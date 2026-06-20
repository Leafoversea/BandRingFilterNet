import argparse
import copy
import csv
import gc
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent

REPEATS = [0, 1, 2, 3, 4]
COOLDOWN_SECONDS = 60

BS1 = 1
BS2048 = 2048

WARMUP_BS1 = 100
ITERS_BS1 = 500

WARMUP_BS2048 = 30
ITERS_BS2048 = 100

USE_LATENCY_AMP = True
AMP_DTYPE = torch.float16

BACKBONE_DESC = "same BRFNet-T-style CIFAR-100 backbone"
TRAINING_RECIPE_DESC = "same CIFAR-100 recipe"

STAGES = [
    {"stage": "stage1", "C": 32, "res": 32, "K": 8, "depth": 3},
    {"stage": "stage2", "C": 48, "res": 16, "K": 6, "depth": 6},
    {"stage": "stage3", "C": 64, "res": 8, "K": 4, "depth": 8},
    {"stage": "stage4", "C": 80, "res": 4, "K": 2, "depth": 3},
]

EXPERIMENTS = {
    "1": {
        "method": "GFNet-style point-wise",
        "script": ROOT / "pointwise.py",
        "save_pattern": "exp_c100_pointwise_seed0",
        "ckpt_glob": "deploy_model_best_swa_acc*.pt",
        "mode": "pointwise",
        "kind": "point",
        "access": "one coefficient per channel-frequency point",
    },
    "2": {
        "method": "Ring-band SIMD",
        "script": ROOT / "ring.py",
        "save_pattern": "exp_c100_ring_seed0",
        "ckpt_glob": "deploy_model_best_ensemble_acc*.pt",
        "mode": "simd",
        "kind": "ring",
        "access": "SIMD-style cached deployment lanes with compact coefficient reuse",
    },
}


def cleanup_cuda(device):
    gc.collect()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

        try:
            torch.backends.cuda.cufft_plan_cache.clear()
        except Exception:
            pass


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

        try:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.reset_accumulated_memory_stats(device)
        except Exception:
            pass


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safe_load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")
    except Exception:
        return torch.load(str(path), map_location="cpu")


def build_train_model(script_path):
    mod = load_module(script_path, f"_train_{script_path.stem}_{time.time_ns()}")

    if script_path.stem == "ring":
        model = mod.RingBandFilterNet(num_classes=100, band_edge_mode="area")
        model.refresh_buffers(torch.device("cpu"))
    else:
        model = mod.RingBandFilterNet(num_classes=100)

    return model


def build_deploy_model(script_path, save_dir, device, ckpt_name=None, ckpt_glob=None):
    mod = load_module(script_path, f"_deploy_{script_path.stem}_{time.time_ns()}")

    if script_path.stem == "ring":
        model = mod.RingBandFilterNet(num_classes=100, band_edge_mode="area")
        model.refresh_buffers(torch.device("cpu"))
    else:
        model = mod.RingBandFilterNet(num_classes=100)

    if hasattr(model, "switch_to_deploy"):
        model.switch_to_deploy()

    if ckpt_name is not None:
        ckpt_path = save_dir / ckpt_name
    elif ckpt_glob is not None:
        matches = sorted(save_dir.glob(ckpt_glob), key=lambda p: p.stat().st_mtime)

        if len(matches) == 0:
            raise FileNotFoundError(f"no checkpoint matched: {save_dir / ckpt_glob}")

        ckpt_path = matches[-1]
    else:
        ckpt_path = save_dir / "deploy_model.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}", flush=True)

    ckpt = safe_load(ckpt_path)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    model.to(device)

    if hasattr(model, "refresh_buffers"):
        model.refresh_buffers(device)

    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    model.eval()
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_spectral_filter_params(model):
    names = {"w_re", "w_im", "amp", "phase", "s_amp", "s_phase"}
    total = 0

    for name, p in model.named_parameters():
        if name.split(".")[-1] in names:
            total += p.numel()

    return int(total)


def real_scalars_to_kb(n_real_scalars, bitwidth):
    return n_real_scalars * bitwidth / 8.0 / 1024.0


def spectral_storage_summary(kind):
    trainable_real = 0
    fused_real = 0

    for s in STAGES:
        c = int(s["C"])
        res = int(s["res"])
        k = int(s["K"])
        d = int(s["depth"])
        f = res * (res // 2 + 1)

        if kind == "point":
            trainable_real += d * c * f * 2
            fused_real += d * c * f * 2

        elif kind == "ring":
            trainable_real += (d + 1) * c * k * 2
            fused_real += d * c * k * 2

    return {
        "Trainable spectral real scalars": trainable_real,
        "Trainable filter storage FP32 (KB)": real_scalars_to_kb(trainable_real, 32),
        "Trainable filter storage INT16 (KB)": real_scalars_to_kb(trainable_real, 16),
        "Trainable filter storage INT8 (KB)": real_scalars_to_kb(trainable_real, 8),
        "Fused deploy real scalars": fused_real,
        "Fused deploy storage FP32 (KB)": real_scalars_to_kb(fused_real, 32),
        "Fused deploy storage INT16 (KB)": real_scalars_to_kb(fused_real, 16),
        "Fused deploy storage INT8 (KB)": real_scalars_to_kb(fused_real, 8),
    }


def ratio(ref, cur):
    if cur is None or cur == 0:
        return None

    return ref / cur


def reduction_metrics(kind):
    if kind not in {"point", "ring"}:
        return {
            "Trainable spectral param reduction vs point-wise (x)": None,
            "Fused deploy storage reduction vs point-wise (x)": None,
        }

    ref = spectral_storage_summary("point")
    cur = spectral_storage_summary(kind)

    return {
        "Trainable spectral param reduction vs point-wise (x)": ratio(
            ref["Trainable spectral real scalars"],
            cur["Trainable spectral real scalars"],
        ),
        "Fused deploy storage reduction vs point-wise (x)": ratio(
            ref["Fused deploy real scalars"],
            cur["Fused deploy real scalars"],
        ),
    }


def install_pointwise_inplace(model):
    for m in model.modules():
        if m.__class__.__name__ != "PointwiseGFShared":
            continue

        g = torch.complex(
            m.w_re.detach().float(),
            m.w_im.detach().float(),
        ).to(torch.complex64).contiguous()

        m.register_buffer("_deploy_G_chw", g, persistent=False)

        def forward_pointwise_inplace(self, x):
            b, c, h, w = x.shape
            x_freq = torch.fft.rfft2(x.float(), s=(h, w), norm="ortho")
            x_freq.mul_(self._deploy_G_chw.unsqueeze(0))
            y = torch.fft.irfft2(x_freq, s=(h, w), norm="ortho")
            return y.to(x.dtype)

        m.forward = MethodType(forward_pointwise_inplace, m)

    return model


def install_ring_simd_inplace(model):
    device = next(model.parameters()).device

    for stage_name in ["stage1", "stage2", "stage3", "stage4"]:
        if not hasattr(model, stage_name):
            continue

        stage = getattr(model, stage_name)

        if not hasattr(stage, "shared"):
            continue

        shared = stage.shared
        bid = shared.band_id.to(device=device)

        for block in stage.blocks:
            amp = shared.amp.detach().float() * (1.0 + block.s_amp.detach().float())
            phase = shared.phase.detach().float() + block.s_phase.detach().float()
            g_ck = torch.polar(amp, phase).to(torch.complex64)
            g_chw = g_ck[:, bid].contiguous()

            block.register_buffer("_simd_G_chw", g_chw, persistent=False)

            def forward_simd_inplace(self, x):
                xa = self.norm_a(x)
                b, c, h, w = xa.shape

                x_freq = torch.fft.rfft2(xa.float(), s=(h, w), norm="ortho")
                x_freq.mul_(self._simd_G_chw.unsqueeze(0))

                xa = torch.fft.irfft2(x_freq, s=(h, w), norm="ortho").to(x.dtype)
                xa = self.act_gf(xa)
                xa = self.beta_a * (self.gamma_a * xa)
                xa = self.droppath(xa)
                x = x + xa

                xb = self.norm_b(x)
                xb = self.branch_ffn(xb)
                xb = self.droppath(self.gamma_b * xb)
                x = x + xb

                return x

            block.forward = MethodType(forward_simd_inplace, block)

    return model


def install_inference_path(model, exp):
    if exp["mode"] == "pointwise":
        return install_pointwise_inplace(model)

    if exp["mode"] == "simd":
        return install_ring_simd_inplace(model)

    return model


def latency_context(device):
    if device.type == "cuda" and USE_LATENCY_AMP:
        return torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE)

    return torch.no_grad()


@torch.inference_mode()
def benchmark(model, device, batch_size, warmup, iters, measure_peak_memory=True):
    if device.type != "cuda":
        return {
            "latency_ms_per_img": None,
            "latency_ms_per_batch": None,
            "throughput_img_s": None,
            "peak_mem_mb": None,
        }

    cleanup_cuda(device)

    model.eval()

    x = torch.randn(batch_size, 3, 32, 32, device=device)
    x = x.contiguous(memory_format=torch.channels_last)

    for _ in range(warmup):
        with latency_context(device):
            _ = model(x)

    torch.cuda.synchronize()

    if measure_peak_memory:
        reset_peak_memory(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(iters):
        with latency_context(device):
            _ = model(x)

    end.record()
    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end)

    latency_ms_per_batch = elapsed / iters
    latency_ms_per_img = latency_ms_per_batch / batch_size
    throughput_img_s = 1000.0 * batch_size / latency_ms_per_batch

    peak_mem_mb = None

    if measure_peak_memory:
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0

    del x
    cleanup_cuda(device)

    return {
        "latency_ms_per_img": latency_ms_per_img,
        "latency_ms_per_batch": latency_ms_per_batch,
        "throughput_img_s": throughput_img_s,
        "peak_mem_mb": peak_mem_mb,
    }


def estimate_flops_m(model, device):
    model = copy.deepcopy(model).to(device)
    model.eval()

    if hasattr(model, "refresh_buffers"):
        model.refresh_buffers(device)

    if hasattr(model, "switch_to_deploy"):
        model.switch_to_deploy()

    totals = {"base": 0.0, "fft": 0.0}
    handles = []

    def conv_hook(m, inputs, output):
        x = inputs[0]
        b = int(x.shape[0])
        oh = int(output.shape[-2])
        ow = int(output.shape[-1])
        kh, kw = m.kernel_size
        in_per_group = m.in_channels // m.groups

        totals["base"] += b * oh * ow * m.out_channels * in_per_group * kh * kw

    def linear_hook(m, inputs, output):
        x = inputs[0]
        b = int(x.shape[0]) if x.ndim >= 2 else 1

        totals["base"] += b * m.in_features * m.out_features

    def spectral_hook(m, inputs, output):
        x = inputs[0]

        if not torch.is_tensor(x) or x.ndim != 4:
            return

        b, c, h, w = [int(v) for v in x.shape]
        wc = w // 2 + 1

        totals["fft"] += 2.0 * b * c * h * w * math.log2(h * w)
        totals["fft"] += b * c * h * wc

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif m.__class__.__name__ in {"PointwiseGFShared", "RingGFShared"}:
            handles.append(m.register_forward_hook(spectral_hook))

    with torch.no_grad():
        x = torch.randn(1, 3, 32, 32, device=device)
        _ = model(x)

    for h in handles:
        h.remove()

    del model
    cleanup_cuda(device)

    return (totals["base"] + totals["fft"]) / 1e6


def profile_worker(choice, repeat):
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = EXPERIMENTS[choice]
    save_dir = ROOT / exp["save_pattern"]

    cleanup_cuda(device)

    train_model = build_train_model(exp["script"])

    params = count_params(train_model)
    spectral_params = count_spectral_filter_params(train_model)
    storage_kb = spectral_params * 4.0 / 1024.0
    flops_m = estimate_flops_m(train_model, device)

    storage_summary = spectral_storage_summary(exp["kind"])
    red = reduction_metrics(exp["kind"])

    del train_model
    cleanup_cuda(device)

    model = build_deploy_model(
        exp["script"],
        save_dir,
        device,
        ckpt_name=exp.get("ckpt_name"),
        ckpt_glob=exp.get("ckpt_glob"),
    )

    model = install_inference_path(model, exp)

    cleanup_cuda(device)

    bs1 = benchmark(
        model,
        device,
        BS1,
        WARMUP_BS1,
        ITERS_BS1,
        measure_peak_memory=True,
    )

    cleanup_cuda(device)

    bs2048 = benchmark(
        model,
        device,
        BS2048,
        WARMUP_BS2048,
        ITERS_BS2048,
        measure_peak_memory=False,
    )

    row = {
        "Method": exp["method"],
        "Repeat": repeat,
        "Backbone": BACKBONE_DESC,
        "Training Recipe": TRAINING_RECIPE_DESC,
        "Params (M)": params / 1e6,
        "FLOPs (M, est.)": flops_m,
        "GPU Latency bs=1 (ms/img)": bs1["latency_ms_per_img"],
        "Peak GPU Mem bs=1 (MB)": bs1["peak_mem_mb"],
        "GPU Latency bs=2048 (ms/img)": bs2048["latency_ms_per_img"],
        "GPU Batch Latency bs=2048 (ms/batch)": bs2048["latency_ms_per_batch"],
        "GPU Throughput bs=2048 (img/s)": bs2048["throughput_img_s"],
        "Spectral Filter Params": spectral_params,
        "Filter Storage (KB, FP32)": storage_kb,
        "Coeff. Access Pattern": exp["access"],
    }

    row.update(storage_summary)
    row.update(red)

    print("RESULT_JSON " + json.dumps(row, ensure_ascii=False), flush=True)

    del model
    cleanup_cuda(device)


def mean_std(values):
    vals = [v for v in values if isinstance(v, (int, float))]

    if len(vals) == 0:
        return None, None

    if len(vals) == 1:
        return vals[0], 0.0

    return statistics.mean(vals), statistics.stdev(vals)


def summarize(rows):
    groups = {}

    for r in rows:
        groups.setdefault(r["Method"], []).append(r)

    out = []

    mean_std_keys = [
        "GPU Latency bs=1 (ms/img)",
        "Peak GPU Mem bs=1 (MB)",
        "GPU Latency bs=2048 (ms/img)",
        "GPU Batch Latency bs=2048 (ms/batch)",
        "GPU Throughput bs=2048 (img/s)",
    ]

    static_keys = [
        "Backbone",
        "Training Recipe",
        "Params (M)",
        "FLOPs (M, est.)",
        "Spectral Filter Params",
        "Filter Storage (KB, FP32)",
        "Coeff. Access Pattern",
        "Trainable spectral real scalars",
        "Trainable filter storage FP32 (KB)",
        "Trainable filter storage INT16 (KB)",
        "Trainable filter storage INT8 (KB)",
        "Fused deploy real scalars",
        "Fused deploy storage FP32 (KB)",
        "Fused deploy storage INT16 (KB)",
        "Fused deploy storage INT8 (KB)",
        "Trainable spectral param reduction vs point-wise (x)",
        "Fused deploy storage reduction vs point-wise (x)",
    ]

    for method, rs in groups.items():
        first = rs[0]

        row = {
            "Method": method,
            "Repeated Runs": len(rs),
        }

        for k in static_keys:
            row[k] = first.get(k)

        for k in mean_std_keys:
            m, s = mean_std([r.get(k) for r in rs])

            mean_name = k.replace(" (ms/img)", " mean (ms/img)")
            mean_name = mean_name.replace(" (ms/batch)", " mean (ms/batch)")
            mean_name = mean_name.replace(" (img/s)", " mean (img/s)")
            mean_name = mean_name.replace(" (MB)", " mean (MB)")

            std_name = k.replace(" (ms/img)", " std")
            std_name = std_name.replace(" (ms/batch)", " std")
            std_name = std_name.replace(" (img/s)", " std")
            std_name = std_name.replace(" (MB)", " std")

            row[mean_name] = m
            row[std_name] = s

        out.append(row)

    return out


def fmt(v, n=4):
    if v is None:
        return "NA"

    if isinstance(v, float):
        return f"{v:.{n}f}"

    return str(v)


def print_row(row):
    print("\n----- Result row -----")

    for k, v in row.items():
        if isinstance(v, float):
            print(f"{k}: {fmt(v, 4)}")
        else:
            print(f"{k}: {v}")


def print_summary(summary):
    print("\n========== SUMMARY ==========")

    for row in summary:
        print("\n----- Summary row -----")

        for k, v in row.items():
            if isinstance(v, float):
                print(f"{k}: {fmt(v, 4)}")
            else:
                print(f"{k}: {v}")


def write_csv(path, rows):
    if len(rows) == 0:
        return

    fields = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def md_value(k, v):
    if v is None:
        return "NA"

    if isinstance(v, float):
        lk = k.lower()

        if "latency" in lk:
            return f"{v:.4f}"

        if "throughput" in lk:
            return f"{v:.2f}"

        if "mem" in lk:
            return f"{v:.4f}"

        if "reduction" in lk:
            return f"{v:.2f}x"

        if "storage" in lk:
            return f"{v:.4f}"

        if "params" in lk or "flops" in lk:
            return f"{v:.4f}"

        return f"{v:.4f}"

    return str(v)


def write_md(path, rows):
    if len(rows) == 0:
        return

    fields = list(rows[0].keys())

    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")

        for row in rows:
            vals = [md_value(k, row.get(k)) for k in fields]
            f.write("| " + " | ".join(vals) + " |\n")


def write_outputs(rows, summary):
    stamp = time.strftime("%Y%m%d_%H%M%S")

    per_repeat_csv = ROOT / f"controlled_deployment_comparison_per_repeat_{stamp}.csv"
    per_repeat_json = ROOT / f"controlled_deployment_comparison_per_repeat_{stamp}.json"
    per_repeat_md = ROOT / f"controlled_deployment_comparison_per_repeat_{stamp}.md"

    summary_csv = ROOT / f"controlled_deployment_comparison_summary_{stamp}.csv"
    summary_json = ROOT / f"controlled_deployment_comparison_summary_{stamp}.json"
    summary_md = ROOT / f"controlled_deployment_comparison_summary_{stamp}.md"

    latest_per_repeat_csv = ROOT / "controlled_deployment_comparison_per_repeat_latest.csv"
    latest_per_repeat_json = ROOT / "controlled_deployment_comparison_per_repeat_latest.json"
    latest_per_repeat_md = ROOT / "controlled_deployment_comparison_per_repeat_latest.md"

    latest_summary_csv = ROOT / "controlled_deployment_comparison_summary_latest.csv"
    latest_summary_json = ROOT / "controlled_deployment_comparison_summary_latest.json"
    latest_summary_md = ROOT / "controlled_deployment_comparison_summary_latest.md"

    for p in [per_repeat_csv, latest_per_repeat_csv]:
        write_csv(p, rows)

    for p in [per_repeat_json, latest_per_repeat_json]:
        write_json(p, rows)

    for p in [per_repeat_md, latest_per_repeat_md]:
        write_md(p, rows)

    for p in [summary_csv, latest_summary_csv]:
        write_csv(p, summary)

    for p in [summary_json, latest_summary_json]:
        write_json(p, summary)

    for p in [summary_md, latest_summary_md]:
        write_md(p, summary)

    print("\nSaved:")
    print(per_repeat_csv)
    print(per_repeat_json)
    print(per_repeat_md)
    print(summary_csv)
    print(summary_json)
    print(summary_md)
    print(latest_per_repeat_csv)
    print(latest_summary_csv)


def run_child(choice, repeat):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--choice",
        str(choice),
        "--repeat",
        str(repeat),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    p = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    row = None

    for line in p.stdout:
        print(line, end="")

        if line.startswith("RESULT_JSON "):
            row = json.loads(line[len("RESULT_JSON "):].strip())

    ret = p.wait()

    if ret != 0:
        raise RuntimeError(f"worker failed, choice={choice}, repeat={repeat}, returncode={ret}")

    if row is None:
        raise RuntimeError(f"worker did not return RESULT_JSON, choice={choice}, repeat={repeat}")

    return row


def resolve_choices(choice):
    if choice == "1":
        return ["1"]

    if choice == "2":
        return ["2"]

    if choice == "a":
        return ["1", "2"]

    raise ValueError(f"invalid choice: {choice}")


def controller(choice=None):
    if choice is None:
        print("1: GFNet-style point-wise")
        print("2: Ring-band SIMD")
        print("a: run both")
        choice = input("choice> ").strip().lower()

    keys = resolve_choices(choice)

    rows = []

    for key_idx, key in enumerate(keys):
        for repeat_idx, repeat in enumerate(REPEATS):
            print(f"\n========== RUN {EXPERIMENTS[key]['method']} repeat{repeat} ==========\n")

            row = run_child(key, repeat)
            print_row(row)
            rows.append(row)

            is_last = key_idx == len(keys) - 1 and repeat_idx == len(REPEATS) - 1

            if not is_last:
                print(f"\nCooling down for {COOLDOWN_SECONDS} seconds...\n")
                time.sleep(COOLDOWN_SECONDS)

    summary = summarize(rows)
    print_summary(summary)
    write_outputs(rows, summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--choice", type=str, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    args = parser.parse_args()

    if args.worker:
        profile_worker(args.choice, args.repeat)
    else:
        controller(args.choice)


if __name__ == "__main__":
    main()
