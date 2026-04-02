import argparse
import contextlib
import json
import time
import traceback
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import script

try:
    import torch._dynamo as dynamo
except Exception:
    dynamo = None


_ORIG_GF_FORWARD = script.GFBlock.forward
_GF_PATCHED = False
_APPLY_FILTER_FN = None

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def get_parser():
    here = Path(__file__).resolve().parent
    default_ckpt = here / script.OUTPUT_DIR / "checkpoint_last.pth"

    parser = argparse.ArgumentParser()

    # paths / data
    parser.add_argument("--ckpt", type=str, default=str(default_ckpt))
    parser.add_argument("--data-root", type=str, default=str(here))
    parser.add_argument("--num-classes", type=int, default=None)

    # global task control
    parser.add_argument(
        "--task",
        type=str,
        default="both",
        choices=["both", "thr", "acc"],
        help="both: throughput first, then accuracy; thr: throughput only; acc: accuracy only",
    )

    # image / precision
    parser.add_argument("--img-size", type=int, default=int(script.INPUT_SIZE))
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "tf32", "fp32"],
    )

    # validation preprocessing
    parser.add_argument("--crop-pct", type=float, default=0.875)
    parser.add_argument(
        "--interpolation",
        type=str,
        default="bicubic",
        choices=["bicubic", "bilinear"],
    )

    # accuracy evaluation
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-workers", type=int, default=8)

    # throughput evaluation
    parser.add_argument("--batch-size", "--fixed-batch", dest="batch_size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--thr-cache-batches",
        type=int,
        default=1,
        help="Number of preprocessed val batches cached on GPU for model-only throughput. "
             "For cudagraph-like/reduce-overhead use, 1 is usually best.",
    )
    parser.add_argument("--thr-workers", type=int, default=8)

    # compile
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="none",
        choices=[
            "none",
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
    )
    parser.add_argument(
        "--compile-scope",
        type=str,
        default="hybrid",
        choices=["whole", "hybrid"],
        help="whole=compile whole model; hybrid=keep FFT helper eager and compile the rest",
    )
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--allow-compile-fallback", action="store_true")

    # misc
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-json", type=str, default="eval_thr_acc_max_thr6.json")
    return parser


def is_oom_error(e: Exception) -> bool:
    s = str(e).lower()
    return ("out of memory" in s) or ("cuda error: out of memory" in s)


def resolve_val_dir(data_root: Path) -> Path:
    data_root = data_root.resolve()
    if (data_root / "val").is_dir():
        return data_root / "val"
    if data_root.is_dir():
        # allow user to pass val dir directly
        return data_root
    raise FileNotFoundError(f"Cannot find validation directory from: {data_root}")


def infer_num_classes(ckpt_obj, data_root: Path) -> int:
    if isinstance(ckpt_obj, dict):
        args = ckpt_obj.get("args", None)
        if isinstance(args, dict) and "nb_classes" in args:
            return int(args["nb_classes"])

    val_dir = resolve_val_dir(data_root)
    ds = ImageFolder(str(val_dir))
    return len(ds.classes)


def load_ckpt(path: Path):
    if not path.is_file():
        return None
    return torch.load(str(path), map_location="cpu")


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_weights(model: nn.Module, ckpt_obj):
    if ckpt_obj is None:
        return
    state = ckpt_obj.get("model", ckpt_obj) if isinstance(ckpt_obj, dict) else ckpt_obj
    state = strip_module_prefix(state)
    model.load_state_dict(state, strict=True)


def set_backend_for_precision(precision: str):
    cudnn.benchmark = True
    try:
        torch.backends.cudnn.benchmark_limit = 0
    except Exception:
        pass

    allow_tf32 = precision in ("tf32", "fp16", "bf16")
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    try:
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    except Exception:
        pass


def get_autocast_context(precision: str):
    if precision == "fp16":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    if precision == "bf16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def build_val_transform(img_size: int, crop_pct: float, interpolation: str):
    resize_size = int(round(img_size / crop_pct))
    interp = InterpolationMode.BICUBIC if interpolation == "bicubic" else InterpolationMode.BILINEAR
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=interp),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


def make_loader(
    val_dir: Path,
    transform,
    batch_size: int,
    workers: int,
    shuffle: bool = False,
    drop_last: bool = False,
):
    ds = ImageFolder(str(val_dir), transform=transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
        drop_last=drop_last,
        persistent_workers=(workers > 0),
    )
    return ds, loader


def build_model(num_classes: int, device: torch.device):
    model = script.RingBandFilterNetImageNet(
        num_classes=num_classes,
        drop_path_rate=float(script.DROP_PATH),
        ffn_expansion=int(script.FFN_EXPANSION),
        se_ratio=float(script.SE_RATIO),
        depths=script.DEPTHS,
        Ks=script.KS,
        Cs=script.CS,
        reses=script.RESES,
        layerscale_init=float(script.LAYERSCALE_INIT),
        beta_init=float(script.BETA_INIT),
        stage_mixers=script.STAGE_MIXERS,
        stage_gates=script.STAGE_GATES,
        local_k=int(script.LOCAL_K),
    ).to(device)
    model.refresh_buffers(device)
    model.eval()
    return model


def patch_model_forward_chlast(model: nn.Module):
    def forward_patched(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem1(x)
        x = self.stem2(x)
        x = self.stem_to_C1(x)
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)

    model.forward = types.MethodType(forward_patched, model)


def fuse_conv_bn_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    w = conv.weight.detach().float()
    if conv.bias is None:
        b = torch.zeros(conv.out_channels, device=w.device, dtype=torch.float32)
    else:
        b = conv.bias.detach().float()

    gamma = bn.weight.detach().float()
    beta = bn.bias.detach().float()
    mean = bn.running_mean.detach().float()
    var = bn.running_var.detach().float()
    std = torch.sqrt(var + bn.eps)

    w_fused = w * (gamma / std).reshape(-1, 1, 1, 1)
    b_fused = beta + (b - mean) * gamma / std

    fused = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)

    fused.weight.data.copy_(w_fused.to(conv.weight.dtype))
    fused.bias.data.copy_(b_fused.to(conv.weight.dtype))
    return fused


def pad_1x1_to_kxk(kernel: torch.Tensor, k: int) -> torch.Tensor:
    if kernel.size(2) == k and kernel.size(3) == k:
        return kernel
    out = torch.zeros(
        (kernel.size(0), kernel.size(1), k, k),
        device=kernel.device,
        dtype=kernel.dtype,
    )
    center = k // 2
    out[:, :, center:center + 1, center:center + 1] = kernel
    return out


def fuse_identity_bn_to_kernel_bias(num_channels: int, k: int, bn: nn.BatchNorm2d, device, dtype):
    kernel = torch.zeros((num_channels, num_channels, k, k), device=device, dtype=torch.float32)
    center = k // 2
    idx = torch.arange(num_channels, device=device)
    kernel[idx, idx, center, center] = 1.0
    bias = torch.zeros(num_channels, device=device, dtype=torch.float32)

    gamma = bn.weight.detach().float()
    beta = bn.bias.detach().float()
    mean = bn.running_mean.detach().float()
    var = bn.running_var.detach().float()
    std = torch.sqrt(var + bn.eps)

    kernel = kernel * (gamma / std).reshape(-1, 1, 1, 1)
    bias = beta + (bias - mean) * gamma / std
    return kernel.to(dtype), bias.to(dtype)


def repvgg_to_deploy(block: script.RepVGGBlock) -> script.RepVGGBlock:
    if block.deploy:
        return block

    device = next(block.parameters()).device
    dtype = next(block.parameters()).dtype
    k = int(block.k)

    fused_dense = fuse_conv_bn_pair(block.rbr_dense[0], block.rbr_dense[1])
    fused_1x1 = fuse_conv_bn_pair(block.rbr_1x1[0], block.rbr_1x1[1])

    k_dense = fused_dense.weight.detach()
    b_dense = fused_dense.bias.detach()

    k_1x1 = pad_1x1_to_kxk(fused_1x1.weight.detach(), k)
    b_1x1 = fused_1x1.bias.detach()

    if block.rbr_identity is not None:
        k_id, b_id = fuse_identity_bn_to_kernel_bias(block.in_channels, k, block.rbr_identity, device, dtype)
    else:
        k_id = torch.zeros_like(k_dense)
        b_id = torch.zeros_like(b_dense)

    kernel = k_dense + k_1x1 + k_id
    bias = b_dense + b_1x1 + b_id

    new_block = script.RepVGGBlock(
        in_channels=block.in_channels,
        out_channels=block.out_channels,
        k=block.k,
        stride=block.stride,
        deploy=True,
    ).to(device=device, dtype=dtype)

    new_block.rbr_reparam.weight.data.copy_(kernel.to(dtype))
    new_block.rbr_reparam.bias.data.copy_(bias.to(dtype))
    return new_block


def recursive_optimize_module(m: nn.Module) -> nn.Module:
    if m.__class__.__name__ == "DropPath":
        return nn.Identity()

    for name, child in list(m.named_children()):
        if name == "droppath":
            setattr(m, name, nn.Identity())
            continue

        new_child = recursive_optimize_module(child)
        if new_child is not child:
            setattr(m, name, new_child)

    if isinstance(m, script.RepVGGBlock):
        return repvgg_to_deploy(m)

    if isinstance(m, script.ConvBNAct):
        if isinstance(m.bn, nn.BatchNorm2d):
            m.conv = fuse_conv_bn_pair(m.conv, m.bn)
            m.bn = nn.Identity()
        return m

    if isinstance(m, script.LocalDW):
        if isinstance(m.bn, nn.BatchNorm2d):
            m.dw = fuse_conv_bn_pair(m.dw, m.bn)
            m.bn = nn.Identity()
        return m

    if isinstance(m, script.MBV3FFN):
        if isinstance(m.dw_bn, nn.BatchNorm2d):
            m.dw = fuse_conv_bn_pair(m.dw, m.dw_bn)
            m.dw_bn = nn.Identity()
        return m

    return m


def _apply_cached_filter_impl(shared: script.RingGFShared, x: torch.Tensor, g_1chw: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    if h != shared.res or w != shared.res or c != shared.C:
        raise RuntimeError(
            f"GF cached filter mismatch: got {tuple(x.shape)}, expect C={shared.C}, res={shared.res}"
        )

    x32 = x if x.dtype == torch.float32 else x.float()
    X = torch.fft.rfft2(x32, s=(h, w), norm="ortho")
    Y = X * g_1chw
    y = torch.fft.irfft2(Y, s=(h, w), norm="ortho")
    return y if y.dtype == x.dtype else y.to(x.dtype)


if dynamo is not None:
    @dynamo.disable
    def _apply_cached_filter_no_compile(shared: script.RingGFShared, x: torch.Tensor, g_1chw: torch.Tensor) -> torch.Tensor:
        return _apply_cached_filter_impl(shared, x, g_1chw)
else:
    def _apply_cached_filter_no_compile(shared: script.RingGFShared, x: torch.Tensor, g_1chw: torch.Tensor) -> torch.Tensor:
        return _apply_cached_filter_impl(shared, x, g_1chw)


def set_apply_filter_fn(compile_mode: str, compile_scope: str):
    global _APPLY_FILTER_FN
    if compile_mode != "none" and compile_scope == "hybrid":
        _APPLY_FILTER_FN = _apply_cached_filter_no_compile
    else:
        _APPLY_FILTER_FN = _apply_cached_filter_impl


def patch_gfblock_forward_once():
    global _GF_PATCHED
    if _GF_PATCHED:
        return

    def forward_cached(self, x: torch.Tensor) -> torch.Tensor:
        xa = self.norm_a(x)

        cached_g = getattr(self, "_cached_G_1chw", None)
        scale_a = getattr(self, "_cached_scale_a", None)
        scale_b = getattr(self, "_cached_scale_b", None)

        if scale_a is None:
            scale_a = (self.gamma_a * self.beta_a).view(1, -1, 1, 1)
        if scale_b is None:
            scale_b = self.gamma_b.view(1, -1, 1, 1)

        if self.mixer == "G":
            xa = _APPLY_FILTER_FN(self.shared, xa, cached_g) if cached_g is not None else self.shared(xa, self.s_amp, self.s_phase)

        elif self.mixer == "L":
            xa = self.local(xa)

        elif self.mixer == "GL":
            xg = _APPLY_FILTER_FN(self.shared, xa, cached_g) if cached_g is not None else self.shared(xa, self.s_amp, self.s_phase)
            xl = self.local(xa)
            if self.gate is None:
                xa = xg + xl
            else:
                w = self.gate(xa)
                xa = xl + w * (xg - xl)
        else:
            return _ORIG_GF_FORWARD(self, x)

        xa = self.act_gf(xa)
        xa = scale_a * xa
        xa = self.droppath(xa)
        x = x + xa

        xb = self.norm_b(x)
        xb = self.branch_ffn(xb)
        xb = self.droppath(scale_b * xb)
        x = x + xb
        return x

    script.GFBlock.forward = forward_cached
    _GF_PATCHED = True


def build_cached_filter_for_block(block: script.GFBlock) -> torch.Tensor:
    shared = block.shared
    bid_flat = shared.band_id_flat

    A = shared.amp.float() * (1.0 + block.s_amp.float())
    P = shared.phase.float() + block.s_phase.float()

    G_ck = torch.polar(A, P).to(torch.complex64)
    G_flat = torch.index_select(G_ck, dim=1, index=bid_flat)
    G_1chw = G_flat.view(1, shared.C, shared.res, shared.res // 2 + 1).contiguous()
    return G_1chw


def cache_all_filters_and_scales(model: nn.Module):
    patch_gfblock_forward_once()

    for m in model.modules():
        if isinstance(m, script.GFBlock):
            if m.mixer in ("G", "GL"):
                g_1chw = build_cached_filter_for_block(m)
                if hasattr(m, "_cached_G_1chw"):
                    m._cached_G_1chw = g_1chw
                else:
                    m.register_buffer("_cached_G_1chw", g_1chw, persistent=False)

            scale_a = (m.gamma_a.detach() * m.beta_a.detach()).view(1, -1, 1, 1).contiguous()
            scale_b = m.gamma_b.detach().view(1, -1, 1, 1).contiguous()

            if hasattr(m, "_cached_scale_a"):
                m._cached_scale_a = scale_a
            else:
                m.register_buffer("_cached_scale_a", scale_a, persistent=False)

            if hasattr(m, "_cached_scale_b"):
                m._cached_scale_b = scale_b
            else:
                m.register_buffer("_cached_scale_b", scale_b, persistent=False)


def prepare_best_model(num_classes: int, device: torch.device, ckpt_obj):
    model = build_model(num_classes=num_classes, device=device)
    load_weights(model, ckpt_obj)

    model = recursive_optimize_module(model)
    patch_model_forward_chlast(model)

    model = model.to(memory_format=torch.channels_last)
    cache_all_filters_and_scales(model)

    model.eval()
    return model


def maybe_compile_model(model: nn.Module, compile_mode: str, fullgraph: bool):
    if compile_mode == "none":
        return model

    if not hasattr(torch, "compile"):
        raise RuntimeError("Your PyTorch version does not provide torch.compile().")

    kwargs = {
        "fullgraph": bool(fullgraph),
        "dynamic": False,
    }
    if compile_mode != "default":
        kwargs["mode"] = compile_mode

    return torch.compile(model, **kwargs)


def build_runtime_model(
    num_classes: int,
    device: torch.device,
    ckpt_obj,
    compile_mode: str,
    compile_scope: str,
    compile_fullgraph: bool,
):
    set_apply_filter_fn(compile_mode=compile_mode, compile_scope=compile_scope)
    model = prepare_best_model(num_classes=num_classes, device=device, ckpt_obj=ckpt_obj)
    model = maybe_compile_model(model, compile_mode=compile_mode, fullgraph=compile_fullgraph)
    model.eval()
    return model


def accuracy_topk(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    maxk = min(max(topk), output.size(1))
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1))

    res = []
    for k in topk:
        kk = min(k, output.size(1))
        correct_k = correct[:kk].reshape(-1).float().sum()
        res.append(correct_k)
    return res


@torch.inference_mode()
def evaluate_accuracy(
    model,
    loader,
    precision: str,
    device: torch.device,
):
    if precision == "bf16" and (not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Current GPU / runtime does not support bf16.")

    criterion = nn.CrossEntropyLoss().to(device)

    total = 0
    correct1 = 0.0
    correct5 = 0.0
    loss_sum = 0.0

    t0 = time.perf_counter()

    for images, target in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        target = target.to(device, non_blocking=True)

        with get_autocast_context(precision):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy_topk(output, target, topk=(1, 5))
        bs = int(images.size(0))
        total += bs
        correct1 += float(acc1.item())
        correct5 += float(acc5.item())
        loss_sum += float(loss.item()) * bs

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return {
        "num_samples": int(total),
        "top1": float(correct1 * 100.0 / total),
        "top5": float(correct5 * 100.0 / total),
        "loss": float(loss_sum / total),
        "wall_total_s": float(t1 - t0),
    }


@torch.inference_mode()
def cache_val_batches_on_gpu(
    loader,
    num_batches: int,
    device: torch.device,
):
    cached = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        cached.append(images)
        if len(cached) >= int(num_batches):
            break

    if len(cached) == 0:
        raise RuntimeError("Failed to cache any validation batch for throughput benchmarking.")

    return cached


def summarize_runs(run_list, key):
    vals = [float(x[key]) for x in run_list]
    return {
        "mean": float(sum(vals) / len(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


@torch.inference_mode()
def benchmark_one_run_model_only(
    model,
    cached_batches,
    precision: str,
    device: torch.device,
    warmup: int,
    iters: int,
):
    if precision == "bf16" and (not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Current GPU / runtime does not support bf16.")

    y = None
    nb = len(cached_batches)

    for i in range(int(warmup)):
        x = cached_batches[i % nb]
        with get_autocast_context(precision):
            y = model(x)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)

    batch_size = int(cached_batches[0].size(0))
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    t0 = time.perf_counter()
    starter.record()
    for i in range(int(iters)):
        x = cached_batches[i % nb]
        with get_autocast_context(precision):
            y = model(x)
    ender.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    cuda_total_ms = float(starter.elapsed_time(ender))
    wall_total_ms = float((t1 - t0) * 1000.0)
    checksum = float(y.float().mean().item()) if y is not None else None

    return {
        "batch_size": batch_size,
        "cuda_total_ms": cuda_total_ms,
        "wall_total_ms": wall_total_ms,
        "cuda_mean_ms_per_batch": float(cuda_total_ms / float(iters)),
        "wall_mean_ms_per_batch": float(wall_total_ms / float(iters)),
        "cuda_event_images_per_sec": float(batch_size * iters * 1000.0 / cuda_total_ms),
        "wall_clock_images_per_sec": float(batch_size * iters * 1000.0 / wall_total_ms),
        "max_mem_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
        "checksum_mean": checksum,
        "cached_batches": int(nb),
    }


def run_throughput(
    args,
    model,
    val_dir: Path,
    transform,
    device: torch.device,
):
    _, thr_loader = make_loader(
        val_dir=val_dir,
        transform=transform,
        batch_size=int(args.batch_size),
        workers=int(args.thr_workers),
        shuffle=False,
        drop_last=True,
    )

    cached_batches = cache_val_batches_on_gpu(
        loader=thr_loader,
        num_batches=int(args.thr_cache_batches),
        device=device,
    )

    runs = []
    for ridx in range(int(args.repeats)):
        one = benchmark_one_run_model_only(
            model=model,
            cached_batches=cached_batches,
            precision=args.precision,
            device=device,
            warmup=int(args.warmup),
            iters=int(args.iters),
        )
        one["run_id"] = ridx + 1
        runs.append(one)

    summary = {
        "cuda_event_images_per_sec": summarize_runs(runs, "cuda_event_images_per_sec"),
        "wall_clock_images_per_sec": summarize_runs(runs, "wall_clock_images_per_sec"),
        "cuda_mean_ms_per_batch": summarize_runs(runs, "cuda_mean_ms_per_batch"),
        "wall_mean_ms_per_batch": summarize_runs(runs, "wall_mean_ms_per_batch"),
        "max_mem_mb": summarize_runs(runs, "max_mem_mb"),
    }
    return runs, summary


def main():
    args = get_parser().parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    set_backend_for_precision(args.precision)

    torch.manual_seed(int(args.seed))

    ckpt_path = Path(args.ckpt).resolve()
    data_root = Path(args.data_root).resolve()
    val_dir = resolve_val_dir(data_root)
    ckpt_obj = load_ckpt(ckpt_path)

    if args.num_classes is not None:
        num_classes = int(args.num_classes)
    else:
        num_classes = infer_num_classes(ckpt_obj, data_root)

    transform = build_val_transform(
        img_size=int(args.img_size),
        crop_pct=float(args.crop_pct),
        interpolation=args.interpolation,
    )

    actual_compile_mode = args.compile_mode
    actual_compile_scope = args.compile_scope
    fallback_reason = None

    try:
        model = build_runtime_model(
            num_classes=num_classes,
            device=device,
            ckpt_obj=ckpt_obj,
            compile_mode=actual_compile_mode,
            compile_scope=actual_compile_scope,
            compile_fullgraph=bool(args.compile_fullgraph),
        )
    except Exception as e:
        if args.allow_compile_fallback and args.compile_mode != "none":
            fallback_reason = f"{type(e).__name__}: {str(e)}"
            print("")
            print("[warning] compile path failed; falling back to eager.")
            print(f"[warning] reason: {fallback_reason}")
            print("")

            if dynamo is not None:
                try:
                    dynamo.reset()
                except Exception:
                    pass
            torch.cuda.empty_cache()

            actual_compile_mode = "none"
            actual_compile_scope = "whole"
            model = build_runtime_model(
                num_classes=num_classes,
                device=device,
                ckpt_obj=ckpt_obj,
                compile_mode=actual_compile_mode,
                compile_scope=actual_compile_scope,
                compile_fullgraph=bool(args.compile_fullgraph),
            )
        else:
            print("")
            print("===== BUILD FAILED =====")
            print(traceback.format_exc())
            raise

    throughput_runs = None
    throughput_summary = None
    accuracy_result = None

    # throughput first to reduce thermal throttling bias
    if args.task in ("both", "thr"):
        throughput_runs, throughput_summary = run_throughput(
            args=args,
            model=model,
            val_dir=val_dir,
            transform=transform,
            device=device,
        )

    if args.task in ("both", "acc"):
        _, acc_loader = make_loader(
            val_dir=val_dir,
            transform=transform,
            batch_size=int(args.eval_batch_size),
            workers=int(args.eval_workers),
            shuffle=False,
            drop_last=False,
        )
        accuracy_result = evaluate_accuracy(
            model=model,
            loader=acc_loader,
            precision=args.precision,
            device=device,
        )

    payload = {
        "device": torch.cuda.get_device_name(0),
        "checkpoint": str(ckpt_path) if ckpt_obj is not None else None,
        "num_classes": int(num_classes),
        "task": args.task,
        "precision": args.precision,
        "img_size": int(args.img_size),
        "crop_pct": float(args.crop_pct),
        "interpolation": args.interpolation,
        "throughput_batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "repeats": int(args.repeats),
        "thr_cache_batches": int(args.thr_cache_batches),
        "requested_compile_mode": args.compile_mode,
        "requested_compile_scope": args.compile_scope,
        "actual_compile_mode": actual_compile_mode,
        "actual_compile_scope": actual_compile_scope,
        "compile_fullgraph": bool(args.compile_fullgraph),
        "allow_compile_fallback": bool(args.allow_compile_fallback),
        "fallback_reason": fallback_reason,
        "throughput_runs": throughput_runs,
        "throughput_summary": throughput_summary,
        "accuracy": accuracy_result,
    }

    out_json = Path(args.out_json).resolve()
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("")
    print("===== EVAL + THROUGHPUT RESULT =====")
    print(f"precision                       : {args.precision}")
    print(f"img size                        : {args.img_size}")
    print(f"actual compile                  : {actual_compile_mode} / {actual_compile_scope}")
    if fallback_reason is not None:
        print(f"fallback reason                 : {fallback_reason}")

    if throughput_runs is not None:
        print("")
        print("---- throughput (model-only, preprocessed val batches cached on GPU) ----")
        for r in throughput_runs:
            print(
                f"run {r['run_id']:>2d} | "
                f"cuda img/s = {r['cuda_event_images_per_sec']:.2f} | "
                f"wall img/s = {r['wall_clock_images_per_sec']:.2f} | "
                f"cuda ms/batch = {r['cuda_mean_ms_per_batch']:.3f} | "
                f"wall ms/batch = {r['wall_mean_ms_per_batch']:.3f} | "
                f"mem MB = {r['max_mem_mb']:.1f} | "
                f"checksum = {r['checksum_mean']:.6f}"
            )
        print(
            f"summary cuda img/s mean/min/max: "
            f"{throughput_summary['cuda_event_images_per_sec']['mean']:.2f} / "
            f"{throughput_summary['cuda_event_images_per_sec']['min']:.2f} / "
            f"{throughput_summary['cuda_event_images_per_sec']['max']:.2f}"
        )
        print(
            f"summary wall img/s mean/min/max: "
            f"{throughput_summary['wall_clock_images_per_sec']['mean']:.2f} / "
            f"{throughput_summary['wall_clock_images_per_sec']['min']:.2f} / "
            f"{throughput_summary['wall_clock_images_per_sec']['max']:.2f}"
        )

    if accuracy_result is not None:
        print("")
        print("---- accuracy (full val) ----")
        print(f"samples                         : {accuracy_result['num_samples']}")
        print(f"top1                            : {accuracy_result['top1']:.4f}")
        print(f"top5                            : {accuracy_result['top5']:.4f}")
        print(f"loss                            : {accuracy_result['loss']:.6f}")
        print(f"accuracy wall time (s)          : {accuracy_result['wall_total_s']:.2f}")

    print(f"saved json                      : {out_json}")


if __name__ == "__main__":
    main()
