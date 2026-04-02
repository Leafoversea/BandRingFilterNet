import argparse
import contextlib
import copy
import struct
import importlib.util
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import torch
def _dequant_i16_qF_on_device(q_i16, Fq: int):
    return q_i16.to(torch.float32) / float(1 << int(Fq))

def dump_q_i16(name: str, q_i16_chw: torch.Tensor, Fq: int, out_dir: str):
    # q_i16_chw: [C,H,W] int16 on ANY device
    q = q_i16_chw.detach().to("cpu", non_blocking=False)
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    # 复用你已有的 save_txt_i16_nchw_qF
    save_txt_i16_nchw_qF(str(p / f"{name}.txt"), q.unsqueeze(0), int(Fq))

def dump_logits_prob(tag: str, logits: torch.Tensor, prob: torch.Tensor, out_dir: str):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{tag}_logits.txt").write_text(" ".join([f"{float(x):.8f}" for x in logits.reshape(-1).tolist()]), encoding="utf-8")
    (p / f"{tag}_prob.txt").write_text(" ".join([f"{float(x):.10f}" for x in prob.reshape(-1).tolist()]), encoding="utf-8")
import json
from pathlib import Path
import torch
import torch.nn.functional as F

def _pick(d, *keys):
    if d is None:
        return None
    for k in keys:
        if isinstance(d, dict) and (k in d) and (d[k] is not None):
            return d[k]
    return None

def _spec_from_path(path_like, use_relu=False):
    if path_like is None:
        return None
    if isinstance(path_like, (str, Path)):
        return {"mem_dir": str(path_like), "use_relu": bool(use_relu)}
    if isinstance(path_like, dict):
        return path_like
    return None

def _normalize_pw_spec(spec, use_relu_default=False):
    if spec is None:
        return None
    if isinstance(spec, (str, Path)):
        return {"mem_dir": str(spec), "use_relu": bool(use_relu_default)}
    if not isinstance(spec, dict):
        return None

    md = _pick(spec, "mem_dir", "mem", "dir", "path")
    if md is None:
        exp = _pick(spec, "export")
        if isinstance(exp, dict):
            md = _pick(exp, "expand_dir", "expand_mem_dir", "project_dir", "project_mem_dir", "dir", "mem_dir")
    if md is None:
        md = _pick(spec, "expand_dir", "expand_mem_dir", "project_dir", "project_mem_dir", "exp_dir", "pr_dir")

    if md is None:
        return None

    use_relu = _pick(spec, "use_relu", "USE_RELU")
    if use_relu is None:
        use_relu = bool(use_relu_default)
    return {"mem_dir": str(md), "use_relu": bool(use_relu)}

def _normalize_dw_spec(spec, use_relu_default=True):
    if spec is None:
        return None
    if isinstance(spec, (str, Path)):
        return {"mem_dir": str(spec), "use_relu": bool(use_relu_default)}
    if not isinstance(spec, dict):
        return None

    md = _pick(spec, "mem_dir", "mem", "dir", "path")
    if md is None:
        exp = _pick(spec, "export")
        if isinstance(exp, dict):
            md = _pick(exp, "dw_dir", "dw_mem_dir", "dir", "mem_dir")
    if md is None:
        md = _pick(spec, "dw_dir", "dw_mem_dir")

    if md is None:
        return None

    use_relu = _pick(spec, "use_relu", "USE_RELU")
    if use_relu is None:
        use_relu = bool(use_relu_default)
    return {"mem_dir": str(md), "use_relu": bool(use_relu)}

@torch.no_grad()
def load_module_from_path(py_path: str, mod_name: str = "user_model_def"):
    p = Path(py_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    spec = importlib.util.spec_from_file_location(mod_name, str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_tokens(txt: str) -> List[str]:
    txt = txt.replace("\r", "\n")
    m_vec = re.search(r"memory_initialization_vector\s*=\s*(.*?);", txt, flags=re.S | re.I)
    if m_vec is not None:
        txt = m_vec.group(1)
    txt = re.sub(r"memory_initialization_radix\s*=\s*\d+\s*;", " ", txt, flags=re.I)
    txt = re.sub(r"//.*?$", " ", txt, flags=re.M)
    txt = re.sub(r"/\*.*?\*/", " ", txt, flags=re.S)
    toks = re.split(r"[, \t\r\n]+", txt)
    return [t.strip() for t in toks if t and t.strip()]


def parse_coe_rgb888(path: str, H: int = 64, W: int = 64) -> torch.Tensor:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    m_rad = re.search(r"memory_initialization_radix\s*=\s*(\d+)\s*;", txt, flags=re.I)
    radix = int(m_rad.group(1)) if m_rad else 16
    toks = _parse_tokens(txt)
    vals = []
    for t in toks:
        tt = t
        if tt.lower().startswith("0x"):
            tt = tt[2:]
        try:
            v = int(tt, radix)
        except Exception:
            continue
        vals.append(int(v))
    need = int(H) * int(W)
    if len(vals) < need:
        raise RuntimeError(f"{path}: not enough entries {len(vals)} < {need}")
    vals = vals[:need]
    v = torch.tensor(vals, dtype=torch.int64)
    r = ((v >> 16) & 255).to(torch.uint8).view(H, W)
    g = ((v >> 8) & 255).to(torch.uint8).view(H, W)
    b = (v & 255).to(torch.uint8).view(H, W)
    return torch.stack([r, g, b], dim=0)


def parse_mem_rgb888(path: str, H: int = 64, W: int = 64) -> torch.Tensor:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    toks = _parse_tokens(txt)
    vals = []
    for t in toks:
        tt = t
        if tt.startswith("@"):
            continue
        if tt.lower().startswith("0x"):
            tt = tt[2:]
        try:
            v = int(tt, 16)
        except Exception:
            try:
                v = int(tt, 10)
            except Exception:
                continue
        vals.append(int(v))
    need = int(H) * int(W)
    if len(vals) < need:
        raise RuntimeError(f"{path}: not enough entries {len(vals)} < {need}")
    vals = vals[:need]
    v = torch.tensor(vals, dtype=torch.int64)
    r = ((v >> 16) & 255).to(torch.uint8).view(H, W)
    g = ((v >> 8) & 255).to(torch.uint8).view(H, W)
    b = (v & 255).to(torch.uint8).view(H, W)
    return torch.stack([r, g, b], dim=0)


def load_img_rgb888(path: str, H: int = 64, W: int = 64) -> torch.Tensor:
    p = Path(path)
    if p.suffix.lower() == ".coe":
        return parse_coe_rgb888(str(p), H=H, W=W)
    return parse_mem_rgb888(str(p), H=H, W=W)


def pool2x2_u8(x_u8_3x64x64: torch.Tensor) -> torch.Tensor:
    x = x_u8_3x64x64.to(torch.int16)
    a = x[:, 0::2, 0::2]
    b = x[:, 0::2, 1::2]
    c = x[:, 1::2, 0::2]
    d = x[:, 1::2, 1::2]
    m1 = torch.maximum(a, b)
    m2 = torch.maximum(c, d)
    y = torch.maximum(m1, m2).to(torch.uint8)
    return y


def read_memh_signed(path: str, bits: int, count: int) -> List[int]:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    txt = re.sub(r"//.*", "", txt)
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.S)
    toks = re.split(r"[\s,]+", txt)
    vals = []
    for t in toks:
        tt = t.strip()
        if not tt:
            continue
        if tt.startswith("@"):
            continue
        if tt.lower().startswith("0x"):
            tt = tt[2:]
        try:
            v = int(tt, 16)
        except Exception:
            try:
                v = int(tt, 10)
            except Exception:
                continue
        mask = (1 << bits) - 1
        v = v & mask
        if v >= (1 << (bits - 1)):
            v = v - (1 << bits)
        vals.append(int(v))
        if len(vals) >= count:
            break
    if len(vals) < count:
        raise RuntimeError(f"{path}: not enough mem entries {len(vals)} < {count}")
    return vals[:count]


def read_memh_uhex_lines(path: str) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    out = []
    for ln in lines:
        s = ln.strip().lower()
        if not s:
            continue
        if s.startswith("//"):
            continue
        s = re.sub(r"[^0-9a-f]", "", s)
        if s:
            out.append(s)
    return out


def write_hex_u16_line(f, v: int):
    f.write(f"{(int(v) & 0xFFFF):04x}\n")


def write_hex_u32_line(f, v: int):
    f.write(f"{(int(v) & 0xFFFFFFFF):08x}\n")


def rounding_shift_right(x: torch.Tensor, shift: int) -> torch.Tensor:
    if int(shift) <= 0:
        return x
    s = int(shift)
    return (x + (1 << (s - 1))) >> s


def quant_w_i16(w: torch.Tensor, r_w: int) -> torch.Tensor:
    s = float(1 << int(r_w))
    q = torch.round(w.to(torch.float32) * s).to(torch.int64)
    q = torch.clamp(q, -32768, 32767).to(torch.int16)
    return q


def quant_b_i32(b: torch.Tensor, r_b: int) -> torch.Tensor:
    s = float(1 << int(r_b))
    q = torch.round(b.to(torch.float32) * s).to(torch.int64)
    q = torch.clamp(q, -2147483648, 2147483647).to(torch.int32)
    return q


def quant_i16_qF(x: torch.Tensor, Fq: int) -> torch.Tensor:
    s = float(1 << int(Fq))
    q = torch.round(x.to(torch.float32) * s).to(torch.int64)
    q = torch.clamp(q, -32768, 32767).to(torch.int16)
    return q


def dequant_i16_qF(q: torch.Tensor, Fq: int, device: torch.device) -> torch.Tensor:
    return (q.to(device=device, dtype=torch.float32) / float(1 << int(Fq)))


def save_txt_i16_nchw_qF(path: str, q_i16: torch.Tensor, Fq: int):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = q_i16.detach().cpu()
    if q.ndim == 3:
        q = q.unsqueeze(0)
    N, C, H, W = q.shape
    if N != 1:
        raise RuntimeError("only support N=1")
    with p.open("w", encoding="utf-8") as f:
        f.write(f"dtype=int16 N=1 C={C} H={H} W={W} F={int(Fq)}\n")
        f.write("pix_idx och val_hex val_dec\n")
        for yy in range(H):
            for xx in range(W):
                pix_idx = yy * W + xx
                for oc in range(C):
                    v = int(q[0, oc, yy, xx].item())
                    u = v & 0xFFFF
                    f.write(f"{pix_idx} {oc} 0x{u:04X} {v}\n")




def save_mem_i16_chw(path: str, q_i16_chw: torch.Tensor):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = q_i16_chw.detach().cpu()
    if q.ndim != 3:
        raise RuntimeError("save_mem_i16_chw expects [C,H,W]")
    C, H, W = q.shape
    with p.open("w", encoding="utf-8") as f:
        for yy in range(H):
            for xx in range(W):
                for oc in range(C):
                    v = int(q[oc, yy, xx].item())
                    write_hex_u16_line(f, v)

def load_mem_i16_chw(path: str, C: int, H: int, W: int) -> torch.Tensor:
    vals = read_memh_signed(str(path), 16, int(C) * int(H) * int(W))
    t = torch.tensor(vals, dtype=torch.int16).view(H, W, C).permute(2, 0, 1).contiguous()
    return t

def dump_q_i16_memtxt_reload(name: str, q_i16_chw: torch.Tensor, Fq: int, out_dir: Path) -> torch.Tensor:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_txt_i16_nchw_qF(str(out_dir / f"{name}.txt"), q_i16_chw.detach().cpu().unsqueeze(0), int(Fq))
    save_mem_i16_chw(str(out_dir / f"{name}.mem"), q_i16_chw)
    C, H, W = int(q_i16_chw.shape[0]), int(q_i16_chw.shape[1]), int(q_i16_chw.shape[2])
    q2 = load_mem_i16_chw(str(out_dir / f"{name}.mem"), C=C, H=H, W=W).to(device=q_i16_chw.device)
    return q2

def read_memh_unsigned(path: str, bits: int, count: int) -> List[int]:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    txt = re.sub(r"//.*", "", txt)
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.S)
    toks = re.split(r"[\s,]+", txt)
    vals = []
    for t in toks:
        tt = t.strip()
        if not tt:
            continue
        if tt.startswith("@"):
            continue
        if tt.lower().startswith("0x"):
            tt = tt[2:]
        try:
            v = int(tt, 16)
        except Exception:
            try:
                v = int(tt, 10)
            except Exception:
                continue
        mask = (1 << bits) - 1
        v = v & mask
        vals.append(int(v))
        if len(vals) >= count:
            break
    if len(vals) < count:
        raise RuntimeError(f"{path}: need {count} words, got {len(vals)}")
    return vals

def _f32_to_u32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]

def _u32_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(u) & 0xFFFFFFFF))[0]

def save_mem_cplx16_nchw(path: str, z: torch.Tensor, Fz: int):
    # 写单个 mem：每个复数两行 16-bit：re, im（顺序固定）
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    zz = z.detach().cpu().to(torch.complex64)
    if zz.ndim == 3:
        zz = zz.unsqueeze(0)
    if zz.ndim != 4:
        raise RuntimeError("save_mem_cplx16_nchw expects [N,C,H,Wc] or [C,H,Wc]")
    N, C, H, Wc = zz.shape
    if N != 1:
        raise RuntimeError("only support N=1")

    re = zz.real.to(torch.float32)
    im = zz.imag.to(torch.float32)
    qre = quant_i16_qF(re, int(Fz))
    qim = quant_i16_qF(im, int(Fz))

    with p.open("w", encoding="utf-8") as f:
        for yy in range(H):
            for xx in range(Wc):
                for oc in range(C):
                    write_hex_u16_line(f, int(qre[0, oc, yy, xx].item()))
                    write_hex_u16_line(f, int(qim[0, oc, yy, xx].item()))

def load_mem_cplx16_nchw(path: str, C: int, H: int, Wc: int, device: torch.device, Fz: int) -> torch.Tensor:
    # 对应 save_mem_cplx16_nchw：两行一组(re,im)
    cnt = int(C) * int(H) * int(Wc)
    vals = read_memh_signed(str(path), 16, 2 * cnt)
    re_list = vals[0::2]
    im_list = vals[1::2]

    re_t = torch.tensor(re_list, dtype=torch.float32).view(H, Wc, C).permute(2, 0, 1).contiguous()
    im_t = torch.tensor(im_list, dtype=torch.float32).view(H, Wc, C).permute(2, 0, 1).contiguous()
    s = float(1 << int(Fz))
    re_t = re_t / s
    im_t = im_t / s
    z = torch.complex(re_t, im_t).unsqueeze(0).to(device=device).to(torch.complex64)
    return z

def save_txt_cplx16_nchw(path: str, z: torch.Tensor, Fz: int):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    zz = z.detach().cpu().to(torch.complex64)
    if zz.ndim == 3:
        zz = zz.unsqueeze(0)
    N, C, H, Wc = zz.shape
    if N != 1:
        raise RuntimeError("only support N=1")

    re = zz.real.to(torch.float32)
    im = zz.imag.to(torch.float32)
    qre = quant_i16_qF(re, int(Fz))
    qim = quant_i16_qF(im, int(Fz))

    with p.open("w", encoding="utf-8") as f:
        f.write(f"dtype=complex_qi16 Fz={int(Fz)} N=1 C={C} H={H} Wc={Wc}\n")
        f.write("k_idx och re_f32 im_f32 re_i16_hex im_i16_hex\n")
        for yy in range(H):
            for xx in range(Wc):
                k_idx = yy * Wc + xx
                for oc in range(C):
                    re0 = float(zz[0, oc, yy, xx].real.item())
                    im0 = float(zz[0, oc, yy, xx].imag.item())
                    rqi = int(qre[0, oc, yy, xx].item()) & 0xFFFF
                    iqi = int(qim[0, oc, yy, xx].item()) & 0xFFFF
                    f.write(f"{k_idx} {oc} {re0:.9g} {im0:.9g} 0x{rqi:04X} 0x{iqi:04X}\n")

def dump_cplx16_memtxt_reload(name: str, z: torch.Tensor, out_dir: Path, device: torch.device, Fz: int) -> torch.Tensor:
    out_dir.mkdir(parents=True, exist_ok=True)
    z4 = z.unsqueeze(0) if z.ndim == 3 else z
    _, C, H, Wc = z4.shape
    save_txt_cplx16_nchw(str(out_dir / f"{name}.txt"), z4, int(Fz))
    save_mem_cplx16_nchw(str(out_dir / f"{name}.mem"), z4, int(Fz))
    z2 = load_mem_cplx16_nchw(str(out_dir / f"{name}.mem"), C=int(C), H=int(H), Wc=int(Wc), device=device, Fz=int(Fz))
    return z2


def fold_hw_norm_into_conv(conv_w: torch.Tensor, conv_b: torch.Tensor, mean: List[float], std: List[float]):
    mean_t = torch.tensor(mean, dtype=conv_w.dtype, device=conv_w.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=conv_w.dtype, device=conv_w.device).view(1, 3, 1, 1)
    scale = (1.0 / 255.0) / std_t
    shift = (-mean_t) / std_t
    w_new = conv_w * scale
    b_new = conv_b + (conv_w * shift).sum(dim=(1, 2, 3))
    return w_new, b_new


def fuse_bn_into_conv(conv_w: torch.Tensor, conv_b: torch.Tensor, bn: nn.BatchNorm2d):
    w = conv_w
    b = conv_b
    gamma = bn.weight.to(dtype=w.dtype, device=w.device)
    beta = bn.bias.to(dtype=w.dtype, device=w.device)
    rm = bn.running_mean.to(dtype=w.dtype, device=w.device)
    rv = bn.running_var.to(dtype=w.dtype, device=w.device)
    eps = torch.tensor(float(bn.eps), dtype=w.dtype, device=w.device)
    std = torch.sqrt(rv + eps)
    s = (gamma / std).view(-1, 1, 1, 1)
    w2 = w * s
    b2 = (b - rm) * (gamma / std) + beta
    return w2, b2


def fold_pre_bn_into_conv(conv_w: torch.Tensor, conv_b: torch.Tensor, bn_in: nn.BatchNorm2d):
    w = conv_w
    b = conv_b
    gamma = bn_in.weight.to(dtype=w.dtype, device=w.device)
    beta = bn_in.bias.to(dtype=w.dtype, device=w.device)
    rm = bn_in.running_mean.to(dtype=w.dtype, device=w.device)
    rv = bn_in.running_var.to(dtype=w.dtype, device=w.device)
    eps = torch.tensor(float(bn_in.eps), dtype=w.dtype, device=w.device)
    std = torch.sqrt(rv + eps)
    mul = (gamma / std).view(1, -1, 1, 1)
    add = (beta - rm * (gamma / std)).view(1, -1, 1, 1)
    w2 = w * mul
    b2 = b + (w * add).sum(dim=(1, 2, 3))
    return w2, b2


def fold_out_channel_scale_into_conv(conv_w: torch.Tensor, conv_b: torch.Tensor, out_scale: torch.Tensor):
    s = out_scale.to(dtype=conv_w.dtype, device=conv_w.device).view(-1, 1, 1, 1)
    w2 = conv_w * s
    b2 = conv_b * out_scale.to(dtype=conv_b.dtype, device=conv_b.device)
    return w2, b2


def auto_r_w_from_weight(w_f: torch.Tensor, r_w_max: int = 15, min_r: int = 0, target: float = 32767.0) -> int:
    w = w_f.detach().abs().max().item()
    if not math.isfinite(w) or w <= 0.0:
        return int(min_r)
    r = int(math.floor(math.log2(target / w)))
    r = max(int(min_r), min(int(r_w_max), int(r)))
    return int(r)


def export_stemconv_mem(out_dir: str, w_f: torch.Tensor, b_f: torch.Tensor, r_w: int, F_act: int):
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    wq = quant_w_i16(w_f.detach().cpu(), r_w)
    bq = quant_b_i32(b_f.detach().cpu(), r_w)
    for bank in range(27):
        ic = bank // 9
        pos = bank % 9
        kh = pos // 3
        kw = pos % 3
        vals = wq[:, ic, kh, kw].to(torch.int64).tolist()
        p = outdir / f"wbank{bank:02d}.mem"
        with p.open("w", encoding="utf-8") as f:
            for q in vals:
                write_hex_u16_line(f, q)
    p = outdir / "bias64_i32.mem"
    with p.open("w", encoding="utf-8") as f:
        for q in bq.to(torch.int64).tolist():
            write_hex_u32_line(f, q)
    meta = {"r_w": int(r_w), "F_act": int(F_act), "shift_out": int(int(r_w) - int(F_act))}
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(outdir)


def export_pointwise_bank128_mem(out_dir: str, W_f_2d: torch.Tensor, b_f_1d: torch.Tensor, r_w: int, F_in: int, F_out: int, bank_size: int = 8):
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    W = W_f_2d.detach().cpu()
    b = b_f_1d.detach().cpu()
    oc = int(W.shape[0])
    ic = int(W.shape[1])
    if ic % bank_size != 0:
        raise RuntimeError(f"ic={ic} must be multiple of bank_size={bank_size}")
    nb = ic // bank_size
    wq = quant_w_i16(W, r_w).to(torch.int16)
    bq = quant_b_i32(b, int(r_w) + int(F_in)).to(torch.int32)
    for grp in range(nb):
        p = outdir / f"wbank{grp:02d}.mem"
        with p.open("w", encoding="utf-8") as f:
            for o in range(oc):
                ws = []
                base = grp * bank_size
                for k in range(bank_size):
                    ws.append(int(wq[o, base + (bank_size - 1 - k)].item()))
                line = "".join([f"{(w & 0xFFFF):04x}" for w in ws])
                f.write(line + "\n")
    p = outdir / f"bias{oc:02d}_i32.mem"
    with p.open("w", encoding="utf-8") as f:
        for q in bq.to(torch.int64).tolist():
            write_hex_u32_line(f, q)
    meta = {
        "r_w": int(r_w),
        "F_in": int(F_in),
        "F_out": int(F_out),
        "shift_out": int(int(r_w) + int(F_in) - int(F_out)),
        "pack_order": "lsb",
        "bank_size": int(bank_size),
        "oc": int(oc),
        "ic": int(ic),
        "kind": "pw",
    }
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(outdir)


def export_dw5x5_4x128_mem(out_dir: str, W_f: torch.Tensor, b_f: torch.Tensor, r_w: int, F_in: int, F_out: int):
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    W = W_f.detach().cpu()
    b = b_f.detach().cpu()
    oc = int(W.shape[0])
    if tuple(W.shape[1:]) != (1, 5, 5):
        raise RuntimeError(f"expect DW weight shape (C,1,5,5), got {tuple(W.shape)}")
    wq = quant_w_i16(W, r_w).to(torch.int16)
    bq = quant_b_i32(b, int(r_w) + int(F_in)).to(torch.int32)
    banks = [outdir / f"dw_wbank{bi:02d}_128.mem" for bi in range(4)]
    fhs = [p.open("w", encoding="utf-8") for p in banks]
    try:
        for c in range(oc):
            w25 = []
            for kh in range(5):
                for kw in range(5):
                    w25.append(int(wq[c, 0, kh, kw].item()))
            while len(w25) < 32:
                w25.append(0)
            for bi in range(4):
                ws = w25[bi * 8:(bi + 1) * 8]
                ws_rev = list(reversed(ws))
                line = "".join([f"{(w & 0xFFFF):04x}" for w in ws_rev])
                fhs[bi].write(line + "\n")
    finally:
        for fh in fhs:
            fh.close()
    p = outdir / f"dw_b{oc:02d}_i32.mem"
    with p.open("w", encoding="utf-8") as f:
        for q in bq.to(torch.int64).tolist():
            write_hex_u32_line(f, q)
    meta = {
        "r_w": int(r_w),
        "F_in": int(F_in),
        "F_out": int(F_out),
        "shift_out": int(int(r_w) + int(F_in) - int(F_out)),
        "oc": int(oc),
        "k": 5,
        "kind": "dw",
        "bank_words_per_ch": 4,
        "weights_per_word": 8,
        "weights_padded": 32,
        "pack_order": "lsb",
        "files": [p.name for p in banks] + [p.name],
    }
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(outdir)


def export_conv2x2_stride2_bank128_mem(out_dir: str, W_f: torch.Tensor, b_f: torch.Tensor, r_w: int, F_in: int, F_out: int, bank_size: int = 8):
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    W = W_f.detach().cpu()
    b = b_f.detach().cpu()
    oc, ic, kh, kw = W.shape
    if kh != 2 or kw != 2:
        raise RuntimeError(f"expect 2x2 weight, got {kh}x{kw}")
    if ic % bank_size != 0:
        raise RuntimeError(f"ic={ic} must be multiple of bank_size={bank_size}")
    nb = ic // bank_size
    wq = quant_w_i16(W, int(r_w)).to(torch.int16)
    bq = quant_b_i32(b, int(r_w) + int(F_in)).to(torch.int32)
    for ty in range(2):
        for tx in range(2):
            tap = ty * 2 + tx
            Wtap = wq[:, :, ty, tx]
            for grp in range(nb):
                p = outdir / f"wbank_t{tap:02d}_g{grp:02d}.mem"
                with p.open("w", encoding="utf-8") as f:
                    base = grp * bank_size
                    for o in range(oc):
                        ws = []
                        for k in range(bank_size):
                            ws.append(int(Wtap[o, base + (bank_size - 1 - k)].item()))
                        line = "".join([f"{(w & 0xFFFF):04x}" for w in ws])
                        f.write(line + "\n")
    p = outdir / f"bias{oc:02d}_i32.mem"
    with p.open("w", encoding="utf-8") as f:
        for q in bq.to(torch.int64).tolist():
            write_hex_u32_line(f, q)
    meta = {
        "kind": "down2x2",
        "k": 2,
        "stride": 2,
        "pad": 0,
        "r_w": int(r_w),
        "F_in": int(F_in),
        "F_out": int(F_out),
        "shift_out": int(int(r_w) + int(F_in) - int(F_out)),
        "bank_size": int(bank_size),
        "oc": int(oc),
        "ic": int(ic),
    }
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(outdir)


def read_pointwise_128b(mem_dir: str, oc: int, ic: int, bank_size: int = 8) -> torch.Tensor:
    mem_dir = Path(mem_dir)
    if ic % bank_size != 0:
        raise RuntimeError(f"ic={ic} must be multiple of bank_size={bank_size}")
    nb = ic // bank_size
    w = torch.zeros((oc, ic), dtype=torch.int16)
    for grp in range(nb):
        p = mem_dir / f"wbank{grp:02d}.mem"
        lines = read_memh_uhex_lines(str(p))
        if len(lines) < oc:
            raise RuntimeError(f"{p}: not enough lines {len(lines)} < oc={oc}")
        for o in range(oc):
            line = lines[o].zfill(32)
            ws = []
            for k in range(8):
                h = line[k * 4:(k + 1) * 4]
                v = int(h, 16) & 0xFFFF
                if v >= 0x8000:
                    v -= 0x10000
                ws.append(v)
            for k in range(8):
                w[o, grp * 8 + k] = int(ws[7 - k])
    return w


def read_dw_4banks_128b(mem_dir: str, oc: int) -> torch.Tensor:
    mem_dir = Path(mem_dir)
    bank_paths = [mem_dir / f"dw_wbank{bi:02d}_128.mem" for bi in range(4)]
    bank_lines = [read_memh_uhex_lines(str(p)) for p in bank_paths]
    for bi in range(4):
        if len(bank_lines[bi]) < oc:
            raise RuntimeError(f"{bank_paths[bi]}: not enough lines {len(bank_lines[bi])} < oc={oc}")
    w = torch.zeros((oc, 32), dtype=torch.int16)
    for c in range(oc):
        all32 = []
        for bi in range(4):
            line = bank_lines[bi][c].zfill(32)
            ws = []
            for k in range(8):
                h = line[k * 4:(k + 1) * 4]
                v = int(h, 16) & 0xFFFF
                if v >= 0x8000:
                    v -= 0x10000
                ws.append(v)
            ws = list(reversed(ws))
            all32.extend(ws)
        w[c] = torch.tensor(all32, dtype=torch.int16)
    return w


def read_conv2x2_stride2_tapbanks(mem_dir: str, oc: int, ic: int, bank_size: int = 8) -> torch.Tensor:
    mem_dir = Path(mem_dir)
    if ic % bank_size != 0:
        raise RuntimeError(f"ic={ic} must be multiple of bank_size={bank_size}")
    nb = ic // bank_size
    w = torch.zeros((4, oc, ic), dtype=torch.int16)
    for tap in range(4):
        for grp in range(nb):
            p = mem_dir / f"wbank_t{tap:02d}_g{grp:02d}.mem"
            lines = read_memh_uhex_lines(str(p))
            if len(lines) < oc:
                raise RuntimeError(f"{p}: not enough lines {len(lines)} < oc={oc}")
            for o in range(oc):
                line = lines[o].zfill(32)
                ws = []
                for k in range(8):
                    h = line[k * 4:(k + 1) * 4]
                    v = int(h, 16) & 0xFFFF
                    if v >= 0x8000:
                        v -= 0x10000
                    ws.append(v)
                for k in range(8):
                    w[tap, o, grp * 8 + k] = int(ws[7 - k])
    return w


def conv3x3_from_mem_qF(pool_u8_3x32x32: torch.Tensor, wbanks_27x64: List[List[int]], bias64_i32: List[int], r_w: int, F_act: int):
    x = pool_u8_3x32x32.to(torch.int64)
    Rp = torch.zeros((34, 34), dtype=torch.int64)
    Gp = torch.zeros((34, 34), dtype=torch.int64)
    Bp = torch.zeros((34, 34), dtype=torch.int64)
    Rp[1:33, 1:33] = x[0]
    Gp[1:33, 1:33] = x[1]
    Bp[1:33, 1:33] = x[2]
    tapsR, tapsG, tapsB = [], [], []
    for kh in range(3):
        for kw in range(3):
            tapsR.append(Rp[kh:kh + 32, kw:kw + 32])
            tapsG.append(Gp[kh:kh + 32, kw:kw + 32])
            tapsB.append(Bp[kh:kh + 32, kw:kw + 32])
    shift_out = int(int(r_w) - int(F_act))
    if shift_out < 0:
        raise RuntimeError(f"shift_out < 0: r_w={r_w}, F_act={F_act}")
    out_q = torch.zeros((64, 32, 32), dtype=torch.int16)
    for oc in range(64):
        acc = torch.full((32, 32), int(bias64_i32[oc]), dtype=torch.int64)
        for i in range(9):
            wr = int(wbanks_27x64[i][oc])
            wg = int(wbanks_27x64[9 + i][oc])
            wb = int(wbanks_27x64[18 + i][oc])
            if wr != 0:
                acc += wr * tapsR[i]
            if wg != 0:
                acc += wg * tapsG[i]
            if wb != 0:
                acc += wb * tapsB[i]
        acc = torch.clamp(acc, min=0)
        acc2 = rounding_shift_right(acc, shift_out) if shift_out > 0 else acc
        acc2 = torch.clamp(acc2, min=0, max=32767).to(torch.int16)
        out_q[oc] = acc2
    return out_q, shift_out


def pointwise1x1_from_mem_qF(x_q_i16_CxHxW: torch.Tensor, w_q_i16_oc_ic: torch.Tensor, b_q_i32: torch.Tensor,
                            F_in: int, r_w: int, F_out: int, relu: bool = False, clip16: bool = True):
    xq = x_q_i16_CxHxW.to(torch.int64)
    wq = w_q_i16_oc_ic.to(torch.int64)
    bq = b_q_i32.to(torch.int64).view(-1)
    C, H, W = int(xq.shape[0]), int(xq.shape[1]), int(xq.shape[2])
    OC = int(wq.shape[0])
    if int(wq.shape[1]) != C:
        raise RuntimeError(f"wq ic {int(wq.shape[1])} != xq C {C}")
    shift_out = int(int(r_w) + int(F_in) - int(F_out))
    if shift_out < 0:
        raise RuntimeError("shift_out < 0")
    x_flat = xq.reshape(C, -1)
    out = torch.zeros((OC, x_flat.shape[1]), dtype=torch.int64)
    for oc in range(OC):
        acc = torch.full((x_flat.shape[1],), int(bq[oc].item()), dtype=torch.int64)
        acc += (wq[oc].view(C, 1) * x_flat).sum(dim=0)
        if shift_out > 0:
            acc = rounding_shift_right(acc, shift_out)
        if relu:
            acc = torch.clamp(acc, min=0)
        out[oc] = acc
    if clip16:
        out16 = torch.clamp(out, -32768, 32767).to(torch.int16).view(OC, H, W)
        return out16, shift_out
    else:
        out32 = torch.clamp(out, -2147483648, 2147483647).to(torch.int32).view(OC, H, W)
        return out32, shift_out


def depthwise5x5_from_mem_qF(x_q_i16_CxHxW: torch.Tensor, w_q_i16_Cx32: torch.Tensor, b_q_i32: torch.Tensor,
                            F_in: int, r_w: int, F_out: int, relu: bool = False):
    xq = x_q_i16_CxHxW.to(torch.int64)
    wq = w_q_i16_Cx32.to(torch.int64)
    bq = b_q_i32.to(torch.int64).view(-1)
    C, H, W = int(xq.shape[0]), int(xq.shape[1]), int(xq.shape[2])
    shift_out = int(int(r_w) + int(F_in) - int(F_out))
    xp = torch.zeros((C, H + 4, W + 4), dtype=torch.int64)
    xp[:, 2:2 + H, 2:2 + W] = xq
    taps = []
    for kh in range(5):
        for kw in range(5):
            taps.append(xp[:, kh:kh + H, kw:kw + W])
    acc = bq.view(C, 1, 1).expand(C, H, W).clone()
    for t in range(25):
        wt = wq[:, t].view(C, 1, 1)
        acc += wt * taps[t]
    if shift_out > 0:
        acc = rounding_shift_right(acc, shift_out)
    if relu:
        acc = torch.clamp(acc, min=0)
    out_q = torch.clamp(acc, -32768, 32767).to(torch.int16)
    return out_q, shift_out


def conv2x2_stride2_from_mem_qF(x_q_i16_CxHxW: torch.Tensor, w_tap4_q_i16_oc_ic: torch.Tensor, b_q_i32: torch.Tensor,
                               F_in: int, r_w: int, F_out: int):
    xq = x_q_i16_CxHxW.to(torch.int64)
    wq = w_tap4_q_i16_oc_ic.to(torch.int64)
    bq = b_q_i32.to(torch.int64).view(-1)
    IC, H, W = int(xq.shape[0]), int(xq.shape[1]), int(xq.shape[2])
    OC = int(wq.shape[1])
    Ho, Wo = H // 2, W // 2
    x00 = xq[:, 0::2, 0::2].reshape(IC, -1)
    x01 = xq[:, 0::2, 1::2].reshape(IC, -1)
    x10 = xq[:, 1::2, 0::2].reshape(IC, -1)
    x11 = xq[:, 1::2, 1::2].reshape(IC, -1)
    out = bq.view(OC, 1).expand(OC, x00.shape[1]).clone()
    out += (wq[0] @ x00)
    out += (wq[1] @ x01)
    out += (wq[2] @ x10)
    out += (wq[3] @ x11)
    shift_out = int(int(r_w) + int(F_in) - int(F_out))
    if shift_out > 0:
        out = rounding_shift_right(out, shift_out)
    out16 = torch.clamp(out, -32768, 32767).to(torch.int16).view(OC, Ho, Wo)
    return out16, shift_out


def make_band_id_rfft2_like_shift(res: int, K: int, device: torch.device) -> torch.Tensor:
    H = int(res)
    W = int(res)
    Wc = W // 2 + 1
    cy = H // 2
    cx = W // 2
    y = torch.arange(H, device=device) - cy
    kx = torch.arange(Wc, device=device)
    x_shift = (kx + cx) % W
    x = x_shift - cx
    try:
        yy, xx = torch.meshgrid(y, x, indexing="ij")
    except TypeError:
        yy, xx = torch.meshgrid(y, x)
    r2 = yy.float() * yy.float() + xx.float() * xx.float()
    r2_max = float(r2.max().item() + 1e-6)
    edges = torch.linspace(0.0, r2_max, int(K) + 1, device=device)
    band_id = torch.bucketize(r2.reshape(-1), edges[1:-1], right=False).reshape(H, Wc).to(torch.long)
    return band_id



def apply_fold_stem(model_base: nn.Module, mean: List[float], std: List[float]) -> nn.Module:
    m = copy.deepcopy(model_base)
    device = next(m.parameters()).device
    m.eval()
    w = m.stem_conv.weight.detach().clone()
    b = m.stem_conv.bias.detach().clone()
    w1, b1 = fold_hw_norm_into_conv(w, b, mean, std)
    w2, b2 = fuse_bn_into_conv(w1, b1, m.stem_bn)
    m.hw_norm = nn.Identity()
    m.stem_bn = nn.Identity()
    with torch.no_grad():
        m.stem_conv.weight.copy_(w2)
        m.stem_conv.bias.copy_(b2)
    if hasattr(m, "refresh_buffers"):
        m.refresh_buffers(device)
    m.eval()
    if hasattr(m, "set_banddrop_p"):
        m.set_banddrop_p(0.0)
    return m


def load_calibrator(mod, calib_pt: str, device: torch.device):
    p = Path(calib_pt)
    if not p.exists():
        return None
    ck = torch.load(str(p), map_location="cpu")
    sd = None
    if isinstance(ck, dict):
        sd = ck.get("calibrator_state", None)
        if sd is None:
            sd = ck.get("model", None)
        if sd is None:
            sd = ck
    else:
        sd = ck
    num_classes = int(getattr(mod, "NUM_CLASSES", 8))
    cal = mod.LogitLinearCalibrator(num_classes).to(device)
    cal.load_state_dict(sd, strict=True)
    cal.eval()
    return cal


def export_block_ffn_folded(block: nn.Module, out_dir: str, r_w_max: int, Fq: int):
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not isinstance(block.ffn.se, nn.Identity):
        raise RuntimeError("se_ratio > 0 not supported in this exporter (your current train_ema uses se_ratio=0)")
    exp_conv = block.ffn.expand.conv
    exp_post_bn = block.ffn.expand.bn
    pre_bn = block.bn_b
    w_exp = exp_conv.weight.detach()
    b_exp = torch.zeros((w_exp.shape[0],), device=w_exp.device, dtype=w_exp.dtype)
    w_exp_pre, b_exp_pre = fold_pre_bn_into_conv(w_exp, b_exp, pre_bn)
    w_exp_f, b_exp_f = fuse_bn_into_conv(w_exp_pre, b_exp_pre, exp_post_bn)
    r_w_exp = auto_r_w_from_weight(w_exp_f, r_w_max=r_w_max)
    exp_dir = export_pointwise_bank128_mem(
        str(outdir / "expand_mem"),
        w_exp_f.cpu().squeeze(-1).squeeze(-1),
        b_exp_f.cpu(),
        r_w=int(r_w_exp),
        F_in=int(Fq),
        F_out=int(Fq),
        bank_size=8,
    )
    dw_conv = block.ffn.dw.rbr_reparam
    dw_bn = block.ffn.dw_bn
    w_dw = dw_conv.weight.detach()
    b_dw = dw_conv.bias.detach()
    w_dw_f, b_dw_f = fuse_bn_into_conv(w_dw, b_dw, dw_bn)
    r_w_dw = auto_r_w_from_weight(w_dw_f, r_w_max=r_w_max)
    dw_dir = export_dw5x5_4x128_mem(
        str(outdir / "dw5x5_mem"),
        w_dw_f.cpu(),
        b_dw_f.cpu(),
        r_w=int(r_w_dw),
        F_in=int(Fq),
        F_out=int(Fq),
    )
    pr_conv = block.ffn.project.conv
    pr_post_bn = block.ffn.project.bn
    w_pr = pr_conv.weight.detach()
    b_pr = torch.zeros((w_pr.shape[0],), device=w_pr.device, dtype=w_pr.dtype)
    w_pr_f, b_pr_f = fuse_bn_into_conv(w_pr, b_pr, pr_post_bn)
    w_pr_f2, b_pr_f2 = w_pr_f, b_pr_f
    r_w_pr = auto_r_w_from_weight(w_pr_f2, r_w_max=r_w_max)
    pr_dir = export_pointwise_bank128_mem(
        str(outdir / "project_mem"),
        w_pr_f2.cpu().squeeze(-1).squeeze(-1),
        b_pr_f2.cpu(),
        r_w=int(r_w_pr),
        F_in=int(Fq),
        F_out=int(Fq),
        bank_size=8,
    )
    meta = {
        "Fq": int(Fq),
        "r_w": {"expand": int(r_w_exp), "dw": int(r_w_dw), "project": int(r_w_pr)},
        "expand_dir": exp_dir,
        "dw_dir": dw_dir,
        "project_dir": pr_dir,
        "note": "bn_b folded into expand; expand.bn/dw_bn/project.bn fused; gamma_b folded into project",
    }
    (outdir / "ffn_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def export_block_ring_params(block: nn.Module, out_dir: str):
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    bn = block.bn_a
    bn_json = {
        "eps": float(bn.eps),
        "weight": [float(x) for x in bn.weight.detach().cpu().view(-1).tolist()],
        "bias": [float(x) for x in bn.bias.detach().cpu().view(-1).tolist()],
        "running_mean": [float(x) for x in bn.running_mean.detach().cpu().view(-1).tolist()],
        "running_var": [float(x) for x in bn.running_var.detach().cpu().view(-1).tolist()],
    }
    (outdir / "bn_a.json").write_text(json.dumps(bn_json, ensure_ascii=False), encoding="utf-8")
    torch.save({
        "s_amp": block.s_amp.detach().cpu().to(torch.float32),
        "s_phase": block.s_phase.detach().cpu().to(torch.float32),
    }, str(outdir / "ring_params.pt"))


def export_stage_shared(stage: nn.Module, out_dir: str):
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    amp = stage.shared.amp.detach().cpu().to(torch.float32)
    phase = stage.shared.phase.detach().cpu().to(torch.float32)
    torch.save({"amp": amp, "phase": phase, "C": int(amp.shape[0]), "K": int(amp.shape[1]), "res": int(stage.shared.res)}, str(outdir / "shared_amp_phase.pt"))

def export_full_net_mem(model_fold: nn.Module, out_root: str, r_w_max: int, Fq: int, calibrator: Optional[nn.Module]):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    meta = {"Fq": int(Fq), "r_w_max": int(r_w_max), "layers": {}}

    # ---------------- stem conv3x3 ----------------
    if not hasattr(model_fold, "stem_conv"):
        raise RuntimeError("model_fold has no stem_conv")
    stem_dir = out_root / "fold_stemconv_mem"
    stem_dir.mkdir(parents=True, exist_ok=True)

    r_stem = auto_r_w_from_weight(model_fold.stem_conv.weight.detach(), r_w_max=r_w_max)
    export_stemconv_mem(
        str(stem_dir),
        model_fold.stem_conv.weight.detach().cpu(),
        model_fold.stem_conv.bias.detach().cpu(),
        r_w=int(r_stem),
        F_act=int(Fq),
    )
    meta["layers"]["stem_conv3x3"] = {"dir": str(stem_dir), "r_w": int(r_stem), "F_act": int(Fq)}

    # ---------------- stem_to_C1 pw ----------------
    if not hasattr(model_fold, "stem_to_C1"):
        raise RuntimeError("model_fold has no stem_to_C1")
    pw0_dir = out_root / "fold_stem_to_C1_mem"
    pw0_dir.mkdir(parents=True, exist_ok=True)
    r_pw0 = auto_r_w_from_weight(model_fold.stem_to_C1.weight.detach(), r_w_max=r_w_max)
    export_pointwise_bank128_mem(
        str(pw0_dir),
        model_fold.stem_to_C1.weight.detach().cpu().squeeze(-1).squeeze(-1),
        model_fold.stem_to_C1.bias.detach().cpu(),
        r_w=int(r_pw0),
        F_in=int(Fq),
        F_out=int(Fq),
        bank_size=8,
    )
    meta["layers"]["stem_to_C1_pw"] = {"dir": str(pw0_dir), "r_w": int(r_pw0), "F_in": int(Fq), "F_out": int(Fq)}

    # ---------------- stages (auto-detect stage1..stageN) ----------------
    stage_names = []
    for i in range(1, 10):
        sn = f"stage{i}"
        if hasattr(model_fold, sn):
            stage_names.append(sn)
        else:
            break
    if not stage_names:
        raise RuntimeError("no stage1/stage2/... found on model_fold")

    for sn in stage_names:
        st = getattr(model_fold, sn)
        st_dir = out_root / sn
        st_dir.mkdir(parents=True, exist_ok=True)

        # 只 stage1 用 ring
        has_ring = (sn == "stage1") and hasattr(st, "shared")

        if has_ring:
            shared_dir = st_dir / "shared"
            shared_dir.mkdir(parents=True, exist_ok=True)
            export_stage_shared(st, str(shared_dir))

        blocks = list(getattr(st, "blocks", []))
        meta["layers"][sn] = {"depth": len(blocks), "blocks": [], "has_ring": bool(has_ring)}

        for bidx, blk in enumerate(blocks):
            blk_dir = st_dir / f"block{bidx}"
            blk_dir.mkdir(parents=True, exist_ok=True)

            if has_ring:
                ring_dir = blk_dir / "ring"
                ring_dir.mkdir(parents=True, exist_ok=True)
                export_block_ring_params(blk, str(ring_dir))

            ffn_dir = blk_dir / "ffn"
            ffn_dir.mkdir(parents=True, exist_ok=True)
            ffn_meta = export_block_ffn_folded(blk, str(ffn_dir), r_w_max=int(r_w_max), Fq=int(Fq))

            meta["layers"][sn]["blocks"].append(
                {"dir": str(blk_dir), "ffn": ffn_meta, "has_ring": bool(has_ring)}
            )

    # ---------------- downs (auto-detect down1/down2/...) ----------------
    for i in range(1, 10):
        dn_name = f"down{i}"
        if not hasattr(model_fold, dn_name):
            break
        dn = getattr(model_fold, dn_name)
        if not isinstance(dn, nn.Conv2d):
            raise RuntimeError(f"{dn_name} expected nn.Conv2d, got {type(dn)}")

        k = int(dn.kernel_size[0])
        s = int(dn.stride[0])
        p = int(dn.padding[0])
        if not (k == 2 and s == 2 and p == 0):
            raise RuntimeError(f"{dn_name} expect k2s2p0, got k/s/p={k}/{s}/{p}")

        w = dn.weight.detach()
        b = dn.bias.detach() if dn.bias is not None else torch.zeros((w.shape[0],), device=w.device, dtype=w.dtype)

        r_dn = auto_r_w_from_weight(w, r_w_max=r_w_max)
        dn_dir = out_root / dn_name
        dn_dir.mkdir(parents=True, exist_ok=True)
        export_conv2x2_stride2_bank128_mem(
            str(dn_dir),
            w.detach().cpu(),
            b.detach().cpu(),
            r_w=int(r_dn),
            F_in=int(Fq),
            F_out=int(Fq),
            bank_size=8,
        )
        meta["layers"][dn_name] = {"dir": str(dn_dir), "r_w": int(r_dn), "F_in": int(Fq), "F_out": int(Fq)}

    # ---------------- head fc ----------------
    if not hasattr(model_fold, "fc"):
        raise RuntimeError("model_fold has no fc")
    fc = model_fold.fc
    w_fc = fc.weight.detach().cpu().to(torch.float32)
    b_fc = fc.bias.detach().cpu().to(torch.float32) if fc.bias is not None else torch.zeros((w_fc.shape[0],), dtype=torch.float32)

    r_fc = auto_r_w_from_weight(w_fc, r_w_max=r_w_max)
    head_dir = out_root / "head"
    head_dir.mkdir(parents=True, exist_ok=True)
    fc_dir = head_dir / "fc_mem"
    fc_dir.mkdir(parents=True, exist_ok=True)
    export_pointwise_bank128_mem(
        str(fc_dir),
        w_fc,
        b_fc,
        r_w=int(r_fc),
        F_in=int(Fq),
        F_out=int(Fq),
        bank_size=8,
    )
    meta["layers"]["fc"] = {"dir": str(fc_dir), "r_w": int(r_fc), "F_in": int(Fq), "F_out": int(Fq)}

    # ---------------- optional calibrator ----------------
    if calibrator is not None:
        cal_dir = head_dir / "calibrator_mem"
        cal_dir.mkdir(parents=True, exist_ok=True)
        w_cal = calibrator.fc.weight.detach().cpu().to(torch.float32)
        b_cal = calibrator.fc.bias.detach().cpu().to(torch.float32)
        r_cal = auto_r_w_from_weight(w_cal, r_w_max=r_w_max)
        export_pointwise_bank128_mem(
            str(cal_dir),
            w_cal,
            b_cal,
            r_w=int(r_cal),
            F_in=int(Fq),
            F_out=int(Fq),
            bank_size=8,
        )
        meta["layers"]["calibrator"] = {"dir": str(cal_dir), "r_w": int(r_cal), "F_in": int(Fq), "F_out": int(Fq)}

    (out_root / "net_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta

def bn_forward_float(x_float: torch.Tensor, bn_json: Dict) -> torch.Tensor:
    w = torch.tensor(bn_json["weight"], dtype=torch.float32, device=x_float.device).view(1, -1, 1, 1)
    b = torch.tensor(bn_json["bias"], dtype=torch.float32, device=x_float.device).view(1, -1, 1, 1)
    rm = torch.tensor(bn_json["running_mean"], dtype=torch.float32, device=x_float.device).view(1, -1, 1, 1)
    rv = torch.tensor(bn_json["running_var"], dtype=torch.float32, device=x_float.device).view(1, -1, 1, 1)
    eps = float(bn_json["eps"])
    y = (x_float - rm) / torch.sqrt(rv + eps)
    y = y * w + b
    return y


@torch.no_grad()
def ringgf_forward_from_mem(x_q_i16: torch.Tensor, Fq: int, shared_amp: torch.Tensor, shared_phase: torch.Tensor,
                           s_amp: torch.Tensor, s_phase: torch.Tensor, band_id: torch.Tensor) -> torch.Tensor:
    device = band_id.device
    x = dequant_i16_qF(x_q_i16.unsqueeze(0), int(Fq), device=device)
    X = torch.fft.rfft2(x.float(), s=(x.shape[2], x.shape[3]), norm="ortho")
    amp = shared_amp.to(device=device, dtype=torch.float32)
    phase = shared_phase.to(device=device, dtype=torch.float32)
    sa = s_amp.to(device=device, dtype=torch.float32)
    sp = s_phase.to(device=device, dtype=torch.float32)
    A = amp * (1.0 + sa)
    P = phase + sp
    G_ck = torch.polar(A, P).to(torch.complex64)
    G_chw = G_ck[:, band_id]
    Y = X * G_chw.unsqueeze(0)
    y = torch.fft.irfft2(Y, s=(x.shape[2], x.shape[3]), norm="ortho")
    y_q = quant_i16_qF(y, int(Fq))[0]
    return y_q
def _first_exist_dir(base: Path, names):
    for n in names:
        p = base / n
        if p.exists():
            return p
    return None

def load_block_mem(block_dir: Path) -> Dict:
    ring_dir = block_dir / "ring"
    ffn_dir = block_dir / "ffn"

    has_ring = (ring_dir / "bn_a.json").exists() and (ring_dir / "ring_params.pt").exists()
    if has_ring:
        bn_a = json.loads((ring_dir / "bn_a.json").read_text(encoding="utf-8"))
        rp = torch.load(str(ring_dir / "ring_params.pt"), map_location="cpu")
        s_amp = rp["s_amp"]
        s_phase = rp["s_phase"]
    else:
        bn_a = None
        s_amp = None
        s_phase = None

    exp_dir = _first_exist_dir(ffn_dir, ["expand", "expand_mem"])
    dw_dir  = _first_exist_dir(ffn_dir, ["dw", "dw5x5_mem"])
    pr_dir  = _first_exist_dir(ffn_dir, ["project", "project_mem"])

    if exp_dir is None or dw_dir is None or pr_dir is None:
        raise RuntimeError(f"ffn subdirs missing under: {ffn_dir}")

    exp_meta = json.loads((exp_dir / "meta.json").read_text(encoding="utf-8"))
    dw_meta  = json.loads((dw_dir  / "meta.json").read_text(encoding="utf-8"))
    pr_meta  = json.loads((pr_dir  / "meta.json").read_text(encoding="utf-8"))

    w_exp = read_pointwise_128b(str(exp_dir), oc=int(exp_meta["oc"]), ic=int(exp_meta["ic"]), bank_size=8)
    b_exp = torch.tensor(read_memh_signed(str(exp_dir / f"bias{int(exp_meta['oc']):02d}_i32.mem"), 32, int(exp_meta["oc"])),
                         dtype=torch.int32)

    w_dw = read_dw_4banks_128b(str(dw_dir), oc=int(dw_meta["oc"]))
    b_dw = torch.tensor(read_memh_signed(str(dw_dir / f"dw_b{int(dw_meta['oc']):02d}_i32.mem"), 32, int(dw_meta["oc"])),
                        dtype=torch.int32)

    w_pr = read_pointwise_128b(str(pr_dir), oc=int(pr_meta["oc"]), ic=int(pr_meta["ic"]), bank_size=8)
    b_pr = torch.tensor(read_memh_signed(str(pr_dir / f"bias{int(pr_meta['oc']):02d}_i32.mem"), 32, int(pr_meta["oc"])),
                        dtype=torch.int32)

    return {
        "has_ring": bool(has_ring),
        "bn_a": bn_a,
        "s_amp": s_amp,
        "s_phase": s_phase,
        "exp": {"meta": exp_meta, "w": w_exp, "b": b_exp},
        "dw":  {"meta": dw_meta,  "w": w_dw,  "b": b_dw},
        "pr":  {"meta": pr_meta,  "w": w_pr,  "b": b_pr},
    }

def load_stage_shared(stage_shared_dir: Path) -> Dict:
    d = torch.load(str(stage_shared_dir / "shared_amp_phase.pt"), map_location="cpu")
    return {"amp": d["amp"], "phase": d["phase"], "C": int(d["C"]), "K": int(d["K"]), "res": int(d["res"])}


@torch.no_grad()
def load_pw_mem(pw_dir: Path):
    meta = json.loads((pw_dir / "meta.json").read_text(encoding="utf-8"))
    oc = int(meta["oc"]); ic = int(meta["ic"])
    w = read_pointwise_128b(str(pw_dir), oc=oc, ic=ic, bank_size=int(meta["bank_size"]))
    b = torch.tensor(read_memh_signed(str(pw_dir / f"bias{oc:02d}_i32.mem"), 32, oc), dtype=torch.int32)
    return {"meta": meta, "w": w, "b": b}
def diff_stats(a, b, eps: float = 1e-12):
    # a/b 可以是 torch.Tensor / list / numpy
    if a is None or b is None:
        return {"ok": False, "reason": "a_or_b_is_none"}

    ta = torch.as_tensor(a).detach().cpu().to(torch.float32)
    tb = torch.as_tensor(b).detach().cpu().to(torch.float32)

    if ta.shape != tb.shape:
        return {"ok": False, "reason": "shape_mismatch", "shape_a": list(ta.shape), "shape_b": list(tb.shape)}

    d = ta - tb
    ad = d.abs()

    out = {
        "ok": True,
        "shape": list(ta.shape),
        "max_abs": float(ad.max().item()) if ad.numel() else 0.0,
        "mean_abs": float(ad.mean().item()) if ad.numel() else 0.0,
        "rmse": float(torch.sqrt((d * d).mean()).item()) if d.numel() else 0.0,
        "l1": float(ad.sum().item()) if ad.numel() else 0.0,
    }

    if ta.numel() and ta.min().item() >= -1e-6 and tb.min().item() >= -1e-6:
        pa = ta.clamp_min(0.0)
        pb = tb.clamp_min(0.0)
        sa = pa.sum().item()
        sb = pb.sum().item()
        if sa > eps and sb > eps:
            pa = pa / sa
            pb = pb / sb
            kl = (pa * (pa.add(eps).log() - pb.add(eps).log())).sum()
            out["kl(a||b)"] = float(kl.item())

    return out

def load_down_mem(dn_dir: Path):
    meta = json.loads((dn_dir / "meta.json").read_text(encoding="utf-8"))
    oc = int(meta["oc"]); ic = int(meta["ic"])
    w = read_conv2x2_stride2_tapbanks(str(dn_dir), oc=oc, ic=ic, bank_size=int(meta["bank_size"]))
    b = torch.tensor(read_memh_signed(str(dn_dir / f"bias{oc:02d}_i32.mem"), 32, oc), dtype=torch.int32)
    return {"meta": meta, "w": w, "b": b}

def avgpool_hw_qF(x_q: torch.Tensor) -> torch.Tensor:
    x = x_q.to(torch.int64)
    C, H, W = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    denom = H * W
    s = x.reshape(C, -1).sum(dim=1)
    s = s + (denom // 2)
    y = (s // denom)
    y = torch.clamp(y, -32768, 32767).to(torch.int16)
    return y.view(C, 1, 1)


@torch.no_grad()
def forward_ref_full(model_fold: nn.Module, calibrator: Optional[nn.Module], x_float_b1: torch.Tensor):
    logits = model_fold(x_float_b1)
    if calibrator is not None:
        logits2 = calibrator(logits.float())
    else:
        logits2 = logits
    prob = F.softmax(logits2.float(), dim=-1)[0].detach().cpu()
    return logits.detach().cpu(), logits2.detach().cpu(), prob

@torch.no_grad()
def run_block_closedloop(x_q: torch.Tensor, Fq: int,
                         shared: Optional[Dict],
                         band_id: Optional[torch.Tensor],
                         blk_mem: Dict,
                         dump_dir: Path) -> torch.Tensor:
    dev = x_q.device
    dump_dir.mkdir(parents=True, exist_ok=True)

    x_q = dump_q_i16_memtxt_reload("00_in", x_q, int(Fq), dump_dir)

    has_ring = bool(blk_mem.get("has_ring", False)) and (shared is not None) and (band_id is not None)

    if has_ring:
        band_id = band_id.to(dev)
        shared_amp   = shared["amp"].to(dev, dtype=torch.float32)
        shared_phase = shared["phase"].to(dev, dtype=torch.float32)
        s_amp   = blk_mem["s_amp"].to(dev, dtype=torch.float32)
        s_phase = blk_mem["s_phase"].to(dev, dtype=torch.float32)

        x_float = dequant_i16_qF(x_q.unsqueeze(0), int(Fq), device=dev)
        xa = bn_forward_float(x_float, blk_mem["bn_a"])
        xa_q = quant_i16_qF(xa, int(Fq))[0].to(dev)
        xa_q = dump_q_i16_memtxt_reload("01_bn_a", xa_q, int(Fq), dump_dir)

        x2 = dequant_i16_qF(xa_q.unsqueeze(0), int(Fq), device=dev)
        X = torch.fft.rfft2(x2.float(), s=(x2.shape[2], x2.shape[3]), norm="ortho")
        Ffft = min(15, int(Fq) + 4)
        X = dump_cplx16_memtxt_reload("02a_rfft2_X", X, dump_dir, device=dev, Fz=Ffft)

        A = shared_amp * (1.0 + s_amp)
        P = shared_phase + s_phase
        G_ck = torch.polar(A, P).to(torch.complex64)              # [C,K]
        G_chw = G_ck[:, band_id].unsqueeze(0)                     # [1,C,H,Wc]
        G_chw = dump_cplx16_memtxt_reload("02b_g_chw", G_chw, dump_dir, device=dev, Fz=Ffft)

        Y = X * G_chw
        Y = dump_cplx16_memtxt_reload("02c_mul_Y", Y, dump_dir, device=dev, Fz=Ffft)

        y = torch.fft.irfft2(Y, s=(x2.shape[2], x2.shape[3]), norm="ortho")
        y_q = quant_i16_qF(y, int(Fq))[0].to(dev)
        y_q = dump_q_i16_memtxt_reload("02_ringgf", y_q, int(Fq), dump_dir)

        x1_q = (x_q.to(torch.int32) + y_q.to(torch.int32))
        x1_q = torch.clamp(x1_q, -32768, 32767).to(torch.int16)
        x1_q = dump_q_i16_memtxt_reload("04_add_a", x1_q, int(Fq), dump_dir)
    else:
        x1_q = dump_q_i16_memtxt_reload("04_add_a", x_q, int(Fq), dump_dir)

    expm = blk_mem["exp"]["meta"]
    dwm  = blk_mem["dw"]["meta"]
    prm  = blk_mem["pr"]["meta"]

    exp_w = blk_mem["exp"]["w"].to(dev)
    exp_b = blk_mem["exp"]["b"].to(dev)
    dw_w  = blk_mem["dw"]["w"].to(dev)
    dw_b  = blk_mem["dw"]["b"].to(dev)
    pr_w  = blk_mem["pr"]["w"].to(dev)
    pr_b  = blk_mem["pr"]["b"].to(dev)

    y_exp_q, _ = pointwise1x1_from_mem_qF(
        x1_q, exp_w, exp_b,
        F_in=int(expm["F_in"]), r_w=int(expm["r_w"]), F_out=int(expm["F_out"]),
        relu=True, clip16=True
    )
    y_exp_q = dump_q_i16_memtxt_reload("05_expand_relu", y_exp_q.to(dev), int(Fq), dump_dir)

    y_dw_q, _ = depthwise5x5_from_mem_qF(
        y_exp_q, dw_w, dw_b,
        F_in=int(dwm["F_in"]), r_w=int(dwm["r_w"]), F_out=int(dwm["F_out"]),
        relu=True
    )
    y_dw_q = dump_q_i16_memtxt_reload("06_dw_relu", y_dw_q.to(dev), int(Fq), dump_dir)

    y_pr_q, _ = pointwise1x1_from_mem_qF(
        y_dw_q, pr_w, pr_b,
        F_in=int(prm["F_in"]), r_w=int(prm["r_w"]), F_out=int(prm["F_out"]),
        relu=False, clip16=True
    )
    y_pr_q = dump_q_i16_memtxt_reload("11_project", y_pr_q.to(dev), int(Fq), dump_dir)

    out_q = (x1_q.to(torch.int32) + y_pr_q.to(torch.int32))
    out_q = torch.clamp(out_q, -32768, 32767).to(torch.int16)
    out_q = dump_q_i16_memtxt_reload("12_out", out_q, int(Fq), dump_dir)
    return out_q
@torch.no_grad()
def forward_mem_full(net_meta: Dict,
                     out_root: Path,
                     img_path: str,
                     H: int,
                     W: int,
                     Fq: int,
                     device: torch.device):
    import json, re
    out_root = Path(out_root)

    dumps_root = out_root / "dumps_mem"
    dumps_root.mkdir(parents=True, exist_ok=True)

    x_u8 = load_img_rgb888(img_path, H=int(H), W=int(W))
    pool_u8 = pool2x2_u8(x_u8)

    stem_dir = out_root / "fold_stemconv_mem"
    stem_meta = json.loads((stem_dir / "meta.json").read_text(encoding="utf-8"))
    wbanks = [read_memh_signed(str(stem_dir / f"wbank{i:02d}.mem"), 16, 64) for i in range(27)]
    bias64 = read_memh_signed(str(stem_dir / "bias64_i32.mem"), 32, 64)

    stem_q, _ = conv3x3_from_mem_qF(
        pool_u8, wbanks, bias64,
        r_w=int(stem_meta["r_w"]),
        F_act=int(stem_meta["F_act"])
    )
    save_txt_i16_nchw_qF(str(dumps_root / "stem_out_q.txt"), stem_q.unsqueeze(0), int(Fq))

    pw0 = load_pw_mem(out_root / "fold_stem_to_C1_mem")
    x_q, _ = pointwise1x1_from_mem_qF(
        stem_q, pw0["w"], pw0["b"],
        F_in=int(pw0["meta"]["F_in"]),
        r_w=int(pw0["meta"]["r_w"]),
        F_out=int(pw0["meta"]["F_out"]),
        relu=False,
        clip16=True
    )
    save_txt_i16_nchw_qF(str(dumps_root / "stem_to_C1_out_q.txt"), x_q.unsqueeze(0), int(Fq))

    stage_names = sorted(
        [k for k in net_meta["layers"].keys() if re.fullmatch(r"stage\d+", str(k))],
        key=lambda s: int(str(s)[5:])
    )

    for stage_name in stage_names:
        st_dir = out_root / stage_name
        has_ring = bool(net_meta["layers"][stage_name].get("has_ring", False))

        if has_ring:
            shared = load_stage_shared(st_dir / "shared")
            band_id = make_band_id_rfft2_like_shift(shared["res"], shared["K"], device=device)
        else:
            shared = None
            band_id = None

        blocks = net_meta["layers"][stage_name]["blocks"]
        for bidx in range(len(blocks)):
            blk_dir = st_dir / f"block{bidx}"
            blk_mem = load_block_mem(blk_dir)
            dump_dir = dumps_root / stage_name / f"block{bidx}"
            x_q = run_block_closedloop(x_q, int(Fq), shared, band_id, blk_mem, dump_dir)

        if stage_name == "stage1" and ("down1" in net_meta["layers"]):
            dn = load_down_mem(out_root / "down1")
            x_q, _ = conv2x2_stride2_from_mem_qF(
                x_q, dn["w"], dn["b"],
                F_in=int(dn["meta"]["F_in"]),
                r_w=int(dn["meta"]["r_w"]),
                F_out=int(dn["meta"]["F_out"])
            )
            save_txt_i16_nchw_qF(str(dumps_root / "down1_out_q.txt"), x_q.unsqueeze(0), int(Fq))

        if stage_name == "stage2" and ("down2" in net_meta["layers"]):
            dn = load_down_mem(out_root / "down2")
            x_q, _ = conv2x2_stride2_from_mem_qF(
                x_q, dn["w"], dn["b"],
                F_in=int(dn["meta"]["F_in"]),
                r_w=int(dn["meta"]["r_w"]),
                F_out=int(dn["meta"]["F_out"])
            )
            save_txt_i16_nchw_qF(str(dumps_root / "down2_out_q.txt"), x_q.unsqueeze(0), int(Fq))

    gap_q = avgpool_hw_qF(x_q)
    save_txt_i16_nchw_qF(str(dumps_root / "gap_q.txt"), gap_q.unsqueeze(0), int(Fq))

    fc = load_pw_mem(out_root / "head" / "fc_mem")
    logits_q32, _ = pointwise1x1_from_mem_qF(
        gap_q.view(gap_q.shape[0], 1, 1),
        fc["w"], fc["b"],
        F_in=int(fc["meta"]["F_in"]),
        r_w=int(fc["meta"]["r_w"]),
        F_out=int(fc["meta"]["F_out"]),
        relu=False,
        clip16=False
    )
    logits_q32 = logits_q32.view(-1).to(torch.int32)
    (dumps_root / "logits_q32.json").write_text(
        json.dumps([int(x) for x in logits_q32.tolist()], ensure_ascii=False),
        encoding="utf-8"
    )

    if "calibrator" in net_meta["layers"]:
        cal = load_pw_mem(out_root / "head" / "calibrator_mem")
        logits_i16 = torch.clamp(logits_q32, -32768, 32767).to(torch.int16)
        logits_cal_q32, _ = pointwise1x1_from_mem_qF(
            logits_i16.view(-1, 1, 1),
            cal["w"], cal["b"],
            F_in=int(cal["meta"]["F_in"]),
            r_w=int(cal["meta"]["r_w"]),
            F_out=int(cal["meta"]["F_out"]),
            relu=False,
            clip16=False
        )
        logits_cal_q32 = logits_cal_q32.view(-1).to(torch.int32)
    else:
        logits_cal_q32 = logits_q32

    (dumps_root / "logits_cal_q32.json").write_text(
        json.dumps([int(x) for x in logits_cal_q32.tolist()], ensure_ascii=False),
        encoding="utf-8"
    )

    logits_f = logits_cal_q32.to(torch.float32) / float(1 << int(Fq))
    prob = F.softmax(logits_f, dim=0).detach().cpu()

    return logits_q32.detach().cpu(), logits_cal_q32.detach().cpu(), prob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_py", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--calib_pt", type=str, default="")
    ap.add_argument("--img", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--Fq", type=int, default=8)
    ap.add_argument("--r_w_max", type=int, default=15)
    ap.add_argument("--no_ref", action="store_true", default=False)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    mod = load_module_from_path(args.model_py)

    MEAN = list(getattr(mod, "MEAN"))
    STD = list(getattr(mod, "STD"))
    num_classes = int(getattr(mod, "NUM_CLASSES", 8))

    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck.get("model_state", ck.get("model", ck))
    ck_args = ck.get("args", {}) if isinstance(ck, dict) else {}

    se_ratio = float(ck_args.get("se_ratio", 0.0))
    ffn_exp = int(ck_args.get("ffn_expansion", 3))
    beta_init = float(ck_args.get("beta_init", 0.0))
    drop_path = float(ck_args.get("drop_path", 0.0))

    model_base = mod.BloodMNISTBandHWNet_MBV3(
        num_classes=num_classes,
        drop_path_rate=float(drop_path),
        ffn_expansion=int(ffn_exp),
        se_ratio=float(se_ratio),
    ).to(device)
    model_base.load_state_dict(sd, strict=True)
    model_base.refresh_buffers(device)
    model_base.eval()
    if hasattr(model_base, "set_banddrop_p"):
        model_base.set_banddrop_p(0.0)

    model_fold = apply_fold_stem(model_base, MEAN, STD).to(device)
    model_fold.eval()
    if hasattr(model_fold, "switch_to_deploy"):
        model_fold.switch_to_deploy()
    if hasattr(model_fold, "refresh_buffers"):
        model_fold.refresh_buffers(device)
    if hasattr(model_fold, "set_banddrop_p"):
        model_fold.set_banddrop_p(0.0)

    calib_pt = args.calib_pt.strip()
    if not calib_pt:
        cand = Path(args.ckpt).parent / "calibrator.pt"
        calib_pt = str(cand) if cand.exists() else ""
    calibrator = load_calibrator(mod, calib_pt, device) if calib_pt else None

    net_meta = export_full_net_mem(model_fold, str(out_root), r_w_max=int(args.r_w_max), Fq=int(args.Fq), calibrator=calibrator)

    x_u8 = load_img_rgb888(args.img, H=int(args.H), W=int(args.W))
    x_float = x_u8.to(torch.float32).unsqueeze(0).to(device, non_blocking=True)

    if not args.no_ref:
        ref_logits_raw, ref_logits_cal, ref_prob = forward_ref_full(model_fold, calibrator, x_float)
    else:
        ref_logits_raw, ref_logits_cal, ref_prob = None, None, None

    mem_logits_raw_q, mem_logits_cal_q, mem_prob = forward_mem_full(net_meta, out_root, args.img, args.H, args.W, int(args.Fq), device=device)

    summary = {
        "img": str(args.img),
        "out_root": str(out_root),
        "device": str(args.device),
        "Fq": int(args.Fq),
        "r_w_max": int(args.r_w_max),
        "use_calibrator": bool(calibrator is not None),
        "pred_mem": int(mem_prob.argmax().item()),
        "prob_mem": [float(x) for x in mem_prob.tolist()],
    }

    if ref_prob is not None:
        summary.update({
            "pred_ref_full": int(ref_prob.argmax().item()),
            "prob_ref_full": [float(x) for x in ref_prob.tolist()],
            "diff_prob_ref_vs_mem": diff_stats(ref_prob, mem_prob),
        })

    (out_root / "summary_fullnet_mem.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

#驱动代码 PS C:\Users\kangk\PycharmProjects\PythonProject> python export_and_test.py --model_py "C:\Users\kangk\PycharmProjects\PythonProject\train_ema.py" --ckpt ".\run_main_only_no_gate_cs_1_2\best_main.pt" --calib_pt ".\run_main_only_no_gate_cs_1_2\calibrator.pt" --img "C:\Users\kangk\Desktop\000036.coe" --out_root "C:\Users\kangk\Desktop\fullnet_mem_run_000036" --device cuda --Fq 8 --r_w_max 15