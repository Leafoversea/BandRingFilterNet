import argparse
import contextlib
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def param_report(model: nn.Module) -> Dict[str, int]:
    report: Dict[str, int] = {}
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        report[name] = int(n)
    report["TOTAL"] = int(count_params(model))
    return report


def save_param_report(report: Dict[str, int], path: str):
    lines = ["# Parameter Report (trainable)\n"]
    total = report.get("TOTAL", 0)
    for k, v in report.items():
        if k == "TOTAL":
            continue
        lines.append(f"{k:>20s}: {v / 1e6:.6f} M ({v})\n")
    lines.append("-" * 40 + "\n")
    lines.append(f"{'TOTAL':>20s}: {total / 1e6:.6f} M ({total})\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def get_autocast_ctx(amp: bool, amp_dtype: str):
    if (not amp) or (not torch.cuda.is_available()):
        return contextlib.nullcontext()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    dtype = dtype_map.get(str(amp_dtype).lower(), torch.bfloat16)
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


class AdjustableRandomErasing:
    def __init__(self, p: float = 0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0):
        self.p = float(p)
        self._re = transforms.RandomErasing(p=1.0, scale=scale, ratio=ratio, value=value)

    def set_p(self, p: float):
        self.p = float(p)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0:
            return img
        if torch.rand(1).item() < self.p:
            return self._re(img)
        return img


class WarmupWrapper:
    def __init__(self, optimizer, base_sched, warmup_epochs: int, base_lr: float):
        self.opt = optimizer
        self.base_sched = base_sched
        self.warmup_epochs = int(warmup_epochs)
        self.base_lr = float(base_lr)
        self.epoch = 0

    def step(self):
        if self.epoch < self.warmup_epochs:
            lr = self.base_lr * (self.epoch + 1) / max(1, self.warmup_epochs)
            for g in self.opt.param_groups:
                g["lr"] = lr
        else:
            self.base_sched.step()
        self.epoch += 1


def tail_anneal_factor(epoch: int, total_epochs: int, tail_epochs: int) -> float:
    if tail_epochs <= 0:
        return 1.0
    start = total_epochs - tail_epochs
    if epoch <= start:
        return 1.0
    t = (epoch - start) / max(1, tail_epochs)
    t = max(0.0, min(1.0, t))
    return 0.5 * (1.0 + math.cos(math.pi * t))


def one_hot_smooth(targets: torch.Tensor, num_classes: int, eps: float):
    if targets.ndim == 2:
        return targets
    t = F.one_hot(targets, num_classes).float()
    if eps <= 0:
        return t
    return (1.0 - eps) * t + eps / float(num_classes)


def cross_entropy_with_soft_targets(logits: torch.Tensor, soft_targets: torch.Tensor):
    logp = F.log_softmax(logits, dim=-1)
    return -(soft_targets * logp).sum(dim=-1).mean()


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float):
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


def freq_smooth_loss(amps: List[torch.Tensor]):
    loss = 0.0
    for amp in amps:
        if amp.shape[1] > 1:
            loss = loss + F.mse_loss(amp[:, :-1], amp[:, 1:], reduction="mean")
    return loss


def group_lasso_loss(groups: List[List[torch.Tensor]]):
    loss = 0.0
    cnt = 0
    for group_list in groups:
        for param in group_list:
            loss = loss + param.norm(2)
            cnt += 1
    return loss / max(cnt, 1)


@torch.no_grad()
def update_ema_params(ema_model: nn.Module, model: nn.Module, decay: float):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


@torch.no_grad()
def sync_bn_buffers(dst: nn.Module, src: nn.Module):
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    for k in dst_sd.keys():
        if ("running_mean" in k) or ("running_var" in k) or ("num_batches_tracked" in k):
            dst_sd[k].copy_(src_sd[k])
    dst.load_state_dict(dst_sd, strict=True)


def get_param_groups(model: nn.Module, weight_decay: float):
    decay = []
    no_decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class MixupCutmixCollate:
    def __init__(self, num_classes: int, mixup_alpha: float = 0.8, cutmix_alpha: float = 1.0, p_mix: float = 0.5):
        self.num_classes = int(num_classes)
        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.p_mix = float(p_mix)

    def set_p_mix(self, p: float):
        self.p_mix = float(max(0.0, min(1.0, p)))

    def __call__(self, batch):
        images, targets = zip(*batch)
        images = torch.stack(images, dim=0)
        targets = torch.tensor(targets, dtype=torch.long)

        if torch.rand(1).item() < self.p_mix:
            if torch.rand(1).item() < 0.5:
                images, targets = self._mixup(images, targets)
            else:
                images, targets = self._cutmix(images, targets)

        return images, targets

    def _mixup(self, images, targets):
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
        index = torch.randperm(images.size(0))
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_targets = lam * F.one_hot(targets, self.num_classes).float() + (1 - lam) * F.one_hot(
            targets[index], self.num_classes
        ).float()
        return mixed_images, mixed_targets

    def _cutmix(self, images, targets):
        lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample().item()
        index = torch.randperm(images.size(0))
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        lam_adj = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        mixed_targets = lam_adj * F.one_hot(targets, self.num_classes).float() + (1 - lam_adj) * F.one_hot(
            targets[index], self.num_classes
        ).float()
        return mixed_images, mixed_targets

    def _rand_bbox(self, size, lam):
        W, H = size[-1], size[-2]
        cut_rat = math.sqrt(1.0 - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = torch.randint(W, (1,)).item(), torch.randint(H, (1,)).item()
        bbx1, bby1 = max(0, cx - cut_w // 2), max(0, cy - cut_h // 2)
        bbx2, bby2 = min(W, cx + cut_w // 2), min(H, cy + cut_h // 2)
        return bbx1, bby1, bbx2, bby2


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(int(num_channels), eps=float(eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.ln(x_perm)
        return x_norm.permute(0, 3, 1, 2)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.base_drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


def set_droppath_factor(model: nn.Module, factor: float):
    factor = float(max(0.0, min(1.0, factor)))
    for m in model.modules():
        if isinstance(m, DropPath):
            if not hasattr(m, "base_drop_prob"):
                m.base_drop_prob = float(m.drop_prob)
            m.drop_prob = float(m.base_drop_prob) * factor


def complex_from_amp_phase(amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    # NOTE: fixed dtype (original had a typo torch.complex6464)
    return torch.polar(amp.float(), phase.float()).to(torch.complex64)


@dataclass
class RingGFConfig:
    C: int
    K: int
    res: int
    band_edge_mode: str = "area"


def make_band_id_rfft2(res: int, K: int, device: torch.device, edge_mode: str = "area") -> torch.Tensor:
    H = int(res)
    W = int(res)
    Wc = W // 2 + 1

    ky = torch.arange(H, device=device)
    ky = torch.where(ky < (H // 2), ky, ky - H)
    kx = torch.arange(Wc, device=device)

    try:
        yy, xx = torch.meshgrid(ky, kx, indexing="ij")
    except TypeError:
        yy, xx = torch.meshgrid(ky, kx)

    rr = torch.sqrt(yy.float() ** 2 + xx.float() ** 2)
    r_max = float(rr.max().item() + 1e-6)

    edge_mode = str(edge_mode).lower()
    if edge_mode == "linear":
        edges = torch.linspace(0.0, r_max, int(K) + 1, device=device)
    elif edge_mode in ["area", "equal_area", "sqrt"]:
        edges = r_max * torch.sqrt(torch.linspace(0.0, 1.0, int(K) + 1, device=device))
    else:
        raise ValueError(f"Unknown edge_mode={edge_mode}. Use 'linear' or 'area'.")

    band_id = torch.bucketize(rr.reshape(-1), edges[1:-1], right=False).reshape(H, Wc).to(torch.long)
    return band_id


class RingGFShared(nn.Module):
    def __init__(self, cfg: RingGFConfig):
        super().__init__()
        self.C = int(cfg.C)
        self.K = int(cfg.K)
        self.res = int(cfg.res)
        self.band_edge_mode = str(cfg.band_edge_mode)

        self.amp = nn.Parameter(torch.ones(self.C, self.K))
        self.phase = nn.Parameter(torch.zeros(self.C, self.K))

        self.register_buffer("band_id", torch.zeros(self.res, self.res // 2 + 1, dtype=torch.long), persistent=False)
        self.register_buffer("band_keep", torch.ones(self.K), persistent=False)

        self.refresh_buffers(torch.device("cpu"))

    def refresh_buffers(self, device: torch.device):
        self.band_id = make_band_id_rfft2(self.res, self.K, device=device, edge_mode=self.band_edge_mode)
        self.band_keep = torch.ones(self.K, device=device)

    def set_band_keep(self, keep: torch.Tensor):
        assert keep.numel() == self.K
        self.band_keep = keep.to(self.amp.device)

    def get_amp_phase(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.amp, self.phase

    def forward(self, x: torch.Tensor, s_amp: torch.Tensor, s_phase: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert h == self.res and w == self.res
        assert c == self.C

        X = torch.fft.rfft2(x.float(), s=(h, w), norm="ortho")

        A = self.amp.float() * (1.0 + s_amp.float()) * self.band_keep.float().unsqueeze(0)
        P = self.phase.float() + s_phase.float()
        G_ck = torch.polar(A, P).to(torch.complex64)

        bid = self.band_id.to(device=x.device)
        G_chw = G_ck[:, bid]

        Y = X * G_chw.unsqueeze(0)
        y = torch.fft.irfft2(Y, s=(h, w), norm="ortho")
        return y.to(x.dtype)


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    W = conv.weight
    bias = torch.zeros(W.size(0), device=W.device, dtype=W.dtype) if conv.bias is None else conv.bias
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    std = torch.sqrt(var + eps)
    scale = (gamma / std).reshape(-1, 1, 1, 1)
    W_fused = W * scale
    b_fused = beta + (bias - mean) * (gamma / std)
    return W_fused, b_fused


def _fuse_identity_bn_to_conv(channels: int, k: int, bn: nn.BatchNorm2d, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    W = torch.zeros((channels, channels, k, k), device=device, dtype=dtype)
    W[torch.arange(channels), torch.arange(channels), k // 2, k // 2] = 1.0
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    std = torch.sqrt(var + eps)
    scale = (gamma / std).reshape(-1, 1, 1, 1)
    W_fused = W * scale
    b_fused = beta + (0.0 - mean) * (gamma / std)
    return W_fused, b_fused


def _pad_1x1_to_kxk(kernel: torch.Tensor, k: int) -> torch.Tensor:
    if kernel.size(2) == k and kernel.size(3) == k:
        return kernel
    pad = k // 2
    return F.pad(kernel, [pad, pad, pad, pad])


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 3, stride: int = 1, deploy: bool = False):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.k = int(k) if int(k) % 2 == 1 else int(k) + 1
        self.stride = int(stride)
        self.padding = self.k // 2
        self.deploy = bool(deploy)
        self.act = nn.GELU()

        if self.deploy:
            self.rbr_reparam = nn.Conv2d(
                self.in_channels, self.out_channels, kernel_size=self.k, stride=self.stride, padding=self.padding, bias=True
            )
            self.rbr_dense = None
            self.rbr_1x1 = None
            self.rbr_identity = None
        else:
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.k, stride=self.stride, padding=self.padding, bias=False),
                nn.BatchNorm2d(self.out_channels),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, padding=0, bias=False),
                nn.BatchNorm2d(self.out_channels),
            )
            self.rbr_identity = None
            if self.in_channels == self.out_channels and self.stride == 1:
                self.rbr_identity = nn.BatchNorm2d(self.in_channels)
            self.rbr_reparam = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.act(self.rbr_reparam(x))
        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)
        return self.act(out)

    @torch.no_grad()
    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.deploy:
            return self.rbr_reparam.weight, self.rbr_reparam.bias

        Wk, bk = _fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        W1, b1 = _fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        W1 = _pad_1x1_to_kxk(W1, self.k)

        if self.rbr_identity is not None:
            Wid, bid = _fuse_identity_bn_to_conv(self.out_channels, self.k, self.rbr_identity, device=Wk.device, dtype=Wk.dtype)
        else:
            Wid = torch.zeros_like(Wk)
            bid = torch.zeros_like(bk)

        W = Wk + W1 + Wid
        b = bk + b1 + bid
        return W, b

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return
        W, b = self.get_equivalent_kernel_bias()
        conv = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=self.k, stride=self.stride, padding=self.padding, bias=True
        ).to(device=W.device, dtype=W.dtype)
        conv.weight.data.copy_(W)
        conv.bias.data.copy_(b.to(dtype=W.dtype, device=W.device))
        self.rbr_reparam = conv
        self.rbr_dense = None
        self.rbr_1x1 = None
        self.rbr_identity = None
        self.deploy = True


def _fuse_dwconv_bn(dw: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    W = dw.weight
    bias = torch.zeros(W.size(0), device=W.device, dtype=W.dtype) if dw.bias is None else dw.bias
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    std = torch.sqrt(var + eps)
    scale = (gamma / std).reshape(-1, 1, 1, 1)
    W_fused = W * scale
    b_fused = beta + (bias - mean) * (gamma / std)
    return W_fused, b_fused


def _fuse_identity_bn_dw(channels: int, k: int, bn: nn.BatchNorm2d, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    W = torch.zeros((channels, 1, k, k), device=device, dtype=dtype)
    W[:, 0, k // 2, k // 2] = 1.0
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    std = torch.sqrt(var + eps)
    scale = (gamma / std).reshape(-1, 1, 1, 1)
    W_fused = W * scale
    b_fused = beta + (0.0 - mean) * (gamma / std)
    return W_fused, b_fused


class RepDepthwiseConv(nn.Module):
    def __init__(self, channels: int, k: int = 5, deploy: bool = False):
        super().__init__()
        self.channels = int(channels)
        self.k = int(k) if int(k) % 2 == 1 else int(k) + 1
        self.deploy = bool(deploy)

        if self.deploy:
            self.rbr_reparam = nn.Conv2d(
                self.channels, self.channels, kernel_size=self.k, padding=self.k // 2, groups=self.channels, bias=True
            )
            self.rbr_dw = None
            self.rbr_bn = None
            self.rbr_idbn = None
        else:
            self.rbr_dw = nn.Conv2d(
                self.channels, self.channels, kernel_size=self.k, padding=self.k // 2, groups=self.channels, bias=False
            )
            self.rbr_bn = nn.BatchNorm2d(self.channels)
            self.rbr_idbn = nn.BatchNorm2d(self.channels)
            self.rbr_reparam = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.rbr_reparam(x)
        a = self.rbr_bn(self.rbr_dw(x))
        b = self.rbr_idbn(x)
        return a + b

    @torch.no_grad()
    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.deploy:
            return self.rbr_reparam.weight, self.rbr_reparam.bias
        W_dw, b_dw = _fuse_dwconv_bn(self.rbr_dw, self.rbr_bn)
        W_id, b_id = _fuse_identity_bn_dw(self.channels, self.k, self.rbr_idbn, device=W_dw.device, dtype=W_dw.dtype)
        W = W_dw + W_id
        b = b_dw + b_id
        return W, b

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return
        W, b = self.get_equivalent_kernel_bias()
        conv = nn.Conv2d(
            self.channels, self.channels, kernel_size=self.k, padding=self.k // 2, groups=self.channels, bias=True
        ).to(device=W.device, dtype=W.dtype)
        conv.weight.data.copy_(W)
        conv.bias.data.copy_(b.to(dtype=W.dtype, device=W.device))
        self.rbr_reparam = conv
        self.rbr_dw = None
        self.rbr_bn = None
        self.rbr_idbn = None
        self.deploy = True


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, se_ratio: float = 0.2):
        super().__init__()
        hidden = max(8, int(channels * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)
        self.gate = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.gate(y)
        return x * y


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, act: str = "hswish"):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        if act == "hswish":
            self.act = nn.Hardswish()
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class MBV3FFN(nn.Module):
    def __init__(self, C: int, expansion: int = 3, dw_k: int = 5, se_ratio: float = 0.2, act: str = "hswish"):
        super().__init__()
        mid = int(C * expansion)
        self.expand = ConvBNAct(C, mid, k=1, s=1, act=act)
        self.dw = RepDepthwiseConv(mid, k=dw_k, deploy=False)
        self.dw_act = nn.Hardswish() if act == "hswish" else (nn.ReLU(inplace=True) if act == "relu" else nn.GELU())
        self.se = SqueezeExcite(mid, se_ratio=se_ratio)
        self.project = nn.Conv2d(mid, C, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.dw(x)
        x = self.dw_act(x)
        x = self.se(x)
        x = self.project(x)
        return x

    @torch.no_grad()
    def switch_to_deploy(self):
        self.dw.switch_to_deploy()


class GFBlock(nn.Module):
    """
    Proposal A implemented here:
    - Add branch-level gate beta_a for the frequency branch.
    - beta_a is initialized to 0, so the frequency branch is "off" at init and can be learned to turn on.
    - Keep LayerScale (gamma_a) and DropPath as-is.
    """
    def __init__(
        self,
        C: int,
        shared: RingGFShared,
        drop_path_prob: float,
        layerscale_init: float = 1e-3,
        ffn_expansion: int = 3,
        se_ratio: float = 0.2,
        beta_init: float = 0.0,
    ):
        super().__init__()
        self.shared = shared
        self.norm_a = LayerNorm2d(C)
        self.norm_b = LayerNorm2d(C)
        self.branch_ffn = MBV3FFN(C, expansion=ffn_expansion, dw_k=5, se_ratio=se_ratio, act="hswish")
        self.act_gf = nn.GELU()
        self.droppath = DropPath(drop_path_prob)
        self.s_amp = nn.Parameter(torch.zeros(C, shared.K))
        self.s_phase = nn.Parameter(torch.zeros(C, shared.K))
        self.gamma_a = nn.Parameter(torch.ones(1, C, 1, 1) * layerscale_init)
        self.gamma_b = nn.Parameter(torch.ones(1, C, 1, 1) * layerscale_init)

        # NEW: branch-level gate for freq branch (per-channel, broadcastable)
        self.beta_a = nn.Parameter(torch.ones(1, C, 1, 1) * float(beta_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xa = self.norm_a(x)
        xa = self.shared(xa, self.s_amp, self.s_phase)
        xa = self.act_gf(xa)

        # Apply LayerScale and branch gate before DropPath
        xa = self.beta_a * (self.gamma_a * xa)
        xa = self.droppath(xa)
        x = x + xa

        xb = self.norm_b(x)
        xb = self.branch_ffn(xb)
        xb = self.droppath(self.gamma_b * xb)
        x = x + xb
        return x

    @torch.no_grad()
    def get_beta_stats(self) -> Dict[str, float]:
        b = self.beta_a.detach()
        return {
            "beta_mean": float(b.mean().item()),
            "beta_abs_mean": float(b.abs().mean().item()),
            "beta_min": float(b.min().item()),
            "beta_max": float(b.max().item()),
        }

    @torch.no_grad()
    def switch_to_deploy(self):
        self.branch_ffn.switch_to_deploy()


class Stage(nn.Module):
    def __init__(
        self,
        C: int,
        K: int,
        res: int,
        depth: int,
        dp_rates: List[float],
        ffn_expansion: int,
        se_ratio: float,
        band_edge_mode: str,
        beta_init: float = 0.0,
    ):
        super().__init__()
        self.shared = RingGFShared(RingGFConfig(C=C, K=K, res=res, band_edge_mode=band_edge_mode))
        self.blocks = nn.Sequential(*[
            GFBlock(
                C=C,
                shared=self.shared,
                drop_path_prob=dp_rates[i],
                layerscale_init=1e-3,
                ffn_expansion=ffn_expansion,
                se_ratio=se_ratio,
                beta_init=beta_init,
            )
            for i in range(depth)
        ])
        self.K = int(K)
        self.banddrop_p = 0.0

    def refresh_buffers(self, device: torch.device):
        self.shared.refresh_buffers(device)

    def set_banddrop_p(self, p: float):
        self.banddrop_p = float(p)

    @torch.no_grad()
    def _sample_band_keep(self, p: float, device: torch.device):
        keep = torch.ones(self.K, device=device)
        if p <= 0:
            return keep
        m = torch.rand(self.K, device=device) < p
        if m.any():
            keep[m] = torch.rand(int(m.sum().item()), device=device) * 0.5
        return keep

    def get_amp_phase(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.shared.get_amp_phase()

    @torch.no_grad()
    def get_band_importance(self, mode: str = "mean_abs") -> List[float]:
        """
        Frequency-band importance report for this stage.
        Default: mean(|amp_c,k|) over channels c, per band k.
        """
        amp, _phase = self.shared.get_amp_phase()
        if mode == "l2":
            imp = torch.sqrt((amp.detach() ** 2).mean(dim=0) + 1e-12)
        else:
            imp = amp.detach().abs().mean(dim=0)
        return [float(x.item()) for x in imp]

    @torch.no_grad()
    def get_beta_stats(self) -> Dict[str, float]:
        """
        Aggregate beta stats across blocks in this stage.
        """
        stats = {"beta_mean": 0.0, "beta_abs_mean": 0.0, "beta_min": 0.0, "beta_max": 0.0}
        n = 0
        bmins = []
        bmaxs = []
        for b in self.blocks:
            if hasattr(b, "beta_a"):
                s = b.get_beta_stats()
                stats["beta_mean"] += s["beta_mean"]
                stats["beta_abs_mean"] += s["beta_abs_mean"]
                bmins.append(s["beta_min"])
                bmaxs.append(s["beta_max"])
                n += 1
        if n > 0:
            stats["beta_mean"] /= n
            stats["beta_abs_mean"] /= n
            stats["beta_min"] = float(min(bmins))
            stats["beta_max"] = float(max(bmaxs))
        return stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.banddrop_p > 0:
            self.shared.set_band_keep(self._sample_band_keep(self.banddrop_p, x.device))
        else:
            self.shared.set_band_keep(torch.ones(self.K, device=x.device))
        return self.blocks(x)

    @torch.no_grad()
    def switch_to_deploy(self):
        for b in self.blocks:
            if hasattr(b, "switch_to_deploy"):
                b.switch_to_deploy()


class DownsampleRD(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class RingBandFilterNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        drop_path_rate: float = 0.15,
        ffn_expansion: int = 3,
        se_ratio: float = 0.2,
        band_edge_mode: str = "area",
        beta_init: float = 0.0,
    ):
        super().__init__()
        self.banddrop_p = 0.0
        self.band_edge_mode = str(band_edge_mode)

        self.stem = RepVGGBlock(3, 64, k=3, stride=1, deploy=False)

        depths = [3, 6, 8, 3]
        Ks = [8, 6, 4, 2]
        reses = [32, 16, 8, 4]
        Cs = [32, 48, 64, 80]

        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        i = 0

        self.stem_to_C1 = nn.Conv2d(64, Cs[0], kernel_size=1, stride=1, bias=True)

        self.stage1 = Stage(
            C=Cs[0], K=Ks[0], res=reses[0], depth=depths[0], dp_rates=dp_rates[i:i + depths[0]],
            ffn_expansion=ffn_expansion, se_ratio=se_ratio, band_edge_mode=self.band_edge_mode, beta_init=beta_init
        )
        i += depths[0]
        self.down1 = DownsampleRD(Cs[0], Cs[1])

        self.stage2 = Stage(
            C=Cs[1], K=Ks[1], res=reses[1], depth=depths[1], dp_rates=dp_rates[i:i + depths[1]],
            ffn_expansion=ffn_expansion, se_ratio=se_ratio, band_edge_mode=self.band_edge_mode, beta_init=beta_init
        )
        i += depths[1]
        self.down2 = DownsampleRD(Cs[1], Cs[2])

        self.stage3 = Stage(
            C=Cs[2], K=Ks[2], res=reses[2], depth=depths[2], dp_rates=dp_rates[i:i + depths[2]],
            ffn_expansion=ffn_expansion, se_ratio=se_ratio, band_edge_mode=self.band_edge_mode, beta_init=beta_init
        )
        i += depths[2]
        self.down3 = DownsampleRD(Cs[2], Cs[3])

        self.stage4 = Stage(
            C=Cs[3], K=Ks[3], res=reses[3], depth=depths[3], dp_rates=dp_rates[i:i + depths[3]],
            ffn_expansion=ffn_expansion, se_ratio=se_ratio, band_edge_mode=self.band_edge_mode, beta_init=beta_init
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(Cs[3], num_classes)

    def refresh_buffers(self, device: torch.device):
        self.stage1.refresh_buffers(device)
        self.stage2.refresh_buffers(device)
        self.stage3.refresh_buffers(device)
        self.stage4.refresh_buffers(device)

    def set_banddrop_p(self, p: float):
        self.banddrop_p = float(p)
        self.stage1.set_banddrop_p(p)
        self.stage2.set_banddrop_p(p)
        self.stage3.set_banddrop_p(p)
        self.stage4.set_banddrop_p(p)

    def get_all_amp_params(self) -> List[torch.Tensor]:
        return [
            self.stage1.get_amp_phase()[0],
            self.stage2.get_amp_phase()[0],
            self.stage3.get_amp_phase()[0],
            self.stage4.get_amp_phase()[0],
        ]

    def get_all_amp_groups(self) -> List[List[torch.Tensor]]:
        groups = []
        for amp in self.get_all_amp_params():
            K = amp.shape[1]
            groups.append([amp[:, k] for k in range(K)])
        return groups

    @torch.no_grad()
    def get_beta_report(self) -> Dict[str, Dict[str, float]]:
        return {
            "stage1": self.stage1.get_beta_stats(),
            "stage2": self.stage2.get_beta_stats(),
            "stage3": self.stage3.get_beta_stats(),
            "stage4": self.stage4.get_beta_stats(),
        }

    @torch.no_grad()
    def get_band_importance_report(self, mode: str = "mean_abs") -> Dict[str, List[float]]:
        return {
            "stage1": self.stage1.get_band_importance(mode=mode),
            "stage2": self.stage2.get_band_importance(mode=mode),
            "stage3": self.stage3.get_band_importance(mode=mode),
            "stage4": self.stage4.get_band_importance(mode=mode),
        }

    @torch.no_grad()
    def switch_to_deploy(self):
        self.stem.switch_to_deploy()
        self.stage1.switch_to_deploy()
        self.stage2.switch_to_deploy()
        self.stage3.switch_to_deploy()
        self.stage4.switch_to_deploy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stem_to_C1(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return self.fc(x)


class DummyScaler:
    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

    def unscale_(self, optimizer):
        pass

    def is_enabled(self):
        return False


def build_transforms():
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    eraser = AdjustableRandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=12),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        eraser,
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, test_tf, eraser


@torch.no_grad()
def recalibrate_bn_stats_for_avg(
    m: nn.Module,
    train_loader: DataLoader,
    args,
    mix_collate: MixupCutmixCollate,
    eraser: AdjustableRandomErasing,
    max_batches: int = 0,
):
    was_training = m.training

    old_p_mix = float(getattr(mix_collate, "p_mix", 0.0))
    old_p_erase = float(getattr(eraser, "p", 0.0))
    mix_collate.set_p_mix(0.0)
    eraser.set_p(0.0)

    if hasattr(m, "set_banddrop_p"):
        m.set_banddrop_p(0.0)
    set_droppath_factor(m, 0.0)

    m.train()

    bn_momentum: Dict[nn.Module, Optional[float]] = {}
    for mod in m.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            bn_momentum[mod] = mod.momentum
            mod.reset_running_stats()
            mod.momentum = None

    autocast_ctx = get_autocast_ctx(args.amp, args.amp_dtype)

    it = 0
    for images, _targets in train_loader:
        images = images.to(args.device, non_blocking=True)
        if args.channels_last and images.is_cuda:
            images = images.contiguous(memory_format=torch.channels_last)
        with autocast_ctx:
            _ = m(images)
        it += 1
        if max_batches > 0 and it >= max_batches:
            break

    for mod, mom in bn_momentum.items():
        mod.momentum = mom

    if not was_training:
        m.eval()

    mix_collate.set_p_mix(old_p_mix)
    eraser.set_p(old_p_erase)


@torch.no_grad()
def load_weight_averaged_state(
    dst: nn.Module,
    srcs: Dict[str, nn.Module],
    weights: Dict[str, float],
):
    items = []
    wsum = 0.0
    for k, m in srcs.items():
        w = float(weights.get(k, 0.0))
        if w > 0.0:
            items.append((k, m, w))
            wsum += w
    if wsum <= 0.0:
        return False

    src_sds = {k: m.state_dict() for k, m, _w in items}
    dst_sd = dst.state_dict()
    out_sd = {}

    first_key = items[0][0]
    for name in dst_sd.keys():
        t0 = src_sds[first_key][name]
        if t0.is_floating_point() or t0.is_complex():
            acc = None
            for k, _m, w in items:
                t = src_sds[k][name]
                if acc is None:
                    acc = t * w
                else:
                    acc = acc + t * w
            out_sd[name] = acc / wsum
        else:
            out_sd[name] = t0
    dst.load_state_dict(out_sd, strict=True)
    return True


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    ema_model: nn.Module,
    swa_model: nn.Module,
    train_loader: DataLoader,
    opt,
    scaler,
    scheduler,
    args,
    mix_collate: MixupCutmixCollate,
    eraser: AdjustableRandomErasing,
):
    model.train()

    tail = tail_anneal_factor(epoch, args.epochs, args.tail_anneal_epochs)

    mix_collate.set_p_mix(args.mix_p * tail)
    eraser.set_p(args.erase_p * tail)

    if hasattr(model, "set_banddrop_p"):
        model.set_banddrop_p(args.band_p * tail)
    set_droppath_factor(model, tail)

    lam_smooth = args.lambda_smooth * tail
    lam_group = args.lambda_group * tail
    label_smoothing = args.label_smoothing * tail
    kd_alpha = args.kd_alpha * tail

    if epoch <= 10:
        ema_decay = 0.90 + (args.ema_decay_min - 0.90) * (epoch / 10.0)
    else:
        ramp = min(1.0, (epoch - 10) / max(1.0, (args.epochs - 10)))
        ema_decay = args.ema_decay_min + (args.ema_decay_max - args.ema_decay_min) * ramp

    autocast_ctx = get_autocast_ctx(args.amp, args.amp_dtype)

    for images, targets in train_loader:
        images = images.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)

        if args.channels_last and images.is_cuda:
            images = images.contiguous(memory_format=torch.channels_last)

        if targets.ndim == 1:
            targets_soft = one_hot_smooth(targets, args.num_classes, eps=label_smoothing)
        else:
            targets_soft = targets

        opt.zero_grad(set_to_none=True)

        with autocast_ctx:
            logits = model(images)
            ce = cross_entropy_with_soft_targets(logits, targets_soft)

            loss_kd = torch.tensor(0.0, device=args.device)
            if epoch >= args.kd_start and kd_alpha > 0:
                with torch.no_grad():
                    tlogits = ema_model(images)
                loss_kd = kd_loss(logits, tlogits, T=args.kd_T)

            amps = model.get_all_amp_params()
            l_smooth = freq_smooth_loss(amps) if lam_smooth > 0 else torch.tensor(0.0, device=args.device)
            l_group = group_lasso_loss(model.get_all_amp_groups()) if lam_group > 0 else torch.tensor(0.0, device=args.device)

            loss = ce + kd_alpha * loss_kd + lam_smooth * l_smooth + lam_group * l_group

        scaler.scale(loss).backward()

        if hasattr(scaler, "unscale_"):
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(opt)
        scaler.update()

        update_ema_params(ema_model, model, ema_decay)

    sync_bn_buffers(ema_model, model)

    if scheduler is not None:
        scheduler.step()

    if epoch >= args.swa_start_epoch:
        swa_n = getattr(train_one_epoch, "swa_n", 0)
        alpha = 1.0 / (swa_n + 1.0)
        with torch.no_grad():
            for p_swa, p in zip(swa_model.parameters(), model.parameters()):
                p_swa.data.mul_(1.0 - alpha).add_(p.data, alpha=alpha)
        train_one_epoch.swa_n = swa_n + 1


@torch.no_grad()
def _collect_beta_and_band_reports(models: Dict[str, nn.Module], band_mode: str = "mean_abs") -> Dict[str, Dict]:
    """
    Collect beta stats + band importance report for each model that supports it.
    """
    out: Dict[str, Dict] = {}
    for k, m in models.items():
        rep: Dict = {}
        if hasattr(m, "get_beta_report"):
            rep["beta"] = m.get_beta_report()
        if hasattr(m, "get_band_importance_report"):
            rep["band_importance"] = m.get_band_importance_report(mode=band_mode)
        if rep:
            out[k] = rep
    return out


@torch.no_grad()
def evaluate(models: Dict[str, nn.Module], loader: DataLoader, args) -> Dict[str, float]:
    for m in models.values():
        m.eval()
        if hasattr(m, "set_banddrop_p"):
            m.set_banddrop_p(0.0)

    autocast_ctx = get_autocast_ctx(args.amp, args.amp_dtype)
    correct = {k: 0 for k in models.keys()}
    total = 0

    for images, targets in loader:
        images = images.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)
        if args.channels_last and images.is_cuda:
            images = images.contiguous(memory_format=torch.channels_last)

        total += targets.size(0)

        for k, m in models.items():
            with autocast_ctx:
                out = m(images)
            correct[k] += (out.argmax(dim=-1) == targets).sum().item()

    return {k: (v / total) * 100.0 for k, v in correct.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./cifar100")
    parser.add_argument("--save", type=str, default="./exp_c100_ring_rfft2_param_ens_beta")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--warmup_epochs", type=int, default=40)

    parser.add_argument("--drop_path", type=float, default=0.15)
    parser.add_argument("--band_p", type=float, default=0.15)

    parser.add_argument("--lambda_smooth", type=float, default=0.1)
    parser.add_argument("--lambda_group", type=float, default=1e-5)

    parser.add_argument("--mix_p", type=float, default=0.4)
    parser.add_argument("--mixup_alpha", type=float, default=0.8)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)

    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--erase_p", type=float, default=0.25)

    parser.add_argument("--kd_T", type=float, default=3.0)
    parser.add_argument("--kd_alpha", type=float, default=0.30)
    parser.add_argument("--kd_start", type=int, default=100)

    parser.add_argument("--ema_decay_min", type=float, default=0.99)
    parser.add_argument("--ema_decay_max", type=float, default=0.999)

    parser.add_argument("--tail_anneal_epochs", type=int, default=20)
    parser.add_argument("--swa_start_epoch", type=int, default=400)

    parser.add_argument("--grad_clip", type=float, default=1.0)

    try:
        parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    except Exception:
        parser.add_argument("--amp", action="store_true", default=True)
        parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--ffn_expansion", type=int, default=3)
    parser.add_argument("--se_ratio", type=float, default=0.2)
    parser.add_argument("--band_edge_mode", type=str, default="area", choices=["area", "linear"])

    # NEW: beta init for Proposal A
    parser.add_argument("--beta_init", type=float, default=0.0)

    # band importance mode
    parser.add_argument("--band_importance_mode", type=str, default="mean_abs", choices=["mean_abs", "l2"])

    try:
        parser.add_argument("--channels_last", action=argparse.BooleanOptionalAction, default=True)
    except Exception:
        parser.add_argument("--channels_last", action="store_true", default=True)
        parser.add_argument("--no_channels_last", dest="channels_last", action="store_false")

    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_until", type=int, default=500)
    parser.add_argument("--eval_dense_after", type=int, default=520)

    parser.add_argument("--bn_recalib_batches", type=int, default=0)

    parser.add_argument("--w_main", type=float, default=0.2)
    parser.add_argument("--w_ema", type=float, default=0.2)
    parser.add_argument("--w_swa", type=float, default=0.6)

    parser.add_argument("--deploy_from", type=str, default="ema", choices=["ema", "swa", "main", "ensemble"])

    args = parser.parse_args()
    args.num_classes = 100

    os.makedirs(args.save, exist_ok=True)
    set_seed(args.seed)

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    model = RingBandFilterNet(
        num_classes=args.num_classes,
        drop_path_rate=args.drop_path,
        ffn_expansion=args.ffn_expansion,
        se_ratio=args.se_ratio,
        band_edge_mode=args.band_edge_mode,
        beta_init=args.beta_init,
    ).to(device)
    model.refresh_buffers(device)

    if args.channels_last and torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)

    report = param_report(model)
    save_param_report(report, os.path.join(args.save, "param_report.txt"))

    ema_model = copy.deepcopy(model).to(device)
    ema_model.refresh_buffers(device)
    if args.channels_last and torch.cuda.is_available():
        ema_model = ema_model.to(memory_format=torch.channels_last)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_model.eval()

    swa_model = copy.deepcopy(model).to(device)
    swa_model.refresh_buffers(device)
    if args.channels_last and torch.cuda.is_available():
        swa_model = swa_model.to(memory_format=torch.channels_last)
    for p in swa_model.parameters():
        p.requires_grad_(False)
    swa_model.eval()

    ens_model = copy.deepcopy(model).to(device)
    ens_model.refresh_buffers(device)
    if args.channels_last and torch.cuda.is_available():
        ens_model = ens_model.to(memory_format=torch.channels_last)
    for p in ens_model.parameters():
        p.requires_grad_(False)
    ens_model.eval()

    train_tf, test_tf, eraser = build_transforms()

    train_set = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True, transform=test_tf)

    mix_collate = MixupCutmixCollate(
        num_classes=args.num_classes,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        p_mix=args.mix_p,
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.workers > 0),
        generator=g,
        collate_fn=mix_collate,
    )
    if args.workers > 0:
        train_kwargs["prefetch_factor"] = max(2, int(args.prefetch_factor))

    test_kwargs = dict(
        batch_size=512,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.workers > 0),
    )
    if args.workers > 0:
        test_kwargs["prefetch_factor"] = max(2, int(args.prefetch_factor))

    train_loader = DataLoader(train_set, **train_kwargs)
    test_loader = DataLoader(test_set, **test_kwargs)

    param_groups = get_param_groups(model, args.weight_decay)
    opt = optim.AdamW(param_groups, lr=args.lr)

    base_sched = CosineAnnealingLR(opt, T_max=max(1, args.epochs - args.warmup_epochs))
    scheduler = WarmupWrapper(opt, base_sched, args.warmup_epochs, args.lr)

    if args.amp and args.amp_dtype == "fp16" and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = DummyScaler()

    best = {"main": 0.0, "ema": 0.0, "swa": 0.0, "ensemble": 0.0}
    last_swa_bn_n = -1

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            epoch=epoch,
            model=model,
            ema_model=ema_model,
            swa_model=swa_model,
            train_loader=train_loader,
            opt=opt,
            scaler=scaler,
            scheduler=scheduler,
            args=args,
            mix_collate=mix_collate,
            eraser=eraser,
        )

        do_eval = (epoch <= args.eval_until and epoch % args.eval_interval == 0) or (epoch > args.eval_dense_after)

        if do_eval:
            eval_models: Dict[str, nn.Module] = {"main": model, "ema": ema_model}

            if epoch >= args.swa_start_epoch:
                swa_n = getattr(train_one_epoch, "swa_n", 0)
                if swa_n != last_swa_bn_n:
                    recalibrate_bn_stats_for_avg(
                        swa_model,
                        train_loader,
                        args,
                        mix_collate,
                        eraser,
                        max_batches=args.bn_recalib_batches,
                    )
                    last_swa_bn_n = swa_n
                eval_models["swa"] = swa_model

            if epoch > args.eval_dense_after:
                srcs = {"main": model, "ema": ema_model}
                if "swa" in eval_models:
                    srcs["swa"] = swa_model
                weights = {"main": args.w_main, "ema": args.w_ema, "swa": args.w_swa}
                ok = load_weight_averaged_state(ens_model, srcs, weights)
                if ok:
                    if hasattr(ens_model, "refresh_buffers"):
                        ens_model.refresh_buffers(args.device)
                    recalibrate_bn_stats_for_avg(
                        ens_model,
                        train_loader,
                        args,
                        mix_collate,
                        eraser,
                        max_batches=args.bn_recalib_batches,
                    )
                    eval_models["ensemble"] = ens_model

            acc = evaluate(eval_models, test_loader, args)

            # NEW: report beta stats + band importance at each evaluation
            reports = _collect_beta_and_band_reports(eval_models, band_mode=args.band_importance_mode)

            print(json.dumps(
                {
                    "epoch": epoch,
                    **acc,
                    "beta_and_band_reports": reports,
                },
                ensure_ascii=False
            ))

            for k in ["main", "ema", "swa", "ensemble"]:
                if k in acc and acc[k] > best.get(k, 0.0):
                    best[k] = acc[k]
                    payload = {
                        "model": model.state_dict(),
                        "ema": ema_model.state_dict(),
                        "swa": swa_model.state_dict(),
                        "epoch": epoch,
                        "acc": acc,
                        "args": vars(args),
                        "swa_n": getattr(train_one_epoch, "swa_n", 0),
                        "last_swa_bn_n": last_swa_bn_n,
                    }
                    torch.save(payload, os.path.join(args.save, f"best_{k}.pt"))

            payload = {
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "swa": swa_model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch,
                "args": vars(args),
                "best": best,
                "swa_n": getattr(train_one_epoch, "swa_n", 0),
                "last_swa_bn_n": last_swa_bn_n,
            }
            torch.save(payload, os.path.join(args.save, "last.pt"))

    ckpt_path = os.path.join(args.save, f"best_{args.deploy_from}.pt")
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(args.save, "last.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    deploy_model = RingBandFilterNet(
        num_classes=args.num_classes,
        drop_path_rate=args.drop_path,
        ffn_expansion=args.ffn_expansion,
        se_ratio=args.se_ratio,
        band_edge_mode=args.band_edge_mode,
        beta_init=args.beta_init,
    )

    if args.deploy_from == "main":
        deploy_model.load_state_dict(ckpt["model"], strict=True)
    elif args.deploy_from == "ema":
        deploy_model.load_state_dict(ckpt["ema"], strict=True)
    elif args.deploy_from == "swa":
        deploy_model.load_state_dict(ckpt["swa"], strict=True)
    elif args.deploy_from == "ensemble":
        tmp = copy.deepcopy(deploy_model)
        tmp.load_state_dict(ckpt["model"], strict=True)
        tmp_ema = copy.deepcopy(deploy_model)
        tmp_ema.load_state_dict(ckpt["ema"], strict=True)
        tmp_swa = copy.deepcopy(deploy_model)
        tmp_swa.load_state_dict(ckpt["swa"], strict=True)
        srcs = {"main": tmp, "ema": tmp_ema, "swa": tmp_swa}
        weights = {"main": args.w_main, "ema": args.w_ema, "swa": args.w_swa}
        _ = load_weight_averaged_state(deploy_model, srcs, weights)
    else:
        deploy_model.load_state_dict(ckpt["ema"], strict=True)

    deploy_model.to(device)
    deploy_model.refresh_buffers(device)
    if args.channels_last and torch.cuda.is_available():
        deploy_model = deploy_model.to(memory_format=torch.channels_last)

    if args.deploy_from in ["swa", "ensemble"]:
        recalibrate_bn_stats_for_avg(
            deploy_model,
            train_loader,
            args,
            mix_collate,
            eraser,
            max_batches=args.bn_recalib_batches,
        )

    deploy_model.eval()
    deploy_model.switch_to_deploy()
    torch.save({"model": deploy_model.state_dict(), "args": vars(args)}, os.path.join(args.save, "deploy_model.pt"))
    print(json.dumps(best, ensure_ascii=False))


if __name__ == "__main__":
    main()
