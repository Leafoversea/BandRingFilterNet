import argparse
import contextlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T


MEAN = [0.7960541844367981, 0.659596860408783, 0.6963490843772888]
STD = [0.22328314185142517, 0.25463026762008667, 0.09473568946123123]
NUM_CLASSES = 8


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_autocast_ctx(amp: bool, amp_dtype: str):
    if (not amp) or (not torch.cuda.is_available()):
        return contextlib.nullcontext()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    dtype = dtype_map.get(str(amp_dtype).lower(), torch.bfloat16)
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def tail_anneal_factor(epoch: int, total_epochs: int, tail_epochs: int) -> float:
    if int(tail_epochs) <= 0:
        return 1.0
    start = int(total_epochs) - int(tail_epochs)
    if epoch <= start:
        return 1.0
    t = (epoch - start) / max(1, int(tail_epochs))
    t = max(0.0, min(1.0, float(t)))
    return 0.5 * (1.0 + math.cos(math.pi * t))


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor):
    logp = F.log_softmax(logits, dim=-1)
    return -(soft_targets * logp).sum(dim=-1).mean()


class RGB888TrainTransform(object):
    def __init__(self):
        self.tf = T.Compose([
            T.Resize((64, 64)),
            T.RandomResizedCrop(size=64, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=90),
            T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.03),
            T.RandAugment(num_ops=2, magnitude=10),
            T.PILToTensor(),
        ])

    def __call__(self, img):
        x = self.tf(img)
        return x.to(torch.float32)


class RGB888EvalTransform(object):
    def __init__(self):
        self.tf = T.Compose([
            T.Resize((64, 64)),
            T.PILToTensor(),
        ])

    def __call__(self, img):
        x = self.tf(img)
        return x.to(torch.float32)


class MixupCutmixCollateRGB888(object):
    def __init__(self, num_classes: int, mixup_alpha: float = 1.0, cutmix_alpha: float = 1.0, p_mix: float = 0.7):
        self.num_classes = int(num_classes)
        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.p_mix = float(p_mix)

    def set_p_mix(self, p: float):
        self.p_mix = float(max(0.0, min(1.0, p)))

    def __call__(self, batch):
        images, targets = zip(*batch)
        images = torch.stack(images, dim=0).float()
        targets = torch.tensor(targets, dtype=torch.long)

        if torch.rand(1).item() < self.p_mix:
            if torch.rand(1).item() < 0.5:
                images, targets_soft = self._mixup(images, targets)
            else:
                images, targets_soft = self._cutmix(images, targets)
        else:
            targets_soft = F.one_hot(targets, self.num_classes).float()

        images = torch.round(images).clamp(0.0, 255.0)
        return images, targets_soft

    def _mixup(self, images, targets):
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
        idx = torch.randperm(images.size(0))
        mixed = lam * images + (1.0 - lam) * images[idx]
        t1 = F.one_hot(targets, self.num_classes).float()
        t2 = F.one_hot(targets[idx], self.num_classes).float()
        tgt = lam * t1 + (1.0 - lam) * t2
        return mixed, tgt

    def _cutmix(self, images, targets):
        lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample().item()
        idx = torch.randperm(images.size(0))
        W = images.size(-1)
        H = images.size(-2)
        cut_rat = (1.0 - lam) ** 0.5
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = torch.randint(W, (1,)).item()
        cy = torch.randint(H, (1,)).item()
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(W, cx + cut_w // 2)
        y2 = min(H, cy + cut_h // 2)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
        area = (x2 - x1) * (y2 - y1)
        lam2 = 1.0 - (area / float(W * H))
        t1 = F.one_hot(targets, self.num_classes).float()
        t2 = F.one_hot(targets[idx], self.num_classes).float()
        tgt = lam2 * t1 + (1.0 - lam2) * t2
        return mixed, tgt


class HWNormRGB888(nn.Module):
    def __init__(self, mean: List[float], std: List[float]):
        super().__init__()
        m = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        s = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", m)
        self.register_buffer("std", s)
        self.inv255 = 1.0 / 255.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.inv255
        x = (x - self.mean) / self.std
        return x


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False):
    if float(drop_prob) == 0.0 or (not training):
        return x
    keep_prob = 1.0 - float(drop_prob)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rnd = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    rnd.floor_()
    return x.div(keep_prob) * rnd


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


@dataclass
class RingGFConfig:
    C: int
    K: int
    res: int


class RingGFShared(nn.Module):
    def __init__(self, cfg: RingGFConfig):
        super().__init__()
        self.C = int(cfg.C)
        self.K = int(cfg.K)
        self.res = int(cfg.res)
        self.amp = nn.Parameter(torch.ones(self.C, self.K))
        self.phase = nn.Parameter(torch.zeros(self.C, self.K))
        self.register_buffer("band_id", torch.zeros(self.res, self.res // 2 + 1, dtype=torch.long), persistent=False)
        self.register_buffer("band_keep", torch.ones(self.K), persistent=False)
        self.refresh_buffers(torch.device("cpu"))

    def refresh_buffers(self, device: torch.device):
        self.band_id = make_band_id_rfft2_like_shift(self.res, self.K, device=device)
        self.band_keep = torch.ones(self.K, device=device)

    def set_band_keep(self, keep: torch.Tensor):
        assert keep.numel() == self.K
        self.band_keep = keep.to(self.amp.device)

    def get_amp_phase(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.amp, self.phase

    def forward(self, x: torch.Tensor, s_amp: torch.Tensor, s_phase: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert h == self.res and w == self.res and c == self.C
        X = torch.fft.rfft2(x.float(), s=(h, w), norm="ortho")
        A = self.amp.float() * (1.0 + s_amp.float()) * self.band_keep.float().unsqueeze(0)
        P = self.phase.float() + s_phase.float()
        G_ck = torch.polar(A, P).to(torch.complex64)
        bid = self.band_id.to(device=x.device)
        G_chw = G_ck[:, bid]
        Y = X * G_chw.unsqueeze(0)
        y = torch.fft.irfft2(Y, s=(h, w), norm="ortho")
        return y.to(x.dtype)


class RepDepthwiseConv(nn.Module):
    def __init__(self, channels: int, k: int = 5, deploy: bool = False):
        super().__init__()
        self.channels = int(channels)
        self.k = int(k) if int(k) % 2 == 1 else int(k) + 1
        self.deploy = bool(deploy)
        if self.deploy:
            self.rbr_reparam = nn.Conv2d(self.channels, self.channels, kernel_size=self.k, padding=self.k // 2, groups=self.channels, bias=True)
            self.rbr_dw = None
            self.rbr_bn = None
            self.rbr_idbn = None
        else:
            self.rbr_dw = nn.Conv2d(self.channels, self.channels, kernel_size=self.k, padding=self.k // 2, groups=self.channels, bias=False)
            self.rbr_bn = nn.BatchNorm2d(self.channels, eps=1e-5, momentum=0.1)
            self.rbr_idbn = nn.BatchNorm2d(self.channels, eps=1e-5, momentum=0.1)
            self.rbr_reparam = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.rbr_reparam(x)
        return self.rbr_bn(self.rbr_dw(x)) + self.rbr_idbn(x)

    @torch.no_grad()
    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.deploy:
            return self.rbr_reparam.weight, self.rbr_reparam.bias
        dw = self.rbr_dw
        bn = self.rbr_bn
        W = dw.weight
        bias = torch.zeros(W.size(0), device=W.device, dtype=W.dtype)
        gamma, beta = bn.weight, bn.bias
        mean, var, eps = bn.running_mean, bn.running_var, bn.eps
        std = torch.sqrt(var + eps)
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        W_dw = W * scale
        b_dw = beta + (bias - mean) * (gamma / std)
        idbn = self.rbr_idbn
        k = self.k
        W_id = torch.zeros((self.channels, 1, k, k), device=W.device, dtype=W.dtype)
        W_id[:, 0, k // 2, k // 2] = 1.0
        gamma, beta = idbn.weight, idbn.bias
        mean, var, eps = idbn.running_mean, idbn.running_var, idbn.eps
        std = torch.sqrt(var + eps)
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        W_id = W_id * scale
        b_id = beta + (0.0 - mean) * (gamma / std)
        return W_dw + W_id, b_dw + b_id

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return
        W, b = self.get_equivalent_kernel_bias()
        conv = nn.Conv2d(self.channels, self.channels, kernel_size=self.k, padding=self.k // 2, groups=self.channels, bias=True).to(device=W.device, dtype=W.dtype)
        conv.weight.data.copy_(W)
        conv.bias.data.copy_(b.to(dtype=W.dtype, device=W.device))
        self.rbr_reparam = conv
        self.rbr_dw = None
        self.rbr_bn = None
        self.rbr_idbn = None
        self.deploy = True


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, groups: int = 1):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvBN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, groups: int = 1):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, se_ratio: float = 0.25):
        super().__init__()
        ch = int(channels)
        hidden = max(8, int(ch * float(se_ratio)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(ch, hidden, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, ch, 1, bias=True)
        self.gate = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.gate(y)
        return x * y


class MBV3FFN_ReLU_Faithful(nn.Module):
    def __init__(self, C: int, expansion: int = 3, dw_k: int = 5, se_ratio: float = 0.25):
        super().__init__()
        C = int(C)
        mid = int(C * int(expansion))
        self.expand = ConvBNReLU(C, mid, k=1, s=1)
        self.dw = RepDepthwiseConv(mid, k=dw_k, deploy=False)
        self.dw_bn = nn.BatchNorm2d(mid, eps=1e-5, momentum=0.1)
        self.dw_act = nn.ReLU(inplace=True)
        if float(se_ratio) > 0.0:
            self.se = SqueezeExcite(mid, se_ratio=float(se_ratio))
        else:
            self.se = nn.Identity()
        self.project = ConvBN(mid, C, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.dw(x)
        x = self.dw_bn(x)
        x = self.dw_act(x)
        x = self.se(x)
        x = self.project(x)
        return x

    @torch.no_grad()
    def switch_to_deploy(self):
        if hasattr(self, "dw") and hasattr(self.dw, "switch_to_deploy"):
            self.dw.switch_to_deploy()


class FFNBlockBN_MBV3(nn.Module):
    def __init__(self, C: int, drop_path_p: float, ffn_expansion: int = 3, se_ratio: float = 0.25):
        super().__init__()
        C = int(C)
        self.bn_b = nn.BatchNorm2d(C, eps=1e-5, momentum=0.1)
        self.ffn = MBV3FFN_ReLU_Faithful(C, expansion=ffn_expansion, dw_k=5, se_ratio=se_ratio)
        self.dp = DropPath(float(drop_path_p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = self.bn_b(x)
        xb = self.ffn(xb)
        xb = self.dp(xb)
        return x + xb

    @torch.no_grad()
    def switch_to_deploy(self):
        self.ffn.switch_to_deploy()


class GFBlockBN_MBV3(nn.Module):
    def __init__(self, C: int, shared: RingGFShared, drop_path_p: float, ffn_expansion: int = 3, se_ratio: float = 0.25):
        super().__init__()
        C = int(C)
        self.shared = shared
        self.bn_a = nn.BatchNorm2d(C, eps=1e-5, momentum=0.1)
        self.bn_b = nn.BatchNorm2d(C, eps=1e-5, momentum=0.1)
        self.ffn = MBV3FFN_ReLU_Faithful(C, expansion=ffn_expansion, dw_k=5, se_ratio=se_ratio)
        self.dp = DropPath(float(drop_path_p))
        self.s_amp = nn.Parameter(torch.zeros(C, shared.K))
        self.s_phase = nn.Parameter(torch.zeros(C, shared.K))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xa = self.bn_a(x)
        xa = self.shared(xa, self.s_amp, self.s_phase)
        xa = self.dp(xa)
        x = x + xa
        xb = self.bn_b(x)
        xb = self.ffn(xb)
        xb = self.dp(xb)
        x = x + xb
        return x

    @torch.no_grad()
    def switch_to_deploy(self):
        self.ffn.switch_to_deploy()


class Stage(nn.Module):
    def __init__(self, C: int, K: int, res: int, depth: int, dp_rates: List[float], ffn_expansion: int, se_ratio: float, use_ring: bool = True):
        super().__init__()
        self.use_ring = bool(use_ring)
        self.K = int(K)
        self.banddrop_p = 0.0
        if self.use_ring:
            self.shared = RingGFShared(RingGFConfig(C=int(C), K=int(K), res=int(res)))
            self.blocks = nn.Sequential(*[
                GFBlockBN_MBV3(int(C), self.shared, float(dp_rates[i]), ffn_expansion=ffn_expansion, se_ratio=se_ratio)
                for i in range(int(depth))
            ])
        else:
            self.shared = None
            self.blocks = nn.Sequential(*[
                FFNBlockBN_MBV3(int(C), float(dp_rates[i]), ffn_expansion=ffn_expansion, se_ratio=se_ratio)
                for i in range(int(depth))
            ])

    def refresh_buffers(self, device: torch.device):
        if self.use_ring and (self.shared is not None):
            self.shared.refresh_buffers(device)

    def set_banddrop_p(self, p: float):
        self.banddrop_p = float(p)

    @torch.no_grad()
    def _sample_band_keep(self, p: float, device: torch.device):
        keep = torch.ones(self.K, device=device)
        p = float(p)
        if p <= 0.0:
            return keep
        m = torch.rand(self.K, device=device) < p
        if m.any():
            keep[m] = torch.rand(int(m.sum().item()), device=device) * 0.5
        return keep

    def get_amp_phase(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.use_ring and (self.shared is not None):
            return self.shared.get_amp_phase()
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_ring and (self.shared is not None):
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


class BloodMNISTBandHWNet_MBV3(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, drop_path_rate: float = 0.2, ffn_expansion: int = 3, se_ratio: float = 0.25):
        super().__init__()
        self.hw_norm = HWNormRGB888(MEAN, STD)
        self.initial_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_bn = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.stem_act = nn.ReLU(inplace=True)
        depths = [1, 2, 1]
        Ks = [8, 6, 4]
        reses = [32, 16, 8]
        Cs = [32, 48, 64]
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, float(drop_path_rate), total_blocks)]
        i = 0
        self.stem_to_C1 = nn.Conv2d(64, Cs[0], kernel_size=1, stride=1, bias=True)
        self.stage1 = Stage(C=Cs[0], K=Ks[0], res=reses[0], depth=depths[0], dp_rates=dp_rates[i:i + depths[0]], ffn_expansion=ffn_expansion, se_ratio=se_ratio, use_ring=True)
        i += depths[0]
        self.down1 = nn.Conv2d(Cs[0], Cs[1], kernel_size=2, stride=2, padding=0, bias=True)
        self.stage2 = Stage(C=Cs[1], K=Ks[1], res=reses[1], depth=depths[1], dp_rates=dp_rates[i:i + depths[1]], ffn_expansion=ffn_expansion, se_ratio=se_ratio, use_ring=False)
        i += depths[1]
        self.down2 = nn.Conv2d(Cs[1], Cs[2], kernel_size=2, stride=2, padding=0, bias=True)
        self.stage3 = Stage(C=Cs[2], K=Ks[2], res=reses[2], depth=depths[2], dp_rates=dp_rates[i:i + depths[2]], ffn_expansion=ffn_expansion, se_ratio=se_ratio, use_ring=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(Cs[2], num_classes)
        self.banddrop_p = 0.0

    def refresh_buffers(self, device: torch.device):
        self.stage1.refresh_buffers(device)
        self.stage2.refresh_buffers(device)
        self.stage3.refresh_buffers(device)

    def set_banddrop_p(self, p: float):
        self.banddrop_p = float(p)
        self.stage1.set_banddrop_p(p)
        self.stage2.set_banddrop_p(0.0)
        self.stage3.set_banddrop_p(0.0)

    def get_all_amp_params(self) -> List[torch.Tensor]:
        ap = self.stage1.get_amp_phase()
        if ap is None:
            return []
        return [ap[0]]

    def get_all_amp_groups(self) -> List[List[torch.Tensor]]:
        amps = self.get_all_amp_params()
        groups = []
        for amp in amps:
            K = amp.shape[1]
            groups.append([amp[:, k] for k in range(K)])
        return groups

    def forward(self, x: torch.Tensor):
        x = self.initial_pool(x)
        x = self.hw_norm(x)
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_act(x)
        x = self.stem_to_C1(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

    @torch.no_grad()
    def switch_to_deploy(self):
        self.stage1.switch_to_deploy()
        self.stage2.switch_to_deploy()
        self.stage3.switch_to_deploy()


def freq_smooth_loss(amps: List[torch.Tensor], device: Optional[torch.device] = None):
    if len(amps) == 0:
        return torch.tensor(0.0, device=device if device is not None else "cpu")
    loss = None
    for amp in amps:
        if amp.shape[1] > 1:
            t = F.mse_loss(amp[:, :-1], amp[:, 1:], reduction="mean")
            loss = t if loss is None else (loss + t)
    if loss is None:
        return torch.tensor(0.0, device=device if device is not None else amps[0].device)
    return loss


def group_lasso_loss(groups: List[List[torch.Tensor]], device: Optional[torch.device] = None):
    if len(groups) == 0:
        return torch.tensor(0.0, device=device if device is not None else "cpu")
    loss = None
    cnt = 0
    for gl in groups:
        for p in gl:
            t = p.norm(2)
            loss = t if loss is None else (loss + t)
            cnt += 1
    if loss is None:
        return torch.tensor(0.0, device=device if device is not None else "cpu")
    return loss / max(1, cnt)


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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool, amp_dtype: str, calibrator: Optional[nn.Module] = None):
    model.eval()
    if calibrator is not None:
        calibrator.eval()
    if hasattr(model, "set_banddrop_p"):
        model.set_banddrop_p(0.0)
    autocast_ctx = get_autocast_ctx(amp, amp_dtype)
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_ctx:
            logits = model(images)
        if calibrator is not None:
            logits = calibrator(logits)
        pred = logits.argmax(dim=-1)
        correct += (pred == targets).sum().item()
        total += targets.numel()
    return 100.0 * correct / max(1, total), correct, total


def train_one_epoch(epoch: int, model: nn.Module, train_loader: DataLoader, opt: optim.Optimizer, scaler, scheduler, args, mix_collate: MixupCutmixCollateRGB888):
    model.train()
    tail = tail_anneal_factor(epoch, args.epochs, args.tail_anneal_epochs)
    mix_collate.set_p_mix(args.mix_p * tail)
    if hasattr(model, "set_banddrop_p"):
        model.set_banddrop_p(args.band_p * tail)
    set_droppath_factor(model, tail)
    lam_smooth = float(args.lambda_smooth) * tail
    lam_group = float(args.lambda_group) * tail
    autocast_ctx = get_autocast_ctx(args.amp, args.amp_dtype)
    for images, targets_soft in train_loader:
        images = images.to(args.device, non_blocking=True)
        targets_soft = targets_soft.to(args.device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast_ctx:
            logits = model(images)
            ce = soft_cross_entropy(logits, targets_soft)
            amps = model.get_all_amp_params()
            l_smooth = freq_smooth_loss(amps, device=args.device) if lam_smooth > 0 else torch.tensor(0.0, device=args.device)
            l_group = group_lasso_loss(model.get_all_amp_groups(), device=args.device) if lam_group > 0 else torch.tensor(0.0, device=args.device)
            loss = ce + lam_smooth * l_smooth + lam_group * l_group
        scaler.scale(loss).backward()
        if hasattr(scaler, "unscale_"):
            scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        scaler.step(opt)
        scaler.update()
    if scheduler is not None:
        scheduler.step()


class LogitLinearCalibrator(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(num_classes, num_classes, bias=True)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(num_classes))
            self.fc.bias.zero_()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        w = self.fc.weight.to(dtype=logits.dtype, device=logits.device)
        b = self.fc.bias.to(dtype=logits.dtype, device=logits.device)
        return F.linear(logits, w, b)


def calibrate_logit_linear(model: nn.Module, calib_loader: DataLoader, device: torch.device, amp: bool, amp_dtype: str, num_classes: int, num_epochs: int = 10, lr: float = 1e-2, weight_decay: float = 1e-4):
    model.eval()
    cal = LogitLinearCalibrator(num_classes).to(device)
    opt = optim.AdamW(cal.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    autocast_ctx = get_autocast_ctx(amp, amp_dtype)
    for _ in range(int(num_epochs)):
        for images, targets in calib_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.no_grad():
                with autocast_ctx:
                    logits = model(images)
            logits = logits.float()
            logits2 = cal(logits)
            loss = F.cross_entropy(logits2, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    cal.eval()
    return cal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=r"C:\Users\kangk\Desktop\BloodMNIST")
    parser.add_argument("--save", type=str, default="./run_main_only_no_gate_cs_1_2")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=40)
    parser.add_argument("--drop_path", type=float, default=0.2)
    parser.add_argument("--band_p", type=float, default=0.15)
    parser.add_argument("--lambda_smooth", type=float, default=0.15)
    parser.add_argument("--lambda_group", type=float, default=1e-5)
    parser.add_argument("--mix_p", type=float, default=0.7)
    parser.add_argument("--mixup_alpha", type=float, default=1.0)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--tail_anneal_epochs", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    try:
        parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    except Exception:
        parser.add_argument("--amp", action="store_true", default=True)
        parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ffn_expansion", type=int, default=3)
    parser.add_argument("--se_ratio", type=float, default=0.0)
    parser.add_argument("--eval_start", type=int, default=395)
    parser.add_argument("--bn_recalib_batches", type=int, default=0)
    parser.add_argument("--calib_max_per_class", type=int, default=5000)
    parser.add_argument("--calib_epochs", type=int, default=12)
    parser.add_argument("--calib_lr", type=float, default=1e-2)
    parser.add_argument("--calib_wd", type=float, default=1e-4)

    args = parser.parse_args()

    os.makedirs(args.save, exist_ok=True)
    set_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    train_tf = RGB888TrainTransform()
    eval_tf = RGB888EvalTransform()

    train_set = ImageFolder(os.path.join(args.data, "train"), transform=train_tf)
    train_set_eval = ImageFolder(os.path.join(args.data, "train"), transform=eval_tf)
    val_set = ImageFolder(os.path.join(args.data, "val"), transform=eval_tf)
    test_set = ImageFolder(os.path.join(args.data, "test"), transform=eval_tf)

    mix_collate = MixupCutmixCollateRGB888(
        num_classes=NUM_CLASSES,
        mixup_alpha=float(args.mixup_alpha),
        cutmix_alpha=float(args.cutmix_alpha),
        p_mix=float(args.mix_p),
    )

    nw = int(args.workers)
    train_loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(nw > 0),
        collate_fn=mix_collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(nw > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(nw > 0),
    )

    model = BloodMNISTBandHWNet_MBV3(
        num_classes=NUM_CLASSES,
        drop_path_rate=float(args.drop_path),
        ffn_expansion=int(args.ffn_expansion),
        se_ratio=float(args.se_ratio),
    ).to(device)
    model.refresh_buffers(device)

    opt = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    base_sched = CosineAnnealingLR(opt, T_max=max(1, int(args.epochs) - int(args.warmup_epochs)))

    class WarmupWrapper:
        def __init__(self, optimizer, base_sched, warmup_epochs, base_lr):
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

    scheduler = WarmupWrapper(opt, base_sched, int(args.warmup_epochs), float(args.lr))

    if bool(args.amp) and str(args.amp_dtype).lower() == "fp16" and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = DummyScaler()

    best_val = -1.0
    best_epoch = 0
    best_path = os.path.join(args.save, "best_main.pt")

    for epoch in range(1, int(args.epochs) + 1):
        train_one_epoch(
            epoch=epoch,
            model=model,
            train_loader=train_loader,
            opt=opt,
            scaler=scaler,
            scheduler=scheduler,
            args=args,
            mix_collate=mix_collate,
        )

        do_eval = epoch >= int(args.eval_start)
        if do_eval:
            val_acc, _, _ = evaluate(model, val_loader, device, bool(args.amp), str(args.amp_dtype), calibrator=None)

            updated = False
            if float(val_acc) > float(best_val) + 1e-12:
                best_val = float(val_acc)
                best_epoch = int(epoch)
                torch.save(
                    {
                        "epoch": int(epoch),
                        "val_acc_raw": float(val_acc),
                        "model_state": model.state_dict(),
                        "args": vars(args),
                    },
                    best_path,
                )
                updated = True
            print(json.dumps({"epoch": int(epoch), "val_raw": float(val_acc), "best": {"val_raw": float(best_val), "epoch": int(best_epoch), "updated": bool(updated)}}, ensure_ascii=False))
        torch.save(
            {
                "epoch": int(epoch),
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "best_val": float(best_val),
                "best_epoch": int(best_epoch),
                "args": vars(args),
            },
            os.path.join(args.save, "last.pt"),
        )

    ckpt = torch.load(best_path, map_location="cpu")
    deploy_model = BloodMNISTBandHWNet_MBV3(
        num_classes=NUM_CLASSES,
        drop_path_rate=float(args.drop_path),
        ffn_expansion=int(args.ffn_expansion),
        se_ratio=float(args.se_ratio),
    ).to(device)
    deploy_model.load_state_dict(ckpt["model_state"], strict=True)
    deploy_model.refresh_buffers(device)
    deploy_model.eval()
    deploy_model.switch_to_deploy()
    torch.save({"model": deploy_model.state_dict(), "args": vars(args), "best_epoch": int(ckpt["epoch"]), "best_val_raw": float(ckpt["val_acc_raw"])}, os.path.join(args.save, "deploy_model.pt"))

    if hasattr(train_set_eval, "targets"):
        targets_all = list(train_set_eval.targets)
    else:
        targets_all = [int(y) for _p, y in train_set_eval.samples]

    indices_per_class = {c: [] for c in range(NUM_CLASSES)}
    for idx, y in enumerate(targets_all):
        indices_per_class[int(y)].append(idx)

    calib_indices = []
    max_per_class = int(args.calib_max_per_class)
    for c in range(NUM_CLASSES):
        idxs = indices_per_class[c]
        random.shuffle(idxs)
        take = min(max_per_class, len(idxs))
        calib_indices.extend(idxs[:take])

    calib_subset = Subset(train_set_eval, calib_indices)
    calib_loader = DataLoader(
        calib_subset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=True,
        persistent_workers=(int(args.workers) > 0),
        drop_last=True,
    )

    cal = calibrate_logit_linear(
        deploy_model,
        calib_loader,
        device=device,
        amp=bool(args.amp),
        amp_dtype=str(args.amp_dtype),
        num_classes=NUM_CLASSES,
        num_epochs=int(args.calib_epochs),
        lr=float(args.calib_lr),
        weight_decay=float(args.calib_wd),
    )

    test_raw, _, _ = evaluate(deploy_model, test_loader, device, bool(args.amp), str(args.amp_dtype), calibrator=None)
    test_cal, _, _ = evaluate(deploy_model, test_loader, device, bool(args.amp), str(args.amp_dtype), calibrator=cal)
    val_raw, _, _ = evaluate(deploy_model, val_loader, device, bool(args.amp), str(args.amp_dtype), calibrator=None)
    val_cal, _, _ = evaluate(deploy_model, val_loader, device, bool(args.amp), str(args.amp_dtype), calibrator=cal)

    torch.save({"calibrator_state": cal.state_dict(), "args": vars(args), "best_epoch": int(ckpt["epoch"]), "best_val_raw": float(ckpt["val_acc_raw"])}, os.path.join(args.save, "calibrator.pt"))

    out = {
        "best_epoch": int(ckpt["epoch"]),
        "val_acc_raw": float(val_raw),
        "val_acc_cal": float(val_cal),
        "test_acc_raw": float(test_raw),
        "test_acc_cal": float(test_cal),
        "save_dir": str(args.save),
    }
    with open(os.path.join(args.save, "final_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
