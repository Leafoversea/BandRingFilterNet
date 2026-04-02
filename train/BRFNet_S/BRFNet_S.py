
import json
import math
import random
import time
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from sympy import false
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from timm.data import create_transform, Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except Exception:
    HAS_FVCORE = False

try:
    import torch._dynamo as _dynamo
    dynamo_disable = _dynamo.disable
except Exception:
    def dynamo_disable(fn=None, recursive=True):
        if fn is None:
            def deco(f):
                return f
            return deco
        return fn


DATA_PATH = r"D:\imagenet"
OUTPUT_DIR = r"F:\save_fixsmall_chlast-plus_ema_distill_regnety160_48_96_192_384"

RESUME_PATH = r""

RESUME_LOAD_EMA = False
RESUME_LOAD_OPTIMIZER = True

DEVICE = "cuda"
SEED = 0

EPOCHS = 300
INPUT_SIZE = 224

MICRO_BATCH = 256
ACCUM_STEPS = 8
EVAL_BATCH = 256

NUM_WORKERS = 24
PIN_MEM = True

TF32 = True
AMP = True
AMP_DTYPE = "bf16"

COMPILE_ENABLED = True
COMPILE_BACKEND = "inductor"
COMPILE_MODE = "default"
COMPILE_FULLGRAPH = False
COMPILE_DYNAMIC = False

LR_BASE_FOR_512 = 1e-3
MIN_LR = 1e-5
WARMUP_LR = 1e-6
WARMUP_EPOCHS = 5

WEIGHT_DECAY = 0.025
OPT_EPS = 1e-8
OPT_BETAS = (0.9, 0.999)

DROP_PATH = 0.1

COLOR_JITTER = 0.4
AA = "rand-m9-mstd0.5-inc1"
TRAIN_INTERPOLATION = "bicubic"
REPROB = 0.25
REMODE = "pixel"
RECOUNT = 1

MIXUP_ALPHA = 0.8
CUTMIX_ALPHA = 1.0
CUTMIX_MINMAX = None
MIXUP_PROB = 1.0
MIXUP_SWITCH_PROB = 0.5
MIXUP_MODE = "batch"

SMOOTHING = 0.1

PRINT_FREQ_OPT = 200

SAVE_FREQ = 20
SAVE_LAST_FREQ = 20

DEPTHS = (3, 3, 12, 4)
KS = (8, 6, 4, 2)
CS = (48, 96, 192, 384)
RESES = (56, 28, 14, 7)

FFN_EXPANSION = 3
SE_RATIO = 0.2
LAYERSCALE_INIT = 1e-3
BETA_INIT = 0.0

CLIP_MODE = "agc"
CLIP_GRAD = 0.02

CHANNELS_LAST = True

STAGE_MIXERS = ("L", "L", "GL", "GL")
STAGE_GATES = (False, False, True, True)
LOCAL_K = 3

USE_EMA = false
EMA_DECAY = 0.9999

USE_DISTILL = True
DISTILL_TYPE = "hard"
DISTILL_ALPHA = 0.5
DISTILL_TAU = 1.0
TEACHER_MODEL = "regnety_160"
TEACHER_CKPT = r"D:\pretrained\regnety_160-a5fe301d.pth"
TEACHER_PRETRAINED_FALLBACK = False
STUDENT_DISTILLED = True


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0).item()
        res.append(correct_k * (100.0 / target.size(0)))
    return res


def clean_state_dict_keys(state_dict):
    out = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        if nk.startswith("module."):
            nk = nk[len("module."):]
        out[nk] = v
    return out


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_ema", "teacher", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def copy_model(model: nn.Module) -> nn.Module:
    import copy
    return copy.deepcopy(model)

class ModelEma:
    """
    修复点：
    - 不再用 state_dict() 逐项 EMA，避免共享模块重复更新
    - 参数用 named_parameters() 去重
    - buffer 用 named_buffers() 去重
    - 逻辑仍然对齐你第二段代码：
        * floating point state -> EMA
        * non-floating state   -> copy
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.module = copy_model(model)
        self.module.eval()
        self.decay = float(decay)
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        # 如果外面不小心传进 compiled wrapper，这里也尽量解一层
        src_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        ema_model = self.module

        d = self.decay

        src_params = dict(src_model.named_parameters())
        ema_params = dict(ema_model.named_parameters())

        if src_params.keys() != ema_params.keys():
            missing_src = sorted(set(ema_params.keys()) - set(src_params.keys()))
            missing_ema = sorted(set(src_params.keys()) - set(ema_params.keys()))
            raise RuntimeError(
                f"EMA parameter keys mismatch. "
                f"missing_in_src={missing_src[:10]}, missing_in_ema={missing_ema[:10]}"
            )

        for k, ema_v in ema_params.items():
            model_v = src_params[k].detach()
            if ema_v.is_floating_point():
                ema_v.mul_(d).add_(model_v, alpha=1.0 - d)
            else:
                ema_v.copy_(model_v)

        src_buffers = dict(src_model.named_buffers())
        ema_buffers = dict(ema_model.named_buffers())

        if src_buffers.keys() != ema_buffers.keys():
            missing_src = sorted(set(ema_buffers.keys()) - set(src_buffers.keys()))
            missing_ema = sorted(set(src_buffers.keys()) - set(ema_buffers.keys()))
            raise RuntimeError(
                f"EMA buffer keys mismatch. "
                f"missing_in_src={missing_src[:10]}, missing_in_ema={missing_ema[:10]}"
            )

        for k, ema_v in ema_buffers.items():
            model_v = src_buffers[k].detach()
            if ema_v.is_floating_point():
                ema_v.mul_(d).add_(model_v, alpha=1.0 - d)
            else:
                ema_v.copy_(model_v)

class DistillationLoss(nn.Module):
    def __init__(
        self,
        base_criterion: nn.Module,
        teacher_model: Optional[nn.Module] = None,
        distillation_type: str = "none",
        alpha: float = 0.5,
        tau: float = 1.0,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.distillation_type = str(distillation_type).lower()
        self.alpha = float(alpha)
        self.tau = float(tau)

    def forward(self, inputs, outputs, labels):
        outputs_kd = None
        if isinstance(outputs, (tuple, list)):
            if len(outputs) != 2:
                raise ValueError("When distillation is enabled, model output must be (outputs, outputs_kd).")
            outputs, outputs_kd = outputs

        base_loss = self.base_criterion(outputs, labels)

        if self.distillation_type == "none":
            return base_loss

        if self.teacher_model is None:
            raise ValueError("Distillation is enabled but teacher_model is None.")

        if outputs_kd is None:
            raise ValueError("Distillation requires the model to return (outputs, outputs_kd) during training.")

        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
            if isinstance(teacher_outputs, (tuple, list)):
                teacher_outputs = teacher_outputs[0]

        if self.distillation_type == "soft":
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.softmax(teacher_outputs / T, dim=1),
                reduction="sum",
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == "hard":
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
        else:
            raise ValueError(f"Unknown distillation type: {self.distillation_type}")

        return base_loss * (1.0 - self.alpha) + distillation_loss * self.alpha


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(int(num_channels), eps=float(eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


def complex_from_amp_phase(amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    return torch.polar(amp.float(), phase.float()).to(torch.complex64)


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
    rr = torch.sqrt(yy.float() ** 2 + xx.float() ** 2)
    r_max = float(rr.max().item() + 1e-6)
    edges = torch.linspace(0.0, r_max, int(K) + 1, device=device)
    band_id = torch.bucketize(rr.reshape(-1), edges[1:-1], right=False).reshape(H, Wc).to(torch.long)
    return band_id


class RingGFShared(nn.Module):
    def __init__(self, C: int, K: int, res: int):
        super().__init__()
        self.C = int(C)
        self.K = int(K)
        self.res = int(res)
        self.amp = nn.Parameter(torch.ones(self.C, self.K))
        self.phase = nn.Parameter(torch.zeros(self.C, self.K))
        self.register_buffer("band_id", torch.empty(self.res, self.res // 2 + 1, dtype=torch.long), persistent=False)
        self.register_buffer("band_id_flat", torch.empty(self.res * (self.res // 2 + 1), dtype=torch.long), persistent=False)

    def refresh_buffers(self, device: torch.device):
        bid = make_band_id_rfft2_like_shift(self.res, self.K, device=device)
        self.band_id = bid
        self.band_id_flat = bid.reshape(-1)

    @dynamo_disable
    def _fft_forward_core(self, x: torch.Tensor, s_amp: torch.Tensor, s_phase: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x32 = x.float()
        X = torch.fft.rfft2(x32, s=(h, w), norm="ortho")
        A = self.amp.float() * (1.0 + s_amp.float())
        P = self.phase.float() + s_phase.float()
        G_ck = complex_from_amp_phase(A, P)

        bid_flat = self.band_id_flat
        G_flat = torch.index_select(G_ck, dim=1, index=bid_flat)
        G_chw = G_flat.view(self.C, h, w // 2 + 1)

        Y = X * G_chw.unsqueeze(0)
        y = torch.fft.irfft2(Y, s=(h, w), norm="ortho")
        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, s_amp: torch.Tensor, s_phase: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h != self.res or w != self.res:
            raise RuntimeError(f"GF res mismatch: got {h}x{w}, expected {self.res}x{self.res}")
        if c != self.C:
            raise RuntimeError(f"GF channels mismatch: got C={c}, expected {self.C}")
        if self.band_id.device != x.device:
            raise RuntimeError(f"band_id on {self.band_id.device}, but x on {x.device}")
        return self._fft_forward_core(x, s_amp, s_phase)


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
                self.in_channels, self.out_channels,
                kernel_size=self.k, stride=self.stride, padding=self.padding, bias=True
            )
            self.rbr_dense = None
            self.rbr_1x1 = None
            self.rbr_identity = None
        else:
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, self.out_channels,
                    kernel_size=self.k, stride=self.stride, padding=self.padding, bias=False
                ),
                nn.BatchNorm2d(self.out_channels),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, self.out_channels,
                    kernel_size=1, stride=self.stride, padding=0, bias=False
                ),
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


class DownsampleRD(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


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
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.Hardswish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class MBV3FFN(nn.Module):
    def __init__(self, C: int, expansion: int = 3, dw_k: int = 5, se_ratio: float = 0.2):
        super().__init__()
        mid = int(C * expansion)
        self.expand = ConvBNAct(C, mid, k=1, s=1)
        self.dw = nn.Conv2d(mid, mid, kernel_size=dw_k, padding=dw_k // 2, groups=mid, bias=False)
        self.dw_bn = nn.BatchNorm2d(mid)
        self.dw_act = nn.Hardswish()
        self.se = SqueezeExcite(mid, se_ratio=se_ratio)
        self.project = nn.Conv2d(mid, C, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.dw_act(self.dw_bn(self.dw(x)))
        x = self.se(x)
        x = self.project(x)
        return x


class LocalDW(nn.Module):
    def __init__(self, C: int, k: int = 3):
        super().__init__()
        kk = int(k) if int(k) % 2 == 1 else int(k) + 1
        self.dw = nn.Conv2d(C, C, kernel_size=kk, padding=kk // 2, groups=C, bias=False)
        self.bn = nn.BatchNorm2d(C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.dw(x))


class GateMLP(nn.Module):
    def __init__(self, C: int, hidden: Optional[int] = None):
        super().__init__()
        h = int(hidden) if hidden is not None else max(8, C // 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(C, h, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(h, C, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.pool(x)
        g = self.fc1(g)
        g = self.act(g)
        g = self.fc2(g)
        return self.gate(g)


class GFBlock(nn.Module):
    def __init__(
        self,
        C: int,
        shared: RingGFShared,
        drop_path_prob: float,
        ffn_expansion: int,
        se_ratio: float,
        layerscale_init: float = 1e-3,
        beta_init: float = 0.0,
        mixer: str = "G",
        local_k: int = 3,
        use_gate: bool = False,
    ):
        super().__init__()
        self.shared = shared
        self.mixer = str(mixer).upper()
        self.use_gate = bool(use_gate)

        self.norm_a = LayerNorm2d(C)
        self.norm_b = LayerNorm2d(C)

        self.local = LocalDW(C, k=int(local_k))
        self.gate = GateMLP(C) if (self.mixer == "GL" and self.use_gate) else None

        self.branch_ffn = MBV3FFN(C, expansion=ffn_expansion, dw_k=5, se_ratio=se_ratio)
        self.act_gf = nn.GELU()
        self.droppath = DropPath(drop_path_prob)

        self.s_amp = nn.Parameter(torch.zeros(C, shared.K))
        self.s_phase = nn.Parameter(torch.zeros(C, shared.K))
        self.gamma_a = nn.Parameter(torch.ones(C) * float(layerscale_init))
        self.gamma_b = nn.Parameter(torch.ones(C) * float(layerscale_init))
        self.beta_a = nn.Parameter(torch.ones(C) * float(beta_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ga = self.gamma_a.view(1, -1, 1, 1)
        gb = self.gamma_b.view(1, -1, 1, 1)
        ba = self.beta_a.view(1, -1, 1, 1)

        xa = self.norm_a(x)

        if self.mixer == "G":
            xa = self.shared(xa, self.s_amp, self.s_phase)
        elif self.mixer == "L":
            xa = self.local(xa)
        elif self.mixer == "GL":
            xg = self.shared(xa, self.s_amp, self.s_phase)
            xl = self.local(xa)
            if self.gate is None:
                xa = xg + xl
            else:
                w = self.gate(xa)
                xa = w * xg + (1.0 - w) * xl
        else:
            raise RuntimeError(f"Unknown mixer: {self.mixer}")

        xa = self.act_gf(xa)
        xa = ba * (ga * xa)
        xa = self.droppath(xa)
        x = x + xa

        xb = self.norm_b(x)
        xb = self.branch_ffn(xb)
        xb = self.droppath(gb * xb)
        x = x + xb
        return x


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
        layerscale_init: float = 1e-3,
        beta_init: float = 0.0,
        mixer: str = "G",
        local_k: int = 3,
        use_gate: bool = False,
    ):
        super().__init__()
        self.mixer = str(mixer).upper()
        self.use_gate = bool(use_gate)
        self.local_k = int(local_k)

        self.shared = RingGFShared(C=C, K=K, res=res)
        self.blocks = nn.Sequential(
            *[
                GFBlock(
                    C=C,
                    shared=self.shared,
                    drop_path_prob=float(dp_rates[i]),
                    ffn_expansion=ffn_expansion,
                    se_ratio=se_ratio,
                    layerscale_init=layerscale_init,
                    beta_init=beta_init,
                    mixer=self.mixer,
                    local_k=self.local_k,
                    use_gate=self.use_gate,
                )
                for i in range(int(depth))
            ]
        )

    def refresh_buffers(self, device: torch.device):
        self.shared.refresh_buffers(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class RingBandFilterNetImageNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        drop_path_rate: float = 0.1,
        ffn_expansion: int = 3,
        se_ratio: float = 0.2,
        depths: Tuple[int, int, int, int] = (3, 3, 10, 3),
        Ks: Tuple[int, int, int, int] = (8, 6, 4, 2),
        Cs: Tuple[int, int, int, int] = (48, 96, 192, 384),
        reses: Tuple[int, int, int, int] = (56, 28, 14, 7),
        layerscale_init: float = 1e-3,
        beta_init: float = 0.0,
        stage_mixers: Tuple[str, str, str, str] = ("L", "L", "GL", "GL"),
        stage_gates: Tuple[bool, bool, bool, bool] = (False, False, True, True),
        local_k: int = 3,
        distilled: bool = True,
    ):
        super().__init__()
        depths = tuple(int(x) for x in depths)
        Ks = tuple(int(x) for x in Ks)
        Cs = tuple(int(x) for x in Cs)
        reses = tuple(int(x) for x in reses)

        stage_mixers = tuple(str(x).upper() for x in stage_mixers)
        stage_gates = tuple(bool(x) for x in stage_gates)
        local_k = int(local_k)
        self.distilled = bool(distilled)

        total_blocks = sum(depths)
        dp_rates_all = [x.item() for x in torch.linspace(0, float(drop_path_rate), total_blocks)]
        idx = 0

        self.stem1 = RepVGGBlock(3, 24, k=3, stride=2, deploy=False)
        self.stem2 = RepVGGBlock(24, 48, k=3, stride=2, deploy=False)
        self.stem_to_C1 = nn.Conv2d(48, Cs[0], kernel_size=1, stride=1, bias=True)

        self.stage1 = Stage(
            C=Cs[0],
            K=Ks[0],
            res=reses[0],
            depth=depths[0],
            dp_rates=dp_rates_all[idx: idx + depths[0]],
            ffn_expansion=ffn_expansion,
            se_ratio=se_ratio,
            layerscale_init=layerscale_init,
            beta_init=beta_init,
            mixer=stage_mixers[0],
            local_k=local_k,
            use_gate=stage_gates[0],
        )
        idx += depths[0]
        self.down1 = DownsampleRD(Cs[0], Cs[1])

        self.stage2 = Stage(
            C=Cs[1],
            K=Ks[1],
            res=reses[1],
            depth=depths[1],
            dp_rates=dp_rates_all[idx: idx + depths[1]],
            ffn_expansion=ffn_expansion,
            se_ratio=se_ratio,
            layerscale_init=layerscale_init,
            beta_init=beta_init,
            mixer=stage_mixers[1],
            local_k=local_k,
            use_gate=stage_gates[1],
        )
        idx += depths[1]
        self.down2 = DownsampleRD(Cs[1], Cs[2])

        self.stage3 = Stage(
            C=Cs[2],
            K=Ks[2],
            res=reses[2],
            depth=depths[2],
            dp_rates=dp_rates_all[idx: idx + depths[2]],
            ffn_expansion=ffn_expansion,
            se_ratio=se_ratio,
            layerscale_init=layerscale_init,
            beta_init=beta_init,
            mixer=stage_mixers[2],
            local_k=local_k,
            use_gate=stage_gates[2],
        )
        idx += depths[2]
        self.down3 = DownsampleRD(Cs[2], Cs[3])

        self.stage4 = Stage(
            C=Cs[3],
            K=Ks[3],
            res=reses[3],
            depth=depths[3],
            dp_rates=dp_rates_all[idx: idx + depths[3]],
            ffn_expansion=ffn_expansion,
            se_ratio=se_ratio,
            layerscale_init=layerscale_init,
            beta_init=beta_init,
            mixer=stage_mixers[3],
            local_k=local_k,
            use_gate=stage_gates[3],
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(Cs[3], num_classes)
        self.fc_dist = nn.Linear(Cs[3], num_classes) if self.distilled else None

    def refresh_buffers(self, device: torch.device):
        self.stage1.refresh_buffers(device)
        self.stage2.refresh_buffers(device)
        self.stage3.refresh_buffers(device)
        self.stage4.refresh_buffers(device)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem1(x)
        x = self.stem2(x)
        x = self.stem_to_C1(x)
        x = x.contiguous()
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        x_cls = self.fc(x)

        if self.fc_dist is None:
            return x_cls

        x_dist = self.fc_dist(x)

        if self.training:
            return x_cls, x_dist

        return (x_cls + x_dist) * 0.5


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def report_model_stats(model: nn.Module, input_size: int = 224):
    n_params = count_params(model)
    mac_g = float("nan")

    if HAS_FVCORE:
        try:
            import copy
            model_eval = copy.deepcopy(model).eval().cpu()
            if hasattr(model_eval, "refresh_buffers"):
                model_eval.refresh_buffers(torch.device("cpu"))

            dummy = torch.randn(1, 3, input_size, input_size)
            flops = FlopCountAnalysis(model_eval, dummy)
            mac_g = float(flops.total()) / 1e9

            print("[fvcore] total MACs(G):", f"{mac_g:.4f}")
            print("[fvcore] total params(M):", f"{n_params / 1e6:.4f}")
            print("[fvcore] MACs * Params:", f"{mac_g * (n_params / 1e6):.4f}")
        except Exception as e:
            print("[fvcore] failed to analyze FLOPs/MACs:", repr(e))
    else:
        print("[fvcore] not available, skip MAC report.")

    return n_params, mac_g


def param_groups_weight_decay(model: nn.Module, weight_decay: float):
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def lr_for_epoch(epoch: int, epochs: int, base_lr: float, min_lr: float, warmup_epochs: int, warmup_lr: float):
    if warmup_epochs > 0 and epoch < warmup_epochs:
        t = float(epoch) / float(warmup_epochs)
        return float(warmup_lr + (base_lr - warmup_lr) * t)
    if epochs <= warmup_epochs + 1:
        return float(min_lr)
    t = epoch - warmup_epochs
    T = epochs - warmup_epochs - 1
    cos_t = math.cos(math.pi * float(t) / float(T))
    return float(min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cos_t))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = float(lr)


def build_loaders():
    train_dir = Path(DATA_PATH) / "train"
    val_dir = Path(DATA_PATH) / "val"
    if not train_dir.is_dir():
        raise RuntimeError(f"train dir not found: {train_dir}")
    if not val_dir.is_dir():
        raise RuntimeError(f"val dir not found: {val_dir}")

    train_tf = create_transform(
        input_size=INPUT_SIZE,
        is_training=True,
        color_jitter=COLOR_JITTER,
        auto_augment=AA,
        interpolation=TRAIN_INTERPOLATION,
        re_prob=REPROB,
        re_mode=REMODE,
        re_count=RECOUNT,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    val_tf = create_transform(
        input_size=INPUT_SIZE,
        is_training=False,
        interpolation="bicubic",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        crop_pct=0.875,
    )

    dataset_train = ImageFolder(str(train_dir), transform=train_tf)
    dataset_val = ImageFolder(str(val_dir), transform=val_tf)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    common_kwargs = {}
    if int(NUM_WORKERS) > 0:
        common_kwargs["persistent_workers"] = True
        common_kwargs["prefetch_factor"] = 4

    loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=MICRO_BATCH,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        drop_last=True,
        **common_kwargs,
    )
    loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(EVAL_BATCH),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        drop_last=False,
        **common_kwargs,
    )
    return dataset_train, dataset_val, loader_train, loader_val


def unitwise_norm(x: torch.Tensor) -> torch.Tensor:
    if x.ndim <= 1:
        return torch.linalg.vector_norm(x, ord=2)
    return torch.linalg.vector_norm(x, ord=2, dim=tuple(range(1, x.ndim)), keepdim=True)


@torch.no_grad()
def adaptive_clip_grad_(parameters, clip_factor: float, eps: float = 1e-3):
    cf = float(clip_factor)
    for p in parameters:
        if p.grad is None:
            continue
        if p.grad.is_sparse:
            continue
        if p.ndim <= 1:
            continue
        g = p.grad
        p_norm = unitwise_norm(p.detach()).clamp_min(eps)
        g_norm = unitwise_norm(g.detach()).clamp_min(eps)
        max_norm = p_norm * cf
        scale = max_norm / g_norm
        need_clip = scale < 1.0
        if torch.any(need_clip):
            g.mul_(torch.where(need_clip, scale, torch.ones_like(scale)))


@torch.no_grad()
def global_grad_norm(parameters) -> float:
    norms = []
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach()
        if g.is_sparse:
            continue
        norms.append(torch.linalg.vector_norm(g.float()).unsqueeze(0))
    if not norms:
        return 0.0
    n = torch.linalg.vector_norm(torch.cat(norms, dim=0))
    return float(n.item())


def to_channels_last(x: torch.Tensor) -> torch.Tensor:
    if not CHANNELS_LAST:
        return x
    return x.contiguous(memory_format=torch.channels_last)


def maybe_compile_model(model: nn.Module, model_name: str = "model") -> nn.Module:
    if not COMPILE_ENABLED:
        return model
    if not hasattr(torch, "compile"):
        print(f"[compile] torch.compile not found, skip {model_name}.")
        return model
    try:
        compiled = torch.compile(
            model,
            backend=COMPILE_BACKEND,
            mode=COMPILE_MODE,
            fullgraph=COMPILE_FULLGRAPH,
            dynamic=COMPILE_DYNAMIC,
        )
        print(
            f"[compile] enabled for {model_name}: "
            f"backend={COMPILE_BACKEND}, mode={COMPILE_MODE}, "
            f"fullgraph={COMPILE_FULLGRAPH}, dynamic={COMPILE_DYNAMIC}"
        )
        print("[compile] FFT core is forced eager via torch._dynamo.disable")
        return compiled
    except Exception as e:
        print(f"[compile] failed for {model_name}, fallback to eager:", repr(e))
        return model


def build_teacher_model(num_classes: int, device: torch.device) -> Optional[nn.Module]:
    if not USE_DISTILL:
        return None

    if int(num_classes) != 1000:
        raise RuntimeError(
            "This RepViT-style setup expects ImageNet-1K (1000 classes), "
            f"but got num_classes={num_classes}."
        )

    teacher = create_model(
        TEACHER_MODEL,
        pretrained=bool(TEACHER_PRETRAINED_FALLBACK and not TEACHER_CKPT),
    )

    if TEACHER_CKPT:
        teacher_path = Path(TEACHER_CKPT)
        if not teacher_path.is_file():
            raise RuntimeError(f"teacher checkpoint not found: {teacher_path}")
        ckpt = torch.load(str(teacher_path), map_location="cpu")
        sd = clean_state_dict_keys(extract_state_dict(ckpt))
        missing, unexpected = teacher.load_state_dict(sd, strict=False)
        print(f"[teacher] loaded from: {teacher_path}")
        print(f"[teacher] missing keys: {len(missing)}")
        print(f"[teacher] unexpected keys: {len(unexpected)}")
        if len(missing) > 0:
            print("[teacher] first missing keys:", missing[:10])
        if len(unexpected) > 0:
            print("[teacher] first unexpected keys:", unexpected[:10])
    else:
        if not TEACHER_PRETRAINED_FALLBACK:
            raise RuntimeError("USE_DISTILL=True but no TEACHER_CKPT provided.")
        print(f"[teacher] using timm pretrained weights for {TEACHER_MODEL}")

    teacher = teacher.to(device)
    if CHANNELS_LAST:
        teacher = teacher.to(memory_format=torch.channels_last)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def get_clean_state_dict(model: nn.Module):
    return clean_state_dict_keys(model.state_dict())


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=True)


def train_one_epoch(
    model_train: nn.Module,
    model_raw: nn.Module,
    model_ema: Optional[ModelEma],
    criterion: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    clip_mode: str,
    clip_grad: float,
    mixup_fn: Optional[Mixup],
    accum_steps: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    print_freq_opt: int,
):
    model_raw.train(True)
    model_train.train(True)
    optimizer.zero_grad(set_to_none=True)

    iters = len(data_loader)
    accum_steps = int(max(1, accum_steps))
    iters_eff = (iters // accum_steps) * accum_steps

    loss_sum = torch.zeros((), device=device)
    n_micro = 0
    n_opt = 0
    last_gn_pre = 0.0
    last_gn_post = 0.0

    for it, (samples, targets) in enumerate(data_loader):
        if it >= iters_eff:
            break

        samples = samples.to(device, non_blocking=True)
        samples = to_channels_last(samples)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            outputs = model_train(samples)
            loss = criterion(samples, outputs, targets)

        loss = loss / float(accum_steps)
        loss.backward()

        do_step = ((it + 1) % accum_steps == 0)
        if do_step:
            gn_pre = global_grad_norm(model_raw.parameters())

            if float(clip_grad) > 0:
                if str(clip_mode).lower() == "agc":
                    adaptive_clip_grad_(model_raw.parameters(), clip_factor=float(clip_grad))
                else:
                    torch.nn.utils.clip_grad_norm_(model_raw.parameters(), float(clip_grad))

            gn_post = global_grad_norm(model_raw.parameters())
            last_gn_pre = float(gn_pre)
            last_gn_post = float(gn_post)

            optimizer.step()
            if model_ema is not None:
                model_ema.update(model_raw)
            optimizer.zero_grad(set_to_none=True)
            n_opt += 1

            if print_freq_opt > 0 and (n_opt % int(print_freq_opt) == 0):
                lr_now = float(optimizer.param_groups[0]["lr"])
                loss_log = float((loss.detach() * float(accum_steps)).float().item())
                print(
                    f"[E{epoch}] opt_step={n_opt} "
                    f"loss={loss_log:.4f} lr={lr_now:.6g} "
                    f"gn_pre={last_gn_pre:.4f} gn_post={last_gn_post:.4f}"
                )

        loss_sum += (loss.detach() * float(accum_steps))
        n_micro += 1

    return {
        "loss": float((loss_sum / max(1, n_micro)).float().item()),
        "opt_steps": float(n_opt),
        "lr": float(optimizer.param_groups[0]["lr"]),
        "gn_pre": float(last_gn_pre),
        "gn_post": float(last_gn_post),
    }


@torch.inference_mode()
def evaluate_one(data_loader: DataLoader, model: nn.Module, device: torch.device, amp_enabled: bool, amp_dtype: torch.dtype):
    model.eval()
    top1_sum = 0.0
    top5_sum = 0.0
    n = 0
    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        samples = to_channels_last(samples)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(samples)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
        acc1, acc5 = accuracy_topk(outputs, targets, topk=(1, 5))
        bs = targets.size(0)
        top1_sum += acc1 * bs
        top5_sum += acc5 * bs
        n += bs
    return {"acc1": top1_sum / max(1, n), "acc5": top5_sum / max(1, n)}


def save_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def should_eval(epoch: int, total_epochs: int) -> bool:
    e = epoch + 1
    if e <= 3:
        return True
    if e <= 270 and (e % 10 == 0):
        return True
    if e > 270:
        return True
    if e == total_epochs:
        return True
    return False


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    set_seed(int(SEED))

    if TF32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    cudnn.benchmark = True

    amp_enabled = bool(AMP) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(AMP_DTYPE).lower() == "bf16" else torch.float16
    if amp_enabled and amp_dtype == torch.bfloat16 and (not torch.cuda.is_bf16_supported()):
        amp_dtype = torch.float16

    dataset_train, dataset_val, loader_train, loader_val = build_loaders()
    nb_classes = len(dataset_train.classes)

    model_raw = RingBandFilterNetImageNet(
        num_classes=nb_classes,
        drop_path_rate=float(DROP_PATH),
        ffn_expansion=int(FFN_EXPANSION),
        se_ratio=float(SE_RATIO),
        depths=DEPTHS,
        Ks=KS,
        Cs=CS,
        reses=RESES,
        layerscale_init=float(LAYERSCALE_INIT),
        beta_init=float(BETA_INIT),
        stage_mixers=STAGE_MIXERS,
        stage_gates=STAGE_GATES,
        local_k=int(LOCAL_K),
        distilled=bool(STUDENT_DISTILLED and USE_DISTILL),
    )

    n_parameters, mac_g = report_model_stats(model_raw, input_size=INPUT_SIZE)

    model_raw = model_raw.to(device)
    model_raw.refresh_buffers(device)

    if CHANNELS_LAST:
        model_raw = model_raw.to(memory_format=torch.channels_last)

    teacher_model = build_teacher_model(nb_classes, device) if USE_DISTILL else None

    model_ema = ModelEma(model_raw, decay=EMA_DECAY) if USE_EMA else None

    global_batch = int(MICRO_BATCH) * int(ACCUM_STEPS)
    base_lr = float(LR_BASE_FOR_512) * float(global_batch) / 512.0

    param_groups = param_groups_weight_decay(model_raw, float(WEIGHT_DECAY))
    try:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=float(base_lr),
            betas=OPT_BETAS,
            eps=float(OPT_EPS),
            fused=True
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=float(base_lr),
            betas=OPT_BETAS,
            eps=float(OPT_EPS)
        )

    start_epoch = 0
    max_acc_main = 0.0
    max_acc_ema = 0.0

    if RESUME_PATH:
        resume_path = Path(RESUME_PATH)
        if not resume_path.is_file():
            raise RuntimeError(f"resume checkpoint not found: {resume_path}")

        ckpt = torch.load(str(resume_path), map_location="cpu")
        model_raw.load_state_dict(clean_state_dict_keys(ckpt["model"]), strict=True)

        if RESUME_LOAD_OPTIMIZER and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            optimizer_to(optimizer, device)

        if model_ema is not None:
            if RESUME_LOAD_EMA and ("model_ema" in ckpt):
                model_ema.module.load_state_dict(clean_state_dict_keys(ckpt["model_ema"]), strict=True)
                max_acc_ema = float(ckpt.get("max_accuracy_ema", 0.0))
                print("[resume] loaded EMA from checkpoint.")
            else:
                model_ema = ModelEma(model_raw, decay=EMA_DECAY)
                max_acc_ema = 0.0
                print("[resume] rebuilt EMA from current main model weights.")

        start_epoch = int(ckpt.get("epoch", -1)) + 1
        max_acc_main = float(ckpt.get("max_accuracy_main", 0.0))

        print(f"[resume] loaded from: {resume_path}")
        print(f"[resume] start_epoch={start_epoch}")
        print(f"[resume] max_acc_main={max_acc_main:.2f} max_acc_ema={max_acc_ema:.2f}")
    else:
        print("[resume] RESUME_PATH is empty, start training from scratch.")

    model_train = maybe_compile_model(model_raw, model_name="student")

    print("number of params:", n_parameters)

    mixup_fn = None
    mixup_active = MIXUP_ALPHA > 0.0 or CUTMIX_ALPHA > 0.0 or CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=float(MIXUP_ALPHA),
            cutmix_alpha=float(CUTMIX_ALPHA),
            cutmix_minmax=CUTMIX_MINMAX,
            prob=float(MIXUP_PROB),
            switch_prob=float(MIXUP_SWITCH_PROB),
            mode=str(MIXUP_MODE),
            label_smoothing=float(SMOOTHING),
            num_classes=int(nb_classes),
        )

    if mixup_active:
        base_criterion = SoftTargetCrossEntropy()
    elif float(SMOOTHING) > 0.0:
        base_criterion = LabelSmoothingCrossEntropy(smoothing=float(SMOOTHING))
    else:
        base_criterion = nn.CrossEntropyLoss()

    criterion = DistillationLoss(
        base_criterion=base_criterion,
        teacher_model=teacher_model,
        distillation_type=str(DISTILL_TYPE if USE_DISTILL else "none"),
        alpha=float(DISTILL_ALPHA),
        tau=float(DISTILL_TAU),
    )

    meta = {
        "data_path": str(DATA_PATH),
        "output_dir": str(OUTPUT_DIR),
        "resume_path": str(RESUME_PATH),
        "resume_load_ema": bool(RESUME_LOAD_EMA),
        "resume_load_optimizer": bool(RESUME_LOAD_OPTIMIZER),
        "epochs": int(EPOCHS),
        "batch_size": int(MICRO_BATCH),
        "accum_steps": int(ACCUM_STEPS),
        "global_batch": int(global_batch),
        "lr_base_for_512": float(LR_BASE_FOR_512),
        "lr_scaled": float(base_lr),
        "min_lr": float(MIN_LR),
        "warmup_lr": float(WARMUP_LR),
        "warmup_epochs": int(WARMUP_EPOCHS),
        "weight_decay": float(WEIGHT_DECAY),
        "clip_mode": str(CLIP_MODE),
        "clip_grad": float(CLIP_GRAD),
        "drop_path": float(DROP_PATH),
        "params": int(n_parameters),
        "params_m": float(n_parameters / 1e6),
        "mac_g": None if not math.isfinite(mac_g) else float(mac_g),
        "nb_classes": int(nb_classes),
        "tf32": bool(TF32),
        "amp": bool(amp_enabled),
        "amp_dtype": "bf16" if amp_dtype == torch.bfloat16 else "fp16",
        "channels_last": bool(CHANNELS_LAST),
        "compile_enabled": bool(COMPILE_ENABLED),
        "compile_backend": str(COMPILE_BACKEND),
        "compile_mode": str(COMPILE_MODE),
        "compile_fullgraph": bool(COMPILE_FULLGRAPH),
        "compile_dynamic": bool(COMPILE_DYNAMIC),
        "depths": list(DEPTHS),
        "channels": list(CS),
        "Ks": list(KS),
        "reses": list(RESES),
        "ffn_expansion": int(FFN_EXPANSION),
        "se_ratio": float(SE_RATIO),
        "layerscale_init": float(LAYERSCALE_INIT),
        "beta_init": float(BETA_INIT),
        "stage_mixers": list(STAGE_MIXERS),
        "stage_gates": list(STAGE_GATES),
        "local_k": int(LOCAL_K),
        "use_ema": bool(USE_EMA),
        "ema_decay": float(EMA_DECAY),
        "use_distill": bool(USE_DISTILL),
        "distill_type": str(DISTILL_TYPE),
        "distill_alpha": float(DISTILL_ALPHA),
        "distill_tau": float(DISTILL_TAU),
        "teacher_model": str(TEACHER_MODEL),
        "teacher_ckpt": str(TEACHER_CKPT),
        "student_distilled": bool(STUDENT_DISTILLED and USE_DISTILL),
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"Start training for {int(EPOCHS)} epochs "
        f"(micro={MICRO_BATCH}, accum={ACCUM_STEPS}, world=1, global={global_batch}) "
        f"amp={amp_enabled}({'bf16' if amp_dtype == torch.bfloat16 else 'fp16'})"
    )
    print(
        f"[distill] enabled={USE_DISTILL} "
        f"type={DISTILL_TYPE if USE_DISTILL else 'none'} "
        f"alpha={DISTILL_ALPHA} tau={DISTILL_TAU} "
        f"teacher={TEACHER_MODEL if USE_DISTILL else 'none'} "
        f"teacher_ckpt={TEACHER_CKPT if USE_DISTILL else 'none'} "
        f"student_distilled={bool(STUDENT_DISTILLED and USE_DISTILL)}"
    )
    print(f"[ema] enabled={USE_EMA} decay={EMA_DECAY}")
    print(
        f"[compile] enabled={COMPILE_ENABLED} "
        f"backend={COMPILE_BACKEND} mode={COMPILE_MODE} "
        f"fullgraph={COMPILE_FULLGRAPH} dynamic={COMPILE_DYNAMIC}"
    )

    start_time = time.time()

    for epoch in range(start_epoch, int(EPOCHS)):
        lr_epoch = lr_for_epoch(
            epoch=epoch,
            epochs=int(EPOCHS),
            base_lr=float(base_lr),
            min_lr=float(MIN_LR),
            warmup_epochs=int(WARMUP_EPOCHS),
            warmup_lr=float(WARMUP_LR),
        )
        set_optimizer_lr(optimizer, lr_epoch)
        print(f"[E{epoch}] lr_begin={float(optimizer.param_groups[0]['lr']):.6g}")

        train_stats = train_one_epoch(
            model_train=model_train,
            model_raw=model_raw,
            model_ema=model_ema,
            criterion=criterion,
            data_loader=loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            clip_mode=str(CLIP_MODE),
            clip_grad=float(CLIP_GRAD),
            mixup_fn=mixup_fn,
            accum_steps=int(ACCUM_STEPS),
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            print_freq_opt=int(PRINT_FREQ_OPT),
        )
        print(f"[E{epoch}] lr_end={float(optimizer.param_groups[0]['lr']):.6g} train_loss={float(train_stats['loss']):.4f}")

        do_eval = should_eval(epoch, int(EPOCHS))
        test_stats = {
            "main_acc1": float("nan"),
            "main_acc5": float("nan"),
            "ema_acc1": float("nan"),
            "ema_acc5": float("nan"),
        }

        if do_eval:
            main_stats = evaluate_one(loader_val, model_raw, device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
            test_stats["main_acc1"] = float(main_stats["acc1"])
            test_stats["main_acc5"] = float(main_stats["acc5"])

            if model_ema is not None:
                ema_stats = evaluate_one(loader_val, model_ema.module, device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
                test_stats["ema_acc1"] = float(ema_stats["acc1"])
                test_stats["ema_acc5"] = float(ema_stats["acc5"])

            print(
                f"Accuracy on {len(loader_val.dataset)} val images: "
                f"main acc1={test_stats['main_acc1']:.2f}% acc5={test_stats['main_acc5']:.2f}% "
                f"ema acc1={test_stats['ema_acc1']:.2f}% acc5={test_stats['ema_acc5']:.2f}%"
            )

            if math.isfinite(test_stats["main_acc1"]):
                max_acc_main = max(max_acc_main, test_stats["main_acc1"])
            if math.isfinite(test_stats["ema_acc1"]):
                max_acc_ema = max(max_acc_ema, test_stats["ema_acc1"])

            print(f"Max accuracy: main={max_acc_main:.2f}% ema={max_acc_ema:.2f}%")

        payload = {
            "model": get_clean_state_dict(model_raw),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "args": meta,
            "max_accuracy_main": float(max_acc_main),
            "max_accuracy_ema": float(max_acc_ema),
            "use_ema": bool(USE_EMA),
            "ema_decay": float(EMA_DECAY),
            "use_distill": bool(USE_DISTILL),
            "distill_type": str(DISTILL_TYPE),
            "distill_alpha": float(DISTILL_ALPHA),
            "distill_tau": float(DISTILL_TAU),
            "teacher_model": str(TEACHER_MODEL),
            "teacher_ckpt": str(TEACHER_CKPT),
        }
        if model_ema is not None:
            payload["model_ema"] = get_clean_state_dict(model_ema.module)

        if ((epoch + 1) % int(SAVE_LAST_FREQ) == 0) or (epoch == int(EPOCHS) - 1):
            save_checkpoint(output_dir / "checkpoint_last.pth", payload)
        if (epoch + 1) % int(SAVE_FREQ) == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch{epoch}.pth", payload)

        log_stats = {
            "epoch": int(epoch),
            "train_loss": float(train_stats["loss"]),
            "train_opt_steps": float(train_stats["opt_steps"]),
            "train_lr": float(optimizer.param_groups[0]["lr"]),
            "test_acc1": float(test_stats["main_acc1"]),
            "test_acc5": float(test_stats["main_acc5"]),
            "test_acc1_ema": float(test_stats["ema_acc1"]),
            "test_acc5_ema": float(test_stats["ema_acc5"]),
            "max_accuracy_main": float(max_acc_main),
            "max_accuracy_ema": float(max_acc_ema),
            "n_parameters": int(n_parameters),
            "params_m": float(n_parameters / 1e6),
            "mac_g": None if not math.isfinite(mac_g) else float(mac_g),
            "use_ema": bool(USE_EMA),
            "ema_decay": float(EMA_DECAY),
            "use_distill": bool(USE_DISTILL),
            "distill_type": str(DISTILL_TYPE),
            "distill_alpha": float(DISTILL_ALPHA),
            "distill_tau": float(DISTILL_TAU),
            "teacher_model": str(TEACHER_MODEL),
            "compile_enabled": bool(COMPILE_ENABLED),
            "compile_backend": str(COMPILE_BACKEND),
        }
        with (output_dir / "log.txt").open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats, ensure_ascii=False) + "\n")

    total_time = time.time() - start_time
    print("Training time {}".format(str(datetime.timedelta(seconds=int(total_time)))))


if __name__ == "__main__":
    main()
