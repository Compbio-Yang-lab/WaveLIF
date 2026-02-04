# physics_factory.py (revised: roll-based Neumann BC everywhere)
import math, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# 0) Haar DWT / IWT (fixed weights, no learnables)
# ---------------------------
class DWT_Haar(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5],[0.5, 0.5]])
        lh = torch.tensor([[0.5,-0.5],[0.5,-0.5]])
        hl = torch.tensor([[0.5, 0.5],[-0.5,-0.5]])
        hh = torch.tensor([[0.5,-0.5],[-0.5, 0.5]])
        k = torch.stack([ll, lh, hl, hh])  # (4,2,2)
        self.register_buffer("weight", k.view(4,1,2,2))
    def forward(self, x):  # (N,C,H,W) -> (N,4C,H/2,W/2)
        N,C,H,W = x.shape
        w = self.weight.repeat(C,1,1,1)  # (4C,1,2,2)
        y = F.conv2d(x, w, stride=2, groups=C)  # (N,4C,H/2,W/2)
        LL, LH, HL, HH = torch.chunk(y, 4, dim=1)
        return LL, LH, HL, HH

class IWT_Haar(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5],[0.5, 0.5]])
        lh = torch.tensor([[0.5,-0.5],[0.5,-0.5]])
        hl = torch.tensor([[0.5, 0.5],[-0.5,-0.5]])
        hh = torch.tensor([[0.5,-0.5],[-0.5, 0.5]])
        k = torch.stack([ll, lh, hl, hh])  # (4,2,2)
        self.register_buffer("weight", k.view(4,1,2,2))
    def forward(self, LL, LH, HL, HH):  # each (N,C,H,W) -> (N,C,2H,2W)
        x = torch.cat([LL, LH, HL, HH], dim=1)  # (N,4C,H,W)
        N,C4,H,W = x.shape
        C = C4 // 4
        w = self.weight.repeat(C,1,1,1)        # (4C,1,2,2)
        # transposed groups conv to invert stride-2 analysis
        y = F.conv_transpose2d(x, w, stride=2, groups=C)  # (N,C,2H,2W)
        return y

class LKA(nn.Module):
    """Large-Kernel-Attention-like: DWConv (k=9) -> DWConv (dilated k=9, d=4) -> 1x1."""
    def __init__(self, ch):
        super().__init__()
        self.dw1 = nn.Conv2d(ch, ch, 9, padding=4, groups=ch)
        self.dw2 = nn.Conv2d(ch, ch, 9, padding=16, dilation=4, groups=ch)
        self.pw  = nn.Conv2d(ch, ch, 1)
    def forward(self, x):
        a = self.dw1(x)
        a = self.dw2(a)
        a = self.pw(a)
        return x * (a.sigmoid())

class WaveletBlock(nn.Module):
    def __init__(self, ch, ratio=2):
        super().__init__()
        hid = ch*ratio
        self.norm = nn.GroupNorm(1, ch)
        self.lka  = LKA(ch)
        self.mlp  = nn.Sequential(nn.Conv2d(ch, hid, 1), nn.GELU(), nn.Conv2d(hid, ch, 1))
    def forward(self, x):
        s = x
        x = self.lka(self.norm(x))
        x = s + x
        s = x
        x = self.mlp(self.norm(x))
        return s + x

def wave_stage(in_ch, out_ch, depth=2):
    layers = [nn.Conv2d(in_ch, out_ch, 1)]
    for _ in range(depth):
        layers.append(WaveletBlock(out_ch))
    return nn.Sequential(*layers)

class WaveDown(nn.Module):
    def __init__(self, in_ch, out_ch, depth=2):
        super().__init__()
        self.dwt = DWT_Haar()
        self.body = wave_stage(in_ch, out_ch, depth)
    def forward(self, x):
        LL, LH, HL, HH = self.dwt(x)
        fe = self.body(LL)      # encode only LL
        return fe, (LH, HL, HH) # keep highs for SSI

class WaveUp(nn.Module):
    def __init__(self, in_ch, out_ch, in_ch_skip, depth=2):
        super().__init__()
        self.iwt = IWT_Haar()
        # 把高频子带从 aux/main 的 in_ch 投到解码 LL 的通道（=in_ch_dec_ll）
        self.proj_LH = nn.Conv2d(in_ch_skip, in_ch, 1)
        self.proj_HL = nn.Conv2d(in_ch_skip, in_ch, 1)
        self.proj_HH = nn.Conv2d(in_ch_skip, in_ch, 1)
        self.fuse = nn.Conv2d(in_ch + in_ch_skip, out_ch, 1)
        self.body = wave_stage(out_ch, out_ch, depth)

    def forward(self, x1_LL, skip_main, LH, HL, HH):
        # 1) 把高频子带投到与 x1_LL 相同的通道数
        LH = self.proj_LH(LH); HL = self.proj_HL(HL); HH = self.proj_HH(HH)
        # 2) 用 IWT 把 (LL,LH,HL,HH) 组合成更高分辨率
        x_up = self.iwt(x1_LL, LH, HL, HH)
        # 3) 与跳连拼接并细化
        x = torch.cat([skip_main, x_up], dim=1)
        x = self.fuse(x)
        return self.body(x)




# ----------------------------- factory -----------------------------
DOWN_REGISTRY = {
    "mde":WaveDown, # kwargs: depth
}
UP_REGISTRY = {
    "mde": WaveUp,
}
BLOCK_REGISTRY = {
    "mde": wave_stage,
}


def make_down(name: str, in_ch: int, out_ch: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in DOWN_REGISTRY:
        raise ValueError(f"Unknown down type '{name}'. Available: {list(DOWN_REGISTRY.keys())}")
    return DOWN_REGISTRY[name](in_ch, out_ch, **kwargs)

def make_up(name: str, in_ch: int, out_ch: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in UP_REGISTRY:
        raise ValueError(f"Unknown up type '{name}'. Available: {list(UP_REGISTRY.keys())}")
    return UP_REGISTRY[name](in_ch, out_ch, **kwargs)

def make_block(name: str, in_ch: int, out_ch: int, **kwargs):
    name = name.lower()
    if name not in BLOCK_REGISTRY:
        raise ValueError(f"Unknown down type '{name}'. Available: {list(BLOCK_REGISTRY.keys())}")
    return BLOCK_REGISTRY[name](in_ch, out_ch, **kwargs)
