import math, torch
import torch.nn as nn
import torch.nn.functional as F

from .down_up import make_down, make_up, make_block


MODEL_REGISTRY = {}
def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator



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

def sinkhorn_ot_gate(f_main, f_aux, grid=8, iters=5, eps=0.05):
    """
    f_main, f_aux: (N,C,H,W). We pool to (N,C,h,w), flatten to P and Q (h*w tokens).
    Cost = 1 - cosine; entropic-regularized OT -> coupling π.
    Return g in [0,1] upsampled to (H,W) using row marginals of π.
    """
    N,C,H,W = f_main.shape
    h = max(1, H // grid); w = max(1, W // grid)
    pm = F.adaptive_avg_pool2d(f_main, (h,w))   # (N,C,h,w)
    pa = F.adaptive_avg_pool2d(f_aux,  (h,w))
    # flatten tokens
    X = pm.view(N, C, -1)                       # (N,C,M)
    Y = pa.view(N, C, -1)                       # (N,C,M)
    # cosine distance
    Xm = F.normalize(X, dim=1)
    Ym = F.normalize(Y, dim=1)
    #Cmat = 1 - torch.einsum('ncm,ncn->nmn', Xm, Ym)  # (N,M,M)
    # Xm, Ym 已按 dim=1 做过 F.normalize，形状都是 (N, C, M)
    sim = torch.matmul(Xm.transpose(1, 2), Ym)  # (N, M, M)
    Cmat = 1 - sim

    # entropic OT with uniform marginals
    M = Cmat.size(-1)
    K = torch.exp(-Cmat / eps)                   # kernel
    u = torch.ones(N, M, device=K.device) / M
    v = torch.ones(N, M, device=K.device) / M
    for _ in range(iters):
        u = 1.0 / (K @ v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
        v = 1.0 / (K.transpose(1,2) @ u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
    Pi = (u.unsqueeze(-1) * K) * v.unsqueeze(-2)        # (N,M,M)
    row_mass = Pi.sum(dim=-1)                           # (N,M)
    g_coarse = row_mass.view(N, 1, h, w)                # more mass -> trust main
    g = F.interpolate(g_coarse, size=(H,W), mode='bilinear', align_corners=False)
    g = g.clamp(0.0, 1.0)
    fused = g * f_main + (1 - g) * f_aux
    return fused, g, {"ot_cost": (Cmat * Pi).mean(), "gate_name":'ssi'}

class SubbandInjector(nn.Module):
    """
    将 aux 子带 (LH/HL/HH) 先从 aux_ch 映射到 skip_ch，再与主干跳连融合。
    """
    def __init__(self, skip_ch, aux_ch):
        super().__init__()
        # 先投影到 skip_ch，再做轻量 3×3
        self.adp_LH = nn.Sequential(
            nn.Conv2d(aux_ch, skip_ch, 1), nn.GELU(),
            nn.Conv2d(skip_ch, skip_ch, 3, padding=1)
        )
        self.adp_HL = nn.Sequential(
            nn.Conv2d(aux_ch, skip_ch, 1), nn.GELU(),
            nn.Conv2d(skip_ch, skip_ch, 3, padding=1)
        )
        self.adp_HH = nn.Sequential(
            nn.Conv2d(aux_ch, skip_ch, 1), nn.GELU(),
            nn.Conv2d(skip_ch, skip_ch, 3, padding=1)
        )
        self.mix = nn.Sequential(
            nn.Conv2d(skip_ch * 2, skip_ch, 1), nn.GELU(),
            nn.Conv2d(skip_ch, skip_ch, 1)
        )

    def forward(self, skip_main, LH, HL, HH):
        # 尺寸对齐
        H, W = skip_main.shape[-2:]
        if LH.shape[-2:] != (H, W):
            LH = F.interpolate(LH, size=(H, W), mode='bilinear', align_corners=False)
            HL = F.interpolate(HL, size=(H, W), mode='bilinear', align_corners=False)
            HH = F.interpolate(HH, size=(H, W), mode='bilinear', align_corners=False)

        enh = self.adp_LH(LH) + self.adp_HL(HL) + self.adp_HH(HH)   # -> (N, skip_ch, H, W)
        return self.mix(torch.cat([skip_main, enh], dim=1))          # -> (N, skip_ch, H, W)

@register_model('unet_wave')
class UNetSegHead_WaveOT(nn.Module):
    """
    - Wavelet-UNet: DWT/IWT instead of pooling/upsampling
    - Sinkhorn OT gate for pre-encoder fusion
    - Subband Skip Injection to pass aux detail into each skip
    The proposed **Wavelet-UNet with Sinkhorn Optimal-Transport Gate** is designed to achieve robust multimodal feature fusion and high-fidelity boundary recovery in histological segmentation tasks. Unlike conventional UNets that rely on local pooling and naïve channel concatenation, this architecture replaces all down- and up-sampling operations with fixed **Discrete and Inverse Wavelet Transforms (DWT/IWT)**. The DWT factorizes the feature map into a low-frequency semantic component (LL) and three high-frequency subbands (LH, HL, HH), allowing the network to model coarse semantic context while explicitly preserving structural edges and textural cues. This multi-band decomposition yields sharper boundaries and consistent detail restoration during decoding without additional learnable parameters or computational overhead.
    To realize stable multimodal integration between the main and auxiliary modalities (e.g., H&E and multiplex IF channels), we introduce a **Sinkhorn Optimal-Transport (OT) Gate**. Instead of using fixed or attention-based weights, the OT Gate formulates feature alignment as an entropic optimal-transport problem between coarse spatial grids of the two feature maps. The resulting coupling plan provides adaptive spatial mixing weights that emphasize geometrically aligned regions while suppressing modality-specific noise. This mechanism ensures that the fusion is not only intensity-aware but also structure-consistent, improving cross-domain generalization.
    Finally, the decoder employs **Subband Skip Injection (SSI)** to pass auxiliary high-frequency subbands into each skip connection via lightweight band-wise adapters. This targeted injection reintroduces biologically relevant microstructures such as nuclei boundaries or proliferation hotspots precisely at their original scales. Together, the Wavelet-UNet, OT-based gating, and SSI modules establish a unified spectral–geometric fusion framework that significantly enhances boundary precision and modality robustness, setting a new direction for interpretable multimodal biomedical segmentation architectures.
    """
    def __init__(self, base=32, depth=(1,1,2,1)):
        super().__init__()
        d1,d2,d3,d4 = depth

        # stems
        self.inc_aux  = make_block('wave',12, base, depth=1)
        self.inc_main = make_block('wave',3, base, depth=1)


        # SSI modules at each scale
        self.ssi1 = SubbandInjector(skip_ch=base, aux_ch=base)  # a1_*：B
        self.ssi2 = SubbandInjector(skip_ch=base * 2, aux_ch=base)  # a1_*：B → 2B
        self.ssi3 = SubbandInjector(skip_ch=base * 4, aux_ch=base * 2)  # a2_*：2B → 4B
        self.ssi4 = SubbandInjector(skip_ch=base * 8, aux_ch=base * 4)  # a3_*：4B → 8B

        # Encoder pyramid (ConvNeXt)
        self.down1 = make_down("wave", base, base * 2, depth=d1)
        self.down2 = make_down("wave", base * 2, base * 4, depth=d2)
        self.down3 = make_down("wave", base * 4, base * 8, depth=d3)
        self.down4 = make_down("wave", base * 8, base * 16, depth=d4)

        # Store aux features at each scale (for adapters)
        self.aux_down1 = make_down("wave", base, base * 2, depth=1)
        self.aux_down2 = make_down("wave", base * 2, base * 4, depth=1)
        self.aux_down3 = make_down("wave", base * 4, base * 8, depth=1)
        self.aux_down4 = make_down("wave", base * 8, base * 16, depth=1)

        self.up1 = make_up("wave", base * 16, base * 8, in_ch_skip=base * 8, depth=2)
        self.up2 = make_up("wave", base * 8, base * 4, in_ch_skip=base * 4, depth=2)
        self.up3 = make_up("wave", base * 4, base * 2, in_ch_skip=base * 2, depth=2)
        self.up4 = make_up("wave", base * 2, base, in_ch_skip=base, depth=2)


        self.dwt = DWT_Haar()
        self.outc = nn.Conv2d(base, 1, 1)

    def forward(self, x):  # x: (N, 15, H, W)
        xa, xm = x[:, :12], x[:, 12:15]

        f_aux0 = self.inc_aux(xa)
        f_main0 = self.inc_main(xm)

        # --- 1) OT Gate fusion ---
        x1, g, ot_stats = sinkhorn_ot_gate(f_main0, f_aux0, grid=8, iters=5, eps=0.05)

        # --- 2) Main encoder with Wavelet decomposition ---
        e2_LL, (m1_LH, m1_HL, m1_HH) = self.down1(x1)
        e3_LL, (m2_LH, m2_HL, m2_HH) = self.down2(e2_LL)
        e4_LL, (m3_LH, m3_HL, m3_HH) = self.down3(e3_LL)
        e5_LL, (m4_LH, m4_HL, m4_HH) = self.down4(e4_LL)

        # --- 3) Aux encoder (for aux subbands) ---
        a0_LL, a0_LH, a0_HL, a0_HH = self.dwt(f_aux0)
        a2_LL, (a1_LH, a1_HL, a1_HH) = self.aux_down1(f_aux0)
        a3_LL, (a2_LH, a2_HL, a2_HH) = self.aux_down2(a2_LL)
        a4_LL, (a3_LH, a3_HL, a3_HH) = self.aux_down3(a3_LL)
        a5_LL, (a4_LH, a4_HL, a4_HH) = self.aux_down4(a4_LL)


        # --- 4) Decoder with IWT upsampling ---
        s4 = self.ssi4(e4_LL, a3_LH, a3_HL, a3_HH)
        y = self.up1(e5_LL, s4, m4_LH, m4_HL, m4_HH)

        s3 = self.ssi3(e3_LL, a2_LH, a2_HL, a2_HH)
        y = self.up2(y, s3, m3_LH, m3_HL, m3_HH)

        s2 = self.ssi2(e2_LL, a1_LH, a1_HL, a1_HH)
        y = self.up3(y, s2, m2_LH, m2_HL, m2_HH)

        s1 = self.ssi1(x1, a0_LH, a0_HL, a0_HH)
        y = self.up4(y, s1, m1_LH, m1_HL, m1_HH)

        logits = self.outc(y)
        return logits, g, ot_stats



