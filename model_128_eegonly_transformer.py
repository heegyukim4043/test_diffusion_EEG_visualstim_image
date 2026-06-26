# model_128_eegonly_transformer.py
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Noise schedules ──────────────────────────────────────────────────────────
def _linear_beta_schedule(num_timesteps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)


def _cosine_beta_schedule(num_timesteps, s=0.008):
    """
    Cosine noise schedule (Nichol & Dhariwal, 2021 — IDDPM).
    alpha_bar(t) = cos^2(((t/T + s)/(1+s)) * pi/2)
    Betas clipped to [1e-4, 0.9999] to avoid numerical issues.
    """
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.9999).float()


def _gaussian_window(win_size, channel, device, sigma=1.5):
    coords = torch.arange(win_size, device=device).float() - win_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]
    window = window.expand(channel, 1, win_size, win_size).contiguous()
    return window


def _ssim(img1, img2, data_range=2.0, k1=0.01, k2=0.03, win_size=11, eps=1e-8):
    # img1/img2 in [-1, 1], so data_range=2.0
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    channel = img1.size(1)
    window = _gaussian_window(win_size, channel, img1.device)

    mu1 = F.conv2d(img1, window, padding=win_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=win_size // 2, groups=channel)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=win_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=win_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=win_size // 2, groups=channel) - mu12

    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + eps
    )
    return ssim_map.mean()


# ---------------------------------------------------------
# 1. Time Embedding
# ---------------------------------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, time_dim: int = 256):
        super().__init__()
        self.time_dim = time_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.time_dim // 2
        device = t.device
        freqs = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / half_dim)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.time_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


# ---------------------------------------------------------
# 2. EEG Encoder (Conv + Transformer)
# ---------------------------------------------------------
class EEGEncoderTransformer(nn.Module):
    def __init__(
        self,
        eeg_channels: int = 32,
        eeg_hidden_dim: int = 256,
        out_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(eeg_channels, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, eeg_hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, eeg_hidden_dim),
            nn.SiLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=eeg_hidden_dim,
            nhead=n_heads,
            dim_feedforward=eeg_hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc = nn.Sequential(
            nn.Linear(eeg_hidden_dim, eeg_hidden_dim),
            nn.SiLU(),
            nn.Linear(eeg_hidden_dim, out_dim),
        )

    @staticmethod
    def _sinusoidal_pos_embed(seq_len: int, dim: int, device) -> torch.Tensor:
        """Sinusoidal positional encoding (고정값, 저장 불필요)."""
        pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
        i   = torch.arange(0, dim, 2, device=device).float()
        div = torch.exp(-i * (math.log(10000.0) / dim))
        emb = torch.zeros(seq_len, dim, device=device)
        emb[:, 0::2] = torch.sin(pos * div)
        emb[:, 1::2] = torch.cos(pos * div[:dim // 2])
        return emb.unsqueeze(0)  # (1, T', D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6
        x_norm = (x - mean) / std

        h = self.conv(x_norm)  # (B, D, T')
        h = h.transpose(1, 2)  # (B, T', D)

        pos = self._sinusoidal_pos_embed(h.size(1), h.size(2), h.device)
        h = h + pos
        h = self.transformer(h)  # (B, T', D)
        h = h.mean(dim=1)  # (B, D)
        return self.fc(h)


# ---------------------------------------------------------
# 3. Enhanced EEG Encoder V2
#    OccipitalChannelGate + MultiScaleStem + Transformer
# ---------------------------------------------------------
class OccipitalChannelGate(nn.Module):
    """
    Learnable per-channel soft gate with occipital prior (BioSemi 32-ch layout).

    BioSemi 32-ch channel order (0-indexed):
      0:FP1  1:AF3  2:F7   3:F3   4:FC1  5:FC5
      6:T7   7:C3   8:CP1  9:CP5 10:P7  11:P3
     12:Pz  13:PO3 14:O1  15:Oz  16:O2  17:PO4
     18:P4  19:P8  20:CP6 21:CP2 22:C4  23:T8
     24:FC6 25:FC2 26:F4  27:F8  28:AF4 29:FP2
     30:FZ  31:Cz

    Default prior (for 32-ch BioSemi):
      core occipital   (O1, Oz, O2)          : logit = +2.0  → sigmoid ≈ 0.88
      parieto-occipital (PO3, PO4)           : logit = +1.5  → sigmoid ≈ 0.82
      parietal (P7, P3, Pz, P4, P8)          : logit = +1.0  → sigmoid ≈ 0.73
      all others                              : logit = 0.0   → sigmoid = 0.50

    The gate is fully learnable — prior is a warm start only.

    Parameters
    ----------
    n_channels        : total EEG channel count
    occipital_indices : list of (idx, bias) tuples, or list of idx (uniform bias).
                        None → use BioSemi 32-ch default (only when n_channels==32)
    occipital_bias    : default logit when plain index list is given (no per-idx bias)
    """

    # BioSemi 32-ch defaults  (idx, logit)
    _BIOSEMI32_PRIOR = [
        (14, 2.0), (15, 2.0), (16, 2.0),   # O1, Oz, O2
        (13, 1.5), (17, 1.5),               # PO3, PO4
        (10, 1.0), (11, 1.0), (12, 1.0),   # P7, P3, Pz
        (18, 1.0), (19, 1.0),               # P4, P8
    ]

    def __init__(
        self,
        n_channels: int,
        occipital_indices=None,
        occipital_bias: float = 1.0,
    ):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(n_channels))

        with torch.no_grad():
            if occipital_indices is None:
                # use BioSemi 32-ch default if channel count matches
                if n_channels == 32:
                    for idx, logit in self._BIOSEMI32_PRIOR:
                        self.gate[idx] = logit
                # else: uniform (all zeros → sigmoid=0.5)
            else:
                # accept plain int list or (idx, logit) list
                for entry in occipital_indices:
                    if isinstance(entry, (list, tuple)):
                        idx, logit = entry[0], entry[1]
                    else:
                        idx, logit = int(entry), occipital_bias
                    if 0 <= idx < n_channels:
                        self.gate[idx] = logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        w = torch.sigmoid(self.gate).view(1, -1, 1)
        return x * w


class MultiScaleStem(nn.Module):
    """
    Multi-scale temporal conv stem with depthwise-separable convolutions.

    3 parallel branches (k = 7 / 15 / 31 samples @ 1024 Hz ≈ 7 / 15 / 30 ms):
      gamma/HFO band → beta band → alpha band

    Each branch:
      depthwise Conv1d (per-channel temporal filter)
      → pointwise Conv1d (cross-channel mixing)
      → GroupNorm → SiLU → Dropout

    Output: (B, n_filters * 3, T)  — concatenated across branches
    """
    def __init__(
        self,
        in_channels: int,
        n_filters: int = 32,    # filters per branch
        dropout: float = 0.1,
    ):
        super().__init__()
        scales = [7, 15, 31]
        self.branches = nn.ModuleList([
            nn.Sequential(
                # depthwise: temporal filter per channel
                nn.Conv1d(in_channels, in_channels, kernel_size=k,
                          padding=k // 2, groups=in_channels, bias=False),
                # pointwise: channel mixing → n_filters
                nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, n_filters), n_filters),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            for k in scales
        ])
        self.out_channels = n_filters * len(scales)   # 96 for n_filters=32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([b(x) for b in self.branches], dim=1)


class EEGEncoderV2(nn.Module):
    """
    Enhanced EEG encoder (V2) for the diffusion generation model.

    Architecture:
      z-score normalisation
      → OccipitalChannelGate          (learnable channel weighting, occipital prior)
      → MultiScaleStem k=7/15/31      (depthwise-separable, 3 temporal scales)
      → Conv1d(96 → hidden_dim, s=2)  (temporal downsampling + channel projection)
      → GroupNorm + SiLU
      → Sinusoidal PE + TransformerEncoder
      → MeanPool → LayerNorm → MLP → out_dim

    Drop-in replacement for EEGEncoderTransformer (same interface).
    """

    def __init__(
        self,
        eeg_channels: int = 32,
        eeg_hidden_dim: int = 256,
        out_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        stem_filters: int = 32,          # filters per scale branch (total = 3×)
        occipital_indices=None,          # None → BioSemi32 default prior
        occipital_bias: float = 1.0,
    ):
        super().__init__()

        # 1. channel gate
        self.channel_gate = OccipitalChannelGate(
            eeg_channels, occipital_indices, occipital_bias
        )

        # 2. multi-scale depthwise-separable stem → (B, stem_filters*3, T)
        self.stem = MultiScaleStem(eeg_channels, n_filters=stem_filters, dropout=dropout)
        stem_ch = self.stem.out_channels  # 96 for stem_filters=32

        # 3. temporal downsampling + projection to hidden_dim
        self.proj = nn.Sequential(
            nn.Conv1d(stem_ch, eeg_hidden_dim, kernel_size=5, stride=2,
                      padding=2, bias=False),
            nn.GroupNorm(8, eeg_hidden_dim),
            nn.SiLU(),
        )

        # 4. Transformer (identical to V1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=eeg_hidden_dim, nhead=n_heads,
                dim_feedforward=eeg_hidden_dim * 4,
                dropout=dropout, batch_first=True,
            ),
            num_layers=n_layers,
        )

        # 5. output head
        self.fc = nn.Sequential(
            nn.LayerNorm(eeg_hidden_dim),
            nn.Linear(eeg_hidden_dim, eeg_hidden_dim),
            nn.SiLU(),
            nn.Linear(eeg_hidden_dim, out_dim),
        )

    @staticmethod
    def _sinusoidal_pos_embed(seq_len: int, dim: int, device) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
        i   = torch.arange(0, dim, 2, device=device).float()
        div = torch.exp(-i * (math.log(10000.0) / dim))
        emb = torch.zeros(seq_len, dim, device=device)
        emb[:, 0::2] = torch.sin(pos * div)
        emb[:, 1::2] = torch.cos(pos * div[:dim // 2])
        return emb.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # z-score per trial (B, C, T)
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)
        x = self.channel_gate(x)          # occipital-biased weighting
        h = self.stem(x)                  # (B, 96, T)
        h = self.proj(h)                  # (B, hidden_dim, T//2)
        h = h.transpose(1, 2)            # (B, T//2, hidden_dim)
        h = h + self._sinusoidal_pos_embed(h.size(1), h.size(2), h.device)
        h = self.transformer(h).mean(1)   # (B, hidden_dim)
        return self.fc(h)                 # (B, out_dim)


# ---------------------------------------------------------
# 4. FiLM Residual Block
# ---------------------------------------------------------
class FiLMResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, cond_scale: float = 2.0):
        super().__init__()
        self.cond_scale = cond_scale

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.emb_proj = nn.Linear(emb_dim, 2 * out_ch)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)

        gamma_beta = self.emb_proj(emb)
        gamma, beta = gamma_beta.chunk(2, dim=1)

        gamma = 1.0 + self.cond_scale * gamma
        beta = self.cond_scale * beta

        h = h * gamma.unsqueeze(-1).unsqueeze(-1) + beta.unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)

        return self.skip(x) + h


# ---------------------------------------------------------
# 4. ResBlock + Down/Up
# ---------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.time_proj = nn.Linear(time_dim, out_ch)
        self.cond_proj = nn.Linear(cond_dim, out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        t = self.time_proj(t_emb)
        c = self.cond_proj(cond_vec)
        tc = (t + c).unsqueeze(-1).unsqueeze(-1)
        h = h + tc

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            time_dim: int,
            cond_dim: int,
            n_res: int = 2,
            downsample: bool = True,
    ):
        super().__init__()
        res_blocks = []
        ch = in_ch
        for _ in range(n_res):
            res_blocks.append(ResBlock(ch, out_ch, time_dim, cond_dim))
            ch = out_ch
        self.res_blocks = nn.ModuleList(res_blocks)

        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor, t_emb: torch.Tensor):
        for res in self.res_blocks:
            x = res(x, cond_vec, t_emb)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(
            self,
            in_ch: int,
            skip_ch: int,
            out_ch: int,
            time_dim: int,
            cond_dim: int,
            n_res: int = 2,
            upsample: bool = True,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, in_ch, 4, stride=2, padding=1
        ) if upsample else nn.Identity()

        res_blocks = []
        res_blocks.append(ResBlock(in_ch + skip_ch, out_ch, time_dim, cond_dim))
        for _ in range(n_res - 1):
            res_blocks.append(ResBlock(out_ch, out_ch, time_dim, cond_dim))
        self.res_blocks = nn.ModuleList(res_blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                cond_vec: torch.Tensor, t_emb: torch.Tensor):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        for res in self.res_blocks:
            x = res(x, cond_vec, t_emb)
        return x


# ---------------------------------------------------------
# 5. UNet (128x128)
# ---------------------------------------------------------
class UNet128(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            base_channels: int = 64,
            time_dim: int = 256,
            cond_dim: int = 256,
            ch_mult=(1, 2, 4, 8),
            emb_dim: int = None,
            cond_scale: float = 2.0,
            n_res_blocks: int = 2,
    ):
        super().__init__()

        if emb_dim is None:
            emb_dim = time_dim + cond_dim

        assert len(ch_mult) == 4

        c0 = base_channels * ch_mult[0]
        c1 = base_channels * ch_mult[1]
        c2 = base_channels * ch_mult[2]
        c3 = base_channels * ch_mult[3]

        self.inc = nn.Conv2d(in_channels, c0, kernel_size=3, padding=1)
        self.res1 = FiLMResBlock(c0, c0, emb_dim, cond_scale)

        self.down1 = DownBlock(c0, c1, time_dim, cond_dim, n_res=n_res_blocks)
        self.down2 = DownBlock(c1, c2, time_dim, cond_dim, n_res=n_res_blocks)
        self.down3 = DownBlock(c2, c3, time_dim, cond_dim, n_res=n_res_blocks)

        self.mid1 = FiLMResBlock(c3, c3, emb_dim, cond_scale)
        self.mid2 = FiLMResBlock(c3, c3, emb_dim, cond_scale)

        self.up3 = UpBlock(c3, c3, c2, time_dim, cond_dim, n_res=n_res_blocks)
        self.up2 = UpBlock(c2, c2, c1, time_dim, cond_dim, n_res=n_res_blocks)
        self.up1 = UpBlock(c1, c1, c0, time_dim, cond_dim, n_res=n_res_blocks)

        self.outc = nn.Conv2d(c0, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([t_emb, cond_emb], dim=1)

        x1 = self.inc(x)
        x1 = self.res1(x1, emb)

        x2, s1 = self.down1(x1, cond_emb, t_emb)
        x3, s2 = self.down2(x2, cond_emb, t_emb)
        x4, s3 = self.down3(x3, cond_emb, t_emb)

        h = self.mid1(x4, emb)
        h = self.mid2(h, emb)

        h = self.up3(h, s3, cond_emb, t_emb)
        h = self.up2(h, s2, cond_emb, t_emb)
        h = self.up1(h, s1, cond_emb, t_emb)

        out = self.outc(h)
        return out


# ---------------------------------------------------------
# 6. Diffusion wrapper (EEG-only conditioning)
# ---------------------------------------------------------
class EEGDiffusionModel128(nn.Module):
    def __init__(
            self,
            img_size: int = 128,
            img_channels: int = 3,
            eeg_channels: int = 32,
            num_classes: int = 9,
            num_timesteps: int = 200,
            base_channels: int = 64,
            time_dim: int = 256,
            cond_dim: int = 256,
            eeg_hidden_dim: int = 256,
            cond_scale: float = 2.0,
            beta_start: float = 1e-4,
            beta_end: float = 2e-2,
            beta_schedule: str = "linear",   # "linear" | "cosine"
            ch_mult=(1, 2, 4, 8),
            n_res_blocks: int = 2,
            lambda_rec: float = 0.02,
            lambda_ssim: float = 0.05,
            eeg_tf_heads: int = 4,
            eeg_tf_layers: int = 2,
            eeg_tf_dropout: float = 0.1,
            encoder_version: str = "v1",    # "v1" = original | "v2" = enhanced
            eeg_stem_filters: int = 32,     # V2: filters per scale branch
            eeg_occipital_indices=None,     # V2: None → BioSemi32 auto-prior; or [(idx,logit),...] or [idx,...]
    ):
        super().__init__()
        assert img_size == 128
        assert encoder_version in ("v1", "v2"), "encoder_version must be 'v1' or 'v2'"

        self.img_size = img_size
        self.img_channels = img_channels
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        self.lambda_rec = lambda_rec
        self.lambda_ssim = lambda_ssim
        self.beta_schedule = beta_schedule

        if beta_schedule == "cosine":
            betas = _cosine_beta_schedule(num_timesteps)
        else:
            betas = _linear_beta_schedule(num_timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

        self.time_embed = TimeEmbedding(time_dim=time_dim)

        _enc_kwargs = dict(
            eeg_channels=eeg_channels,
            eeg_hidden_dim=eeg_hidden_dim,
            out_dim=cond_dim,
            n_heads=eeg_tf_heads,
            n_layers=eeg_tf_layers,
            dropout=eeg_tf_dropout,
        )
        if encoder_version == "v2":
            self.eeg_encoder = EEGEncoderV2(
                **_enc_kwargs,
                stem_filters=eeg_stem_filters,
                occipital_indices=eeg_occipital_indices,
            )
        else:
            self.eeg_encoder = EEGEncoderTransformer(**_enc_kwargs)
        self.class_emb = nn.Embedding(num_classes, cond_dim)

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        emb_dim = time_dim + cond_dim

        self.unet = UNet128(
            in_channels=img_channels,
            base_channels=base_channels,
            time_dim=time_dim,
            cond_dim=cond_dim,
            ch_mult=ch_mult,
            emb_dim=emb_dim,
            cond_scale=cond_scale,
            n_res_blocks=n_res_blocks,
        )

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int]) -> torch.Tensor:
        out = a.gather(-1, t)
        return out.view(-1, 1, 1, 1).expand(x_shape)

    def get_cond_emb_eeg_only(self, eeg: torch.Tensor) -> torch.Tensor:
        eeg_emb = self.eeg_encoder(eeg)
        cond = self.cond_proj(eeg_emb)
        return cond

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha_bar * x_start + sqrt_one_minus * noise

    def p_losses(
            self,
            x_start: torch.Tensor,
            eeg: torch.Tensor,
            labels: torch.Tensor,
            t: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        t_emb = self.time_embed(t)
        cond_emb = self.get_cond_emb_eeg_only(eeg)
        cond_emb = cond_emb * 1.5

        eps_pred = self.unet(x_noisy, t_emb, cond_emb)

        with torch.no_grad():
            alpha_bar_t = self._extract(self.alphas_cumprod, t, eps_pred.shape)
            w_eps = torch.sqrt(1.0 - alpha_bar_t)

        mse_eps = (eps_pred - noise) ** 2
        loss_noise = (w_eps * mse_eps).mean()

        sqrt_alpha_bar = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        x0_pred = (x_noisy - sqrt_one_minus * eps_pred) / (sqrt_alpha_bar + 1e-8)
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        with torch.no_grad():
            w_x0 = alpha_bar_t

        loss_recon = (w_x0 * (x0_pred - x_start).abs()).mean()
        loss_ssim = 1.0 - _ssim(x0_pred, x_start)
        return loss_noise + self.lambda_rec * loss_recon + self.lambda_ssim * loss_ssim

    @torch.no_grad()
    def sample(
            self,
            eeg: torch.Tensor,
            labels: torch.Tensor,
            num_steps: int = None,
            guidance_scale: float = 2.5,
    ) -> torch.Tensor:
        device = eeg.device
        b = eeg.size(0)
        T = self.num_timesteps if num_steps is None else min(self.num_timesteps, num_steps)

        x_t = torch.randn(b, self.img_channels, self.img_size, self.img_size, device=device)

        cond = self.get_cond_emb_eeg_only(eeg)
        cond = cond * guidance_scale

        for i in reversed(range(T)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            t_emb = self.time_embed(t)
            eps_theta = self.unet(x_t, t_emb, cond)

            beta_t = self.betas[i]
            alpha_t = self.alphas[i]
            alpha_bar_t = self.alphas_cumprod[i]

            sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bar_t)
            sqrt_recip_alpha = torch.sqrt(1.0 / alpha_t)

            mean = sqrt_recip_alpha * (x_t - beta_t / sqrt_one_minus_ab * eps_theta)

            if i > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t) * noise
            else:
                x_t = mean

        return x_t.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample_ddim(
            self,
            eeg: torch.Tensor,
            num_steps: int = None,
            guidance_scale: float = 2.5,
            eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM sampling with EEG-only conditioning. eta=0.0 is deterministic.
        """
        device = eeg.device
        b = eeg.size(0)
        T = self.num_timesteps if num_steps is None else min(self.num_timesteps, num_steps)

        x_t = torch.randn(b, self.img_channels, self.img_size, self.img_size, device=device)

        cond = self.get_cond_emb_eeg_only(eeg)
        cond = cond * guidance_scale

        for i in reversed(range(T)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            t_emb = self.time_embed(t)
            eps_theta = self.unet(x_t, t_emb, cond)

            alpha_bar_t = self.alphas_cumprod[i]
            alpha_bar_prev = self.alphas_cumprod[i - 1] if i > 0 else torch.tensor(1.0, device=device)

            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)

            x0_pred = (x_t - sqrt_one_minus * eps_theta) / (sqrt_alpha_bar_t + 1e-8)

            if i > 0 and eta > 0.0:
                sigma = (
                    eta
                    * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
                    * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
                )
                noise = torch.randn_like(x_t)
            else:
                sigma = 0.0
                noise = torch.zeros_like(x_t)

            dir_xt = torch.sqrt(1.0 - alpha_bar_prev - sigma ** 2) * eps_theta
            x_t = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

        return x_t.clamp(-1.0, 1.0)
