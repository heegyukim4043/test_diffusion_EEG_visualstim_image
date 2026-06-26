# model_128_eegonly_transformer.py
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.pos_embed = None

        self.fc = nn.Sequential(
            nn.Linear(eeg_hidden_dim, eeg_hidden_dim),
            nn.SiLU(),
            nn.Linear(eeg_hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6
        x_norm = (x - mean) / std

        h = self.conv(x_norm)  # (B, D, T')
        h = h.transpose(1, 2)  # (B, T', D)

        if self.pos_embed is None or self.pos_embed.size(1) != h.size(1):
            self.pos_embed = nn.Parameter(
                torch.zeros(1, h.size(1), h.size(2), device=h.device)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        h = h + self.pos_embed
        h = self.transformer(h)  # (B, T', D)
        h = h.mean(dim=1)  # (B, D)
        return self.fc(h)


# ---------------------------------------------------------
# 3. FiLM Residual Block
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
            ch_mult=(1, 2, 4, 8),
            n_res_blocks: int = 2,
            lambda_rec: float = 0.02,
            eeg_tf_heads: int = 4,
            eeg_tf_layers: int = 2,
            eeg_tf_dropout: float = 0.1,
    ):
        super().__init__()
        assert img_size == 128

        self.img_size = img_size
        self.img_channels = img_channels
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        self.lambda_rec = lambda_rec

        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
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
        self.eeg_encoder = EEGEncoderTransformer(
            eeg_channels=eeg_channels,
            eeg_hidden_dim=eeg_hidden_dim,
            out_dim=cond_dim,
            n_heads=eeg_tf_heads,
            n_layers=eeg_tf_layers,
            dropout=eeg_tf_dropout,
        )
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
        return loss_noise + self.lambda_rec * loss_recon

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
