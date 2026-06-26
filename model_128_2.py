import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# 1. 시간 임베딩
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
        """
        t: (B,) integer timesteps
        return: (B, time_dim)
        """
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
# 2. EEG Encoder (1D CNN + Global pooling)
#    - 여기서 EEG z-score 정규화 수행
# ---------------------------------------------------------
class EEGEncoder(nn.Module):
    def __init__(self, eeg_channels: int = 32, eeg_hidden_dim: int = 256, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(eeg_channels, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, eeg_hidden_dim),
            nn.SiLU(),
            nn.Linear(eeg_hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        """
        # z-score 정규화 (time 축 기준)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6
        x_norm = (x - mean) / std

        h = self.conv(x_norm)
        h = self.pool(h).squeeze(-1)  # (B, 128)
        h = self.fc(h)  # (B, out_dim)
        return h


# ---------------------------------------------------------
# 3. FiLM Residual Block (조건 스케일 강화)
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
        """
        x: (B, C, H, W)
        emb: (B, emb_dim)
        """
        h = self.conv1(x)
        h = self.norm1(h)

        gamma_beta = self.emb_proj(emb)  # (B, 2*out_ch)
        gamma, beta = gamma_beta.chunk(2, dim=1)

        gamma = 1.0 + self.cond_scale * gamma
        beta = self.cond_scale * beta

        h = h * gamma.unsqueeze(-1).unsqueeze(-1) + beta.unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)

        return self.skip(x) + h


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, cond_scale: float = 2.0):
        super().__init__()
        self.res = FiLMResBlock(in_ch, out_ch, emb_dim, cond_scale)
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.res(x, emb)
        x = self.down(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, emb_dim: int, cond_scale: float = 2.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.res = FiLMResBlock(out_ch + skip_ch, out_ch, emb_dim, cond_scale)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, emb)
        return x


# ---------------------------------------------------------
# 4. UNet (128x128 전용)
# ---------------------------------------------------------
class UNet128(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        emb_dim: int = 512,
        cond_scale: float = 2.0,
    ):
        super().__init__()
        c = base_channels

        self.inc = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
        self.res1 = FiLMResBlock(c, c, emb_dim, cond_scale)

        self.down1 = DownBlock(c, c * 2, emb_dim, cond_scale)      # 128 -> 64
        self.down2 = DownBlock(c * 2, c * 4, emb_dim, cond_scale)  # 64 -> 32
        self.down3 = DownBlock(c * 4, c * 4, emb_dim, cond_scale)  # 32 -> 16

        self.mid1 = FiLMResBlock(c * 4, c * 4, emb_dim, cond_scale)
        self.mid2 = FiLMResBlock(c * 4, c * 4, emb_dim, cond_scale)

        self.up3 = UpBlock(c * 4, c * 4, c * 4, emb_dim, cond_scale)  # 16 -> 32
        self.up2 = UpBlock(c * 4, c * 2, c * 2, emb_dim, cond_scale)  # 32 -> 64
        self.up1 = UpBlock(c * 2, c, c, emb_dim, cond_scale)          # 64 -> 128

        self.outc = nn.Conv2d(c, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,128,128)
        t_emb: (B, time_dim)
        cond_emb: (B, cond_dim)
        """
        emb = torch.cat([t_emb, cond_emb], dim=1)  # (B, emb_dim)

        x1 = self.inc(x)
        x1 = self.res1(x1, emb)     # (B, c, 128, 128)

        x2 = self.down1(x1, emb)    # (B, 2c, 64, 64)
        x3 = self.down2(x2, emb)    # (B, 4c, 32, 32)
        x4 = self.down3(x3, emb)    # (B, 4c, 16, 16)

        h = self.mid1(x4, emb)
        h = self.mid2(h, emb)

        h = self.up3(h, x3, emb)    # (B, 4c, 32, 32)
        h = self.up2(h, x2, emb)    # (B, 2c, 64, 64)
        h = self.up1(h, x1, emb)    # (B, c, 128, 128)

        out = self.outc(h)          # (B, 3, 128, 128)
        return out


# ---------------------------------------------------------
# 5. Diffusion wrapper (EEG + class conditioning)
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
        cfg_scale: float = 1.5,      # ← cond_emb 스케일 (train & sample 공통)
        lambda_x0: float = 0.1,      # ← x0 L1 보조 손실 weight (0이면 사용 안 함)
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        assert img_size == 128, "이 모델은 128x128 이미지를 기준으로 설계되었습니다."

        self.img_size = img_size
        self.img_channels = img_channels
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        self.cfg_scale = cfg_scale
        self.lambda_x0 = lambda_x0

        # diffusion schedule
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

        # 모듈들
        self.time_embed = TimeEmbedding(time_dim=time_dim)
        self.eeg_encoder = EEGEncoder(
            eeg_channels=eeg_channels,
            eeg_hidden_dim=eeg_hidden_dim,
            out_dim=cond_dim,
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
            emb_dim=emb_dim,
            cond_scale=cond_scale,
        )

    # ------------------ helper ------------------
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int]) -> torch.Tensor:
        """
        a: (T,), t: (B,)
        -> (B, 1, 1, 1) broadcasting shape
        """
        out = a.gather(-1, t)
        return out.view(-1, 1, 1, 1).expand(x_shape)

    def get_cond_emb(self, eeg: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        eeg: (B, C, T)
        labels: (B,) in {1..9} or {0..8}
        """
        # label 1~9 -> 0~8로 맞추기
        y = labels.long()
        if y.min() >= 1:
            y = y - 1
        y = torch.clamp(y, 0, self.num_classes - 1)

        eeg_emb = self.eeg_encoder(eeg)          # (B, cond_dim)
        cls_emb = self.class_emb(y)              # (B, cond_dim)

        cond = eeg_emb + cls_emb                 # EEG + class embedding
        cond = self.cond_proj(cond)              # 추가 non-linear
        return cond

    # ------------------ diffusion core ------------------
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*noise
        """
        sqrt_alpha_bar = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alpha_bar * x_start + sqrt_one_minus * noise

    def p_losses(
        self,
        x_start: torch.Tensor,
        eeg: torch.Tensor,
        labels: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training loss: MSE between predicted noise and true noise + (optional) x0 L1.
        x_start: (B,3,128,128), in [-1,1]
        """
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        t_emb = self.time_embed(t)
        cond_emb = self.get_cond_emb(eeg, labels)

        # 학습/샘플링 모두 cfg_scale 사용
        cond_emb = cond_emb * self.cfg_scale

        eps_pred = self.unet(x_noisy, t_emb, cond_emb)

        # 기본 noise MSE
        loss = F.mse_loss(eps_pred, noise)

        # 보조 x0 재구성 L1 (엣지/디테일 강제)
        if self.lambda_x0 > 0.0:
            sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            x0_pred = (x_noisy - sqrt_one_minus * eps_pred) / (sqrt_alpha_bar + 1e-8)
            loss_x0 = F.l1_loss(x0_pred, x_start)
            loss = loss + self.lambda_x0 * loss_x0

        return loss

    # ------------------ sampling (DDPM) ------------------
    @torch.no_grad()
    def sample_guided(self, eeg, labels, num_steps: int = None, cfg_scale: float = None):
        """
        sample_unseen_subject_128_2.py 에서 사용하는 래퍼.
        - cfg_scale 이 주어지면 일시적으로 self.cfg_scale 을 바꿔서 sample() 호출
        - 주어지지 않으면 현재 self.cfg_scale 로 sample() 호출
        """
        if cfg_scale is None:
            # 현재 설정된 self.cfg_scale 그대로 사용
            return self.sample(eeg, labels, num_steps=num_steps)

        # 임시로 cfg_scale 변경
        old_cfg = self.cfg_scale
        self.cfg_scale = cfg_scale
        try:
            return self.sample(eeg, labels, num_steps=num_steps)
        finally:
            # 원래 값 복원
            self.cfg_scale = old_cfg


    def sample(
        self,
        eeg: torch.Tensor,
        labels: torch.Tensor,
        num_steps: int = None,
    ) -> torch.Tensor:
        """
        eeg: (B,C,T)
        labels: (B,)
        return: (B,3,128,128) in [-1,1]
        """
        device = eeg.device
        b = eeg.size(0)
        T = self.num_timesteps if num_steps is None else min(
            num_steps, self.num_timesteps
        )

        x_t = torch.randn(b, self.img_channels, self.img_size, self.img_size, device=device)

        cond_emb = self.get_cond_emb(eeg, labels)
        cond_emb = cond_emb * self.cfg_scale  # 학습과 동일 스케일

        for i in reversed(range(T)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            t_emb = self.time_embed(t)
            eps_theta = self.unet(x_t, t_emb, cond_emb)

            beta_t = self.betas[i]
            alpha_t = self.alphas[i]
            alpha_bar_t = self.alphas_cumprod[i]

            sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bar_t)
            sqrt_recip_alpha = torch.sqrt(1.0 / alpha_t)

            mean = sqrt_recip_alpha * (
                x_t - beta_t / sqrt_one_minus_ab * eps_theta
            )

            if i > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t) * noise
            else:
                x_t = mean

        return x_t
