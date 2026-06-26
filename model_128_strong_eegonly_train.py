# model_128.py
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
# 2. EEG Encoder
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
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, eeg_hidden_dim),
            nn.SiLU(),
            nn.Linear(eeg_hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        """
        # z-score (time 축 기준)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6
        x_norm = (x - mean) / std

        h = self.conv(x_norm)
        h = self.pool(h).squeeze(-1)  # (B, 128)
        h = self.fc(h)  # (B, out_dim)
        return h


# ---------------------------------------------------------
# 3. FiLM Residual Block (t_emb + cond_emb 결합)
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
        x:   (B, C, H, W)
        emb: (B, emb_dim)  # concat[t_emb, cond_emb]
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


# ---------------------------------------------------------
# 4. ResBlock + Down/Up (t_emb, cond_emb 분리)
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
        """
        x:        (B, C, H, W)
        cond_vec: (B, cond_dim)
        t_emb:    (B, time_dim)
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        t = self.time_proj(t_emb)
        c = self.cond_proj(cond_vec)
        tc = (t + c).unsqueeze(-1).unsqueeze(-1)  # (B, out_ch, 1, 1)
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

        # 1) 먼저 x를 upsample 할 때 쓸 convtranspose
        self.up = nn.ConvTranspose2d(
            in_ch, in_ch, 4, stride=2, padding=1
        ) if upsample else nn.Identity()

        # 2) upsample 후 skip과 concat → (in_ch + skip_ch, H, W)
        #    첫 ResBlock은 (in_ch + skip_ch) → out_ch
        res_blocks = []
        res_blocks.append(ResBlock(in_ch + skip_ch, out_ch, time_dim, cond_dim))
        # 나머지 ResBlock들은 out_ch → out_ch
        for _ in range(n_res - 1):
            res_blocks.append(ResBlock(out_ch, out_ch, time_dim, cond_dim))

        self.res_blocks = nn.ModuleList(res_blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                cond_vec: torch.Tensor, t_emb: torch.Tensor):
        # x: (B, in_ch,   H_small, W_small)
        # skip: (B, skip_ch, H_big, W_big)

        # 1) 먼저 upsample (H_small -> H_big)
        x = self.up(x)  # (B, in_ch, H_big, W_big)

        # 2) 이제 skip과 해상도가 맞으므로 concat 가능
        x = torch.cat([x, skip], dim=1)  # (B, in_ch+skip_ch, H_big, W_big)

        # 3) ResBlock들 통과
        for res in self.res_blocks:
            x = res(x, cond_vec, t_emb)

        return x


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)

        attn = torch.softmax(torch.einsum('bnct,bncs->bnts', q, k) / (head_dim ** 0.5), dim=-1)
        out = torch.einsum('bnts,bncs->bnct', attn, v)
        out = out.contiguous().view(b, c, h, w)
        out = self.proj(out)
        return x + out

# ---------------------------------------------------------
# 5. UNet (128x128용)
# ---------------------------------------------------------
class UNet128(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            base_channels: int = 64,
            time_dim: int = 256,  # ★ time embedding dim
            cond_dim: int = 256,  # ★ cond embedding dim (EEG+class)
            ch_mult=(1, 2, 4, 8),  # ★ 해상도별 채널 multiplier
            emb_dim: int = None,  # = time_dim + cond_dim
            cond_scale: float = 2.0,
            n_res_blocks: int = 2,
    ):
        super().__init__()

        if emb_dim is None:
            emb_dim = time_dim + cond_dim  # t_emb(256) + cond_emb(256) = 512

        # ----- 채널 설정 (128→64→32→16) -----
        assert len(ch_mult) == 4, "현재 UNet128은 4단계 down/up (128→64→32→16)을 가정합니다."

        c0 = base_channels * ch_mult[0]  # 128×128
        c1 = base_channels * ch_mult[1]  #  64×64
        c2 = base_channels * ch_mult[2]  #  32×32
        c3 = base_channels * ch_mult[3]  #  16×16

        # ----- 입력 + 첫 ResBlock (FiLM: t_emb+cond_emb 사용) -----
        self.inc = nn.Conv2d(in_channels, c0, kernel_size=3, padding=1)
        self.res1 = FiLMResBlock(c0, c0, emb_dim, cond_scale)

        # ----- Down path -----
        # 여기서 time_dim/cond_dim을 ResBlock에 넘긴다 (절대 emb_dim/cond_scale 아님!)
        self.down1 = DownBlock(c0, c1, time_dim, cond_dim, n_res=n_res_blocks)  # 128→64
        self.down2 = DownBlock(c1, c2, time_dim, cond_dim, n_res=n_res_blocks)  # 64→32
        self.attn2 = SelfAttention2d(c2, num_heads=4)
        self.down3 = DownBlock(c2, c3, time_dim, cond_dim, n_res=n_res_blocks)  # 32→16

        # ----- Mid (여기서는 FiLMResBlock 그대로 사용) -----
        self.mid1 = FiLMResBlock(c3, c3, emb_dim, cond_scale)
        self.mid2 = FiLMResBlock(c3, c3, emb_dim, cond_scale)
        self.attn3 = SelfAttention2d(c3, num_heads=8)

        # ----- Up path -----
        self.up3 = UpBlock(c3, c3, c2, time_dim, cond_dim, n_res=n_res_blocks)  # 16→32
        self.up2 = UpBlock(c2, c2, c1, time_dim, cond_dim, n_res=n_res_blocks)  # 32→64
        self.up1 = UpBlock(c1, c1, c0, time_dim, cond_dim, n_res=n_res_blocks)  # 64→128

        self.outc = nn.Conv2d(c0, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """
        x:       (B,3,128,128)
        t_emb:   (B,time_dim)  = (B,256)
        cond_emb:(B,cond_dim) = (B,256)
        """
        # FiLMResBlock용으로 time+cond를 합친 embedding
        emb = torch.cat([t_emb, cond_emb], dim=1)  # (B, emb_dim=512)

        # ----- Down -----
        x1 = self.inc(x)
        x1 = self.res1(x1, emb)  # (B,c0,128,128)

        x2, s1 = self.down1(x1, cond_emb, t_emb)  # x2:(B,c1,64,64),  s1:(B,c1,128,128)
        x3, s2 = self.down2(x2, cond_emb, t_emb)  # x3:(B,c2,32,32),  s2:(B,c2,64,64)
        x3 = self.attn2(x3)
        x4, s3 = self.down3(x3, cond_emb, t_emb)  # x4:(B,c3,16,16),  s3:(B,c3,32,32)

        # ----- Mid -----
        h = self.mid1(x4, emb)
        h = self.mid2(h, emb)
        h = self.attn3(h)

        # ----- Up -----
        h = self.up3(h, s3, cond_emb, t_emb)  # (B,c2,32,32)
        h = self.up2(h, s2, cond_emb, t_emb)  # (B,c1,64,64)
        h = self.up1(h, s1, cond_emb, t_emb)  # (B,c0,128,128)

        out = self.outc(h)  # (B,3,128,128)
        return out


# ---------------------------------------------------------
# 6. Diffusion wrapper (EEG + class conditioning)
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
            lambda_rec: float = 0.02,  # ← 추가
    ):
        super().__init__()
        assert img_size == 128, "이 모델은 128x128 이미지를 기준으로 설계되었습니다."

        self.img_size = img_size
        self.img_channels = img_channels
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        self.lambda_rec = lambda_rec

        # ---------- diffusion schedule ----------
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

        # ---------- modules ----------
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

        emb_dim = time_dim + cond_dim  # 256 + 256 = 512

        self.unet = UNet128(
            in_channels=img_channels,
            base_channels=base_channels,
            time_dim=time_dim,  # ★ 여기 중요
            cond_dim=cond_dim,  # ★ 여기 중요
            ch_mult=ch_mult,
            emb_dim=emb_dim,
            cond_scale=cond_scale,
            n_res_blocks=n_res_blocks,
        )

    # helper
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int]) -> torch.Tensor:
        out = a.gather(-1, t)
        return out.view(-1, 1, 1, 1).expand(x_shape)

    def get_cond_emb(self, eeg: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        eeg: (B,C,T)
        labels: (B,) in {1..9} or {0..8}
        """
        y = labels.long()
        if y.min() >= 1:
            y = y - 1
        y = torch.clamp(y, 0, self.num_classes - 1)

        eeg_emb = self.eeg_encoder(eeg)  # (B, cond_dim)
        cls_emb = self.class_emb(y)  # (B, cond_dim)

        cond = eeg_emb + cls_emb
        cond = self.cond_proj(cond)
        return cond

    # diffusion core
    def get_cond_emb_eeg_only(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        eeg-only conditioning (no class label)
        eeg: (B,C,T)
        """
        eeg_emb = self.eeg_encoder(eeg)  # (B, cond_dim)
        cond = self.cond_proj(eeg_emb)
        return cond

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha_bar * x_start + sqrt_one_minus * noise

    def p_losses(
            self,
            x_start: torch.Tensor,  # (B, 3, H, W)  in [-1, 1]
            eeg: torch.Tensor,  # (B, C, T)
            labels: torch.Tensor,  # (B,)
            t: torch.Tensor,  # (B,) long
    ) -> torch.Tensor:
        """
        DDPM training loss with
        - t-dependent weight on noise loss
        - x0 reconstruction term with clamp [-1, 1]
        """

        # ----------------------------
        # 1) forward diffusion: q(x_t | x_0)
        # ----------------------------
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)  # x_t

        # ----------------------------
        # 2) embeddings
        # ----------------------------
        t_emb = self.time_embed(t)  # (B, time_dim)
        cond_emb = self.get_cond_emb_eeg_only(eeg)  # (B, cond_dim)

        # 약한 "guidance" 효과를 위해 조건 임베딩 스케일 조정
        cond_emb = cond_emb * 1.5

        # UNet 예측: epsilon_theta(x_t, t, cond)
        eps_pred = self.unet(x_noisy, t_emb, cond_emb)

        # ----------------------------
        # 3) t-dependent weight for noise loss
        #    w_eps(t) ~ sqrt(1 - alpha_bar_t)
        #    (큰 t에서는 더 강하게, 작은 t에서는 살짝 줄여서 high-freq 보존)
        # ----------------------------
        with torch.no_grad():
            alpha_bar_t = self._extract(self.alphas_cumprod, t, eps_pred.shape)
            w_eps = torch.sqrt(1.0 - alpha_bar_t)  # (B,1,1,1)

        mse_eps = (eps_pred - noise) ** 2
        loss_noise = (w_eps * mse_eps).mean()

        # ----------------------------
        # 4) x0 reconstruction term with clamp in [-1, 1]
        #    x0_pred = (x_t - sqrt(1-a_bar)*eps) / sqrt(a_bar)
        # ----------------------------
        sqrt_alpha_bar = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        x0_pred = (x_noisy - sqrt_one_minus * eps_pred) / (sqrt_alpha_bar + 1e-8)

        # 이미지 범위를 강제로 [-1, 1]로 클램프 (톤 튀는 것 완화)
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        # 작은 t(후반부)에 더 비중을 주기 위해 alpha_bar_t로 가중
        with torch.no_grad():
            w_x0 = alpha_bar_t  # (B,1,1,1)

        loss_recon = (w_x0 * (x0_pred - x_start).abs()).mean()

        # 두 손실을 합침 (가중치는 필요에 따라 조정 가능)
        loss = loss_noise + self.lambda_rec * loss_recon
        return loss

    @torch.no_grad()
    def sample(
            self,
            eeg: torch.Tensor,
            labels: torch.Tensor,
            num_steps: int = None,
            guidance_scale: float = 2.5,  # CLI에서 쓰던 값 유지
    ) -> torch.Tensor:
        """
        DDPM sampling with simple "guidance_scale" (조건 임베딩 스케일),
        최종 결과는 [-1, 1]로 클램프.
        """
        device = eeg.device
        b = eeg.size(0)
        T = self.num_timesteps if num_steps is None else min(self.num_timesteps, num_steps)

        # 시작은 pure Gaussian noise
        x_t = torch.randn(b, self.img_channels, self.img_size, self.img_size, device=device)

        # 조건 임베딩 한 번만 계산
        cond = self.get_cond_emb(eeg, labels)
        cond = cond * guidance_scale  # 샤프니스 조절용 스케일

        for i in reversed(range(T)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            t_emb = self.time_embed(t)

            # eps_theta(x_t, t, cond)
            eps_theta = self.unet(x_t, t_emb, cond)

            beta_t = self.betas[i]
            alpha_t = self.alphas[i]
            alpha_bar_t = self.alphas_cumprod[i]

            sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bar_t)
            sqrt_recip_alpha = torch.sqrt(1.0 / alpha_t)

            # DDPM mean
            mean = sqrt_recip_alpha * (x_t - beta_t / sqrt_one_minus_ab * eps_theta)

            if i > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t) * noise
            else:
                x_t = mean

        # 최종 x0를 [-1,1]로 제한 (저장할 때는 0~255로 다시 스케일)
        x_t = x_t.clamp(-1.0, 1.0)
        return x_t
"""
    def p_losses(
            self,
            x_start: torch.Tensor,  # (B,3,128,128), [-1,1]
            eeg: torch.Tensor,
            labels: torch.Tensor,
            t: torch.Tensor,
            lambda_rec: float = 0.02,  # ★ 재구성 loss 비율
    ) -> torch.Tensor:

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        t_emb = self.time_embed(t)
        cond_emb = self.get_cond_emb(eeg, labels)

        eps_pred = self.unet(x_noisy, t_emb, cond_emb)

        # (1) 기존 noise 예측 loss
        loss_noise = F.mse_loss(eps_pred, noise)

        # (2) x0 재구성: x0_hat = (x_t - sqrt(1-ab)*eps) / sqrt(ab)
        alpha_bar_t = self._extract(self.alphas_cumprod, t, x_start.shape)
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bar_t)

        x0_pred = (x_noisy - sqrt_one_minus_ab * eps_pred) / sqrt_alpha_bar

        # high-freq를 살리려면 L1가 보통 더 낫습니다.
        loss_rec = F.l1_loss(x0_pred, x_start)

        return loss_noise + lambda_rec * loss_rec
    @torch.no_grad()
    def sample_eeg_only(
            self,
            eeg: torch.Tensor,
            num_steps: int = None,
            guidance_scale: float = 2.5,
    ) -> torch.Tensor:
        """
      #   DDPM sampling using EEG-only conditioning (no class label).






