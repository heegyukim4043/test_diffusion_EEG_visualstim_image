"""
model_128_eegonly_transformer_repa.py
─────────────────────────────────────────────────────────────────────────────
EEG-only Transformer diffusion model
+ Perceptual Loss (ResNet18 ImageNet feature cosine similarity)

conditioning
────────────
  cond = EEGEncoder(eeg) * 1.5   ← EEG 신호만 사용 (class label 미사용)
  추론 시에도 EEG만으로 이미지 생성 → 실제 BCI 시나리오에 맞는 구조

※ 이전 이름 'REPA loss' 는 사실 Perceptual Loss 임:
   loss_percept = 1 - cos( ResNet18(x̂₀), ResNet18(x₀) )
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from model_128_eegonly_transformer import EEGDiffusionModel128 as BaseModel, _ssim

# LPIPS (optional — graceful fallback if not installed)
try:
    import lpips as _lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


class EEGDiffusionModel128(BaseModel):
    """
    EEG Transformer diffusion model + Perceptual Loss + Class Conditioning.
    """

    def __init__(
        self,
        *args,
        lambda_percept: float = 0.05,
        percept_feat_dim: int = 512,
        lambda_lpips: float = 0.0,   # LPIPS training loss weight (0 = disabled)
        # 하위 호환: 이전 코드에서 lambda_repa 로 전달하는 경우 처리
        lambda_repa: float = None,
        repa_feat_dim: int = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # 이전 인자명 호환
        if lambda_repa is not None:
            lambda_percept = lambda_repa
        if repa_feat_dim is not None:
            percept_feat_dim = repa_feat_dim

        self.lambda_percept = lambda_percept
        self.lambda_lpips = lambda_lpips

        try:
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            backbone = resnet18(weights=None)
        self.percept_backbone = nn.Sequential(*list(backbone.children())[:-1])
        for p in self.percept_backbone.parameters():
            p.requires_grad = False
        self.percept_backbone.eval()

        self.percept_proj = nn.Sequential(
            nn.Linear(512, percept_feat_dim),
            nn.SiLU(),
            nn.Linear(percept_feat_dim, percept_feat_dim),
        )

        self.register_buffer(
            "percept_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "percept_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

        # LPIPS network (frozen, optional)
        self.lpips_fn = None
        if lambda_lpips > 0.0:
            if _LPIPS_AVAILABLE:
                self.lpips_fn = _lpips_lib.LPIPS(net="vgg")
                for p in self.lpips_fn.parameters():
                    p.requires_grad = False
                self.lpips_fn.eval()
            else:
                import warnings
                warnings.warn(
                    "lambda_lpips > 0 but 'lpips' package is not installed. "
                    "LPIPS loss will be skipped. Install with: pip install lpips",
                    RuntimeWarning,
                )

    def train(self, mode: bool = True):
        """model.train() 호출 시 percept_backbone과 lpips_fn은 항상 eval 유지."""
        super().train(mode)
        self.percept_backbone.eval()
        if self.lpips_fn is not None:
            self.lpips_fn.eval()
        return self

    # ── Perceptual feature 추출 ───────────────────────────────────────────────
    def _extract_percept_feat(self, x: torch.Tensor) -> torch.Tensor:
        """x: [-1,1] → ImageNet 정규화 → ResNet18 → L2 normalized feature"""
        x = (x.clamp(-1, 1) + 1.0) * 0.5
        x = (x - self.percept_mean) / (self.percept_std + 1e-8)
        f = self.percept_backbone(x).flatten(1)
        f = self.percept_proj(f)
        return F.normalize(f, dim=1)

    # ── EEG-only conditioning 벡터 ───────────────────────────────────────────
    def _get_cond(
        self,
        eeg: torch.Tensor,
        labels: torch.Tensor = None,  # 하위 호환용 — 실제로는 사용하지 않음
    ) -> torch.Tensor:
        """EEG 신호만으로 conditioning 벡터 생성. labels는 무시."""
        return self.get_cond_emb_eeg_only(eeg) * 1.5

    # ── 학습 손실 ─────────────────────────────────────────────────────────────
    def p_losses(
        self,
        x_start: torch.Tensor,
        eeg: torch.Tensor,
        labels: torch.Tensor = None,  # 하위 호환용 — 사용하지 않음
        t: torch.Tensor = None,
        return_x0: bool = False,
    ):
        if t is None:
            raise ValueError("t must be provided")

        noise   = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        t_emb    = self.time_embed(t)
        cond_emb = self._get_cond(eeg)               # EEG만 사용

        eps_pred = self.unet(x_noisy, t_emb, cond_emb)

        with torch.no_grad():
            alpha_bar_t = self._extract(self.alphas_cumprod, t, eps_pred.shape)
            w_eps       = torch.sqrt(1.0 - alpha_bar_t)

        loss_noise = (w_eps * (eps_pred - noise) ** 2).mean()

        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        x0_pred = (x_noisy - sqrt_one_minus * eps_pred) / (sqrt_alpha_bar + 1e-8)
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        with torch.no_grad():
            w_x0 = alpha_bar_t

        loss_recon   = (w_x0 * (x0_pred - x_start).abs()).mean()
        loss_ssim    = 1.0 - _ssim(x0_pred, x_start)

        # Perceptual loss (ResNet18 feature cosine similarity)
        feat_pred = self._extract_percept_feat(x0_pred)
        with torch.no_grad():
            feat_tgt = self._extract_percept_feat(x_start)
        loss_percept = 1.0 - (feat_pred * feat_tgt).sum(dim=1).mean()

        # LPIPS loss (VGG perceptual distance, optional)
        loss_lpips = torch.tensor(0.0, device=x_start.device)
        if self.lambda_lpips > 0.0 and self.lpips_fn is not None:
            lp = self.lpips_fn(
                x0_pred.clamp(-1, 1),
                x_start.clamp(-1, 1),
            )
            loss_lpips = lp.mean()

        total = (
            loss_noise
            + self.lambda_rec     * loss_recon
            + self.lambda_ssim    * loss_ssim
            + self.lambda_percept * loss_percept
            + self.lambda_lpips   * loss_lpips
        )
        if return_x0:
            return total, x0_pred.detach()
        return total

    # ── DDPM 샘플링 (EEG-only) ───────────────────────────────────────────────
    @torch.no_grad()
    def sample(
        self,
        eeg: torch.Tensor,
        labels: torch.Tensor = None,  # 하위 호환용 — 무시
        num_steps: int = None,
        guidance_scale: float = 1.5,
    ) -> torch.Tensor:
        """DDPM 샘플링. EEG 신호만으로 이미지 생성."""
        device = eeg.device
        b      = eeg.size(0)
        # DDPM은 full schedule이어야 올바름 — num_steps 무시하고 항상 full
        T      = self.num_timesteps

        x_t  = torch.randn(b, self.img_channels, self.img_size, self.img_size, device=device)
        cond = self._get_cond(eeg, labels) * (guidance_scale / 1.5)

        for i in reversed(range(T)):
            t     = torch.full((b,), i, device=device, dtype=torch.long)
            t_emb = self.time_embed(t)
            eps_theta = self.unet(x_t, t_emb, cond)

            beta_t            = self.betas[i]
            alpha_t           = self.alphas[i]
            alpha_bar_t       = self.alphas_cumprod[i]
            sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bar_t)
            sqrt_recip_alpha  = torch.sqrt(1.0 / alpha_t)

            mean = sqrt_recip_alpha * (x_t - beta_t / sqrt_one_minus_ab * eps_theta)

            if i > 0:
                x_t = mean + torch.sqrt(beta_t) * torch.randn_like(x_t)
            else:
                x_t = mean

        return x_t.clamp(-1.0, 1.0)

    # ── DDIM 샘플링 (EEG-only) ───────────────────────────────────────────────
    @torch.no_grad()
    def sample_ddim(
        self,
        eeg: torch.Tensor,
        labels: torch.Tensor = None,
        num_steps: int = None,
        guidance_scale: float = 1.5,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM 샘플링. num_steps < num_timesteps 이면 균등 간격으로 타임스텝 선택.
        guidance_scale=1.5 : 학습 시 conditioning 강도와 동일 (권장).
        더 선명한 이미지를 원하면 2.0~3.0 으로 높여볼 수 있음.
        """
        device = eeg.device
        b      = eeg.size(0)
        T_full = self.num_timesteps
        n_steps = T_full if num_steps is None else min(T_full, num_steps)

        # 균등 간격 타임스텝 선택 (0 ~ T_full-1)
        step_seq      = np.linspace(0, T_full - 1, n_steps, dtype=int).tolist()
        step_seq_prev = [-1] + step_seq[:-1]   # 이전 타임스텝 (-1 → alpha_bar=1)

        x_t  = torch.randn(b, self.img_channels, self.img_size, self.img_size, device=device)
        cond = self._get_cond(eeg, labels) * (guidance_scale / 1.5)

        for i, prev_i in zip(reversed(step_seq), reversed(step_seq_prev)):
            t     = torch.full((b,), i, device=device, dtype=torch.long)
            t_emb = self.time_embed(t)
            eps_theta = self.unet(x_t, t_emb, cond)

            alpha_bar_t    = self.alphas_cumprod[i]
            alpha_bar_prev = (self.alphas_cumprod[prev_i]
                              if prev_i >= 0
                              else torch.tensor(1.0, device=device))

            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus   = torch.sqrt(1.0 - alpha_bar_t)

            x0_pred = (x_t - sqrt_one_minus * eps_theta) / (sqrt_alpha_bar_t + 1e-8)
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            if prev_i >= 0 and eta > 0.0:
                sigma = (
                    eta
                    * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
                    * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
                )
                noise = torch.randn_like(x_t)
            else:
                sigma = 0.0
                noise = torch.zeros_like(x_t)

            dir_xt = torch.sqrt((1.0 - alpha_bar_prev - sigma ** 2).clamp(min=0)) * eps_theta
            x_t    = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

        return x_t.clamp(-1.0, 1.0)
