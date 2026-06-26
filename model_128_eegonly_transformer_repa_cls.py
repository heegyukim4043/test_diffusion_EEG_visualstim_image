"""
model_128_eegonly_transformer_repa_cls.py
─────────────────────────────────────────────────────────────────────────────
EEG → Image 생성 (DDPM/DDIM) + EEG → Class 분류를 동시에 학습.

베이스 모델
───────────
  model_128_eegonly_transformer_repa.EEGDiffusionModel128
    └── model_128_eegonly_transformer.EEGDiffusionModel128
          ├── EEGEncoderTransformer   (EEG → cond_emb 256-d)
          ├── UNet128                 (ε-prediction)
          └── REPA backbone          (ResNet18, frozen)

추가 모듈
─────────
  ClassificationHead  : cond_emb(256) → FC → num_classes
      256 → 128 → SiLU → Dropout → num_classes

총 손실
───────
  L = L_diff(ε-MSE) + λ_rec·L_recon + λ_ssim·L_ssim
      + λ_repa·L_repa + λ_cls·L_cls(CrossEntropy)

공개 메서드
──────────
  p_losses(x_start, eeg, labels, t)    → 총 손실 + 분류 정확도 반환
  classify(eeg)                         → logits (B, num_classes)
  sample / sample_ddim                  → 이미지 생성 (부모 클래스 그대로)

사용 예시
─────────
  model = EEGDiffusionModel128Cls(
      img_size=128, eeg_channels=32, num_classes=3,
      lambda_cls=0.5,
  )
  loss, acc = model.p_losses(img, eeg, labels, t)
  logits    = model.classify(eeg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_128_eegonly_transformer_repa import (
    EEGDiffusionModel128 as BaseREPA,
    _ssim,
)


class ClassificationHead(nn.Module):
    """EEG 컨디셔닝 벡터 → 클래스 로짓."""

    def __init__(self, in_dim: int = 256, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EEGDiffusionModel128Cls(BaseREPA):
    """
    EEG → Image 생성 + EEG → Classification 동시 학습 모델.

    Parameters
    ----------
    lambda_cls   : float  분류 손실 가중치 (default 0.5)
    cls_dropout  : float  분류 헤드 드롭아웃 (default 0.3)
    *args/**kwargs → BaseREPA (EEGDiffusionModel128) 에 전달
    """

    def __init__(
        self,
        *args,
        lambda_cls: float = 0.5,
        cls_dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_cls = lambda_cls

        cond_dim = kwargs.get("cond_dim", 256)
        self.cls_head = ClassificationHead(
            in_dim=cond_dim,
            num_classes=self.num_classes,
            dropout=cls_dropout,
        )

    # ── 분류만 수행 ───────────────────────────────────────────────────────────
    def classify(self, eeg: torch.Tensor) -> torch.Tensor:
        """EEG → 로짓 (B, num_classes)"""
        cond = self.get_cond_emb_eeg_only(eeg)   # (B, cond_dim)
        return self.cls_head(cond)

    # ── 학습 손실 (생성 + 분류) ───────────────────────────────────────────────
    def p_losses(
        self,
        x_start: torch.Tensor,
        eeg: torch.Tensor,
        labels: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Returns
        -------
        total_loss : torch.Tensor  (scalar)
        cls_acc    : float         (미니배치 정확도, 0~1)
        """
        noise   = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        t_emb    = self.time_embed(t)
        cond_emb = self.get_cond_emb_eeg_only(eeg)   # (B, cond_dim)
        cond_emb_scaled = cond_emb * 1.5

        eps_pred = self.unet(x_noisy, t_emb, cond_emb_scaled)

        # ── 확산 손실 ──────────────────────────────────────────────────────
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

        loss_recon = (w_x0 * (x0_pred - x_start).abs()).mean()
        loss_ssim  = 1.0 - _ssim(x0_pred, x_start)

        # ── REPA 손실 ──────────────────────────────────────────────────────
        feat_pred = self._extract_percept_feat(x0_pred)
        with torch.no_grad():
            feat_tgt = self._extract_percept_feat(x_start)
        loss_repa = 1.0 - (feat_pred * feat_tgt).sum(dim=1).mean()

        # ── 분류 손실 ──────────────────────────────────────────────────────
        logits   = self.cls_head(cond_emb)              # (B, num_classes)
        loss_cls = F.cross_entropy(logits, labels)

        with torch.no_grad():
            preds   = logits.argmax(dim=1)
            cls_acc = (preds == labels).float().mean().item()

        # ── 총 손실 ────────────────────────────────────────────────────────
        total_loss = (
            loss_noise
            + self.lambda_rec  * loss_recon
            + self.lambda_ssim * loss_ssim
            + self.lambda_percept * loss_repa
            + self.lambda_cls  * loss_cls
        )

        return total_loss, cls_acc
