"""
model_eeg_transformer_cls_only.py
─────────────────────────────────────────────────────────────────────────────
EEG Transformer → Classification 전용 모델 (UNet/Diffusion 없음).

아키텍처 (V2)
─────────────
  EEG (B, ch, T)
      │
      ▼
  MultiScaleStem
    ├── Conv1d(ch→64, k=7)   ┐
    ├── Conv1d(ch→64, k=15)  ├─ 병렬 → Concat(192) → Linear(hidden) → GN+SiLU
    └── Conv1d(ch→64, k=31)  ┘
    └── Conv1d(hidden, hidden, k=5, s=2)  (시간 다운샘플)
      │
      ▼
  CLS Token + Positional Embedding
      │
      ▼
  TransformerBlock × n_layers  (StochasticDepth 포함)
    ├── PreNorm MultiheadAttention
    └── PreNorm MLP (hidden × mlp_ratio)
      │
      ▼
  CLS Token → (B, hidden)
      │
      ▼
  ClassificationHead
    ├── LayerNorm(hidden)
    ├── Linear(hidden, hidden//2)
    ├── GELU
    ├── Dropout
    └── Linear(hidden//2, num_classes)
      │
      ▼
  logits (B, num_classes)

사용 예시
─────────
  model = EEGTransformerClassifier(
      eeg_channels=32,
      num_classes=3,
      eeg_hidden_dim=256,
      n_heads=8,
      n_layers=4,
      stochastic_depth=0.1,
  )
  logits          = model(eeg)
  loss, acc       = model.compute_loss(eeg, labels)
  loss, acc       = model.compute_loss_mixup(eeg, labels_a, labels_b, lam)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Stochastic Depth (DropPath)
# ─────────────────────────────────────────────────────────────────────────────

class StochasticDepth(nn.Module):
    """학습 시 레이어 전체를 확률적으로 skip (잔차 연결만 통과)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.rand(shape, dtype=x.dtype, device=x.device).floor_().add_(keep).clamp_(max=1.0)
        return x * mask / keep


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Scale Stem
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleStem(nn.Module):
    """
    단기·중기·장기 시간 패턴을 동시에 포착하는 병렬 Conv1d stem.

    Input : (B, ch, T)
    Output: (B, hidden, T//2)
    """

    def __init__(self, in_channels: int, hidden: int):
        super().__init__()
        mid = 64
        # 병렬 멀티스케일 컨볼루션
        self.branch_s = nn.Conv1d(in_channels, mid, kernel_size=7,  padding=3)
        self.branch_m = nn.Conv1d(in_channels, mid, kernel_size=15, padding=7)
        self.branch_l = nn.Conv1d(in_channels, mid, kernel_size=31, padding=15)

        # 채널 합성 + 정규화
        self.proj = nn.Sequential(
            nn.Conv1d(mid * 3, hidden, kernel_size=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
        )
        # 시간 다운샘플 (×½)
        self.down = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.branch_s(x), self.branch_m(x), self.branch_l(x)], dim=1)
        feat = self.proj(feat)
        return self.down(feat)   # (B, hidden, T//2)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block (PreNorm + StochasticDepth)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 dropout: float = 0.1, drop_path: float = 0.0,
                 mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.norm2  = nn.LayerNorm(d_model)
        dim_ff = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# EEG Encoder V2
# ─────────────────────────────────────────────────────────────────────────────

class EEGEncoderV2(nn.Module):
    """
    Multi-Scale Stem + CLS Token + Transformer (StochasticDepth).

    Input : (B, ch, T)
    Output: (B, out_dim)
    """

    def __init__(
        self,
        eeg_channels: int = 32,
        hidden: int = 256,
        out_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        stochastic_depth: float = 0.1,
        max_len: int = 1024,
    ):
        super().__init__()

        self.stem = MultiScaleStem(eeg_channels, hidden)

        # CLS 토큰
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 위치 임베딩 (최대 max_len+1 : CLS 포함)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len + 1, hidden))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # stochastic depth 비율 선형 증가 (0 → stochastic_depth)
        dp_rates = [stochastic_depth * i / max(n_layers - 1, 1) for i in range(n_layers)]

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden, n_heads, dropout=dropout, drop_path=dp_rates[i])
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden)

        # 출력 MLP
        self.head_proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        # eeg: (B, ch, T)
        x = self.stem(eeg)                        # (B, hidden, T//2)
        x = x.permute(0, 2, 1)                    # (B, T//2, hidden)

        B, L, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)    # (B, 1, hidden)
        x   = torch.cat([cls, x], dim=1)          # (B, L+1, hidden)

        # 위치 임베딩 (길이 맞춤)
        x = x + self.pos_emb[:, : L + 1, :]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]                         # CLS 토큰만 사용
        return self.head_proj(cls_out)            # (B, out_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Classification Head
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# EEGTransformerClassifier
# ─────────────────────────────────────────────────────────────────────────────

class EEGTransformerClassifier(nn.Module):
    """
    EEG Transformer V2 인코더 + 분류 헤드.

    Parameters
    ----------
    eeg_channels     : EEG 채널 수
    num_classes      : 분류 클래스 수
    eeg_hidden_dim   : Transformer 내부 차원 (default 256)
    out_dim          : 인코더 출력 차원 (default 256)
    n_heads          : Attention head 수 (default 8)
    n_layers         : Transformer 레이어 수 (default 4)
    tf_dropout       : Attention/MLP dropout (default 0.1)
    cls_dropout      : 분류 헤드 dropout (default 0.3)
    stochastic_depth : DropPath 최대 비율 (default 0.1)
    """

    def __init__(
        self,
        eeg_channels: int = 32,
        num_classes: int = 3,
        eeg_hidden_dim: int = 256,
        out_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        tf_dropout: float = 0.1,
        cls_dropout: float = 0.3,
        stochastic_depth: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.encoder = EEGEncoderV2(
            eeg_channels=eeg_channels,
            hidden=eeg_hidden_dim,
            out_dim=out_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=tf_dropout,
            stochastic_depth=stochastic_depth,
        )
        self.cls_head = ClassificationHead(out_dim, num_classes, cls_dropout)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.cls_head(self.encoder(eeg))

    def compute_loss(
        self,
        eeg: torch.Tensor,
        labels: torch.Tensor,
        label_smoothing: float = 0.1,
    ):
        """
        Returns
        -------
        loss : torch.Tensor
        acc  : float  (0~1)
        """
        logits = self.forward(eeg)
        loss   = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        with torch.no_grad():
            acc = (logits.argmax(1) == labels).float().mean().item()
        return loss, acc

    def compute_loss_mixup(
        self,
        eeg: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
        label_smoothing: float = 0.1,
    ):
        """
        Mixup 적용 시 사용. eeg는 이미 혼합된 상태.

        Returns
        -------
        loss : torch.Tensor
        acc  : float  (원본 labels_a 기준)
        """
        logits = self.forward(eeg)
        loss = (lam * F.cross_entropy(logits, labels_a, label_smoothing=label_smoothing)
                + (1 - lam) * F.cross_entropy(logits, labels_b, label_smoothing=label_smoothing))
        with torch.no_grad():
            acc = (logits.argmax(1) == labels_a).float().mean().item()
        return loss, acc

    @torch.no_grad()
    def predict(self, eeg: torch.Tensor):
        logits = self.forward(eeg)
        probs  = F.softmax(logits, dim=1)
        return probs.argmax(1), probs
