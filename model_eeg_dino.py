"""
model_eeg_dino.py
─────────────────────────────────────────────────────────────────────────────
Stage 1: EEG → shared 512-dim latent space (DINO-aligned)

구조
────
  EEG (B, ch, T)
    └─ Conv1D + Transformer → 256
  Subject ID
    └─ Embedding(n_subjects, 64)
  Fusion MLP: (256+64=320) → 512  →  L2 norm  =  eeg_latent

  Image
    └─ frozen DINO (ViT-S/14, dim=384)
    └─ Image projector: 384 → 512  →  L2 norm  =  img_latent

Loss
────
  cosine loss : 1 - mean(eeg_latent · img_latent)
  InfoNCE     : optional, in-batch negatives
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_128_eegonly_transformer import EEGEncoderV2   # V2: OccipitalChannelGate + MultiScaleStem


LATENT_DIM = 512
DINO_DIM   = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024}
ENCODER_TYPES = ("transformer", "conv", "v2")


# ── DINO teacher 로드 ────────────────────────────────────────────────────
def load_dino_encoder(model_name: str = "dinov2_vits14", device=None):
    """frozen DINOv2 — 학습 중 업데이트 없음."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
    dino.eval().to(device)
    for p in dino.parameters():
        p.requires_grad = False
    return dino


# ── EEG Encoder ──────────────────────────────────────────────────────────
class EEGEncoderTransformer(nn.Module):
    def __init__(
        self,
        eeg_channels: int = 32,
        hidden_dim:   int = 256,
        out_dim:      int = 256,
        n_heads:      int = 4,
        n_layers:     int = 4,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(eeg_channels, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout, batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    @staticmethod
    def _sinusoidal(seq_len, dim, device):
        pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
        i   = torch.arange(0, dim, 2, device=device).float()
        div = torch.exp(-i * (math.log(10000.0) / dim))
        emb = torch.zeros(seq_len, dim, device=device)
        emb[:, 0::2] = torch.sin(pos * div)
        emb[:, 1::2] = torch.cos(pos * div[:dim // 2])
        return emb.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # per-trial z-score
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)
        h = self.conv(x).transpose(1, 2)
        h = h + self._sinusoidal(h.size(1), h.size(2), h.device)
        h = self.transformer(h).mean(1)
        return self.fc(h)                   # (B, out_dim)


# ── EEGNet-inspired Conv Encoder ─────────────────────────────────────────
class EEGEncoderConv(nn.Module):
    """
    Multi-scale temporal conv + depthwise separable encoder.

    Inspired by EEGNet (Lawhern et al. 2018) but adapted for 1D input (B, C, T).

    Architecture:
      1. Multi-scale temporal conv  (3 scales: ~15ms / 62ms / 250ms @ 1024Hz)
         → concat along filter dim
      2. Depthwise separable conv   (channel mixing within each scale group)
      3. AdaptiveAvgPool + MLP      → out_dim

    Input : (B, eeg_channels, T)    e.g. (B, 32, 2048)
    Output: (B, out_dim)
    """

    def __init__(
        self,
        eeg_channels: int   = 32,
        out_dim:      int   = 256,
        n_filters:    int   = 32,    # filters per temporal scale
        dropout:      float = 0.25,
    ):
        super().__init__()
        # temporal kernel sizes (samples @ 1024 Hz)
        scales = [16, 64, 256]       # ~15ms, ~62ms, ~250ms

        self.multiscale = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(eeg_channels, n_filters, kernel_size=k,
                          padding=k // 2, bias=False),
                nn.BatchNorm1d(n_filters),
                nn.ELU(),
                nn.AvgPool1d(8),     # 2048 → 256
                nn.Dropout(dropout),
            )
            for k in scales
        ])

        total = n_filters * len(scales)   # 96

        # depthwise + pointwise (separable)
        self.separable = nn.Sequential(
            nn.Conv1d(total, total, kernel_size=16,
                      padding=8, groups=total, bias=False),   # depthwise
            nn.Conv1d(total, total, kernel_size=1, bias=False),  # pointwise
            nn.BatchNorm1d(total),
            nn.ELU(),
            nn.AvgPool1d(8),         # 256 → 32
            nn.Dropout(dropout),
        )

        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(total, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # per-trial z-score (same as transformer encoder)
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)
        feats = [branch(x) for branch in self.multiscale]
        h = torch.cat(feats, dim=1)      # (B, 96, ~256)
        h = self.separable(h)            # (B, 96, ~32)
        return self.proj(h)              # (B, out_dim)


# ── 메인 모델 ────────────────────────────────────────────────────────────
class EEGDINORegressor(nn.Module):
    """
    EEG + subject_id → 512-dim shared latent (L2 normalized)
    Image → DINO → 512-dim shared latent (L2 normalized)

    두 latent 간 cosine similarity로 alignment 학습.
    """

    def __init__(
        self,
        eeg_channels:  int   = 32,
        n_subjects:    int   = 20,
        dino_feat_dim: int   = 384,        # DINO 출력 dim (ViT-S=384, ViT-B=768)
        latent_dim:    int   = LATENT_DIM, # 공유 latent 차원
        eeg_hidden:    int   = 256,
        eeg_out:       int   = 256,
        subj_emb_dim:  int   = 64,
        n_heads:       int   = 4,
        n_layers:      int   = 4,
        dropout:       float = 0.1,
        temperature:   float = 0.1,
        encoder_type:  str   = "transformer",  # "transformer" | "conv" | "v2"
        n_classes:     int   = 9,              # aux classification head
        eeg_occipital_indices=None,            # V2 only: None=BioSemi32 prior, []=no prior
    ):
        super().__init__()
        assert encoder_type in ENCODER_TYPES, f"encoder_type must be one of {ENCODER_TYPES}"
        self.latent_dim   = latent_dim
        self.encoder_type = encoder_type
        self.log_temp = nn.Parameter(torch.tensor(math.log(temperature)))

        # ── EEG branch ───────────────────────────────────────────────────
        if encoder_type == "transformer":
            self.eeg_encoder = EEGEncoderTransformer(
                eeg_channels=eeg_channels, hidden_dim=eeg_hidden,
                out_dim=eeg_out, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
            )
        elif encoder_type == "v2":
            self.eeg_encoder = EEGEncoderV2(
                eeg_channels=eeg_channels, eeg_hidden_dim=eeg_hidden,
                out_dim=eeg_out, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
                occipital_indices=eeg_occipital_indices,
            )
        else:  # conv
            self.eeg_encoder = EEGEncoderConv(
                eeg_channels=eeg_channels, out_dim=eeg_out, dropout=dropout,
            )

        self.subject_emb = nn.Embedding(n_subjects, subj_emb_dim)
        nn.init.normal_(self.subject_emb.weight, std=0.02)

        # Fusion MLP: 320 → 512
        fusion_in = eeg_out + subj_emb_dim
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, fusion_in * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_in * 2, latent_dim),
        )

        # ── Image branch (frozen DINO → projector) ───────────────────────
        self.img_projector = nn.Sequential(
            nn.LayerNorm(dino_feat_dim),
            nn.Linear(dino_feat_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # ── Auxiliary classification head ─────────────────────────────────
        # EEG latent → class logits (병렬 CE 손실로 class discrimination 강화)
        self.aux_cls_head = nn.Linear(latent_dim, n_classes)

    # ── Encoding ─────────────────────────────────────────────────────────
    def encode_eeg(self, eeg: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """EEG + subject_id → L2-norm 512-dim latent."""
        eeg_feat  = self.eeg_encoder(eeg)
        subj_feat = self.subject_emb(subject_ids)
        fused     = torch.cat([eeg_feat, subj_feat], dim=1)
        return F.normalize(self.fusion_mlp(fused), dim=1)

    def encode_img(self, dino_feat: torch.Tensor) -> torch.Tensor:
        """DINO feature → L2-norm 512-dim latent."""
        return F.normalize(self.img_projector(dino_feat), dim=1)

    # ── Loss ─────────────────────────────────────────────────────────────
    def compute_loss(
        self,
        eeg:            torch.Tensor,   # (B, ch, T)
        subject_ids:    torch.Tensor,   # (B,)
        target_dino:    torch.Tensor,   # (B, dino_feat_dim)
        proto_dino:     torch.Tensor,   # (n_cls, dino_feat_dim)
        class_labels:   torch.Tensor,   # (B,) 0-based
        use_infonce:    bool  = False,
        w_cos:          float = 1.0,
        w_proto:        float = 1.0,
        w_infonce:      float = 1.0,
        w_aux:          float = 0.5,    # auxiliary classification head weight
    ):
        eeg_lat   = self.encode_eeg(eeg, subject_ids)    # (B, 512)
        img_lat   = self.encode_img(target_dino)         # (B, 512)
        proto_lat = self.encode_img(proto_dino)          # (n_cls, 512)
        temp      = self.log_temp.exp().clamp(0.01, 1.0)

        # ── 1. Cosine alignment (trial-level) ────────────────────────────
        loss_cos = (1.0 - (eeg_lat * img_lat).sum(dim=1)).mean()

        # ── 2. Prototypical cross-entropy ────────────────────────────────
        sim        = eeg_lat @ proto_lat.T / temp        # (B, n_cls)
        loss_proto = F.cross_entropy(sim, class_labels)
        acc        = (sim.argmax(1) == class_labels).float().mean()

        # ── 3. Supervised InfoNCE ─────────────────────────────────────────
        loss_infonce = torch.tensor(0.0, device=eeg.device)
        if use_infonce:
            B = eeg_lat.size(0)
            sim_ei = eeg_lat @ img_lat.T / temp          # (B, B)

            pos_mask = class_labels.unsqueeze(1).eq(class_labels.unsqueeze(0))
            neg_inf  = torch.zeros_like(sim_ei)
            neg_inf[pos_mask] = -1e9
            eye = torch.eye(B, device=eeg.device).bool()
            neg_inf[eye] = -1e9

            denom_ei = torch.logsumexp(sim_ei + neg_inf, dim=1)
            denom_ie = torch.logsumexp(sim_ei.T + neg_inf.T, dim=1)
            loss_ei  = -(sim_ei.diagonal() - denom_ei).mean()
            loss_ie  = -(sim_ei.diagonal() - denom_ie).mean()
            loss_infonce = (loss_ei + loss_ie) * 0.5

        # ── 4. Auxiliary classification (EEG latent → class) ─────────────
        # 직접 latent에서 class를 예측 → class discrimination 명시적 강화
        aux_logits   = self.aux_cls_head(eeg_lat)        # (B, n_classes)
        loss_aux_cls = F.cross_entropy(aux_logits, class_labels)

        loss = (w_cos     * loss_cos
              + w_proto   * loss_proto
              + w_infonce * loss_infonce
              + w_aux     * loss_aux_cls)
        return loss, loss_cos, loss_proto, loss_infonce, acc

    # ── Supervised Contrastive Loss (Khosla et al. NeurIPS 2020) ─────────
    @staticmethod
    def supcon_loss(
        z1:          torch.Tensor,   # (B, D) L2-normed EEG latents
        z2:          torch.Tensor,   # (B, D) L2-normed image latents
        labels:      torch.Tensor,   # (B,) class indices (0-based)
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Multi-view SupCon: both EEG (z1) and image (z2) serve as anchors.
        Positives = same-class pairs (cross- and within-modal), excluding self.
        Negatives = different-class pairs.

        Compared to InfoNCE with false-negative masking:
          InfoNCE: excludes same-class from denominator.
          SupCon : explicitly uses same-class pairs in numerator (stronger signal).
        """
        B      = z1.size(0)
        device = z1.device

        z_all = torch.cat([z1, z2], dim=0)               # (2B, D)
        y_all = torch.cat([labels, labels], dim=0)        # (2B,)

        sim = z_all @ z_all.T / temperature               # (2B, 2B)

        eye      = torch.eye(2 * B, device=device, dtype=torch.bool)
        pos_mask = y_all.unsqueeze(1).eq(y_all.unsqueeze(0)) & ~eye  # (2B, 2B)

        n_pos   = pos_mask.float().sum(dim=1)              # (2B,)
        has_pos = n_pos > 0
        if not has_pos.any():
            return torch.tensor(0.0, device=device)

        sim_noself = sim.masked_fill(eye, float('-inf'))
        log_denom  = torch.logsumexp(sim_noself, dim=1, keepdim=True)  # (2B, 1)
        log_prob   = sim - log_denom                       # (2B, 2B)

        loss_per = -(pos_mask.float() * log_prob).sum(dim=1) / n_pos.clamp(min=1)
        return loss_per[has_pos].mean()

    # ── Retrieval ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        eeg:          torch.Tensor,
        subject_ids:  torch.Tensor,
        proto_dino:   torch.Tensor,   # (n_cls, dino_feat_dim)
    ):
        """Returns: logits (B, n_cls), pred_cls (B,), eeg_latent (B, 512)"""
        eeg_lat   = self.encode_eeg(eeg, subject_ids)
        proto_lat = self.encode_img(proto_dino)
        temp      = self.log_temp.exp().clamp(0.01, 1.0)
        logits    = eeg_lat @ proto_lat.T / temp
        return logits, logits.argmax(1), eeg_lat
