"""
train_vs_re_latent_gen.py
────────────────────────────────────────────────────────────────────────────
Exp32: Latent diffusion in SD 1.5 VAE space.

Pipeline:
  1. Frozen SD VAE encodes 128x128 images → 4×16×16 latents
  2. Small DDPM UNet is trained in latent space
  3. SupCon EEG encoder (Exp23-B) initializes EEG conditioning (optional freeze)
  4. At eval: sample latents → decode with VAE → DINO collapse diagnostics

Usage:
  python train_vs_re_latent_gen.py --subject_ids 1,2,18 --epochs 300
  python train_vs_re_latent_gen.py --subject_ids 1,2,18 --supcon_ckpt checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon --freeze_encoder
"""

import argparse
import copy
import csv
import datetime
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_vs_re import VSReDataset, collate_fn, available_subjects, session_counts
from model_eeg_dino import EEGDINORegressor, load_dino_encoder, DINO_DIM, LATENT_DIM
from train_crosssubj_dino import set_seed, EEGAugment, compute_class_prototypes

# ── VAE ───────────────────────────────────────────────────────────────────────
def load_vae(device):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae

VAE_SCALE = 0.18215   # SD standard latent scale factor

# ── Latent UNet ───────────────────────────────────────────────────────────────
class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / (half - 1))
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

class FiLMResBlock(nn.Module):
    def __init__(self, ch, cond_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.film  = nn.Linear(cond_dim, ch * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
    def forward(self, x, cond):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return x + h

class LatentUNet(nn.Module):
    """Small UNet for 4×16×16 latents."""
    def __init__(self, in_ch=4, base_ch=128, ch_mult=(1,2,4), time_dim=256, cond_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbed(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        full_cond = time_dim + cond_dim

        # Encoder
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = base_ch
        enc_chs = [ch]
        for m in ch_mult[:-1]:
            out_ch = base_ch * m
            self.enc.append(FiLMResBlock(ch, full_cond))
            self.downs.append(nn.Conv2d(ch, out_ch, 4, stride=2, padding=1))
            ch = out_ch
            enc_chs.append(ch)

        # Bottleneck
        out_ch = base_ch * ch_mult[-1]
        self.bot_in  = FiLMResBlock(ch, full_cond)
        self.bot_mid = FiLMResBlock(ch, full_cond)
        self.bot_proj = nn.Conv2d(ch, out_ch, 1) if ch != out_ch else nn.Identity()
        ch = out_ch

        # Decoder
        self.ups   = nn.ModuleList()
        self.dec   = nn.ModuleList()
        skip_chs = list(reversed(enc_chs))
        for i, m in enumerate(reversed(ch_mult[:-1])):
            sk = skip_chs[i]
            out_ch = base_ch * m
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch + sk, out_ch, 3, padding=1),
            ))
            self.dec.append(FiLMResBlock(out_ch, full_cond))
            ch = out_ch

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)

    def forward(self, x, t, eeg_cond):
        temb = self.time_mlp(t)
        cond = torch.cat([temb, eeg_cond], dim=-1)

        h = self.in_conv(x)
        skips = [h]
        for blk, down in zip(self.enc, self.downs):
            h = blk(h, cond)
            h = down(h)
            skips.append(h)

        h = self.bot_in(h, cond)
        h = self.bot_mid(h, cond)
        h = self.bot_proj(h)

        for up, blk in zip(self.ups, self.dec):
            sk = skips.pop()
            h = up(torch.cat([h, sk], dim=1))
            h = blk(h, cond)

        return self.out_conv(F.silu(self.out_norm(h)))

# ── Cross-Attention UNet ──────────────────────────────────────────────────────
class CrossAttnBlock(nn.Module):
    """Image spatial tokens attend to EEG condition tokens."""
    def __init__(self, ch, eeg_dim, n_heads=4, n_eeg_tokens=8):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.eeg_proj = nn.Linear(eeg_dim, n_eeg_tokens * ch)
        self.n_tokens = n_eeg_tokens
        self.attn = nn.MultiheadAttention(ch, n_heads, batch_first=True, dropout=0.0)
        self.out_proj = nn.Linear(ch, ch)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, eeg_cond):
        B, C, H, W = x.shape
        xr = x.flatten(2).transpose(1, 2)          # (B, H*W, C)
        xr_n = self.norm(xr)
        kv = self.eeg_proj(eeg_cond).view(B, self.n_tokens, C)  # (B, K, C)
        out, _ = self.attn(xr_n, kv, kv)
        out = self.out_proj(out).transpose(1, 2).view(B, C, H, W)
        return x + out


class CAResBlock(nn.Module):
    """ResBlock with time-FiLM + EEG cross-attention."""
    def __init__(self, ch, time_dim, eeg_dim, n_heads=4, n_eeg_tokens=8, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.time_film = nn.Linear(time_dim, ch * 2)
        self.cross_attn = CrossAttnBlock(ch, eeg_dim, n_heads, n_eeg_tokens)
        nn.init.zeros_(self.time_film.weight)
        nn.init.zeros_(self.time_film.bias)

    def forward(self, x, temb, eeg_cond):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.time_film(temb).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.cross_attn(h, eeg_cond)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return x + h


class LatentUNetCA(nn.Module):
    """Small UNet for 4×16×16 latents with cross-attention EEG conditioning."""
    def __init__(self, in_ch=4, base_ch=128, ch_mult=(1,2,4),
                 time_dim=256, eeg_dim=512, n_heads=4, n_eeg_tokens=8):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbed(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = base_ch
        enc_chs = [ch]
        for m in ch_mult[:-1]:
            out_ch = base_ch * m
            self.enc.append(CAResBlock(ch, time_dim, eeg_dim, n_heads, n_eeg_tokens))
            self.downs.append(nn.Conv2d(ch, out_ch, 4, stride=2, padding=1))
            ch = out_ch
            enc_chs.append(ch)

        out_ch = base_ch * ch_mult[-1]
        self.bot_in  = CAResBlock(ch, time_dim, eeg_dim, n_heads, n_eeg_tokens)
        self.bot_mid = CAResBlock(ch, time_dim, eeg_dim, n_heads, n_eeg_tokens)
        self.bot_proj = nn.Conv2d(ch, out_ch, 1) if ch != out_ch else nn.Identity()
        ch = out_ch

        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        skip_chs = list(reversed(enc_chs))
        for i, m in enumerate(reversed(ch_mult[:-1])):
            sk = skip_chs[i]
            out_ch = base_ch * m
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch + sk, out_ch, 3, padding=1),
            ))
            self.dec.append(CAResBlock(out_ch, time_dim, eeg_dim, n_heads, n_eeg_tokens))
            ch = out_ch

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)

    def forward(self, x, t, eeg_cond):
        temb = self.time_mlp(t)

        h = self.in_conv(x)
        skips = [h]
        for blk, down in zip(self.enc, self.downs):
            h = blk(h, temb, eeg_cond)
            h = down(h)
            skips.append(h)

        h = self.bot_in(h, temb, eeg_cond)
        h = self.bot_mid(h, temb, eeg_cond)
        h = self.bot_proj(h)

        for up, blk in zip(self.ups, self.dec):
            sk = skips.pop()
            h = up(torch.cat([h, sk], dim=1))
            h = blk(h, temb, eeg_cond)

        return self.out_conv(F.silu(self.out_norm(h)))


# ── DDPM helpers ──────────────────────────────────────────────────────────────
def make_schedule(T, device):
    betas = torch.linspace(1e-4, 2e-2, T, device=device)
    alphas = 1.0 - betas
    acp = torch.cumprod(alphas, dim=0)
    return betas, alphas, acp

def q_sample(x0, t, acp, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    s = acp[t][:, None, None, None].sqrt()
    r = (1 - acp[t])[:, None, None, None].sqrt()
    return s * x0 + r * noise, noise

@torch.no_grad()
def sample_ddim(model, eeg_cond, acp, T, steps=50, latent_shape=(4,16,16), device='cuda'):
    seq = list(range(0, T, T // steps))
    x = torch.randn(eeg_cond.size(0), *latent_shape, device=device)
    for i in reversed(range(len(seq))):
        t = torch.full((x.size(0),), seq[i], dtype=torch.long, device=device)
        pred = model(x, t, eeg_cond)
        a  = acp[seq[i]]
        a_ = acp[seq[i-1]] if i > 0 else torch.tensor(1.0, device=device)
        x0_pred = (x - (1-a).sqrt() * pred) / a.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)
        x = a_.sqrt() * x0_pred + (1 - a_).sqrt() * pred
    return x

# ── EEG encoder ──────────────────────────────────────────────────────────────
def build_eeg_encoder(n_ch, dino_feat_dim, args, device):
    raw_occ = args.eeg_occipital_ids.strip().lower()
    occ_idx = None if raw_occ == "auto" else ([] if raw_occ == "none" else [int(x) for x in raw_occ.split(",") if x.strip()])
    model = EEGDINORegressor(
        eeg_channels=n_ch, n_subjects=1,
        dino_feat_dim=dino_feat_dim, latent_dim=LATENT_DIM,
        eeg_hidden=256, eeg_out=256, subj_emb_dim=32,
        n_heads=4, n_layers=4, dropout=0.1,
        temperature=0.1, encoder_type="v2",
        n_classes=9, eeg_occipital_indices=occ_idx,
    ).to(device)
    return model

def load_supcon_encoder(encoder, supcon_ckpt_dir, sid, device):
    path = os.path.join(supcon_ckpt_dir, f"subj{sid:02d}_best.pt")
    if not os.path.isfile(path):
        print(f"  [WARN] SupCon ckpt not found: {path}", flush=True)
        return False
    ckpt = torch.load(path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt.get("model", ckpt), strict=True)
    print(f"  [SupCon encoder] Loaded {path}", flush=True)
    return True

# ── Image latent precompute ───────────────────────────────────────────────────
IMG_TRANSFORM = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def encode_class_images(vae, img_root, cls_list, device):
    latents = []
    for c in cls_list:
        img = Image.open(os.path.join(img_root, f"{c:02d}.png")).convert("RGB")
        x = IMG_TRANSFORM(img).unsqueeze(0).to(device)
        with torch.no_grad():
            z = vae.encode(x).latent_dist.sample() * VAE_SCALE
        latents.append(z)
    return torch.cat(latents, dim=0)  # (9, 4, 16, 16)


def encode_class_images_multi(vae, img_root, cls_list, device):
    """Load all available images per class (original + alternative).
    Returns (9, K, 4, 16, 16) where K = number of images per class."""
    import glob
    all_class_latents = []
    for c in cls_list:
        imgs_for_class = []
        # Original image
        orig_path = os.path.join(img_root, f"{c:02d}.png")
        if os.path.isfile(orig_path):
            imgs_for_class.append(orig_path)
        # Alternative images (e.g. 01_air.png, 01_cup.png)
        for alt in sorted(glob.glob(os.path.join(img_root, f"{c:02d}_*.png"))):
            imgs_for_class.append(alt)
        class_latents = []
        for p in imgs_for_class:
            img = Image.open(p).convert("RGB")
            x = IMG_TRANSFORM(img).unsqueeze(0).to(device)
            with torch.no_grad():
                z = vae.encode(x).latent_dist.sample() * VAE_SCALE
            class_latents.append(z)
        all_class_latents.append(torch.cat(class_latents, dim=0))  # (K, 4, 16, 16)
    return all_class_latents  # list of 9 tensors, each (K, 4, 16, 16)

# ── Collapse diagnostics ──────────────────────────────────────────────────────
@torch.no_grad()
def collapse_diagnostics(unet, eeg_encoder, test_loader, vae, dino, proto_dino,
                          acp, num_timesteps, device, n_samples=50):
    cls_image_latents = None
    pred_classes = []
    true_classes  = []

    subj = torch.zeros(1, dtype=torch.long, device=device)
    unet.eval()
    eeg_encoder.eval()

    for eeg, _, lbl in test_loader:
        eeg = eeg.to(device)
        lbl_list = lbl.tolist()

        eeg_lat = eeg_encoder.encode_eeg(eeg, subj.expand(eeg.size(0)))
        # Project to 256-dim for UNet conditioning (use eeg_out directly)
        gen = sample_ddim(unet, eeg_lat, acp, num_timesteps, steps=50, device=device)

        # Decode latents
        decoded = vae.decode(gen / VAE_SCALE).sample  # (B,3,128,128)
        decoded = (decoded.clamp(-1, 1) + 1) / 2      # 0~1

        # DINO features of decoded images
        dino_tf = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        imgs_for_dino = torch.stack([dino_tf(x) for x in decoded])
        with torch.no_grad():
            feats = dino(imgs_for_dino)
        feats = F.normalize(feats, dim=-1)
        sims  = feats @ F.normalize(proto_dino, dim=-1).T
        preds = sims.argmax(dim=-1).cpu().tolist()
        pred_classes.extend(preds)
        true_classes.extend(lbl_list)

        if len(pred_classes) >= n_samples:
            break

    pred_classes = pred_classes[:n_samples]
    true_classes = true_classes[:n_samples]
    n = len(pred_classes)
    dino_top1 = sum(p == t for p, t in zip(pred_classes, true_classes)) / n

    from collections import Counter
    cnt = Counter(pred_classes)
    dominant_frac = cnt.most_common(1)[0][1] / n
    counts = np.array(list(cnt.values()), dtype=float)
    probs  = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-9))

    return {"dino_top1": dino_top1, "dominant": dominant_frac, "entropy": entropy}


# ── Training ──────────────────────────────────────────────────────────────────
def train_subject(sid, args, vae, dino, proto_dino, dino_feat_dim, device, save_dir):
    print(f"\n{'='*55}\n  Subject {sid:02d}", flush=True)
    set_seed(args.seed)

    # Dataset
    train_ds = VSReDataset(args.data_root, [sid], n_ch=args.n_ch, split="train", seed=args.seed, max_sessions=args.max_sessions)
    val_ds   = VSReDataset(args.data_root, [sid], n_ch=args.n_ch, split="val",   seed=args.seed, max_sessions=args.max_sessions)
    test_ds  = VSReDataset(args.data_root, [sid], n_ch=args.n_ch, split="test",  seed=args.seed, max_sessions=args.max_sessions)
    if len(train_ds) == 0:
        print(f"  [SKIP] S{sid:02d}: no training data", flush=True)
        return None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f"  S{sid:02d}  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}", flush=True)

    # Pre-encode class images to latents
    use_multi = getattr(args, 'multi_img', False)
    if use_multi:
        cls_latents_multi = encode_class_images_multi(vae, args.img_root, list(range(1, 10)), device)
        cls_latents = torch.stack([x[0] for x in cls_latents_multi])  # (9,4,16,16) primary
        print(f"  [MultiImg] K per class: {[x.size(0) for x in cls_latents_multi]}", flush=True)
    else:
        cls_latents = encode_class_images(vae, args.img_root, list(range(1, 10)), device)  # (9,4,16,16)
        cls_latents_multi = None

    # EEG encoder
    eeg_enc = build_eeg_encoder(args.n_ch, dino_feat_dim, args, device)
    if args.supcon_ckpt:
        load_supcon_encoder(eeg_enc, args.supcon_ckpt, sid, device)
    if args.freeze_encoder:
        for p in eeg_enc.parameters():
            p.requires_grad_(False)
        print(f"  [EEG encoder] Frozen", flush=True)

    # Latent UNet — cross-attention or FiLM conditioning
    if getattr(args, 'use_cross_attn', False):
        unet = LatentUNetCA(in_ch=4, base_ch=args.base_ch,
                            ch_mult=tuple(int(x) for x in args.ch_mult.split(",")),
                            time_dim=256, eeg_dim=LATENT_DIM,
                            n_heads=4, n_eeg_tokens=8).to(device)
        print(f"  [UNet] Cross-attention conditioning (n_eeg_tokens=8)", flush=True)
    else:
        unet = LatentUNet(in_ch=4, base_ch=args.base_ch,
                          ch_mult=tuple(int(x) for x in args.ch_mult.split(",")),
                          time_dim=256, cond_dim=LATENT_DIM).to(device)

    # DDPM schedule
    betas, _, acp = make_schedule(args.num_timesteps, device)

    params = list(unet.parameters())
    if not args.freeze_encoder:
        params += list(eeg_enc.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    augmenter = EEGAugment(noise_std=0.03, scale_range=(0.9,1.1), ch_drop_prob=0.05,
                            max_shift=10, freq_noise_std=0.0, p_noise=0.5, p_scale=0.5,
                            p_drop=0.2, p_shift=0.2, p_freq=0.0)

    best_val_loss = float("inf")
    best_path = os.path.join(save_dir, f"subj{sid:02d}_best.pt")
    subj_t = torch.zeros(1, dtype=torch.long, device=device)

    print(f"    Ep    TrLoss   ValLoss", flush=True)
    for epoch in range(1, args.epochs + 1):
        unet.train()
        eeg_enc.train()
        tr_loss = 0.0
        for eeg, _, lbl in train_loader:
            eeg = augmenter(eeg.to(device))
            lbl = lbl.to(device)

            # EEG encoding → 256-dim
            eeg_lat = eeg_enc.encode_eeg(eeg, subj_t.expand(eeg.size(0)))

            # Target: class image latents (use true label)
            if use_multi and cls_latents_multi is not None:
                # Randomly sample one image per sample from the available class images
                x0_list = []
                for li in lbl:
                    imgs = cls_latents_multi[li.item()]  # (K, 4, 16, 16)
                    idx = torch.randint(imgs.size(0), (1,)).item()
                    x0_list.append(imgs[idx])
                x0 = torch.stack(x0_list)  # (B, 4, 16, 16)
            else:
                x0 = cls_latents[lbl]  # (B,4,16,16)
            t  = torch.randint(0, args.num_timesteps, (eeg.size(0),), device=device)
            xt, noise = q_sample(x0, t, acp)

            pred = unet(xt, t, eeg_lat)
            loss = F.mse_loss(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        # Validation (MSE on random samples)
        unet.eval()
        eeg_enc.eval()
        val_loss = 0.0
        with torch.no_grad():
            for eeg, _, lbl in DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn):
                eeg = eeg.to(device)
                lbl = lbl.to(device)
                eeg_lat = eeg_enc.encode_eeg(eeg, subj_t.expand(eeg.size(0)))
                x0  = cls_latents[lbl]
                t   = torch.randint(0, args.num_timesteps, (eeg.size(0),), device=device)
                xt, noise = q_sample(x0, t, acp)
                pred = unet(xt, t, eeg_lat)
                val_loss += F.mse_loss(pred, noise).item()
        val_loss /= max(len(val_ds) // args.batch_size, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"unet": unet.state_dict(), "eeg_enc": eeg_enc.state_dict(),
                        "sid": sid, "config": vars(args)}, best_path)

        if epoch % 60 == 0:
            print(f"  {epoch:5d}  {tr_loss:.5f}  {val_loss:.5f}", flush=True)

    # Load best and evaluate collapse
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    unet.load_state_dict(ckpt["unet"])
    eeg_enc.load_state_dict(ckpt["eeg_enc"])

    diag = collapse_diagnostics(unet, eeg_enc, test_loader, vae, dino, proto_dino,
                                acp, args.num_timesteps, device)
    print(f"  [Done S{sid:02d}]  best_val={best_val_loss:.5f}  "
          f"DINO@1={diag['dino_top1']:.4f}  entropy={diag['entropy']:.3f}  dominant={diag['dominant']*100:.1f}%", flush=True)

    return {"sid": sid, "best_val_loss": best_val_loss, **diag}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",       default="./preproc_vs_re")
    parser.add_argument("--img_root",        default="./preproc_data_vi/images")
    parser.add_argument("--subject_ids",     default="1,2,18")
    parser.add_argument("--n_ch",            type=int,   default=32)
    parser.add_argument("--max_sessions",    type=int,   default=None)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--epochs",          type=int,   default=300)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--wd",              type=float, default=1e-4)
    parser.add_argument("--num_timesteps",   type=int,   default=200)
    parser.add_argument("--base_ch",         type=int,   default=128)
    parser.add_argument("--ch_mult",         type=str,   default="1,2,4")
    parser.add_argument("--eeg_occipital_ids", default="auto")
    parser.add_argument("--supcon_ckpt",     default="", help="SupCon checkpoint dir for EEG encoder init")
    parser.add_argument("--freeze_encoder",   action="store_true")
    parser.add_argument("--use_cross_attn",   action="store_true",
                        help="Use cross-attention UNet instead of FiLM")
    parser.add_argument("--multi_img",        action="store_true",
                        help="Randomly sample from all available class images per class during training")
    parser.add_argument("--dino_model",      default="dinov2_vits14")
    parser.add_argument("--ckpt_root",       default="./checkpoints_vsre_latent_gen")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}  latent diffusion (SD VAE)  epochs={args.epochs}", flush=True)

    all_sids = available_subjects(args.data_root)
    if args.subject_ids.strip().lower() == "all":
        subject_ids = all_sids
    else:
        subject_ids = []
        for tok in args.subject_ids.split(","):
            tok = tok.strip()
            if "-" in tok:
                a, b = tok.split("-")
                subject_ids.extend(range(int(a), int(b)+1))
            else:
                subject_ids.append(int(tok))
        subject_ids = [s for s in subject_ids if s in all_sids]
    print(f"[INFO] Subjects: {subject_ids}", flush=True)

    print("[INFO] Loading SD VAE...", flush=True)
    vae = load_vae(device)

    print(f"[INFO] Loading DINO: {args.dino_model}", flush=True)
    _hub_dir = os.path.expanduser("~/.cache/torch/hub")
    torch.hub.set_dir(_hub_dir)
    dino = torch.hub.load(
        os.path.join(_hub_dir, "facebookresearch_dinov2_main"),
        args.dino_model, source="local", verbose=False,
    )
    dino = dino.to(device).eval()
    for p in dino.parameters():
        p.requires_grad_(False)
    dino_feat_dim = DINO_DIM[args.dino_model]
    proto_dino = compute_class_prototypes(dino, args.img_root, list(range(1, 10)), device)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    enc_tag = "_supcon" if args.supcon_ckpt else ""
    frz_tag = "_frozen" if args.freeze_encoder else ""
    ca_tag  = "_ca" if getattr(args, 'use_cross_attn', False) else ""
    mi_tag  = "_mi" if getattr(args, 'multi_img', False) else ""
    save_dir = os.path.join(args.ckpt_root, f"{ts}_ch{args.n_ch}_ep{args.epochs}{enc_tag}{frz_tag}{ca_tag}{mi_tag}")
    os.makedirs(save_dir, exist_ok=True)

    results = []
    for sid in subject_ids:
        r = train_subject(sid, args, vae, dino, proto_dino, dino_feat_dim, device, save_dir)
        if r:
            results.append(r)

    # Summary
    print(f"\n{'='*55}\nSummary")
    print(f"  {'Subj':>5}  {'DINO@1':>7}  {'entropy':>8}  {'dominant':>9}")
    for r in results:
        print(f"  S{r['sid']:02d}    {r['dino_top1']:>7.4f}  {r['entropy']:>8.3f}  {r['dominant']*100:>8.1f}%")
    if results:
        print(f"  Mean   {np.mean([r['dino_top1'] for r in results]):>7.4f}  "
              f"{np.mean([r['entropy'] for r in results]):>8.3f}  "
              f"{np.mean([r['dominant'] for r in results])*100:>8.1f}%")

    out_csv = os.path.join(save_dir, "results_latent_gen.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sid","best_val_loss","dino_top1","entropy","dominant"])
        w.writeheader()
        w.writerows(results)
    print(f"[INFO] Saved: {save_dir}", flush=True)


if __name__ == "__main__":
    main()
