"""
train_vs_re_gen.py
────────────────────────────────────────────────────────────────────────────
preproc_vs_re 데이터 기반  Subject-wise EEG-only 생성 모델 (Stage gen)

전략: subject-wise first
  - 피험자별 독립 모델
  - EEG 32ch, 0~2 sec
  - EEG-only conditioning (class label 없음)
  - 기존 EEGDiffusionModel128 재사용

사용 예시:
  python train_vs_re_gen.py --subject_ids 1
  python train_vs_re_gen.py --subject_ids 1,2,18 --epochs 300
  python train_vs_re_gen.py --subject_ids all --max_sessions 2
"""

import os, sys, random, copy, argparse, math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_vs_re import (
    VSReDataset, collate_fn, available_subjects, session_counts,
)
from model_128_eegonly_transformer_repa import EEGDiffusionModel128
from model_128_eegonly_transformer import _ssim
from model_eeg_dino import load_dino_encoder, DINO_DIM
from train_crosssubj_dino import compute_class_prototypes

# LPIPS (optional — graceful fallback if not installed)
try:
    import lpips as _lpips_lib
    _lpips_fn = None   # lazy init per device
    _LPIPS_OK = True
except ImportError:
    _LPIPS_OK = False


# ── Image-level DINO prototype losses (Priority 1, HUMAN_DIRECTIVE) ──────────
DINO_MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINO_STD_T  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def _dino_feat_from_x0(x0: torch.Tensor, dino, device: torch.device) -> torch.Tensor:
    """x0 in [-1,1] → DINOv2 L2-normalized feature (B, DINO_DIM). Frozen, no grad."""
    x01  = (x0.clamp(-1, 1) + 1.0) * 0.5
    x224 = F.interpolate(x01, size=(224, 224), mode="bilinear", align_corners=False)
    mean = DINO_MEAN_T.to(device)
    std  = DINO_STD_T.to(device)
    x224 = (x224 - mean) / (std + 1e-8)
    return F.normalize(dino(x224), dim=1)


def compute_proto_loss(
    x0: torch.Tensor,
    lbl: torch.Tensor,
    dino,
    proto_dino: torch.Tensor,
    lambda_attract: float,
    lambda_sep: float,
    sep_margin: float = 0.1,
) -> torch.Tensor:
    """
    Image-level DINO prototype losses.

    attraction: pull generated image toward true-label prototype
                loss = 1 - cos(feat_gen, proto_true)
    separation: push generated image away from nearest wrong prototype
                loss = relu(sim_wrong_max - sim_true + margin)
    """
    if lambda_attract <= 0.0 and lambda_sep <= 0.0:
        return torch.tensor(0.0, device=x0.device)

    feat = _dino_feat_from_x0(x0, dino, x0.device)   # (B, DINO_DIM)
    loss = torch.tensor(0.0, device=x0.device)

    if lambda_attract > 0.0:
        true_proto = proto_dino[lbl]                           # (B, DINO_DIM)
        attract    = 1.0 - (feat * true_proto).sum(dim=1)     # (B,) in [0,2]
        loss = loss + lambda_attract * attract.mean()

    if lambda_sep > 0.0:
        sim_all   = feat @ proto_dino.T                        # (B, n_classes)
        true_sim  = sim_all.gather(1, lbl.unsqueeze(1)).squeeze(1)
        # mask out true-label to find nearest wrong
        sim_all   = sim_all.clone()
        sim_all.scatter_(1, lbl.unsqueeze(1), -1e4)
        wrong_max = sim_all.max(dim=1).values
        margin_l  = F.relu(wrong_max - true_sim + sep_margin)
        loss = loss + lambda_sep * margin_l.mean()

    return loss


# ── Anti-collapse auxiliary head ──────────────────────────────────────────────
class AntiCollapseHead(nn.Module):
    """
    Lightweight auxiliary head attached to the EEG encoder output (cond_emb).

    Two sub-losses (each ablatable via lambda weight):
      1. dino_align: project EEG latent → DINO space, cosine proto-CE
         (pulls EEG latent toward the correct DINO class prototype)
      2. aux_ce: linear classifier head on EEG latent
         (explicit class discrimination in the generation latent space)

    Both losses are regularizers — they do NOT receive GT class label at inference.
    """
    def __init__(self, cond_dim: int = 256, dino_dim: int = 384, n_classes: int = 9,
                 temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dino_proj = nn.Sequential(
            nn.Linear(cond_dim, dino_dim),
            nn.LayerNorm(dino_dim),
        )
        self.ce_head = nn.Linear(cond_dim, n_classes)

    def forward(
        self,
        cond_emb: torch.Tensor,         # (B, cond_dim)
        labels: torch.Tensor,           # (B,) long, 0-based
        proto_dino: torch.Tensor,       # (n_classes, dino_dim) L2-normalized
        lambda_dino_align: float,
        lambda_aux_ce: float,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=cond_emb.device)

        if lambda_dino_align > 0.0:
            proj = F.normalize(self.dino_proj(cond_emb), dim=1)   # (B, dino_dim)
            sim  = proj @ proto_dino.T / self.temperature          # (B, n_classes)
            loss = loss + lambda_dino_align * F.cross_entropy(sim, labels)

        if lambda_aux_ce > 0.0:
            logits = self.ce_head(cond_emb)                        # (B, n_classes)
            loss = loss + lambda_aux_ce * F.cross_entropy(logits, labels)

        return loss


def _get_lpips(device):
    """Lazy-initialize LPIPS network (VGG) on the given device."""
    global _lpips_fn
    if not _LPIPS_OK:
        return None
    if _lpips_fn is None:
        _lpips_fn = _lpips_lib.LPIPS(net="vgg").to(device)
        _lpips_fn.eval()
        for p in _lpips_fn.parameters():
            p.requires_grad_(False)
    return _lpips_fn


# ── 유틸 ─────────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def ema_update(model, ema_model, decay=0.999):
    msd = model.state_dict()
    for name, param in ema_model.state_dict().items():
        if name in msd:
            param.copy_(param * decay + msd[name] * (1.0 - decay))


img_transform = T.Compose([
    T.Resize(128), T.CenterCrop(128), T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ── 생성 모델 용 Dataset wrapper ─────────────────────────────────────────────
class GenDataset(Dataset):
    """
    VSReDataset에서 클래스 이미지를 함께 반환하는 Dataset.
    (eeg, label, image_tensor) 반환
    """
    def __init__(self, vsre_ds, img_root, n_classes=9):
        self.base = vsre_ds
        # 클래스 이미지 미리 로드 (0-based index → 1-based file name)
        self.class_imgs = []
        for c in range(n_classes):
            p = os.path.join(img_root, f"{c+1:02d}.png")
            img = img_transform(Image.open(p).convert("RGB"))
            self.class_imgs.append(img)
        self.class_imgs = torch.stack(self.class_imgs)  # (9, 3, 128, 128)

    def __len__(self): return len(self.base)

    def __getitem__(self, i):
        eeg, subj, lbl = self.base[i]
        img = self.class_imgs[lbl]   # target image for perceptual loss
        return eeg, lbl, img


def gen_collate(batch):
    eeg  = torch.stack([b[0] for b in batch])
    lbl  = torch.tensor([b[1] for b in batch], dtype=torch.long)
    imgs = torch.stack([b[2] for b in batch])
    return eeg, lbl, imgs


@torch.no_grad()
def evaluate_generation(
    model,
    loader,
    dino,
    proto_dino,
    device,
    ddim_steps=50,
    guidance_scale=1.5,
    eta=0.0,
):
    dino_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    dino_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    lpips_fn = _get_lpips(device)   # None if lpips not installed

    n_classes = proto_dino.size(0)
    total = 0
    l1_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    proto_sim_sum    = 0.0
    proto_margin_sum = 0.0
    top1 = top3 = top5 = 0
    pred_hist = np.zeros(n_classes, dtype=int)

    model.eval()
    dino.eval()

    for eeg, lbl, img_tgt in loader:
        eeg = eeg.to(device)
        lbl = lbl.to(device)
        img_tgt = img_tgt.to(device)

        gen = model.sample_ddim(
            eeg,
            num_steps=ddim_steps,
            guidance_scale=guidance_scale,
            eta=eta,
        )
        bsz = eeg.size(0)

        l1_sum   += (gen - img_tgt).abs().mean().item() * bsz
        ssim_sum += _ssim(gen, img_tgt).item() * bsz

        # LPIPS: expects [-1, 1] inputs (already in that range)
        if lpips_fn is not None:
            lp = lpips_fn(gen.clamp(-1, 1), img_tgt.clamp(-1, 1))
            lpips_sum += lp.mean().item() * bsz

        gen_01 = (gen.clamp(-1, 1) + 1.0) * 0.5
        gen_224 = F.interpolate(gen_01, size=(224, 224), mode="bilinear", align_corners=False)
        gen_224 = (gen_224 - dino_mean) / (dino_std + 1e-8)

        feat = F.normalize(dino(gen_224), dim=1)
        sim = feat @ proto_dino.T
        pred = sim.argmax(dim=1).cpu().numpy()
        for p in pred:
            pred_hist[p] += 1
        top1 += sim.topk(1, dim=1).indices.eq(lbl.unsqueeze(1)).any(1).sum().item()
        top3 += sim.topk(min(3, sim.size(1)), dim=1).indices.eq(lbl.unsqueeze(1)).any(1).sum().item()
        top5 += sim.topk(min(5, sim.size(1)), dim=1).indices.eq(lbl.unsqueeze(1)).any(1).sum().item()
        # True-label prototype similarity and true-vs-best-wrong margin (Priority 1)
        true_sim = sim.gather(1, lbl.unsqueeze(1)).squeeze(1)
        proto_sim_sum += true_sim.sum().item()
        sim_mask = sim.clone()
        sim_mask.scatter_(1, lbl.unsqueeze(1), -1e4)
        wrong_max = sim_mask.max(dim=1).values
        proto_margin_sum += (true_sim - wrong_max).sum().item()
        total += bsz

    if total == 0:
        return {"l1": 0.0, "ssim": 0.0, "lpips": None,
                "dino_top1": 0.0, "dino_top3": 0.0, "dino_top5": 0.0,
                "pred_entropy": 0.0, "dominant_frac": 0.0,
                "proto_sim": 0.0, "proto_margin": 0.0}

    # Predicted-class entropy (normalized)
    hist_norm = pred_hist / total
    p = hist_norm + 1e-12
    entropy_norm = -np.sum(p * np.log(p)) / np.log(n_classes)
    dominant_frac = hist_norm.max()

    return {
        "l1":           l1_sum / total,
        "ssim":         ssim_sum / total,
        "lpips":        lpips_sum / total if lpips_fn is not None else None,
        "dino_top1":    top1 / total,
        "dino_top3":    top3 / total,
        "dino_top5":    top5 / total,
        "pred_entropy": float(entropy_norm),
        "dominant_frac": float(dominant_frac),
        "proto_sim":    proto_sim_sum / total,      # true-label prototype cosine sim
        "proto_margin": proto_margin_sum / total,   # true-label sim minus best-wrong sim
    }


# ── 샘플 그리드 저장 ─────────────────────────────────────────────────────────
@torch.no_grad()
def save_sample_grid(
    model,
    eeg_sample,
    lbl_sample,
    save_path,
    device,
    ddim_steps=50,
    guidance_scale=1.5,
    eta=0.0,
):
    model.eval()
    eeg_sample = eeg_sample.to(device)
    n = eeg_sample.size(0)
    if hasattr(model, "sample_ddim"):
        samples = model.sample_ddim(
            eeg_sample,
            num_steps=ddim_steps,
            guidance_scale=guidance_scale,
            eta=eta,
        )
    else:
        samples = model.sample(eeg_sample, guidance_scale=guidance_scale)
    samples = (samples.clamp(-1, 1) + 1) / 2   # [0,1]
    ncols = min(n, 9)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = np.array(axes).reshape(-1)
    for ax in axes: ax.axis("off")
    for i in range(n):
        img = samples[i].cpu().permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(f"cls {lbl_sample[i].item()+1}", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


# ── 단일 피험자 학습 ─────────────────────────────────────────────────────────
def train_subject(sid, args, device, save_dir, dino, proto_dino):
    set_seed(args.seed)

    subj_map = {sid: 0}
    base_train = VSReDataset(args.data_root, [sid], subj_map, args.n_ch,
                             "train", args.seed, args.max_sessions)
    base_val   = VSReDataset(args.data_root, [sid], subj_map, args.n_ch,
                             "val",   args.seed, args.max_sessions)
    base_test  = VSReDataset(args.data_root, [sid], subj_map, args.n_ch,
                             "test",  args.seed, args.max_sessions)

    if len(base_train) == 0:
        print(f"  [SKIP] S{sid:02d}: no data")
        return

    train_ds = GenDataset(base_train, args.img_root)
    val_ds   = GenDataset(base_val,   args.img_root)
    test_ds  = GenDataset(base_test,  args.img_root)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, collate_fn=gen_collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, collate_fn=gen_collate)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0, collate_fn=gen_collate)

    eeg_ch = args.n_ch

    raw_occ = args.eeg_occipital_ids.strip().lower()
    if raw_occ in ("auto", ""):
        occipital_indices = None          # OccipitalChannelGate uses BioSemi32 default
    elif raw_occ == "none":
        occipital_indices = []            # uniform init (no prior)
    else:
        occipital_indices = [int(x) for x in raw_occ.split(",") if x.strip()]
    model = EEGDiffusionModel128(
        eeg_channels=eeg_ch,
        num_classes=9,
        num_timesteps=args.num_timesteps,
        base_channels=args.base_channels,
        ch_mult=tuple(int(x) for x in args.ch_mult.split(",")),
        lambda_percept=args.lambda_percept,
        lambda_rec=args.lambda_rec,
        lambda_ssim=args.lambda_ssim,
        lambda_lpips=args.lambda_lpips,
        beta_schedule=args.beta_schedule,
        encoder_version=args.encoder_version,
        eeg_stem_filters=args.eeg_stem_filters,
        eeg_occipital_indices=occipital_indices,
        eeg_tf_layers=args.eeg_tf_layers,
    ).to(device)

    # Load SupCon encoder weights if provided
    if args.supcon_ckpt:
        sc_path = os.path.join(args.supcon_ckpt, f"subj{sid:02d}_best.pt")
        if os.path.isfile(sc_path):
            sc_ckpt = torch.load(sc_path, map_location=device, weights_only=False)
            sc_state = sc_ckpt.get("model", sc_ckpt)
            enc_state = {k[len("eeg_encoder."):]: v
                         for k, v in sc_state.items() if k.startswith("eeg_encoder.")}
            missing, unexpected = model.eeg_encoder.load_state_dict(enc_state, strict=True)
            print(f"  [SupCon encoder] Loaded from {sc_path}  (missing={missing} unexpected={unexpected})", flush=True)
            if args.freeze_encoder:
                for p in model.eeg_encoder.parameters():
                    p.requires_grad_(False)
                print(f"  [SupCon encoder] Frozen", flush=True)
        else:
            print(f"  [WARN] SupCon checkpoint not found: {sc_path} — training from random init", flush=True)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters(): p.requires_grad_(False)

    # Anti-collapse auxiliary head (only created when enabled)
    use_anticollapse = (args.lambda_dino_align > 0.0 or args.lambda_aux_ce > 0.0)
    ac_head = None
    if use_anticollapse:
        ac_head = AntiCollapseHead(
            cond_dim=256,
            dino_dim=DINO_DIM[args.dino_model],
            n_classes=9,
            temperature=0.1,
        ).to(device)
        print(f"  [AntiCollapse] lambda_dino_align={args.lambda_dino_align}  "
              f"lambda_aux_ce={args.lambda_aux_ce}", flush=True)

    opt_params = list(model.parameters())
    if ac_head is not None:
        opt_params += list(ac_head.parameters())
    optimizer = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.wd)
    warmup = max(1, args.epochs // 10)
    def lr_lambda(ep):
        if ep < warmup: return ep / warmup
        p = (ep - warmup) / max(1, args.epochs - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    subj_dir  = os.path.join(save_dir, f"subj{sid:02d}")
    os.makedirs(subj_dir, exist_ok=True)
    ckpt_path = os.path.join(subj_dir, "best.pt")
    sample_dir = os.path.join(subj_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    print(f"  S{sid:02d}  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}", flush=True)
    print(f"  {'Ep':>4}  {'TrLoss':>8}  {'ValLoss':>8}", flush=True)

    # 고정 샘플 (시각화용)
    fixed_eeg, fixed_lbl, _ = next(iter(val_loader))
    fixed_eeg = fixed_eeg[:9]
    fixed_lbl = fixed_lbl[:9]

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = ep_total = 0
        for eeg, lbl, img_tgt in train_loader:
            eeg     = eeg.to(device)
            lbl     = lbl.to(device)
            img_tgt = img_tgt.to(device)

            t = torch.randint(0, model.num_timesteps, (eeg.size(0),),
                              device=device, dtype=torch.long)

            use_image_proto = (args.lambda_proto_attract > 0 or args.lambda_proto_sep > 0)
            loss, x0_pred = model.p_losses(img_tgt, eeg, t=t, return_x0=use_image_proto) \
                            if use_image_proto else (model.p_losses(img_tgt, eeg, t=t), None)

            # Image-level DINO prototype loss (Priority 1)
            # t-filter: only apply at low-noise timesteps where x0_pred is accurate
            if use_image_proto and x0_pred is not None:
                t_thresh = getattr(args, "t_max_proto", model.num_timesteps)
                if t_thresh < model.num_timesteps:
                    low_t = (t < t_thresh)
                    if low_t.any():
                        loss = loss + compute_proto_loss(
                            x0_pred[low_t], lbl[low_t], dino, proto_dino,
                            args.lambda_proto_attract, args.lambda_proto_sep,
                        )
                else:
                    loss = loss + compute_proto_loss(
                        x0_pred, lbl, dino, proto_dino,
                        args.lambda_proto_attract, args.lambda_proto_sep,
                    )

            # Anti-collapse auxiliary loss (Priority 2)
            if ac_head is not None:
                cond_emb = model.get_cond_emb_eeg_only(eeg).detach()
                loss = loss + ac_head(
                    cond_emb, lbl, proto_dino,
                    args.lambda_dino_align, args.lambda_aux_ce,
                )

            optimizer.zero_grad(); loss.backward()
            params_to_clip = list(model.parameters())
            if ac_head is not None:
                params_to_clip += list(ac_head.parameters())
            nn.utils.clip_grad_norm_(params_to_clip, 1.0)
            optimizer.step()
            ema_update(model, ema_model, decay=0.999)
            ep_loss  += loss.item() * eeg.size(0)
            ep_total += eeg.size(0)

        scheduler.step()

        model.eval()
        val_loss = val_total = 0
        with torch.no_grad():
            for eeg, lbl, img_tgt in val_loader:
                eeg     = eeg.to(device)
                img_tgt = img_tgt.to(device)
                t_v = torch.randint(0, model.num_timesteps, (eeg.size(0),),
                                    device=device, dtype=torch.long)
                l = model.p_losses(img_tgt, eeg, t=t_v)
                val_loss  += l.item() * eeg.size(0)
                val_total += eeg.size(0)
        val_loss /= val_total

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "ema_model": ema_model.state_dict(),
                "model": model.state_dict(),
                "config": {
                    "eeg_channels": eeg_ch,
                    "num_classes": 9,
                    "num_timesteps": args.num_timesteps,
                    "base_channels": args.base_channels,
                    "ch_mult": args.ch_mult,
                    "n_ch": args.n_ch,
                    "sid": sid,
                    "encoder_version": args.encoder_version,
                    "eeg_stem_filters": args.eeg_stem_filters,
                    "eeg_occipital_ids": args.eeg_occipital_ids,
                    "eeg_occipital_indices": occipital_indices,
                    "beta_schedule": args.beta_schedule,
                    "lambda_lpips": args.lambda_lpips,
                    "lambda_percept": args.lambda_percept,
                    "lambda_rec": args.lambda_rec,
                    "lambda_ssim": args.lambda_ssim,
                    "guidance_scale": args.guidance_scale,
                    "eta": args.eta,
                    "eval_ddim_steps": args.eval_ddim_steps,
                    "sample_ddim_steps": args.sample_ddim_steps,
                    "subject_ids": args.subject_ids,
                    "max_sessions": args.max_sessions,
                    "seed": args.seed,
                    "lambda_proto_attract": args.lambda_proto_attract,
                    "lambda_proto_sep": args.lambda_proto_sep,
                    "lambda_dino_align": args.lambda_dino_align,
                    "lambda_aux_ce": args.lambda_aux_ce,
                },
            }, ckpt_path)

        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            tr_l = ep_loss / ep_total
            print(f"  {epoch:>4}  {tr_l:>8.5f}  {val_loss:>8.5f}", flush=True)
            # 샘플 이미지 저장 (EMA 모델)
            save_sample_grid(
                ema_model, fixed_eeg, fixed_lbl,
                os.path.join(sample_dir, f"ep{epoch:04d}.png"),
                device,
                ddim_steps=args.sample_ddim_steps,
                guidance_scale=args.guidance_scale,
                eta=args.eta,
            )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ema_model.load_state_dict(ckpt["ema_model"])
    ema_model.eval()
    metrics = evaluate_generation(
        ema_model,
        test_loader,
        dino,
        proto_dino,
        device,
        ddim_steps=args.eval_ddim_steps,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
    )

    lpips_str = f"{metrics['lpips']:.4f}" if metrics.get("lpips") is not None else "N/A"
    print(
        f"  [Done S{sid:02d}]  best val loss={best_val_loss:.5f}  "
        f"L1={metrics['l1']:.5f}  SSIM={metrics['ssim']:.4f}  "
        f"LPIPS={lpips_str}  DINO@1={metrics['dino_top1']:.4f}  "
        f"proto_sim={metrics['proto_sim']:.4f}  margin={metrics['proto_margin']:.4f}  "
        f"entropy={metrics['pred_entropy']:.3f}  dominant={metrics['dominant_frac']:.1%}",
        flush=True,
    )
    return {
        "sid": sid,
        "best_val_loss":  round(best_val_loss, 6),
        "l1":             round(metrics["l1"], 6),
        "ssim":           round(metrics["ssim"], 6),
        "lpips":          round(metrics["lpips"], 6) if metrics.get("lpips") is not None else "",
        "dino_top1":      round(metrics["dino_top1"], 6),
        "dino_top3":      round(metrics["dino_top3"], 6),
        "dino_top5":      round(metrics["dino_top5"], 6),
        "pred_entropy":   round(metrics["pred_entropy"], 4),
        "dominant_frac":  round(metrics["dominant_frac"], 4),
        "proto_sim":      round(metrics["proto_sim"], 4),
        "proto_margin":   round(metrics["proto_margin"], 4),
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    type=str, default="./preproc_vs_re")
    parser.add_argument("--img_root",     type=str, default="./preproc_data_vi/images")
    parser.add_argument("--subject_ids",  type=str, default="all")
    parser.add_argument("--n_ch",         type=int, default=32)
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--epochs",       type=int, default=300)
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--wd",           type=float, default=1e-4)
    parser.add_argument("--num_timesteps",type=int, default=200)
    parser.add_argument("--base_channels",type=int, default=64)
    parser.add_argument("--ch_mult",      type=str, default="1,2,4,4")
    parser.add_argument("--lambda_percept",type=float, default=0.1,
                        help="ResNet18 perceptual loss weight")
    parser.add_argument("--lambda_rec",   type=float, default=0.01,
                        help="L1 reconstruction loss weight")
    parser.add_argument("--lambda_ssim",  type=float, default=0.05)
    parser.add_argument("--lambda_lpips", type=float, default=0.0,
                        help="LPIPS VGG loss weight (0=disabled; test separately at 0.01)")
    parser.add_argument("--beta_schedule",type=str,   default="linear",
                        choices=["linear", "cosine"],
                        help="Noise schedule (linear | cosine); cosine not better in current setup")
    parser.add_argument("--encoder_version", type=str, default="v1",
                        choices=["v1", "v2"],
                        help="EEG encoder: v1=original Conv+Transformer, v2=enhanced (OccipitalGate+MultiScaleStem+Transformer)")
    parser.add_argument("--eeg_stem_filters",    type=int, default=32,
                        help="V2 encoder: filters per temporal scale branch (total = 3x)")
    parser.add_argument("--eeg_occipital_ids",   type=str, default="auto",
                        help="V2 encoder: 'auto'=BioSemi32 default prior | 'none'=uniform | comma-sep indices e.g. '14,15,16'")
    parser.add_argument("--dino_model",   type=str, default="dinov2_vits14",
                        choices=list(DINO_DIM.keys()))
    parser.add_argument("--sample_ddim_steps", type=int, default=50)
    parser.add_argument("--eval_ddim_steps",   type=int, default=50)
    parser.add_argument("--guidance_scale",    type=float, default=1.5)
    parser.add_argument("--eta",               type=float, default=0.0,
                        help="DDIM eta: 0.0 = deterministic, >0 adds stochasticity")
    # Image-level prototype losses (Priority 1, HUMAN_DIRECTIVE)
    parser.add_argument("--lambda_proto_attract", type=float, default=0.0,
                        help="Generated image -> true-label DINO prototype attraction (0=disabled, try 0.1)")
    parser.add_argument("--lambda_proto_sep",     type=float, default=0.0,
                        help="Generated image -> nearest wrong prototype margin separation (0=disabled, try 0.05)")
    parser.add_argument("--t_max_proto", type=int, default=9999,
                        help="Only apply proto loss at timesteps t < t_max_proto (9999=all; try T//3 e.g. 67 for T=200)")
    # Anti-collapse auxiliary losses (Priority 2, HUMAN_DIRECTIVE)
    parser.add_argument("--lambda_dino_align", type=float, default=0.0,
                        help="EEG latent -> DINO prototype cosine CE loss weight (0=disabled, try 0.1)")
    parser.add_argument("--lambda_aux_ce",     type=float, default=0.0,
                        help="Auxiliary classification CE head on EEG latent (0=disabled, try 0.1)")
    parser.add_argument("--eeg_tf_layers",  type=int, default=2,
                        help="EEG encoder transformer layers (use 4 to match SupCon Exp23-B)")
    parser.add_argument("--supcon_ckpt",    type=str, default="",
                        help="Dir containing subj{sid:02d}_best.pt SupCon checkpoints; loads encoder weights into generator")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze EEG encoder after loading SupCon weights (linear probe mode)")
    parser.add_argument("--ckpt_root",    type=str, default="./checkpoints_vsre_gen")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}  n_ch={args.n_ch}  max_sessions={args.max_sessions}")

    all_sids = available_subjects(args.data_root)
    if args.subject_ids == "all":
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

    sc = session_counts(args.data_root)
    print(f"[INFO] Subjects ({len(subject_ids)}): {subject_ids}")

    print(f"[INFO] Loading DINO for generation eval: {args.dino_model}")
    dino = load_dino_encoder(args.dino_model, device)
    proto_dino = compute_class_prototypes(dino, args.img_root, list(range(1, 10)), device)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    sess_tag = f"cap{args.max_sessions}" if args.max_sessions else "merged"
    enc_tag  = f"_supcon{'_frozen' if args.freeze_encoder else '_init'}" if args.supcon_ckpt else ""
    save_dir = os.path.join(args.ckpt_root,
                            f"{ts}_ch{args.n_ch}_{sess_tag}_ep{args.epochs}{enc_tag}")
    os.makedirs(save_dir, exist_ok=True)

    import csv
    all_results = []
    for sid in subject_ids:
        print(f"\n{'='*55}")
        print(f"  Subject {sid:02d}  (sessions={sc.get(sid,0)})")
        r = train_subject(sid, args, device, save_dir, dino, proto_dino)
        if r:
            all_results.append(r)

    # 요약 CSV
    out_csv = os.path.join(save_dir, "results_gen.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "best_val_loss", "l1", "ssim", "lpips",
                    "dino_top1", "dino_top3", "dino_top5",
                    "pred_entropy", "dominant_frac", "proto_sim", "proto_margin"])
        for r in all_results:
            w.writerow([
                f"S{r['sid']:02d}",
                r["best_val_loss"],
                r["l1"],
                r["ssim"],
                r.get("lpips", ""),
                r["dino_top1"],
                r["dino_top3"],
                r["dino_top5"],
                r.get("pred_entropy", ""),
                r.get("dominant_frac", ""),
                r.get("proto_sim", ""),
                r.get("proto_margin", ""),
            ])

    print(f"\n[INFO] All done. Results: {save_dir}")
    print(f"[INFO] CSV: {out_csv}")


if __name__ == "__main__":
    main()
