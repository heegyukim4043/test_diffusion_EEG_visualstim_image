"""
train_vs_re_lora_gen.py
────────────────────────────────────────────────────────────────────────────
Exp41: SD 1.5 LoRA generator with EEG cross-attention conditioning.

Architecture:
  - SD 1.5 UNet (mostly frozen) + LoRA adapters on attention layers
  - SD 1.5 VAE (frozen) for 512x512 image encoding (64x64x4 latents)
  - EEG SupCon encoder (frozen) → 512-dim → projected to N_tok x 768 SD conditioning tokens
  - DDPM training in VAE latent space

Usage:
  python train_vs_re_lora_gen.py --subject_ids 18 --epochs 100
"""

import argparse, csv, datetime, os, sys, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from collections import Counter
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_vs_re import VSReDataset, collate_fn, available_subjects
from model_eeg_dino import EEGDINORegressor, DINO_DIM, LATENT_DIM
from train_crosssubj_dino import set_seed, EEGAugment, compute_class_prototypes
from train_vs_re_latent_gen import make_schedule, sample_ddim, build_eeg_encoder, VAE_SCALE

N_CLASSES = 9
CLS_LIST  = list(range(1, 10))

# 512x512 transform for SD 1.5
IMG_TRANSFORM_512 = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

DINO_EVAL_TF = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def encode_class_images_512(vae, img_root, cls_list, device):
    """Encode 512x512 class images to 64x64x4 SD latents."""
    latents = []
    for c in cls_list:
        img = Image.open(os.path.join(img_root, f"{c:02d}.png")).convert("RGB")
        x = IMG_TRANSFORM_512(img).unsqueeze(0).to(device)
        with torch.no_grad():
            z = vae.encode(x).latent_dist.sample() * VAE_SCALE
        latents.append(z)
    return torch.cat(latents, dim=0)  # (9, 4, 64, 64)


# Class-preserving augmentation: natural vs symbolic classes
# Classes 1-9: airplane, cup, tree, digit1, digit3, digit5, heart, star, triangle
# Natural (safe for flip/color): airplane(1), cup(2), tree(3), heart(7), star(8)
# Symbolic (no flip, careful rotation): digit1(4), digit3(5), digit5(6), triangle(9)
_NATURAL_CLASSES = {0, 1, 2, 6, 7}   # 0-indexed: airplane, cup, tree, heart, star
_SYMBOLIC_CLASSES = {3, 4, 5, 8}      # 0-indexed: digit1, digit3, digit5, triangle

_AUG_NATURAL = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)], p=0.5),
    T.RandomApply([T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1))], p=0.3),
])
_AUG_SYMBOLIC = T.Compose([
    T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15)], p=0.4),
    T.RandomApply([T.RandomAffine(degrees=5, scale=(0.92, 1.08))], p=0.3),
])

def encode_class_images_512_aug(vae, img_root, cls_list, device, n_aug=4):
    """Encode 512x512 class images with augmentation → (9, n_aug, 4, 64, 64)."""
    to_tensor = T.ToTensor()
    all_latents = []
    for idx, c in enumerate(cls_list):
        img = Image.open(os.path.join(img_root, f"{c:02d}.png")).convert("RGB")
        base = to_tensor(img)  # (3, 512, 512) float32
        aug_fn = _AUG_NATURAL if idx in _NATURAL_CLASSES else _AUG_SYMBOLIC
        cls_lats = []
        # Always include original
        x0 = T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])(base).unsqueeze(0).to(device)
        with torch.no_grad():
            cls_lats.append(vae.encode(x0).latent_dist.sample() * VAE_SCALE)
        # Add augmented versions
        for _ in range(n_aug - 1):
            xa = T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])(aug_fn(base.clone())).unsqueeze(0).to(device)
            with torch.no_grad():
                cls_lats.append(vae.encode(xa).latent_dist.sample() * VAE_SCALE)
        all_latents.append(torch.cat(cls_lats, dim=0))  # (n_aug, 4, 64, 64)
    return all_latents  # list of 9 tensors


def load_sd15_unet_lora(lora_r=16, lora_alpha=32):
    """Load SD 1.5 UNet and apply LoRA to cross-attention layers."""
    from diffusers import UNet2DConditionModel
    from peft import LoraConfig, get_peft_model

    print("  Loading SD 1.5 UNet...", flush=True)
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet"
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    return unet


class EEGConditionProjector(nn.Module):
    """Projects EEG latent (512-dim) to SD cross-attention tokens (N_tok x 768).
    deep=False: 2-layer MLP  (default, Exp41/42-A/B-step2)
    deep=True:  3-layer MLP with LayerNorm  (Exp42-B Step 3)
    """
    def __init__(self, eeg_dim=512, sd_dim=768, n_tokens=8, deep=False):
        super().__init__()
        self.n_tokens = n_tokens
        if deep:
            self.proj = nn.Sequential(
                nn.Linear(eeg_dim, eeg_dim * 4),
                nn.LayerNorm(eeg_dim * 4),
                nn.SiLU(),
                nn.Linear(eeg_dim * 4, eeg_dim * 2),
                nn.LayerNorm(eeg_dim * 2),
                nn.SiLU(),
                nn.Linear(eeg_dim * 2, n_tokens * sd_dim),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(eeg_dim, eeg_dim * 2),
                nn.SiLU(),
                nn.Linear(eeg_dim * 2, n_tokens * sd_dim),
            )
        self.norm = nn.LayerNorm(sd_dim)

    def forward(self, eeg_lat):  # (B, 512) → (B, N_tok, 768)
        B = eeg_lat.size(0)
        out = self.proj(eeg_lat).view(B, self.n_tokens, -1)
        return self.norm(out)


@torch.no_grad()
def ddpm_q_sample(x0, t, acp, noise=None):
    if noise is None: noise = torch.randn_like(x0)
    s = acp[t][:, None, None, None].sqrt()
    r = (1 - acp[t])[:, None, None, None].sqrt()
    return s * x0 + r * noise, noise


@torch.no_grad()
def sample_sd_ddim(unet, cond_proj, eeg_lat, acp, T, steps=50, device='cuda', guidance=1.0):
    """DDIM sampling with SD 1.5 UNet conditioned on EEG."""
    B = eeg_lat.size(0)
    cond_tokens = cond_proj(eeg_lat)  # (B, N_tok, 768)

    seq = list(range(0, T, T // steps))
    x = torch.randn(B, 4, 64, 64, device=device)

    for i in reversed(range(len(seq))):
        t_val = seq[i]
        t_tensor = torch.full((B,), t_val, dtype=torch.long, device=device)
        noise_pred = unet(x, t_tensor, encoder_hidden_states=cond_tokens).sample
        a  = acp[seq[i]]
        a_ = acp[seq[i-1]] if i > 0 else torch.tensor(1.0, device=device)
        x0_pred = (x - (1-a).sqrt() * noise_pred) / a.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)
        x = a_.sqrt() * x0_pred + (1 - a_).sqrt() * noise_pred
    return x


@torch.no_grad()
def evaluate_lora(unet, cond_proj, eeg_enc, test_loader, vae, dino,
                   proto_dino, acp, T, device, n_samples=54):
    """Evaluate with DDIM sampling, decode, DINO@1."""
    subj = torch.zeros(1, dtype=torch.long, device=device)
    correct = {1:0, 3:0, 5:0}
    pred_classes, true_classes = [], []
    total = 0
    unet.eval(); cond_proj.eval(); eeg_enc.eval()

    for eeg, _, lbl in test_loader:
        eeg = eeg.to(device); lbl = lbl.to(device)
        eeg_lat = eeg_enc.encode_eeg(eeg, subj.expand(eeg.size(0)))
        gen_lat = sample_sd_ddim(unet, cond_proj, eeg_lat, acp, T, steps=30, device=device)
        # Decode 64x64 latent → 512x512 image → resize to 224 for DINO
        decoded = vae.decode(gen_lat / VAE_SCALE).sample.clamp(-1, 1)
        decoded = (decoded + 1) / 2
        imgs = torch.stack([DINO_EVAL_TF(x) for x in decoded])
        feats = F.normalize(dino(imgs), dim=-1)
        sims  = feats @ F.normalize(proto_dino, dim=-1).T
        preds = sims.argmax(dim=-1).cpu().tolist()
        pred_classes.extend(preds)
        true_classes.extend(lbl.cpu().tolist())
        for k in correct:
            topk = sims.topk(min(k, 9), dim=1).indices
            correct[k] += topk.eq(lbl.unsqueeze(1)).any(1).sum().item()
        total += eeg.size(0)
        if total >= n_samples:
            break

    cnt = Counter(pred_classes)
    dominant = cnt.most_common(1)[0][1] / max(len(pred_classes), 1)
    counts = np.array(list(cnt.values()), dtype=float)
    entropy = float(-np.sum((counts/counts.sum()) * np.log(counts/counts.sum() + 1e-9)))
    return {
        "top1": correct[1]/max(total,1),
        "top3": correct[3]/max(total,1),
        "top5": correct[5]/max(total,1),
        "dominant": dominant,
        "entropy": entropy,
    }


def train_subject(sid, args, vae, dino, proto_dino, dino_feat_dim, acp, device, save_dir):
    print(f"\n{'='*55}\n  Subject {sid:02d}", flush=True)
    set_seed(args.seed)

    train_ds = VSReDataset(args.data_root, [sid], n_ch=args.n_ch, split="train", seed=args.seed)
    val_ds   = VSReDataset(args.data_root, [sid], n_ch=args.n_ch, split="val",   seed=args.seed)
    test_ds  = VSReDataset(args.data_root, [sid], n_ch=args.n_ch, split="test",  seed=args.seed)
    if len(train_ds) == 0:
        return None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f"  S{sid:02d}  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}", flush=True)

    # Encode 512x512 class images to 64x64x4 latents
    use_aug_targets = getattr(args, 'augment_targets', False)
    if use_aug_targets:
        cls_latents_aug = encode_class_images_512_aug(vae, args.img_root, CLS_LIST, device, n_aug=4)
        cls_latents = torch.stack([x[0] for x in cls_latents_aug])  # primary (9,4,64,64)
        print(f"  Class latents: augmented (4 views/class)", flush=True)
    else:
        cls_latents = encode_class_images_512(vae, args.img_root, CLS_LIST, device)
        cls_latents_aug = None
        print(f"  Class latents: {cls_latents.shape}", flush=True)

    # EEG encoder (frozen)
    eeg_enc = build_eeg_encoder(args.n_ch, dino_feat_dim,
                                 type('a', (), {'eeg_occipital_ids': 'auto'})(), device)
    encoder_source = "random"
    supcon_ckpt_path_used = None
    if args.supcon_ckpt:
        ckpt_path = os.path.join(args.supcon_ckpt, f"subj{sid:02d}_best.pt")
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "eeg_enc" in ckpt:
                eeg_enc.load_state_dict(ckpt["eeg_enc"])
            elif "model" in ckpt:
                eeg_enc.load_state_dict(ckpt["model"])
            print(f"  [Encoder] Loaded SupCon checkpoint: {ckpt_path}", flush=True)
            encoder_source = "supcon"
            supcon_ckpt_path_used = ckpt_path
        else:
            if getattr(args, 'allow_random_encoder', False):
                print(f"  [Encoder] WARNING: {ckpt_path} not found — using random init (--allow_random_encoder set)", flush=True)
            else:
                raise FileNotFoundError(
                    f"SupCon checkpoint not found: {ckpt_path}\n"
                    f"Use --allow_random_encoder to explicitly allow random EEG encoder init."
                )
    for p in eeg_enc.parameters():
        p.requires_grad_(False)
    print(f"  [Encoder] Frozen (source={encoder_source})", flush=True)

    # SD 1.5 UNet with LoRA
    unet = load_sd15_unet_lora(args.lora_r, args.lora_alpha)
    unet = unet.to(device)
    if getattr(args, 'grad_ckpt', False):
        unet.enable_gradient_checkpointing()
        print("  [UNet] gradient checkpointing enabled", flush=True)

    # EEG conditioning projector
    cond_proj = EEGConditionProjector(eeg_dim=LATENT_DIM, sd_dim=768,
                                       n_tokens=args.n_eeg_tokens,
                                       deep=getattr(args, 'deep_proj', False)).to(device)

    # Only train: LoRA weights + cond_proj
    train_params = [p for p in unet.parameters() if p.requires_grad] + list(cond_proj.parameters())
    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.wd)

    augmenter = EEGAugment(noise_std=0.03, scale_range=(0.9,1.1), ch_drop_prob=0.05,
                            max_shift=10, freq_noise_std=0.0,
                            p_noise=0.5, p_scale=0.5, p_drop=0.2, p_shift=0.2, p_freq=0.0)

    best_val = float("inf")
    best_ep  = 0
    subj_t = torch.zeros(1, dtype=torch.long, device=device)
    best_path = os.path.join(save_dir, f"subj{sid:02d}_lora_best.pt")
    use_fp16 = getattr(args, 'fp16', False)
    scaler = GradScaler() if use_fp16 else None

    print(f"    Ep    TrLoss", flush=True)
    for epoch in range(1, args.epochs + 1):
        unet.train(); cond_proj.train()
        tr_loss = 0.0
        for eeg, _, lbl in train_loader:
            eeg = augmenter(eeg.to(device))
            lbl = lbl.to(device)
            with torch.no_grad():
                eeg_lat = eeg_enc.encode_eeg(eeg, subj_t.expand(eeg.size(0)))
            cond_tokens = cond_proj(eeg_lat)  # (B, N_tok, 768)
            if use_aug_targets and cls_latents_aug is not None:
                x0_list = []
                for li in lbl:
                    views = cls_latents_aug[li.item()]  # (n_aug, 4, 64, 64)
                    idx = torch.randint(views.size(0), (1,)).item()
                    x0_list.append(views[idx])
                x0 = torch.stack(x0_list)
            else:
                x0 = cls_latents[lbl]             # (B, 4, 64, 64)
            t  = torch.randint(0, args.num_timesteps, (eeg.size(0),), device=device)
            xt, noise = ddpm_q_sample(x0, t, acp)
            with autocast(enabled=use_fp16):
                noise_pred = unet(xt, t, encoder_hidden_states=cond_tokens).sample
                loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(train_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(train_params, 1.0)
                optimizer.step()
            tr_loss += loss.item()
        tr_loss /= max(len(train_loader), 1)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  {epoch:5d}  {tr_loss:.5f}", flush=True)

        if tr_loss < best_val:
            best_val = tr_loss; best_ep = epoch
            torch.save({
                "unet_lora": {k:v for k,v in unet.named_parameters() if v.requires_grad},
                "cond_proj": cond_proj.state_dict(),
                "sid": sid,
                "provenance": {
                    "supcon_ckpt_dir": args.supcon_ckpt,
                    "supcon_ckpt_path": supcon_ckpt_path_used,
                    "encoder_source": encoder_source,
                    "allow_random_encoder": getattr(args, 'allow_random_encoder', False),
                    "lora_r": args.lora_r,
                    "n_eeg_tokens": args.n_eeg_tokens,
                },
            }, best_path)

        if epoch - best_ep >= args.patience:
            print(f"  Early stop ep={epoch} (best={best_ep})", flush=True)
            break

    # Load best and evaluate
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    cond_proj.load_state_dict(ckpt["cond_proj"])
    # LoRA params already in unet, reload best
    for name, param in unet.named_parameters():
        if name in ckpt["unet_lora"]:
            param.data.copy_(ckpt["unet_lora"][name])

    diag = evaluate_lora(unet, cond_proj, eeg_enc, test_loader, vae, dino, proto_dino, acp, args.num_timesteps, device)
    print(f"  [Done S{sid:02d}]  DINO@1={diag['top1']:.4f} @3={diag['top3']:.4f} @5={diag['top5']:.4f}  entropy={diag['entropy']:.3f}  dominant={diag['dominant']*100:.1f}%  best_ep={best_ep}", flush=True)
    return {"sid": sid, "best_ep": best_ep, **diag}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",     default="./preproc_vs_re")
    parser.add_argument("--img_root",      default="./preproc_data_vi/images")
    parser.add_argument("--subject_ids",   default="18")
    parser.add_argument("--n_ch",          type=int,   default=32)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--patience",      type=int,   default=30)
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--wd",            type=float, default=1e-4)
    parser.add_argument("--num_timesteps", type=int,   default=1000)
    parser.add_argument("--lora_r",        type=int,   default=16)
    parser.add_argument("--lora_alpha",    type=int,   default=32)
    parser.add_argument("--n_eeg_tokens",  type=int,   default=8)
    parser.add_argument("--deep_proj",        action="store_true",
                        help="Use deeper 3-layer MLP projection (Exp42-B Step 3)")
    parser.add_argument("--augment_targets",  action="store_true",
                        help="Apply class-preserving augmentation to target images (Exp42-B Step 5)")
    parser.add_argument("--allow_random_encoder", action="store_true",
                        help="Allow random EEG encoder init when supcon_ckpt file is missing (debug only)")
    parser.add_argument("--supcon_ckpt",   default="checkpoints_vsre_latent_gen/20260610_105755_ch32_ep300_supcon_frozen_ca",
                        help="Dir with subj{sid:02d}_best.pt containing eeg_enc state")
    parser.add_argument("--dino_model",    default="dinov2_vits14")
    parser.add_argument("--ckpt_root",     default="./checkpoints_vsre_lora_gen")
    parser.add_argument("--fp16",      action="store_true", help="Mixed precision fp16 training (halves GPU memory)")
    parser.add_argument("--grad_ckpt", action="store_true", help="Gradient checkpointing on UNet (saves memory, slower)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}  SD1.5 LoRA generator", flush=True)

    subject_ids = []
    for tok in args.subject_ids.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-"); subject_ids.extend(range(int(a), int(b)+1))
        else:
            subject_ids.append(int(tok))
    avail = available_subjects(args.data_root)
    subject_ids = [s for s in subject_ids if s in avail]

    _hub_dir = os.path.expanduser("~/.cache/torch/hub")
    _dino_local = os.path.join(_hub_dir, "facebookresearch_dinov2_main")
    print("[INFO] Loading DINO...", flush=True)
    if os.path.isdir(_dino_local):
        dino = torch.hub.load(_dino_local, args.dino_model, source="local", verbose=False)
    else:
        print("[INFO]  local cache not found, downloading from facebookresearch/dinov2...", flush=True)
        dino = torch.hub.load("facebookresearch/dinov2", args.dino_model, verbose=False)
    dino = dino.to(device).eval()
    for p in dino.parameters(): p.requires_grad_(False)
    dino_feat_dim = DINO_DIM[args.dino_model]
    proto_dino = compute_class_prototypes(dino, args.img_root, CLS_LIST, device)

    print("[INFO] Loading SD VAE...", flush=True)
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    for p in vae.parameters(): p.requires_grad_(False)

    _, _, acp = make_schedule(args.num_timesteps, device)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.ckpt_root, f"{ts}_lora_r{args.lora_r}_ep{args.epochs}")
    os.makedirs(save_dir, exist_ok=True)

    all_results = []
    for sid in subject_ids:
        r = train_subject(sid, args, vae, dino, proto_dino, dino_feat_dim, acp, device, save_dir)
        if r: all_results.append(r)

    print("\nSummary:")
    for r in all_results:
        print(f"  S{r['sid']:02d}: DINO@1={r['top1']:.4f} @3={r['top3']:.4f} dominant={r['dominant']*100:.1f}% entropy={r['entropy']:.3f}")

    out_csv = os.path.join(save_dir, "results_lora_gen.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sid","best_ep","top1","top3","top5","dominant","entropy"])
        w.writeheader(); w.writerows(all_results)
    print(f"[INFO] Saved: {save_dir}", flush=True)


if __name__ == "__main__":
    main()
