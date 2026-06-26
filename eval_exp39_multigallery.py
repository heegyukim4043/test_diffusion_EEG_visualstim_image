"""
eval_exp39_multigallery.py
────────────────────────────────────────────────────────────────────────────
Exp39: Multi-image retrieval gallery evaluation.

Current: 9 classes × 1 image → DINO@1
Target:  9 classes × 2 images (original + alternative) → DINO@1
Also:    9 classes × N augmented views (10 per class via DINOv2 aug)

Evaluates existing Exp37 VS checkpoints (S18, S01) without retraining.

Usage:
  python eval_exp39_multigallery.py
  python eval_exp39_multigallery.py --subject_ids 1,18 --ckpt_dir checkpoints_vsre_latent_gen/...
"""

import argparse, os, sys, csv
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_vs_re import VSReDataset, collate_fn
from model_eeg_dino import DINO_DIM, LATENT_DIM
from train_vs_re_latent_gen import (
    LatentUNetCA, make_schedule, sample_ddim,
    build_eeg_encoder, VAE_SCALE,
)
from train_crosssubj_dino import set_seed

N_CLASSES = 9
CLS_LIST  = list(range(1, 10))

IMG_TRANSFORM = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

DINO_EVAL_TF = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Augmentation for multi-view gallery
AUG_TF = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_gallery_images(img_root, n_aug=8):
    """Load original + alternative images, return dict {cls: [PIL, ...]}"""
    images = {}
    for c in CLS_LIST:
        orig = Image.open(os.path.join(img_root, f"{c:02d}.png")).convert("RGB")
        alt_candidates = [f for f in os.listdir(img_root)
                          if f.startswith(f"{c:02d}_") and f.endswith(".png")]
        imgs = [orig]
        for af in sorted(alt_candidates):
            imgs.append(Image.open(os.path.join(img_root, af)).convert("RGB"))
        images[c] = imgs
    return images


def build_gallery_features(dino, images, device, n_aug=8, mode="single"):
    """
    mode: 'single'  - 1 image per class (original only)
          'dual'    - original + alternative (2 per class)
          'augmented' - original × n_aug random augmentations per class
    Returns: (gallery_feats, gallery_labels) — normalized DINO features
    """
    feats_list, labels_list = [], []
    to_tensor = T.ToTensor()

    with torch.no_grad():
        for c in CLS_LIST:
            img_list = images[c]
            if mode == "single":
                img_list = [img_list[0]]
            elif mode == "dual":
                pass  # use all (2 if available)
            # augmented: create n_aug views of the first image
            if mode == "augmented":
                base = to_tensor(img_list[0])
                aug_imgs = []
                for _ in range(n_aug):
                    aug_imgs.append(AUG_TF(base.clone()))
                batch = torch.stack(aug_imgs).to(device)
            else:
                batch = []
                for img in img_list:
                    t = DINO_EVAL_TF(to_tensor(img))
                    batch.append(t)
                batch = torch.stack(batch).to(device)

            f = F.normalize(dino(batch), dim=-1)  # (k, dim)
            feats_list.append(f)
            labels_list.extend([c - 1] * len(f))

    gallery_feats  = torch.cat(feats_list, dim=0)   # (N_gallery, dim)
    gallery_labels = torch.tensor(labels_list, device=device)
    return gallery_feats, gallery_labels


def eval_with_gallery(unet, eeg_enc, test_loader, vae, dino,
                       gallery_feats, gallery_labels, acp, num_timesteps, device):
    """Generate images from EEG, retrieve from gallery, return Top-1/3/5 and collapse stats."""
    correct = {1: 0, 3: 0, 5: 0}
    pred_classes = []
    total = 0
    subj = torch.zeros(1, dtype=torch.long, device=device)
    unet.eval(); eeg_enc.eval()

    with torch.no_grad():
        for eeg, _, lbl in test_loader:
            eeg = eeg.to(device)
            lbl = lbl.to(device)
            eeg_lat = eeg_enc.encode_eeg(eeg, subj.expand(eeg.size(0)))
            gen = sample_ddim(unet, eeg_lat, acp, num_timesteps, steps=50, device=device)
            decoded = vae.decode(gen / VAE_SCALE).sample.clamp(-1, 1)
            decoded = (decoded + 1) / 2

            # DINO features of generated images
            imgs = torch.stack([DINO_EVAL_TF(x) for x in decoded])
            gen_feats = F.normalize(dino(imgs), dim=-1)  # (B, dim)

            # Similarity to gallery
            sims = gen_feats @ gallery_feats.T  # (B, N_gallery)

            # For each generated image: aggregate scores per class
            n_gallery = gallery_feats.size(0)
            for b in range(gen_feats.size(0)):
                s = sims[b]  # (N_gallery,)
                # Max-sim per class
                class_scores = torch.full((N_CLASSES,), -1.0, device=device)
                for gc_idx in range(n_gallery):
                    cls_i = gallery_labels[gc_idx].item()
                    if s[gc_idx] > class_scores[cls_i]:
                        class_scores[cls_i] = s[gc_idx]
                pred_class = class_scores.argmax().item()
                pred_classes.append(pred_class)

            for k in correct:
                class_scores_batch = torch.zeros(gen_feats.size(0), N_CLASSES, device=device)
                for gc_idx in range(n_gallery):
                    cls_i = gallery_labels[gc_idx].item()
                    class_scores_batch[:, cls_i] = torch.max(class_scores_batch[:, cls_i], sims[:, gc_idx])
                topk = class_scores_batch.topk(min(k, N_CLASSES), dim=1).indices
                correct[k] += topk.eq(lbl.unsqueeze(1)).any(1).sum().item()
            total += eeg.size(0)

    cnt = Counter(pred_classes)
    dominant = cnt.most_common(1)[0][1] / max(len(pred_classes), 1)
    counts = np.array(list(cnt.values()), dtype=float)
    entropy = float(-np.sum((counts / counts.sum()) * np.log(counts / counts.sum() + 1e-9)))
    return {
        "top1": correct[1] / max(total, 1),
        "top3": correct[3] / max(total, 1),
        "top5": correct[5] / max(total, 1),
        "dominant": dominant,
        "entropy": entropy,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir",   default="checkpoints_vsre_latent_gen/20260610_105755_ch32_ep300_supcon_frozen_ca")
    parser.add_argument("--data_root",  default="./preproc_vs_re")
    parser.add_argument("--img_root",   default="./preproc_data_vi/images")
    parser.add_argument("--subject_ids",default="18,1")
    parser.add_argument("--n_ch",       type=int, default=32)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--n_aug",      type=int, default=8, help="Augmented views per class for 'augmented' gallery")
    parser.add_argument("--dino_model", default="dinov2_vits14")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _hub_dir = os.path.expanduser("~/.cache/torch/hub")
    print("[INFO] Loading DINO...", flush=True)
    dino = torch.hub.load(os.path.join(_hub_dir, "facebookresearch_dinov2_main"),
                          args.dino_model, source="local", verbose=False)
    dino = dino.to(device).eval()
    for p in dino.parameters(): p.requires_grad_(False)

    print("[INFO] Loading SD VAE...", flush=True)
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    for p in vae.parameters(): p.requires_grad_(False)

    _, _, acp = make_schedule(200, device)

    # Load gallery images
    images = load_gallery_images(args.img_root, args.n_aug)
    print(f"[INFO] Gallery images per class: { {c: len(v) for c, v in images.items()} }", flush=True)

    # Build galleries
    galleries = {}
    for mode in ["single", "dual", "augmented"]:
        gf, gl = build_gallery_features(dino, images, device, args.n_aug, mode)
        galleries[mode] = (gf, gl)
        print(f"  Gallery '{mode}': {gf.size(0)} images total", flush=True)

    subject_ids = [int(x.strip()) for x in args.subject_ids.split(",")]
    all_results = []

    for sid in subject_ids:
        ckpt_path = os.path.join(args.ckpt_dir, f"subj{sid:02d}_best.pt")
        if not os.path.isfile(ckpt_path):
            print(f"  [SKIP] S{sid:02d}: checkpoint not found", flush=True)
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        eeg_enc = build_eeg_encoder(args.n_ch, DINO_DIM[args.dino_model],
                                     type('a', (), {'eeg_occipital_ids':'auto'})(), device)
        eeg_enc.load_state_dict(ckpt["eeg_enc"])
        eeg_enc.eval()

        unet = LatentUNetCA(in_ch=4, base_ch=128, ch_mult=(1,2,4),
                            time_dim=256, eeg_dim=LATENT_DIM, n_heads=4, n_eeg_tokens=8).to(device)
        unet.load_state_dict(ckpt["unet"])
        unet.eval()

        test_ds = VSReDataset(args.data_root, [sid], n_ch=args.n_ch, split="test", seed=args.seed)
        loader  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn)

        print(f"\n=== S{sid:02d} (test n={len(test_ds)}) ===", flush=True)
        for mode, (gf, gl) in galleries.items():
            r = eval_with_gallery(unet, eeg_enc, loader, vae, dino, gf, gl, acp, 200, device)
            print(f"  Gallery '{mode:10s}': DINO@1={r['top1']:.4f} @3={r['top3']:.4f} @5={r['top5']:.4f}  dominant={r['dominant']*100:.1f}%  entropy={r['entropy']:.3f}", flush=True)
            all_results.append({"sid": sid, "gallery": mode, **r})

    out_csv = os.path.join(args.ckpt_dir, "exp39_multigallery_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sid","gallery","top1","top3","top5","dominant","entropy"])
        w.writeheader(); w.writerows(all_results)
    print(f"\n[INFO] Saved: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
