"""
eval_lora_ckpt.py
────────────────────────────────────────────────────────────────────────────
SD 1.5 LoRA 체크포인트 평가 (dual-gallery DINO retrieval).
train_vs_re_lora_gen.py 로 저장된 체크포인트에 사용.

Usage:
  python eval_lora_ckpt.py --ckpt_dir checkpoints_vsre_lora_gen/20260625_111012_lora_r16_ep100 --subject_ids 24
  python eval_lora_ckpt.py --ckpt_dir ... --subject_ids 1,18,24 --lora_r 16 --n_eeg_tokens 16
"""

import argparse, csv, os, sys
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_vs_re import VSReDataset, collate_fn, available_subjects
from model_eeg_dino import DINO_DIM, LATENT_DIM
from train_crosssubj_dino import set_seed, compute_class_prototypes
from train_vs_re_latent_gen import make_schedule, build_eeg_encoder, VAE_SCALE
from train_vs_re_lora_gen import (
    load_sd15_unet_lora, EEGConditionProjector,
    sample_sd_ddim, DINO_EVAL_TF,
)

N_CLASSES = 9
CLS_LIST  = list(range(1, 10))

IMG_ALT_ROOT = "./preproc_data_vi/images_alt"  # alternative class images (if exists)


def build_dual_gallery(img_root, dino, device):
    """Build dual gallery: original + alternative image per class → (18, feat_dim)."""
    to_t = T.ToTensor()
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    resize = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224)])

    gallery_feats = []
    gallery_labels = []

    alt_root = os.path.join(os.path.dirname(img_root), "images_alt")
    has_alt = os.path.isdir(alt_root)

    for c in CLS_LIST:
        # Original image
        img = Image.open(os.path.join(img_root, f"{c:02d}.png")).convert("RGB")
        x = norm(resize(to_t(img))).unsqueeze(0).to(device)
        with torch.no_grad():
            gallery_feats.append(F.normalize(dino(x), dim=-1))
        gallery_labels.append(c - 1)

        # Alternative image (if available)
        alt_path = os.path.join(alt_root, f"{c:02d}.png") if has_alt else None
        if alt_path and os.path.isfile(alt_path):
            img2 = Image.open(alt_path).convert("RGB")
            x2 = norm(resize(to_t(img2))).unsqueeze(0).to(device)
            with torch.no_grad():
                gallery_feats.append(F.normalize(dino(x2), dim=-1))
            gallery_labels.append(c - 1)

    gallery_feats  = torch.cat(gallery_feats, dim=0)   # (N_gallery, feat_dim)
    gallery_labels = torch.tensor(gallery_labels)        # (N_gallery,)
    return gallery_feats, gallery_labels


@torch.no_grad()
def evaluate_dual_gallery(unet, cond_proj, eeg_enc, test_loader,
                           vae, dino, gallery_feats, gallery_labels,
                           proto_dino, acp, T_steps, device, ddim_steps=20):
    subj = torch.zeros(1, dtype=torch.long, device=device)
    unet.eval(); cond_proj.eval(); eeg_enc.eval()

    pred_classes, true_classes = [], []
    correct = {1: 0, 3: 0, 5: 0}
    total = 0

    for eeg, _, lbl in test_loader:
        eeg = eeg.to(device); lbl = lbl.to(device)
        eeg_lat = eeg_enc.encode_eeg(eeg, subj.expand(eeg.size(0)))
        gen_lat = sample_sd_ddim(unet, cond_proj, eeg_lat, acp, T_steps,
                                  steps=ddim_steps, device=device)
        if total % 20 == 0:
            print(f"    [{total}/{len(test_loader.dataset)}] done", flush=True)
        decoded = vae.decode(gen_lat / VAE_SCALE).sample.clamp(-1, 1)
        decoded = (decoded + 1) / 2
        imgs = torch.stack([DINO_EVAL_TF(x) for x in decoded])
        feats = F.normalize(dino(imgs), dim=-1)

        # Dual-gallery retrieval
        sims = feats @ gallery_feats.T  # (B, N_gallery)
        # Aggregate per class: max sim
        B = feats.size(0)
        cls_sims = torch.zeros(B, N_CLASSES, device=device)
        for gi, gl in enumerate(gallery_labels):
            cls_sims[:, gl] = torch.max(cls_sims[:, gl], sims[:, gi])

        preds = cls_sims.argmax(dim=-1).cpu().tolist()
        pred_classes.extend(preds)
        true_classes.extend(lbl.cpu().tolist())
        for k in correct:
            topk = cls_sims.topk(min(k, 9), dim=1).indices
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
        "n_test": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir",     required=True)
    parser.add_argument("--data_root",    default="./preproc_vs_re")
    parser.add_argument("--img_root",     default="./preproc_data_vi/images")
    parser.add_argument("--subject_ids",  default="24")
    parser.add_argument("--supcon_ckpt",  default="checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon")
    parser.add_argument("--n_ch",         type=int, default=32)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--lora_r",       type=int, default=16)
    parser.add_argument("--lora_alpha",   type=int, default=32)
    parser.add_argument("--n_eeg_tokens", type=int, default=16)
    parser.add_argument("--num_timesteps",type=int, default=1000)
    parser.add_argument("--ddim_steps",   type=int, default=20)
    parser.add_argument("--dino_model",   default="dinov2_vits14")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}  Eval: {args.ckpt_dir}", flush=True)

    subject_ids = []
    for tok in args.subject_ids.split(","):
        tok = tok.strip()
        subject_ids.append(int(tok))

    _hub_dir = os.path.expanduser("~/.cache/torch/hub")
    print("[INFO] Loading DINO...", flush=True)
    dino = torch.hub.load(os.path.join(_hub_dir, "facebookresearch_dinov2_main"),
                          args.dino_model, source="local", verbose=False)
    dino = dino.to(device).eval()
    for p in dino.parameters(): p.requires_grad_(False)
    dino_feat_dim = DINO_DIM[args.dino_model]

    print("[INFO] Loading SD VAE...", flush=True)
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    for p in vae.parameters(): p.requires_grad_(False)

    _, _, acp = make_schedule(args.num_timesteps, device)

    # Build dual gallery
    proto_dino = compute_class_prototypes(dino, args.img_root, CLS_LIST, device)
    gallery_feats, gallery_labels = build_dual_gallery(args.img_root, dino, device)
    n_gallery = gallery_feats.size(0)
    print(f"[INFO] Gallery size: {n_gallery} images ({n_gallery//N_CLASSES} per class)", flush=True)

    all_results = []
    for sid in subject_ids:
        ckpt_path = os.path.join(args.ckpt_dir, f"subj{sid:02d}_lora_best.pt")
        if not os.path.isfile(ckpt_path):
            print(f"  [SKIP] No checkpoint: {ckpt_path}", flush=True)
            continue

        print(f"\n  Evaluating S{sid:02d}...", flush=True)
        test_ds = VSReDataset(args.data_root, [sid], n_ch=args.n_ch, split="test", seed=args.seed)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=0, collate_fn=collate_fn)
        print(f"  test={len(test_ds)}", flush=True)

        # Build models
        eeg_enc = build_eeg_encoder(args.n_ch, dino_feat_dim,
                                     type('a', (), {'eeg_occipital_ids': 'auto'})(), device)
        enc_path = os.path.join(args.supcon_ckpt, f"subj{sid:02d}_best.pt")
        if os.path.isfile(enc_path):
            enc_ckpt = torch.load(enc_path, map_location=device, weights_only=False)
            key = "eeg_enc" if "eeg_enc" in enc_ckpt else "model"
            eeg_enc.load_state_dict(enc_ckpt[key])
        for p in eeg_enc.parameters(): p.requires_grad_(False)

        unet = load_sd15_unet_lora(args.lora_r, args.lora_alpha).to(device)
        cond_proj = EEGConditionProjector(eeg_dim=LATENT_DIM, sd_dim=768,
                                           n_tokens=args.n_eeg_tokens).to(device)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        for name, param in unet.named_parameters():
            if name in ckpt["unet_lora"]:
                param.data.copy_(ckpt["unet_lora"][name])
        cond_proj.load_state_dict(ckpt["cond_proj"])
        print(f"  Checkpoint loaded.", flush=True)

        diag = evaluate_dual_gallery(unet, cond_proj, eeg_enc, test_loader,
                                      vae, dino, gallery_feats, gallery_labels,
                                      proto_dino, acp, args.num_timesteps, device,
                                      ddim_steps=args.ddim_steps)
        print(f"  [S{sid:02d}] dual-gallery DINO@1={diag['top1']:.4f} @3={diag['top3']:.4f} "
              f"@5={diag['top5']:.4f}  entropy={diag['entropy']:.3f}  "
              f"dominant={diag['dominant']*100:.1f}%  n_test={diag['n_test']}", flush=True)
        all_results.append({"sid": sid, **diag})

        # Free GPU memory before next subject
        del unet, cond_proj, eeg_enc
        torch.cuda.empty_cache()

    out_csv = os.path.join(args.ckpt_dir, "results_dual_gallery.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sid","top1","top3","top5","dominant","entropy","n_test"])
        w.writeheader(); w.writerows(all_results)
    print(f"\n[INFO] Saved: {out_csv}", flush=True)

    print("\nSummary:")
    for r in all_results:
        print(f"  S{r['sid']:02d}: dual-gallery DINO@1={r['top1']:.4f}  @3={r['top3']:.4f}  dominant={r['dominant']*100:.1f}%  entropy={r['entropy']:.3f}")


if __name__ == "__main__":
    main()
