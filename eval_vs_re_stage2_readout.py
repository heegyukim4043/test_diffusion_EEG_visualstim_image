"""
eval_vs_re_stage2_readout.py
────────────────────────────────────────────────────────────────────────────
Exp021: Stage 2 latent/readout evaluation

Strategy (HUMAN_DIRECTIVE Block E):
  1. EEG -> frozen Exp010 DINO encoder -> 512-dim latent
  2. latent -> nearest-neighbor DINO prototype -> predicted class
  3. nearest-neighbor image readout (return class stimulus image)
  4. evaluate: DINO@1/3/5, SSIM, LPIPS vs ground-truth class image
  5. compare vs gen probe (Exp018 best: DINO@1=0.1270)

Key point: Stage 2 bypasses pixel-space diffusion collapse entirely.
Expected: DINO@1 ≈ Exp010 Top-1 (0.2824) >> gen probe (0.1270)

Usage:
  python eval_vs_re_stage2_readout.py --ckpt_dir checkpoints_vsre_dino/20260427_095215_ch32_merged_ep200
"""

import os, sys, csv, argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_vs_re import VSReDataset, collate_fn, session_counts
from model_eeg_dino import EEGDINORegressor, load_dino_encoder, DINO_DIM, LATENT_DIM
from train_crosssubj_dino import set_seed, compute_class_prototypes, dino_transform

try:
    import lpips as lpips_lib
    _LPIPS_OK = True
except ImportError:
    _LPIPS_OK = False

try:
    from skimage.metrics import structural_similarity as ssim_fn
    _SSIM_OK = True
except ImportError:
    _SSIM_OK = False

N_CLASSES = 9
CLS_LIST  = list(range(1, 10))
IMG_SIZE  = 128   # resize to match gen probe output size


def load_class_images(img_root, size=128):
    """Load 9 class stimulus images as tensors. Returns (9, 3, H, W) float32 [0,1]."""
    tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])
    imgs = []
    for c in CLS_LIST:
        p = os.path.join(img_root, f"{c:02d}.png")
        imgs.append(tf(Image.open(p).convert("RGB")))
    return torch.stack(imgs)   # (9, 3, H, W)


@torch.no_grad()
def evaluate_subject(model, test_loader, proto_dino, class_imgs, device,
                     lpips_fn=None, k_list=(1, 3, 5)):
    """
    For each test EEG:
      1. Encode EEG -> latent
      2. NN retrieval -> predicted class
      3. Retrieve class image
      4. Compute SSIM / LPIPS vs true class image

    Returns metrics dict.
    """
    model.eval()
    correct = {k: 0 for k in k_list}
    total   = 0
    ssim_sum  = 0.0
    lpips_sum = 0.0

    for eeg, subj, lbl in test_loader:
        eeg  = eeg.to(device)
        subj = subj.to(device)
        lbl  = lbl.to(device)

        logits, pred_cls, eeg_lat = model.predict(eeg, subj, proto_dino)

        for k in k_list:
            topk = logits.topk(min(k, logits.size(1)), dim=1).indices
            correct[k] += topk.eq(lbl.unsqueeze(1)).any(1).sum().item()
        total += eeg.size(0)

        # Image readout: retrieve predicted class image
        for i in range(eeg.size(0)):
            p_cls  = pred_cls[i].item()   # predicted class index (0-based)
            t_cls  = lbl[i].item()        # true class index (0-based)
            pred_img = class_imgs[p_cls]   # (3, H, W) tensor [0,1]
            true_img = class_imgs[t_cls]

            # SSIM
            if _SSIM_OK:
                pi = pred_img.numpy().transpose(1,2,0)
                ti = true_img.numpy().transpose(1,2,0)
                s  = ssim_fn(pi, ti, data_range=1.0, channel_axis=2)
                ssim_sum += s

            # LPIPS
            if lpips_fn is not None:
                pi_t = (pred_img.unsqueeze(0) * 2 - 1).to(device)
                ti_t = (true_img.unsqueeze(0) * 2 - 1).to(device)
                lpips_sum += lpips_fn(pi_t, ti_t).item()

    metrics = {k: correct[k] / max(total, 1) for k in k_list}
    metrics["ssim"]  = ssim_sum  / max(total, 1)
    metrics["lpips"] = lpips_sum / max(total, 1)
    metrics["total"] = total
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir",    type=str,
        default="./checkpoints_vsre_dino/20260427_095215_ch32_merged_ep200",
        help="Exp010 checkpoint directory")
    parser.add_argument("--data_root",   type=str, default="./preproc_vs_re")
    parser.add_argument("--img_root",    type=str, default="./preproc_data_vi/images")
    parser.add_argument("--subject_ids", type=str, default="1,2,18")
    parser.add_argument("--n_ch",        type=int, default=32)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--dino_model",  type=str, default="dinov2_vits14")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Stage 2 readout eval  device={device}")
    print(f"[INFO] Checkpoint: {args.ckpt_dir}")

    subject_ids = [int(x) for x in args.subject_ids.split(",")]
    sc = session_counts(args.data_root)

    # DINO + prototypes
    dino          = load_dino_encoder(args.dino_model, device)
    dino_feat_dim = DINO_DIM[args.dino_model]
    proto_dino    = compute_class_prototypes(dino, args.img_root, CLS_LIST, device)
    print(f"  proto_dino: {proto_dino.shape}")

    # Class images for readout
    class_imgs = load_class_images(args.img_root, size=IMG_SIZE)  # (9,3,H,W)
    print(f"  class_imgs: {class_imgs.shape}")

    # LPIPS
    lpips_fn = None
    if _LPIPS_OK:
        lpips_fn = lpips_lib.LPIPS(net="vgg").to(device)
        lpips_fn.eval()
        print("  LPIPS: enabled (vgg)")
    else:
        print("  LPIPS: disabled (pip install lpips)")

    # Per-subject evaluation
    all_results = []
    for sid in subject_ids:
        print(f"\n{'='*55}")
        print(f"  Subject {sid:02d}  (sessions={sc.get(sid,0)})", flush=True)

        ckpt_path = os.path.join(args.ckpt_dir, f"subj{sid:02d}_best.pt")
        if not os.path.isfile(ckpt_path):
            print(f"  [SKIP] checkpoint not found: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg  = ckpt.get("config", {})

        # Reconstruct model from saved config
        raw_occ = cfg.get("eeg_occipital_ids", "auto").strip().lower()
        if raw_occ == "auto":
            occ_idx = None
        elif raw_occ == "none":
            occ_idx = []
        else:
            occ_idx = [int(x) for x in raw_occ.split(",") if x.strip()]

        model = EEGDINORegressor(
            eeg_channels=cfg.get("n_ch", args.n_ch),
            n_subjects=1,
            dino_feat_dim=dino_feat_dim,
            latent_dim=LATENT_DIM,
            eeg_hidden=cfg.get("eeg_hidden", 256),
            eeg_out=cfg.get("eeg_out", 256),
            subj_emb_dim=cfg.get("subj_emb_dim", 32),
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 4),
            dropout=cfg.get("dropout", 0.1),
            temperature=cfg.get("temperature", 0.1),
            encoder_type=cfg.get("encoder_type", "v2"),
            n_classes=N_CLASSES,
            eeg_occipital_indices=occ_idx,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        subj_map = {sid: 0}
        test_ds  = VSReDataset(args.data_root, [sid], subj_map, args.n_ch, "test",
                               args.seed)
        if len(test_ds) == 0:
            print(f"  [SKIP] no test data")
            continue

        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0, collate_fn=collate_fn)

        print(f"  test={len(test_ds)} trials", flush=True)
        metrics = evaluate_subject(model, test_loader, proto_dino,
                                   class_imgs, device, lpips_fn)

        print(f"  [Stage2 S{sid:02d}]  "
              f"Top-1={metrics[1]:.4f}  Top-3={metrics[3]:.4f}  Top-5={metrics[5]:.4f}  "
              f"SSIM={metrics['ssim']:.4f}  LPIPS={metrics['lpips']:.4f}",
              flush=True)
        all_results.append({"sid": sid, **metrics})

    # Summary
    print(f"\n{'='*55}")
    print(f"  Stage 2 Readout Summary (Exp021)")
    print(f"  {'Subj':>5}  {'Top-1':>7}  {'Top-3':>7}  {'Top-5':>7}  {'SSIM':>7}  {'LPIPS':>7}")
    print(f"  {'-'*50}")
    for r in all_results:
        print(f"  S{r['sid']:02d}    {r[1]:>7.4f}  {r[3]:>7.4f}  {r[5]:>7.4f}  "
              f"{r['ssim']:>7.4f}  {r['lpips']:>7.4f}")

    if all_results:
        t1m = np.mean([r[1]    for r in all_results])
        t3m = np.mean([r[3]    for r in all_results])
        t5m = np.mean([r[5]    for r in all_results])
        sm  = np.mean([r["ssim"]  for r in all_results])
        lm  = np.mean([r["lpips"] for r in all_results])
        print(f"  {'-'*50}")
        print(f"  Mean   {t1m:>7.4f}  {t3m:>7.4f}  {t5m:>7.4f}  {sm:>7.4f}  {lm:>7.4f}")
        print(f"  Random {1/N_CLASSES:>7.4f}")
        print(f"\n  [vs Gen Probe Exp018]  Exp018 mean DINO@1=0.1138  (Stage2 = {t1m:.4f})")
        print(f"  [vs DINO Exp010]       Exp010 mean Top-1=0.2824   (Stage2 = {t1m:.4f})")

        # CSV
        out_csv = os.path.join(args.ckpt_dir, "stage2_readout_results.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subject","top1","top3","top5","ssim","lpips"])
            for r in all_results:
                w.writerow([f"S{r['sid']:02d}", r[1], r[3], r[5],
                            r["ssim"], r["lpips"]])
            w.writerow(["Mean", round(t1m,4), round(t3m,4), round(t5m,4),
                        round(sm,4), round(lm,4)])
        print(f"\n[INFO] Saved: {out_csv}")


if __name__ == "__main__":
    main()
