"""
eval_vs_re_exp22_tta.py
────────────────────────────────────────────────────────────────────────────
Exp022: Test-Time Augmentation (TTA) evaluation

Strategy (HUMAN_DIRECTIVE Exp22):
  - No retraining
  - Use best current DINO encoder baseline (Exp010: V2+auto-prior)
    checkpoint: 20260427_095215_ch32_merged_ep200
  - Apply TTA: small noise + time shift only, average N_TTA latents
  - Compare TTA vs no-TTA retrieval (Top-1/3/5)
  - Re-run Stage 2 readout if retrieval improves

TTA augmentations (inference-safe):
  1. Gaussian noise (noise_std=0.03, relative to signal std)
  2. Time shift (circular, max ±10 samples ~10ms @1024Hz)
  No freq noise / channel dropout (too destructive at inference)

Usage:
  python eval_vs_re_exp22_tta.py
  python eval_vs_re_exp22_tta.py --ckpt_dir checkpoints_vsre_dino/20260427_095215_ch32_merged_ep200
  python eval_vs_re_exp22_tta.py --n_tta 8 --subject_ids 1,2,18
"""

import os, sys, csv, random, argparse
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
IMG_SIZE  = 128


# ── TTA augmentation (noise + shift only) ───────────────────────────────────
def tta_augment(eeg: torch.Tensor, noise_std: float = 0.03, max_shift: int = 10) -> torch.Tensor:
    """Single TTA augmentation pass: small Gaussian noise + circular time shift.

    eeg: (B, ch, T)  —  returns augmented copy on same device.
    """
    x = eeg.clone()
    # Gaussian noise relative to per-trial signal std
    sig_std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    x = x + torch.randn_like(x) * sig_std * noise_std
    # Circular time shift
    shift = random.randint(-max_shift, max_shift)
    if shift != 0:
        x = torch.roll(x, shift, dims=-1)
    return x


# ── Load class stimulus images ───────────────────────────────────────────────
def load_class_images(img_root, size=128):
    tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])
    imgs = []
    for c in CLS_LIST:
        p = os.path.join(img_root, f"{c:02d}.png")
        imgs.append(tf(Image.open(p).convert("RGB")))
    return torch.stack(imgs)   # (9, 3, H, W)


# ── Per-sample evaluation: no-TTA vs TTA ────────────────────────────────────
@torch.no_grad()
def evaluate_subject_tta(
    model, test_loader, proto_dino, class_imgs, device,
    n_tta: int = 8, noise_std: float = 0.03, max_shift: int = 10,
    lpips_fn=None, k_list=(1, 3, 5),
):
    """
    Evaluates a subject with and without TTA in a single pass.

    Returns (no_tta_metrics, tta_metrics), each with Top-1/3/5, ssim, lpips.
    """
    model.eval()

    correct_base = {k: 0 for k in k_list}
    correct_tta  = {k: 0 for k in k_list}
    total = 0

    ssim_base  = 0.0;  ssim_tta  = 0.0
    lpips_base = 0.0;  lpips_tta = 0.0

    for eeg, subj, lbl in test_loader:
        eeg  = eeg.to(device)
        subj = subj.to(device)
        lbl  = lbl.to(device)
        B    = eeg.size(0)

        proto_lat = model.encode_img(proto_dino)  # (n_cls, 512)

        # ── no-TTA: single forward ──────────────────────────────────────────
        lat_base = model.encode_eeg(eeg, subj)    # (B, 512), already L2-normed
        temp     = model.log_temp.exp().clamp(0.01, 1.0)
        logits_base = lat_base @ proto_lat.T / temp

        # ── TTA: average N_TTA augmented latents ───────────────────────────
        lat_sum = torch.zeros_like(lat_base)
        for _ in range(n_tta):
            eeg_aug = tta_augment(eeg, noise_std=noise_std, max_shift=max_shift)
            lat_aug = model.encode_eeg(eeg_aug, subj)   # L2-normed
            lat_sum = lat_sum + lat_aug
        lat_avg     = F.normalize(lat_sum, dim=1)        # re-normalize average
        logits_tta  = lat_avg @ proto_lat.T / temp

        # ── Retrieval accuracy ─────────────────────────────────────────────
        for k in k_list:
            topk_base = logits_base.topk(min(k, N_CLASSES), dim=1).indices
            topk_tta  = logits_tta .topk(min(k, N_CLASSES), dim=1).indices
            correct_base[k] += topk_base.eq(lbl.unsqueeze(1)).any(1).sum().item()
            correct_tta [k] += topk_tta .eq(lbl.unsqueeze(1)).any(1).sum().item()
        total += B

        # ── Image readout metrics ──────────────────────────────────────────
        pred_base = logits_base.argmax(1)
        pred_tta  = logits_tta .argmax(1)

        for i in range(B):
            t_cls   = lbl[i].item()
            true_img = class_imgs[t_cls]

            for pred, mode in [
                (pred_base[i].item(), "base"), (pred_tta[i].item(), "tta")
            ]:
                pred_img = class_imgs[pred]

                s = 0.0
                if _SSIM_OK:
                    pi = pred_img.numpy().transpose(1, 2, 0)
                    ti = true_img.numpy().transpose(1, 2, 0)
                    s  = ssim_fn(pi, ti, data_range=1.0, channel_axis=2)

                lv = 0.0
                if lpips_fn is not None:
                    pi_t = (pred_img.unsqueeze(0) * 2 - 1).to(device)
                    ti_t = (true_img.unsqueeze(0) * 2 - 1).to(device)
                    lv   = lpips_fn(pi_t, ti_t).item()

                if mode == "base":
                    ssim_base  += s;  lpips_base += lv
                else:
                    ssim_tta   += s;  lpips_tta  += lv

    n = max(total, 1)
    base_metrics = {k: correct_base[k] / n for k in k_list}
    base_metrics["ssim"]  = ssim_base  / n
    base_metrics["lpips"] = lpips_base / n
    base_metrics["total"] = total

    tta_metrics = {k: correct_tta[k] / n for k in k_list}
    tta_metrics["ssim"]  = ssim_tta  / n
    tta_metrics["lpips"] = lpips_tta / n
    tta_metrics["total"] = total

    return base_metrics, tta_metrics


# ── Load model from checkpoint ──────────────────────────────────────────────
def load_subject_model(ckpt_path, dino_feat_dim, n_ch, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})

    raw_occ = cfg.get("eeg_occipital_ids", "auto").strip().lower()
    if raw_occ == "auto":
        occ_idx = None
    elif raw_occ == "none":
        occ_idx = []
    else:
        occ_idx = [int(x) for x in raw_occ.split(",") if x.strip()]

    model = EEGDINORegressor(
        eeg_channels=cfg.get("n_ch", n_ch),
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
    return model, cfg


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir",    type=str,
        default="./checkpoints_vsre_dino/20260427_095215_ch32_merged_ep200",
        help="Exp010 checkpoint directory (best DINO baseline)")
    parser.add_argument("--data_root",   type=str, default="./preproc_vs_re")
    parser.add_argument("--img_root",    type=str, default="./preproc_data_vi/images")
    parser.add_argument("--subject_ids", type=str, default="1,2,18")
    parser.add_argument("--n_ch",        type=int, default=32)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--dino_model",  type=str, default="dinov2_vits14")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--n_tta",       type=int, default=8,
        help="Number of TTA augmentation passes (5~10 recommended)")
    parser.add_argument("--tta_noise_std", type=float, default=0.03,
        help="TTA Gaussian noise std (relative to signal std)")
    parser.add_argument("--tta_max_shift", type=int, default=10,
        help="TTA max circular time shift in samples (~10ms @1024Hz)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp022] TTA evaluation  device={device}")
    print(f"[Exp022] Checkpoint: {args.ckpt_dir}")
    print(f"[Exp022] N_TTA={args.n_tta}  noise_std={args.tta_noise_std}  max_shift={args.tta_max_shift}")

    subject_ids = [int(x) for x in args.subject_ids.split(",")]
    sc = session_counts(args.data_root)

    # DINO teacher + prototypes
    dino          = load_dino_encoder(args.dino_model, device)
    dino_feat_dim = DINO_DIM[args.dino_model]
    proto_dino    = compute_class_prototypes(dino, args.img_root, CLS_LIST, device)
    print(f"  proto_dino: {proto_dino.shape}")

    # Class images for Stage 2 readout
    class_imgs = load_class_images(args.img_root, size=IMG_SIZE)
    print(f"  class_imgs: {class_imgs.shape}")

    # LPIPS
    lpips_fn = None
    if _LPIPS_OK:
        lpips_fn = lpips_lib.LPIPS(net="vgg").to(device)
        lpips_fn.eval()
        print("  LPIPS: enabled (vgg)")
    else:
        print("  LPIPS: disabled (pip install lpips)")

    all_base = []
    all_tta  = []

    for sid in subject_ids:
        print(f"\n{'='*60}")
        print(f"  Subject {sid:02d}  (sessions={sc.get(sid,0)})", flush=True)

        ckpt_path = os.path.join(args.ckpt_dir, f"subj{sid:02d}_best.pt")
        if not os.path.isfile(ckpt_path):
            print(f"  [SKIP] checkpoint not found: {ckpt_path}")
            continue

        model, cfg = load_subject_model(ckpt_path, dino_feat_dim, args.n_ch, device)

        subj_map = {sid: 0}
        test_ds  = VSReDataset(args.data_root, [sid], subj_map, args.n_ch,
                               "test", args.seed)
        if len(test_ds) == 0:
            print(f"  [SKIP] no test data")
            continue

        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0, collate_fn=collate_fn)

        n_sessions_effective = sc.get(sid, 0)
        print(f"  test={len(test_ds)} trials  loaded_sessions={n_sessions_effective}",
              flush=True)

        base_m, tta_m = evaluate_subject_tta(
            model, test_loader, proto_dino, class_imgs, device,
            n_tta=args.n_tta, noise_std=args.tta_noise_std,
            max_shift=args.tta_max_shift, lpips_fn=lpips_fn,
        )

        delta1 = tta_m[1] - base_m[1]
        print(f"  [Base S{sid:02d}]  Top-1={base_m[1]:.4f}  Top-3={base_m[3]:.4f}  "
              f"Top-5={base_m[5]:.4f}  SSIM={base_m['ssim']:.4f}  LPIPS={base_m['lpips']:.4f}")
        print(f"  [TTA  S{sid:02d}]  Top-1={tta_m[1]:.4f}  Top-3={tta_m[3]:.4f}  "
              f"Top-5={tta_m[5]:.4f}  SSIM={tta_m['ssim']:.4f}  LPIPS={tta_m['lpips']:.4f}  "
              f"(Δ Top-1 {delta1:+.4f})")

        all_base.append({"sid": sid, **base_m})
        all_tta .append({"sid": sid, **tta_m})

    # ── Summary ─────────────────────────────────────────────────────────────
    if not all_base:
        print("\n[ERROR] No subjects evaluated.")
        return

    print(f"\n{'='*60}")
    print(f"  Exp022 TTA Summary  (N_TTA={args.n_tta})")
    print(f"  {'Subj':>5}  {'Base T1':>8}  {'TTA T1':>8}  {'Δ T1':>8}  "
          f"{'Base T3':>8}  {'TTA T3':>8}  {'Base T5':>8}  {'TTA T5':>8}")
    print(f"  {'-'*72}")
    for b, t in zip(all_base, all_tta):
        print(f"  S{b['sid']:02d}    {b[1]:>8.4f}  {t[1]:>8.4f}  {t[1]-b[1]:>+8.4f}  "
              f"{b[3]:>8.4f}  {t[3]:>8.4f}  {b[5]:>8.4f}  {t[5]:>8.4f}")

    bm1 = np.mean([r[1] for r in all_base])
    bm3 = np.mean([r[3] for r in all_base])
    bm5 = np.mean([r[5] for r in all_base])
    tm1 = np.mean([r[1] for r in all_tta ])
    tm3 = np.mean([r[3] for r in all_tta ])
    tm5 = np.mean([r[5] for r in all_tta ])
    bss = np.mean([r["ssim"]  for r in all_base])
    tss = np.mean([r["ssim"]  for r in all_tta ])
    blp = np.mean([r["lpips"] for r in all_base])
    tlp = np.mean([r["lpips"] for r in all_tta ])
    print(f"  {'-'*72}")
    print(f"  Mean   {bm1:>8.4f}  {tm1:>8.4f}  {tm1-bm1:>+8.4f}  "
          f"{bm3:>8.4f}  {tm3:>8.4f}  {bm5:>8.4f}  {tm5:>8.4f}")
    print(f"\n  SSIM   Base={bss:.4f}  TTA={tss:.4f}  Δ={tss-bss:+.4f}")
    print(f"  LPIPS  Base={blp:.4f}  TTA={tlp:.4f}  Δ={tlp-blp:+.4f}")
    print(f"\n  [vs Exp010 Top-1=0.2824]  Base={bm1:.4f}  TTA={tm1:.4f}")
    print(f"  [vs Exp021 Top-1=0.2725]  Base={bm1:.4f}  TTA={tm1:.4f}")

    tta_improved = (tm1 > bm1)
    print(f"\n  TTA improved Top-1: {'YES (+' + f'{tm1-bm1:.4f})' if tta_improved else 'NO (' + f'{tm1-bm1:+.4f})'}")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    out_csv = os.path.join(args.ckpt_dir, "exp022_tta_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "mode", "top1", "top3", "top5", "ssim", "lpips"])
        for b, t in zip(all_base, all_tta):
            sid = b["sid"]
            w.writerow([f"S{sid:02d}", "base", b[1], b[3], b[5], b["ssim"], b["lpips"]])
            w.writerow([f"S{sid:02d}", "tta",  t[1], t[3], t[5], t["ssim"], t["lpips"]])
        w.writerow(["Mean", "base", round(bm1,4), round(bm3,4), round(bm5,4),
                    round(bss,4), round(blp,4)])
        w.writerow(["Mean", "tta",  round(tm1,4), round(tm3,4), round(tm5,4),
                    round(tss,4), round(tlp,4)])
    print(f"\n[INFO] Saved: {out_csv}")

    if tta_improved:
        print(f"\n[INFO] TTA improved retrieval → Stage 2 readout re-run recommended.")
        print(f"       Run: python eval_vs_re_stage2_readout.py --ckpt_dir {args.ckpt_dir}")
    else:
        print(f"\n[INFO] TTA did not improve Top-1.  Stage 2 readout re-run skipped.")
        print(f"       Exp021 Stage 2 result stands as the current best (Top-1=0.2725).")


if __name__ == "__main__":
    main()
