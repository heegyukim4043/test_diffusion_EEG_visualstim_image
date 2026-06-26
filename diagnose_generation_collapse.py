"""
diagnose_generation_collapse.py
────────────────────────────────────────────────────────────────────────────
HUMAN_DIRECTIVE Priority 1: Generation collapse diagnosis

Loads a trained generation checkpoint, runs DDIM sampling on the test set,
classifies generated images via DINOv2, and reports:
  - Per-class predicted histogram (generated DINO-predicted class distribution)
  - Confusion matrix (true label vs DINOv2-predicted class)
  - Shannon entropy of the predicted class distribution
  - Per-subject collapse pattern summary

Usage:
  python diagnose_generation_collapse.py \
      --ckpt_dir checkpoints_vsre_gen/20260417_113129_ch32_merged_ep300

  python diagnose_generation_collapse.py \
      --ckpt_dir checkpoints_vsre_gen/20260411_165536_ch32_merged_ep300 \
      --subject_ids all
"""

import os, sys, argparse, csv
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_vs_re import VSReDataset, collate_fn, available_subjects
from model_128_eegonly_transformer_repa import EEGDiffusionModel128
from model_eeg_dino import load_dino_encoder, DINO_DIM
from train_crosssubj_dino import compute_class_prototypes
from train_vs_re_gen import GenDataset, gen_collate


CLASS_NAMES = [f"cls{i+1}" for i in range(9)]
IMG_TRANSFORM = T.Compose([
    T.Resize(128), T.CenterCrop(128), T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINO_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> EEGDiffusionModel128:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    raw_occ = cfg.get("eeg_occipital_ids", "auto").strip().lower()
    if raw_occ in ("auto", ""):
        occ_idx = None
    elif raw_occ == "none":
        occ_idx = []
    else:
        occ_idx = [int(x) for x in raw_occ.split(",") if x.strip()]

    model = EEGDiffusionModel128(
        eeg_channels    = cfg.get("eeg_channels", cfg.get("n_ch", 32)),
        num_classes     = cfg.get("num_classes", 9),
        num_timesteps   = cfg.get("num_timesteps", 200),
        base_channels   = cfg.get("base_channels", 64),
        ch_mult         = tuple(int(x) for x in cfg.get("ch_mult", "1,2,4,4").split(","))
                          if isinstance(cfg.get("ch_mult"), str)
                          else tuple(cfg.get("ch_mult", [1, 2, 4, 4])),
        lambda_percept  = cfg.get("lambda_percept", 0.1),
        lambda_rec      = cfg.get("lambda_rec", 0.01),
        lambda_ssim     = cfg.get("lambda_ssim", 0.05),
        lambda_lpips    = cfg.get("lambda_lpips", 0.0),
        beta_schedule   = cfg.get("beta_schedule", "linear"),
        encoder_version = cfg.get("encoder_version", "v1"),
        eeg_stem_filters= cfg.get("eeg_stem_filters", 32),
        eeg_occipital_indices = cfg.get("eeg_occipital_indices", occ_idx),
    ).to(device)

    state = ckpt.get("ema_model", ckpt.get("model"))
    model.load_state_dict(state)
    model.eval()
    return model, cfg


@torch.no_grad()
def diagnose_subject(
    sid: int,
    ckpt_dir: str,
    args,
    dino,
    proto_dino: torch.Tensor,
    device: torch.device,
) -> dict | None:
    subj_ckpt = os.path.join(ckpt_dir, f"subj{sid:02d}", "best.pt")
    if not os.path.isfile(subj_ckpt):
        print(f"  [SKIP] S{sid:02d}: checkpoint not found at {subj_ckpt}")
        return None

    model, cfg = load_model_from_ckpt(subj_ckpt, device)

    subj_map = {sid: 0}
    base_test = VSReDataset(args.data_root, [sid], subj_map, args.n_ch,
                            "test", args.seed, cfg.get("max_sessions"))
    if len(base_test) == 0:
        print(f"  [SKIP] S{sid:02d}: no test data")
        return None

    test_ds = GenDataset(base_test, args.img_root)
    loader  = DataLoader(test_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=0, collate_fn=gen_collate)

    n_classes = 9
    true_labels  = []
    pred_classes = []

    ddim_steps     = cfg.get("eval_ddim_steps", args.eval_ddim_steps)
    guidance_scale = cfg.get("guidance_scale",  args.guidance_scale)
    eta            = cfg.get("eta",             args.eta)

    dino_mean = DINO_MEAN.to(device)
    dino_std  = DINO_STD.to(device)

    for eeg, lbl, _ in loader:
        eeg = eeg.to(device)
        lbl = lbl.to(device)

        gen = model.sample_ddim(eeg, num_steps=ddim_steps,
                                guidance_scale=guidance_scale, eta=eta)
        gen_01 = (gen.clamp(-1, 1) + 1.0) * 0.5
        gen_224 = F.interpolate(gen_01, size=(224, 224), mode="bilinear", align_corners=False)
        gen_224 = (gen_224 - dino_mean) / (dino_std + 1e-8)

        feat = F.normalize(dino(gen_224), dim=1)
        sim  = feat @ proto_dino.T          # (B, 9)
        pred = sim.argmax(dim=1)            # (B,)

        true_labels.extend(lbl.cpu().tolist())
        pred_classes.extend(pred.cpu().tolist())

    true_labels  = np.array(true_labels)
    pred_classes = np.array(pred_classes)
    n_total = len(true_labels)

    # ── 1. Predicted-class histogram ─────────────────────────────────────────
    hist = np.bincount(pred_classes, minlength=n_classes)
    hist_norm = hist / n_total

    # ── 2. Shannon entropy of predicted distribution ─────────────────────────
    p = hist_norm + 1e-12
    entropy = -np.sum(p * np.log(p)) / np.log(n_classes)   # normalized [0,1]
    max_entropy = np.log(n_classes)

    # ── 3. Confusion matrix ──────────────────────────────────────────────────
    conf_mat = np.zeros((n_classes, n_classes), dtype=int)
    for t, p_ in zip(true_labels, pred_classes):
        conf_mat[t, p_] += 1

    # ── 4. Top-1 accuracy from DINO classification ───────────────────────────
    top1 = np.mean(true_labels == pred_classes)

    dominant_class = hist.argmax()
    dominant_frac  = hist_norm[dominant_class]
    is_collapsed   = dominant_frac > 0.5    # >50% predicted as single class

    summary = {
        "sid":            sid,
        "n_total":        n_total,
        "dino_top1":      round(float(top1), 4),
        "entropy_norm":   round(float(entropy), 4),
        "dominant_class": int(dominant_class + 1),   # 1-based
        "dominant_frac":  round(float(dominant_frac), 4),
        "collapsed":      is_collapsed,
        "hist":           hist_norm.tolist(),
    }

    print(
        f"  S{sid:02d}  n={n_total:4d}  DINO@1={top1:.4f}  "
        f"entropy={entropy:.3f}  dominant=cls{dominant_class+1}({dominant_frac:.2%})  "
        f"{'[COLLAPSED]' if is_collapsed else '[OK]'}"
    )

    # ── Save outputs ─────────────────────────────────────────────────────────
    out_dir = os.path.join(ckpt_dir, f"subj{sid:02d}", "collapse_diag")
    os.makedirs(out_dir, exist_ok=True)

    # Histogram plot
    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.bar(range(1, n_classes + 1), hist_norm, color="steelblue", edgecolor="white")
    bars[dominant_class].set_color("tomato")
    ax.axhline(1.0 / n_classes, ls="--", color="gray", label="uniform (1/9)")
    ax.set_xlabel("Predicted class (DINOv2)")
    ax.set_ylabel("Fraction")
    ax.set_title(f"S{sid:02d}  entropy={entropy:.3f}  dominant=cls{dominant_class+1}({dominant_frac:.0%})")
    ax.set_xticks(range(1, n_classes + 1))
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pred_histogram.png"), dpi=120)
    plt.close()

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf_mat, cmap="Blues")
    ax.set_xticks(range(n_classes)); ax.set_xticklabels([f"p{i+1}" for i in range(n_classes)])
    ax.set_yticks(range(n_classes)); ax.set_yticklabels([f"t{i+1}" for i in range(n_classes)])
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(f"S{sid:02d} confusion (DINO top-1)")
    for r in range(n_classes):
        for c in range(n_classes):
            ax.text(c, r, str(conf_mat[r, c]), ha="center", va="center", fontsize=7,
                    color="white" if conf_mat[r, c] > conf_mat.max() * 0.5 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=120)
    plt.close()

    # Histogram CSV
    with open(os.path.join(out_dir, "pred_histogram.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "count", "fraction"])
        for c in range(n_classes):
            w.writerow([c + 1, int(hist[c]), f"{hist_norm[c]:.6f}"])

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir",       type=str, required=True,
                        help="Path to checkpoint dir, e.g. checkpoints_vsre_gen/20260417_113129_ch32_merged_ep300")
    parser.add_argument("--data_root",      type=str, default="./preproc_vs_re")
    parser.add_argument("--img_root",       type=str, default="./preproc_data_vi/images")
    parser.add_argument("--subject_ids",    type=str, default="1,2,18")
    parser.add_argument("--n_ch",           type=int, default=32)
    parser.add_argument("--batch_size",     type=int, default=16)
    parser.add_argument("--eval_ddim_steps",type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--eta",            type=float, default=0.0)
    parser.add_argument("--dino_model",     type=str, default="dinov2_vits14")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}  ckpt_dir={args.ckpt_dir}")

    all_sids = available_subjects(args.data_root)
    if args.subject_ids.strip().lower() == "all":
        subject_ids = all_sids
    else:
        subject_ids = [int(x) for x in args.subject_ids.split(",") if x.strip()]
        subject_ids = [s for s in subject_ids if s in all_sids]

    print(f"[INFO] Loading DINOv2 ({args.dino_model}) ...")
    dino = load_dino_encoder(args.dino_model).to(device)
    dino.eval()

    print("[INFO] Computing DINO class prototypes ...")
    proto_dino = compute_class_prototypes(
        dino, args.img_root, list(range(1, 10)), device
    ).to(device)    # (9, DINO_DIM)

    results = []
    for sid in subject_ids:
        print(f"\n── S{sid:02d} ─────────────────────────────────────")
        r = diagnose_subject(sid, args.ckpt_dir, args, dino, proto_dino, device)
        if r is not None:
            results.append(r)

    if not results:
        print("\n[WARN] No results collected.")
        return

    # ── Cross-subject summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COLLAPSE DIAGNOSIS SUMMARY")
    print("=" * 60)
    print(f"{'SID':<6} {'DINO@1':>7} {'Entropy':>8} {'Dominant':>10} {'Frac':>6} {'Status'}")
    print("-" * 60)
    for r in results:
        status = "[COLLAPSED]" if r["collapsed"] else "[OK]"
        print(f"S{r['sid']:02d}   {r['dino_top1']:>7.4f} {r['entropy_norm']:>8.3f} "
              f"  cls{r['dominant_class']:>2d}     {r['dominant_frac']:>5.1%} {status}")

    n_collapsed = sum(1 for r in results if r["collapsed"])
    mean_top1   = np.mean([r["dino_top1"]     for r in results])
    mean_ent    = np.mean([r["entropy_norm"]   for r in results])
    print("-" * 60)
    print(f"Mean         {mean_top1:>7.4f} {mean_ent:>8.3f}    collapsed={n_collapsed}/{len(results)}")
    print("=" * 60)

    # Save summary CSV
    summary_path = os.path.join(args.ckpt_dir, "collapse_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sid", "n_total", "dino_top1",
                                           "entropy_norm", "dominant_class",
                                           "dominant_frac", "collapsed"])
        w.writeheader()
        for r in results:
            w.writerow({k: v for k, v in r.items() if k != "hist"})
    print(f"\n[INFO] Summary saved: {summary_path}")

    # Collapse verdict
    print("\n[VERDICT]")
    if n_collapsed == 0:
        print("  No collapse detected. Generated-class distribution is spread across classes.")
        print("  Low DINO Top-1 is due to within-class misretrieval, not collapse.")
    elif n_collapsed == len(results):
        print("  FULL COLLAPSE: All subjects show dominant-class bias.")
        print("  → Add anti-collapse auxiliary loss (--lambda_dino_align / --lambda_aux_ce)")
    else:
        print(f"  PARTIAL COLLAPSE: {n_collapsed}/{len(results)} subjects collapsed.")
        print("  → Inspect per-subject histograms; add anti-collapse loss for collapsed subjects.")


if __name__ == "__main__":
    main()
