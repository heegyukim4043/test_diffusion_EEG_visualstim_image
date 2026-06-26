"""
eval_vs_re_exp25_vi_transfer.py
────────────────────────────────────────────────────────────────────────────
Exp025-VI: VS → VI zero-shot transfer readiness probe

Strategy (HUMAN_DIRECTIVE Priority 0 / Exp25-VI):
  1. Load best VS-trained SupCon DINO encoder (Exp23-B checkpoint)
  2. Apply frozen encoder to VI EEG data (zero-shot, no VI fine-tuning)
  3. Retrieve against 9 class DINO image prototypes
  4. Report VI Top-1/3/5 and diagnose failure mode

Key format difference:
  VS re-dataset : 1024 Hz, 2048 samples (0~2 sec)
  VI data       : scipy mat, X=(ch, time, trial), time=512 (~512 Hz, 1 sec)

The V2 encoder uses MeanPool over time → handles variable-length input.
Temporal kernel sizes (k=7/15/31) have 2× wider effective window on VI data
(512 Hz vs 1024 Hz), which is a known limitation of zero-shot transfer.

Usage:
  python eval_vs_re_exp25_vi_transfer.py
  python eval_vs_re_exp25_vi_transfer.py --ckpt_dir checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon
  python eval_vs_re_exp25_vi_transfer.py --subject_ids 1,2,18
"""

import os, sys, csv, argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_eeg_dino import EEGDINORegressor, load_dino_encoder, DINO_DIM, LATENT_DIM
from train_crosssubj_dino import set_seed, compute_class_prototypes

N_CLASSES = 9
CLS_LIST  = list(range(1, 10))


# ── VI data loader ───────────────────────────────────────────────────────────
class VIDataset(Dataset):
    """
    Load VI EEG from preproc_data_vi/subj_XX.mat.
    Format: X=(ch, time, trial), y=(trial,), labels 1~9.
    Returns: (eeg: ch×time float32, label: 0-based int)
    """
    def __init__(self, vi_root: str, sid: int, n_ch: int = 32,
                 split: str = "test", seed: int = 42,
                 target_time: int | None = None):
        path = os.path.join(vi_root, f"subj_{sid:02d}.mat")
        mat  = loadmat(path)
        X    = mat["X"].astype(np.float32)          # (ch, time, trial)
        y    = mat["y"].squeeze().astype(np.int64)  # (trial,) labels 1~9

        ch, time, n_trial = X.shape
        eeg = X[:n_ch, :, :]                        # (n_ch, time, trial)
        eeg = eeg.transpose(2, 0, 1)                # (trial, n_ch, time)

        self.orig_time = time
        if target_time is not None and target_time > 0 and target_time != time:
            eeg_t = torch.from_numpy(eeg)
            eeg_t = F.interpolate(
                eeg_t,
                size=target_time,
                mode="linear",
                align_corners=False,
            )
            eeg = eeg_t.numpy()
            time = target_time

        labels_0 = y - 1                            # 0-based

        # Stratified 80:10:10 split per class
        train_idx, val_idx, test_idx = [], [], []
        for cls in range(N_CLASSES):
            idx = np.where(labels_0 == cls)[0]
            rng = np.random.RandomState(seed + cls)
            rng.shuffle(idx)
            n      = len(idx)
            n_tr   = int(n * 0.8)
            n_val  = int(n * 0.1)
            train_idx.extend(idx[:n_tr])
            val_idx  .extend(idx[n_tr:n_tr + n_val])
            test_idx .extend(idx[n_tr + n_val:])

        sel = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
        self.eeg    = [torch.from_numpy(eeg[i]) for i in sel]
        self.labels = [int(labels_0[i]) for i in sel]
        self.n_time = time
        self.n_ch   = n_ch

    def __len__(self):   return len(self.eeg)
    def __getitem__(self, i): return self.eeg[i], self.labels[i]


def collate_vi(batch):
    eeg = torch.stack([b[0] for b in batch])
    lbl = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return eeg, lbl


# ── Load VS-trained model ────────────────────────────────────────────────────
def load_vs_model(ckpt_path: str, dino_feat_dim: int, n_ch: int, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})

    raw_occ = cfg.get("eeg_occipital_ids", "auto").strip().lower()
    if raw_occ == "auto":   occ_idx = None
    elif raw_occ == "none": occ_idx = []
    else: occ_idx = [int(x) for x in raw_occ.split(",") if x.strip()]

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


# ── Retrieval evaluation on VI data ──────────────────────────────────────────
@torch.no_grad()
def evaluate_vi_retrieval(model, loader, proto_dino, device, k_list=(1, 3, 5)):
    correct = {k: 0 for k in k_list}
    total   = 0
    subj_idx = torch.zeros(1, dtype=torch.long, device=device)  # single-subj model

    for eeg, lbl in loader:
        eeg = eeg.to(device)
        lbl = lbl.to(device)
        logits, _, _ = model.predict(eeg, subj_idx.expand(eeg.size(0)), proto_dino)
        for k in k_list:
            topk = logits.topk(min(k, N_CLASSES), dim=1).indices
            correct[k] += topk.eq(lbl.unsqueeze(1)).any(1).sum().item()
        total += eeg.size(0)

    return {k: correct[k] / max(total, 1) for k in k_list}, total


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir",    type=str,
        default="./checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon",
        help="Exp23-B SupCon checkpoint (best VS representation)")
    parser.add_argument("--vi_root",     type=str, default="./preproc_data_vi")
    parser.add_argument("--img_root",    type=str, default="./preproc_data_vi/images")
    parser.add_argument("--subject_ids", type=str, default="1,2,18",
        help="Subjects to evaluate (must have VS encoder checkpoint)")
    parser.add_argument("--n_ch",        type=int, default=32)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--dino_model",  type=str, default="dinov2_vits14")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--split",       type=str, default="test",
        choices=["train", "val", "test"])
    parser.add_argument("--target_time", type=int, default=0,
        help="If >0, interpolate VI EEG time axis to this length before evaluation. Use 2048 to match VS.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp025-VI] VS→VI zero-shot transfer  device={device}")
    print(f"[Exp025-VI] VS checkpoint: {args.ckpt_dir}")
    print(f"[Exp025-VI] VI data: {args.vi_root}")
    if args.target_time > 0:
        print(f"[Exp025-VI] VI temporal interpolation target_time={args.target_time}")

    subject_ids = [int(x) for x in args.subject_ids.split(",")]

    # DINO teacher + prototypes
    dino          = load_dino_encoder(args.dino_model, device)
    dino_feat_dim = DINO_DIM[args.dino_model]
    proto_dino    = compute_class_prototypes(dino, args.img_root, CLS_LIST, device)
    print(f"  proto_dino: {proto_dino.shape}")

    all_results = []
    for sid in subject_ids:
        print(f"\n{'='*60}")
        print(f"  Subject {sid:02d}", flush=True)

        # Check VS encoder exists
        ckpt_path = os.path.join(args.ckpt_dir, f"subj{sid:02d}_best.pt")
        if not os.path.isfile(ckpt_path):
            print(f"  [SKIP] VS encoder not found: {ckpt_path}")
            continue

        # Check VI data exists
        vi_path = os.path.join(args.vi_root, f"subj_{sid:02d}.mat")
        if not os.path.isfile(vi_path):
            print(f"  [SKIP] VI data not found: {vi_path}")
            continue

        # Load VS-trained encoder
        model, cfg = load_vs_model(ckpt_path, dino_feat_dim, args.n_ch, device)

        # Load VI EEG
        try:
            vi_ds = VIDataset(
                args.vi_root, sid, args.n_ch, args.split, args.seed,
                target_time=args.target_time if args.target_time > 0 else None,
            )
        except Exception as e:
            print(f"  [SKIP] VI load error: {e}")
            continue

        vi_loader = DataLoader(vi_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=0, collate_fn=collate_vi)

        print(
            f"  VI {args.split}={len(vi_ds)} trials  "
            f"time_pts={vi_ds.n_time} (orig={getattr(vi_ds, 'orig_time', vi_ds.n_time)})",
            flush=True,
        )
        print(f"  VS encoder: {cfg.get('encoder_type','v2')} "
              f"occipital={cfg.get('eeg_occipital_ids','auto')} "
              f"loss={cfg.get('loss_type','supcon')}")

        metrics, n_eval = evaluate_vi_retrieval(model, vi_loader, proto_dino, device)

        print(f"  [VI S{sid:02d}]  Top-1={metrics[1]:.4f}  Top-3={metrics[3]:.4f}  "
              f"Top-5={metrics[5]:.4f}  (n={n_eval}  random={1/N_CLASSES:.4f})")

        all_results.append({"sid": sid, **metrics, "n": n_eval})

    # ── Summary ──────────────────────────────────────────────────────────────
    if not all_results:
        print("\n[ERROR] No subjects evaluated.")
        return

    print(f"\n{'='*60}")
    print(f"  Exp025-VI: VS→VI Zero-Shot Transfer Summary")
    print(f"  {'Subj':>5}  {'VI Top-1':>9}  {'VI Top-3':>9}  {'VI Top-5':>9}  {'N':>6}")
    print(f"  {'-'*48}")
    for r in all_results:
        print(f"  S{r['sid']:02d}    {r[1]:>9.4f}  {r[3]:>9.4f}  {r[5]:>9.4f}  {r['n']:>6}")

    if len(all_results) > 1:
        m1 = np.mean([r[1] for r in all_results])
        m3 = np.mean([r[3] for r in all_results])
        m5 = np.mean([r[5] for r in all_results])
        print(f"  {'-'*48}")
        print(f"  Mean   {m1:>9.4f}  {m3:>9.4f}  {m5:>9.4f}")
    else:
        m1 = all_results[0][1]; m3 = all_results[0][3]; m5 = all_results[0][5]

    print(f"  Random {1/N_CLASSES:>9.4f}")
    print(f"\n  [vs VS Exp23-B]  VS mean Top-1=0.3333  VI mean Top-1={m1:.4f}")
    print(f"  Transfer gap: {0.3333 - m1:+.4f}")

    above_chance = m1 > (1 / N_CLASSES)
    print(f"\n  Diagnosis:")
    print(f"    Above chance (>{1/N_CLASSES:.4f}): {'YES' if above_chance else 'NO'}")
    if m1 > 0.25:
        print(f"    Transfer quality: GOOD — encoder generalizes to VI imagery")
    elif m1 > 1/N_CLASSES:
        print(f"    Transfer quality: PARTIAL — some signal, but VS/VI distribution gap exists")
    else:
        print(f"    Transfer quality: FAILED — at or below chance")
        print(f"    Likely causes: VS/VI distribution gap, temporal resolution mismatch (512 vs 2048 samples),")
        print(f"                   or subject-specific model not generalizing across paradigms")

    # CSV
    suffix = f"_t{args.target_time}" if args.target_time > 0 else ""
    out_csv = os.path.join(args.ckpt_dir, f"exp025_vi_transfer_results{suffix}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "vi_top1", "vi_top3", "vi_top5", "n_eval"])
        for r in all_results:
            w.writerow([f"S{r['sid']:02d}", r[1], r[3], r[5], r["n"]])
        if len(all_results) > 1:
            w.writerow(["Mean", round(m1,4), round(m3,4), round(m5,4), ""])
    print(f"\n[INFO] Saved: {out_csv}")


if __name__ == "__main__":
    main()
