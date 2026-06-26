"""
Compare VS and VI EEG latent alignment to DINO class prototypes.

This diagnoses whether VS->VI failure comes from weak image/prototype alignment
in latent space rather than only reporting top-k retrieval.

Usage:
  python diagnose_vs_vi_latent_gap.py --subject_ids 1,2,18
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_vs_re import VSReDataset, collate_fn as collate_vs
from eval_vs_re_exp25_vi_transfer import VIDataset, collate_vi, load_vs_model
from model_eeg_dino import load_dino_encoder, DINO_DIM
from train_crosssubj_dino import set_seed, compute_class_prototypes


N_CLASSES = 9
CLS_LIST = list(range(1, 10))


def parse_subject_ids(raw):
    ids = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(tok))
    return ids


@torch.no_grad()
def collect_alignment_stats(model, loader, proto_lat, device, source):
    rows = []
    subj_idx = torch.zeros(1, dtype=torch.long, device=device)

    for batch in loader:
        if source == "vs":
            eeg, subj, lbl = batch
            subj = subj.to(device)
        else:
            eeg, lbl = batch
            subj = subj_idx.expand(eeg.size(0))

        eeg = eeg.to(device)
        lbl = lbl.to(device)
        lat = model.encode_eeg(eeg, subj)
        sims = lat @ proto_lat.T
        pred = sims.argmax(dim=1)

        for i in range(eeg.size(0)):
            y = int(lbl[i].item())
            sim_row = sims[i]
            true_sim = float(sim_row[y].item())
            wrong = torch.cat([sim_row[:y], sim_row[y + 1:]])
            max_wrong = float(wrong.max().item())
            margin = true_sim - max_wrong
            rows.append({
                "source": source,
                "label": y,
                "pred": int(pred[i].item()),
                "correct": int(pred[i].item() == y),
                "true_sim": true_sim,
                "max_wrong_sim": max_wrong,
                "margin": margin,
                "mean_sim": float(sim_row.mean().item()),
                "std_sim": float(sim_row.std(unbiased=False).item()),
            })
    return rows


def summarize(rows):
    if not rows:
        return {
            "n": 0, "top1": 0.0, "true_sim": 0.0, "max_wrong_sim": 0.0,
            "margin": 0.0, "mean_sim": 0.0, "std_sim": 0.0,
        }
    return {
        "n": len(rows),
        "top1": float(np.mean([r["correct"] for r in rows])),
        "true_sim": float(np.mean([r["true_sim"] for r in rows])),
        "max_wrong_sim": float(np.mean([r["max_wrong_sim"] for r in rows])),
        "margin": float(np.mean([r["margin"] for r in rows])),
        "mean_sim": float(np.mean([r["mean_sim"] for r in rows])),
        "std_sim": float(np.mean([r["std_sim"] for r in rows])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="./checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon")
    parser.add_argument("--data_root", default="./preproc_vs_re")
    parser.add_argument("--vi_root", default="./preproc_data_vi")
    parser.add_argument("--img_root", default="./preproc_data_vi/images")
    parser.add_argument("--subject_ids", default="1,2,18")
    parser.add_argument("--n_ch", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dino_model", default="dinov2_vits14")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vs_split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--vi_split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject_ids = parse_subject_ids(args.subject_ids)

    dino = load_dino_encoder(args.dino_model, device)
    dino_feat_dim = DINO_DIM[args.dino_model]
    proto_dino = compute_class_prototypes(dino, args.img_root, CLS_LIST, device)

    summary_rows = []
    detail_rows = []
    for sid in subject_ids:
        ckpt_path = os.path.join(args.ckpt_dir, f"subj{sid:02d}_best.pt")
        vi_path = os.path.join(args.vi_root, f"subj_{sid:02d}.mat")
        if not os.path.isfile(ckpt_path):
            print(f"[SKIP] S{sid:02d}: checkpoint not found")
            continue
        if not os.path.isfile(vi_path):
            print(f"[SKIP] S{sid:02d}: VI data not found")
            continue

        model, cfg = load_vs_model(ckpt_path, dino_feat_dim, args.n_ch, device)
        model.eval()
        proto_lat = model.encode_img(proto_dino)

        subj_map = {sid: 0}
        vs_ds = VSReDataset(args.data_root, [sid], subj_map, args.n_ch, args.vs_split, args.seed)
        vi_ds = VIDataset(args.vi_root, sid, args.n_ch, args.vi_split, args.seed)
        vs_loader = DataLoader(vs_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_vs)
        vi_loader = DataLoader(vi_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_vi)

        vs_rows = collect_alignment_stats(model, vs_loader, proto_lat, device, "vs")
        vi_rows = collect_alignment_stats(model, vi_loader, proto_lat, device, "vi")
        vs_sum = summarize(vs_rows)
        vi_sum = summarize(vi_rows)

        print(
            f"S{sid:02d}: VS top1={vs_sum['top1']:.4f} margin={vs_sum['margin']:.4f} "
            f"| VI top1={vi_sum['top1']:.4f} margin={vi_sum['margin']:.4f}"
        )

        for source, summ in [("vs", vs_sum), ("vi", vi_sum)]:
            summary_rows.append({
                "subject": f"S{sid:02d}",
                "source": source,
                **summ,
            })
        for row in vs_rows + vi_rows:
            row["subject"] = f"S{sid:02d}"
            detail_rows.append(row)

    if args.out_csv is None:
        args.out_csv = os.path.join(args.ckpt_dir, "vs_vi_latent_gap_summary.csv")
    detail_csv = args.out_csv.replace("_summary.csv", "_detail.csv")

    with open(args.out_csv, "w", newline="") as f:
        fields = ["subject", "source", "n", "top1", "true_sim", "max_wrong_sim", "margin", "mean_sim", "std_sim"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    with open(detail_csv, "w", newline="") as f:
        fields = ["subject", "source", "label", "pred", "correct", "true_sim", "max_wrong_sim", "margin", "mean_sim", "std_sim"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)

    print(f"Saved summary: {args.out_csv}")
    print(f"Saved detail : {detail_csv}")


if __name__ == "__main__":
    main()
