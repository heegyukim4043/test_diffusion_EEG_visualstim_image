"""
Audit per-subject, per-class trial counts for VS-re and VI datasets.

This is required before VI fine-tuning because some subjects/classes may have
different trial counts. It reports raw counts and deterministic train/val/test
split counts using the same 80/10/10 class-stratified rule used by the loaders.

Usage:
  python diagnose_class_trial_distribution.py --dataset both
  python diagnose_class_trial_distribution.py --dataset vi --subject_ids all
  python diagnose_class_trial_distribution.py --dataset vs --subject_ids 11,16,23
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from dataset_vs_re import available_subjects, load_subject_vsre


N_CLASSES = 9


def parse_subject_ids(raw, dataset, root):
    raw = raw.strip().lower()
    if raw == "all":
        if dataset == "vs":
            return available_subjects(root)
        paths = sorted(Path(root).glob("subj_*.mat"))
        return [int(p.stem.replace("subj_", "")) for p in paths]

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


def split_counts(count, seed, cls, subj_index=0):
    idx = np.arange(count)
    rng = np.random.RandomState(seed + subj_index * 100 + cls)
    rng.shuffle(idx)
    n_train = int(count * 0.8)
    n_val = int(count * 0.1)
    return n_train, n_val, count - n_train - n_val


def count_vs_subject(root, sid, n_ch, seed):
    eeg, labels, eff_sessions = load_subject_vsre(root, sid, n_ch=n_ch)
    counts = np.bincount(labels, minlength=N_CLASSES)
    rows = []
    for cls in range(N_CLASSES):
        tr, va, te = split_counts(int(counts[cls]), seed, cls, subj_index=0)
        rows.append({
            "dataset": "vs",
            "subject": f"S{sid:02d}",
            "class": cls + 1,
            "total": int(counts[cls]),
            "train": tr,
            "val": va,
            "test": te,
            "effective_sessions": eff_sessions,
            "time_points": eeg.shape[-1] if len(eeg) else "",
            "channels": eeg.shape[1] if len(eeg) else "",
            "status": "ok",
        })
    return rows


def count_vi_subject(root, sid, n_ch, seed):
    path = Path(root) / f"subj_{sid:02d}.mat"
    if not path.exists():
        return [{
            "dataset": "vi", "subject": f"S{sid:02d}", "class": "",
            "total": 0, "train": 0, "val": 0, "test": 0,
            "effective_sessions": "", "time_points": "", "channels": "",
            "status": "missing",
        }]
    mat = loadmat(path)
    if "X" not in mat or "y" not in mat:
        return [{
            "dataset": "vi", "subject": f"S{sid:02d}", "class": "",
            "total": 0, "train": 0, "val": 0, "test": 0,
            "effective_sessions": "", "time_points": "", "channels": "",
            "status": "missing_X_or_y",
        }]
    X = mat["X"]
    y = mat["y"].squeeze().astype(np.int64)
    labels = y - 1
    counts = np.bincount(labels[(labels >= 0) & (labels < N_CLASSES)], minlength=N_CLASSES)
    rows = []
    for cls in range(N_CLASSES):
        tr, va, te = split_counts(int(counts[cls]), seed, cls, subj_index=0)
        rows.append({
            "dataset": "vi",
            "subject": f"S{sid:02d}",
            "class": cls + 1,
            "total": int(counts[cls]),
            "train": tr,
            "val": va,
            "test": te,
            "effective_sessions": "",
            "time_points": X.shape[1] if X.ndim >= 2 else "",
            "channels": min(n_ch, X.shape[0]) if X.ndim >= 1 else "",
            "status": "ok",
        })
    return rows


def summarize(rows):
    by_subject = {}
    for row in rows:
        key = (row["dataset"], row["subject"])
        by_subject.setdefault(key, []).append(row)

    summary = []
    for (dataset, subject), group in sorted(by_subject.items()):
        totals = [int(r["total"]) for r in group if r["class"] != ""]
        tests = [int(r["test"]) for r in group if r["class"] != ""]
        nonzero = sum(1 for x in totals if x > 0)
        min_total = min(totals) if totals else 0
        max_total = max(totals) if totals else 0
        min_test = min(tests) if tests else 0
        balanced = int(min_total == max_total and nonzero == N_CLASSES)
        summary.append({
            "dataset": dataset,
            "subject": subject,
            "n_classes_nonzero": nonzero,
            "min_total_per_class": min_total,
            "max_total_per_class": max_total,
            "min_test_per_class": min_test,
            "total_trials": sum(totals),
            "balanced_9class": balanced,
            "status": "ok" if nonzero == N_CLASSES and min_test > 0 else "check",
        })
    return summary


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["vs", "vi", "both"], default="both")
    parser.add_argument("--vs_root", default="./preproc_vs_re")
    parser.add_argument("--vi_root", default="./preproc_data_vi")
    parser.add_argument("--subject_ids", default="all")
    parser.add_argument("--n_ch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_prefix", default="class_trial_distribution")
    args = parser.parse_args()

    rows = []
    if args.dataset in ("vs", "both"):
        for sid in parse_subject_ids(args.subject_ids, "vs", args.vs_root):
            try:
                rows.extend(count_vs_subject(args.vs_root, sid, args.n_ch, args.seed))
            except Exception as exc:
                rows.append({
                    "dataset": "vs", "subject": f"S{sid:02d}", "class": "",
                    "total": 0, "train": 0, "val": 0, "test": 0,
                    "effective_sessions": "", "time_points": "", "channels": "",
                    "status": f"error:{type(exc).__name__}:{exc}",
                })

    if args.dataset in ("vi", "both"):
        for sid in parse_subject_ids(args.subject_ids, "vi", args.vi_root):
            rows.extend(count_vi_subject(args.vi_root, sid, args.n_ch, args.seed))

    summary_rows = summarize(rows)

    detail_csv = f"{args.out_prefix}_detail.csv"
    summary_csv = f"{args.out_prefix}_summary.csv"
    write_csv(detail_csv, rows, [
        "dataset", "subject", "class", "total", "train", "val", "test",
        "effective_sessions", "time_points", "channels", "status",
    ])
    write_csv(summary_csv, summary_rows, [
        "dataset", "subject", "n_classes_nonzero", "min_total_per_class",
        "max_total_per_class", "min_test_per_class", "total_trials",
        "balanced_9class", "status",
    ])

    print(f"Saved: {detail_csv}")
    print(f"Saved: {summary_csv}")
    print("\nSubjects requiring check:")
    for row in summary_rows:
        if row["status"] != "ok" or not row["balanced_9class"]:
            print(row)


if __name__ == "__main__":
    main()
