"""
Diagnose raw preproc_vs_re subject/session validity.

This intentionally bypasses VSReDataset's skip logic so anomalous files can be
inspected before deciding whether to include or exclude them.

Usage:
  python diagnose_vsre_subject_data.py --subject_ids 23
  python diagnose_vsre_subject_data.py --subject_ids 2,19,23,28
"""

import argparse
import csv
import os
from pathlib import Path

import h5py
import numpy as np


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


def list_session_files(data_root, sid):
    root = Path(data_root)
    prefix = f"preproc_subj_{sid:02d}_"
    files = []
    for path in sorted(root.glob(f"{prefix}*.mat")):
        try:
            sess = int(path.stem.replace(prefix[:-1], "").split("_")[-1])
        except ValueError:
            sess = -1
        files.append((sess, path))
    return sorted(files, key=lambda x: x[0])


def inspect_file(path, n_ch):
    row = {
        "file": str(path),
        "exists": path.exists(),
        "shape": "",
        "n_rep": "",
        "n_blk": "",
        "n_cls": "",
        "n_time": "",
        "n_ch_all": "",
        "total_trials": 0,
        "per_class": "",
        "finite_frac": "",
        "mean": "",
        "std": "",
        "status": "missing",
    }
    if not path.exists():
        return row

    with h5py.File(path, "r") as f:
        if "results/data" not in f:
            row["status"] = "missing_results_data"
            return row
        data = np.array(f["results/data"])

    row["shape"] = str(tuple(data.shape))
    if data.ndim != 5:
        row["status"] = f"bad_ndim_{data.ndim}"
        return row

    n_rep, n_blk, n_cls, n_time, n_ch_all = data.shape
    row.update({
        "n_rep": n_rep,
        "n_blk": n_blk,
        "n_cls": n_cls,
        "n_time": n_time,
        "n_ch_all": n_ch_all,
        "total_trials": n_rep * n_blk * n_cls,
        "per_class": ";".join([str(n_rep * n_blk)] * n_cls),
    })

    sig = data[:, :, :, 1024:3072, :min(n_ch, n_ch_all)]
    finite = np.isfinite(sig)
    row["finite_frac"] = float(finite.mean())
    row["mean"] = float(np.nanmean(sig))
    row["std"] = float(np.nanstd(sig))

    if n_cls != 9:
        row["status"] = "bad_n_cls"
    elif n_time < 3072:
        row["status"] = "bad_time"
    elif n_ch_all < n_ch:
        row["status"] = "bad_channel_count"
    elif n_rep > 20:
        row["status"] = "large_n_rep_needs_decision"
    elif row["finite_frac"] < 1.0:
        row["status"] = "nonfinite_values"
    elif row["std"] == 0:
        row["status"] = "zero_variance"
    else:
        row["status"] = "ok"
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./preproc_vs_re")
    parser.add_argument("--subject_ids", default="23")
    parser.add_argument("--n_ch", type=int, default=32)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    subject_ids = parse_subject_ids(args.subject_ids)
    rows = []
    for sid in subject_ids:
        files = list_session_files(args.data_root, sid)
        print(f"\nS{sid:02d}: {len(files)} files")
        if not files:
            rows.append({"subject": f"S{sid:02d}", "session": "", "status": "no_files"})
            continue
        for sess, path in files:
            row = inspect_file(path, args.n_ch)
            row["subject"] = f"S{sid:02d}"
            row["session"] = sess
            rows.append(row)
            print(
                f"  sess{sess}: status={row['status']} shape={row['shape']} "
                f"trials={row['total_trials']} per_class={row['per_class']}"
            )

    if args.out_csv is None:
        args.out_csv = f"diagnose_vsre_subject_data_{args.subject_ids.replace(',', '_')}.csv"

    if rows:
        fieldnames = [
            "subject", "session", "status", "file", "exists", "shape",
            "n_rep", "n_blk", "n_cls", "n_time", "n_ch_all",
            "total_trials", "per_class", "finite_frac", "mean", "std",
        ]
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
