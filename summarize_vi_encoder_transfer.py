"""Summarize paired VI encoder-transfer conditions across subjects."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MODES = ("zero_shot", "vi_only", "vs_to_vi")
METRICS = ("top1", "top3", "top5", "balanced_accuracy", "mean_true_score", "mean_true_margin")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--subjects", default="1,2,9,18,24,28,29")
    return p.parse_args()


def parse_subjects(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def read_json(path: Path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    subjects = parse_subjects(args.subjects)
    long_rows = []
    wide_rows = []

    for sid in subjects:
        found = {
            mode: read_json(root / f"S{sid:02d}" / mode / "metrics.json")
            for mode in MODES
        }
        for mode, result in found.items():
            if result is not None:
                long_rows.append({
                    "subject": sid,
                    "mode": mode,
                    "vi_sessions": result.get("vi_sessions"),
                    "n_test": result.get("n_test"),
                    **{metric: result.get(metric) for metric in METRICS},
                })
        if not all(found.values()):
            missing = [mode for mode, result in found.items() if result is None]
            print(f"[SKIP] S{sid:02d}: incomplete ({', '.join(missing)})")
            continue

        row = {
            "subject": sid,
            "vi_sessions": found["vi_only"]["vi_sessions"],
            "n_test": found["vi_only"]["n_test"],
        }
        for mode in MODES:
            for metric in METRICS:
                row[f"{mode}_{metric}"] = found[mode][metric]
        for metric in METRICS:
            row[f"delta_vs_to_vi_minus_vi_only_{metric}"] = (
                found["vs_to_vi"][metric] - found["vi_only"][metric]
            )
            row[f"delta_vs_to_vi_minus_zero_shot_{metric}"] = (
                found["vs_to_vi"][metric] - found["zero_shot"][metric]
            )
        wide_rows.append(row)

    root.mkdir(parents=True, exist_ok=True)
    long_path = root / "vi_encoder_transfer_long.csv"
    wide_path = root / "vi_encoder_transfer_summary.csv"

    if long_rows:
        with long_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
            writer.writeheader()
            writer.writerows(long_rows)
    if wide_rows:
        with wide_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(wide_rows[0]))
            writer.writeheader()
            writer.writerows(wide_rows)

    print(f"\nPaired subjects: {len(wide_rows)} / {len(subjects)}")
    if wide_rows:
        print("mode          Top-1    Top-3    Top-5    BalAcc")
        for mode in MODES:
            means = [
                np.mean([row[f"{mode}_{metric}"] for row in wide_rows])
                for metric in ("top1", "top3", "top5", "balanced_accuracy")
            ]
            print(f"{mode:12s} " + " ".join(f"{x:8.4f}" for x in means))

        delta = np.array([
            row["delta_vs_to_vi_minus_vi_only_top1"] for row in wide_rows
        ])
        better = int((delta > 1e-12).sum())
        tie = int((np.abs(delta) <= 1e-12).sum())
        worse = int((delta < -1e-12).sum())
        print(f"\nVS->VI minus VI-only Top-1 mean: {delta.mean():+.4f}")
        print(f"VS->VI better / tie / worse: {better} / {tie} / {worse}")

    print(f"[Saved] {long_path}")
    print(f"[Saved] {wide_path}")


if __name__ == "__main__":
    main()
