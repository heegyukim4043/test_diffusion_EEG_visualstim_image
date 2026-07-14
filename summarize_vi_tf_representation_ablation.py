"""Summarize paired raw/TF/raw+TF VS-to-VI representation experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPRESENTATIONS = ("raw", "tf", "raw_tf")
VI_STAGES = ("zero_shot", "vi_only", "vs_to_vi")
METRICS = (
    "top1",
    "top3",
    "top5",
    "balanced_accuracy",
    "mean_true_margin",
    "normalized_entropy",
    "dominant_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--subjects", default="24")
    return parser.parse_args()


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    subjects = [int(value) for value in args.subjects.split(",") if value.strip()]
    long_rows = []
    paired_rows = []

    for sid in subjects:
        for representation in REPRESENTATIONS:
            found = {
                stage: read_json(root / f"S{sid:02d}" / representation / stage / "metrics.json")
                for stage in VI_STAGES
            }
            for stage, metrics in found.items():
                if metrics is not None:
                    long_rows.append({
                        "subject": sid,
                        "representation": representation,
                        "stage": stage,
                        "n_sessions": metrics.get("n_sessions"),
                        "parameter_count": metrics.get("parameter_count"),
                        **{metric: metrics.get(metric) for metric in METRICS},
                    })
            if not all(found.values()):
                missing = [stage for stage, value in found.items() if value is None]
                print(f"[SKIP] S{sid:02d} {representation}: missing {', '.join(missing)}")
                continue
            row = {
                "subject": sid,
                "representation": representation,
                "n_sessions": found["vi_only"].get("n_sessions"),
            }
            for stage in VI_STAGES:
                for metric in METRICS:
                    row[f"{stage}_{metric}"] = found[stage][metric]
            for metric in METRICS:
                row[f"delta_vs_to_vi_minus_vi_only_{metric}"] = (
                    found["vs_to_vi"][metric] - found["vi_only"][metric]
                )
            paired_rows.append(row)

    root.mkdir(parents=True, exist_ok=True)
    long_path = root / "tf_representation_long.csv"
    paired_path = root / "tf_representation_paired.csv"
    write_csv(long_path, long_rows)
    write_csv(paired_path, paired_rows)

    print(f"\nComplete subject/representation pairs: {len(paired_rows)}")
    if paired_rows:
        print("representation    zero-shot   VI-only    VS->VI    transfer_delta")
        for representation in REPRESENTATIONS:
            subset = [row for row in paired_rows if row["representation"] == representation]
            if not subset:
                continue
            zero = np.mean([row["zero_shot_balanced_accuracy"] for row in subset])
            vi = np.mean([row["vi_only_balanced_accuracy"] for row in subset])
            transfer = np.mean([row["vs_to_vi_balanced_accuracy"] for row in subset])
            delta = np.mean([
                row["delta_vs_to_vi_minus_vi_only_balanced_accuracy"] for row in subset
            ])
            print(f"{representation:16s} {zero:9.4f} {vi:10.4f} {transfer:9.4f} {delta:+15.4f}")

    print(f"[Saved] {long_path}")
    print(f"[Saved] {paired_path}")


if __name__ == "__main__":
    main()
