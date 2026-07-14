"""Summarize transfer-safe raw+TF fine-tuning against fixed baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_subjects(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--baseline_root", required=True)
    parser.add_argument("--subjects", default="24")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    baseline_root = Path(args.baseline_root)
    rows = []
    for sid in parse_subjects(args.subjects):
        base = baseline_root / f"S{sid:02d}" / "raw_tf"
        paths = {
            "vi_only": base / "vi_only" / "metrics.json",
            "full_vs_to_vi": base / "vs_to_vi" / "metrics.json",
            "safe_vs_to_vi": root / f"S{sid:02d}" / "safe_vs_to_vi" / "metrics.json",
        }
        if not all(path.is_file() for path in paths.values()):
            print(f"[WARN] S{sid:02d}: incomplete; skipped")
            continue
        metrics = {name: load(path) for name, path in paths.items()}
        row = {
            "subject": sid,
            "vi_sessions": metrics["safe_vs_to_vi"]["n_sessions"],
            "best_epoch": metrics["safe_vs_to_vi"]["best_epoch"],
            "best_phase": metrics["safe_vs_to_vi"]["best_phase"],
        }
        for name, result in metrics.items():
            for key in ("balanced_accuracy", "top3", "top5", "dominant_ratio", "normalized_entropy"):
                row[f"{name}_{key}"] = result[key]
        for baseline in ("vi_only", "full_vs_to_vi"):
            for key in ("balanced_accuracy", "top3", "top5", "dominant_ratio"):
                row[f"safe_minus_{baseline}_{key}"] = (
                    metrics["safe_vs_to_vi"][key] - metrics[baseline][key]
                )
        rows.append(row)

    if not rows:
        raise RuntimeError("No complete subjects found")
    output = root / "safe_transfer_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Subjects: {len(rows)}")
    print("mode             BAC      Top-3    Top-5    Dominant")
    for name, label in (
        ("vi_only", "VI-only"),
        ("full_vs_to_vi", "Full VS->VI"),
        ("safe_vs_to_vi", "Safe VS->VI"),
    ):
        values = {
            key: float(np.mean([row[f"{name}_{key}"] for row in rows]))
            for key in ("balanced_accuracy", "top3", "top5", "dominant_ratio")
        }
        print(
            f"{label:15s} {values['balanced_accuracy']:.4f}   "
            f"{values['top3']:.4f}   {values['top5']:.4f}   "
            f"{values['dominant_ratio']:.4f}"
        )
    for baseline, label in (("vi_only", "VI-only"), ("full_vs_to_vi", "Full VS->VI")):
        delta = np.asarray([
            row[f"safe_minus_{baseline}_balanced_accuracy"] for row in rows
        ])
        print(
            f"Safe minus {label} BAC: mean={delta.mean():+.4f} "
            f"B/T/W={(delta > 1e-12).sum()}/{(np.abs(delta) <= 1e-12).sum()}/{(delta < -1e-12).sum()}"
        )
    print(f"[Saved] {output}")


if __name__ == "__main__":
    main()
