"""Summarize S24 VI-primary gated residual transfer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--baseline_root", required=True)
    parser.add_argument("--subjects", default="24")
    args = parser.parse_args()
    root, baseline = Path(args.root), Path(args.baseline_root)
    selected = [int(x.strip()) for x in args.subjects.split(",") if x.strip()]
    rows = []
    for sid in selected:
        base = baseline / f"S{sid:02d}" / "raw_tf"
        paths = {
            "vi_only": base / "vi_only" / "metrics.json",
            "full_vs_to_vi": base / "vs_to_vi" / "metrics.json",
            "gated": root / f"S{sid:02d}" / "gated_residual_vs_to_vi" / "metrics.json",
        }
        if not all(path.is_file() for path in paths.values()):
            print(f"[WARN] S{sid:02d}: incomplete")
            continue
        results = {name: load(path) for name, path in paths.items()}
        row = {
            "subject": sid,
            "best_epoch": results["gated"]["best_epoch"],
            "best_phase": results["gated"]["best_phase"],
            "gate_value": results["gated"]["gate_value"],
            "mean_residual_norm": results["gated"]["mean_residual_norm"],
        }
        for name, result in results.items():
            for metric in (
                "balanced_accuracy", "top3", "top5",
                "dominant_ratio", "normalized_entropy",
            ):
                row[f"{name}_{metric}"] = result[metric]
        for name in ("vi_only", "full_vs_to_vi"):
            for metric in ("balanced_accuracy", "top3", "top5", "dominant_ratio"):
                row[f"gated_minus_{name}_{metric}"] = (
                    results["gated"][metric] - results[name][metric]
                )
        rows.append(row)
    if not rows:
        raise RuntimeError("No complete results")
    output = root / "gated_residual_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("subject VI-only_BAC Full_BAC Gated_BAC delta_VI delta_Full gate")
    for row in rows:
        print(
            f"S{row['subject']:02d} {row['vi_only_balanced_accuracy']:.4f} "
            f"{row['full_vs_to_vi_balanced_accuracy']:.4f} "
            f"{row['gated_balanced_accuracy']:.4f} "
            f"{row['gated_minus_vi_only_balanced_accuracy']:+.4f} "
            f"{row['gated_minus_full_vs_to_vi_balanced_accuracy']:+.4f} "
            f"{row['gate_value']:+.5f}"
        )
    print(f"[Saved] {output}")


if __name__ == "__main__":
    main()
