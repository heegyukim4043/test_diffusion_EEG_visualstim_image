"""Print the frozen-LoRA raw+TF downstream generation comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()
    path = Path(args.metrics)
    with path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    rows = []
    for encoder, result in metrics["results"].items():
        correct = result["generation"]["correct"]
        shuffled = result["generation"]["shuffled"]
        zero = result["generation"]["zero"]
        rows.append({
            "encoder": encoder,
            "best_epoch": result["best_epoch"],
            "val_diffusion_loss": result["best_validation_diffusion_loss"],
            "top1": correct["top1"],
            "top3": correct["top3"],
            "top5": correct["top5"],
            "entropy": correct["normalized_entropy"],
            "dominant_ratio": correct["dominant_ratio"],
            "shuffled_change_rate": shuffled["prediction_change_rate_vs_correct"],
            "shuffled_dino_cosine": shuffled["mean_dino_cosine_vs_correct"],
            "zero_change_rate": zero["prediction_change_rate_vs_correct"],
            "zero_dino_cosine": zero["mean_dino_cosine_vs_correct"],
            "prediction_counts": correct["prediction_counts"],
            "test_evaluations": result["test_evaluations"],
        })
    frame = pd.DataFrame(rows)
    output = path.with_name("generation_comparison.csv")
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"[Saved] {output}")


if __name__ == "__main__":
    main()
