"""Summarize VI Riemannian baseline metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out_dir", default="")
    return parser.parse_args()


def read_metric(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        m = json.load(handle)
    val = m.get("validation", {})
    test = m.get("test", {})
    perm = m.get("permutation_test", {})
    return {
        "subject": m.get("subject"),
        "seed": m.get("seed"),
        "feature": m.get("feature"),
        "classifier": m.get("classifier"),
        "shrinkage": m.get("config", {}).get("shrinkage"),
        "feature_dim": m.get("feature_dim"),
        "n_train": m.get("n_train"),
        "n_val": m.get("n_val"),
        "n_test": m.get("n_test"),
        "val_BAC": val.get("balanced_accuracy"),
        "val_top3": val.get("top3"),
        "val_top5": val.get("top5"),
        "test_BAC": test.get("balanced_accuracy"),
        "test_top1": test.get("top1"),
        "test_top3": test.get("top3"),
        "test_top5": test.get("top5"),
        "test_margin": test.get("mean_true_margin"),
        "test_entropy": test.get("normalized_entropy"),
        "test_dominant_ratio": test.get("dominant_ratio"),
        "prediction_counts": test.get("prediction_counts"),
        "perm_p": perm.get("permutation_p_ge_observed"),
        "path": str(path),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    metrics = sorted(root.rglob("metrics.json"))
    if not metrics:
        raise FileNotFoundError(f"No metrics.json under {root}")
    rows = [read_metric(path) for path in metrics]
    df = pd.DataFrame(rows).sort_values(["subject", "feature", "classifier", "shrinkage"])

    # Validation-selected best config per subject. Test is read once after selection.
    best = (
        df.sort_values(
            ["subject", "val_BAC", "val_top3", "val_top5", "test_dominant_ratio"],
            ascending=[True, False, False, False, True],
        )
        .groupby("subject", as_index=False)
        .head(1)
        .sort_values("subject")
        .reset_index(drop=True)
    )

    by_config = (
        df.groupby(["feature", "classifier", "shrinkage"], dropna=False)
        .agg(
            n=("subject", "nunique"),
            val_BAC_mean=("val_BAC", "mean"),
            test_BAC_mean=("test_BAC", "mean"),
            test_BAC_median=("test_BAC", "median"),
            test_top3_mean=("test_top3", "mean"),
            test_top5_mean=("test_top5", "mean"),
            dominant_mean=("test_dominant_ratio", "mean"),
        )
        .reset_index()
        .sort_values(["test_BAC_mean", "test_top3_mean"], ascending=False)
    )

    out_dir = Path(args.out_dir) if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "riemannian_all_metrics.csv", index=False)
    best.to_csv(out_dir / "riemannian_validation_selected_by_subject.csv", index=False)
    by_config.to_csv(out_dir / "riemannian_by_config_summary.csv", index=False)

    print(f"metrics: {len(df)}")
    print("\nValidation-selected by subject:")
    print(best[[
        "subject", "feature", "classifier", "shrinkage",
        "val_BAC", "test_BAC", "test_top3", "test_top5", "test_dominant_ratio",
    ]].to_string(index=False))
    print("\nBy config:")
    print(by_config.head(20).to_string(index=False))
    print(f"\nsaved: {out_dir}")


if __name__ == "__main__":
    main()
