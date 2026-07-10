"""Summarize full-test Exp43 manifests and make paired C0/C1 comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        default="/content/drive/MyDrive/vsvi_data/gen_images_full",
    )
    p.add_argument("--out_csv", default=None)
    return p.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_counts(c0: list[dict], c1: list[dict]) -> tuple[int, int, int]:
    keys = ("test_index", "true_label", "seed")
    map0 = {tuple(row[key] for key in keys): row for row in c0}
    map1 = {tuple(row[key] for key in keys): row for row in c1}
    if set(map0) != set(map1):
        missing0 = len(set(map1) - set(map0))
        missing1 = len(set(map0) - set(map1))
        raise ValueError(
            f"C0/C1 are not paired: missing_from_c0={missing0}, missing_from_c1={missing1}"
        )
    c0_only = 0
    c1_only = 0
    both = 0
    for key in map0:
        hit0 = int(map0[key]["correct"])
        hit1 = int(map1[key]["correct"])
        c0_only += int(hit0 == 1 and hit1 == 0)
        c1_only += int(hit0 == 0 and hit1 == 1)
        both += int(hit0 == 1 and hit1 == 1)
    return c0_only, c1_only, both


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    metric_paths = sorted(root.glob("S??/c?/metrics.json"))
    if not metric_paths:
        raise FileNotFoundError(f"No metrics.json found under {root}")

    by_subject: dict[int, dict[str, dict]] = {}
    metric_rows = []
    for path in metric_paths:
        with path.open(encoding="utf-8") as handle:
            metric = json.load(handle)
        subject = int(metric["subject"])
        condition = metric["condition"]
        by_subject.setdefault(subject, {})[condition] = metric
        metric_rows.append(metric)

    summary_rows = []
    for subject in sorted(by_subject):
        conditions = by_subject[subject]
        row = {"subject": subject}
        for condition in ("c0", "c1"):
            metric = conditions.get(condition)
            for key in (
                "n_test",
                "top1",
                "top3",
                "top5",
                "balanced_accuracy",
                "entropy",
                "normalized_entropy",
                "dominant_ratio",
                "mean_true_similarity",
                "mean_true_margin",
            ):
                row[f"{condition}_{key}"] = "" if metric is None else metric[key]

        if "c0" in conditions and "c1" in conditions:
            row["delta_top1_c1_minus_c0"] = conditions["c1"]["top1"] - conditions["c0"]["top1"]
            row["delta_entropy_c1_minus_c0"] = conditions["c1"]["entropy"] - conditions["c0"]["entropy"]
            row["delta_dominant_c1_minus_c0"] = conditions["c1"]["dominant_ratio"] - conditions["c0"]["dominant_ratio"]
            c0_manifest = read_csv(root / f"S{subject:02d}" / "c0" / "manifest.csv")
            c1_manifest = read_csv(root / f"S{subject:02d}" / "c1" / "manifest.csv")
            c0_only, c1_only, both = paired_counts(c0_manifest, c1_manifest)
            row["paired_c0_only_correct"] = c0_only
            row["paired_c1_only_correct"] = c1_only
            row["paired_both_correct"] = both
        summary_rows.append(row)

    output = Path(args.out_csv) if args.out_csv else root / "exp43_full_test_summary.csv"
    all_fields = []
    for row in summary_rows:
        for key in row:
            if key not in all_fields:
                all_fields.append(key)
    for row in summary_rows:
        for field in all_fields:
            row.setdefault(field, "")
    write_csv(output, summary_rows)

    paired = [row for row in summary_rows if row.get("delta_top1_c1_minus_c0", "") != ""]
    if paired:
        deltas = np.array([float(row["delta_top1_c1_minus_c0"]) for row in paired])
        print(f"Subjects paired: {len(paired)}")
        print(f"Mean C1-C0 Top1: {deltas.mean():.4f}")
        print(f"Median C1-C0 Top1: {np.median(deltas):.4f}")
        print(f"C1 better / tie / worse: {(deltas > 0).sum()} / {(deltas == 0).sum()} / {(deltas < 0).sum()}")
    print(f"[Saved] {output}")


if __name__ == "__main__":
    main()
