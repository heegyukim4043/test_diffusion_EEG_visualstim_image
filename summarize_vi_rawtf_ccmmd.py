"""Summarize sequential, joint-replay, and class-conditional MMD results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


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
    parser.add_argument("--baseline_roots", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_baseline(args: argparse.Namespace, sid: int) -> tuple[dict, dict] | None:
    for raw_root in args.baseline_roots.split(","):
        root = Path(raw_root.strip()) / f"seed{args.seed}" / f"S{sid:02d}" / "raw_tf"
        vi_only = read_json(root / "vi_only" / "metrics.json")
        sequential = read_json(root / "vs_to_vi" / "metrics.json")
        if vi_only is not None and sequential is not None:
            return vi_only, sequential
    return None


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
    subjects = [int(value.strip()) for value in args.subjects.split(",") if value.strip()]
    long_rows = []
    paired_rows = []
    for sid in subjects:
        baseline = resolve_baseline(args, sid)
        replay = read_json(root / f"S{sid:02d}" / "replay" / "metrics.json")
        ccmmd = read_json(root / f"S{sid:02d}" / "ccmmd" / "metrics.json")
        if baseline is None or replay is None or ccmmd is None:
            missing = []
            if baseline is None:
                missing.append("baseline")
            if replay is None:
                missing.append("replay")
            if ccmmd is None:
                missing.append("ccmmd")
            print(f"[SKIP] S{sid:02d}: missing {', '.join(missing)}")
            continue
        vi_only, sequential = baseline
        conditions = {
            "vi_only": vi_only,
            "sequential": sequential,
            "replay": replay,
            "ccmmd": ccmmd,
        }
        for condition, metrics in conditions.items():
            long_rows.append({
                "subject": sid,
                "condition": condition,
                "selected_lambda": metrics.get("selected_lambda", 0.0),
                **{metric: metrics.get(metric) for metric in METRICS},
            })
        row = {
            "subject": sid,
            "n_test": vi_only.get("n_test"),
            "ccmmd_selected_lambda": ccmmd.get("selected_lambda"),
        }
        for condition, metrics in conditions.items():
            for metric in METRICS:
                row[f"{condition}_{metric}"] = metrics.get(metric)
        for metric in METRICS:
            row[f"delta_sequential_minus_vi_only_{metric}"] = (
                sequential[metric] - vi_only[metric]
            )
            row[f"delta_replay_minus_sequential_{metric}"] = (
                replay[metric] - sequential[metric]
            )
            row[f"delta_ccmmd_minus_replay_{metric}"] = ccmmd[metric] - replay[metric]
            row[f"delta_ccmmd_minus_vi_only_{metric}"] = ccmmd[metric] - vi_only[metric]
        paired_rows.append(row)

    root.mkdir(parents=True, exist_ok=True)
    long_path = root / "rawtf_ccmmd_long.csv"
    paired_path = root / "rawtf_ccmmd_summary.csv"
    write_csv(long_path, long_rows)
    write_csv(paired_path, paired_rows)
    print(f"\nComplete subjects: {len(paired_rows)} / {len(subjects)}")
    if paired_rows:
        print("condition       Top-1    Top-3    Top-5    BalAcc    Dominant")
        for condition in ("vi_only", "sequential", "replay", "ccmmd"):
            means = [
                np.mean([row[f"{condition}_{metric}"] for row in paired_rows])
                for metric in ("top1", "top3", "top5", "balanced_accuracy", "dominant_ratio")
            ]
            print(f"{condition:12s} " + " ".join(f"{value:8.4f}" for value in means))
        for name in (
            "delta_replay_minus_sequential_balanced_accuracy",
            "delta_ccmmd_minus_replay_balanced_accuracy",
            "delta_ccmmd_minus_vi_only_balanced_accuracy",
        ):
            values = np.asarray([row[name] for row in paired_rows], dtype=float)
            print(
                f"{name}: mean={values.mean():+.4f} "
                f"B/T/W={(values > 1e-12).sum()}/"
                f"{(np.abs(values) <= 1e-12).sum()}/"
                f"{(values < -1e-12).sum()}"
            )
    print(f"[Saved] {long_path}")
    print(f"[Saved] {paired_path}")


if __name__ == "__main__":
    main()
