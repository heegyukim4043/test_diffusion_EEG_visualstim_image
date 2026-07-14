"""Foreground runner for the VI-primary gated residual experiment."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dataset_vs_re import session_counts


FIXED = {
    "epochs": 80,
    "residual_only_epochs": 10,
    "batch_size": 32,
    "lr": 3e-4,
    "student_lr_ratio": 0.1,
    "weight_decay": 1e-4,
    "patience": 20,
    "w_supcon": 0.2,
    "temperature": 0.07,
    "label_smoothing": 0.1,
    "gate_penalty": 1e-3,
    "sampling_rate": 1024.0,
    "n_ch": 32,
    "n_fft": 256,
    "hop_length": 64,
    "hidden_dim": 256,
    "latent_dim": 256,
    "n_heads": 4,
    "n_layers": 2,
    "dropout": 0.1,
}


def subjects(raw: str) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(x.strip()) for x in raw.split(",") if x.strip()))
    if not result:
        raise ValueError("At least one subject is required")
    return result


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "run", "all", "summary"), default="audit")
    parser.add_argument("--subject_id", type=int)
    parser.add_argument("--subjects", default="24")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo", default=str(repo))
    parser.add_argument("--vi_root", default="/content/drive/MyDrive/vsvi_data/preproc_vi_re")
    parser.add_argument(
        "--baseline_root",
        default="/content/drive/MyDrive/vsvi_data/vi_tf_representation_ablation",
    )
    parser.add_argument(
        "--out_root",
        default="/content/drive/MyDrive/vsvi_data/vi_rawtf_gated_residual",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def make_paths(args) -> dict[str, Path]:
    return {
        "repo": Path(args.repo),
        "vi_root": Path(args.vi_root),
        "baseline": Path(args.baseline_root) / f"seed{args.seed}",
        "root": Path(args.out_root) / f"seed{args.seed}",
        "logs": Path(args.out_root) / f"seed{args.seed}" / "logs",
    }


def baselines(paths, sid: int) -> dict[str, Path]:
    root = paths["baseline"] / f"S{sid:02d}" / "raw_tf"
    result = {
        "vs_ckpt": root / "vs_pretrain" / "encoder_best.pt",
        "vi_ckpt": root / "vi_only" / "encoder_best.pt",
        "vi_metrics": root / "vi_only" / "metrics.json",
        "full_metrics": root / "vs_to_vi" / "metrics.json",
    }
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing raw_tf baseline: " + ", ".join(missing))
    return result


def out_dir(paths, sid: int) -> Path:
    return paths["root"] / f"S{sid:02d}" / "gated_residual_vs_to_vi"


def result_path(paths, sid: int) -> Path:
    return out_dir(paths, sid) / "metrics.json"


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        return (
            result.get("stage") == "gated_residual_vs_to_vi"
            and result.get("test_evaluations") == 1
            and result.get("epoch0_is_vi_only") is True
        )
    except (OSError, ValueError):
        return False


def audit(paths, selected_subjects) -> None:
    counts = session_counts(str(paths["vi_root"]))
    rows = []
    print("subject vi_sessions baseline gated_residual")
    for sid in selected_subjects:
        try:
            baselines(paths, sid)
            baseline = 1
        except FileNotFoundError:
            baseline = 0
        row = {
            "subject": sid,
            "vi_sessions": counts.get(sid, 0),
            "baseline": baseline,
            "gated_residual": int(complete(result_path(paths, sid))),
        }
        rows.append(row)
        print(
            f"S{sid:02d} {row['vi_sessions']:11d} {baseline:8d} "
            f"{row['gated_residual']:14d}"
        )
    paths["root"].mkdir(parents=True, exist_ok=True)
    path = paths["root"] / "audit.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] {path}")


def ensure_protocol(paths, args, selected_subjects) -> None:
    payload = {
        **FIXED,
        "subjects": list(selected_subjects),
        "seed": args.seed,
        "fp16": args.fp16,
        "method": "VI_primary_frozen_VS_identity_adapter_zero_gate",
        "selection": "epoch0_included_VI_validation_BAC_Top3_Top5",
        "test_evaluations": 1,
        "vi_root": args.vi_root,
        "baseline_root": args.baseline_root,
    }
    path = paths["root"] / "protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous != payload:
            raise RuntimeError(f"Protocol mismatch at {path}; use a new --out_root")
        return
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def command(paths, args, sid: int) -> list[str]:
    base = baselines(paths, sid)
    result = [
        sys.executable, "-u", str(paths["repo"] / "train_vi_rawtf_gated_residual.py"),
        "--subject_id", str(sid),
        "--vi_root", str(paths["vi_root"]),
        "--vi_ckpt", str(base["vi_ckpt"]),
        "--vs_ckpt", str(base["vs_ckpt"]),
        "--out_dir", str(out_dir(paths, sid)),
        "--seed", str(args.seed),
    ]
    for key, value in FIXED.items():
        result.extend((f"--{key}", str(value)))
    if args.fp16:
        result.append("--fp16")
    if args.force:
        result.append("--overwrite")
    return result


def foreground(command_line, log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[RUN] " + " ".join(command_line), flush=True)
    print(f"[LOG] {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command_line,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command_line)


def main() -> None:
    args = parse_args()
    selected_subjects = subjects(args.subjects)
    paths = make_paths(args)
    if args.stage == "audit":
        audit(paths, selected_subjects)
        return
    if args.stage == "summary":
        foreground([
            sys.executable, "-u",
            str(paths["repo"] / "summarize_vi_rawtf_gated_residual.py"),
            "--root", str(paths["root"]),
            "--baseline_root", str(paths["baseline"]),
            "--subjects", ",".join(map(str, selected_subjects)),
        ], paths["logs"] / f"summary_{datetime.now():%Y%m%d_%H%M%S}.log", paths["repo"])
        return
    ensure_protocol(paths, args, selected_subjects)
    run_subjects = selected_subjects if args.stage == "all" else (args.subject_id,)
    if args.stage == "run" and args.subject_id is None:
        raise ValueError("--subject_id is required for --stage run")
    for sid in run_subjects:
        assert sid is not None
        if complete(result_path(paths, sid)) and not args.force:
            print(f"[SKIP] S{sid:02d}: already complete")
            continue
        foreground(
            command(paths, args, sid),
            paths["logs"] / f"S{sid:02d}_{datetime.now():%Y%m%d_%H%M%S}.log",
            paths["repo"],
        )
        audit(paths, selected_subjects)


if __name__ == "__main__":
    main()
