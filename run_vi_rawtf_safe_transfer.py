"""Foreground runner for transfer-safe raw+TF VS-to-VI fine-tuning."""

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
    "sampling_rate": 1024.0,
    "n_ch": 32,
    "epochs": 100,
    "batch_size": 32,
    "lr": 3e-4,
    "fusion_lr_ratio": 1.0 / 3.0,
    "backbone_lr_ratio": 0.1,
    "weight_decay": 1e-4,
    "patience": 20,
    "head_epochs": 10,
    "fusion_epochs": 20,
    "w_supcon": 0.2,
    "temperature": 0.07,
    "label_smoothing": 0.1,
    "n_fft": 256,
    "hop_length": 64,
    "hidden_dim": 256,
    "latent_dim": 256,
    "n_heads": 4,
    "n_layers": 2,
    "dropout": 0.1,
}


def parse_subjects(raw: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(value.strip()) for value in raw.split(",") if value.strip()))
    if not values:
        raise ValueError("At least one subject is required")
    return values


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
        default="/content/drive/MyDrive/vsvi_data/vi_rawtf_safe_transfer",
    )
    parser.add_argument("--epochs", type=int, default=FIXED["epochs"])
    parser.add_argument("--batch_size", type=int, default=FIXED["batch_size"])
    parser.add_argument("--patience", type=int, default=FIXED["patience"])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def paths_for(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "repo": Path(args.repo),
        "vi_root": Path(args.vi_root),
        "baseline": Path(args.baseline_root) / f"seed{args.seed}",
        "root": Path(args.out_root) / f"seed{args.seed}",
        "logs": Path(args.out_root) / f"seed{args.seed}" / "logs",
    }


def baseline_paths(paths: dict[str, Path], sid: int) -> dict[str, Path]:
    root = paths["baseline"] / f"S{sid:02d}" / "raw_tf"
    result = {
        "vs_ckpt": root / "vs_pretrain" / "encoder_best.pt",
        "vi_only": root / "vi_only" / "metrics.json",
        "vs_to_vi": root / "vs_to_vi" / "metrics.json",
    }
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing raw_tf baseline files: " + ", ".join(missing))
    return result


def output_dir(paths: dict[str, Path], sid: int) -> Path:
    return paths["root"] / f"S{sid:02d}" / "safe_vs_to_vi"


def metrics_path(paths: dict[str, Path], sid: int) -> Path:
    return output_dir(paths, sid) / "metrics.json"


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        return (
            metrics.get("stage") == "safe_vs_to_vi"
            and metrics.get("classifier_reinitialized") is True
            and metrics.get("test_evaluations") == 1
        )
    except (OSError, ValueError):
        return False


def protocol_payload(args: argparse.Namespace, subjects: tuple[int, ...]) -> dict:
    return {
        **FIXED,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "subjects": list(subjects),
        "seed": args.seed,
        "representation": "raw_tf",
        "condition": "VS_backbone_reinitialized_VI_head_gradual_unfreeze",
        "selection": "VI_validation_BAC_Top3_Top5_rounded_12_decimals",
        "test_evaluations": 1,
        "vi_root": args.vi_root,
        "baseline_root": args.baseline_root,
        "fp16": args.fp16,
    }


def ensure_protocol(paths: dict[str, Path], args: argparse.Namespace, subjects: tuple[int, ...]) -> None:
    path = paths["root"] / "protocol.json"
    payload = protocol_payload(args, subjects)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous != payload:
            raise RuntimeError(f"Protocol mismatch at {path}; use a new --out_root")
        return
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def audit(paths: dict[str, Path], subjects: tuple[int, ...]) -> None:
    counts = session_counts(str(paths["vi_root"]))
    rows = []
    print("subject vi_sessions baseline safe_vs_to_vi")
    for sid in subjects:
        try:
            baseline_paths(paths, sid)
            baseline = 1
        except FileNotFoundError:
            baseline = 0
        row = {
            "subject": sid,
            "vi_sessions": counts.get(sid, 0),
            "baseline": baseline,
            "safe_vs_to_vi": int(complete(metrics_path(paths, sid))),
        }
        rows.append(row)
        print(
            f"S{sid:02d} {row['vi_sessions']:11d} {baseline:8d} "
            f"{row['safe_vs_to_vi']:13d}"
        )
    path = paths["root"] / "audit.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] {path}")


def command_for(
    paths: dict[str, Path],
    args: argparse.Namespace,
    sid: int,
) -> list[str]:
    baseline = baseline_paths(paths, sid)
    command = [
        sys.executable,
        "-u",
        str(paths["repo"] / "train_vi_rawtf_safe_transfer.py"),
        "--subject_id", str(sid),
        "--vi_root", str(paths["vi_root"]),
        "--vs_ckpt", str(baseline["vs_ckpt"]),
        "--out_dir", str(output_dir(paths, sid)),
        "--seed", str(args.seed),
        "--sampling_rate", str(FIXED["sampling_rate"]),
        "--n_ch", str(FIXED["n_ch"]),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(FIXED["lr"]),
        "--fusion_lr_ratio", str(FIXED["fusion_lr_ratio"]),
        "--backbone_lr_ratio", str(FIXED["backbone_lr_ratio"]),
        "--weight_decay", str(FIXED["weight_decay"]),
        "--patience", str(args.patience),
        "--head_epochs", str(FIXED["head_epochs"]),
        "--fusion_epochs", str(FIXED["fusion_epochs"]),
        "--w_supcon", str(FIXED["w_supcon"]),
        "--temperature", str(FIXED["temperature"]),
        "--label_smoothing", str(FIXED["label_smoothing"]),
        "--n_fft", str(FIXED["n_fft"]),
        "--hop_length", str(FIXED["hop_length"]),
        "--hidden_dim", str(FIXED["hidden_dim"]),
        "--latent_dim", str(FIXED["latent_dim"]),
        "--n_heads", str(FIXED["n_heads"]),
        "--n_layers", str(FIXED["n_layers"]),
        "--dropout", str(FIXED["dropout"]),
    ]
    if args.fp16:
        command.append("--fp16")
    if args.force:
        command.append("--overwrite")
    return command


def run_foreground(command: list[str], log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[RUN] " + " ".join(command), flush=True)
    print(f"[LOG] {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
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
        raise subprocess.CalledProcessError(code, command)


def main() -> None:
    args = parse_args()
    subjects = parse_subjects(args.subjects)
    paths = paths_for(args)
    ensure_protocol(paths, args, subjects)
    if args.stage == "audit":
        audit(paths, subjects)
        return
    if args.stage == "summary":
        run_foreground(
            [
                sys.executable,
                "-u",
                str(paths["repo"] / "summarize_vi_rawtf_safe_transfer.py"),
                "--root", str(paths["root"]),
                "--baseline_root", str(paths["baseline"]),
                "--subjects", ",".join(map(str, subjects)),
            ],
            paths["logs"] / f"summary_{datetime.now():%Y%m%d_%H%M%S}.log",
            paths["repo"],
        )
        return
    run_subjects = subjects if args.stage == "all" else (args.subject_id,)
    if args.stage == "run" and args.subject_id is None:
        raise ValueError("--subject_id is required for --stage run")
    counts = session_counts(str(paths["vi_root"]))
    for sid in run_subjects:
        assert sid is not None
        if counts.get(sid, 0) < 2:
            raise RuntimeError(f"S{sid:02d} requires at least two VI sessions")
        if complete(metrics_path(paths, sid)) and not args.force:
            print(f"[SKIP] S{sid:02d}: already complete")
            continue
        run_foreground(
            command_for(paths, args, sid),
            paths["logs"] / f"S{sid:02d}_{datetime.now():%Y%m%d_%H%M%S}.log",
            paths["repo"],
        )
        audit(paths, subjects)


if __name__ == "__main__":
    main()
