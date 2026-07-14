"""Foreground, resumable runner for the leakage-controlled S24 EA pilot."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dataset_vs_re import session_counts


STAGES = (
    "vs_pretrain",
    "zero_shot_strict",
    "zero_shot_calibrated",
    "vi_only",
    "vs_to_vi",
)
FIXED = {
    "n_ch": 32,
    "epochs": 100,
    "batch_size": 32,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "patience": 20,
    "w_supcon": 0.2,
    "temperature": 0.07,
    "label_smoothing": 0.1,
    "hidden_dim": 256,
    "latent_dim": 256,
    "n_heads": 4,
    "n_layers": 2,
    "dropout": 0.1,
    "shrinkage": 0.05,
    "eigen_eps": 1e-6,
    "alignment": "domain train split only",
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "next", *STAGES), default="audit")
    parser.add_argument("--subject_id", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo", default=str(repo))
    parser.add_argument("--vs_root", default="/content/drive/MyDrive/vsvi_data/preproc_vs_re")
    parser.add_argument("--vi_root", default="/content/drive/MyDrive/vsvi_data/preproc_vi_re")
    parser.add_argument("--out_root", default="/content/drive/MyDrive/vsvi_data/vi_ea_ablation")
    parser.add_argument("--batch_size", type=int, default=FIXED["batch_size"])
    parser.add_argument("--epochs", type=int, default=FIXED["epochs"])
    parser.add_argument("--patience", type=int, default=FIXED["patience"])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def paths(args: argparse.Namespace) -> dict[str, Path]:
    root = Path(args.out_root) / f"seed{args.seed}" / f"S{args.subject_id:02d}"
    return {
        "repo": Path(args.repo),
        "vs_root": Path(args.vs_root),
        "vi_root": Path(args.vi_root),
        "root": root,
        "logs": root / "logs",
    }


def metrics_path(p: dict[str, Path], stage: str) -> Path:
    return p["root"] / stage / "metrics.json"


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        return all(key in result for key in ("subject", "stage", "balanced_accuracy"))
    except (OSError, ValueError):
        return False


def ensure_protocol(p: dict[str, Path], args: argparse.Namespace) -> None:
    payload = dict(FIXED)
    payload.update({
        "subject": args.subject_id,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "fp16": args.fp16,
        "vs_root": args.vs_root,
        "vi_root": args.vi_root,
        "stages": list(STAGES),
    })
    path = p["root"] / "protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            old = json.load(handle)
        if old != payload:
            raise RuntimeError(f"Protocol mismatch at {path}; use a different out_root")
        return
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def audit(p: dict[str, Path], args: argparse.Namespace) -> Path:
    vs_counts = session_counts(str(p["vs_root"]))
    vi_counts = session_counts(str(p["vi_root"]))
    row = {
        "subject": args.subject_id,
        "vs_sessions": vs_counts.get(args.subject_id, 0),
        "vi_sessions": vi_counts.get(args.subject_id, 0),
        **{stage: int(complete(metrics_path(p, stage))) for stage in STAGES},
    }
    print("subject vs_sessions vi_sessions " + " ".join(STAGES))
    print(
        f"S{args.subject_id:02d} {row['vs_sessions']:11d} {row['vi_sessions']:11d} "
        + " ".join(str(row[stage]) for stage in STAGES)
    )
    path = p["root"] / "audit.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"[Saved] {path}")
    return path


def next_stage(p: dict[str, Path]) -> str | None:
    for stage in STAGES:
        if not complete(metrics_path(p, stage)):
            return stage
    return None


def command(args: argparse.Namespace, p: dict[str, Path], stage: str) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(p["repo"] / "train_vi_ea_ablation.py"),
        "--stage", stage,
        "--subject_id", str(args.subject_id),
        "--vs_root", str(p["vs_root"]),
        "--vi_root", str(p["vi_root"]),
        "--out_dir", str(p["root"] / stage),
        "--seed", str(args.seed),
        "--n_ch", str(FIXED["n_ch"]),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(FIXED["lr"]),
        "--weight_decay", str(FIXED["weight_decay"]),
        "--patience", str(args.patience),
        "--w_supcon", str(FIXED["w_supcon"]),
        "--temperature", str(FIXED["temperature"]),
        "--label_smoothing", str(FIXED["label_smoothing"]),
        "--hidden_dim", str(FIXED["hidden_dim"]),
        "--latent_dim", str(FIXED["latent_dim"]),
        "--n_heads", str(FIXED["n_heads"]),
        "--n_layers", str(FIXED["n_layers"]),
        "--dropout", str(FIXED["dropout"]),
        "--shrinkage", str(FIXED["shrinkage"]),
        "--eigen_eps", str(FIXED["eigen_eps"]),
    ]
    if stage in ("zero_shot_strict", "zero_shot_calibrated", "vs_to_vi"):
        checkpoint = p["root"] / "vs_pretrain" / "encoder_best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"EA VS checkpoint missing: {checkpoint}")
        cmd.extend(("--init_ckpt", str(checkpoint)))
    if stage == "zero_shot_strict":
        alignment = p["root"] / "vs_pretrain" / "alignment_matrix.npy"
        if not alignment.is_file():
            raise FileNotFoundError(f"VS alignment matrix missing: {alignment}")
        cmd.extend(("--source_alignment", str(alignment)))
    if args.fp16:
        cmd.append("--fp16")
    if args.force:
        cmd.append("--overwrite")
    return cmd


def run_foreground(cmd: list[str], log: Path, cwd: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("[RUN] " + " ".join(cmd), flush=True)
    print(f"[LOG] {log}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
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
        raise subprocess.CalledProcessError(code, cmd)


def main() -> None:
    args = parse_args()
    p = paths(args)
    ensure_protocol(p, args)
    if args.stage == "audit":
        audit(p, args)
        return
    stage = next_stage(p) if args.stage == "next" else args.stage
    if stage is None:
        print(f"S{args.subject_id:02d}: all EA stages complete")
        return
    if complete(metrics_path(p, stage)) and not args.force:
        print(f"S{args.subject_id:02d} {stage}: already complete; use --force to rerun")
        return
    script = p["repo"] / "train_vi_ea_ablation.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    log = p["logs"] / f"{stage}_{datetime.now():%Y%m%d_%H%M%S}.log"
    run_foreground(command(args, p, stage), log, p["repo"])
    print(f"[DONE] S{args.subject_id:02d} EA {stage}")
    audit(p, args)


if __name__ == "__main__":
    main()
