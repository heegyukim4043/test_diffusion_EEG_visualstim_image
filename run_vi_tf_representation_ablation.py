"""Foreground, resumable runner for the raw/TF/raw+TF transfer ablation.

The default cohort is the S24 pilot.  After the pilot is frozen, pass the full
multi-session cohort explicitly with ``--subjects 1,2,9,18,24,28,29``.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dataset_vs_re import session_counts


REPRESENTATIONS = ("raw", "tf", "raw_tf")
STAGES = ("vs_pretrain", "zero_shot", "vi_only", "vs_to_vi")
FIXED = {
    "sampling_rate": 1024.0,
    "n_ch": 32,
    "epochs": 100,
    "batch_size": 32,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "patience": 20,
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
    "split": "stratified 80/10/10 per class using the same seed in VS and VI",
}


def parse_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("At least one subject is required")
    return values


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "next", "summary", *STAGES), default="audit")
    parser.add_argument("--subject_id", type=int)
    parser.add_argument("--representation", choices=REPRESENTATIONS)
    parser.add_argument("--subjects", default="24")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo", default=str(repo))
    parser.add_argument("--vs_root", default="/content/drive/MyDrive/vsvi_data/preproc_vs_re")
    parser.add_argument("--vi_root", default="/content/drive/MyDrive/vsvi_data/preproc_vi_re")
    parser.add_argument(
        "--out_root",
        default="/content/drive/MyDrive/vsvi_data/vi_tf_representation_ablation",
    )
    parser.add_argument("--batch_size", type=int, default=FIXED["batch_size"])
    parser.add_argument("--epochs", type=int, default=FIXED["epochs"])
    parser.add_argument("--patience", type=int, default=FIXED["patience"])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_paths(args: argparse.Namespace) -> dict[str, Path]:
    seed_root = Path(args.out_root) / f"seed{args.seed}"
    return {
        "repo": Path(args.repo),
        "vs_root": Path(args.vs_root),
        "vi_root": Path(args.vi_root),
        "seed_root": seed_root,
        "logs": seed_root / "logs",
    }


def stage_dir(paths: dict[str, Path], sid: int, representation: str, stage: str) -> Path:
    return paths["seed_root"] / f"S{sid:02d}" / representation / stage


def metrics_path(paths: dict[str, Path], sid: int, representation: str, stage: str) -> Path:
    return stage_dir(paths, sid, representation, stage) / "metrics.json"


def vs_checkpoint(paths: dict[str, Path], sid: int, representation: str) -> Path:
    return stage_dir(paths, sid, representation, "vs_pretrain") / "encoder_best.pt"


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        return all(key in result for key in ("subject", "stage", "representation", "balanced_accuracy"))
    except (OSError, ValueError):
        return False


def protocol_payload(args: argparse.Namespace, subjects: tuple[int, ...]) -> dict:
    payload = dict(FIXED)
    payload.update({
        "seed": args.seed,
        "subjects": list(subjects),
        "representations": list(REPRESENTATIONS),
        "stages": list(STAGES),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "fp16": args.fp16,
        "vs_root": args.vs_root,
        "vi_root": args.vi_root,
    })
    return payload


def ensure_protocol(paths: dict[str, Path], args: argparse.Namespace, subjects: tuple[int, ...]) -> None:
    path = paths["seed_root"] / "protocol.json"
    payload = protocol_payload(args, subjects)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous != payload:
            raise RuntimeError(
                f"Protocol mismatch at {path}. Use a new out_root/seed rather than mixing settings."
            )
        return
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def audit(paths: dict[str, Path], subjects: tuple[int, ...]) -> Path:
    vs_counts = session_counts(str(paths["vs_root"]))
    vi_counts = session_counts(str(paths["vi_root"]))
    rows = []
    print("subject representation vs_sessions vi_sessions vs_pretrain zero_shot vi_only vs_to_vi")
    for sid in subjects:
        for representation in REPRESENTATIONS:
            row = {
                "subject": sid,
                "representation": representation,
                "vs_sessions": vs_counts.get(sid, 0),
                "vi_sessions": vi_counts.get(sid, 0),
                **{
                    stage: int(complete(metrics_path(paths, sid, representation, stage)))
                    for stage in STAGES
                },
            }
            rows.append(row)
            print(
                f"S{sid:02d}     {representation:8s}       {row['vs_sessions']:3d}        "
                f"{row['vi_sessions']:3d}          {row['vs_pretrain']}          "
                f"{row['zero_shot']}        {row['vi_only']}        {row['vs_to_vi']}"
            )
    path = paths["seed_root"] / "audit.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] {path}")
    return path


def preflight(paths: dict[str, Path], sid: int) -> None:
    script = paths["repo"] / "train_vi_tf_representation_ablation.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    vs_counts = session_counts(str(paths["vs_root"]))
    vi_counts = session_counts(str(paths["vi_root"]))
    if vs_counts.get(sid, 0) < 1:
        raise FileNotFoundError(f"S{sid:02d} VS data missing")
    if vi_counts.get(sid, 0) < 2:
        raise RuntimeError(f"S{sid:02d} requires multi-session VI data; found {vi_counts.get(sid, 0)}")


def command_for(
    args: argparse.Namespace,
    paths: dict[str, Path],
    sid: int,
    representation: str,
    stage: str,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(paths["repo"] / "train_vi_tf_representation_ablation.py"),
        "--stage", stage,
        "--representation", representation,
        "--subject_id", str(sid),
        "--vs_root", str(paths["vs_root"]),
        "--vi_root", str(paths["vi_root"]),
        "--out_dir", str(stage_dir(paths, sid, representation, stage)),
        "--seed", str(args.seed),
        "--sampling_rate", str(FIXED["sampling_rate"]),
        "--n_ch", str(FIXED["n_ch"]),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(FIXED["lr"]),
        "--weight_decay", str(FIXED["weight_decay"]),
        "--patience", str(args.patience),
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
    if stage in ("zero_shot", "vs_to_vi"):
        checkpoint = vs_checkpoint(paths, sid, representation)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Matching VS checkpoint missing: {checkpoint}. Run vs_pretrain first."
            )
        command.extend(("--init_ckpt", str(checkpoint)))
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


def next_incomplete(paths: dict[str, Path], sid: int, representation: str) -> str | None:
    for stage in STAGES:
        if not complete(metrics_path(paths, sid, representation, stage)):
            return stage
    return None


def main() -> None:
    args = parse_args()
    subjects = parse_list(args.subjects)
    paths = build_paths(args)
    ensure_protocol(paths, args, subjects)

    if args.stage == "audit":
        audit(paths, subjects)
        return
    if args.stage == "summary":
        run_foreground(
            [
                sys.executable,
                "-u",
                str(paths["repo"] / "summarize_vi_tf_representation_ablation.py"),
                "--root", str(paths["seed_root"]),
                "--subjects", ",".join(map(str, subjects)),
            ],
            paths["logs"] / f"summary_{datetime.now():%Y%m%d_%H%M%S}.log",
            paths["repo"],
        )
        return

    if args.subject_id is None or args.representation is None:
        raise ValueError("--subject_id and --representation are required")
    if args.subject_id not in subjects:
        raise ValueError(f"S{args.subject_id:02d} is not in --subjects {subjects}")
    preflight(paths, args.subject_id)
    stage = (
        next_incomplete(paths, args.subject_id, args.representation)
        if args.stage == "next"
        else args.stage
    )
    if stage is None:
        print(f"S{args.subject_id:02d} {args.representation}: all stages complete")
        return
    artifact = metrics_path(paths, args.subject_id, args.representation, stage)
    if complete(artifact) and not args.force:
        print(
            f"S{args.subject_id:02d} {args.representation} {stage}: "
            "already complete; use --force to rerun"
        )
        return

    log = paths["logs"] / (
        f"S{args.subject_id:02d}_{args.representation}_{stage}_"
        f"{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    command = command_for(args, paths, args.subject_id, args.representation, stage)
    run_foreground(command, log, paths["repo"])
    print(f"[DONE] S{args.subject_id:02d} {args.representation} {stage}")
    audit(paths, subjects)


if __name__ == "__main__":
    main()
