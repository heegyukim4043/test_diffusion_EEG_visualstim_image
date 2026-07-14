"""Foreground, resumable runner for raw+TF joint replay and CC-MMD."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dataset_vs_re import session_counts


STAGES = ("replay", "ccmmd")
FIXED = {
    "representation": "raw_tf",
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
    "vs_replay_weight": 0.5,
    "samples_per_class": 2,
    "selection": "VI validation BAC, Top-3 tie-break, then smaller lambda",
}


def parse_subjects(raw: str) -> tuple[int, ...]:
    subjects = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not subjects:
        raise ValueError("At least one subject is required")
    return subjects


def parse_roots(raw: str) -> tuple[Path, ...]:
    roots = tuple(Path(value.strip()) for value in raw.split(",") if value.strip())
    if not roots:
        raise ValueError("At least one baseline root is required")
    return roots


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "next", "summary", *STAGES), default="audit")
    parser.add_argument("--subject_id", type=int)
    parser.add_argument("--subjects", default="24")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo", default=str(repo))
    parser.add_argument("--vs_root", default="/content/drive/MyDrive/vsvi_data/preproc_vs_re")
    parser.add_argument("--vi_root", default="/content/drive/MyDrive/vsvi_data/preproc_vi_re")
    parser.add_argument(
        "--baseline_roots",
        default=(
            "/content/drive/MyDrive/vsvi_data/vi_tf_representation_ablation,"
            "/content/drive/MyDrive/vsvi_data/vi_rawtf_confirmatory_multi"
        ),
    )
    parser.add_argument(
        "--out_root",
        default="/content/drive/MyDrive/vsvi_data/vi_rawtf_ccmmd_s24",
    )
    parser.add_argument("--lambda_candidates", default="0.01,0.05,0.1")
    parser.add_argument("--fixed_lambda", type=float, default=-1.0)
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


def stage_dir(paths: dict[str, Path], sid: int, stage: str) -> Path:
    return paths["seed_root"] / f"S{sid:02d}" / stage


def metrics_path(paths: dict[str, Path], sid: int, stage: str) -> Path:
    return stage_dir(paths, sid, stage) / "metrics.json"


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        return all(
            key in result
            for key in ("subject", "mode", "balanced_accuracy", "selected_lambda")
        )
    except (OSError, ValueError):
        return False


def baseline_seed_roots(args: argparse.Namespace) -> tuple[Path, ...]:
    return tuple(root / f"seed{args.seed}" for root in parse_roots(args.baseline_roots))


def resolve_baseline(paths: dict[str, Path], args: argparse.Namespace, sid: int) -> dict[str, Path]:
    candidates = []
    for root in baseline_seed_roots(args):
        subject_root = root / f"S{sid:02d}" / "raw_tf"
        checkpoint = subject_root / "vs_pretrain" / "encoder_best.pt"
        vi_only = subject_root / "vi_only" / "metrics.json"
        sequential = subject_root / "vs_to_vi" / "metrics.json"
        if checkpoint.is_file() and vi_only.is_file() and sequential.is_file():
            candidates.append({
                "root": subject_root,
                "checkpoint": checkpoint,
                "vi_only": vi_only,
                "sequential": sequential,
            })
    if not candidates:
        searched = ", ".join(str(root) for root in baseline_seed_roots(args))
        raise FileNotFoundError(
            f"Complete raw_tf baseline missing for S{sid:02d}; searched {searched}"
        )
    selected = candidates[0]
    if len(candidates) > 1:
        print(f"[WARN] Multiple S{sid:02d} baselines; selecting first:", flush=True)
        for item in candidates:
            print(f"  candidate: {item['root']}", flush=True)
    print(f"[Baseline S{sid:02d}] {selected['root']}", flush=True)
    return selected


def protocol_payload(args: argparse.Namespace, subjects: tuple[int, ...]) -> dict:
    payload = dict(FIXED)
    payload.update({
        "subjects": list(subjects),
        "seed": args.seed,
        "stages": list(STAGES),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "fp16": args.fp16,
        "lambda_candidates": args.lambda_candidates,
        "fixed_lambda": args.fixed_lambda,
        "vs_root": args.vs_root,
        "vi_root": args.vi_root,
        "baseline_roots": args.baseline_roots,
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
            raise RuntimeError(f"Protocol mismatch at {path}; use a new out_root")
        return
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def audit(paths: dict[str, Path], args: argparse.Namespace, subjects: tuple[int, ...]) -> Path:
    vs_counts = session_counts(str(paths["vs_root"]))
    vi_counts = session_counts(str(paths["vi_root"]))
    rows = []
    print("subject vs_sessions vi_sessions baseline replay ccmmd")
    for sid in subjects:
        try:
            baseline = resolve_baseline(paths, args, sid)
            baseline_ok = 1
            baseline_root = str(baseline["root"])
        except FileNotFoundError:
            baseline_ok = 0
            baseline_root = ""
        row = {
            "subject": sid,
            "vs_sessions": vs_counts.get(sid, 0),
            "vi_sessions": vi_counts.get(sid, 0),
            "baseline": baseline_ok,
            "baseline_root": baseline_root,
            **{stage: int(complete(metrics_path(paths, sid, stage))) for stage in STAGES},
        }
        rows.append(row)
        print(
            f"S{sid:02d} {row['vs_sessions']:11d} {row['vi_sessions']:11d} "
            f"{baseline_ok:8d} {row['replay']:6d} {row['ccmmd']:5d}"
        )
    path = paths["seed_root"] / "audit.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] {path}")
    return path


def preflight(paths: dict[str, Path], args: argparse.Namespace, sid: int) -> dict[str, Path]:
    script = paths["repo"] / "train_vi_rawtf_ccmmd.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    vs_counts = session_counts(str(paths["vs_root"]))
    vi_counts = session_counts(str(paths["vi_root"]))
    if vs_counts.get(sid, 0) < 1 or vi_counts.get(sid, 0) < 2:
        raise RuntimeError(
            f"S{sid:02d} requires VS and >=2 VI sessions; "
            f"found VS={vs_counts.get(sid, 0)}, VI={vi_counts.get(sid, 0)}"
        )
    return resolve_baseline(paths, args, sid)


def next_stage(paths: dict[str, Path], sid: int) -> str | None:
    for stage in STAGES:
        if not complete(metrics_path(paths, sid, stage)):
            return stage
    return None


def command_for(
    paths: dict[str, Path],
    args: argparse.Namespace,
    sid: int,
    stage: str,
    baseline: dict[str, Path],
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(paths["repo"] / "train_vi_rawtf_ccmmd.py"),
        "--mode", stage,
        "--subject_id", str(sid),
        "--vs_root", str(paths["vs_root"]),
        "--vi_root", str(paths["vi_root"]),
        "--vs_ckpt", str(baseline["checkpoint"]),
        "--out_dir", str(stage_dir(paths, sid, stage)),
        "--seed", str(args.seed),
        "--lambda_candidates", args.lambda_candidates,
        "--fixed_lambda", str(args.fixed_lambda),
        "--vs_replay_weight", str(FIXED["vs_replay_weight"]),
        "--samples_per_class", str(FIXED["samples_per_class"]),
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
    paths = build_paths(args)
    ensure_protocol(paths, args, subjects)
    if args.stage == "audit":
        audit(paths, args, subjects)
        return
    if args.stage == "summary":
        run_foreground(
            [
                sys.executable,
                "-u",
                str(paths["repo"] / "summarize_vi_rawtf_ccmmd.py"),
                "--root", str(paths["seed_root"]),
                "--subjects", ",".join(map(str, subjects)),
                "--baseline_roots", args.baseline_roots,
                "--seed", str(args.seed),
            ],
            paths["logs"] / f"summary_{datetime.now():%Y%m%d_%H%M%S}.log",
            paths["repo"],
        )
        return
    if args.subject_id is None:
        raise ValueError("--subject_id is required")
    if args.subject_id not in subjects:
        raise ValueError(f"S{args.subject_id:02d} is not in --subjects {subjects}")
    baseline = preflight(paths, args, args.subject_id)
    stage = next_stage(paths, args.subject_id) if args.stage == "next" else args.stage
    if stage is None:
        print(f"S{args.subject_id:02d}: all CC-MMD stages complete")
        return
    if complete(metrics_path(paths, args.subject_id, stage)) and not args.force:
        print(f"S{args.subject_id:02d} {stage}: already complete; use --force to rerun")
        return
    log = paths["logs"] / (
        f"S{args.subject_id:02d}_{stage}_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    run_foreground(
        command_for(paths, args, args.subject_id, stage, baseline),
        log,
        paths["repo"],
    )
    print(f"[DONE] S{args.subject_id:02d} {stage}")
    audit(paths, args, subjects)


if __name__ == "__main__":
    main()
