"""Foreground, resumable runner for the multi-session VI transfer study.

Fixed cohort: S01, S02, S09, S18, S24, S28, S29.
One invocation runs at most one GPU-heavy subject/condition stage.
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


SUBJECTS = (1, 2, 9, 18, 24, 28, 29)
STAGES = ("zero_shot", "vi_only", "vs_to_vi")
FIXED_CONFIG = {
    "seed": 42,
    "epochs": 200,
    "batch_size": 64,
    "lr": 3e-4,
    "wd": 1e-4,
    "loss_type": "supcon",
    "w_supcon": 1.0,
    "w_proto": 1.0,
    "w_aux": 0.5,
    "patience": 0,
    "n_ch": 32,
    "dino_model": "dinov2_vits14",
    "split": "stratified 80/10/10 per class, seed 42",
}


def parse_args() -> argparse.Namespace:
    repo_default = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("audit", "next", "summary", *STAGES), default="audit")
    p.add_argument("--subject_id", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repo", default=str(repo_default))
    p.add_argument("--vi_root", default="/content/drive/MyDrive/vsvi_data/preproc_vi_re")
    p.add_argument(
        "--vs_ckpt_dir",
        default=str(repo_default / "checkpoints_vsre_dino" / "20260604_091352_ch32_merged_ep200_supcon"),
    )
    p.add_argument("--img_root", default=str(repo_default / "preproc_data_vi" / "images"))
    p.add_argument(
        "--out_root",
        default="/content/drive/MyDrive/vsvi_data/vi_encoder_transfer_multi",
    )
    p.add_argument("--batch_size", type=int, default=FIXED_CONFIG["batch_size"])
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def paths_for(args: argparse.Namespace) -> dict[str, Path]:
    seed_root = Path(args.out_root) / f"seed{args.seed}"
    return {
        "repo": Path(args.repo),
        "vi_root": Path(args.vi_root),
        "vs_ckpt_dir": Path(args.vs_ckpt_dir),
        "img_root": Path(args.img_root),
        "seed_root": seed_root,
        "logs": seed_root / "logs",
    }


def ensure_protocol(paths: dict[str, Path], args: argparse.Namespace) -> None:
    protocol = dict(FIXED_CONFIG)
    protocol["seed"] = args.seed
    protocol["batch_size"] = args.batch_size
    protocol["subjects"] = list(SUBJECTS)
    protocol["vi_root"] = str(paths["vi_root"])
    protocol["vs_ckpt_dir"] = str(paths["vs_ckpt_dir"])
    path = paths["seed_root"] / "protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            old = json.load(handle)
        if old != protocol:
            raise RuntimeError(
                f"Protocol mismatch at {path}. Use a different seed/out_root instead of mixing runs."
            )
        return
    with path.open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False)


def metrics_path(paths: dict[str, Path], sid: int, stage: str) -> Path:
    return paths["seed_root"] / f"S{sid:02d}" / stage / "metrics.json"


def is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        return all(key in result for key in ("subject", "mode", "top1", "top3", "top5"))
    except (OSError, ValueError):
        return False


def write_audit(paths: dict[str, Path]) -> Path:
    vi_counts = session_counts(str(paths["vi_root"]))
    rows = []
    print("subject  vi_sessions  vs_ckpt  zero_shot  vi_only  vs_to_vi")
    for sid in SUBJECTS:
        row = {
            "subject": sid,
            "vi_sessions": vi_counts.get(sid, 0),
            "vs_checkpoint": int((paths["vs_ckpt_dir"] / f"subj{sid:02d}_best.pt").is_file()),
            **{stage: int(is_complete(metrics_path(paths, sid, stage))) for stage in STAGES},
        }
        rows.append(row)
        print(
            f"S{sid:02d}      {row['vi_sessions']:>3}          {row['vs_checkpoint']}          "
            f"{row['zero_shot']}         {row['vi_only']}        {row['vs_to_vi']}"
        )
    path = paths["seed_root"] / "audit.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] {path}")
    return path


def require_preflight(paths: dict[str, Path], sid: int) -> None:
    for script in ("train_vi_encoder_transfer.py", "summarize_vi_encoder_transfer.py"):
        path = paths["repo"] / script
        if not path.is_file():
            raise FileNotFoundError(path)
    if not paths["img_root"].is_dir():
        raise FileNotFoundError(paths["img_root"])
    vi_counts = session_counts(str(paths["vi_root"]))
    if vi_counts.get(sid, 0) < 2:
        raise RuntimeError(f"S{sid:02d} is not multi-session in VI root: {vi_counts.get(sid, 0)}")
    checkpoint = paths["vs_ckpt_dir"] / f"subj{sid:02d}_best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)


def stage_command(stage: str, args: argparse.Namespace, paths: dict[str, Path], sid: int) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(paths["repo"] / "train_vi_encoder_transfer.py"),
        "--mode", stage,
        "--subject_id", str(sid),
        "--vi_root", str(paths["vi_root"]),
        "--img_root", str(paths["img_root"]),
        "--vs_ckpt", str(paths["vs_ckpt_dir"] / f"subj{sid:02d}_best.pt"),
        "--out_dir", str(paths["seed_root"] / f"S{sid:02d}" / stage),
        "--seed", str(args.seed),
        "--n_ch", str(FIXED_CONFIG["n_ch"]),
        "--dino_model", FIXED_CONFIG["dino_model"],
        "--epochs", str(FIXED_CONFIG["epochs"]),
        "--batch_size", str(args.batch_size),
        "--lr", str(FIXED_CONFIG["lr"]),
        "--wd", str(FIXED_CONFIG["wd"]),
        "--loss_type", FIXED_CONFIG["loss_type"],
        "--w_supcon", str(FIXED_CONFIG["w_supcon"]),
        "--w_proto", str(FIXED_CONFIG["w_proto"]),
        "--w_aux", str(FIXED_CONFIG["w_aux"]),
        "--patience", str(FIXED_CONFIG["patience"]),
    ]
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
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


def next_stage(paths: dict[str, Path], sid: int) -> str | None:
    for stage in STAGES:
        if not is_complete(metrics_path(paths, sid, stage)):
            return stage
    return None


def main() -> None:
    args = parse_args()
    paths = paths_for(args)
    ensure_protocol(paths, args)

    if args.stage == "audit":
        write_audit(paths)
        return
    if args.stage == "summary":
        run_foreground(
            [
                sys.executable,
                "-u",
                str(paths["repo"] / "summarize_vi_encoder_transfer.py"),
                "--root", str(paths["seed_root"]),
                "--subjects", ",".join(map(str, SUBJECTS)),
            ],
            paths["logs"] / f"summary_{datetime.now():%Y%m%d_%H%M%S}.log",
            paths["repo"],
        )
        return

    if args.subject_id is None:
        raise ValueError("--subject_id is required")
    sid = args.subject_id
    if sid not in SUBJECTS:
        raise ValueError(f"S{sid:02d} is not in fixed multi-session cohort {SUBJECTS}")
    require_preflight(paths, sid)
    stage = next_stage(paths, sid) if args.stage == "next" else args.stage
    if stage is None:
        print(f"S{sid:02d}: all stages complete")
        return
    artifact = metrics_path(paths, sid, stage)
    if is_complete(artifact) and not args.force:
        print(f"S{sid:02d} {stage}: already complete; use --force to rerun")
        return

    log_path = paths["logs"] / f"S{sid:02d}_{stage}_{datetime.now():%Y%m%d_%H%M%S}.log"
    run_foreground(stage_command(stage, args, paths, sid), log_path, paths["repo"])
    print(f"[DONE] S{sid:02d} seed={args.seed} stage={stage}")
    write_audit(paths)


if __name__ == "__main__":
    main()
