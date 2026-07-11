"""Foreground/resumable Colab queue for the confirmatory single-session cohort.

One invocation runs at most one GPU-heavy stage. All checkpoints, logs, protocol
metadata, and evaluations are stored on Drive, so ``--stage next`` can resume
after a Colab runtime reset without selecting subjects based on their results.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SUBJECTS = (3, 5, 6, 11, 20, 21, 22, 23)
STAGES = ("supcon", "vs_lora", "exp43_c0", "exp43_c1", "eval_c0", "eval_c1")

FIXED_CONFIG = {
    "cohort": "single_session_confirmatory",
    "subjects": list(SUBJECTS),
    "n_channels": 32,
    "supcon_epochs": 200,
    "supcon_lr": 1e-4,
    "supcon_temperature": 0.07,
    "supcon_loss_type": "supcon",
    "supcon_encoder_type": "v2",
    "vs_lora_epochs": 100,
    "vi_lora_epochs": 100,
    "lora_rank": 32,
    "lora_alpha": 32,
    "n_eeg_tokens": 16,
    "per_class_total": 0,
    "ddim_steps": 30,
    "evaluation_noise_seed": 20260711,
    "evaluation_split_seed": None,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        choices=("audit", "next", "summary", *STAGES),
        default="audit",
    )
    p.add_argument("--subject_id", type=int, default=None)
    p.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    p.add_argument("--repo_root", default="/content/vsvi_project")
    p.add_argument("--drive_root", default="/content/drive/MyDrive/vsvi_data")
    p.add_argument("--batch_size_supcon", type=int, default=64)
    p.add_argument("--batch_size_lora", type=int, default=2)
    p.add_argument("--batch_size_eval", type=int, default=2)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def newest(paths) -> Path | None:
    files = [Path(path) for path in paths if Path(path).is_file()]
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def ensure_protocol(seed_root: Path, seed: int) -> None:
    path = seed_root / "protocol.json"
    payload = dict(FIXED_CONFIG)
    payload["training_seed"] = seed
    payload["evaluation_split_seed"] = seed
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                f"Frozen protocol mismatch in {path}. Use another root; do not overwrite it."
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def paths_for(args: argparse.Namespace) -> dict[str, Path]:
    repo = Path(args.repo_root)
    drive = Path(args.drive_root)
    seed_root = drive / "confirmatory_single_session" / f"seed{args.seed}"
    return {
        "repo": repo,
        "drive": drive,
        "seed_root": seed_root,
        "vs_data": drive / "preproc_vs_re",
        "vi_data": drive / "preproc_vi_re",
        "images": repo / "preproc_data_vi" / "images",
        "supcon_root": seed_root / "checkpoints_supcon",
        "vs_lora_root": seed_root / "checkpoints_vs_lora",
        "exp43_root": seed_root / "checkpoints_exp43",
        "eval_root": seed_root / "full_test_eval",
        "logs": seed_root / "logs",
    }


def artifacts(paths: dict[str, Path], sid: int) -> dict[str, Path | None]:
    return {
        "supcon": newest(paths["supcon_root"].glob(f"**/subj{sid:02d}_best.pt")),
        "vs_lora": newest(paths["vs_lora_root"].glob(f"**/subj{sid:02d}_lora_best.pt")),
        "exp43_c0": newest(paths["exp43_root"].glob(f"**/subj{sid:02d}_exp43_c0_lora_best.pt")),
        "exp43_c1": newest(paths["exp43_root"].glob(f"**/subj{sid:02d}_exp43_c1_lora_best.pt")),
        "eval_c0": (paths["eval_root"] / f"S{sid:02d}" / "c0" / "manifest.csv"),
        "eval_c1": (paths["eval_root"] / f"S{sid:02d}" / "c1" / "manifest.csv"),
    }


def is_complete(value: Path | None) -> bool:
    return value is not None and value.is_file()


def data_count(root: Path, sid: int) -> int:
    return len(list(root.glob(f"preproc_subj_{sid:02d}_*.npz")))


def write_audit(paths: dict[str, Path]) -> Path:
    rows = []
    for sid in SUBJECTS:
        found = artifacts(paths, sid)
        row = {
            "subject": sid,
            "vs_npz": data_count(paths["vs_data"], sid),
            "vi_npz": data_count(paths["vi_data"], sid),
        }
        row.update({stage: int(is_complete(found[stage])) for stage in STAGES})
        rows.append(row)
    output = paths["seed_root"] / "audit.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("subject vs vi  supcon vs c0 c1 e0 e1")
    for row in rows:
        print(
            f"S{row['subject']:02d}     {row['vs_npz']}  {row['vi_npz']}     "
            f"{row['supcon']}     {row['vs_lora']}  {row['exp43_c0']}  "
            f"{row['exp43_c1']}  {row['eval_c0']}  {row['eval_c1']}"
        )
    print(f"[Saved] {output}")
    return output


def run_foreground(cmd: list[str], log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[RUN]", " ".join(cmd), flush=True)
    print(f"[LOG] {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
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
            log.write(line)
            log.flush()
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def require_subject(args: argparse.Namespace) -> int:
    if args.subject_id is None:
        raise ValueError("--subject_id is required for this stage")
    if args.subject_id not in SUBJECTS:
        raise ValueError(f"S{args.subject_id:02d} is not in fixed cohort {SUBJECTS}")
    return args.subject_id


def require_preflight(paths: dict[str, Path], sid: int) -> None:
    if data_count(paths["vs_data"], sid) != 1:
        raise RuntimeError(f"S{sid:02d} must have exactly one VS npz")
    if data_count(paths["vi_data"], sid) != 1:
        raise RuntimeError(f"S{sid:02d} must have exactly one VI npz")
    for script in (
        "train_vs_re_dino.py",
        "train_vs_re_lora_gen.py",
        "train_exp43_vi_lora.py",
        "eval_exp43_full_test.py",
        "summarize_exp43_full_test.py",
    ):
        if not (paths["repo"] / script).is_file():
            raise FileNotFoundError(paths["repo"] / script)
    if importlib.util.find_spec("torchao") is not None:
        raise RuntimeError("torchao is installed. Run: pip uninstall -y torchao")


def stage_command(
    stage: str,
    args: argparse.Namespace,
    paths: dict[str, Path],
    sid: int,
    found: dict[str, Path | None],
) -> list[str]:
    python = sys.executable
    common_data = ["--img_root", str(paths["images"]), "--subject_ids", str(sid)]

    if stage == "supcon":
        return [
            python,
            "-u",
            str(paths["repo"] / "train_vs_re_dino.py"),
            "--data_root",
            str(paths["vs_data"]),
            *common_data,
            "--seed",
            str(args.seed),
            "--epochs",
            str(FIXED_CONFIG["supcon_epochs"]),
            "--batch_size",
            str(args.batch_size_supcon),
            "--lr",
            str(FIXED_CONFIG["supcon_lr"]),
            "--temperature",
            str(FIXED_CONFIG["supcon_temperature"]),
            "--loss_type",
            str(FIXED_CONFIG["supcon_loss_type"]),
            "--encoder_type",
            str(FIXED_CONFIG["supcon_encoder_type"]),
            "--ckpt_root",
            str(paths["supcon_root"]),
        ]

    supcon = found["supcon"]
    if not is_complete(supcon):
        raise FileNotFoundError(f"S{sid:02d} confirmatory SupCon checkpoint is missing")
    supcon_dir = str(supcon.parent)

    if stage == "vs_lora":
        return [
            python,
            "-u",
            str(paths["repo"] / "train_vs_re_lora_gen.py"),
            "--data_root",
            str(paths["vs_data"]),
            *common_data,
            "--seed",
            str(args.seed),
            "--epochs",
            str(FIXED_CONFIG["vs_lora_epochs"]),
            "--batch_size",
            str(args.batch_size_lora),
            "--lora_r",
            str(FIXED_CONFIG["lora_rank"]),
            "--lora_alpha",
            str(FIXED_CONFIG["lora_alpha"]),
            "--n_eeg_tokens",
            str(FIXED_CONFIG["n_eeg_tokens"]),
            "--supcon_ckpt",
            supcon_dir,
            "--ckpt_root",
            str(paths["vs_lora_root"]),
            "--fp16",
        ]

    vs_lora = found["vs_lora"]
    if stage in ("exp43_c0", "exp43_c1"):
        if not is_complete(vs_lora):
            raise FileNotFoundError(f"S{sid:02d} confirmatory VS LoRA checkpoint is missing")
        condition = stage[-2:]
        return [
            python,
            "-u",
            str(paths["repo"] / "train_exp43_vi_lora.py"),
            "--data_root",
            str(paths["vi_data"]),
            "--img_root",
            str(paths["images"]),
            "--subject_ids",
            str(sid),
            "--conditions",
            condition,
            "--seed",
            str(args.seed),
            "--epochs",
            str(FIXED_CONFIG["vi_lora_epochs"]),
            "--batch_size",
            str(args.batch_size_lora),
            "--lora_r",
            str(FIXED_CONFIG["lora_rank"]),
            "--lora_alpha",
            str(FIXED_CONFIG["lora_alpha"]),
            "--n_eeg_tokens",
            str(FIXED_CONFIG["n_eeg_tokens"]),
            "--per_class_total",
            "0",
            "--eval_n_samples",
            "18",
            "--supcon_ckpt",
            supcon_dir,
            "--init_lora_ckpt",
            str(vs_lora),
            "--ckpt_root",
            str(paths["exp43_root"]),
            "--fp16",
        ]

    if stage in ("eval_c0", "eval_c1"):
        condition = stage[-2:]
        checkpoint = found[f"exp43_{condition}"]
        if not is_complete(checkpoint):
            raise FileNotFoundError(f"S{sid:02d} {condition.upper()} checkpoint is missing")
        cmd = [
            python,
            "-u",
            str(paths["repo"] / "eval_exp43_full_test.py"),
            "--ckpt",
            str(checkpoint),
            "--supcon_ckpt",
            supcon_dir,
            "--data_root",
            str(paths["vi_data"]),
            "--img_root",
            str(paths["images"]),
            "--subject_id",
            str(sid),
            "--condition",
            condition,
            "--lora_r",
            str(FIXED_CONFIG["lora_rank"]),
            "--n_eeg_tokens",
            str(FIXED_CONFIG["n_eeg_tokens"]),
            "--per_class_total",
            "0",
            "--ddim_steps",
            str(FIXED_CONFIG["ddim_steps"]),
            "--batch_size",
            str(args.batch_size_eval),
            "--seed",
            str(FIXED_CONFIG["evaluation_noise_seed"]),
            "--split_seed",
            str(args.seed),
            "--out_root",
            str(paths["eval_root"]),
        ]
        if args.force:
            cmd.append("--overwrite")
        return cmd

    raise ValueError(stage)


def next_stage(found: dict[str, Path | None]) -> str | None:
    for stage in STAGES:
        if not is_complete(found[stage]):
            return stage
    return None


def main() -> None:
    args = parse_args()
    paths = paths_for(args)
    ensure_protocol(paths["seed_root"], args.seed)

    if args.stage == "audit":
        write_audit(paths)
        return
    if args.stage == "summary":
        run_foreground(
            [
                sys.executable,
                "-u",
                str(paths["repo"] / "summarize_exp43_full_test.py"),
                "--root",
                str(paths["eval_root"]),
            ],
            paths["logs"] / f"summary_{datetime.now():%Y%m%d_%H%M%S}.log",
            paths["repo"],
        )
        return

    sid = require_subject(args)
    require_preflight(paths, sid)
    found = artifacts(paths, sid)
    stage = next_stage(found) if args.stage == "next" else args.stage
    if stage is None:
        print(f"S{sid:02d} seed={args.seed}: all stages complete")
        return
    if is_complete(found[stage]) and not args.force:
        print(f"S{sid:02d} {stage}: already complete; use --force to rerun")
        return
    command = stage_command(stage, args, paths, sid, found)
    log_path = paths["logs"] / f"S{sid:02d}_{stage}_{datetime.now():%Y%m%d_%H%M%S}.log"
    run_foreground(command, log_path, paths["repo"])
    print(f"[DONE] S{sid:02d} seed={args.seed} stage={stage}")
    write_audit(paths)


if __name__ == "__main__":
    main()
