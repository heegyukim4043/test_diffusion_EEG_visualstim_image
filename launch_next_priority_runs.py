#!/usr/bin/env python3
"""
launch_next_priority_runs.py
────────────────────────────────────────────────────────────────────────────
Background launcher for the next run queue requested after the S24 r=32 VS result.

Priority presets:
  p0_exp43_s24  = Exp43 VI C0/C1, S24 first, VS init r=32
  p1_exp43_s01  = Exp43 VI C0/C1, S01 parallel candidate, VS init r=32
  p2_vs_s28     = S28 r=32 SD LoRA VS generation
  p3_vs_s02     = S02 r=32 SD LoRA VS generation

All outputs are written to Google Drive. This launcher does not create a
PROGRESS.md update and does not push result files to GitHub.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_SUPCON = "checkpoints_vsre_dino/20260604_091352_ch32_merged_ep200_supcon"
DRIVE_ROOT = "/content/drive/MyDrive/vsvi_data"
DRIVE_VS_CKPT = f"{DRIVE_ROOT}/checkpoints_vsre_lora_gen"
DRIVE_VI_CKPT = f"{DRIVE_ROOT}/checkpoints_exp43_vi_lora"
DRIVE_LOGS = f"{DRIVE_ROOT}/logs"


PRESET_ORDER = ("p0_exp43_s24", "p1_exp43_s01", "p2_vs_s28", "p3_vs_s02")


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print("$", shlex.join(cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def in_colab() -> bool:
    return os.path.isdir("/content")


def mount_drive_if_needed(force_remount: bool = False) -> None:
    if not in_colab():
        return
    if Path("/content/drive/MyDrive").exists() and not force_remount:
        return
    try:
        from google.colab import drive  # type: ignore
    except Exception as exc:
        raise RuntimeError("Google Colab drive module is unavailable") from exc
    drive.mount("/content/drive", force_remount=force_remount)


def remove_incompatible_torchao() -> None:
    if importlib.util.find_spec("torchao") is not None:
        run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
    probe = run(
        [
            sys.executable,
            "-c",
            "import importlib.util, peft; "
            "print('torchao_absent=', importlib.util.find_spec('torchao') is None); "
            "print('peft=', peft.__version__)",
        ],
        check=True,
        capture=True,
    )
    print(probe.stdout, flush=True)


def gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return ""


def active_training_jobs() -> list[str]:
    proc = subprocess.run(["ps", "-eo", "pid,args"], text=True, capture_output=True)
    if proc.returncode != 0:
        return []
    self_pid = str(os.getpid())
    needles = ("train_vs_re_lora_gen.py", "train_exp43_vi_lora.py")
    jobs = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("PID "):
            continue
        pid = stripped.split(maxsplit=1)[0]
        if pid == self_pid:
            continue
        if any(n in stripped for n in needles):
            jobs.append(stripped)
    return jobs


def assert_common_paths() -> None:
    for p in [DRIVE_ROOT, DRIVE_LOGS, DRIVE_VS_CKPT, DRIVE_VI_CKPT]:
        Path(p).mkdir(parents=True, exist_ok=True)
    if not Path("preproc_data_vi/images").is_dir():
        raise FileNotFoundError("Missing image root: preproc_data_vi/images")
    for i in range(1, 10):
        if not Path(f"preproc_data_vi/images/{i:02d}.png").is_file():
            raise FileNotFoundError(f"Missing target image: preproc_data_vi/images/{i:02d}.png")
    if not Path(DEFAULT_SUPCON).is_dir():
        raise FileNotFoundError(f"Missing SupCon root: {DEFAULT_SUPCON}")


def has_subject_data(root: str, sid: int) -> bool:
    p = Path(root)
    return bool(list(p.glob(f"preproc_subj_{sid:02d}_*.npz")) or list(p.glob(f"preproc_subj_{sid:02d}_*.mat")))


def preflight_preset(preset: str) -> None:
    assert_common_paths()
    if preset == "p0_exp43_s24":
        sid, root = 24, "preproc_vi_re"
    elif preset == "p1_exp43_s01":
        sid, root = 1, "preproc_vi_re"
    elif preset == "p2_vs_s28":
        sid, root = 28, "preproc_vs_re"
    elif preset == "p3_vs_s02":
        sid, root = 2, "preproc_vs_re"
    else:
        raise ValueError(f"Unknown preset: {preset}")

    if not Path(root).is_dir():
        raise FileNotFoundError(f"Missing data root: {root}")
    if not has_subject_data(root, sid):
        raise FileNotFoundError(f"No subject data for S{sid:02d} in {root}")
    supcon_file = Path(DEFAULT_SUPCON) / f"subj{sid:02d}_best.pt"
    if not supcon_file.is_file():
        raise FileNotFoundError(f"Missing SupCon checkpoint: {supcon_file}")

    print(f"preset={preset}")
    print(f"subject=S{sid:02d}")
    print(f"data_root={root}")
    print(f"supcon={supcon_file}")
    print("preflight=OK")


def maybe_grad_ckpt(args) -> bool:
    if args.grad_ckpt:
        return True
    return "t4" in gpu_name().lower()


def build_cmd(args) -> tuple[list[str], str]:
    grad_ckpt = maybe_grad_ckpt(args)
    preset = args.preset

    if preset == "p0_exp43_s24":
        cmd = [
            sys.executable, "-u", "train_exp43_vi_lora.py",
            "--subject_ids", "24",
            "--conditions", "c0,c1",
            "--data_root", "preproc_vi_re",
            "--img_root", "preproc_data_vi/images",
            "--supcon_ckpt", DEFAULT_SUPCON,
            "--vs_ckpt_roots", DRIVE_VS_CKPT, "checkpoints_vsre_lora_gen",
            "--init_lora_ckpt", "auto",
            "--ckpt_root", DRIVE_VI_CKPT,
            "--lora_r", "32",
            "--n_eeg_tokens", "16",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--per_class_total", "0" if args.full_vi else str(args.per_class_total),
            "--eval_n_samples", str(args.eval_n_samples),
            "--fp16",
        ]
        log_prefix = "exp43_s24_c0c1_r32_tok16"

    elif preset == "p1_exp43_s01":
        cmd = [
            sys.executable, "-u", "train_exp43_vi_lora.py",
            "--subject_ids", "1",
            "--conditions", "c0,c1",
            "--data_root", "preproc_vi_re",
            "--img_root", "preproc_data_vi/images",
            "--supcon_ckpt", DEFAULT_SUPCON,
            "--vs_ckpt_roots", DRIVE_VS_CKPT, "checkpoints_vsre_lora_gen",
            "--init_lora_ckpt", "auto",
            "--ckpt_root", DRIVE_VI_CKPT,
            "--lora_r", "32",
            "--n_eeg_tokens", "16",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--per_class_total", "0" if args.full_vi else str(args.per_class_total),
            "--eval_n_samples", str(args.eval_n_samples),
            "--fp16",
        ]
        log_prefix = "exp43_s01_c0c1_r32_tok16"

    elif preset == "p2_vs_s28":
        cmd = [
            sys.executable, "-u", "train_vs_re_lora_gen.py",
            "--subject_ids", "28",
            "--lora_r", "32",
            "--n_eeg_tokens", "16",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--img_root", "preproc_data_vi/images",
            "--supcon_ckpt", DEFAULT_SUPCON,
            "--ckpt_root", DRIVE_VS_CKPT,
            "--fp16",
        ]
        log_prefix = "vs_s28_lora_r32_tok16"

    elif preset == "p3_vs_s02":
        cmd = [
            sys.executable, "-u", "train_vs_re_lora_gen.py",
            "--subject_ids", "2",
            "--lora_r", "32",
            "--n_eeg_tokens", "16",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--img_root", "preproc_data_vi/images",
            "--supcon_ckpt", DEFAULT_SUPCON,
            "--ckpt_root", DRIVE_VS_CKPT,
            "--fp16",
        ]
        log_prefix = "vs_s02_lora_r32_tok16"

    else:
        raise ValueError(f"Unknown preset: {preset}")

    if grad_ckpt:
        cmd.append("--grad_ckpt")
    return cmd, log_prefix


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch next prioritized EEG-to-image experiments in background.")
    parser.add_argument("--preset", required=True, choices=PRESET_ORDER)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--per_class_total", type=int, default=60,
                        help="Exp43 VI cap per class before split. Use --full_vi for all trials.")
    parser.add_argument("--full_vi", action="store_true",
                        help="Exp43 only: use all VI trials instead of per-class 60 cap.")
    parser.add_argument("--eval_n_samples", type=int, default=54)
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--allow_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_remount", action="store_true")
    args = parser.parse_args()

    mount_drive_if_needed(force_remount=args.force_remount)
    remove_incompatible_torchao()

    try:
        import torch
        print(f"torch={torch.__version__}")
        print(f"cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu={torch.cuda.get_device_name(0)}")
    except Exception as exc:
        raise RuntimeError("torch import/GPU check failed") from exc

    preflight_preset(args.preset)

    jobs = active_training_jobs()
    if jobs and not args.allow_existing:
        print("Active training jobs detected:")
        for job in jobs:
            print("  ", job)
        raise SystemExit("Refusing duplicate launch. Stop the existing training process or pass --allow_existing.")

    cmd, log_prefix = build_cmd(args)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(DRIVE_LOGS) / f"{log_prefix}_{ts}.log"
    pid_path = Path(DRIVE_LOGS) / f"{log_prefix}_latest.pid"

    print("Command:", shlex.join(cmd))
    print("Log    :", log_path)
    print("PID file:", pid_path)
    print("Results: Drive only")

    if args.dry_run:
        print("Dry run only. No process launched.")
        return

    if args.foreground:
        with open(log_path, "a", buffering=1) as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        raise SystemExit(proc.returncode)

    with open(log_path, "a", buffering=1) as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pid_path.write_text(str(proc.pid))

    print(f"Started background run: {args.preset}")
    print("PID:", proc.pid)
    print("Monitor:")
    print(f"  tail -f {log_path}")
    print("  pgrep -af 'train_exp43_vi_lora.py|train_vs_re_lora_gen.py'")
    print("  nvidia-smi")


if __name__ == "__main__":
    main()
