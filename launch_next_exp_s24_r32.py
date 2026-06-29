#!/usr/bin/env python3
"""
launch_next_exp_s24_r32.py
────────────────────────────────────────────────────────────────────────────
Next experiment launcher derived from the active HUMAN_DIRECTIVE.md Priority 0:
S24 SD1.5 LoRA VS generation, rank ablation r=32, 16 EEG tokens.

This launcher is designed for Google Colab GPU runtimes. It does not generate
results_lora_gen.csv and does not update PROGRESS.md. It only starts the next
verified training run and writes logs/checkpoints to Google Drive.
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


DEFAULT_DATA_ROOT = "preproc_vs_re"
DEFAULT_IMG_ROOT = "preproc_data_vi/images"
DEFAULT_SUPCON = "checkpoints_vsre_dino/20260604_091352_ch32_merged_ep200_supcon"
DEFAULT_CKPT_ROOT = "/content/drive/MyDrive/vsvi_data/checkpoints_vsre_lora_gen"
DEFAULT_LOG_DIR = "/content/drive/MyDrive/vsvi_data/logs"


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print("$", shlex.join(cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def in_colab() -> bool:
    return os.path.isdir("/content")


def mount_drive_if_needed(force_remount: bool = False) -> None:
    if not in_colab():
        return
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists() and not force_remount:
        return
    try:
        from google.colab import drive  # type: ignore
    except Exception as exc:  # pragma: no cover - only meaningful in Colab
        raise RuntimeError("Google Colab drive module is unavailable") from exc
    drive.mount("/content/drive", force_remount=force_remount)


def remove_incompatible_torchao() -> None:
    """PEFT 0.19 can error when an old torchao package is present.

    This project does not require torchao, so the safe setup is to remove it
    before LoRA injection occurs inside train_vs_re_lora_gen.py.
    """
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


def active_training_jobs() -> list[str]:
    proc = subprocess.run(["pgrep", "-af", "train_vs_re_lora_gen.py"], text=True, capture_output=True)
    if proc.returncode not in (0, 1):
        return []
    self_pid = str(os.getpid())
    jobs = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        if line.split(maxsplit=1)[0] == self_pid:
            continue
        jobs.append(line)
    return jobs


def preflight(args: argparse.Namespace) -> None:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch import failed") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. Use a Colab GPU runtime.")
    print(f"torch={torch.__version__}")
    print(f"gpu={torch.cuda.get_device_name(0)}")

    data_root = Path(args.data_root)
    img_root = Path(args.img_root)
    supcon_dir = Path(args.supcon_ckpt)
    supcon_file = supcon_dir / "subj24_best.pt"

    npz_files = sorted(data_root.glob("preproc_subj_24_*.npz"))
    checks = {
        "data_root": data_root.is_dir(),
        "S24_npz_nonzero": len(npz_files) > 0,
        "S24_supcon": supcon_file.is_file(),
        "images_01_09": all((img_root / f"{i:02d}.png").is_file() for i in range(1, 10)),
    }
    for key, ok in checks.items():
        print(f"{key:16s}: {'OK' if ok else 'FAIL'}")
    print(f"S24 npz count   : {len(npz_files)}")
    if not all(checks.values()):
        raise RuntimeError("Preflight failed. Fix paths before launching training.")

    Path(args.ckpt_root).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)


def build_train_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "train_vs_re_lora_gen.py",
        "--subject_ids",
        "24",
        "--lora_r",
        "32",
        "--n_eeg_tokens",
        "16",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--img_root",
        args.img_root,
        "--supcon_ckpt",
        args.supcon_ckpt,
        "--ckpt_root",
        args.ckpt_root,
        "--fp16",
    ]
    if args.grad_ckpt:
        cmd.append("--grad_ckpt")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch HUMAN_DIRECTIVE Priority-0 next experiment: S24 r=32 LoRA.")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--img_root", default=DEFAULT_IMG_ROOT)
    parser.add_argument("--supcon_ckpt", default=DEFAULT_SUPCON)
    parser.add_argument("--ckpt_root", default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--foreground", action="store_true", help="Run in foreground instead of background.")
    parser.add_argument("--allow_existing", action="store_true", help="Allow launch even if another train_vs_re_lora_gen.py is active.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_remount", action="store_true")
    args = parser.parse_args()

    mount_drive_if_needed(force_remount=args.force_remount)
    remove_incompatible_torchao()
    preflight(args)

    jobs = active_training_jobs()
    if jobs and not args.allow_existing:
        print("Active training jobs detected:")
        for job in jobs:
            print("  ", job)
        raise SystemExit("Refusing to start a duplicate run. Stop the existing process or pass --allow_existing.")

    cmd = build_train_cmd(args)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_dir) / f"s24_lora_r32_tok16_{ts}.log"
    pid_path = Path(args.log_dir) / "s24_lora_r32_latest.pid"

    print("Command:", shlex.join(cmd))
    print("Log    :", log_path)
    print("PID file:", pid_path)

    if args.dry_run:
        print("Dry run only. No training process launched.")
        return

    if args.foreground:
        with open(log_path, "a", buffering=1) as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        raise SystemExit(proc.returncode)

    with open(log_path, "a", buffering=1) as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pid_path.write_text(str(proc.pid))

    print("Started background S24 r=32 training")
    print("PID:", proc.pid)
    print("Monitor:")
    print(f"  tail -f {log_path}")
    print("  pgrep -af train_vs_re_lora_gen.py")
    print("  nvidia-smi")
    print("After epoch 1, verify:")
    print("  find /content/drive/MyDrive/vsvi_data/checkpoints_vsre_lora_gen -path '*lora_r32_ep100/subj24_lora_best.pt' -ls")


if __name__ == "__main__":
    main()
