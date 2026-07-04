#!/usr/bin/env python3
"""
audit_launch_vs_lora_subjects.py
────────────────────────────────────────────────────────────────────────────
Audit VS SD1.5 LoRA checkpoints for the subject set used in the paper table and
launch one missing VS training job at a time.

Default subject set:
  S24, S01, S02, S18, S28, S29, S09

Outputs are Drive-only:
  - audit CSV: /content/drive/MyDrive/vsvi_data/audits/
  - logs:      /content/drive/MyDrive/vsvi_data/logs/
  - ckpts:     /content/drive/MyDrive/vsvi_data/checkpoints_vsre_lora_gen/

The script treats an existing subjXX_lora_best.pt as a usable VS model. If the
matching results_lora_gen.csv is also present, DINO metrics are reported. If only
the checkpoint exists, status is CKPT_ONLY, not MISSING_CKPT.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import importlib.util
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DRIVE_ROOT = "/content/drive/MyDrive/vsvi_data"
DRIVE_CKPT_ROOT = f"{DRIVE_ROOT}/checkpoints_vsre_lora_gen"
DRIVE_LOG_DIR = f"{DRIVE_ROOT}/logs"
DRIVE_AUDIT_DIR = f"{DRIVE_ROOT}/audits"

DEFAULT_DATA_ROOT = "preproc_vs_re"
DEFAULT_IMG_ROOT = "preproc_data_vi/images"

# Planned paper subject set. S18 keeps r=16 by default because the confirmed VS
# model/result is r=16. Use --rank_policy r32 to force r=32 checks/training.
SUBJECT_SPECS: dict[int, dict[str, int]] = {
    24: {"order": 1, "vs_per_class": 150, "vi_per_class": 150, "vs_total": 1350, "planned_rank": 32},
    1:  {"order": 2, "vs_per_class": 135, "vi_per_class": 135, "vs_total": 1215, "planned_rank": 32},
    2:  {"order": 3, "vs_per_class": 129, "vi_per_class": 135, "vs_total": 1161, "planned_rank": 32},
    18: {"order": 4, "vs_per_class": 120, "vi_per_class": 120, "vs_total": 1080, "planned_rank": 16},
    28: {"order": 5, "vs_per_class": 87,  "vi_per_class": 90,  "vs_total": 783,  "planned_rank": 32},
    29: {"order": 6, "vs_per_class": 75,  "vi_per_class": 75,  "vs_total": 675,  "planned_rank": 32},
    9:  {"order": 7, "vs_per_class": 57,  "vi_per_class": 60,  "vs_total": 513,  "planned_rank": 32},
}

# Known verified SupCon roots are tried first, then any checkpoints_vsre_dino/*
# directory containing subjXX_best.pt is accepted as a fallback.
KNOWN_SUPCON_DIRS = [
    "checkpoints_vsre_dino/20260604_091352_ch32_merged_ep200_supcon",
    "checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon",
    f"{DRIVE_ROOT}/checkpoints_vsre_dino/20260604_091352_ch32_merged_ep200_supcon",
    f"{DRIVE_ROOT}/checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon",
]


@dataclass
class ModelCandidate:
    ckpt_path: str
    ckpt_dir: str
    lora_r: Optional[int]
    top1: Optional[float]
    top3: Optional[float]
    top5: Optional[float]
    best_ep: Optional[int]
    result_csv: str
    mtime: float


@dataclass
class AuditRow:
    order: int
    sid: int
    vs_per_class: int
    vi_per_class: int
    vs_total: int
    planned_rank: int
    check_rank: str
    data_ok: bool
    supcon_ok: bool
    status: str
    ckpt_path: str = ""
    ckpt_dir: str = ""
    result_csv: str = ""
    top1: str = ""
    top3: str = ""
    top5: str = ""
    best_ep: str = ""
    supcon_dir: str = ""


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


def parse_subject_ids(text: str) -> list[int]:
    out: list[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    # keep requested order while removing duplicates
    seen = set()
    final = []
    for sid in out:
        if sid not in seen:
            final.append(sid)
            seen.add(sid)
    return final


def planned_rank_for(sid: int, rank_policy: str) -> Optional[int]:
    if rank_policy == "planned":
        return SUBJECT_SPECS.get(sid, {}).get("planned_rank", 32)
    if rank_policy == "r32":
        return 32
    if rank_policy == "r16":
        return 16
    if rank_policy == "any":
        return None
    raise ValueError(f"Unknown rank_policy: {rank_policy}")


def parse_lora_r_from_dir(path: str) -> Optional[int]:
    m = re.search(r"lora_r(\d+)", os.path.basename(path))
    if m:
        return int(m.group(1))
    return None


def read_result_row(csv_path: str, sid: int) -> tuple[Optional[int], Optional[float], Optional[float], Optional[float]]:
    if not csv_path or not os.path.isfile(csv_path):
        return None, None, None, None
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if int(row.get("sid", -1)) == sid:
                    best_ep = row.get("best_ep", "")
                    return (
                        int(float(best_ep)) if best_ep != "" else None,
                        float(row["top1"]) if row.get("top1", "") != "" else None,
                        float(row["top3"]) if row.get("top3", "") != "" else None,
                        float(row["top5"]) if row.get("top5", "") != "" else None,
                    )
    except Exception:
        return None, None, None, None
    return None, None, None, None


def checkpoint_roots(extra_roots: Iterable[str]) -> list[str]:
    roots = [DRIVE_CKPT_ROOT, "checkpoints_vsre_lora_gen"]
    roots.extend(extra_roots)
    out = []
    seen = set()
    for r in roots:
        if not r or r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


def find_model_candidates(sid: int, rank: Optional[int], roots: Iterable[str]) -> list[ModelCandidate]:
    candidates: list[ModelCandidate] = []
    for root in roots:
        if not root or not Path(root).exists():
            continue
        pattern = os.path.join(root, "**", f"subj{sid:02d}_lora_best.pt")
        for ckpt_path in glob.glob(pattern, recursive=True):
            ckpt_dir = os.path.dirname(ckpt_path)
            ckpt_rank = parse_lora_r_from_dir(ckpt_dir)
            if rank is not None and ckpt_rank != rank:
                continue
            result_csv = os.path.join(ckpt_dir, "results_lora_gen.csv")
            best_ep, top1, top3, top5 = read_result_row(result_csv, sid)
            candidates.append(
                ModelCandidate(
                    ckpt_path=ckpt_path,
                    ckpt_dir=ckpt_dir,
                    lora_r=ckpt_rank,
                    top1=top1,
                    top3=top3,
                    top5=top5,
                    best_ep=best_ep,
                    result_csv=result_csv if os.path.isfile(result_csv) else "",
                    mtime=os.path.getmtime(ckpt_path),
                )
            )
    candidates.sort(
        key=lambda c: (
            c.top1 is not None,
            c.top1 if c.top1 is not None else -1.0,
            c.mtime,
        ),
        reverse=True,
    )
    return candidates


def has_vs_data(data_root: str, sid: int) -> bool:
    p = Path(data_root)
    return p.is_dir() and bool(list(p.glob(f"preproc_subj_{sid:02d}_*.npz")) or list(p.glob(f"preproc_subj_{sid:02d}_*.mat")))


def discover_supcon_dir(sid: int, extra_dirs: Iterable[str]) -> str:
    dirs = []
    dirs.extend(extra_dirs)
    dirs.extend(KNOWN_SUPCON_DIRS)

    # Ordered known dirs first.
    for d in dirs:
        if d and os.path.isfile(os.path.join(d, f"subj{sid:02d}_best.pt")):
            return d

    # Fallback: scan both repo-local and Drive checkpoints_vsre_dino trees.
    scan_roots = ["checkpoints_vsre_dino", f"{DRIVE_ROOT}/checkpoints_vsre_dino"]
    found: list[str] = []
    for root in scan_roots:
        if not Path(root).is_dir():
            continue
        for f in glob.glob(os.path.join(root, "*", f"subj{sid:02d}_best.pt")):
            found.append(os.path.dirname(f))
    if not found:
        return ""
    found.sort(key=lambda d: os.path.getmtime(os.path.join(d, f"subj{sid:02d}_best.pt")), reverse=True)
    return found[0]


def audit_subjects(args) -> list[AuditRow]:
    subjects = parse_subject_ids(args.subject_ids)
    roots = checkpoint_roots(args.ckpt_roots)
    rows: list[AuditRow] = []

    for sid in subjects:
        spec = SUBJECT_SPECS.get(sid, {"order": sid, "vs_per_class": -1, "vi_per_class": -1, "vs_total": -1, "planned_rank": 32})
        rank = planned_rank_for(sid, args.rank_policy)
        candidates = find_model_candidates(sid, rank, roots)
        data_ok = has_vs_data(args.data_root, sid)
        supcon_dir = discover_supcon_dir(sid, args.supcon_dirs)
        supcon_ok = bool(supcon_dir)

        status = "MISSING_CKPT"
        chosen: Optional[ModelCandidate] = candidates[0] if candidates else None
        if chosen is not None:
            status = "COMPLETE" if chosen.result_csv else "CKPT_ONLY"
        elif not data_ok:
            status = "BLOCKED_DATA"
        elif not supcon_ok:
            status = "BLOCKED_SUPCON"

        rows.append(
            AuditRow(
                order=int(spec["order"]),
                sid=sid,
                vs_per_class=int(spec["vs_per_class"]),
                vi_per_class=int(spec["vi_per_class"]),
                vs_total=int(spec["vs_total"]),
                planned_rank=int(spec["planned_rank"]),
                check_rank="any" if rank is None else str(rank),
                data_ok=data_ok,
                supcon_ok=supcon_ok,
                status=status,
                ckpt_path=chosen.ckpt_path if chosen else "",
                ckpt_dir=chosen.ckpt_dir if chosen else "",
                result_csv=chosen.result_csv if chosen else "",
                top1=f"{chosen.top1:.10f}" if chosen and chosen.top1 is not None else "",
                top3=f"{chosen.top3:.10f}" if chosen and chosen.top3 is not None else "",
                top5=f"{chosen.top5:.10f}" if chosen and chosen.top5 is not None else "",
                best_ep=str(chosen.best_ep) if chosen and chosen.best_ep is not None else "",
                supcon_dir=supcon_dir,
            )
        )

    rows.sort(key=lambda r: r.order)
    return rows


def print_audit(rows: list[AuditRow]) -> None:
    header = (
        "order sid rank status data supcon top1 best_ep ckpt"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        ckpt_short = r.ckpt_path.replace(DRIVE_ROOT, "$DRIVE") if r.ckpt_path else ""
        print(
            f"{r.order:>5} S{r.sid:02d} r={r.check_rank:<3} {r.status:<14} "
            f"data={str(r.data_ok):<5} supcon={str(r.supcon_ok):<5} "
            f"top1={r.top1 or '-':<12} best_ep={r.best_ep or '-':<4} {ckpt_short}",
            flush=True,
        )


def save_audit_csv(rows: list[AuditRow]) -> str:
    Path(DRIVE_AUDIT_DIR).mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(DRIVE_AUDIT_DIR, f"vs_lora_model_audit_{ts}.csv")
    fieldnames = list(AuditRow.__dataclass_fields__.keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fieldnames})
    print(f"[INFO] Saved audit CSV: {out}", flush=True)
    return out


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


def assert_images(img_root: str) -> None:
    if not Path(img_root).is_dir():
        raise FileNotFoundError(f"Missing image root: {img_root}")
    for i in range(1, 10):
        if not Path(img_root, f"{i:02d}.png").is_file():
            raise FileNotFoundError(f"Missing target image: {img_root}/{i:02d}.png")


def build_train_cmd(args, row: AuditRow) -> list[str]:
    rank = int(row.check_rank) if row.check_rank != "any" else SUBJECT_SPECS.get(row.sid, {}).get("planned_rank", 32)
    cmd = [
        sys.executable,
        "-u",
        "train_vs_re_lora_gen.py",
        "--subject_ids",
        str(row.sid),
        "--data_root",
        args.data_root,
        "--lora_r",
        str(rank),
        "--n_eeg_tokens",
        str(args.n_eeg_tokens),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--img_root",
        args.img_root,
        "--supcon_ckpt",
        row.supcon_dir,
        "--ckpt_root",
        DRIVE_CKPT_ROOT,
        "--fp16",
    ]
    if args.grad_ckpt or "t4" in gpu_name().lower():
        cmd.append("--grad_ckpt")
    return cmd


def gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return ""


def select_launch_row(rows: list[AuditRow], args) -> AuditRow:
    if args.force_sid is not None:
        for r in rows:
            if r.sid == args.force_sid:
                if not r.data_ok:
                    raise FileNotFoundError(f"S{r.sid:02d} VS data missing in {args.data_root}")
                if not r.supcon_ok:
                    raise FileNotFoundError(f"S{r.sid:02d} SupCon checkpoint missing")
                return r
        raise ValueError(f"force_sid S{args.force_sid:02d} was not included in --subject_ids")

    for r in rows:
        if r.status == "MISSING_CKPT" and r.data_ok and r.supcon_ok:
            return r
    raise SystemExit("No launchable missing VS model found. Use --force_sid SID to retrain an existing subject.")


def launch_training(args, row: AuditRow) -> None:
    jobs = active_training_jobs()
    if jobs and not args.allow_existing:
        print("Active training jobs detected:")
        for job in jobs:
            print("  ", job)
        raise SystemExit("Refusing duplicate launch. Stop the existing training process or pass --allow_existing.")

    assert_images(args.img_root)
    Path(DRIVE_LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(DRIVE_CKPT_ROOT).mkdir(parents=True, exist_ok=True)

    cmd = build_train_cmd(args, row)
    rank = row.check_rank if row.check_rank != "any" else str(SUBJECT_SPECS.get(row.sid, {}).get("planned_rank", 32))
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(DRIVE_LOG_DIR) / f"vs_s{row.sid:02d}_lora_r{rank}_tok{args.n_eeg_tokens}_{ts}.log"
    pid_path = Path(DRIVE_LOG_DIR) / f"vs_s{row.sid:02d}_lora_r{rank}_tok{args.n_eeg_tokens}_latest.pid"

    print("Launch target:")
    print(f"  subject      : S{row.sid:02d}")
    print(f"  rank         : r={rank}")
    print(f"  VS/class     : {row.vs_per_class}")
    print(f"  SupCon       : {row.supcon_dir}")
    print("Command:", shlex.join(cmd))
    print("Log    :", log_path)
    print("PID file:", pid_path)

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
    print(f"Started background VS training for S{row.sid:02d}")
    print("PID:", proc.pid)
    print("Monitor:")
    print(f"  tail -f {log_path}")
    print("  pgrep -af 'train_vs_re_lora_gen.py|train_exp43_vi_lora.py'")
    print("  nvidia-smi")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit VS LoRA models and launch one missing subject.")
    parser.add_argument("--subject_ids", default="24,1,2,18,28,29,9")
    parser.add_argument("--rank_policy", choices=["planned", "r32", "r16", "any"], default="planned")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--img_root", default=DEFAULT_IMG_ROOT)
    parser.add_argument("--ckpt_roots", nargs="*", default=[])
    parser.add_argument("--supcon_dirs", nargs="*", default=[])
    parser.add_argument("--save_csv", action="store_true", help="Save audit CSV to Drive")
    parser.add_argument("--launch_next_missing", action="store_true", help="Launch the first missing/checkpoint-free model in subject order")
    parser.add_argument("--force_sid", type=int, default=None, help="Retrain/launch a specific subject even if a checkpoint exists")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n_eeg_tokens", type=int, default=16)
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--allow_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_remount", action="store_true")
    args = parser.parse_args()

    mount_drive_if_needed(force_remount=args.force_remount)
    Path(DRIVE_LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(DRIVE_CKPT_ROOT).mkdir(parents=True, exist_ok=True)

    rows = audit_subjects(args)
    print_audit(rows)
    if args.save_csv:
        save_audit_csv(rows)

    if args.launch_next_missing or args.force_sid is not None:
        remove_incompatible_torchao()
        try:
            import torch
            print(f"torch={torch.__version__}")
            print(f"cuda={torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"gpu={torch.cuda.get_device_name(0)}")
        except Exception as exc:
            raise RuntimeError("torch import/GPU check failed") from exc
        row = select_launch_row(rows, args)
        launch_training(args, row)


if __name__ == "__main__":
    main()
