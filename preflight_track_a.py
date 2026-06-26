"""
preflight_track_a.py
─────────────────────────────────────────────────────────────────
Pre-flight check before running Track A (S24 SD LoRA VS generation).

Checks (in order):
  1. CUDA available + torch CUDA functional
  2. peft importable
  3. diffusers importable
  4. img_root exists with 9 class images (01.png ~ 09.png)
  5. supcon_ckpt/subj{sid:02d}_best.pt exists
  6. data_root has at least one S24 .mat file  (--check_data only)
  7. ckpt_root parent is writable

Usage:
  python preflight_track_a.py \
    --subject_id 24 \
    --img_root preproc_data_vi/images \
    --supcon_ckpt checkpoints_vsre_dino/20260604_091352_ch32_merged_ep200_supcon \
    --data_root preproc_vs_re \
    --ckpt_root checkpoints_vsre_lora_gen \
    --check_data

Exit code 0 = all checks passed (safe to launch training).
Exit code 1 = one or more checks failed (do not launch training).
"""

import argparse
import os
import sys


def check(label: str, ok: bool, detail: str = ""):
    status = "OK  " if ok else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" -- {detail}"
    print(msg, flush=True)
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id",  type=int, default=24)
    parser.add_argument("--img_root",    default="preproc_data_vi/images")
    parser.add_argument("--supcon_ckpt", required=True)
    parser.add_argument("--data_root",   default="preproc_vs_re")
    parser.add_argument("--ckpt_root",   default="checkpoints_vsre_lora_gen")
    parser.add_argument("--check_data",  action="store_true",
                        help="Also verify EEG .mat files for the subject exist")
    args = parser.parse_args()
    sid = args.subject_id

    print(f"=== Track A Preflight (subject {sid}) ===\n")
    results = []

    # ── 1. CUDA ────────────────────────────────────────────────────────────
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            _ = torch.zeros(1).cuda()
            dev = torch.cuda.get_device_name(0)
            results.append(check("CUDA available + functional", True, dev))
        else:
            results.append(check("CUDA available", False, "torch.cuda.is_available() = False"))
    except Exception as e:
        results.append(check("CUDA / torch", False, str(e)))

    # ── 2. peft ────────────────────────────────────────────────────────────
    try:
        import peft
        results.append(check("peft importable", True, f"v{peft.__version__}"))
    except ImportError as e:
        results.append(check("peft importable", False, str(e)))

    # ── 3. diffusers ───────────────────────────────────────────────────────
    try:
        import diffusers
        results.append(check("diffusers importable", True, f"v{diffusers.__version__}"))
    except ImportError as e:
        results.append(check("diffusers importable", False, str(e)))

    # ── 4. class images ────────────────────────────────────────────────────
    img_root = args.img_root
    if os.path.isdir(img_root):
        found = [f"{i:02d}.png" for i in range(1, 10) if os.path.isfile(os.path.join(img_root, f"{i:02d}.png"))]
        ok = len(found) == 9
        results.append(check(
            f"img_root 9 class images",
            ok,
            f"{img_root}  ({len(found)}/9 found)"
        ))
    else:
        results.append(check("img_root exists", False, f"{img_root} not found"))

    # ── 5. SupCon checkpoint ────────────────────────────────────────────────
    ckpt_path = os.path.join(args.supcon_ckpt, f"subj{sid:02d}_best.pt")
    exists = os.path.isfile(ckpt_path)
    size_mb = os.path.getsize(ckpt_path) / 1e6 if exists else 0
    results.append(check(
        f"SupCon ckpt subj{sid:02d}_best.pt",
        exists,
        f"{ckpt_path}  ({size_mb:.1f} MB)" if exists else ckpt_path
    ))

    # ── 6. EEG data (optional) ─────────────────────────────────────────────
    if args.check_data:
        import glob
        prefix = os.path.join(args.data_root, f"preproc_subj_{sid:02d}_")
        npz_files = sorted(glob.glob(prefix + "*.npz"))
        mat_files = sorted(glob.glob(prefix + "*.mat"))
        n_npz, n_mat = len(npz_files), len(mat_files)
        ok = (n_npz + n_mat) > 0
        if ok:
            detail = f"{n_npz} .npz + {n_mat} .mat in {args.data_root}"
        else:
            detail = f"none found in {args.data_root}"
        results.append(check(f"EEG data for S{sid}", ok, detail))

    # ── 7. ckpt_root writable ──────────────────────────────────────────────
    ckpt_parent = args.ckpt_root
    os.makedirs(ckpt_parent, exist_ok=True)
    test_file = os.path.join(ckpt_parent, ".preflight_write_test")
    try:
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        results.append(check("ckpt_root writable", True, ckpt_parent))
    except Exception as e:
        results.append(check("ckpt_root writable", False, str(e)))

    # ── Summary ────────────────────────────────────────────────────────────
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print(f"\n{'='*45}")
    print(f"  {n_pass}/{len(results)} checks passed,  {n_fail} failed")

    if n_fail == 0:
        print("\n  PREFLIGHT PASSED -- safe to launch Track A training.")
        print(f"\n  Suggested command:")
        print(f"    python train_vs_re_lora_gen.py \\")
        print(f"      --subject_ids {sid} --lora_r 16 --n_eeg_tokens 16 --epochs 100 \\")
        print(f"      --img_root {args.img_root} \\")
        print(f"      --supcon_ckpt {args.supcon_ckpt} \\")
        print(f"      --ckpt_root {args.ckpt_root}")
        sys.exit(0)
    else:
        print("\n  PREFLIGHT FAILED -- resolve above issues before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
