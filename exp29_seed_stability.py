"""
Exp29: Low-session outlier analysis for S11 and S16.
Runs train_vs_re_dino.py with seeds 0,1,2,42 and reports Top-1 variance.
"""

import subprocess
import sys
import re
import csv
import os

PYTHON = r"C:\Users\Biocomputing\anaconda3\envs\eegdiff\python.exe"
SEEDS = [0, 1, 2, 42]
SUBJECTS = [11, 16]

results = []

for sid in SUBJECTS:
    for seed in SEEDS:
        print(f"\n>>> S{sid:02d} seed={seed}", flush=True)
        cmd = [
            PYTHON, "train_vs_re_dino.py",
            "--loss_type", "supcon",
            "--encoder_type", "v2",
            "--eeg_occipital_ids", "auto",
            "--subject_ids", str(sid),
            "--epochs", "200",
            "--seed", str(seed),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = proc.stdout + proc.stderr
        match = re.search(r"\[Test S\d+\]\s+Top-1=([\d.]+)\s+Top-3=([\d.]+)\s+Top-5=([\d.]+)", out)
        if match:
            t1, t3, t5 = float(match.group(1)), float(match.group(2)), float(match.group(3))
        else:
            t1, t3, t5 = None, None, None
            print("  [WARN] No test result found", flush=True)
            print(out[-500:], flush=True)
        results.append({"subject": f"S{sid:02d}", "seed": seed, "top1": t1, "top3": t3, "top5": t5})
        print(f"  Top-1={t1}  Top-3={t3}  Top-5={t5}", flush=True)

# Summary
print("\n=== Summary ===")
for sid in SUBJECTS:
    rows = [r for r in results if r["subject"] == f"S{sid:02d}" and r["top1"] is not None]
    vals = [r["top1"] for r in rows]
    import numpy as np
    if vals:
        print(f"S{sid:02d}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}  seeds={[r['seed'] for r in rows]}")
        print(f"       per-seed: {[f\"{r['seed']}:{r['top1']:.4f}\" for r in rows]}")

# Save CSV
out_csv = "exp29_seed_stability_results.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["subject", "seed", "top1", "top3", "top5"])
    writer.writeheader()
    writer.writerows(results)
print(f"\nSaved: {out_csv}")
