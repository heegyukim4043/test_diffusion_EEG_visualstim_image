# Raw+TF joint replay and class-conditional MMD

## Question

The raw+TF representation improved absolute VI balanced accuracy, but its
sequential VS-to-VI benefit was small and inconsistent.  This experiment asks
whether retaining VS samples during VI training helps, and whether explicit
class-conditional feature alignment adds benefit beyond replay alone.

The paired conditions are:

1. `vi_only`: existing raw+TF VI-only baseline.
2. `sequential`: existing raw+TF VS-pretrain then VI fine-tuning.
3. `replay`: joint VI training plus VS replay, with `lambda_MMD=0`.
4. `ccmmd`: the identical replay objective plus class-conditional MMD.

The loss is

```text
L = CE_VI + 0.5 CE_VS
  + 0.2 (SupCon_VI + 0.5 SupCon_VS)
  + lambda_MMD CC-MMD(z_VS, z_VI).
```

VS and VI use separate class-balanced batches with two trials per class.  MMD
is computed separately for each of the nine classes on normalized raw+TF
latents, then averaged.  The RBF kernel uses detached median bandwidth and
fixed scales 0.5, 1, 2, and 4.

## Leakage control

- S24 is the development subject.
- Candidate lambdas are `0.01,0.05,0.1`.
- Lambda selection uses only S24 VI validation balanced accuracy.
- Top-3 is the tie-break; an exact tie chooses the smaller lambda.
- Candidate checkpoints are not evaluated on the test set.
- The VI test split is evaluated once after selecting the candidate.
- The selected lambda is then fixed for S01, S02, S09, S18, S28, and S29.
- No target-subject lambda changes are allowed.

The present split is the existing stratified trial split.  A final selected
method still requires session-held-out evaluation before a cross-session claim.

## Colab setup

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/vsvi_project
!git pull

import json
import os
import subprocess
import sys

RUNNER = "/content/vsvi_project/run_vi_rawtf_ccmmd.py"
BASELINE_ROOTS = (
    "/content/drive/MyDrive/vsvi_data/vi_tf_representation_ablation,"
    "/content/drive/MyDrive/vsvi_data/vi_rawtf_confirmatory_multi"
)
```

## Phase A: S24 development

```python
S24_COMMON = [
    sys.executable, "-u", RUNNER,
    "--subjects", "24",
    "--seed", "42",
    "--baseline_roots", BASELINE_ROOTS,
    "--out_root", "/content/drive/MyDrive/vsvi_data/vi_rawtf_ccmmd_s24",
    "--lambda_candidates", "0.01,0.05,0.1",
    "--batch_size", "32",
    "--epochs", "100",
    "--patience", "20",
    "--fp16",
]

subprocess.run(S24_COMMON + ["--stage", "audit"], check=True)
```

Run replay, then CC-MMD tuning.  Both run in the foreground and are resumable.

```python
for stage in ("replay", "ccmmd"):
    subprocess.run(
        S24_COMMON + [
            "--stage", stage,
            "--subject_id", "24",
        ],
        check=True,
    )
```

Read the selected lambda and validation-only candidate table:

```python
S24_CCMMD = (
    "/content/drive/MyDrive/vsvi_data/vi_rawtf_ccmmd_s24/"
    "seed42/S24/ccmmd/metrics.json"
)
with open(S24_CCMMD, encoding="utf-8") as handle:
    s24 = json.load(handle)

print("selected lambda:", s24["selected_lambda"])
print("test evaluations:", s24["test_evaluations"])
s24["candidate_validation"]
```

## Phase B: fixed-lambda confirmatory subjects

Replace `SELECTED_LAMBDA` only with the value selected above.  Do not choose a
different value from confirmatory-subject results.

```python
SELECTED_LAMBDA = s24["selected_lambda"]
SUBJECTS = "1,2,9,18,28,29"
CONFIRM_ROOT = "/content/drive/MyDrive/vsvi_data/vi_rawtf_ccmmd_confirmatory"

CONFIRM_COMMON = [
    sys.executable, "-u", RUNNER,
    "--subjects", SUBJECTS,
    "--seed", "42",
    "--baseline_roots", BASELINE_ROOTS,
    "--out_root", CONFIRM_ROOT,
    "--fixed_lambda", str(SELECTED_LAMBDA),
    "--lambda_candidates", "0.01,0.05,0.1",
    "--batch_size", "32",
    "--epochs", "100",
    "--patience", "20",
    "--fp16",
]

subprocess.run(CONFIRM_COMMON + ["--stage", "audit"], check=True)
```

Foreground, resumable execution:

```python
for sid in (1, 2, 9, 18, 28, 29):
    for stage in ("replay", "ccmmd"):
        metrics = (
            f"{CONFIRM_ROOT}/seed42/S{sid:02d}/"
            f"{stage}/metrics.json"
        )
        if os.path.isfile(metrics):
            print(f"[SKIP] S{sid:02d} {stage}")
            continue
        subprocess.run(
            CONFIRM_COMMON + [
                "--stage", stage,
                "--subject_id", str(sid),
            ],
            check=True,
        )
```

Summarize paired results:

```python
subprocess.run(CONFIRM_COMMON + ["--stage", "summary"], check=True)
```

Expected summary:

```text
/content/drive/MyDrive/vsvi_data/vi_rawtf_ccmmd_confirmatory/seed42/
  rawtf_ccmmd_long.csv
  rawtf_ccmmd_summary.csv
```

## Primary comparisons

1. `ccmmd - replay`: MMD-specific benefit.
2. `replay - sequential`: benefit of retaining VS during VI training.
3. `ccmmd - vi_only`: total VS-to-VI benefit relative to no VS data.

Balanced accuracy is primary.  Top-3, Top-5, dominant ratio, and normalized
entropy are secondary collapse/ranking diagnostics.  A Top-1 increase with
worse Top-k and stronger dominance is not treated as robust transfer.
