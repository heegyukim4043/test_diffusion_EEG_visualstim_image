# VI time-frequency representation ablation

## Purpose

This encoder-only experiment tests whether a time-frequency EEG representation
improves VI decoding and the benefit of VS initialization.  It does not load
DINO, Stable Diffusion, or LoRA.

The primary comparison is made **within each representation**:

`VS->VI balanced accuracy - VI-only balanced accuracy`

## Prespecified S24 pilot

- Subject: S24 only
- Sampling rate: 1024 Hz
- Window: current 0-2 s NPZ epoch, 2048 samples
- Split: existing stratified 80/10/10 split, seed 42
- Representations: `raw`, `tf`, `raw_tf`
- Stages: `vs_pretrain`, `zero_shot`, `vi_only`, `vs_to_vi`
- Model selection: validation balanced accuracy, then Top-3 and Top-5
- Test set is evaluated only after checkpoint selection

TF bands are 4-7, 8-13, 14-20, 21-30, and 31-45 Hz.  The TF branch uses STFT
band power and keeps channel, band, and time dimensions separate.

## Colab setup

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/vsvi_project
!git pull
```

Run the audit first:

```python
import subprocess, sys

RUNNER = "/content/vsvi_project/run_vi_tf_representation_ablation.py"
COMMON = [
    sys.executable, "-u", RUNNER,
    "--subjects", "24",
    "--seed", "42",
    "--batch_size", "32",
    "--epochs", "100",
    "--patience", "20",
    "--fp16",
]
subprocess.run(COMMON + ["--stage", "audit"], check=True)
```

## Run one foreground stage at a time

The runner is resumable.  Re-running the cell skips completed stages and runs
the next missing stage for each representation.

```python
for representation in ("raw", "tf", "raw_tf"):
    subprocess.run(
        COMMON + [
            "--stage", "next",
            "--subject_id", "24",
            "--representation", representation,
        ],
        check=True,
    )
```

Run that cell four times.  The order for each representation is:

1. VS pretraining
2. VI zero-shot evaluation
3. VI-only training
4. VS-to-VI fine-tuning

Do not use `nohup` or background execution in Colab.

## Check completion and summarize

```python
subprocess.run(COMMON + ["--stage", "audit"], check=True)
subprocess.run(COMMON + ["--stage", "summary"], check=True)
```

Expected files:

```text
/content/drive/MyDrive/vsvi_data/vi_tf_representation_ablation/seed42/
  protocol.json
  audit.csv
  tf_representation_long.csv
  tf_representation_paired.csv
  S24/{raw,tf,raw_tf}/{vs_pretrain,zero_shot,vi_only,vs_to_vi}/metrics.json
```

## Pilot decision rule

Do not select a representation from test Top-1 alone.  Continue to the full
seven-subject experiment only if the S24 pilot satisfies both:

1. VS-to-VI does not show one-class collapse (`dominant_ratio < 0.8`).
2. At least one TF representation has a positive VS-to-VI minus VI-only BAC.

S24 is a development pilot.  After freezing the representation and settings,
the final claim must use all seven multi-session subjects without target-wise
hyperparameter changes.
