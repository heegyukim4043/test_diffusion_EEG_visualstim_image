# Multi-session VI encoder transfer confirmatory protocol

## Question

Does VS EEG pretraining improve direct nine-class classification of VI EEG?
This experiment evaluates the EEG encoder directly; SD/LoRA image generation is
not involved.

## Fixed cohort

`S01, S02, S09, S18, S24, S28, S29`

All subjects have at least two VI sessions and a subject-specific VS SupCon
checkpoint in `20260604_091352_ch32_merged_ep200_supcon`.

## Conditions

1. `zero_shot`: frozen VS encoder evaluated directly on VI test trials.
2. `vi_only`: identical architecture, random initialization, trained on VI.
3. `vs_to_vi`: initialized from the VS encoder, then all weights fine-tuned on VI.

The VI train/validation/test split is identical across conditions. The primary
transfer contrast is `vs_to_vi - vi_only`.

## Locked primary configuration

- Split seed: 42, stratified 80/10/10 within each class
- Epochs: 200
- Batch size: 64
- Optimizer: AdamW, learning rate 3e-4, weight decay 1e-4
- Loss: SupCon + 0.5 auxiliary classification CE
- Encoder architecture: read from each subject's VS checkpoint
- Test metrics: Top-1/3/5, balanced accuracy, true-class score and margin
- Chance Top-1: 1/9

## Colab execution

First update the repository and audit inputs:

```python
!git -C /content/vsvi_project pull

!/usr/bin/python3 -u /content/vsvi_project/run_multi_session_vi_encoder_transfer.py \
  --stage audit --seed 42
```

Run the complete cohort in the foreground. Completed subject/condition pairs
are skipped, so the cell can safely be re-run after a runtime reset.

```python
import subprocess

PYTHON = "/usr/bin/python3"
RUNNER = "/content/vsvi_project/run_multi_session_vi_encoder_transfer.py"
SUBJECTS = [1, 2, 9, 18, 24, 28, 29]
STAGES = ["zero_shot", "vi_only", "vs_to_vi"]

for stage in STAGES:
    for sid in SUBJECTS:
        print(f"\n{'=' * 70}\nSTART S{sid:02d} {stage}\n{'=' * 70}", flush=True)
        subprocess.run(
            [
                PYTHON, "-u", RUNNER,
                "--stage", stage,
                "--subject_id", str(sid),
                "--seed", "42",
            ],
            check=True,
        )
```

Do not add `--force` during normal resume. `torchao` is irrelevant to this
encoder-only experiment and does not need to be uninstalled.

Generate the paired summary after all three conditions finish:

```python
!/usr/bin/python3 -u /content/vsvi_project/run_multi_session_vi_encoder_transfer.py \
  --stage summary --seed 42
```

Outputs are stored under:

```text
/content/drive/MyDrive/vsvi_data/vi_encoder_transfer_multi/seed42/
```

Completion requires seven rows with all three stage flags equal to one in
`audit.csv`. The paired paper table is `vi_encoder_transfer_summary.csv`.
