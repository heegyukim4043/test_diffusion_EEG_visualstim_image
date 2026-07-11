# Confirmatory single-session computation

This protocol freezes the single-session cohort and settings before inspecting
their results. It is intended for the paper supplement and must include failed,
chance-level, and negative-transfer subjects.

## Cohort

`S03, S05, S06, S11, S20, S21, S22, S23`

Each subject must have exactly one VS and one VI NPZ file. A subject may only be
excluded using a signal/data-integrity rule recorded before model evaluation.

## Frozen settings

- seeds: `42, 43, 44`
- SupCon: `lr=1e-4`, `temperature=0.07`, `epochs=200`, V2 encoder
- VS/VI LoRA: `r=32`, `alpha=32`, `16 EEG tokens`, `epochs=100`
- use all available trials (`per_class_total=0`)
- full VI test export, DDIM 30 steps
- paired C0/C1 generation noise seed: `20260711`
- no subject-wise test-driven hyperparameter selection

Seed 42 is the first complete confirmatory pass. Seeds 43/44 are repeated
robustness runs and must not be selected or discarded based on their scores.

## Colab setup

```python
from google.colab import drive
drive.mount("/content/drive")

%cd /content/vsvi_project
!git pull origin main
!pip uninstall -y torchao
```

## Audit

```python
!python -u run_confirmatory_single_session.py --stage audit --seed 42
```

The audit must report one VS and one VI NPZ for all eight subjects.

## Foreground/resumable execution

Run one stage at a time. Re-running the same command after a runtime reset moves
to the next incomplete stage:

```python
!python -u run_confirmatory_single_session.py \
  --stage next --subject_id 3 --seed 42
```

The six calls for each subject are:

```text
supcon -> vs_lora -> exp43_c0 -> exp43_c1 -> eval_c0 -> eval_c1
```

Repeat `--stage next` only after the foreground command finishes. Do not use
`nohup`, and do not run two GPU training processes in one runtime.

After S03 completes, use the same command for `5, 6, 11, 20, 21, 22, 23`.

## Summary

After all subjects for a seed complete:

```python
!python -u run_confirmatory_single_session.py --stage summary --seed 42
```

Durable outputs are stored under:

```text
/content/drive/MyDrive/vsvi_data/confirmatory_single_session/seed42/
```

Run seeds 43 and 44 only by changing `--seed`; each seed receives an isolated
protocol, checkpoint, log, and evaluation directory.
