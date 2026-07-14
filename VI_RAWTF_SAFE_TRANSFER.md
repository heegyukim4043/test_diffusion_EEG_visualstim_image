# Raw+TF transfer-safe VS→VI experiment

This development experiment tests whether negative transfer is reduced by
discarding the VS classifier and gradually exposing the VS-pretrained encoder
to VI optimization.

The fixed schedule is:

1. epochs 1–10: newly initialized VI classifier only;
2. epochs 11–30: fusion block and classifier;
3. epochs 31–100: full raw+TF encoder, with classifier LR `3e-4`, fusion LR
   `1e-4`, and raw/TF backbone LR `3e-5`.

The data split, waveform augmentation, CE+SupCon objective, and raw+TF model
match the representation ablation. Selection is performed on VI validation
BAC, Top-3, and Top-5 after rounding metrics to 12 decimals. The VI test split
is evaluated once after the selected checkpoint is restored.

## S24 development run

```bash
python -u run_vi_rawtf_safe_transfer.py \
  --stage audit --subjects 24 --seed 42

python -u run_vi_rawtf_safe_transfer.py \
  --stage run --subject_id 24 --subjects 24 --seed 42 --fp16

python -u run_vi_rawtf_safe_transfer.py \
  --stage summary --subjects 24 --seed 42
```

Do not run the confirmatory subjects until the S24 decision is recorded. The
method advances only if Safe VS→VI improves BAC over both raw+TF VI-only and
the existing full VS→VI baseline without materially increasing collapse.
