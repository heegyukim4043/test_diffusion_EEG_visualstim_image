# VSRE Progress Summary

Updated: 2026-04-15

## Scope

This file summarizes the work done so far for the new repeated-session VS dataset:

- Dataset: `preproc_vs_re`
- Goal: EEG-only image generation from stimulus-evoked EEG
- Current policy:
  - use `0~2 sec` only
  - use `32 EEG channels` by default
  - start with `subject-wise` experiments first
  - do not use class-label conditioning in the main generation model

---

## 1. Dataset Understanding

New dataset structure was analyzed and documented.

- Source format: MATLAB v7.3 / HDF5
- Main array: `results/data`
- Shape:
  - `(5, 3, 9, 4096, 40)`
- Time axis interpretation:
  - `-1 ~ 0 sec`: index `0:1024`
  - `0 ~ 2 sec`: index `1024:3072`
  - `2 ~ 3 sec`: index `3072:4096`
- Current experiment window:
  - `1024:3072` only
- Channel policy:
  - default `32 EEG`
  - `8 EX` excluded from baseline experiments

Reasoning:
- stimulus exists only in `0~2 sec`
- using non-stimulus periods would weaken interpretability
- starting with `32 EEG only` keeps the decoding claim cleaner

---

## 2. Loader Implementation

Implemented subject-wise loader for `preproc_vs_re`.

Main file:
- [dataset_vs_re.py](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/dataset_vs_re.py)

Implemented features:
- `load_subject_vsre(data_root, sid, n_ch=32)`
  - loads one subject across sessions
  - returns `(eeg, labels)`
- `VSReDataset(...)`
  - supports subject-wise `train/val/test`
  - split is class-balanced
  - supports `max_sessions` for session-capped experiments
- direct smoke-test block fixed so the file can be executed directly

Observed subject-wise data behavior:
- sessions vary substantially by subject
- some subjects have only `1 session`
- some subjects have up to `9 sessions`
- this creates large per-subject sample count imbalance

---

## 3. Stage 1: DINO Alignment Baseline

Built a subject-wise retrieval/alignment baseline before generation.

Main files:
- [model_eeg_dino.py](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/model_eeg_dino.py)
- [train_vs_re_dino.py](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/train_vs_re_dino.py)

Design:
- input: subject-wise EEG
- target: DINO image prototype / retrieval space
- evaluation:
  - `Top-1`
  - `Top-3`
  - `Top-5`
  - confusion matrix

Important fix:
- the original evaluation path reused a cross-subject retrieval function
- that function expected 4-item batches
- `VSReDataset` returns 3-item batches
- added a VSRE-specific retrieval evaluation path in `train_vs_re_dino.py`

Completed DINO experiments:

1. Subject-wise merged baseline
- checkpoint:
  - [checkpoints_vsre_dino/20260411_152315_ch32_merged_ep200](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/checkpoints_vsre_dino/20260411_152315_ch32_merged_ep200)
- mean result:
  - `Top-1 = 0.1386`
  - `Top-3 = 0.4305`
  - `Top-5 = 0.6461`

2. Session-capped comparison
- checkpoint:
  - [checkpoints_vsre_dino/20260412_163351_ch32_cap2_ep200](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/checkpoints_vsre_dino/20260412_163351_ch32_cap2_ep200)

3. Temperature / batch tuning run
- checkpoint:
  - [checkpoints_vsre_dino/20260412_172051_ch32_merged_ep200](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/checkpoints_vsre_dino/20260412_172051_ch32_merged_ep200)

Current interpretation:
- retrieval works above random for some subjects
- subject variance is large
- alignment is usable as a baseline but still weak for stable generation

---

## 4. Stage 2: Subject-Wise EEG-Only Generation Baseline

Built the first subject-wise generation pipeline on top of the new dataset.

Main files:
- [model_128_eegonly_transformer.py](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/model_128_eegonly_transformer.py)
- [model_128_eegonly_transformer_repa.py](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/model_128_eegonly_transformer_repa.py)
- [train_vs_re_gen.py](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/train_vs_re_gen.py)

Current generation design:
- subject-wise training
- EEG-only conditioning
- no class-label conditioning
- image size: `128x128`
- EMA enabled
- sample grid saved during training
- quantitative evaluation after training

Evaluation metrics now implemented:
- `L1`
- `SSIM`
- `LPIPS` if available
- generated-image `DINO Top-1 / Top-3 / Top-5`

Important implementation fixes applied during development:
- `p_losses()` now receives explicit `t`
- `sample_ddim(steps=...)` fixed to `sample_ddim(num_steps=...)`
- result CSV writing fixed
- generation evaluation added
- `eta`, `guidance_scale`, `eval_ddim_steps`, `sample_ddim_steps` wired through

---

## 5. Baseline Generation Results

### 5.1 Original subject-wise generation baseline

Checkpoint:
- [checkpoints_vsre_gen/20260411_165536_ch32_merged_ep300](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/checkpoints_vsre_gen/20260411_165536_ch32_merged_ep300)

Mean result:
- `best_val_loss = 0.002779`
- `L1 = 0.805959`
- `SSIM = 0.056175`
- `DINO Top-1 = 0.115998`
- `DINO Top-3 = 0.297325`
- `DINO Top-5 = 0.552396`

This remains the practical baseline to beat.

### 5.2 DDIM 200 + guidance 3.0

Checkpoint:
- [checkpoints_vsre_gen/20260413_121829_ch32_merged_ep300](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/checkpoints_vsre_gen/20260413_121829_ch32_merged_ep300)

Mean result:
- `best_val_loss = 0.002764`
- `L1 = 0.795035`
- `SSIM = 0.035236`
- `DINO Top-1 = 0.096451`
- `DINO Top-3 = 0.326830`
- `DINO Top-5 = 0.541079`

Interpretation:
- did not improve the practical baseline
- strong guidance degraded retrieval-oriented quality

### 5.3 Cosine schedule + LPIPS 0.05 + eta 0.0

Checkpoint:
- [checkpoints_vsre_gen/20260414_100452_ch32_merged_ep300](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/checkpoints_vsre_gen/20260414_100452_ch32_merged_ep300)

Mean result:
- `best_val_loss = 0.009470`
- `L1 = 0.903295`
- `SSIM = -0.013619`
- `DINO Top-1 = 0.081643`
- `DINO Top-3 = 0.302285`
- `DINO Top-5 = 0.547362`

Interpretation:
- clearly worse than the original baseline
- LPIPS `0.05` was too strong under the current setup
- training became unstable

Current conclusion:
- baseline `[20260411_165536]` is still the best generation run

---

## 6. Model/Training Improvements Already Implemented

### 6.1 Noise schedule support

In [model_128_eegonly_transformer.py](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/model_128_eegonly_transformer.py):
- added `_cosine_beta_schedule()`
- added `beta_schedule` argument to `EEGDiffusionModel128`
- supports:
  - `linear`
  - `cosine`

### 6.2 LPIPS training loss support

In [model_128_eegonly_transformer_repa.py](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/model_128_eegonly_transformer_repa.py):
- optional `lpips` import
- `lambda_lpips` parameter added
- frozen `lpips_fn` support
- `train()` override keeps LPIPS net in eval mode
- LPIPS added to combined training loss

### 6.3 Generation script controls

In [train_vs_re_gen.py](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/train_vs_re_gen.py):
- added / exposed:
  - `--beta_schedule`
  - `--lambda_lpips`
  - `--eta`
  - `--sample_ddim_steps`
  - `--eval_ddim_steps`
  - `--guidance_scale`
- `LPIPS` metric added to evaluation CSV

Current key default:
- `--lambda_lpips` is now `0.0`
- LPIPS must be enabled explicitly for testing

---

## 7. Current Best Practice

For now, the recommended baseline remains:

- subject-wise
- `0~2 sec`
- `32 EEG channels`
- EEG-only conditioning
- merged sessions
- original generation baseline as the comparison anchor

Practical rule:
- treat new changes as ablations against the baseline
- do not assume perceptual losses or stronger guidance help
- verify by `DINO Top-1`, `SSIM`, and `best_val_loss`

Current stop criteria used in practice:
- `DINO Top-1 < 0.1160`
- `SSIM < 0`
- `best_val_loss > 0.005`

---

## 8. Ongoing / Recent Ablations

### Step A

Purpose:
- test `cosine` schedule alone
- `LPIPS = 0`

Example command:

```powershell
conda run -n eegdiff --no-capture-output python .\train_vs_re_gen.py --subject_ids all --epochs 300 --beta_schedule cosine --lambda_lpips 0.0 --lambda_percept 0.1 --lambda_rec 0.01 --guidance_scale 1.5 --eta 0.0 --eval_ddim_steps 50
```

### Step B

Purpose:
- test weak LPIPS only
- `linear` schedule
- `lambda_lpips = 0.01`

Example command:

```powershell
conda run -n eegdiff --no-capture-output python .\train_vs_re_gen.py --subject_ids all --epochs 300 --beta_schedule linear --lambda_lpips 0.01 --lambda_percept 0.1 --lambda_rec 0.01 --guidance_scale 1.5 --eta 0.0 --eval_ddim_steps 50
```

Note:
- if entered manually, `--eval_ddim_steps` must include a numeric value
- an incomplete command will not start the intended run

---

## 9. Main Takeaways So Far

1. The new dataset pipeline is working end-to-end.
2. Subject-wise DINO alignment is usable, but not yet strong.
3. Subject-wise EEG-only generation works technically, but image quality is still weak.
4. Stronger DDIM guidance did not help.
5. LPIPS `0.05` hurt training under the current setup.
6. The original generation baseline remains the best reference point.
7. The next gains are more likely to come from:
   - better EEG encoder quality
   - cleaner loss balance
   - careful ablation, not aggressive complexity

