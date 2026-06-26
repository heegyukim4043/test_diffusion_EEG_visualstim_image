# Goal

Run subject-wise EEG-only image generation on `preproc_vs_re`, keep the original full-subject baseline as the anchor, and compare EEG Encoder `V1` vs `V2` under reproducible subset runs.

Primary metric:
- `DINO Top-1 retrieval accuracy`

Current full-subject baseline:
- checkpoint: [`20260411_165536_ch32_merged_ep300`](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/checkpoints_vsre_gen/20260411_165536_ch32_merged_ep300/results_gen.csv)
- baseline `DINO Top-1 = 0.115998` (use `0.1160` as shorthand threshold)


# Current State

What is already done:
- `dataset_vs_re.py`: loader for `preproc_vs_re`
- `train_vs_re_dino.py`: subject-wise DINO alignment baseline completed
- `train_vs_re_gen.py`: subject-wise generation baseline completed
- `model_128_eegonly_transformer.py`: supports EEG encoder `v1` and `v2`
- `v2` includes:
  - `OccipitalChannelGate`
  - `MultiScaleStem`
  - BioSemi32 auto prior support

Known result summary:
- original full baseline remains best on full-subject comparison
- `cosine` schedule alone did not beat baseline
- `LPIPS 0.05` harmed stability
- recent subset runs exist, but exact V1/V2 identity tracking was previously incomplete


# Codebase

Existing PyTorch project. Update files incrementally; do not rewrite unrelated parts.

Key files:
- `train_vs_re_gen.py`:
  subject-wise generation training
- `model_128_eegonly_transformer.py`:
  EEG encoder V1/V2
- `model_128_eegonly_transformer_repa.py`:
  generation loss wrapper
- `dataset_vs_re.py`:
  HDF5 loader

Related summary/docs:
- `VSRE_PROGRESS_SUMMARY.md`
- `PROGRESS.md`
- `HUMAN_DIRECTIVE.md`


# What to Try

## Phase 1: Reproducible subset comparison

Run these first on `subjects 1,2,18`:

1. V1 subset baseline
- `--encoder_version v1`

2. V2 + auto prior
- `--encoder_version v2 --eeg_occipital_ids auto`

3. V2 + no prior
- `--encoder_version v2 --eeg_occipital_ids none`

Goal:
- determine whether V2 beats V1
- determine whether auto prior helps vs no prior

## Phase 2: Only if V2 shows value

4. Temporal-window / multi-scale branch
5. Channel attention / spatial attention
6. Session-merged vs session-capped follow-up

Do not jump to higher-complexity branches before the V1/V2 comparison is clear.


# Key Args (current comparison setting)

Use these as the fixed default comparison setting unless the experiment is explicitly about changing one of them:

- `--subject_ids 1,2,18`
- `--epochs 300`
- `--beta_schedule linear`
- `--lambda_lpips 0.0`
- `--lambda_percept 0.1`
- `--lambda_rec 0.01`
- `--guidance_scale 1.5`
- `--eta 0.0`
- `--eval_ddim_steps 50`


# Stop Conditions

- Mean `DINO Top-1 < 0.1160`
- Mean `SSIM < 0`
- Mean `best_val_loss > 0.005`
- Run metadata is insufficient to reconstruct the experiment

Additional rules:
- Do not add class-label conditioning
- Do not start cross-subject training yet
- Do not move to latent diffusion before subject-wise baseline is stable


# Constraints

- Use GPU 0 only
- `32 EEG channels` only
- Time window: `0~2 sec` only (`1024:3072`)
- Keep the main claim EEG-only
- Do not modify `dataset_vs_re.py` unless the experiment is specifically about data handling
- Do not modify `model_eeg_dino.py` for generation-side ablations


# Reproducibility Requirement

Every run must be traceable from checkpoint metadata.

Required metadata to save with checkpoints:
- `encoder_version`
- `eeg_stem_filters`
- `eeg_occipital_ids` or parsed occipital mode
- `beta_schedule`
- `lambda_lpips`
- `lambda_percept`
- `lambda_rec`
- `lambda_ssim`
- `guidance_scale`
- `eta`
- `eval_ddim_steps`
- `sample_ddim_steps`
- `subject_ids`
- `max_sessions`
- `seed`

