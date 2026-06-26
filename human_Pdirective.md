# human_Pdirective

Last Updated: 2026-05-17
Owner: user
Status: active

---

## 1. Current Performance Snapshot

Current best confirmed result is on the VS DINO alignment track.

- Best representation checkpoint:
  - [20260427_095215_ch32_merged_ep200](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/checkpoints_vsre_dino/20260427_095215_ch32_merged_ep200)
- Config:
  - `train_vs_re_dino.py --encoder_type v2 --eeg_occipital_ids auto --subject_ids 1,2,18 --epochs 200`
- Mean performance:
  - `Top-1 = 0.2824`
  - `Top-3 = 0.5648`
  - `Top-5 = 0.7731`

Current best confirmed generation probe is weaker.

- Best generation subset checkpoint:
  - [20260417_113129_ch32_merged_ep300](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/checkpoints_vsre_gen/20260417_113129_ch32_merged_ep300)
- Config:
  - `train_vs_re_gen.py --encoder_version v2 --eeg_occipital_ids none --subject_ids 1,2,18 --epochs 300`
- Mean performance:
  - `best_val_loss = 0.000516`
  - `SSIM = 0.043`
  - `DINO Top-1 = 0.127`
  - `DINO Top-3 = 0.353`
  - `DINO Top-5 = 0.571`

Conclusion:
- the main performance axis is `representation`, not `generation`
- the next cycle must optimize EEG representation first

---

## 2. Primary Objective

The actual near-term goal is:
- improve `VS EEG -> visual representation` quality
- use generation only as a secondary probe
- prepare for later transfer to `VI`

Do not optimize image sharpness ahead of identity metrics.

---

## 3. Priority Order

### Priority 1: Improve DINO alignment beyond Exp010

Base config to keep:
- `train_vs_re_dino.py`
- `--encoder_type v2`
- `--eeg_occipital_ids auto`
- `--subject_ids 1,2,18`

Immediate experiments:
1. `temperature` sweep
   - `0.05 / 0.07 / 0.10 / 0.15`
2. `w_aux` sweep
   - `0.2 / 0.5 / 1.0`
3. `w_proto / w_infonce / w_cos` rebalance
4. confirm whether gains hold without unstable negative-loss behavior

Success condition:
- beat `Top-1 = 0.2824`
- keep or improve `Top-3 / Top-5`
- no collapse of one or two subjects while mean rises

### Priority 2: Add EEG inductive bias only on the DINO track

Allowed direction:
- early/late temporal split encoder
  - branch A: `0~250 ms`
  - branch B: `250~2000 ms`
- stronger temporal multi-scale branch
- depthwise/separable temporal conv refinement

Important:
- test these on `train_vs_re_dino.py` first
- do not start from the generation model for this ablation

Success condition:
- representation metrics improve before any generation follow-up

### Priority 3: CLIP image embedding comparison

Goal:
- compare `DINOv2 image space` vs `CLIP image space`

Required direction:
- use `CLIP image embedding`, not text embedding
- keep the task as retrieval/alignment only
- do not connect CLIP directly to generation first

Success condition:
- determine whether CLIP is better, worse, or complementary to DINO

### Priority 4: Generation collapse diagnosis

Before any new generation optimization, add:
- generated DINO-predicted class histogram
- generated-vs-true confusion matrix
- predicted-class entropy
- per-subject collapse summary

Reason:
- low `Top-1` alone is insufficient
- must distinguish weak generation from one-class collapse

### Priority 5: Anti-collapse auxiliary loss for generation

Only after diagnostics are added.

Preferred direction:
- EEG latent -> DINO prototype retrieval loss
- EEG latent class CE regularizer
- explicit loss-weight ablation

Important:
- keep generation `EEG-only`
- do not add true class-label conditioning as the main result

### Priority 6: SD 1.5 migration planning

This is not the immediate next run.

Only start after:
- DINO/CLIP direction is clearer
- encoder-side evidence is stronger
- generation probe behavior is better understood

---

## 4. What Not To Prioritize Now

- do not spend more time on sampler-only tuning
- do not prioritize DDIM steps, cosine schedule, or LPIPS sweeps
- do not optimize image sharpness without identity gains
- do not use `Exp007` as the next main task
- do not treat unfinished generation checkpoints as valid results

Reason:
- previous generation-only tuning did not beat baseline
- the current bottleneck is not sampler detail, but representation quality

---

## 5. Active Baselines

Representation baseline to beat:
- DINO `Exp010`
- `Top-1 = 0.2824`
- `Top-3 = 0.5648`
- `Top-5 = 0.7731`

Generation probe baseline to beat:
- generation subset `Exp006`
- `DINO Top-1 = 0.127`

Full-subject generation reference:
- [20260411_165536_ch32_merged_ep300](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/checkpoints_vsre_gen/20260411_165536_ch32_merged_ep300/results_gen.csv)
- `DINO Top-1 = 0.115998`

---

## 6. Decision Rule

If a new experiment improves `generation` but not `representation`:
- do not treat it as the main advance

If a new experiment improves `representation` clearly:
- it is worth following with a generation probe

If `CLIP` does not beat `DINO`:
- keep DINO as the main target and move on

If a generation run lacks histogram/confusion/entropy outputs:
- treat the result as incomplete

---

## 7. Immediate Next Run Order

1. DINO `temperature` sweep on `Exp010` base
2. DINO `w_aux` sweep
3. DINO loss rebalance (`w_proto / w_infonce / w_cos`)
4. early/late temporal split encoder
5. CLIP image embedding comparison
6. generation collapse diagnostic
7. generation anti-collapse auxiliary loss

