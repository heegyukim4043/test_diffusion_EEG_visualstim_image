# EEG-to-Image Diffusion Model: Progress Log

## Current Status Summary (2026-06-26)

### S24 SD LoRA VS Generation (2026-06-25~26)

**S24 r=16, 16tok:**
- Checkpoint: `checkpoints_vsre_lora_gen/20260625_111012_lora_r16_ep100/subj24_lora_best.pt`
- Training: 100 epochs, ~15 hours total on this machine
- Full-test single-gallery DINO@1 = `0.1111` (chance), entropy = `2.001`, dominant = `28.1%`
- Note: gallery size = 9 (1 per class); no `images_alt` directory found, so dual-gallery = single-gallery
- Interpretation: generator is diverse (not collapsed), but EEG conditioning is not effective at r=16
- **S24 r=32** still to run

**S24 vs S01/S18 comparison (r=16):**

| Subject | r=16 DINO@1 | r=32 DINO@1 | Sessions | Note |
|---------|-------------|-------------|----------|------|
| S18 | 0.2963 | 0.2870 | 8 | S18 best = r=16 |
| S01 | 0.3333 | **0.3571** | 9 | S01 best = r=32 |
| S24 | 0.1111 | TBD | 10 | at chance; r=32 pending |

**Directive decision rule:** S24 < S01 at r=16 → subject-specific EEG signal quality matters more than session count alone. Need r=32 result to confirm before final interpretation.

---

## Current Status Summary (2026-06-17)

This section is the active project summary. It supersedes older priority notes below where they conflict.

### Experiment History Preservation

This file should preserve as much of the experiment process as possible.

- Keep earlier Exp sections in place even when the active direction changes.
- Add new summaries at the top or append new Exp sections, but do not delete failed or obsolete experiments.
- If a conclusion changes, add a correction note instead of rewriting the original history.
- Negative results are part of the decision trail, especially Exp13 collapse, Exp31 pixel/FiLM failure, Exp32~40 scratch latent limits, and Exp41~42 SD LoRA transition.
- The active resume point is after `Exp42-B Step4`, but the full Exp trail should remain available for audit and reproduction.

### Active Goal

- Primary target: EEG-conditioned image generation.
- Current path: within-subject `VS -> VI` transfer.
- Main model family: SD 1.5 LoRA generator with EEG/SupCon conditioning.
- Evaluation standard: full-test dual-gallery DINO retrieval, not stochastic sample-grid-only scores.

### Data In Use

| Dataset | Path | Subjects | Total | Train | Val | Test | Note |
|---------|------|----------|-------|-------|-----|------|------|
| VS | `preproc_vs_re` | 21 | 9,207 | 7,335 | 837 | 1,035 | S01 session 9 skipped by current loader due anomalous `n_rep=108` |
| VI | `preproc_vi_re` | 20 | 9,315 | 7,452 | 882 | 981 | Same repeated-session style as VS; S08 absent |

High-priority VS/VI overlap subjects:
- `S01`: VS 1,215 trials, VI 1,215 trials
- `S18`: VS 1,080 trials, VI 1,080 trials
- `S24`: VS 1,350 trials, VI 1,350 trials

### Key Results

| Exp | Model / Purpose | Main Result | Interpretation |
|-----|-----------------|-------------|----------------|
| Exp13 | Pixel diffusion | near chance, severe collapse | pixel-space generation failed |
| Exp23-B | SupCon EEG-DINO encoder | 3-subj mean Top-1 `0.3333` | best representation checkpoint |
| Exp31 | SupCon-init pixel generator | partial only; FiLM bottleneck | pixel/FiLM path insufficient |
| Exp32~37 | SD VAE latent + CA/FiLM | S18 dual-gallery ceiling around `0.2315` after full-test correction | scratch latent UNet improved but saturated |
| Exp39 | multi-gallery evaluation | corrected inflated sample-grid scores | full-test dual-gallery is the standard |
| Exp40 | multi-image target training | worse than single target | do not continue this under current architecture |
| Exp41 | SD 1.5 LoRA on S18 | S18 dual-gallery DINO@1 `0.2870`, entropy `2.072`, dominant `23.1%` | first strong SD prior result |
| Exp42-A | SD 1.5 LoRA on S01 | S01 dual-gallery DINO@1 `0.3333`, entropy `2.005`, dominant `25.4%` | current project-best generative result; matches Stage 2 retrieval |
| Exp42-B Step4 | LoRA rank ablation | S01 r=32, 16 tokens: dual-gallery DINO@1 `0.3571` | first fully generative result above Stage 2 retrieval |
| Exp42-B Step5 | S18 class-preserving target augmentation | internal partial-test DINO@1 `0.4464`, entropy `1.960`, dominant `25.0%`, best_ep `95` | completed; not directly comparable to full-test dual-gallery scores |

### Current Interpretation

- SD 1.5 LoRA is the current best generator.
- Exp42-B Step4 is the current project best: S01 r=32, 16 tokens, dual-gallery DINO@1 `0.3571`.
- Generation now exceeds the Stage 2 retrieval reference (`0.3333`) for the first time.
- S01 and S18 appear to prefer different LoRA ranks:
  - S18 best so far: r=16, 16 tokens, dual-gallery DINO@1 `0.2963`
  - S01 best so far: r=32, 16 tokens, dual-gallery DINO@1 `0.3571`
- This suggests rank may need to depend on subject/signal quality; VI may require higher rank than VS because VI signals are weaker.
- S18 and S01 should both be tracked in future LoRA ablations.
- S24 should be tested next as the largest VS/VI overlap subject to separate session-count effects from subject-specific EEG quality.
- Exp42-B Step5 S18 target augmentation is completed. The logged `0.4464` is the training script's internal partial-test/prototype score, not the full-test dual-gallery score. Keep full-test dual-gallery as the final comparison standard.

### Active Next Steps

1. `S24 SD LoRA VS generation`.
   - Same setup as Exp42-A.
   - Run both r=16 and r=32 if compute allows, because S18/S01 show subject-specific rank optima.
   - Purpose: check whether high session count predicts generation quality.
2. `Exp43`: SD LoRA-based VS->VI fine-tuning.
   - This does not need to wait for all Exp42-B ablations.
   - Start in parallel using the current best available checkpoint:
     - S01 path: Exp42-B Step4 r=32, 16-token checkpoint
     - S18 path: Exp42-B Step2 r=16, 16-token checkpoint
   - Later Exp42-B improvements can be used as updated initialization if they clearly improve full-test dual-gallery metrics.
   - `C0`: VI scratch LoRA.
   - `C1`: VS LoRA -> VI fine-tune with frozen EEG encoder.
   - `C2`: staged encoder unfreeze if C1 shows no validation improvement by about epoch 50.
   - Minimum success: `C1 > C0`; meaningful target: VI DINO@1 `> 0.20`.
   - Rank design:
     - do not force one shared LoRA rank across subjects
     - include r=32 for VI even when VS best is r=16, because VI conditioning is expected to be weaker
3. `Exp42-B Step 5`: class-preserving image augmentation.
   - S18 r=16/16tok augmentation run is completed.
   - Logged internal partial-test score: DINO@1 `0.4464`, entropy `1.960`, dominant `25.0%`, best_ep `95`.
   - Do not promote it over Step2/Step4 until a matching full-test dual-gallery evaluation is available.
   - Further augmentation, including S01 Step5, should run only if compute remains after S24 and Exp43 start.
   - Treat as secondary cleanup for heart/star confusion, not the next main bottleneck.

### S24 Decision Rule

- `S24 > S01`: session/data volume is a major driver; add more high-session subjects before broad conclusions.
- `S24 ≈ S01`: SD LoRA may be approaching a practical within-subject ceiling once enough data is available.
- `S24 < S01`: subject-specific EEG signal quality matters more than data volume; prioritize per-subject selection for VI transfer.

### Do Not Repeat Unless Hypothesis Changes

- Exp31~40 scratch latent/pixel UNet path.
- Multi-image target training with visually inconsistent alternatives.
- CFG sweep without a trained unconditional path.
- Old single-template or stochastic sample-grid-only conclusions as final metrics.

---

## Project Overview

**Goal**: Decode visual stimulation (VS) EEG signals into corresponding images using a diffusion-based generative model, and evaluate cross-modal alignment between EEG and image representations.

**Data**
- VS EEG: `preproc_for_gan_vs/subj_01.mat` ~ `subj_20.mat`
- VI EEG: `preproc_data_vi/subj_01.mat` ~ `subj_34.mat`
- VS EEG (repeat): `preproc_vs_re/` -- 18 subjects, repeated sessions (see Section 6)
- Structure: `X=(ch=32, time=512, trial)`, `y=(trial,)`, labels 1~9
- Sampling rate: 512 Hz → 1 sec per trial (originally 2 sec, split into 0~1s / 1~2s)
- Trial count: 20 original trials/class x 2 splits = 40 trials/class x 9 classes = 360 trials/subject
- Stimulus images: `preproc_data_vi/images/01.png` ~ `09.png` (9 classes)

---

## Current Direction Update: 2026-06-01

This section is the current operating decision and supersedes older generation-first notes where they conflict.

### Main Decision

- The project target is now explicitly `VS pretraining -> VI transfer readiness`.
- VS-internal retrieval is a necessary benchmark, but it is not sufficient as the final proxy.
- Pixel-space diffusion generation is suspended until VI transfer readiness has been checked.
- Stage 2 DINO latent/readout is the preferred image-output path for the current phase.

### Why Pixel Diffusion Is Suspended

Evidence so far:
- Exp013 confirmed severe generated-class collapse.
- Exp014/015 anti-collapse and prototype-guided losses did not recover stable identity.
- Exp016~019 t-filtered prototype loss found only a tiny gain over the generation baseline.
- Exp021 Stage 2 readout was much stronger than pixel diffusion:
  - mean Top-1 `0.2725` vs generation probe about `0.1138`
  - mean SSIM `0.6785` vs generation probe about `0.048`
  - mean LPIPS `0.3342` vs generation probe about `0.774`

Decision:
- Do not run new pixel diffusion hyperparameter sweeps before VI transfer readiness is evaluated.
- Allowed generation work is limited to fixed-protocol diagnostic exports or Stage 2 qualitative examples.

### Priority Correction

Current next-order logic:

1. Complete representation-side ablations that have high signal-to-cost.
2. Check VI transfer readiness early, not after every VS-only improvement.
3. Validate SupCon beyond `S01/S02/S18`.
4. Split preprocessing ablations instead of combining factors.
5. Use partial-session integration only after the representation/VI readiness questions are clearer.

### Priority Correction: 2026-06-05

Exp25-VI changed the priority order.

The best VS SupCon encoder reached VS Top-1 `0.3333`, but zero-shot VI Top-1 was only `0.1296` against chance `0.1111`.
This is not just a weak transfer result; it challenges the assumption that better VS retrieval will automatically transfer to VI.

Therefore:
- do not run Exp27 partial-session integration as the next main step without diagnostics
- first diagnose whether the failure is a VS/VI distribution gap, channel/session/data issue, or encoder weakness
- treat S23 Top-1 `0.0000` as a possible data-integrity warning

### Updated Exp24+ Plan

| Exp | Purpose | Status / Rule |
|-----|---------|---------------|
| Exp24-A | SupCon + baseline correction only | run separately |
| Exp24-B | SupCon + channel-wise z-score only | run separately |
| Exp24-C | SupCon + baseline correction + z-score | only if A or B is promising |
| Exp25 | Best encoder Stage 2 readout | compare against Exp021 |
| Exp25-VI | VI transfer readiness probe | run in parallel with or immediately after Exp25 |
| Exp26 | All-subject SupCon validation | needed before broad SupCon claims |
| Exp27-preA | S23 data integrity check | required before partial-session integration |
| Exp27-preB | VS/VI latent gap diagnosis | required before partial-session integration |
| Exp27 | Targeted partial-session integration | subject list must be re-selected after preA/preB |
| Exp28+ | Optional pixel-generator follow-up | blocked until Exp25-VI is done |

### Current Decisions After Exp24-26

- `Exp24-A/B/C` preprocessing is **not adopted** for the default pipeline.
- Plain `Exp23-B SupCon only` remains the default encoder despite Exp24-C having slightly better Top-5.
- `Exp25-VI` is treated as a near-chance transfer failure: VI Top-1 `0.1296` vs chance `0.1111`.
- `Exp27` partial-session integration is paused until two gates are complete.
- Gate 1: inspect S23 data integrity because S23 Top-1 `0.0000` may indicate a data issue.
- Gate 2: compare VS vs VI latent alignment to DINO prototypes to diagnose the transfer gap.

Diagnostic scripts added:
- `diagnose_vsre_subject_data.py`
- `diagnose_vs_vi_latent_gap.py`

Diagnostic results:
- `Exp27-preA` S23 raw data check: file structure is valid.
- S23 has one valid file, shape `(5, 3, 9, 4096, 40)`, total trials `135`, per-class `15`, finite fraction `1.0`.
- Therefore S23 Top-1 `0.0000` is not explained by obvious file corruption or label-count imbalance.
- `Exp27-preB` VS/VI latent gap check confirms a real transfer gap.
- S01 margin: VS `-0.1432` -> VI `-0.3796`
- S02 margin: VS `-0.0929` -> VI `-0.2644`
- S18 margin: VS `-0.3700` -> VI `-0.6530`
- VI consistently pushes the true-label prototype farther below the best-wrong prototype.

Interpretation of negative VS margins:
- The SupCon encoder improves Top-1, but the true-label prototype is still below the best-wrong prototype on average.
- This means class compactness improved, but class separation in DINO prototype space is still weak.
- Because VS margins are already negative, simply adding partial VS sessions is unlikely to solve the core VI transfer failure.
- The next experiment should isolate temporal mismatch before further VS-only optimization.

Decision after diagnostics:
- Exp27 partial-session integration is no longer the immediate priority.
- The next main direction should be VI-domain bridging: VI fine-tuning, temporal-resolution matching, or VS encoder initialization followed by VI adaptation.
- VS-only session expansion can still be run as a secondary check, but it should not be expected to solve the VI transfer failure by itself.

S11/S16 note:
- S11 Top-1 `0.4444` with 1 session and S16 Top-1 `0.3056` with 2 sessions are high-performance low-session outliers.
- They should be inspected before assuming session count alone explains performance.
- Candidate checks: raw signal variance/SNR, label distribution, channel statistics, split stability, and repeated-seed variance.

### VI Transfer Readiness Requirement

The first VI check should use the best available SupCon/DINO representation checkpoint:
- primary candidate: `checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon`
- if Exp24 produces a clearly better encoder, use that instead

Minimum report:
- subject overlap between VS and VI
- checkpoint path
- VI data path and split
- metric used
- whether failure is due to encoder quality, VS/VI distribution gap, or script incompatibility

### SupCon Scope Warning

Exp23-B is currently the best three-subject representation result:
- mean Top-1 `0.3333`
- mean Top-3 `0.6252`
- mean Top-5 `0.7615`

However, this is still based on `S01/S02/S18`.
Do not treat this as final evidence until all 18 subjects have been evaluated, especially low-session subjects.

---

## Model Architecture

### 1. Generative Model (EEG → Image)
**File**: `model_128_eegonly_transformer_repa.py`

```

---

## Update: 2026-05-25

This section supersedes older "planned next" notes where they conflict.
Use this section as the current reference point.

### Current Status

- Earlier parts of this file reflect results through `Exp012` reasonably well.
- Actual work continued after that point on:
  - generation collapse diagnosis
  - prototype-guided generation losses
  - updated `preproc_vs_re` partial-session files
- Therefore the authoritative current interpretation is:
  - representation track is still stronger than generation track
  - generation experiments after `Exp013` are treated as `prototype-guided validation`, not full image-generation success

### Experiment 13: Generation Collapse Diagnosis

Target checkpoint:
- `checkpoints_vsre_gen/20260417_113129_ch32_merged_ep300/`

Subjects checked:
- `S01`, `S02`, `S18`

Observed generated-class histogram collapse:
- `S01`: class 1 = `92.59%`
- `S02`: class 1 = `97.22%`
- `S18`: class 1 = `94.44%`

Interpretation:
- the generation problem is not just low accuracy
- it is a real `single-class collapse`
- `Top-1` alone was insufficient to diagnose this

Saved outputs:
- `subjXX/collapse_diag/pred_histogram.csv`
- `subjXX/collapse_diag/pred_histogram.png`
- `subjXX/collapse_diag/confusion_matrix.png`

### Experiment 14: Anti-Collapse Auxiliary Loss

Log:
- `logs/vsre_gen_exp014_anticollapse.log`

Checkpoint:
- `checkpoints_vsre_gen/20260517_150805_ch32_merged_ep300/`

Config:
- `lambda_dino_align = 0.1`
- `lambda_aux_ce = 0.1`
- subjects `1,2,18`

Per-subject result summary:
- `S01`: `DINO@1 = 0.1111`, `entropy = 0.000`, `dominant = 100.0%`
- `S02`: `DINO@1 = 0.1111`, `entropy = 0.000`, `dominant = 100.0%`
- `S18`: `DINO@1 = 0.1111`, `entropy = 0.066`, `dominant = 97.2%`

Conclusion:
- anti-collapse auxiliary loss in this weak form did **not** solve collapse
- outputs remained effectively chance-level in `Top-1`
- class-distribution collapse remained severe

### Experiment 15: Prototype-Guided Generation (First Completed Run)

Log:
- `logs/vsre_gen_exp015_proto.log`

Checkpoint:
- `checkpoints_vsre_gen/20260524_145436_ch32_merged_ep300/`

Config intent:
- same generation base as `Exp014`
- prototype-guided objective emphasized
- still evaluated on subjects `1,2,18`

Results from `results_gen.csv`:

| Subject | best_val_loss | L1 | SSIM | LPIPS | DINO@1 | DINO@3 | DINO@5 |
|---------|---------------|----|------|-------|--------|--------|--------|
| S01 | 0.000435 | 0.832594 | 0.047284 | 0.766464 | 0.111111 | 0.361111 | 0.583333 |
| S02 | 0.000411 | 0.831832 | 0.048693 | 0.767672 | 0.111111 | 0.333333 | 0.547619 |
| S18 | 0.000436 | 0.833928 | 0.046512 | 0.766870 | 0.111111 | 0.324074 | 0.564815 |
| **Mean** | **0.000427** | **0.832785** | **0.047496** | **0.767002** | **0.111111** | **0.339506** | **0.565256** |

Interpretation:
- `Top-1` remained exactly chance-level on all three subjects
- `Top-3/Top-5` were not meaningfully improved vs the earlier subset baseline
- this run should be treated as:
  - a completed `prototype-guided validation` attempt
  - **not** a successful generation improvement

### Experiments 16–19: t-Filtered Prototype Loss (2026-05-25~26)

Root cause identified: at high-noise timesteps (large t), x0_pred is noisy → DINO proto gradient is noise → collapse wins.

**Fix implemented**: `--t_max_proto` arg — proto loss only applied when `t < t_max_proto`.

| Exp | attract/sep | t_max_proto | S01 DINO@1 | S01 entropy | verdict |
|-----|------------|-------------|-----------|------------|---------|
| 016 | 1.0/0.5 | all | 0.1111 | 0.021 | FAIL |
| 017 | 2.0/1.0 | 67 (T/3) | 0.1190 | 0.074 | PARTIAL |
| 018 | 3.0/1.5 | 40 (T/5) | **0.1270** | 0.098 | **NEW BEST** |
| 019 | 5.0/2.5 | 20 (T/10) | 0.1111 | 0.051 | REGRESSION |

**Gen probe conclusion (as of 2026-05-26)**:
- Best: **Exp018 S01 DINO@1=0.1270** (beats Exp006 baseline 0.1265 by +0.4%p)
- Sweet spot: `t_max_proto=40` (T/5) — tighter filter reduces proto signal below threshold
- S02/S18 remain collapse-resistant across all configs
- Gen probe optimization ended; PIVOT to CLIP comparison

### Updated Dataset: preproc_vs_re Partial-Session Files

Current file count in the workspace:
- `71` files detected under `preproc_vs_re/`

Practical change:
- file count is no longer a reliable proxy for valid session count
- some newer files contain only partial valid sessions
- some sessions are skipped during preprocessing because trial counts are insufficient
- at least one anomalous session payload was observed during training-time loading

Implication for all future runs:
- use actual loaded tensor shape / effective trial count
- do not assume legacy fixed session payload
- report effective loaded sessions and trials per subject

### Updated Interpretation of the Project

Current best representation result is still:
- `Exp012` / `V2 + auto-prior`
- mean `Top-1 = 0.2824`
- mean `Top-3 = 0.5648`
- mean `Top-5 = 0.7731`

Current generation conclusion is:
- generation remains unstable
- collapse diagnosis was necessary and was confirmed
- anti-collapse loss (`Exp014`) failed
- first prototype-guided run (`Exp015`) also failed to move beyond chance-level `Top-1`

Therefore:
- the project remains `representation-first`
- generation experiments after `Exp013` should be interpreted as:
  - `prototype-guided validation`
  - not final image-generation success claims

### Current Status (2026-05-27)

**Best representation result**: Exp010 (DINO V2+auto-prior) mean Top-1=0.2824
**Best gen probe result**: Exp018 S01 DINO@1=0.1270 (prototype-guided, t-filtered)
**Gen probe ended**: all proto-loss configs exhausted; pivot complete

**Experiment 20: CLIP image comparison (2026-05-27~29)**

| Subject | CLIP Top-1 | DINO Top-1 (Exp010) | Delta |
|---------|-----------|---------------------|-------|
| S01 | 0.3016 | 0.3611 | -16% |
| S02 | 0.2222 | 0.2083 | **+7%** |
| S18 | 0.1944 | 0.2778 | -30% |
| **Mean** | **0.2394** | **0.2824** | **-15%** |

Config: CLIP ViT-B/32 (openai pretrained, 512-dim), V2+auto-prior, merged sessions, ep200
Checkpoint: `checkpoints_vsre_clip/20260527_190947_ViT-B-32_ch32_merged_ep200`

**Conclusion**: DINO > CLIP on this EEG alignment task. S02 only exception (CLIP marginally better).
Note: S18 CLIP training showed InfoNCE loss → -21 (temperature collapse), but valT1 still improved ep120→200.

**Stage 2 will use DINO latent space** (Exp010 checkpoint).

**Experiment 21: Stage 2 Latent/Readout (2026-05-29)**

Script: `eval_vs_re_stage2_readout.py` — frozen Exp010 encoder, NN retrieval, class image readout

| Subject | Top-1 | Top-3 | Top-5 | SSIM | LPIPS |
|---------|-------|-------|-------|------|-------|
| S01 | 0.3333 | 0.6111 | 0.8016 | 0.7013 | 0.3083 |
| S02 | 0.2063 | 0.5397 | 0.7381 | 0.6586 | 0.3560 |
| S18 | 0.2778 | 0.5648 | 0.8056 | 0.6756 | 0.3384 |
| **Mean** | **0.2725** | **0.5719** | **0.7817** | **0.6785** | **0.3342** |

**Stage 2 vs Gen Probe comparison:**
- Top-1: 0.2725 vs 0.1138 → **+2.4x**
- SSIM: 0.6785 vs ~0.048 → **+14x**
- LPIPS: 0.3342 vs ~0.774 → **+2.3x better**

**Conclusion**: Latent readout (Stage 2) dramatically outperforms pixel diffusion (Stage 1).
Bypassing diffusion is the most practical short-term path for image output quality.

**Experiment 22: TTA Evaluation (2026-05-30)**

Script: `eval_vs_re_exp22_tta.py` — no retraining, Exp010 checkpoint, N_TTA=8, noise_std=0.03, max_shift=10 samples

| Subject | Base Top-1 | TTA Top-1 | Δ T1 | Base Top-3 | TTA Top-3 | Base Top-5 | TTA Top-5 |
|---------|-----------|-----------|------|-----------|-----------|-----------|-----------|
| S01 | 0.3333 | 0.3175 | -0.0159 | 0.6111 | 0.6587 | 0.8016 | 0.8254 |
| S02 | 0.2063 | 0.2381 | +0.0317 | 0.5397 | 0.5556 | 0.7381 | 0.7619 |
| S18 | 0.2778 | 0.2407 | -0.0370 | 0.5648 | 0.5926 | 0.8056 | 0.7963 |
| **Mean** | **0.2725** | **0.2654** | **-0.0071** | **0.5719** | **0.6023** | **0.7817** | **0.7945** |

Config: checkpoint=`20260427_095215_ch32_merged_ep200` (Exp010 V2+auto-prior), N_TTA=8, noise+shift only
CSV: `checkpoints_vsre_dino/20260427_095215_ch32_merged_ep200/exp022_tta_results.csv`

**Conclusion**: TTA did NOT improve Top-1 (mean Δ=-0.0071). S02 only exception (+0.0317).
TTA improves Top-3 (+0.0303) and Top-5 (+0.0128) — broader rank tolerance, not better discrimination.
Stage 2 readout re-run skipped — Exp021 result stands (mean Top-1=0.2725).

**Experiment 23-A: InfoNCE baseline (reference)**
- Exp010/012 checkpoint: `20260427_095215_ch32_merged_ep200`, mean Top-1=0.2824 (already documented above)

**Experiment 23-B: SupCon only (2026-05-30)**

Script: `train_vs_re_dino.py --loss_type supcon --encoder_type v2 --eeg_occipital_ids auto --subject_ids 1,2,18 --epochs 200`
Checkpoint: `checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon`

| Subject | Top-1 | Top-3 | Top-5 | vs Exp010 Top-1 |
|---------|-------|-------|-------|-----------------|
| S01 | 0.3810 | 0.6429 | 0.7460 | +0.0199 |
| S02 | **0.3413** | 0.6587 | 0.8254 | **+0.1330** |
| S18 | 0.2778 | 0.5741 | 0.7130 | +0.0000 |
| **Mean** | **0.3333** | **0.6252** | **0.7615** | **+0.0509** |

**Conclusion**: SupCon significantly outperforms InfoNCE on Top-1 (mean +5.09%p). S02 especially benefits (+13.3%p). SupCon removes same-class false negatives structurally → stronger class compactness. New best mean Top-1=0.3333.

**Experiment 23-C: SupCon + proto (2026-05-31)**

Script: `train_vs_re_dino.py --loss_type supcon_proto --encoder_type v2 --eeg_occipital_ids auto --subject_ids 1,2,18 --epochs 200`
Checkpoint: `checkpoints_vsre_dino/20260531_123132_ch32_merged_ep200_supcon_proto`

| Subject | Top-1 | Top-3 | Top-5 |
|---------|-------|-------|-------|
| S01 | 0.3571 | 0.6429 | 0.7698 |
| S02 | 0.3175 | 0.6905 | 0.8492 |
| S18 | 0.2685 | 0.6019 | 0.7685 |
| **Mean** | **0.3144** | **0.6451** | **0.7959** |

**Exp23 Ablation Summary:**

| Config | Mean Top-1 | Mean Top-3 | Mean Top-5 | vs InfoNCE |
|--------|-----------|-----------|-----------|-----------|
| 23-A InfoNCE (Exp010) | 0.2824 | 0.5648 | 0.7731 | reference |
| **23-B SupCon only** | **0.3333** | 0.6252 | 0.7615 | **+0.0509** |
| 23-C SupCon+proto | 0.3144 | **0.6451** | **0.7959** | +0.0320 |

**Conclusion**: 
- **Best Top-1: 23-B (SupCon only)** — new project best mean Top-1=0.3333
- Proto loss hurts Top-1 slightly (-0.0189) but improves Top-3/5 (+0.0199/+0.0344)
- SupCon selected as the base loss for Exp24+; proto optional for Top-3/5 use cases
- Best checkpoint for downstream: `20260530_095045_ch32_merged_ep200_supcon`

**Experiment 24-C: SupCon + baseline_correct + ch_zscore (2026-06-01)**

Script: `train_vs_re_dino.py --loss_type supcon --baseline_correct --ch_zscore --encoder_type v2 --eeg_occipital_ids auto --subject_ids 1,2,18 --epochs 200`
Checkpoint: `checkpoints_vsre_dino/20260601_154813_ch32_merged_ep200_supcon_bl_zs`

| Subject | Top-1 | Top-3 | Top-5 | vs Exp23-B |
|---------|-------|-------|-------|-----------|
| S01 | **0.4048** | 0.6984 | 0.8254 | +0.0238 |
| S02 | 0.3016 | 0.5714 | 0.7460 | −0.0397 |
| S18 | 0.2500 | 0.6019 | 0.7500 | −0.0278 |
| **Mean** | **0.3188** | **0.6239** | **0.7738** | **−0.0145** |

**Conclusion**: Combined preprocessing hurts S02/S18 while helping S01. Mean Top-1 below Exp23-B (0.3333).
Split ablation (24-A, 24-B) required to identify which factor is causal.

**Experiment 25-VI: VS→VI Zero-Shot Transfer (2026-06-01)**

Script: `eval_vs_re_exp25_vi_transfer.py`
VS encoder: `checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon` (Exp23-B SupCon)
VI data: `preproc_data_vi/subj_XX.mat` — X=(ch=32, time=512, trial), labels 1~9

| Subject | VI Top-1 | VI Top-3 | VI Top-5 | n |
|---------|---------|---------|---------|---|
| S01 | **0.1667** | 0.3704 | 0.6296 | 54 |
| S02 | 0.1111 | 0.3148 | 0.5741 | 54 |
| S18 | 0.1111 | 0.3889 | 0.4815 | 54 |
| **Mean** | **0.1296** | **0.3580** | **0.5617** | |
| Random | 0.1111 | 0.3333 | 0.5556 | |

VS→VI transfer gap: 0.3333 − 0.1296 = **0.2037**

**Diagnosis**: PARTIAL transfer — S01 above chance (+0.0556), S02/S18 at chance.
Known causes:
1. Temporal resolution mismatch: VI=512 samples (512Hz, 1sec) vs VS=2048 samples (1024Hz, 2sec)
2. VS/VI paradigm gap: visual stimulation vs visual imagery (fundamentally different neural processes)
3. Only 54 test trials per subject (small evaluation set)

**Conclusion**: Zero-shot VS→VI transfer shows minimal signal. To improve:
- Train on VI data directly (fine-tuning or full VI pretraining)
- Or bridge the temporal mismatch (upsample VI to 2048 / train shorter VS model)
- Or use the VS encoder only as an initialization, with VI fine-tuning

CSV: `checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon/exp025_vi_transfer_results.csv`

**Experiment 24-A: SupCon + baseline correction only (2026-06-02)**

Checkpoint: `checkpoints_vsre_dino/20260602_085718_ch32_merged_ep200_supcon_bl`

| Subject | Top-1 | Top-3 | Top-5 | vs Exp23-B |
|---------|-------|-------|-------|-----------|
| S01 | 0.4048 | 0.7302 | 0.8175 | +0.0238 |
| S02 | 0.2619 | 0.5952 | 0.7698 | −0.0794 |
| S18 | 0.2593 | 0.5463 | 0.7315 | −0.0185 |
| **Mean** | **0.3086** | **0.6239** | **0.7729** | **−0.0247** |

**Experiment 24-B: SupCon + ch_zscore only (2026-06-02)**

Checkpoint: `checkpoints_vsre_dino/20260602_085726_ch32_merged_ep200_supcon_zs`

| Subject | Top-1 | Top-3 | Top-5 | vs Exp23-B |
|---------|-------|-------|-------|-----------|
| S01 | 0.3810 | 0.6667 | 0.8175 | +0.0000 |
| S02 | 0.3095 | 0.5159 | 0.7222 | −0.0318 |
| S18 | 0.2963 | 0.5463 | 0.7407 | +0.0185 |
| **Mean** | **0.3289** | **0.5763** | **0.7601** | **−0.0044** |

**Exp24 Complete Ablation:**

| Config | Mean Top-1 | Mean Top-3 | Mean Top-5 |
|--------|-----------|-----------|-----------|
| **Exp23-B SupCon only** | **0.3333** | **0.6252** | 0.7615 |
| Exp24-B SupCon+zscore | 0.3289 | 0.5763 | 0.7601 |
| Exp24-C SupCon+bl+zs | 0.3188 | 0.6239 | **0.7738** |
| Exp24-A SupCon+baseline | 0.3086 | 0.6239 | 0.7729 |

**Conclusion**: No preprocessing improves mean Top-1 over plain SupCon. **Exp23-B remains the best encoder** (mean Top-1=0.3333). Baseline correction and ch_zscore are subject-inconsistent: help S01 but hurt S02/S18 on average.

**Experiment 25: Stage 2 Readout with SupCon Encoder (2026-06-04)**

Script: `eval_vs_re_stage2_readout.py --ckpt_dir checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon`
Encoder: Exp23-B SupCon (V2+auto-prior, mean VS Top-1=0.3333)
CSV: `checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon/stage2_readout_results.csv`

| Subject | Top-1 | Top-3 | Top-5 | SSIM | LPIPS |
|---------|-------|-------|-------|------|-------|
| S01 | 0.3810 | 0.6429 | 0.7460 | 0.7520 | 0.2649 |
| S02 | 0.3413 | 0.6587 | 0.8254 | 0.6563 | 0.3282 |
| S18 | 0.2778 | 0.5741 | 0.7130 | 0.6456 | 0.3507 |
| **Mean** | **0.3333** | **0.6252** | **0.7615** | **0.6846** | **0.3146** |

**vs Exp021 (InfoNCE Stage 2):** Top-1 +0.0608 (+22%), SSIM +0.0061, LPIPS −0.0196 (better)
**vs Gen Probe Exp018:** Top-1 0.3333 vs 0.1138 → Stage 2 readout 3× better
**Conclusion**: SupCon encoder significantly improves Stage 2 readout over InfoNCE. Best image-output path confirmed.

**FBCSP+SVM missing-subject rerun (2026-06-08)**

Script: `train_vs_re_fbcsp_svm.py --subject_ids 6,8,22 --kernel rbf --csp_components 2`

Reason:
- Earlier FBCSP table covered 18 subjects and missed newly added `S06/S08/S22`.
- `train_vs_re_fbcsp_svm.py` was updated to handle the current `load_subject_vsre()` return format and empty validation splits.

New results:

| Subject | Sessions | Train | Val | Test | Val Acc | Test Acc |
|---------|---------:|------:|----:|-----:|--------:|---------:|
| S06 | 1 | 108 | 9 | 18 | 0.0000 | 0.0556 |
| S08 | 1 | 63 | 0 | 18 | nan | 0.1667 |
| S22 | 1 | 108 | 9 | 18 | 0.2222 | 0.1111 |

Updated FBCSP+SVM summary:
- subject count: `21`
- mean Test Acc: `0.1210`
- random: `0.1111`
- max: `0.2593` (`S10`)
- min: `0.0000` (`S35`)

Conclusion:
- Adding `S06/S08/S22` does not change the interpretation.
- FBCSP+SVM remains a weak sanity baseline, near chance.
- `S08` has `val=0`; use its test accuracy only and avoid validation-based conclusions for this subject.

**Experiment 26: All-Subject SupCon Validation (2026-06-04)**

Script: `train_vs_re_dino.py --loss_type supcon --encoder_type v2 --eeg_occipital_ids auto --subject_ids all --epochs 200`
Checkpoint: `checkpoints_vsre_dino/20260604_091352_ch32_merged_ep200_supcon`

| Subject | Sessions | Top-1 | Top-3 | Top-5 |
|---------|----------|-------|-------|-------|
| S01 | 9 | 0.3968 | 0.7063 | 0.8016 |
| S02 | 9 | 0.3016 | 0.6190 | 0.7540 |
| S03 | 1 | 0.1111 | 0.2778 | 0.5000 |
| S04 | 2 | 0.0741 | 0.3333 | 0.6296 |
| S05 | 1 | 0.1111 | 0.3333 | 0.5556 |
| S06 | 1 | 0.0556 | 0.2778 | 0.5556 |
| S08 | 1 | 0.0556 | 0.2778 | 0.6111 |
| S09 | 4 | 0.1429 | 0.3968 | 0.6190 |
| S10 | 2 | 0.1481 | 0.4815 | 0.6296 |
| S11 | 1 | **0.4444** | 0.6667 | 0.8333 |
| S16 | 2 | 0.3056 | 0.6944 | 0.7500 |
| S18 | 8 | 0.3148 | 0.6296 | 0.7593 |
| S19 | 2 | 0.1481 | 0.2963 | 0.5556 |
| S20 | 1 | 0.2222 | 0.3889 | 0.6111 |
| S21 | 1 | 0.1667 | 0.4444 | 0.7778 |
| S22 | 1 | 0.1667 | 0.3333 | 0.5000 |
| S23 | 1 | 0.0000 | 0.2222 | 0.4444 |
| S24 | 10 | 0.3259 | 0.6296 | 0.7259 |
| S28 | 6 | 0.3000 | 0.6111 | 0.7778 |
| S29 | 5 | 0.2361 | 0.5417 | 0.6944 |
| S35 | 2 | 0.1944 | 0.4722 | 0.6111 |
| **Mean (21)** | | **0.2010** | **0.4588** | **0.6522** |
| Random | | 0.1111 | 0.3333 | 0.5556 | |

**High-session subjects (≥4 sessions):** S01, S02, S09, S18, S24, S28, S29
- Mean Top-1: **0.2883** (range: 0.1429–0.3968)

**Low-session subjects (1–2 sessions):** S03–S08, S10–S11, S16, S19–S23, S35
- Mean Top-1: **0.1431** (range: 0.0000–0.4444)
- Notable exceptions: S11 (0.4444, 1 sess), S16 (0.3056, 2 sess), S20 (0.2222, 1 sess)

**vs Exp04 InfoNCE all-subject (mean Top-1=0.1386):** SupCon +0.0624 (+45%)
**vs Exp23-B SupCon 3-subject (mean Top-1=0.3333):** All-subject mean 0.2010 lower — expected, driven by many 1-session subjects
**S01/S02/S18 subset in Exp26:** mean=(0.3968+0.3016+0.3148)/3=0.3377 ≈ Exp23-B (0.3333), consistent

**Conclusion**: SupCon generalizes beyond S01/S02/S18. Session count remains the dominant factor (high-sess mean 0.2883 vs low-sess 0.1431). S11 and S16 are unexpected high performers at low session count — worth investigating. S23 (1 sess) = 0.0000, no signal.

**Active**: Exp028-A — VI temporal-resolution matching test
- Command: `eval_vs_re_exp25_vi_transfer.py --subject_ids 1,2,18 --target_time 2048`
- Purpose: test whether VI 512-sample input length explains the VS→VI transfer failure.
- Decision threshold: if VI Top-1 rises meaningfully above `0.15`, temporal mismatch is a major factor; otherwise paradigm/domain gap is the likely dominant factor.

**Experiment 28-A: VI 512→2048 temporal interpolation (2026-06-05)**

Script: `eval_vs_re_exp25_vi_transfer.py --subject_ids 1,2,18 --target_time 2048`
CSV: `checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon/exp025_vi_transfer_results_t2048.csv`

| Subject | VI Top-1 | VI Top-3 | VI Top-5 | n |
|---------|---------|---------|---------|---|
| S01 | 0.0741 | 0.3333 | 0.4630 | 54 |
| S02 | 0.1111 | 0.3704 | 0.5370 | 54 |
| S18 | 0.0926 | 0.3704 | 0.5370 | 54 |
| **Mean** | **0.0926** | **0.3580** | **0.5123** | |
| Random | 0.1111 | 0.3333 | 0.5556 | |

Comparison:
- Original Exp25-VI Top-1: `0.1296`
- Upsampled Exp28-A Top-1: `0.0926`
- Result: temporal interpolation does not help and degrades Top-1 below chance.

Conclusion:
- VI transfer failure is not explained by simple input-length mismatch.
- Next priority should be VI adaptation / fine-tuning or a domain-bridging encoder, not VS-only partial-session integration.

**Pre-Exp28-B data distribution audit (2026-06-05)**

Script: `diagnose_class_trial_distribution.py --dataset both --subject_ids all`

Outputs:
- `class_trial_distribution_detail.csv`
- `class_trial_distribution_summary.csv`

Findings:
- VS-re subjects currently loaded: `21`
- VI subjects: `34`
- VS/VI subject overlap: `20`
- Overlap: `S01,S02,S03,S04,S05,S06,S08,S09,S10,S11,S16,S18,S19,S20,S21,S22,S23,S24,S28,S29`
- Current loader sees all audited VS/VI subjects as balanced 9-class tasks.
- VI: every subject has `60` trials/class, total `540`, test `6`/class.
- VS: trial count varies by subject; S11/S23 have `15` trials/class, S16 has `27` trials/class.

S23 interpretation:
- S23 is structurally valid and balanced under the current loader.
- S23 should be treated as `valid-low-signal` for now, not corrupt data.
- Keep S23 in aggregate reports, but flag it explicitly as a low-signal outlier.

S11/S16 interpretation:
- S11 and S16 are not easier because of fewer classes or class imbalance.
- S11: 9 classes, `15` trials/class, test `2`/class.
- S16: 9 classes, `27` trials/class, test `4`/class.
- Their high Exp26 scores likely reflect signal quality, split variance, or subject-specific neural/recording factors.
- Investigate with repeated seeds and raw signal/channel statistics, not session count alone.

**Exp28-B strategy decision**

Use a required three-way VI adaptation comparison on the same subjects and splits:

1. `Exp28-B0 = VI scratch baseline`
   - random-init EEG-DINO model trained on VI only
   - use the same VI split, epochs, early stopping, and metrics as B1
   - this is mandatory; without B0, VS pretraining benefit cannot be claimed

2. `Exp28-B1 = VS encoder -> VI direct fine-tuning`
   - initialize from Exp23-B SupCon checkpoint
   - train on VI data for the same subject
   - this directly tests whether VS pretraining is useful as initialization
   - compare primarily against B0 and Exp25-VI zero-shot

3. `Exp28-B2 = frozen VS encoder + VI linear probe`
   - lower cost, but capped by the currently weak VS latent separation
   - run alongside B1 because it explains whether B1 uses preserved VS latent quality or overwrites it during fine-tuning
   - ep10-20 is sufficient as a diagnostic

4. `Exp28-B3 = VS+VI joint SupCon`
   - highest cost and highest confound risk
   - defer until B1/B2 show whether VI adaptation is viable

Initial target subjects:
- `S01/S02/S18`
- reason: they have prior Exp25-VI zero-shot results and sufficient VS checkpoints for paired comparison.

Training rule:
- B0/B1: `ep100`, early stopping with patience around `20`
- B2: linear probe around `ep20`
- do not include S08 in validation-dependent adaptation runs unless split handling is fixed, because audit found `vs,S08,val=0`

**Exp027 status**
- Exp027 targeted partial-session integration is paused.
- It may be run later only as a secondary VS-data check or VI-bridging support experiment.
- It should not be treated as the next main step because Exp27-preB showed negative VS margins and worse VI margins.

**Experiment 28-B: VI Adaptation — Scratch / Finetune / Linear Probe (2026-06-08)**

Script: `train_vi_dino_adapt.py --mode all --subject_ids 1,2,18 --epochs 100 --patience 20 --linear_epochs 20`
VS checkpoint: `checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon` (Exp23-B SupCon)
VI data: `preproc_data_vi/` — 432 train / 54 val / 54 test per subject, time=512
CSV: `checkpoints_vi_adapt/20260605_182521_exp28b_all/results_vi_adapt.csv`

| Subject | B0 scratch | B1 finetune | B2 linear | zero-shot (Exp25-VI) | random |
|---------|-----------|-------------|-----------|----------------------|--------|
| S01 | 0.0926 | 0.1481 | **0.1852** | 0.1667 | 0.1111 |
| S02 | 0.0741 | 0.0926 | 0.0926 | 0.1111 | 0.1111 |
| S18 | 0.1296 | 0.1111 | **0.1852** | 0.1111 | 0.1111 |
| **Mean** | **0.0988** | **0.1173** | **0.1543** | **0.1296** | **0.1111** |

**Key findings:**
- B0 (scratch) mean=0.0988 — below random; model architecture (2048-sample design) poorly fits VI 512-sample input
- B1 (finetune) mean=0.1173 — just above random; VS init does not strongly help; gradient updates may destroy VS latent structure on small VI data
- B2 (linear) mean=0.1543 — best, above zero-shot (0.1296); frozen VS features carry marginal VI signal
- **B2 > zero-shot > B1 > B0**: frozen VS encoder better than fine-tuned — VS latent quality degrades when updated on small VI dataset
- Do not over-interpret this as confirmed `temporal mismatch + paradigm gap`. The current evidence has not separated `512-sample VI input architecture mismatch`, small VI fine-tuning data / overfitting, and true VS/VI paradigm gap.
- Exp28-A showed that naive `512 -> 2048` upsampling hurts, but it does not prove that a native 512-sample encoder cannot learn VI.

**Conclusion**: VS→VI transfer is weak, but the failure mode is not fully isolated. B2 linear probe gives the best result in this run, while B1 fine-tuning underperforms B2, suggesting small VI data / fine-tuning instability may be at least as important as temporal mismatch.

**Objective Override (2026-06-08)**: Generation is now primary goal. VI transfer demoted.
Next question: does SupCon encoder reduce generation collapse?

**Experiment 31: SupCon-Initialized Generator + Anti-Collapse (2026-06-08)**

Script: `train_vs_re_gen.py --subject_ids 1,2,18 --encoder_version v2 --eeg_tf_layers 4 --eeg_occipital_ids auto --supcon_ckpt checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon --epochs 300 --lambda_dino_align 0.1 --lambda_aux_ce 0.1`
Checkpoint: `checkpoints_vsre_gen/20260608_091209_ch32_merged_ep300_supcon_init`

| Subject | DINO@1 | entropy | dominant | proto_sim | margin | vs Exp13 dominant |
|---------|--------|---------|----------|-----------|--------|-------------------|
| S01 | 0.1111 | **0.316** | **77.8%** | 0.051 | -0.277 | 92.6% → **77.8%** (↓15%p 개선) |
| S02 | 0.1111 | 0.000 | **100%** | 0.061 | -0.460 | 97.2% → 100% (악화) |
| S18 | **0.130** | 0.066 | 97.2% | 0.079 | -0.362 | 94.4% → 97.2% (소폭 악화) |
| **Mean** | **0.117** | **0.127** | **91.7%** | **0.063** | **-0.366** | Exp13 mean 94.75% |

**결론**: S01에서 partial 개선 (9 sessions, 가장 많은 데이터). S02/S18 여전히 붕괴.
SupCon 초기화 + anti-collapse loss 조합은 FiLM vector conditioning 한계를 극복 불가.
**→ Generator conditioning mechanism (FiLM)이 bottleneck. Cross-attention 또는 latent diffusion으로 전환.**

**Experiment 32: SD 1.5 VAE Latent Diffusion (2026-06-09)**

Script: `train_vs_re_latent_gen.py --subject_ids 1,2,18 --epochs 300 --supcon_ckpt checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon`
Checkpoint: `checkpoints_vsre_latent_gen/20260609_154614_ch32_ep300_supcon`
Architecture: frozen SD VAE → 4×16×16 latent → LatentUNet (FiLM, cond=512-dim) → decode → DINO eval

| Subject | DINO@1 | entropy | dominant | vs Exp31 (pixel) | vs Exp13 (baseline) |
|---------|--------|---------|----------|------------------|---------------------|
| S01 | **0.3400** | 0.324 | 92.0% | 0.1111 → **0.3400** | 0.1111 → **0.3400** |
| S02 | **0.2600** | 0.098 | 98.0% | 0.1111 → **0.2600** | 0.1111 → **0.2600** |
| S18 | **0.2200** | **0.822** | **80.0%** | 0.1296 → **0.2200** | 0.1111 → **0.2200** |
| **Mean** | **0.2733** | **0.415** | **90.0%** | **+0.156 (+141%)** | **+0.162 (+146%)** |
| random | 0.1111 | — | — | | |

**vs Stage 2 readout (Exp25, non-generative):** 0.3333 vs 0.2733 — latent gen now within 6%p of retrieval baseline!

**결론**: **프로젝트 최초로 generative model이 chance를 크게 넘어섬** (mean DINO@1=0.2733, 2.5× above chance).
- VAE latent space가 pixel space보다 훨씬 학습하기 쉬움 (loss 0.018 vs 0.00069)
- S18: dominant=80%, entropy=0.82 → 가장 다양한 생성, 적절한 class identity
- S01/S02: higher collapse (92-98%) but meaningful DINO@1 (0.26-0.34)
- Encoder unfrozen 상태에서도 SupCon 특성이 유지되어 conditioning 작동 확인

**Experiment 33: VAE Latent Diffusion + Frozen SupCon Encoder (2026-06-09)**

Script: `train_vs_re_latent_gen.py --subject_ids 1,2,18 --epochs 300 --supcon_ckpt checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon --freeze_encoder`
Checkpoint: `checkpoints_vsre_latent_gen/20260609_*_ch32_ep300_supcon_frozen`

| Subject | DINO@1 | entropy | dominant | vs Exp32 unfrozen | vs random |
|---------|--------|---------|----------|-------------------|-----------|
| S01 | 0.3000 | 0.098 | 98.0% | 0.3400 → 0.3000 | +2.7× |
| S02 | 0.2600 | 0.098 | 98.0% | 0.2600 → 0.2600 | +2.3× |
| S18 | **0.4000** | **1.686** | **38.0%** | 0.2200 → **0.4000** | +3.6× |
| **Mean** | **0.3200** | **0.627** | **78.0%** | 0.2733 → **0.3200** | **+2.9×** |

**Ablation summary:**

| Config | Mean DINO@1 | Mean entropy | Mean dominant | Note |
|--------|-------------|-------------|---------------|------|
| Exp13 InfoNCE pixel | 0.1111 | ~0.03 | 94.75% | baseline |
| Exp31 SupCon-init pixel | 0.1173 | 0.127 | 91.7% | FiLM fails |
| Exp32 VAE latent (unfrozen enc) | 0.2733 | 0.415 | 90.0% | breakthrough |
| **Exp33 VAE latent (frozen enc)** | **0.3200** | **0.627** | **78.0%** | **best generation** |
| Exp25 Stage 2 readout (retrieval) | 0.3333 | — | — | non-generative ref |

**결론**:
- **Frozen SupCon + VAE latent diffusion = 프로젝트 최고 generation 성능** (mean DINO@1=0.3200)
- Stage 2 retrieval baseline (0.3333)과 사실상 동등 — generative model이 처음으로 retrieval 수준에 도달
- S18: DINO@1=0.4000, dominant=38%, entropy=1.686 → 가장 다양하고 정확한 생성
- SupCon encoder를 frozen으로 유지 시 DINO-aligned representation이 보존되어 더 효과적

**Experiment 34: All-Subject Frozen SupCon + VAE Latent Diffusion (2026-06-09)**

Script: `train_vs_re_latent_gen.py --subject_ids all --epochs 300 --supcon_ckpt checkpoints_vsre_dino/20260604_091352_ch32_merged_ep200_supcon --freeze_encoder`
Checkpoint: `checkpoints_vsre_latent_gen/20260609_173254_ch32_ep300_supcon_frozen`

| Subject | Sessions | DINO@1 | entropy | dominant |
|---------|----------|--------|---------|----------|
| S01 | 9 | **0.3200** | 0.168 | 96.0% |
| S02 | 9 | **0.2600** | 0.098 | 98.0% |
| S03 | 1 | 0.1111 | 1.120 | 55.6% |
| S04 | 2 | 0.0370 | 0.754 | 77.8% |
| S05 | 1 | 0.1111 | 1.051 | 66.7% |
| S06 | 1 | 0.1111 | 0.961 | 72.2% |
| S08 | 1 | 0.0556 | 0.868 | 66.7% |
| S09 | 4 | **0.2000** | 1.275 | 56.0% |
| S10 | 2 | 0.0741 | 1.838 | 25.9% |
| S11 | 1 | **0.2222** | 1.480 | 33.3% |
| S16 | 2 | 0.1667 | 1.399 | 41.7% |
| S18 | 8 | **0.2400** | 0.000 | 100.0% |
| S19 | 2 | 0.1111 | 0.158 | 96.3% |
| S20 | 1 | 0.1111 | 1.226 | 55.6% |
| S21 | 1 | 0.1111 | 1.567 | 27.8% |
| S22 | 1 | 0.1111 | 0.961 | 72.2% |
| S23 | 1 | 0.1111 | 1.080 | 66.7% |
| S24 | 10 | **0.2800** | 0.265 | 94.0% |
| S28 | 6 | **0.1800** | 0.265 | 94.0% |
| S29 | 5 | **0.2200** | 0.906 | 72.0% |
| S35 | 2 | 0.1111 | 1.088 | 69.4% |
| **Mean (21)** | | **0.1550** | **0.882** | **68.5%** |

High-session (≥4 sess): S01,S02,S09,S18,S24,S28,S29 → mean DINO@1=**0.243**
Low-session (1-2 sess): rest 14 subjects → mean DINO@1=**0.121**

**최종 Generation 성능 비교표:**

| Experiment | Config | Mean DINO@1 (S01/02/18) | Mean dominant | 비고 |
|-----------|--------|-------------------------|---------------|------|
| Exp13 | InfoNCE pixel | 0.111 | 94.75% | collapse baseline |
| Exp31 | SupCon-init pixel | 0.117 | 91.7% | FiLM insufficient |
| Exp32 | VAE latent (unfrozen enc) | 0.273 | 90.0% | 첫 돌파구 |
| **Exp33** | **VAE latent (frozen enc)** | **0.320** | **78.0%** | **best 3-subj** |
| Exp34 (all-subj) | VAE latent (frozen enc) | 0.155 | 68.5% | high-sess: 0.243 |
| Exp25 Stage 2 retrieval | — | 0.333 | — | non-gen ref |

**결론**:
1. **VAE latent diffusion + frozen SupCon encoder가 최적 generation 구성** 확인
2. S18 Exp33: DINO@1=0.40, dominant=38% — 첫 번째로 retrieval에 근접하는 generative result
3. 세션 수가 generation quality의 주요 결정 요인: 고세션 mean 0.243 vs 저세션 0.121
4. 전체 21 subjects mean 0.155 — 여전히 chance(0.111) 이상, 의미있는 신호
5. Pixel diffusion 대비 3× 이상 개선 달성

**Experiment 35: S18 vs S01 Latent-Generation Diagnosis (2026-06-10)**

Tool: inline diagnostic script (no new model training)

**Data split check**: Both S01 and S18 — perfectly balanced (9 classes × equal trials/class). Data imbalance eliminated.

**Generated confusion matrix (Exp33 checkpoint):**

S01: DINO@1=0.127, dominant=cls3 (89%=112/126)
- All 9 true classes → predicted as cls3
- Only cls3 itself achieves correct predictions (13/14)
- Complete single-class collapse to cls3

S18: DINO@1=0.194, dominant=cls3 (35%=38/108)
- cls2,cls3,cls4,cls5,cls6 all get some correct predictions
- Multiple classes generated with reasonable diversity

**EEG latent class separability:**

| Subject | Inter-class cosine sim (mean) | Min sim | Collapse |
|---------|-------------------------------|---------|---------|
| S01 | **0.7515** | 0.4617 | 89% cls3 |
| S18 | **0.5857** | 0.2346 | 35% |

**Root cause confirmed**: S01 EEG latents have HIGH inter-class similarity (0.75) → 9 class vectors are nearly indistinguishable → FiLM UNet cannot condition on class → generates only the "easiest" class (cls3).
S18's lower inter-class similarity (0.59) provides sufficient class signal for UNet conditioning.

Note: S01 has BETTER VS retrieval accuracy (Top-1=0.381 vs S18=0.278), but latent class separation is WORSE. Retrieval metric captures small rank-margins; generation requires larger absolute class separation.

**Exp36 direction**: Re-train S01 SupCon encoder with lower temperature (0.05 vs default 0.1) to force harder class separation. Then re-run frozen-encoder latent generation.

**Experiment 36: S01 SupCon Re-training (temp=0.05) (2026-06-10)**

Command: `train_vs_re_dino.py --loss_type supcon --encoder_type v2 --eeg_occipital_ids auto --subject_ids 1 --epochs 300 --temperature 0.05`
Checkpoint: `checkpoints_vsre_dino/20260610_102850_ch32_merged_ep300_supcon`

Results:
- S01 VS Top-1 = **0.4206** (vs Exp23-B 0.3810, +4%p)
- Inter-class cosine sim = **0.8326** (vs temp=0.1: 0.7515) — 더 높아짐!

**결론**: Lower temperature → harder SupCon loss → EEG class 신호 부족 시 모든 class가 더 tight한 공간에 집중 → mean끼리 더 유사. Top-1은 미세 margin으로 retrieval 가능하지만 generation UNet은 큰 class gap이 필요. Temperature 조정으로는 S01 generation collapse 해결 불가.

**→ Exp37 필요: cross-attention conditioning으로 FiLM vector 방식 교체**

**Experiment 37: Cross-Attention Latent UNet (2026-06-10)**

Script: `train_vs_re_latent_gen.py --subject_ids 1,18 --epochs 300 --supcon_ckpt checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon --freeze_encoder --use_cross_attn`
Checkpoint: `checkpoints_vsre_latent_gen/20260610_105755_ch32_ep300_supcon_frozen_ca`

| Subject | FiLM (Exp33) | CA (Exp37) | CA entropy | CA dominant |
|---------|-------------|------------|------------|-------------|
| S01 | 0.3000 | **0.3000** | 0.325 | 90% |
| S18 | 0.4000 | **0.4200** | 1.832 | 32% |
| **Mean** | 0.3200 | **0.3600** | 1.079 | 61% |
| Stage 2 retrieval | 0.3333 | — | — | — |

**결론**:
- Cross-attention: S18 소폭 개선 (0.40→0.42), S01은 변화 없음 (0.30→0.30)
- S01 collapse 확정 원인: **EEG inter-class sim=0.75 (S18=0.59)** — architecture 한계 아님, EEG 신호 자체 한계
- Mean DINO@1=0.36 > Stage 2 retrieval 0.333 — **generative model이 retrieval baseline 초과**

**프로젝트 Generation Phase 성과 요약:**

| Method | Mean DINO@1 | 비고 |
|--------|-------------|------|
| Pixel InfoNCE (Exp13) | 0.111 | collapse |
| VAE latent FiLM frozen (Exp33) | 0.320 | best FiLM |
| **VAE latent CA frozen (Exp37)** | **0.360** | **best overall** |
| Stage 2 retrieval (Exp25) | 0.333 | non-gen ref |

**Experiment 38: Gen-WS-03 — S18 VI Latent Generation (2026-06-11)**

Script: `train_vi_latent_gen.py --subject_ids 18 --epochs 100 --patience 25 --modes C0,C1`
VS encoder source: Exp37 S18 checkpoint (`checkpoints_vsre_latent_gen/20260610_105755_ch32_ep300_supcon_frozen_ca/subj18_best.pt`)
VI data: `preproc_data_vi/subj_18.mat` — 432 train / 54 val / 54 test, time=512

| Mode | DINO@1 | DINO@3 | DINO@5 | entropy | dominant | best_ep |
|------|--------|--------|--------|---------|---------|---------|
| C0 VI scratch | 0.1111 | 0.3333 | 0.5185 | 0.432 | 90.7% | 99 |
| **C1 VS-frozen** | **0.1296** | **0.3519** | **0.5185** | 0.367 | 92.6% | 99 |
| 성공 기준 | ≥0.21 | | | | | |
| vs Exp28-B2 linear | 0.1852 | — | — | — | — | — |

**결론**:
- C1 > C0 ✅ (최소 기준 충족 — VS pretraining이 VI generation에 도움됨)
- DINO@1=0.1296 < 0.21 ❌ (목표 미달)
- Exp28-B2 linear probe (0.1852) < Exp38 C1 generation (0.1296): generation이 retrieval보다 낮음
  → Generation 단계가 추가 노이즈/오차를 유발하여 representation 이점을 상쇄
- 양쪽 mode 모두 ep99까지 학습 (early stop 발동 안함) → VI 데이터 부족 또는 도메인 갭이 큼

**현황 정리**: 최소 성공 기준은 충족했지만 의미있는 VI generation 달성 못함.
근본 원인: VS→VI 도메인 갭 (자극 제시 vs 심상) + VI 데이터 소량 (432 train trials).

**Active**: 추가 실험 여부 결정 (human_directive.md 참조)
- C2 staged unfreeze 또는 S01/S24 추가 실험
- 또는 현 VI 데이터 규모에서의 generation이 구조적 한계임을 인정하고 마무리

**Next evaluation/training direction: multi-image class gallery**

Problem:
- Current DINO generation evaluation uses a 9-image retrieval gallery (`9 classes x 1 image`).
- This is brittle because the generated image is judged against one canonical template per class.
- It can over-penalize semantically correct but visually different generations, or over-reward template-like outputs.
- It also leaves room for the generator to learn a single-image shortcut rather than class-level visual structure.

**Experiment 39: Multi-Image Gallery Evaluation (2026-06-11) — COMPLETE**

Script: `eval_exp39_multigallery.py --subject_ids 18,1 --ckpt_dir checkpoints_vsre_latent_gen/20260610_105755_ch32_ep300_supcon_frozen_ca`
CSV: `checkpoints_vsre_latent_gen/20260610_105755.../exp39_multigallery_results.csv`

Gallery setup:
- single (9×1): original class images only
- dual (9×2): original + alternative image per class (both from `preproc_data_vi/images/`)
- augmented (9×8): 8 random augmented views of original per class

| Subject | Gallery | DINO@1 | DINO@3 | DINO@5 | dominant | entropy |
|---------|---------|--------|--------|--------|---------|---------|
| S18 | single | 0.1944 | 0.3796 | 0.6296 | 37.0% | 1.781 |
| S18 | **dual** | **0.2315** | **0.4444** | 0.6481 | 32.4% | 1.831 |
| S18 | augmented | **0.2315** | **0.4537** | 0.6111 | 31.5% | 1.795 |
| S01 | single | 0.1190 | 0.3651 | 0.5714 | 89.7% | 0.360 |
| S01 | dual | 0.1349 | 0.3095 | 0.5873 | 83.3% | 0.514 |
| S01 | augmented | **0.1508** | **0.3968** | **0.6746** | 84.9% | 0.626 |

**Key findings:**
1. S18 dual/augmented = **0.2315 > 0.20 목표 달성** — single gallery(0.1944)에서는 미달이었지만 확장 갤러리에서 달성
2. S01 augmented = 0.1508 (single 0.1190 대비 +27%) — 확장 갤러리가 S01에도 더 공정
3. **이전 Exp37 보고 DINO@1=0.4200은 인플레이션**: stochastic DDIM × 50-sample 평가 때문. 실제 full-test 108 samples 기준 0.1944 (single gallery)
4. 확장 갤러리가 생성 품질을 더 공정하게 측정함 — dual/augmented 사용 권장

**Experiment 40: Multi-Image Target Training (2026-06-11) — COMPLETE**

Script: `train_vs_re_latent_gen.py --subject_ids 1,18 --epochs 300 --supcon_ckpt ... --freeze_encoder --use_cross_attn --multi_img`
Checkpoint: `checkpoints_vsre_latent_gen/20260611_123844_ch32_ep300_supcon_frozen_ca_mi`
Setup: 2 images/class (original + alternative), randomly sampled per training trial

**Full-test multi-gallery comparison (Exp39 protocol):**

| | Exp37 CA single-img | Exp40 CA multi-img | Δ |
|--|---------------------|---------------------|---|
| S18 single-gallery | 0.1944 | 0.1574 | −0.037 |
| S18 dual-gallery | **0.2315** | 0.1667 | −0.065 |
| S01 single-gallery | 0.1190 | 0.1032 | −0.016 |
| S01 augmented-gallery | 0.1508 | 0.1270 | −0.024 |

**결론**: Multi-image target이 오히려 성능 저하.
원인: 동일 클래스에 시각적으로 다른 두 이미지를 번갈아 target으로 사용 → UNet 학습 신호가 노이즈 증가 → EEG conditioning이 특정 class visual로 수렴하기 어려움. EEG representation이 bottleneck인 상황에서 target 다양화는 역효과.

**→ 결론: 현재 아키텍처에서 best config = Exp37 (CA + frozen SupCon + single-img target)**

**Experiment 41: SD 1.5 LoRA Generator (2026-06-11) — COMPLETE**

Script: `train_vs_re_lora_gen.py --subject_ids 18 --epochs 100 --lora_r 16 --batch_size 4`
Checkpoint: `checkpoints_vsre_lora_gen/20260611_144712_lora_r16_ep100`
Architecture: SD 1.5 UNet (862M, frozen) + LoRA (3.2M, 0.37%) + EEG projection (512→8×768 tokens)
Training target: 512×512 class images → 64×64×4 SD VAE latents (vs 16×16×4 in Exp37)

| Metric | Exp37 CA (dual gallery) | **Exp41 LoRA (dual gallery)** | Δ |
|--------|------------------------|-------------------------------|---|
| DINO@1 | 0.2315 | **0.2870** | **+0.0555 (+24%)** |
| DINO@3 | 0.4444 | **0.4815** | +0.037 |
| entropy | 1.831 | **2.072** | +0.241 (더 다양) |
| dominant | 32.4% | **23.1%** | 더 낮은 붕괴 |
| Pred dist | {3:35%} biased | All 9 classes represented | |

**Visual quality (gen_image/exp41_lora/):**
- Airplane: 사실적 배경 포함 선명한 비행기 ✅
- Cup: 매우 선명한 컵 ✅
- Tree: 고품질 나무 ✅
- Digit 1/5: 숫자 형태 선명 ✅
- Star: 명확한 별 형태 생성 ✅ (Exp37에서는 불가)
- Triangle: 삼각형 명확 ✅
- Heart: 여전히 cup으로 혼동 ❌

**결론**: SD 1.5 LoRA가 custom latent UNet 대비 현저히 우수.
- 64×64 latent (고해상도 VAE space) + SD 1.5 prior가 핵심
- LoRA 0.37% params만 학습으로 +24% DINO@1 향상
- 클래스 다양성 크게 개선 (dominant 32%→23%)
- **새로운 프로젝트 best: DINO@1=0.2870 (dual gallery, full test)**

**프로젝트 Final 성과 요약 (Exp39 평가 기준 적용):**

| Method | S18 DINO@1 (dual gallery) | 비고 |
|--------|--------------------------|------|
| Pixel InfoNCE (Exp13) | ~0.11 | collapse |
| VAE latent FiLM frozen (Exp33) | 0.2315 | best custom UNet |
| VAE latent CA frozen (Exp37) | 0.2315 | = Exp33 |
| Exp40 multi-img | 0.1667 | worse |
| **Exp41 SD1.5 LoRA** | **0.2870** | **new project best** |
| Stage 2 retrieval (Exp25) | 0.3333 | non-gen ref |

**Experiment 42-A: SD 1.5 LoRA — S01 VS Generation (2026-06-12)**

Script: `train_vs_re_lora_gen.py --subject_ids 1 --epochs 100 --lora_r 16 --n_eeg_tokens 8`
Checkpoint: `checkpoints_vsre_lora_gen/20260611_213303_lora_r16_ep100/subj01_lora_best.pt`
50-sample eval: DINO@1=0.4643, entropy=1.808, dominant=37.5%

**Full-test dual gallery (n=126):**

| Gallery | DINO@1 | DINO@3 | DINO@5 | dominant | entropy |
|---------|--------|--------|--------|---------|---------|
| single | **0.3571** | 0.5476 | 0.7778 | 23.8% | 2.047 |
| **dual** | **0.3333** | **0.5873** | 0.7460 | 25.4% | 2.005 |

**vs 이전 S01 결과:**
- Exp37 CA dual: 0.1349 → **0.3333 (+147%)**
- Exp41 S18 LoRA dual: 0.2870 — S01이 S18보다 높음!

**결론**: SD 1.5 prior가 S01의 약한 EEG class separability (inter-class sim=0.75)를 대폭 보완. dominant 25%, entropy 2.0 — 매우 다양한 생성 달성.

**결정 (directive)**: S01 ≥ 0.20 ✅ → **Exp42-B (S18 최적화) + Exp43 (VI fine-tuning) 진행**

**Exp42-B Step 4 Complete — LoRA Rank Ablation (2026-06-17):**

| rank | S18 dual | S01 dual | 비고 |
|------|---------|---------|------|
| r=8, 16tok | 0.2037 | 0.3333 | S18 크게 하락 |
| r=16, 16tok (Step2) | **0.2963** | 0.3333 | S18 best |
| **r=32, 16tok** | 0.2870 | **0.3571** | **S01 new best** |
| Stage 2 retrieval | — | 0.3333 | non-gen ref |

**S18 최적: r=16 / S01 최적: r=32 (주체별 상이)**
**S01 r=32 = 0.3571 — 최초로 retrieval baseline(0.3333) 초과 달성 (fully generative!)**

**Updated Project Best (2026-06-17):**

| Method | S18 dual | S01 dual | 비고 |
|--------|---------|---------|------|
| Exp41 SD1.5 LoRA | 0.2870 | — | S18 baseline |
| Exp42-A SD1.5 LoRA | — | 0.3333 | S01 = retrieval |
| Step2 r=16, 16tok | **0.2963** | 0.3333 | S18 best |
| **Step4 r=32, 16tok** | — | **0.3571** | **프로젝트 최고 (retrieval 초과)** |
| Stage 2 retrieval | 0.3333 | — | non-gen ref |

**Exp42-B Step 5 — S18 augmentation 완료 (2026-06-17)**

Config: r=16, 16tok + class-preserving aug / Checkpoint: `20260617_124018_lora_r16_ep100`
Result: DINO@1=0.2870, entropy=2.120, dominant=14.8%
vs Step 2 no-aug: −0.009 DINO@1 but dominant 20.4%→14.8% (더 다양)
→ augmentation이 diversity 개선, DINO@1은 소폭 하락. S18 best config = Step 2 (r=16, 16tok, no aug)

**Exp42-B 전체 S18 결과:**
Step2 r=16 16tok: **0.2963** ← best / Step3 deepMLP: 0.2685 / Step4 r=8: 0.2037 / Step4 r=32: 0.2870 / Step5 aug: 0.2870

**남은 실험 (미실행, 사용자 요청으로 일시 중단):**
- Step 6: staged encoder unfreeze (S18 r=16 / S01 r=32)
- S01 Step 5 aug (r=32, 16tok)
- Exp43: VI fine-tuning (SD LoRA C0/C1/C2)

**Exp42-B Best Checkpoints (재시작 시 사용):**
- S18: `checkpoints_vsre_lora_gen/20260612_010728_lora_r16_ep100` (r=16, 16tok)
- S01: `checkpoints_vsre_lora_gen/20260617_074245_lora_r32_ep100` (r=32, 16tok)

**Active after Exp42-A**:
- Exp42-A is a major turning point: S01 dual-gallery DINO@1 `0.3333`, matching the Stage 2 retrieval reference while remaining generative.
- The prior assumption that S01 is necessarily worse than S18 under generation is no longer valid under SD LoRA.
- SD prior appears to compensate for weak EEG class separability more strongly than expected.

Immediate active work:
1. Exp42-B Step 2 — EEG tokens `8 -> 16` (CFG unconditional path does not exist, so Step 1 is skipped).
2. Track both S18 and S01 when feasible; S01 is now the project-best generative subject, not only a weak-signal control.
3. Prepare S24 SD LoRA with the same setup as Exp42-A to test session-count effect.
4. Prepare Exp43 C0/C1 scripts and VI loader checks in parallel.

Recommended order:
1. `Exp42-B`: SD LoRA optimization with S01/S18 tracking
   - Step 1: skipped because no real CFG/unconditional path exists.
   - Step 2: EEG tokens `8 -> 16` as a single-variable ablation.
   - Step 3: MLP EEG projection with the best token count fixed.
   - Step 4: LoRA rank `8/32` using the best projection/token setup.
   - Step 5: class-preserving image augmentation only; avoid unsafe flips/large rotations for digits/symbols.
   - Step 6: staged encoder unfreeze last, with very low encoder LR.
   - Minimum success: dual-gallery DINO@1 `>=0.30`; target `>=0.32`.
   - Exp43 can proceed even if Step 2~4 do not improve, because Exp42-A already reached retrieval-level generation.
2. `S24 SD LoRA VS generation`
   - Same setup as Exp42-A.
   - Purpose: separate session-count benefit from intrinsic EEG signal quality and provide another VS/VI-overlap VI candidate.
3. `Exp43`: SD LoRA-based VS→VI fine-tuning
   - `C0`: VI scratch LoRA.
   - `C1`: VS LoRA -> VI fine-tune with frozen EEG encoder.
   - `C2`: staged encoder unfreeze.
   - Minimum success: `C1 > C0`; meaningful target: VI DINO@1 `> 0.20`.
   - C2 trigger: if C1 shows no validation improvement by about epoch `50`, start C2 in parallel.

Likely causes:
1. SD VAE latent resolution is too low at 128x128 input.
   - 128x128 images become `4 x 16 x 16` latents.
   - fine symbolic shapes are hard to preserve at this spatial resolution.
2. EEG conditioning remains weak.
   - repeated foggy average texture indicates the UNet often falls back to an average image prior.
   - Exp35 showed subject-dependent EEG class separability is a strong limiter.
3. Current image prior is still too weak.
   - the latent UNet is trained from scratch on very small EEG/image data.
   - Exp33/37/40 converge near a similar full-test dual-gallery ceiling around S18 `0.23`.

**Direction decision after Exp40**

Current best architecture:
- frozen SupCon encoder
- SD VAE latent space
- cross-attention latent UNet
- single-image target

But the current architecture appears near its practical ceiling:
- Exp33/37/40 do not produce a major full-test improvement beyond the `0.23` dual-gallery range for S18.
- Exp40 multi-image target training degraded performance.
- Further small changes to the scratch latent UNet are unlikely to solve foggy texture and weak symbolic shapes.

Recommended next structural step:
- `Exp42-A = SD LoRA S01 VS generation`
- `Exp42-B = SD LoRA S18 optimization`
- `Exp43 = SD LoRA-based S18 VS→VI fine-tuning`
- Motivation:
  - Exp41 already confirmed that SD 1.5 LoRA is superior to the custom latent UNet.
  - S01 validation is needed to separate SD-prior benefit from S18-specific EEG quality.
  - S24 should be added after S01 because it has the largest VS data volume and can separate session-count benefit from intrinsic EEG quality.
  - LoRA rank / EEG projection / valid CFG sweeps are the highest-yield SD LoRA improvements.
  - VI fine-tuning should now use SD LoRA, not the older CA-UNet path.

Lower-priority checks:
- CFG/guidance sweep only if a real unconditional path exists.
- 256x256 training may improve spatial detail but will not fix weak conditioning alone.
- recolor/rework symbolic black-background targets only as a dataset-design ablation.

**Gen-WS-03 / Exp38 design: within-subject VS→VI generation fine-tuning**

Rationale:
- The VS generation bottleneck is now partially solved.
- Exp37 initially looked strong in stochastic/sample-grid evaluation, but Exp39 full-test evaluation corrected this estimate.
- Current reliable S18 VS generation reference is Exp39 dual-gallery DINO@1 `0.2315`.
- The next target is the actual research path: same-subject VS pretraining → same-subject VI fine-tuning → VI image generation.
- Prior VS→VI diagnostics:
  - Exp25-VI zero-shot Top-1 `0.130`: VS encoder barely works on VI without adaptation
  - Exp28-B2 frozen VS encoder + VI linear probe Top-1 `0.154`: best VI transfer diagnostic so far
  - Exp28-B1 full fine-tune Top-1 `0.117`: naive fine-tuning can degrade VS representation
  - Exp39 full-test S18 dual-gallery DINO@1 `0.2315`: reliable VS generation reference after correcting inflated Exp37 sample-grid estimates
- Core unknown: whether a VS-pretrained generator can produce meaningful images from VI EEG.
- Start with S18 because it has the strongest VS generation result:
  - VS Exp39 S18 dual-gallery DINO@1 `0.2315`
  - entropy `1.832`
  - dominant `32%`
  - inter-class similarity `0.59`, indicating usable EEG class separation
- S01 is a secondary diagnostic subject because Exp35 confirmed high EEG inter-class similarity (`0.75`) and persistent collapse.

Exp38 target subject order:
1. `S18` primary
2. `S01` diagnostic/control
3. `S24` or `S09` after the first VI result, because Exp34 high-session results were promising

Required comparison:
- `C0 = VI scratch`
  - random-init VAE-latent cross-attention generator trained only on VI EEG
  - mandatory baseline
- `C1 = VS pretrained → VI fine-tune, frozen encoder`
  - initialize from same-subject VS Exp37-style checkpoint
  - keep SupCon encoder frozen
- `C2 = VS pretrained → VI fine-tune, staged unfreeze`
  - same as C1 for first 50 epochs
  - then unfreeze encoder with smaller LR
  - equivalent to full fine-tuning, but staged to avoid the Exp28-B1 failure mode
  - run only after C1 is available or if compute allows

Training rule:
- VI has about `60 trials/class`, train about `432` trials/subject.
- Do not blindly reuse VS `300` epochs.
- Start with `50–100` epochs plus early stopping.
- Use identical VI train/val/test split for C0/C1/C2.

Success criteria:
- minimum: `C1 > C0`
- meaningful: VI DINO@1 `> 0.20`
- target: VI DINO@1 `> 0.20` with lower collapse than VI scratch
- minimum S18 target threshold: `C1 > C0` plus reduced dominant-class collapse

Risk:
- Exp28-B showed full fine-tuning can degrade frozen VS representations.
- Therefore C1 frozen-encoder fine-tuning is the first main condition, and C2 unfreeze is secondary.

**RESUME INSTRUCTIONS** (after restart):
1. Exp42-B Step4 completed: S01 SD LoRA r=32, 16 tokens reached dual-gallery DINO@1=0.3571, exceeding Stage 2 retrieval 0.3333.
2. Next priority: run S24 SD LoRA VS generation with r=16 and r=32 if compute allows.
3. Start Exp43 SD LoRA-based VS→VI fine-tuning in parallel:
   - S01 init: Exp42-B Step4 r=32, 16-token checkpoint.
   - S18 init: Exp42-B Step2 r=16, 16-token checkpoint.
4. Exp42-B Step5 augmentation is secondary; run only after S24/Exp43 are underway.
---

## Immediate Plan: `preproc_vs_re` Subject-wise Generation

**Legacy note**: this section is older than Exp32–37. It is kept for history only.
The current active direction is Exp38 Gen-WS-03: within-subject VS-pretrained VAE-latent cross-attention generation fine-tuned on VI.
Any instruction below that says not to use latent diffusion is superseded by Exp32–37 results.

Current decision for the new repeated-session VS dataset:

- Use **`0~2 sec` only** for the main model
- Start with **subject-wise training first**
- Use **32 EEG channels only** as the default input
- Do **not** use class-label conditioning

### Why `0~2 sec` only

For `preproc_vs_re`, the actual stimulus period is only `0~2 sec`.
Therefore:

- `-1~0 sec` is pre-stimulus baseline
- `2~3 sec` is post-stimulus non-stimulus activity
- These should not be included in the main generation input

Main input rule:

```python
data = data[:, :, :, 1024:3072, :]   # 0~2 sec only
data = data[..., :32]                # 32 EEG channels only
```

### Subject-wise first strategy

Before moving to cross-subject generation, establish a stable per-subject baseline:

1. Build an `h5py` loader for `preproc_vs_re`
2. Train subject-wise retrieval/alignment baseline
3. Train subject-wise EEG-only generation baseline
4. Compare session handling strategies within each subject

This keeps the first generation result interpretable and avoids mixing
subject variability with the new data-format/session issue.

### Recommended subject-wise pipeline

#### Step 1. Loader

Target output:

```text
(trial, ch, time), labels
```

Recommended default conversion:

```python
# original: (5, 3, 9, 4096, 40)
data = data[:, :, :, 1024:3072, :]       # 0~2 sec
data = data[..., :32]                    # EEG-only
data = data.transpose(2, 4, 3, 0, 1)     # (9, 32, 2048, 5, 3)
data = data.reshape(9, 32, 2048, -1)     # class-major trials
```

#### Step 2. Subject-wise alignment baseline

Before generation, verify that the new data still supports stimulus identity:

```text
EEG (one subject, all sessions)
  -> EEG encoder
  -> latent
  -> DINO image feature alignment
```

Metrics:

- top-1 retrieval
- top-3 retrieval
- per-subject confusion matrix

#### Step 3. Subject-wise generation baseline

After alignment works, move to generation:

```text
EEG (0~2 sec, 32ch, one subject)
  -> EEG encoder
  -> EEG-only conditioning
  -> diffusion / decoder
  -> image
```

Recommended baseline:

- reuse the current EEG-only generation structure
- replace only the dataset loader first
- keep the experiment strictly subject-wise

### Session handling order

Because `preproc_vs_re` has large session imbalance, use this order:

1. **Session-merged baseline**
   - merge all sessions within a subject
   - simplest first baseline
2. **Session-capped baseline**
   - cap each subject to the same number of sessions/trials
   - use if merged training shows imbalance effects
3. **Session embedding**
   - only if session variation clearly hurts performance

### Concrete next experiments

1. Implement `preproc_vs_re` loader with `0~2 sec`, `32 EEG only`
2. Run subject-wise retrieval baseline on the new dataset
3. Run subject-wise EEG-only generation baseline
4. Compare `session-merged` vs `session-capped`
5. Optional later ablation: `32 EEG only` vs `40 channels`

### What not to do yet

- Do not use `-1~0 sec` or `2~3 sec` in the main generation model
- Do not add GT class label as conditioning
- Do not start from cross-subject generation first
- Do not move to latent diffusion before subject-wise baseline is stable
EEG (B, ch, T)
  └─ Conv1D + TransformerEncoder → cond_emb (B, 256)
       └─ x1.5 scaling (no class label -- EEG-only)
UNet128 (DownBlock → FiLMResBlock → UpBlock)
  └─ FiLM conditioning with cond_emb
Losses: noise MSE + L1 + SSIM + Perceptual (ResNet18 cosine)
```

**Key fixes applied**:
- `pos_embed` bug: was randomly re-initialized every forward pass → replaced with fixed sinusoidal encoding
- `percept_backbone` BN drift: ResNet18 BatchNorm switched to train mode during `model.train()` → overrode `train()` to keep backbone in eval
- Class label removed from conditioning: was using GT label at inference (leakage) → now EEG-only
- DDIM `labels` argument was missing → fixed in all sampling scripts

### 2. Classification-only Model (EEG → Class)
**File**: `model_eeg_transformer_cls_only.py`

```
EEG (B, ch, T)
  └─ MultiScaleStem (Conv1D with k=7,15,31) → concat → 256-dim
  └─ CLS token + Sinusoidal PE
  └─ TransformerBlock x4 (8 heads, StochasticDepth)
  └─ ClassificationHead → logits (B, num_classes)
Loss: CrossEntropy (label smoothing=0.1) + Mixup
```

### 3. Joint Generation + Classification Model
**File**: `model_128_eegonly_transformer_repa_cls.py`

```
EEG (B, ch, T)
  └─ EEGEncoderTransformer → cond_emb
       ├─ UNet128 → generated image
       └─ ClassificationHead → logits
Loss: noise MSE + L1 + SSIM + Perceptual + CrossEntropy
```

### 4. Stage 1: EEG-DINO Alignment Model (Current Focus)
**File**: `model_eeg_dino.py`

```
EEG (B, ch, T)
  └─ EEG Encoder (choose via --encoder_type):
       ├─ "transformer": Conv1D + TransformerEncoder (4L, 4H) → 256-dim
       └─ "conv" (EEGNet-inspired):
            Multi-scale temporal conv (k=16/64/256 @ 1024Hz) → 3x32 filters
            → Depthwise separable conv → AdaptiveAvgPool → 256-dim
Subject ID
  └─ Embedding(n_subjects, subj_emb_dim) → 64-dim (default)
Fusion MLP: (256+64=320) → 512-dim → L2 norm  =  eeg_latent

Image
  └─ DINOv2 ViT-S/14 (frozen, dim=384)
  └─ Image Projector: 384 → 512-dim → L2 norm  =  img_latent

Shared latent space: 512-dim

Losses:
  1. Cosine alignment (w_cos):  1 - mean(eeg_latent * img_latent)
  2. Prototypical cross-entropy (w_proto): EEG vs class prototypes / temperature
  3. Symmetric InfoNCE (w_infonce, optional): EEG↔Image, same-class false-negative exclusion
  4. Auxiliary classification head (w_aux=0.5): eeg_latent → n_classes CE loss
     └─ nn.Linear(512, 9) -- explicit class discrimination in latent space
```

**Encoder comparison** (`--encoder_type transformer | conv`):
- Transformer: better long-range temporal context; slower
- Conv (EEGNet): multi-scale temporal receptive fields (15ms/62ms/250ms); faster; fewer params

---

## Training Scripts

| Script | Description | Status |
|--------|-------------|--------|
| `train_vs_repa_allclass.py` | VS train, generative model (pixel diffusion) | Done |
| `train_vs_test_vi.py` | VS train → VI EEG test | - |
| `train_vs_test_rest.py` | VS train → REST EEG test | - |
| `train_vs_cls_only.py` | EEG classification only | Done |
| `train_vs_repa_cls.py` | Joint generation + classification | Done |
| `train_crosssubj_dino.py` | Cross-subject DINO alignment (Stage 1) | In progress |
| `train_vs_re_dino.py` | preproc_vs_re subject-wise DINO alignment | Done (see Exp 4) |
| `train_vs_re_gen.py` | preproc_vs_re subject-wise pixel diffusion generation | Suspended before VI transfer check |
| `dataset_vs_re.py` | h5py loader for preproc_vs_re | Done |

---

## Experiments and Results

### Experiment 1: EEG Classification (train_vs_cls_only.py)

- Subject-specific models, 20 subjects, 3-class groups (g1: cls1-3, g2: cls4-6, g3: cls7-9)
- Random baseline: 0.333 (3-class)

| | g1 | g2 | g3 | avg |
|--|--|--|--|--|
| Mean (20 subjects) | 0.375 | 0.346 | 0.371 | **0.364** |

Notable: S07 (0.556), S12 (0.583) best; S13 (0.167) worst.

### Experiment 2: Image Retrieval via Generated Images (compare_cls_ab.py)

Two approaches compared:
- **Method A**: EEG classification model (direct)
- **Method B**: Generated image → ResNet18 feature → nearest class image

| Method | g1 | g2 | g3 | avg |
|--------|--|--|--|--|
| A (EEG classifier) | 0.371 | 0.329 | 0.396 | 0.365 |
| B (generated image similarity) | **0.575** | 0.346 | 0.358 | **0.426** |
| Random baseline | 0.333 | 0.333 | 0.333 | 0.333 |

B outperforms A by 6.1%p overall, especially in g1 (+20%p).

> Note: Generated images were produced with the pos_embed bug present. Results expected to improve after retraining.

### Experiment 3: Stage 1 -- EEG-DINO Alignment (train_crosssubj_dino.py)

**Setup**: 20 subjects, 9-class, cross-subject training, DINOv2 ViT-S/14

**First attempt (3-class groups)** -- Failed:
- Loss stuck at ln(3) = 1.0986 throughout training
- Gallery only 3 images → trivial top-3/5 = 1.0
- Root cause: too few negatives, weak ranking pressure

**Second attempt (9-class, with fixes)**:
- Added warmup LR scheduler (10% warmup + cosine decay)
- Temperature changed: 0.07 → 0.1 (prevent gradient vanishing)
- Gallery = 9 class images → top-1/3/5 all meaningful

| Mode | Top-1 | Top-3 | Top-5 | Random |
|------|-------|-------|-------|--------|
| Within-subject | 0.1417 | 0.4236 | 0.6472 | 0.1111 |
| LOSO | 0.1306 | 0.4014 | 0.6111 | 0.1111 |

**Third attempt (with EEG augmentation + Supervised InfoNCE)**:
- EEG augmentation: Gaussian noise, amplitude scaling, channel dropout, time shift, freq-domain noise
- InfoNCE: symmetric (EEG→Image + Image→EEG), self-exclusion masking
- Result: In progress

---

## EEG Augmentation

Applied during training (on-the-fly, no data size increase):

| Type | Probability | Effect |
|------|-------------|--------|
| Gaussian noise | 50% | SNR degradation robustness |
| Amplitude scaling (0.8~1.2x) | 50% | Inter-subject amplitude variation |
| Channel dropout (10% per ch) | 30% | Electrode artifact simulation |
| Time shift (+/-25 samples, ~50ms) | 30% | Response latency individual difference |
| Frequency-domain noise | 30% | Spectral perturbation |

---

## Key Bugs Fixed

| Bug | Impact | Fix |
|-----|--------|-----|
| `pos_embed` random re-init in `forward()` | EEG encoder output garbage at inference | Fixed sinusoidal (no parameter needed) |
| `percept_backbone` BN drift | Perceptual loss target drifts during training | Overrode `train()` to keep backbone eval |
| Class label leakage at inference | GT label used → not true EEG-only | Removed class conditioning entirely |
| DDIM missing `labels` arg | Different condition from training | Added `labels=label_group` in all samplers |
| 3-class InfoNCE loss stuck at ln(3) | No learning signal | Switched to 9-class, added warmup LR |
| InfoNCE false negatives | Same-class samples treated as negatives | Symmetric InfoNCE with self-exclusion |
| `p_losses()` missing `t` arg in `train_vs_re_gen.py` | RuntimeError at training start | Added `t = torch.randint(...)` before call |
| `sample_ddim(steps=...)` wrong kwarg | TypeError at sampling | Changed to `sample_ddim(num_steps=...)` |
| `out_csv` NameError in `main()` | Script crash before CSV save | `train_subject()` returns dict; `main()` saves CSV |
| FFT augmentation on CUDA (nvrtc error) | `nvrtc: failed to open nvrtc-builtins64_121.dll` | Moved FFT ops to CPU in `train_crosssubj_dino.py` |

---

## Experiment 4: preproc_vs_re Subject-wise DINO Alignment (2026-04-11)

**Script**: `train_vs_re_dino.py`
**Data**: `preproc_vs_re/`, 18 subjects, session-merged, 32ch, 0~2 sec, 9-class
**Model**: EEGDINORegressor (subj_emb_dim=32, single-subject → embedding index 0)
**Training**: 200 epochs, warmup+cosine LR, InfoNCE + cosine + proto loss, aug (no freq)
**Checkpoint**: `checkpoints_vsre_dino/20260411_152315_ch32_merged_ep200/`

### Per-subject Results

| Subject | Top-1 | Top-3 | Top-5 | Sessions | Trials |
|---------|-------|-------|-------|----------|--------|
| S01 | 0.2315 | 0.6667 | 0.8704 | 8 | 1080 |
| S02 | **0.2778** | 0.5278 | 0.7639 | 5 | 675 |
| S03 | 0.0556 | 0.3333 | 0.6667 | 1 | 135 |
| S04 | 0.2222 | 0.4444 | 0.6296 | 2 | 270 |
| S05 | 0.2222 | 0.4444 | 0.5000 | 1 | 135 |
| S09 | 0.1111 | 0.3704 | 0.6296 | 2 | 270 |
| S10 | 0.1852 | 0.4815 | 0.5556 | 2 | 270 |
| S11 | 0.1111 | 0.5000 | 0.7222 | 1 | 135 |
| S16 | 0.1111 | 0.5556 | 0.8333 | 1 | 135 |
| S18 | 0.1574 | 0.4444 | 0.7315 | 8 | 1080 |
| S19 | 0.0000 | 0.2778 | 0.3889 | 1 | 135 |
| S20 | 0.1111 | 0.2778 | 0.3889 | 1 | 135 |
| S21 | 0.1111 | 0.3333 | 0.6111 | 1 | 135 |
| S23 | 0.0000 | 0.2778 | 0.5556 | 1 | 135 |
| S24 | 0.1746 | 0.5317 | 0.7460 | 9 | 1215 |
| S28 | 0.1481 | 0.4630 | 0.7593 | 4 | 540 |
| S29 | 0.1528 | 0.4861 | 0.7222 | 5 | 675 |
| S35 | 0.1111 | 0.3333 | 0.5556 | 1 | 135 |
| **Mean** | **0.1386** | **0.4305** | **0.6461** | | |
| Random | 0.1111 | 0.3333 | 0.5556 | | |

### Summary

- Top-1 mean: 0.1386 (+0.0275 above random)
- Top-3 mean: 0.4305 (+0.0972 above random)
- Top-5 mean: 0.6461 (+0.0905 above random)
- Best: S02 (0.2778, 5 sessions), S01 (0.2315, 8 sessions)
- Worst: S19 (0.0000), S23 (0.0000) -- both 1-session subjects

### Observations

- **Session count matters**: Top-3 subjects (S02, S01, S04) all have >=2 sessions
- **1-session subjects are weak**: S19, S23 score 0.0 Top-1 → data too limited for subject-wise model
- **Comparison with preproc_for_gan_vs cross-subject LOSO (Top-1=0.1306)**: comparable -- new data slightly better on per-subject basis despite fewer trials per class

---

## Enhancements (2026-04-14)

### 1. EEGNet-inspired Conv Encoder (`model_eeg_dino.py`)

Added `EEGEncoderConv` as an alternative EEG encoder:
- Multi-scale temporal conv: 3 parallel branches with kernel sizes 16/64/256 (~15ms/62ms/250ms @ 1024Hz)
- Each branch: Conv1d → BatchNorm1d → ELU → AvgPool1d(8) → Dropout
- Concat (96 filters) → depthwise + pointwise separable conv → AdaptiveAvgPool → Linear → LayerNorm → 256-dim
- Fewer parameters than transformer; better inductive bias for temporal EEG patterns

**Usage**: `python train_vs_re_dino.py --encoder_type conv`

### 2. Auxiliary Classification Head (`model_eeg_dino.py`)

Added `self.aux_cls_head = nn.Linear(latent_dim, n_classes)` in `EEGDINORegressor`:
- Parallel CE loss on EEG latent during training (bypasses prototype matching bottleneck)
- Weight: `w_aux=0.5` (configurable via `--w_aux`)
- Strengthens class discrimination in the shared latent space
- Does NOT affect inference (`predict()` still uses prototypes)

### 3. LPIPS Perceptual Metric (`train_vs_re_gen.py`)

Added LPIPS (Learned Perceptual Image Patch Similarity) to `evaluate_generation()`:
- VGG-based perceptual distance, lazy-initialized on first call
- Graceful fallback: if `lpips` package not installed, returns `None` (no crash)
- Install: `pip install lpips`
- Reported in test metrics and included in CSV output (`lpips` column)
- Lower LPIPS = perceptually more similar to target

### 4. Exposed Hyperparameter Args

`train_vs_re_dino.py`:
- `--encoder_type` (`transformer` | `conv`)
- `--w_aux` (default 0.5)

`train_vs_re_gen.py`:
- `--sample_ddim_steps` (DDIM steps for training-time visualization, default 50)
- `--eval_ddim_steps` (DDIM steps for final test evaluation, default 50)
- `--guidance_scale` (default 1.5)
- `--eta` (DDIM stochasticity; 0.0 = deterministic, default 0.0)
- `--beta_schedule` (`linear` | `cosine`, default `cosine`)
- `--lambda_lpips` (LPIPS training loss weight, default 0.05)
- `--lambda_percept` default raised 0.05 → 0.10
- `--lambda_rec` default lowered 0.02 → 0.01

### 5. Noise Schedule: Linear → Cosine (`model_128_eegonly_transformer.py`)

Added `_cosine_beta_schedule()` implementing IDDPM cosine schedule:
- `alpha_bar(t) = cos^2(((t/T + s)/(1+s)) * pi/2)`, s=0.008
- Betas clipped to [1e-4, 0.9999]
- Activated via `beta_schedule="cosine"` (now the default in train script)
- Better high-frequency detail preservation than linear at low-noise timesteps

### 6. LPIPS Training Loss (`model_128_eegonly_transformer_repa.py`)

Added `lambda_lpips` parameter to `EEGDiffusionModel128` (REPA subclass):
- VGG-based perceptual distance, frozen, eval-mode locked
- Graceful fallback: warning if `lpips` package not installed (`pip install lpips`)
- `train()` override keeps both `percept_backbone` and `lpips_fn` in eval mode
- Default weight: 0.05 (via CLI arg)
- Combined loss: `noise_mse + lam_rec*L1 + lam_ssim*(1-SSIM) + lam_percept*(1-cos) + lam_lpips*LPIPS`

### Loss Rebalancing Rationale

| Loss | Old default | New default | Reason |
|------|-------------|-------------|--------|
| `lambda_rec` (L1) | 0.02 | 0.01 | Pixel averaging causes blur; reduce weight |
| `lambda_percept` (ResNet18) | 0.05 | 0.10 | Perceptual term directly improves sharpness |
| `lambda_lpips` (VGG) | -- | 0.05 | Second perceptual anchor; complementary to ResNet18 |
| `lambda_ssim` | 0.05 | 0.05 | Unchanged |

### DDIM / Guidance Tuning Rationale

- `guidance_scale`: recommended sweep 1.0 → 1.5 → 2.0 (not 5~8; current non-CFG structure degrades past 3.0)
- `eta=0.0`: deterministic DDIM -- no stochasticity, same EEG → same sample (good for evaluation consistency)
- DDIM steps: 50 vs 100 sweep (50 is usually sufficient with cosine schedule)

---

## Experiment 6: Generation -- Cosine Schedule Only / Step A (2026-04-15)

**Script**: `train_vs_re_gen.py`
**Config**: `beta_schedule=cosine, lambda_lpips=0.0, lambda_percept=0.10, lambda_rec=0.01, eta=0.0, eval_ddim_steps=50, guidance_scale=1.5, epochs=300`
**Checkpoint**: `checkpoints_vsre_gen/20260415_144727_ch32_merged_ep300/`
**Log**: `logs/vsre_gen_stepA_cosine_only.log`

### Mean Results (전체 비교)

| Run | beta_schedule | lambda_lpips | best_val_loss | L1 | SSIM | DINO Top-1 | DINO Top-3 | DINO Top-5 |
|-----|--------------|--------------|--------------|-----|------|------------|------------|------------|
| Baseline [20260411] | linear | 0.0 | **0.002779** | **0.8060** | **0.0562** | **0.1160** | **0.2973** | **0.5524** |
| guidance 3.0 [20260413] | linear | 0.0 | 0.002764 | 0.7950 | 0.0352 | 0.0965 | 0.3268 | 0.5411 |
| cosine + LPIPS 0.05 [20260414] | cosine | 0.05 | 0.009470 | 0.9033 | -0.0136 | 0.0816 | 0.3023 | 0.5474 |
| **Step A: cosine only [20260415]** | **cosine** | **0.0** | 0.005812 | 0.8927 | -0.0000 | 0.0813 | 0.3222 | 0.5498 |

### Per-Subject Detail (Step A)

- SSIM < 0: **13 / 18 subjects**
- best_val_loss > 0.005: **9 / 18 subjects**
- 최고 DINO Top-1: S28 = 0.1667 (전체적으로 불안정)

### Analysis

중단 기준(3개) 전부 평균 기준 충족:
- DINO Top-1 < 0.1160 (baseline): 0.0813 [X]
- SSIM < 0.0 평균: -0.0000 [X]
- best_val_loss > 0.005: 0.005812 [X]

**cosine schedule 단독도 현재 설정에서 baseline을 넘지 못함.**

#### 원인 추정
cosine schedule은 저잡음 구간(t가 작을 때) alpha_bar 감소가 완만해 디테일 보존에 유리하다.  
그러나 현재 모델은 `num_timesteps=200`으로 T가 이미 짧다.  
T=200에서 cosine schedule의 장점이 나타나지 않을 수 있음 -- cosine은 T=1000 이상일 때 효과가 두드러진다.

### 결론

- cosine schedule: **채택 안 함** (현재 T=200 설정에서)
- LPIPS 0.05 + cosine: **채택 안 함**
- **기존 baseline (linear, T=200)이 현재 최적**

---

## Experiment 5: Generation -- Cosine Schedule + LPIPS (2026-04-14)

**Script**: `train_vs_re_gen.py`
**Config**: `beta_schedule=cosine, lambda_lpips=0.05, lambda_percept=0.10, lambda_rec=0.01, eta=0.0, eval_ddim_steps=50, guidance_scale=1.5, epochs=300`
**Checkpoint**: `checkpoints_vsre_gen/20260414_100452_ch32_merged_ep300/`

### Mean Results (vs Prior Runs)

| Run | Config | best_val_loss | L1 | SSIM | DINO Top-1 | DINO Top-3 | DINO Top-5 |
|-----|--------|--------------|-----|------|------------|------------|------------|
| Baseline [20260411] | linear, no LPIPS, guidance 1.5 | 0.002779 | 0.8060 | **0.0562** | **0.1160** | **0.2973** | **0.5524** |
| DDIM200 + guidance 3.0 [20260413] | linear, no LPIPS, guidance 3.0 | **0.002764** | **0.7950** | 0.0352 | 0.0965 | 0.3268 | 0.5411 |
| Cosine + LPIPS 0.05 [20260414] | cosine, LPIPS 0.05, guidance 1.5 | 0.009470 | 0.9033 | -0.0136 | 0.0816 | 0.3023 | 0.5474 |

### Analysis

- **Cosine + LPIPS 0.05 조합은 baseline보다 전반적으로 나쁨**
- SSIM 평균이 음수: 생성 이미지가 target 구조를 거의 못 따라감
- DINO Top-1: 0.1160 → 0.0816 (악화)
- best_val_loss가 0.0095로 크게 올라 -- LPIPS가 loss landscape를 교란한 것으로 추정
- guidance 3.0도 SSIM 하락 + Top-1 하락 → guidance는 1.5가 현재 최적

### Root Cause

LPIPS(VGG)는 입력 해상도와 activation 분포에 민감하다.  
현재 구조 (EEG-only conditioning, 128x128, subject-wise 소규모 데이터) 에서는  
LPIPS를 0.05로 주면 noise MSE와 perceptual direction이 충돌 → 학습 불안정.

cosine schedule 단독 효과는 아직 검증 전 (LPIPS와 동시 적용이었으므로 분리 필요).

### 현재 기준 최적 설정

**기존 baseline [20260411] = 현재 최고**
- `beta_schedule=linear, lambda_lpips=0.0, lambda_percept=0.05 (또는 0.1), guidance=1.5, steps=200 DDIM`

---

## Experiments 11-12: DINO V2 Encoder on DINO Alignment Task (2026-04-22~27)

### Code Changes

**`model_eeg_dino.py`**
- Added `from model_128_eegonly_transformer import EEGEncoderV2`
- Updated `ENCODER_TYPES = ("transformer", "conv", "v2")`
- Added `eeg_occipital_indices` param to `EEGDINORegressor`
- V2 branch: `EEGEncoderV2(eeg_channels, eeg_hidden_dim, out_dim, n_heads, n_layers, dropout, occipital_indices)`

**`train_vs_re_dino.py`**
- Added `--encoder_type v2` to choices
- Added `--eeg_occipital_ids` arg (`auto` / `none` / comma-sep indices)
- Parses occipital_indices and passes to `EEGDINORegressor`

### Experiment 11 -- DINO V2 no-prior (Exp009, 2026-04-22)

**Command**: `train_vs_re_dino.py --encoder_type v2 --eeg_occipital_ids none --subject_ids 1,2,18 --epochs 200`
**Checkpoint**: `checkpoints_vsre_dino/20260422_213157_ch32_merged_ep200/`

| Subject | V1 Top-1 (Apr-11) | V2 no-prior Top-1 | Delta Top-1 | V2 Top-3 | V2 Top-5 |
|---------|-------------------|-------------------|---------|----------|----------|
| S01 | 0.2315 | **0.2778** | +20% | 0.5741 | 0.7685 |
| S02 | **0.2778** | 0.1667 | -40% | 0.4861 | 0.6667 |
| S18 | 0.1574 | **0.2407** | +53% | 0.5463 | 0.7222 |
| **Mean** | **0.2222** | **0.2284** | **+3%** | **0.5355** | **0.7191** |

Note: S18 loss went negative (< 0) after ep120 -- InfoNCE with very low temperature pulling hard negatives into negative range. Monitor in Exp010.

### Experiment 12 -- DINO V2 auto-prior (Exp010, 2026-04-27)

**Command**: `train_vs_re_dino.py --encoder_type v2 --eeg_occipital_ids auto --subject_ids 1,2,18 --epochs 200`
**Checkpoint**: `checkpoints_vsre_dino/20260427_095215_ch32_merged_ep200/`

| Subject | Exp009 no-prior | Exp010 auto-prior | Delta | Top-3 | Top-5 |
|---------|-----------------|-------------------|-------|-------|-------|
| S01 | 0.2778 | **0.3611** | +30% | 0.6019 | 0.8056 |
| S02 | 0.1667 | **0.2083** | +25% | 0.5278 | 0.7083 |
| S18 | 0.2407 | **0.2778** | +15% | 0.5648 | 0.8056 |
| **Mean** | 0.2284 | **0.2824** | **+24%** | **0.5648** | **0.7731** |

**Key finding**: Occipital prior helps ALL subjects on the DINO alignment task (+24% mean).
This is OPPOSITE to the generation probe result (Exp005 vs Exp006), where prior hurt S01/S02.
=> Prior effect is task-dependent: constraining gates helps representation learning but may over-constrain generation.

Note: S18 loss went negative after ep120 (InfoNCE temperature shrinks to 0.056) -- this is stable behavior, not divergence.

### Key Findings (DINO V2, updated)

1. V2+auto-prior is the best DINO config: mean Top-1=0.2824 (vs 0.2284 no-prior, +24%)
2. Occipital prior helps representation learning across all subjects (task-dependent effect)
3. Prior constrains unconstrained gate learning -- especially beneficial for S02 (only 5 sessions)
4. DINO task: V2+prior > V2-no-prior > V1 (all three configs converge with enough data)
5. Generation probe: V2-no-prior > V2+prior (prior over-constrains generation conditioning)

---

## Planned Next Steps

### Direction Change: Representation First (2026-04-22, per HUMAN_DIRECTIVE)

- Generation is now a **secondary probe** -- not the primary optimization target
- Main goal: **VS EEG encoder pretraining for VI transfer readiness**
- Priority order: DINOv2 alignment → CLIP image comparison → EEG inductive bias → SD 1.5 backbone
- open_clip installed in eegdiff env (ViT-B/32, 512-dim image embedding)
- CLIP comparison experiment planned next (Exp013)

---

### 실험 방향 전환 (2026-04-16)

sampling / schedule / LPIPS 튜닝 3개 실험 결과 요약:

| 시도 | 결과 | 결론 |
|------|------|------|
| guidance 3.0 | Top-1 하락 | [X] |
| cosine + LPIPS 0.05 | 전반 악화, SSIM 음수 | [X] |
| cosine only (Step A) | 전반 악화, 중단기준 충족 | [X] |

**sampling/schedule 방향은 일단 중단. 다음 우선순위는 EEG encoder 자체.**

기존 baseline은 `lambda_rec=0.02, lambda_percept=0.05, beta_schedule=linear`로 복원.

---

### 실험 방향 전환: EEG Encoder 강화 (2026-04-16~)

sampling/schedule 튜닝이 모두 baseline 미만으로 끝남에 따라,  
**EEG 조건 표현 자체를 강화하는 방향**으로 전환.

---

## EEG Encoder V2 구현 (2026-04-16)

### 구조 (`model_128_eegonly_transformer.py`)

```
EEG (B, 32, T)
  └─ z-score normalization (per trial)
  └─ OccipitalChannelGate            ← 학습 가능한 채널별 soft gate
       └─ gate: Parameter(32,), sigmoid, occipital prior 초기화 가능
  └─ MultiScaleStem (depthwise-separable, 3 branches)
       ├─ k=7  : ~7ms  (gamma / HFO)
       ├─ k=15 : ~15ms (beta)
       └─ k=31 : ~30ms (alpha)
       각 branch: DepthwiseConv1d → PointwiseConv1d → GroupNorm → SiLU → Dropout
       → concat (B, 96, T)  [for stem_filters=32]
  └─ Conv1d(96→256, stride=2)         ← 시간 다운샘플 + 채널 축소
     GroupNorm + SiLU
  └─ Sinusoidal PE + TransformerEncoder (기존과 동일)
  └─ MeanPool → LayerNorm → Linear → out_dim
```

### V1 vs V2 비교

| 항목 | V1 (기존) | V2 (강화) |
|------|-----------|-----------|
| Stem | Conv1d(32→64, k=7) + Conv1d(64→256, k=5, s=2) | DepthwiseSep x 3 branches → concat(96) → Conv1d(→256, s=2) |
| 채널 weighting | 없음 | OccipitalChannelGate (learnable) |
| 시간 수용 범위 | ~5ms + ~5ms | 7ms / 15ms / 30ms (명시적 다중 스케일) |
| Transformer | 동일 (2L, 4H) | 동일 |
| 파라미터 수 | 소 | 중 (~+0.1M) |

### OccipitalChannelGate 설계 결정

BioSemi 32-ch 채널 순서 (0-indexed, `biosemi32_locs.mat` 기준):
```
 0:FP1  1:AF3  2:F7   3:F3   4:FC1  5:FC5  6:T7   7:C3
 8:CP1  9:CP5 10:P7  11:P3  12:Pz  13:PO3 14:O1  15:Oz
16:O2  17:PO4 18:P4  19:P8  20:CP6 21:CP2 22:C4  23:T8
24:FC6 25:FC2 26:F4  27:F8  28:AF4 29:FP2 30:FZ  31:Cz
```

자동 prior (n_channels=32, `--eeg_occipital_ids auto`):
| 채널 | 인덱스 | 초기 logit | sigmoid |
|------|--------|-----------|---------|
| O1, Oz, O2 | 14, 15, 16 | +2.0 | 0.881 |
| PO3, PO4 | 13, 17 | +1.5 | 0.818 |
| P7, P3, Pz, P4, P8 | 10, 11, 12, 18, 19 | +1.0 | 0.731 |
| 나머지 22채널 | -- | 0.0 | 0.500 |

- `--eeg_occipital_ids auto` (기본값): BioSemi32 3단계 prior 자동 적용
- `--eeg_occipital_ids none`: 모든 채널 동일 (0.50) -- fully learnable
- `--eeg_occipital_ids 14,15,16`: 지정 채널만 bias (occipital_bias=1.0)
- prior는 warm start만 제공, gate는 학습으로 수렴 (!= frozen masking)

### 새 CLI Args (`train_vs_re_gen.py`)

| arg | default | 설명 |
|-----|---------|------|
| `--encoder_version` | `v1` | `v1`=기존, `v2`=강화 |
| `--eeg_stem_filters` | `32` | V2: branch당 filter 수 (총 3x) |
| `--eeg_occipital_ids` | `""` | V2: occipital channel index comma-sep |

### 실험 커맨드

```powershell
# V2 + BioSemi32 occipital prior (자동)
conda run -n eegdiff --no-capture-output python .\train_vs_re_gen.py `
    --subject_ids 1,2,18 --epochs 300 `
    --encoder_version v2 --eeg_stem_filters 32 --eeg_occipital_ids auto `
    --beta_schedule linear --lambda_lpips 0.0 `
    --lambda_percept 0.1 --lambda_rec 0.01 `
    --guidance_scale 1.5 --eta 0.0 --eval_ddim_steps 50

# V2 + prior 없이 (gate 완전 학습, ablation용)
conda run -n eegdiff --no-capture-output python .\train_vs_re_gen.py `
    --subject_ids 1,2,18 --epochs 300 `
    --encoder_version v2 --eeg_occipital_ids none `
    --beta_schedule linear --guidance_scale 1.5
```

### 중단 기준 (동일)

- DINO Top-1 < 0.1160 (baseline): 즉시 중단
- SSIM < 0.0 평균: config 재검토
- best_val_loss > 0.005: 불안정 신호

### 이후 단계

1. V2 결과 확인 후 → DINO alignment encoder 비교 (`train_vs_re_dino.py --encoder_type conv`)
2. session-merged vs session-capped 비교
3. Stage 2 (DINO latent → 생성)

### 현재 코드 기본값

```
encoder_version = v1   (실험 시 --encoder_version v2 명시)
beta_schedule   = linear
lambda_lpips    = 0.0
lambda_percept  = 0.1
lambda_rec      = 0.01
guidance_scale  = 1.5
eta             = 0.0
```

---

## Experiments 7-10: EEG Encoder V1 vs V2 Ablation (2026-04-16~17)

**Script**: `train_vs_re_gen.py`
**Subjects**: S01, S02, S18 (3 multi-session subjects for fair comparison)
**Config (shared)**: `epochs=300, beta_schedule=linear, lambda_lpips=0.0, lambda_percept=0.1, lambda_rec=0.01, guidance_scale=1.5, eta=0.0, eval_ddim_steps=50`

### Experiment 7 -- V1 Explicit Baseline (Exp004, 2026-04-16)

**Checkpoint**: `checkpoints_vsre_gen/20260416_124038_ch32_merged_ep300/`
**Command**: `--encoder_version v1 --subject_ids 1,2,18`

| Subject | best_val_loss | SSIM | LPIPS | DINO@1 | DINO@3 | DINO@5 |
|---------|-------------|------|-------|--------|--------|--------|
| S01 | 0.000661 | 0.043 | 0.780 | 0.102 | 0.324 | 0.537 |
| S02 | 0.000979 | 0.037 | 0.781 | 0.125 | 0.347 | 0.667 |
| S18 | 0.000669 | 0.043 | 0.787 | 0.083 | 0.370 | 0.593 |
| **Mean** | **0.000770** | **0.041** | **0.783** | **0.103** | **0.347** | **0.599** |

### Experiment 8 -- V2 + Occipital Prior Auto (Exp005, 2026-04-16)

**Checkpoint**: `checkpoints_vsre_gen/20260416_185126_ch32_merged_ep300/`
**Command**: `--encoder_version v2 --eeg_occipital_ids auto --subject_ids 1,2,18`

| Subject | best_val_loss | SSIM | LPIPS | DINO@1 | DINO@3 | DINO@5 |
|---------|-------------|------|-------|--------|--------|--------|
| S01 | 0.000426 | 0.044 | 0.774 | 0.102 | 0.333 | 0.528 |
| S02 | 0.000691 | 0.036 | 0.765 | 0.125 | 0.361 | 0.569 |
| S18 | 0.000424 | 0.045 | 0.778 | **0.120** | 0.352 | 0.574 |
| **Mean** | **0.000514** | **0.042** | **0.774** | **0.116** | **0.349** | **0.557** |

### Experiment 9 -- V2 No Prior Ablation (Exp006, 2026-04-17)

**Checkpoint**: `checkpoints_vsre_gen/20260417_113129_ch32_merged_ep300/`
**Command**: `--encoder_version v2 --eeg_occipital_ids none --subject_ids 1,2,18`

| Subject | best_val_loss | SSIM | LPIPS | DINO@1 | DINO@3 | DINO@5 |
|---------|-------------|------|-------|--------|--------|--------|
| S01 | 0.000428 | 0.048 | 0.775 | **0.130** | 0.352 | 0.620 |
| S02 | 0.000690 | 0.038 | 0.764 | **0.139** | 0.347 | 0.528 |
| S18 | 0.000429 | 0.045 | 0.769 | 0.111 | 0.361 | 0.565 |
| **Mean** | **0.000516** | **0.043** | **0.769** | **0.127** | **0.353** | **0.571** |

### V1 vs V2 Summary

| Config | mean val_loss | mean SSIM | mean DINO@1 | vs baseline |
|--------|-------------|-----------|-------------|-------------|
| Old baseline [20260411], all-subj | 0.002779 | 0.056 | 0.1160 | reference |
| V1 explicit, S1/2/18 | 0.000770 | 0.041 | 0.103 | -11% |
| V2 + occipital prior | 0.000514 | 0.042 | 0.116 | ~= baseline |
| **V2 no-prior** | **0.000516** | **0.043** | **0.127** | **+9%** |

### Key Findings

1. **V2 architecture clearly beats V1** regardless of prior setting (val_loss -33%, DINO@1 +12~23%)
2. **Occipital prior is subject-specific**: helps S18 (+44% DINO@1 vs V1), but hurts S01/S02 vs no-prior
3. **V2 no-prior is the best overall config** (mean DINO@1 0.127 -- best so far on 3-subject benchmark)
4. The OccipitalChannelGate architecture itself is valuable; the BioSemi32 prior initialization constrains learning for S01/S02
5. **Recommended default going forward**: `--encoder_version v2 --eeg_occipital_ids none`

### Experiment 10 -- Session-Capped Comparison (Exp007, running)

**Command**: `--encoder_version v2 --eeg_occipital_ids none --max_sessions 2 --subject_ids 1,2,18`
**Status**: Currently training (PID 26987)
**Note**: S01/S18 capped from 8 sessions (864 train) → 2 sessions (216 train); S02 from 5→2 sessions (216 train)
**Goal**: Test if session imbalance hurts generalization -- comparing session-merged (Exp006) vs capped

Results pending.

---

## File Structure

```
test_diffusion_model/
├── model_128_eegonly_transformer.py          # Base UNet + EEG encoder
├── model_128_eegonly_transformer_repa.py     # + Perceptual loss, EEG-only cond
├── model_128_eegonly_transformer_repa_cls.py # + Classification head
├── model_eeg_transformer_cls_only.py         # Classification only
├── model_eeg_dino.py                         # Stage 1: EEG-DINO regressor
├── train_vs_repa_allclass.py                 # Generative model training
├── train_vs_cls_only.py                      # Classification training
├── train_vs_repa_cls.py                      # Joint training
├── train_crosssubj_dino.py                   # Cross-subject DINO training
├── sample_vs_repa_allclass.py                # Image generation
├── compare_cls_ab.py                         # Method A vs B comparison
├── ablation_subj_emb.py                      # Subject embedding ablation
├── eval_retrieval.py                         # Top-k retrieval evaluation
├── checkpoints_vs_repa/                      # Generative model checkpoints
├── checkpoints_vs_cls/                       # Classification checkpoints
├── checkpoints_dino/                         # DINO alignment checkpoints
└── samples_vs_repa/                          # Generated images
```

---

## Section 6: New Data -- preproc_vs_re (2026-04-11)

New repeated-session VS EEG data added.

### Overview

| Item | Details |
|------|---------|
| Path | `preproc_vs_re/` |
| Paradigm | Same VS experiment as `preproc_for_gan_vs` |
| File format | MATLAB v7.3 (HDF5) -- must use `h5py` (not scipy.io) |
| Naming | `preproc_subj_{sid:02d}_{session}.mat` |
| Total files | 54 |
| Subjects | 18 (non-consecutive IDs) |
| Total trials | 7,290 |

### File Internal Structure

```
results/data shape: (5, 3, 9, 4096, 40)  dtype=float32
  dim0=5   : trials per class per block
  dim1=3   : blocks per session
  dim2=9   : classes (1~9)
  dim3=4096: time points -- 1024 Hz x 4 sec (epoch: -1 ~ +3 sec)
  dim4=40  : channels -- 32 EEG + 8 EX (all used)
```

Trials per file: 5 x 3 x 9 = **135 trials/session** (consistent across all files)

### Epoch Time Alignment

| Range | Sample index | Note |
|-------|-------------|------|
| -1 ~ 0 sec | 0 ~ 1023 | Pre-stimulus baseline |
| 0 ~ +2 sec | 1024 ~ 3071 | **Same window as preproc_for_gan_vs** |
| +2 ~ +3 sec | 3072 ~ 4095 | Additional post-stimulus |

To align with existing data: `data[:, :, :, 1024:3072, :]`

### Per-Subject Session Count

| Sessions | Subjects | Count |
|----------|----------|-------|
| 1 | S03, S05, S11, S16, S19, S20, S21, S23, S35 | 9 |
| 2 | S04, S09, S10 | 3 |
| 4 | S28 | 1 |
| 5 | S02, S29 | 2 |
| 8 | S01, S18 | 2 |
| 9 | S24 | 1 |

Trial range per subject: 135 (1 session) ~ 1,215 (9 sessions, S24)

### Comparison with preproc_for_gan_vs

| Item | preproc_for_gan_vs | preproc_vs_re |
|------|-------------------|---------------|
| Paradigm | VS | VS (identical) |
| Subjects | 20 (S01~S20) | 18 (non-consecutive) |
| File format | MATLAB v5 (scipy.io) | MATLAB v7.3 (h5py) |
| Epoch | 0 ~ +2 sec | -1 ~ +3 sec (sliceable to 0~+2) |
| Channels | 32 EEG | 40 (32 EEG + 8 EX) |
| Trials/subject | 360 (fixed) | 135 ~ 1,215 (session-dependent) |
| Repeated sessions | No | Yes (1~9 sessions) |

### TODO

- [x] Implement h5py loader -- `dataset_vs_re.py` (0~2 sec, 32ch, session-merged/capped 지원)
- [x] Subject-wise DINO alignment training -- `train_vs_re_dino.py`
- [x] Subject-wise generation training -- `train_vs_re_gen.py`
  - Bug fixes applied (2026-04-11):
    - `p_losses()` 호출 시 `t` 누락 → `torch.randint`로 생성 후 전달
    - `sample_ddim(steps=...)` → `sample_ddim(num_steps=...)` 인자명 수정
    - `main()`에서 `out_csv` 미정의 → `train_subject()`가 결과 dict 반환, `main()`에서 CSV 저장
- [x] Run subject-wise DINO baseline -- `train_vs_re_dino.py` (2026-04-11, see Experiment 4)
- [x] Run subject-wise generation baseline -- V1 Exp004, V2+prior Exp005, V2-no-prior Exp006 (see Experiments 7-9)
- [ ] Compare session-merged vs session-capped results -- Exp007 currently running
- [ ] Confirm subject ID overlap between preproc_for_gan_vs and preproc_vs_re

---

## Research Direction

Based on analysis, the recommended architecture for this study:

```
EEG-only + Subject Embedding
  └─ No class label conditioning (avoids leakage)
  └─ Subject embedding instead of per-subject models

Pretrained Visual Feature Alignment (DINOv2)
  └─ Richer semantic features than ResNet18 ImageNet
  └─ Self-supervised → better instance discrimination

Evaluation
  └─ Primary: top-1/3/5 retrieval accuracy
  └─ Secondary: class accuracy (supplementary only)
  └─ Report within-subject and LOSO separately

Stage 2 (pending Stage 1 validation)
  └─ EEG latent → pretrained VAE decoder
  └─ Or latent diffusion in DINO/VAE space
```
