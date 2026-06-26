# HUMAN_DIRECTIVE

Last Updated: 2026-06-17
Owner: user
Status: active

---

## 1. Objective

### Current Directive: after Exp42-B Step4, 2026-06-17

This section is the active instruction. It supersedes older Priority 0/1/2 text below where they conflict.

Documentation rule:
- Preserve as much Exp history as possible in `PROGRESS.md`.
- Do not remove older experiment sections when priorities change.
- If an old directive is superseded, leave it in place and rely on the newest dated/current section to override it.
- Failed experiments and negative results must remain visible because they justify the current SD LoRA and VS->VI direction.
- When updating this file, add a new current directive section rather than deleting prior reasoning unless it is factually wrong.

Current confirmed results:
- Exp31 connected the SupCon encoder back to pixel generation and confirmed the old FiLM/pixel path was a bottleneck.
- Exp32~37 moved to frozen SD VAE latent generation and cross-attention; this solved the worst pixel-collapse failure but did not close the gap to retrieval.
- Exp39 replaced brittle sample-grid/single-template reporting with full-test multi-gallery evaluation.
- Exp40 multi-image target training degraded performance; do not continue multi-image target training in the current scratch latent-UNet architecture.
- Exp41 SD 1.5 LoRA is the new project best: S18 dual-gallery DINO@1 `0.2870`, entropy `2.072`, dominant `23.1%`.
- Exp42-A SD 1.5 LoRA on S01 reached dual-gallery DINO@1 `0.3333`, matching the Stage 2 retrieval reference while remaining fully generative.
- Exp42-B Step4 reached S01 dual-gallery DINO@1 `0.3571` with r=32 and 16 EEG tokens.
- Exp42-B Step5 S18 target augmentation completed: internal partial-test DINO@1 `0.4464`, entropy `1.960`, dominant `25.0%`, best_ep `95`. This is not yet a full-test dual-gallery score.
- Stage 2 retrieval is now exceeded by the best generator; SD LoRA has surpassed retrieval-level performance on S01.

Current bottleneck interpretation:
- The scratch VAE-latent diffusion generator reached a practical ceiling around S18 dual-gallery DINO@1 `0.2315`.
- SD 1.5 LoRA improved image prior, sharpness, and class diversity, so the active path should now optimize and transfer LoRA rather than continue scratch latent-UNet tweaks.
- Remaining weak classes include heart/star and some symbolic/geometric classes.
- The earlier assumption that `S01` would remain worse than `S18` is no longer valid under SD LoRA.
- SD prior can compensate for weak EEG class separability more than expected.
- S01-vs-S18 differences should now be tracked empirically rather than inferred from inter-class similarity alone.
- S18 and S01 have different apparent rank optima:
  - S18: r=16, 16 tokens is best so far
  - S01: r=32, 16 tokens is best so far
- Treat LoRA rank as subject/signal-quality dependent, not a global constant.
- Do not repeat Exp31~40 unless the code changed enough to test a new hypothesis.

Active priority order:

1. `Priority 0 = S24 SD LoRA VS generation`
   - Run S24 with the same SD LoRA setup as Exp42-A/Step4.
   - Test both `r=16` and `r=32` if compute allows.
   - Purpose:
     - test whether high session count alone predicts LoRA generation success
     - separate session-count benefit from intrinsic EEG signal quality
     - provide another VS/VI-overlap candidate for later VI fine-tuning
   - Interpretation:
     - if `S24 > S01`, session/data volume is a major driver; add more high-session subjects before broad conclusions
     - if `S24 ≈ S01`, SD LoRA may be approaching a practical within-subject ceiling once enough data is available
     - if `S24 < S01`, subject-specific EEG signal quality matters more than data volume; prioritize per-subject selection for VI transfer

2. `Priority 1 = Exp43 SD LoRA-based VI fine-tuning`
   - Do not wait for Exp42-B Step5 augmentation.
   - Replace the older Exp38 CA-UNet VI plan with SD LoRA.
   - Required comparison:
     - `C0`: VI scratch LoRA
     - `C1`: VS LoRA -> VI fine-tune with frozen EEG encoder
     - `C2`: VS LoRA -> VI fine-tune with staged encoder unfreeze
   - First checkpoints:
     - S01: use Exp42-B Step4 r=32, 16-token checkpoint
     - S18: use Exp42-B Step2 r=16, 16-token checkpoint
   - Rank design:
     - do not force the same rank for all subjects
     - include r=32 in VI runs even when VS best is r=16, because VI EEG is expected to provide weaker conditioning
   - Scheduling:
     - start in parallel with S24 if compute allows
     - if later ablations improve full-test dual-gallery metrics, use the new checkpoint as updated initialization
   - Success:
     - minimum: `C1 > C0`
     - meaningful: VI DINO@1 `> 0.20`
     - collapse lower than VI scratch
   - C2 trigger:
     - if C1 does not show validation improvement by about epoch `50`, start C2 in parallel rather than waiting for full C1 completion
     - C2 should keep the encoder frozen initially and unfreeze with a much smaller LR only after the generator/projection stabilizes

3. `Priority 2 = Exp42-B Step5 class-preserving image augmentation`
   - S18 r=16/16tok augmentation has already been run once.
   - Treat its `0.4464` score as internal/preliminary until full-test dual-gallery evaluation exists.
   - Run additional augmentation only if compute remains after S24 and Exp43 have started.
   - Treat as secondary cleanup for heart/star confusion, not the main next bottleneck.
   - Use class-preserving transforms only:
     - natural classes: mild color/contrast, small rotation, horizontal flip only where semantically safe
     - symbol/digit classes: avoid flip, keep rotation small, add style/background variants carefully

4. `Priority 3 = Remaining Exp42-B optimization with S01/S18 tracking`
   - Exp42-A decision outcome:
     - `S01 >= 0.20` condition passed strongly (`0.3333`)
     - proceed with Exp42-B and Exp43
   - Run ablations one variable at a time in this order:
     - Step 1: CFG was skipped because no real unconditional path exists
     - Step 2: increase EEG tokens `8 -> 16` while keeping projection type and LoRA rank fixed
     - Step 3: replace linear EEG projection with MLP projection using the best token count from Step 2
     - Step 4: test LoRA rank `r=8` and `r=32` using the best projection/token setup
     - Step 5: add class-preserving image augmentation using the best rank/projection setup
     - Step 6: staged encoder unfreeze only at the end, with encoder LR about `1e-6`
   - Track both `S18` and `S01` as mandatory paired subjects:
     - S18 remains the original LoRA benchmark subject
     - S01 is now the project-best generative subject and should not be treated as only a weak-signal control
     - do not accept an Exp42-B ablation result from S18 alone because token/rank/projection changes may affect S01 and S18 differently
   - Success:
     - minimum: dual-gallery DINO@1 `>= 0.30`
     - target: dual-gallery DINO@1 `>= 0.32`
     - heart/star confusion decreases
     - entropy does not collapse
   - Proceed-to-VI rule:
     - Exp43 may proceed even if every ablation does not improve, because S01 already reached retrieval-level generation
     - prefer the best validated LoRA config across S01/S18, not S18 alone

5. `Priority 4 = Direct EEG -> SD VAE latent mapper`
   - Keep as a fallback/control, not the immediate main path after Exp41.
   - Use if SD LoRA optimization fails or if a very small VI-friendly generator is needed.
   - Structure:
     - EEG -> frozen SupCon latent
     - small MLP/ResMLP -> SD VAE latent
     - frozen SD VAE decoder -> image

Lower priority:
- 256x256/custom latent training is lower priority because SD LoRA already gives a stronger 512x512 image prior.
- Recolor/rework symbolic targets only as dataset-design ablation, not as the main model fix.
- More multi-image target training is deferred unless images are curated to be visually class-consistent.

### Objective Override: 2026-06-08

The active project goal is now:
- generate images from EEG as the primary target
- use representation learning only if it improves generator conditioning
- evaluate whether the improved SupCon encoder actually reduces generation collapse
- run the main VS -> VI transfer path within-subject first, not cross-subject

This supersedes the earlier `representation first, generation as probe` priority where they conflict.

Immediate consequence:
- VI transfer readiness is no longer Priority 0.
- Pixel diffusion / image generation is no longer suspended as a whole.
- The next main question is whether the best VS SupCon encoder can make the generator use EEG conditioning.
- Stage 2 readout remains useful as a reference, but it is not a substitute for image generation.
- The active transfer structure is:
  - same subject VS EEG -> pretrain encoder/generator
  - same subject VI EEG -> fine-tune/adapt
  - same subject VI EEG -> generate imagined image
- Subject-to-subject generalization is deferred until within-subject generation is stable.

Primary failure to solve:
- Exp13 confirmed severe single-class generation collapse.
- Exp14~19 anti-collapse/prototype losses did not solve it.
- Exp23-B SupCon improved representation, but it has not been connected back to the generator.
- Therefore the first required generator experiment is: current generator + Exp23-B SupCon encoder initialization/freeze + full collapse diagnostics.

Decision rule:
- If SupCon-conditioned generation reduces collapse versus Exp13, then encoder conditioning was a major bottleneck and generator-side refinement is justified.
- If collapse remains, the generator conditioning mechanism is the bottleneck, and FiLM/vector conditioning should be replaced or augmented before more encoder-only work.

Initial within-subject targets:
- `S01`: strongest VS data, prior Exp23-B checkpoint exists
- `S18`: strong VS data, prior Exp23-B checkpoint exists
- `S24`: high VS session count, should be added to the SupCon-conditioned generator test
- Secondary: `S02/S29/S28/S09` after the first collapse diagnostic

---

## Previous Objective Context

The actual project goal is:
- pretrain on `VS` EEG
- learn a transfer-friendly visual EEG representation
- later improve `VI` (visual imagery) performance

This previous VI-transfer objective is temporarily demoted while the project tests whether generation collapse can be fixed.

Current interpretation:
- generation is a supporting probe
- the current generation task is closer to `class prototype generation`
- not trial-specific image reconstruction

Therefore the next cycles must prioritize representation learning first.

Current priority correction as of 2026-06-01:
- do not keep delaying `VS -> VI` transfer checks
- do not conclude from `S01/S02/S18` alone
- do not merge preprocessing variables into one ablation when the result must be interpretable
- do not resume pixel diffusion before VI transfer readiness is checked

Current priority correction as of 2026-06-05:
- treat `Exp25-VI` as a major warning, not a minor weak result
- diagnose the `VS -> VI` latent gap before spending more cycles on VS-only improvements
- validate `S23` data integrity before using all-subject averages or partial-session follow-up
- do not run `Exp27` until the above two checks are complete
- after `Exp27-preB`, pause Exp27 unless its purpose is redefined as secondary support
- run the VI temporal-resolution test next: VI `512 -> 2048` interpolation and repeat zero-shot transfer

---

## 2. Current Decision

- Dataset: `preproc_vs_re`
- Updated dataset state:
  - `preproc_vs_re` now includes additional files with partial-session saves
  - file count is no longer equivalent to valid session count
  - some files contain fewer valid sessions because low-trial sessions were skipped during preprocessing
- Time window: `0~2 sec` only
- Channels: `32 EEG only`
- Main focus: EEG-conditioned image generation and collapse recovery
- Representation learning is useful only insofar as it improves generator conditioning
- Do not use class-label conditioning in generation
- Pixel diffusion is allowed again only for controlled collapse-diagnostic experiments
- Stage 2 latent/readout is a reference baseline, not a replacement for generation

Known partial-save examples:
- `preproc_subj_19_2.mat` ~ `16.1 MB`  -> estimated `1` valid session
- `preproc_subj_28_5.mat` ~ `32.4 MB`  -> estimated `2` valid sessions
- `preproc_subj_02_8.mat` ~ `48.2 MB`  -> estimated `3` valid sessions
- `preproc_subj_06_1.mat` ~ `48.2 MB`  -> estimated `3` valid sessions
- `preproc_subj_08_1.mat` ~ `47.9 MB`  -> estimated `3` valid sessions
- `preproc_subj_09_4.mat` ~ `48.1 MB`  -> estimated `3` valid sessions
- several files around `~64 MB` -> estimated `4` valid sessions
- one known missing file case:
  - `preproc_subj_02_10` missing because all sessions in the source were skipped

Therefore:
- never assume a fixed `5-session` payload per file
- use actual loaded trial count as the ground truth
- use additional data if valid trials exist, even when files are partially saved

Current full-subject generation reference:
- checkpoint: [`20260411_165536_ch32_merged_ep300`](/c:/Users/Biocomputing/Desktop/workspace_VIVS/test_diffusion_model/checkpoints_vsre_gen/20260411_165536_ch32_merged_ep300/results_gen.csv)
- mean `best_val_loss = 0.002779`
- mean `L1 = 0.805959`
- mean `SSIM = 0.056175`
- mean `DINO Top-1 = 0.115998`
- mean `DINO Top-3 = 0.297325`
- mean `DINO Top-5 = 0.552396`

Current best subset generation probe:
- `V2 + no prior`
- mean `DINO Top-1 = 0.127` on `subjects 1,2,18`

Important:
- `Exp007` is currently running elsewhere.
- Do not use `Exp007` as the next main task.

---

## 3. Priority Task

### Active Priority Override: Generation First

The priorities below supersede older VI-first sections where they conflict.

### Priority 0: Collapse recovery with SupCon-conditioned generator

- Task type: implement + run + summarize
- Target files:
  - `train_vs_re_gen.py`
  - `model_128_eegonly_transformer_repa.py`
  - `model_eeg_dino.py`
  - collapse diagnostic/export scripts
- Required experiment:
  - use the existing generator training path
  - initialize or freeze the generator EEG encoder from the Exp23-B SupCon checkpoint
  - run within-subject first
  - first subjects: `S01/S18/S24`
  - compare against Exp13/Exp006 generation collapse baseline
  - keep the generation task EEG-only; do not add true class labels as main conditioning
  - for S24, train or use a matching SupCon encoder before connecting the generator if no Exp23-B-style checkpoint exists
- Required diagnostics:
  - generated DINO-predicted class histogram
  - generated-vs-true confusion matrix
  - predicted-class entropy
  - true-label prototype cosine similarity
  - true-vs-best-wrong prototype margin
  - Top-1/Top-3/Top-5 as secondary identity metrics
- Decision rule:
  - if collapse drops materially versus Exp13, the encoder conditioning path was a major bottleneck
  - if collapse remains, the generator conditioning mechanism is the bottleneck, not just the encoder
  - if VS generation still collapses, do not proceed to VI fine-tuning yet
  - if VS generation becomes class/prototype-stable, run same-subject VI fine-tuning next and compare VI scratch vs VS-pretrained generator
- Success condition:
  - generated-class distribution is not dominated by one class
  - true-label prototype similarity and margin improve
  - exact Top-1 can remain modest, but identity signal must move in the right direction

### Priority 1: Generator architecture choice after SupCon connection test

Recommended generator order:

1. `Latent diffusion with frozen SD 1.5 VAE`
   - image target: SD VAE latent instead of 128x128 pixels
   - conditioning: SupCon EEG latent through cross-attention or strong conditional modulation
   - reason: frozen VAE already knows image structure, so the model learns EEG-to-latent conditioning instead of pixel synthesis from scratch
   - if FiLM is reused for the first latent-diffusion probe, treat the result as provisional; a failed run may still be a FiLM failure
   - freeze the SupCon EEG encoder for the first 50–100 epochs unless explicitly testing encoder fine-tuning
   - meaningful collapse reduction requires roughly `dominant < 70%` and `entropy > 0.5`; `70–85%` is only partial

2. `Direct EEG -> SD VAE latent mapper`
   - EEG -> frozen SupCon latent -> small MLP/ResMLP decoder -> SD VAE latent -> frozen VAE decoder
   - use if latent diffusion remains unstable or collapses
   - reason: no diffusion trajectory, lower data requirement, lower collapse risk
   - this is the diffusion-free version of Stage 2 readout and is appropriate for small within-subject data

3. `Retrieval-augmented generation`
   - EEG -> SupCon/DINO latent -> nearest visual prototype/retrieved image -> diffusion refinement
   - reason: Stage 2 readout already works better than pixel diffusion and can provide a strong prior against collapse
   - limitation: less pure as generation, especially with only 9 class images

4. `Conditional VAE`
   - use if diffusion remains unstable after SupCon conditioning
   - reason: stable on small data and useful as a lower-capacity sanity generator
   - expected drawback: blurrier images than latent diffusion

Not recommended as the next main path:
- more sampler-only DDIM/guidance tuning
- pixel-space UNet128 from scratch without changing conditioning
- SDXL before SD 1.5 VAE feasibility is tested

### Priority 2: Generator conditioning mechanism if collapse remains

- If SupCon-conditioned generator still collapses, replace or augment FiLM/vector conditioning.
- If latent diffusion still uses FiLM and collapses, do not conclude latent diffusion failed; conclude the failure is still confounded with FiLM conditioning.
- Before implementing cross-attention, run low-cost diagnosis of why Exp33 S18 succeeds while S01 collapses under the same frozen-SupCon VAE-latent setup.
- Required pre-cross-attention checks:
  - S18 vs S01 per-class trial counts
  - session-wise class distribution
  - train/val/test class balance
  - generated confusion matrix and dominant generated class
  - S01 class-balanced sampling rerun if imbalance/session skew is present
  - S01 longer-training rerun to check whether collapse is an under-training artifact
- Cross-attention is justified only if S01 remains collapsed after those checks.
- Preferred next mechanism:
  - EEG latent sequence or multi-token condition
  - cross-attention inside UNet blocks
  - classifier-free conditioning dropout only if a real unconditional path is trained
- Prepare cross-attention latent UNet design notes in parallel, but do not prioritize full implementation until the S18/S01 diagnosis is complete.
- Do not spend more cycles on encoder-only retrieval improvements until the generator conditioning path is proven to use the EEG signal.

### Deferred Priority: VI transfer readiness check

- Task type: run + summarize
- Target files:
  - `train_vs_test_vi.py`
  - if needed, a small VI zero-shot/readout evaluation helper
- Problem:
  - the final research target is `VS pretraining -> VI transfer`
  - current results mostly measure VS-internal retrieval
  - if VS and VI distributions have a structural gap, delaying this check can waste later VS-only optimization
- Required direction:
  - run a minimal `VS -> VI` transfer test immediately after or in parallel with `Exp025`
  - use the best available SupCon checkpoint from `Exp23-B` as the first candidate
  - report even poor results; failure is informative
- Minimum accepted output:
  - subject overlap used
  - VI dataset path and split
  - checkpoint path
  - VI retrieval or generated/readout image metric
  - a short diagnosis: encoder gap vs data-distribution gap vs script incompatibility
- Success condition:
  - transfer path runs end-to-end
  - VI performance is measured early enough to influence the next representation experiments

Current result interpretation:
- `Exp25-VI` with the best SupCon VS encoder produced VI Top-1 close to chance.
- This is not enough to justify continuing VS-only optimization blindly.
- Before `Exp27`, compare VS vs VI latent distributions using:
  - `diagnose_vs_vi_latent_gap.py`
- Required measurements:
  - VS and VI true-label prototype cosine similarity
  - VS and VI best-wrong prototype similarity
  - VS and VI true-vs-wrong margin
  - per-subject Top-1/Top-3/Top-5 gap
- Decision rule:
  - if VI margins collapse while VS margins are positive, prioritize VI adaptation / domain bridging over partial-session VS optimization
  - if both VS and VI margins are weak, improve encoder representation
  - if only specific subjects fail, inspect subject/channel/session data

Current diagnosis:
- VS margins are already negative for S01/S02/S18, and VI margins are worse.
- This means SupCon improved rank accuracy but did not create robust prototype separation.
- Therefore Exp27 is not the main next step.
- Next main test: temporal mismatch isolation via VI 512-sample to 2048-sample interpolation.

### Priority 1: All-subject SupCon validation

- Task type: run + summarize
- Target files:
  - `train_vs_re_dino.py`
  - `model_eeg_dino.py`
- Problem:
  - `S01/S02/S18` are not representative of all 18 subjects
  - S01 and S18 have many sessions, and S02 is also not the lowest-data case
  - a three-subject benchmark can overestimate the usefulness of SupCon
- Required direction:
  - run the best current SupCon config on all available `preproc_vs_re` subjects
  - compare against the current all-subject DINO baseline where available
  - explicitly separate high-session and low-session subjects in the summary
- Success condition:
  - SupCon remains useful beyond `S01/S02/S18`
  - low-session subjects are not hidden by mean performance

### Priority 1.5: S23 data integrity check

- Task type: verify + summarize
- Target files:
  - `dataset_vs_re.py`
  - `diagnose_vsre_subject_data.py`
- Problem:
  - `S23 Top-1 = 0.0000` in both early and all-subject runs is a data-quality warning
  - this is qualitatively different from merely low performance
- Required direction:
  - inspect raw `preproc_vs_re` files for S23
  - report shape, trial count, per-class count, finite fraction, mean/std
  - confirm label balance and session validity
  - decide whether S23 should be included, excluded, or handled separately
- Success condition:
  - S23 is classified as valid-low-signal vs invalid/corrupt/misloaded
  - all-subject averages are interpreted accordingly

### Priority 2: Split preprocessing ablation

- Task type: run + summarize
- Target files:
  - `dataset_vs_re.py`
  - `train_vs_re_dino.py`
- Problem:
  - applying baseline correction and channel z-score simultaneously makes the result ambiguous
  - if the combined run improves, the causal factor is unknown
- Required direction:
  - split `Exp24` into:
    - `Exp24-A`: SupCon + baseline correction only
    - `Exp24-B`: SupCon + channel-wise z-score only
    - `Exp24-C`: SupCon + baseline correction + channel-wise z-score only if A or B is promising
  - keep subject set, seed, encoder, loss, and epochs fixed
- Success condition:
  - one preprocessing factor can be selected or rejected with clear evidence

### Legacy / Deferred: Keep pixel diffusion generation suspended

- Task type: documentation + enforcement
- Target files:
  - `PROGRESS.md`
  - generation scripts only if needed for diagnostics
- Decision:
  - pixel diffusion generation is not the next optimization target
  - current evidence shows Stage 2 latent/readout beats pixel diffusion strongly
  - do not restart sampler/loss/sharpness tuning before VI transfer readiness is checked
- Allowed generation work:
  - qualitative examples from Stage 2 readout
  - fixed-protocol diagnostic exports
  - true-label prototype/collapse diagnostics if needed
- Not allowed for the next cycles:
  - new pixel diffusion hyperparameter sweeps
  - sharpness-only tuning
  - repeated DDIM/guidance/schedule searches

### Priority 4: Make generated images resemble the true-label prototype before optimizing exact class accuracy

- Task type: analyze + implement + run + summarize
- Target files:
  - `train_vs_re_gen.py`
  - `model_128_eegonly_transformer_repa.py`
  - evaluation/export scripts as needed
- Problem:
  - current generation often fails before class accuracy even matters
  - some outputs collapse toward one class or produce weak, non-prototype-like images
  - even incorrect outputs should move toward the `true-label` visual prototype first
- Required primary direction:
  - optimize `prototype similarity` before optimizing strict top-1 class accuracy
  - generated images should be semantically closer to the true-label image than to unrelated classes
  - keep generation `EEG-only` for the main result
- Required losses and supervision:
  - add `generated image -> true-label DINO prototype` attraction loss
  - add `generated image -> non-target prototypes` separation loss or margin term
  - allow weak `EEG latent -> class CE` and `EEG latent -> DINO prototype` auxiliary losses
  - make all new loss weights explicit and ablatable
- Required diagnostics:
  - save generated DINO-predicted class histogram
  - save generated-vs-true confusion matrix
  - save predicted-class entropy
  - report per-subject collapse pattern
  - report true-label prototype similarity directly
- Required sanity check:
  - allow `true-label-conditioned generation` only as an `upper-bound diagnostic`
  - do **not** use true-label conditioning as the main reported result
  - use it only to distinguish `generator failure` from `EEG conditioning failure`
- Checkpoint selection:
  - do not select best checkpoint only by stochastic validation diffusion loss
  - include `true-label prototype similarity`, `generated DINO Top-1`, and `predicted-class entropy` in model selection/reporting
- Success condition:
  - generated images become more similar to the true-label prototype
  - generated-class histogram is not dominated by a single class
  - top-k accuracy may remain imperfect, but semantic identity should improve first

### Priority 5: Generation collapse diagnosis and EEG-only anti-collapse loss

- Task type: analyze + implement + run + summarize
- Target files:
  - `train_vs_re_gen.py`
  - `model_128_eegonly_transformer_repa.py`
  - evaluation/export scripts as needed
- Problem:
  - current generation may collapse into one dominant class prototype
  - low performance should not be interpreted only from Top-k scores
  - generated-class distribution must be inspected directly
- Required anti-collapse direction:
  - add weak EEG latent auxiliary supervision before generator-heavy changes
  - preferred auxiliary losses:
    - EEG latent -> DINO prototype contrastive / retrieval loss
    - EEG latent class CE head as a representation regularizer
  - keep anti-collapse changes compatible with Priority 1 prototype-similarity goals
- Success condition:
  - generated-class histogram is not dominated by a single class
  - entropy improves without harming prototype similarity

### Priority 6: DINOv2-centered VS pretraining

- Task type: analyze + implement + run + summarize
- Target files:
  - `model_eeg_dino.py`
  - `train_vs_re_dino.py`
  - if needed, a new VS pretraining script
- Goal:
  - make DINOv2 the primary representation target
  - optimize EEG encoder quality for transfer
- Required direction:
  - DINO/image-feature alignment remains the main objective
  - auxiliary classification head is allowed
  - multi-task representation learning is allowed if controlled
- Immediate preferred order inside this priority:
  1. `test-time augmentation (TTA)` for retrieval
  2. `SupCon` ablation against current InfoNCE-based setup
  3. `baseline correction` using `-1~0 sec`
  4. `channel-wise z-score`
  5. optional `bandpass` ablation after the above
- Specific guidance:
  - `SupCon` is preferred because it removes same-class false negatives structurally
  - do not replace everything at once; compare:
    - `InfoNCE only`
    - `SupCon only`
    - `SupCon + proto`
  - TTA should be low-risk and low-cost:
    - small noise / shift only
    - average `5~10` latent predictions
- Success condition:
  - VS retrieval metrics improve or stabilize
  - encoder quality is clearer than current generation-only evidence

### Priority 7: Robust loading and sampling for partially saved `preproc_vs_re`

- Task type: analyze + implement + verify
- Target files:
  - `dataset_vs_re.py`
  - related training scripts that assume subject/session counts
- Problem:
  - new `preproc_vs_re` files may contain `1~4` valid sessions instead of the previously expected fixed payload
  - missing/partial sessions can silently bias sampling and subject comparisons
- Required direction:
  - loader must infer usable trials from actual tensor shape, not file-name assumptions
  - training/validation/test splits must use actual per-class trial counts after loading
  - session-capped logic must cap by `valid loaded sessions/trials`, not nominal session id
  - report subject-wise effective trial count and effective session count
  - keep partial files if they contain valid data
- Recommended use of the new data:
  - include all valid partial-session files in training
  - compare `old subset only` vs `old + partial new files`
  - track whether subjects with previously low data (e.g. S02, S19, S28) improve
- Success condition:
  - no loader failure on partial files
  - reproducible subject-wise trial counts are reported
  - representation metrics improve or at least variance across subjects is reduced

### Priority 8: Stage 2 latent/readout path to bypass generation collapse

- Task type: analyze + implement + run + summarize
- Goal:
  - avoid direct pixel-space collapse by using the stronger representation model as the main bridge
- Required direction:
  - treat this as `latent/readout` first, not final image generation
  - preferred structure:
    - `EEG -> best representation encoder`
    - `representation latent -> DINO latent prediction / retrieval`
    - nearest-neighbor image readout first
  - use `Exp012`-class representation quality as the foundation
  - only add a lightweight decoder after latent/readout quality is verified
- Important:
  - this is the most realistic short-term escape route from pixel-space collapse
  - evaluate retrieval/readout quality before decoder complexity
- Success condition:
  - collapse is bypassed at the readout level
  - true-label nearest-neighbor retrieval becomes more stable than direct pixel diffusion

### Priority 9: CLIP image-embedding comparison

- Task type: analyze + implement + run + summarize
- Target files:
  - new or adapted CLIP-based pretraining module/script
  - related encoder files as needed
- Goal:
  - compare DINOv2 image space vs CLIP image space for EEG alignment
- Required direction:
  - use CLIP **image embedding**, not text embedding, as the first comparison target
  - compare transfer-relevant representation quality against DINOv2
- Success condition:
  - determine whether CLIP image space is better, worse, or complementary to DINOv2

### Priority 10: EEG inductive bias strengthening

- Task type: implement + run + summarize
- Target files:
  - `model_eeg_dino.py`
  - `model_128_eegonly_transformer.py`
  - `train_vs_re_dino.py`
  - `train_vs_re_gen.py`
- Goal:
  - strengthen encoder structure with EEG-specific bias
- Allowed direction:
  - V2 encoder
  - multi-scale temporal conv
  - depthwise / separable temporal conv
  - occipital / parietal inductive bias
  - early / late temporal window split
- Success condition:
  - encoder-side metrics improve first
  - generation probe may improve secondarily

### Priority 11: Stable Diffusion 1.5 as downstream generator backbone

- Task type: analyze + design + limited prototype
- Goal:
  - use a pretrained large image generator instead of training the current pixel-space generator from scratch
- Required direction:
  - use SD 1.5 as the first generator candidate
  - keep generator mostly frozen or lightly adapted
  - connect EEG representation to a visual latent / conditioning space
- Important:
  - this is **after** DINO/CLIP representation direction is clearer
  - do not jump straight to SDXL
- Success condition:
  - a realistic migration plan exists from current probe generator to pretrained latent generator

### Legacy / Deferred: Generation as probe only

- Task type: analyze + limited run + summarize
- Goal:
  - keep generation as a readout of encoder quality
- Allowed use:
  - compare encoder variants under fixed generation settings
  - export qualitative examples
- Not allowed as primary direction:
  - repeated sampler-only tuning
  - sharpness chasing without identity improvement

### Priority 13: Pixel-generator structural upgrades only after Stage 2 evidence

- Task type: analyze + implement + limited run
- Allowed direction:
  - proper classifier-free guidance training
  - cross-attention conditioning instead of vector-only FiLM
- Important:
  - these are not the immediate next step
  - use only if Stage 2 latent/readout still leaves a strong reason to keep pixel diffusion alive
  - do not prioritize these ahead of representation, TTA, SupCon, or Stage 2

---

## 4. Recommended Model Order

1. `Current generator + Exp23-B SupCon encoder`
   - immediate diagnostic, not the final generator
   - required to test whether representation improvements actually reduce collapse

2. `Latent diffusion with frozen SD 1.5 VAE`
   - first recommended generator upgrade
   - train in VAE latent space instead of 128x128 pixel space
   - use SupCon/DINO EEG representation as conditioning

3. `Retrieval-augmented latent generation`
   - use Stage 2 retrieval/readout as a visual prior
   - refine retrieved/VAE latent with EEG conditioning

4. `Conditional VAE`
   - stable small-data fallback if diffusion remains unstable
   - useful sanity generator, but likely blurrier

5. `DINOv2 / CLIP image embedding`
   - still useful as frozen evaluation/alignment spaces
   - not the final generator by themselves

6. `SDXL` and `ImageBind`
   - not current priority
   - only consider after SD 1.5 VAE / latent diffusion feasibility is tested

---

## 5. Constraints

- Do not change `0~2 sec`
- Keep `32 EEG only` as default
- Do not assume fixed session payload from file names or legacy file size expectations
- Do not add class-label conditioning to generation
- Do not optimize sharpness before prototype identity
- Do not interpret low generation Top-k without checking generated-class histogram
- Do not delay VI transfer readiness checks beyond `Exp025`
- Do not prioritize generator tuning over encoder work
- Do not use `Exp007` as the next main task
- Keep new encoder experiments on `subjects 1,2,18` before all-subject scale-up
- Keep checkpoint metadata complete and traceable
- Do not treat `S01/S02/S18` as enough evidence for final SupCon selection
- Do not combine baseline correction and channel-wise z-score as the first preprocessing ablation
- Do not restart pixel diffusion optimization before VI transfer readiness is measured

---

## 6. Experiment Spec

### Active Within-Subject Generation Order

This order supersedes older Exp28+ VI-transfer-only plans where they conflict.

1. `Gen-WS-01 = SupCon encoder -> current generator collapse diagnostic`
   - subjects: `S01/S18/S24`
   - train/evaluate within-subject
   - compare to Exp13/Exp006 collapse baseline
   - required outputs: histogram, confusion matrix, entropy, true-label prototype similarity, true-vs-wrong margin

2. `Gen-WS-02 = generator conditioning decision`
   - if Gen-WS-01 reduces collapse: continue with current generator and improve losses/conditioning
   - if Gen-WS-01 does not reduce collapse: replace FiLM/vector-only conditioning with stronger conditioning, preferably cross-attention or latent diffusion

3. `Gen-WS-03 = same-subject VI fine-tuning`
   - only after VS generation is class/prototype-stable
   - prior evidence:
     - Exp25-VI zero-shot Top-1 `0.130`: VS encoder barely works on VI without adaptation
     - Exp28-B2 frozen VS encoder + VI linear probe Top-1 `0.154`: best VI transfer diagnostic so far
     - Exp28-B1 full fine-tune Top-1 `0.117`: naive fine-tuning can degrade VS representation
     - Exp37 sample-grid DINO@1 was initially reported as mean `0.360`, but Exp39 full-test multi-gallery evaluation showed this was inflated
     - Current reliable S18 VS generation reference is Exp39 dual-gallery DINO@1 `0.2315`
   - compare:
     - VI scratch generator
     - VS-pretrained generator -> VI fine-tune
   - first subject: `S18`
   - reason: S18 remains the cleanest first transfer subject under full-test evaluation (`Exp39 dual-gallery DINO@1=0.2315`) and has the best non-collapsed visual samples
   - S18 also has lower EEG inter-class similarity (`0.59`) than S01 (`0.75`), so it is the cleanest first transfer test
   - S01 is secondary diagnostic only, because Exp35 confirmed high EEG inter-class similarity and persistent collapse
   - subject order: `S18`, then `S01`, then `S24/S09`, then `S02/S29/S28`
   - required comparison:
     - `C0 = VI scratch`: random-init VAE-latent cross-attention generator trained only on VI
     - `C1 = VS pretrained -> VI fine-tune, frozen encoder`: main condition
     - `C2 = VS pretrained -> VI fine-tune, staged unfreeze`: optional/secondary full fine-tune, unfreeze after about 50 epochs with smaller LR
   - training rule:
     - VI has about `60 trials/class`, train about `432` trials/subject
     - use `50–100` epochs plus early stopping, not the full VS `300` epochs by default
     - C0/C1/C2 must use the same VI split
   - success criteria:
     - minimum: `C1 > C0`
     - meaningful: VI DINO@1 `> 0.20`
     - target: VI DINO@1 `> 0.20`
     - minimum S18 target threshold: `C1 > C0` plus reduced collapse versus VI scratch
   - warning:
     - Exp28-B showed full fine-tuning can underperform frozen representations
     - therefore frozen encoder VI fine-tuning must be tested before unfrozen VI fine-tuning

4. `Gen-WS-04 = latent diffusion / SD 1.5 VAE migration`
   - first migration target if current pixel generator remains unstable
   - keep VAE frozen
   - condition latent UNet with SupCon/DINO EEG latent

5. `Exp35 = S18 vs S01 latent-generation diagnosis`
   - purpose: explain why Exp33 S18 works while S01 collapses under the same frozen-SupCon VAE-latent setup
   - no new heavy model training required
   - required checks:
     - per-class trial counts
     - session-wise class distribution
     - train/val/test class balance
     - generated confusion matrix
     - dominant generated class by subject
   - output: decide whether S01 collapse is caused by data/split/session imbalance or by conditioning architecture

6. `Exp36 = S01 low-cost recovery ablation`
   - run only after Exp35
   - first option: class-balanced sampling if Exp35 finds imbalance/session skew
   - second option: longer S01 training if Exp35 suggests under-training
   - keep frozen SupCon + VAE latent setup fixed
   - success: S01 dominant decreases and entropy increases without DINO@1 collapse

7. `Exp37 = cross-attention latent generator`
   - run only if Exp35/36 cannot explain or reduce S01 collapse
   - purpose: replace FiLM/vector conditioning with cross-attention over EEG condition tokens
   - do not start before Exp35 diagnosis is summarized

8. `Exp38 = Gen-WS-03 first run`
   - subject: `S18`
   - model: frozen-SupCon + SD VAE latent cross-attention generator
   - comparisons: `C0 VI scratch`, `C1 VS-pretrained frozen-encoder VI fine-tune`, optional `C2 staged unfreeze`
   - first report must include DINO@1/3/5, entropy, dominant class ratio, confusion matrix, and C1-vs-C0 delta

9. `Exp39 = multi-image retrieval gallery evaluation`
   - completed
   - purpose: make DINO@1 evaluation less brittle than the current 9-image gallery
   - current gallery: `9 classes x 1 image`
   - target gallery: `9 classes x 10-20 images`, using ImageNet/COCO or a curated equivalent set
   - evaluation rule:
     - generate image from EEG
     - retrieve nearest image from the expanded DINO gallery
     - count correct if the retrieved image's class matches the target class
   - required output:
     - class-level Top-1/Top-3/Top-5
     - image-level nearest-neighbor examples
     - comparison against old 9-image gallery
   - interpretation:
     - this is an evaluation upgrade first, not a training change
     - use it to check whether current DINO@1 is inflated or deflated by the single-template gallery
   - result:
     - expanded gallery gives a more conservative and reliable full-test estimate
     - S18 dual-gallery DINO@1 around `0.2315`
     - single-image stochastic/sample-grid numbers should not be treated as final full-test performance

10. `Exp40 = multi-image target training`
   - completed
   - purpose: reduce the generator's shortcut of memorizing one fixed class image
   - replace the fixed one-image-per-class target with multiple images per class
   - training target should sample one class-compatible image per trial or use class prototype / multi-positive visual targets
   - especially important for VI because imagined images may not match the exact stimulus template
   - success criteria:
     - generated images remain class-consistent under expanded-gallery evaluation
     - diversity improves without dominant-class collapse
     - DINO class accuracy does not depend on one canonical image per class
   - result:
     - multi-image targets degraded performance under expanded-gallery evaluation
     - current best remains single-image target + frozen SupCon + VAE latent cross-attention
     - do not continue multi-image target training in the current architecture unless the image prior is changed

11. `Exp41 = SD 1.5 LoRA generator`
   - next recommended structural step
   - reason:
     - Exp33/37/40 converge near the same full-test dual-gallery ceiling
     - current small latent UNet still produces foggy/smoky images and weak fine-detail classes
     - SD 1.5 provides a much stronger pretrained image prior than the scratch latent UNet
   - structure:
     - SD 1.5 UNet mostly frozen
     - train LoRA layers only
     - EEG/SupCon latent projected into SD conditioning space
     - SD VAE remains frozen
   - first target:
     - within-subject VS generation on S18
     - compare against Exp37/39 dual-gallery baseline
   - success criteria:
     - sharper class-identifiable images
     - dual-gallery DINO@1 improves over current S18 `0.2315`
     - dominant class ratio does not increase
   - caution:
     - keep EEG-only conditioning for the main result
     - class/text prompts may be used only as an upper-bound diagnostic, not as the main claim

12. `Optional low-cost sampling checks before or during Exp41`
   - CFG/guidance sweep only if the implementation has a real unconditional path
   - do not over-invest in sampler-only tuning if guidance is not properly trained
   - 256x256 can improve spatial detail, but it increases compute and does not solve weak EEG conditioning by itself
   - recoloring/replacing black-background symbolic targets is a dataset-design option, not the main model fix

### Exp15+ interpretation rule

From `Exp15` onward, generation experiments must be interpreted as:
- `prototype-guided generation validation`
- not `final image-generation success`
- not `strict class-accuracy optimization`

This means:
- first evaluate whether generated images move closer to the `true-label prototype`
- then evaluate whether collapse is reduced
- only after that should exact `Top-1` improvement be emphasized

Recommended Exp15+ order:
1. `prototype attraction`
2. `non-target prototype separation`
3. `checkpoint selection update`
4. `true-label-conditioned upper-bound diagnostic`
5. `CLIP-vs-DINO image-space comparison for generation targets`

### Exp20+ renumbered order

Use the following order from this point forward:

1. `Exp20 = CLIP image comparison`
   - run `train_vs_re_clip.py`
   - compare `CLIP image embedding` against the current `DINO` representation baseline
   - do not use generation checkpoints as the primary comparison baseline

2. `Exp21 = Stage 2 latent/readout`
   - choose the better image latent space from `Exp20`
   - start with nearest-neighbor retrieval / latent readout
   - only then consider a lightweight decoder

3. `Exp22 = TTA evaluation`
   - no retraining
   - use the best current DINO encoder baseline
   - evaluate DINO retrieval first
   - re-run Stage 2 readout if retrieval improves

4. `Exp23-A/B/C = SupCon ablation`
   - `Exp23-A`: current InfoNCE-style baseline reproduction
   - `Exp23-B`: `SupCon only`
   - `Exp23-C`: `SupCon + proto`
   - keep subject set and split fixed for clean comparison

5. `Exp24-A/B/C = preprocessing ablation`
   - apply to the best SupCon config
   - `Exp24-A`: baseline correction only
   - `Exp24-B`: channel-wise z-score only
   - `Exp24-C`: baseline correction + channel-wise z-score only if A or B is promising
   - use `-1~0 sec` baseline where available
   - keep bandpass as a later ablation, not the first preprocessing change

6. `Exp25 = best encoder Stage 2 readout`
   - freeze the best encoder from `Exp22~24`
   - re-run latent/readout evaluation
   - compare directly against `Exp21`

7. `Exp25-VI = VI transfer readiness probe`
   - run immediately after or in parallel with `Exp25`
   - first candidate: best `Exp23-B` or best post-Exp24 encoder
   - use existing `train_vs_test_vi.py` if sufficient
   - if the existing script cannot evaluate the DINO/SupCon encoder, add a minimal zero-shot retrieval/readout helper instead of training a new generator
   - report failure cases explicitly

8. `Exp26 = all-subject SupCon validation`
   - run best SupCon config on all available `preproc_vs_re` subjects
   - summarize high-session vs low-session subjects separately
   - use this before making broad claims about SupCon

9. `Exp27-preA = S23 data integrity check`
   - run before partial-session integration
   - use `diagnose_vsre_subject_data.py --subject_ids 23`
   - if S23 is invalid, exclude or quarantine it from aggregate interpretation

10. `Exp27-preB = VS/VI latent gap diagnosis`
   - run before partial-session integration
   - use `diagnose_vs_vi_latent_gap.py --subject_ids 1,2,18`
   - compare VS vs VI true-label cosine, best-wrong cosine, and margin
   - decide whether partial-session VS optimization is still justified

11. `Exp27 = targeted partial-session integration`
   - run only after `Exp27-preA` and `Exp27-preB`
   - do not hard-code `S02/S19/S28` without checking Exp26 individual results
   - select subjects by:
     - low current Top-1
     - meaningful additional valid sessions/trials
     - no obvious data-integrity issue
   - current candidates to re-evaluate:
     - `S02`
     - `S19`
     - `S28`
     - any subject with new partial-session files and low Exp26 Top-1
   - compare `old data only` vs `old + partial-session files`
   - use effective loaded trial/session counts, not nominal file/session assumptions

12. `Exp28-A = VI temporal-resolution matching`
   - run before any resumed VS-only partial-session optimization
   - use `eval_vs_re_exp25_vi_transfer.py --target_time 2048`
   - compare against original Exp25-VI Top-1 `0.1296`
   - if Top-1 rises above about `0.15`, temporal mismatch is a major factor
   - if it remains near chance, prioritize VI fine-tuning / domain adaptation
   - completed result: Top-1 dropped to `0.0926`; simple VI upsampling is not the fix

13. `Exp28-B = VI adaptation if Exp28-A fails`
   - completed initial B0/B1/B2 comparison shows weak VI transfer, but the cause is not fully isolated
   - run the required three-way comparison on the same VI split:
     - `Exp28-B0`: random-init VI scratch baseline
     - `Exp28-B1`: initialize from the best VS SupCon encoder and directly fine-tune on VI
     - `Exp28-B2`: freeze the VS encoder and train a VI linear probe
     - `Exp28-B3`: VS+VI joint SupCon training
   - rationale:
     - B0 is mandatory because it is the only clean baseline for whether VS pretraining helps
     - B1 gives the clearest answer to whether VS pretraining helps VI adaptation
     - B2 must run alongside B1 to distinguish useful frozen VS latents from fine-tuning overwrite
     - B3 is expensive and has data-imbalance/domain-confound risk
   - initial subjects: `S01/S02/S18`
   - recommended schedule: B0/B1 `ep100` with early stopping, B2 `ep20`
   - compare against zero-shot VI transfer and against B0 scratch
   - do not include S08 in validation-dependent adaptation runs unless split handling is fixed, because the audit found `vs,S08,val=0`

14. `Exp30-A = native VI-512 scratch encoder`
   - this is the next main direction after Exp28-B
   - purpose: isolate whether B0 failure came from using a VS/2048-oriented encoder on VI 512-sample input
   - do not use naive `512 -> 2048` upsampling as the only temporal test
   - modify only the input temporal handling / projection needed to learn directly from VI `512` samples
   - run VI scratch first on `S01/S02/S18` with the same split as Exp28-B
   - decision rule:
     - if native B0-512 rises clearly above random, prioritize VI-aware architecture and data-size/fine-tuning fixes
     - if native B0-512 remains near or below random, treat paradigm/domain gap as the stronger bottleneck
   - hold `Exp28-B3` joint SupCon until this result is known

15. `Exp29 = VI-relevant low-session outlier check`
   - demoted to side check
   - do not analyze S11/S16 only as VS outliers
   - reframe the question as whether S11/S16 also perform unusually well on VI or VS->VI transfer
   - if they are high in both VS and VI, inspect subject-specific signal/channel statistics

16. `Exp30+ = optional pixel-generator follow-up`
   - only if Stage 2 still leaves a strong reason to continue pixel-space generation
   - examples:
     - proper CFG
     - cross-attention conditioning
     - decoder/generator migration
   - do not run before `Exp25-VI`

### Block A: DINOv2-centered pretraining

- Dataset: `preproc_vs_re`
- Subjects: start with `1,2,18`
- Primary metrics:
  - `DINO Top-1`
  - `DINO Top-3`
  - subject stability
- Secondary metrics:
  - classical classification sanity baseline
  - generation probe

Preferred ablation order:
1. `TTA`
2. `SupCon`
3. `baseline correction`
4. `channel-wise z-score`
5. optional `bandpass`

Additional dataset rule:
- every run must print effective `loaded trials` and `loaded sessions` per subject
- when comparing old vs new data, explicitly separate:
  - `legacy valid files only`
  - `legacy + partial-session files`

### Block B: Generation collapse diagnostic

Required for every generation run:
- generated DINO-predicted class histogram
- generated-vs-true confusion matrix
- predicted-class entropy
- per-subject collapse summary
- true-label prototype similarity
- true-vs-best-wrong prototype margin

Required comparison:
- current checkpoint baseline
- anti-collapse auxiliary loss enabled
- prototype-similarity loss enabled
- same sampler and hyperparameters unless the sampler itself is being tested

Exp15+ success order:
1. `true-label prototype similarity` improves
2. `true-vs-best-wrong prototype margin` improves
3. `collapse` decreases (`entropy up`, dominant fraction down)
4. `Top-1 / Top-3 / Top-5` improve if possible
5. `SSIM / LPIPS / val_loss` are treated as secondary stability checks

### Block C: Minimal encoder comparison

Required subset comparison:
1. `V1`
2. `V2 + auto prior`
3. `V2 + no prior`

Keep fixed unless explicitly testing another factor:
- `--beta_schedule linear`
- `--lambda_lpips 0.0`
- `--lambda_percept 0.1`
- `--lambda_rec 0.01`
- `--guidance_scale 1.5`
- `--eta 0.0`
- `--eval_ddim_steps 50`

### Block D: CLIP image-space comparison

- compare against DINOv2, not against text conditioning
- keep the comparison representation-focused

### Block E: Stage 2 latent/readout check

Required order:
1. `EEG -> representation latent`
2. latent-level prediction or retrieval
3. nearest-neighbor image readout
4. only then optional decoder

Do not start from pixel decoding in this block.

### Block F: Partial-session data integration check

Required checks:
- confirm loader works on partial files of different sizes
- confirm split logic remains class-balanced after partial loading
- confirm subject-level trial counts match actual loaded tensors
- compare metrics before vs after adding the new partial-session files

### Block G: Classical sanity line

Use `FBCSP + SVM` only as:
- sanity baseline
- signal-presence check

Do not optimize the project around FBCSP itself.

---

## 7. Stop Criteria

- Stop a config if:
  - encoder-side metrics do not improve at all
  - generated images do not get closer to the true-label prototype
  - true-vs-best-wrong prototype margin does not improve
  - generated-class histogram collapses to one class without improving identity metrics
  - generation probe gives `mean DINO Top-1 < 0.1160`
  - generation probe gives `mean SSIM < 0`
  - generation probe gives `mean best_val_loss > 0.005`

- Stop generator-heavy work if:
  - retrieval/classification still indicate weak representation
  - only visual sharpness changes without identity improvement

---

## 8. Required Outputs

- Update files:
  - `MEMORY_LOG.md`
  - `PROGRESS.md`
  - `VSRE_PROGRESS_SUMMARY.md` if the summary materially changes

- Must report:
  - VS retrieval metrics
  - VS classification sanity results
  - generation probe metrics
  - true-label prototype similarity
  - true-vs-best-wrong prototype margin
  - effective loaded trial count per subject
  - effective loaded session count per subject
  - exact encoder config
  - whether the change improves transfer-relevant representation quality

- Must compare against:
  - full-subject generation baseline `[20260411_165536]`
  - V1 vs V2 subset comparisons
  - FBCSP+SVM sanity baseline when relevant

---

## 9. Reflection Questions

- Did the encoder improve transfer-relevant representation quality?
- Is DINOv2 clearly better than CLIP image space here, or not yet?
- Is generation still useful as a probe, or mostly redundant?
- Is the project ready to connect EEG representation to SD 1.5 latent conditioning?

---

## 10. Free Notes

- Current evidence indicates the main bottleneck is encoder / representation quality, not pure generator capacity.
- Weak generation does not automatically justify more generator training.
- For generation, semantic similarity to the true-label prototype is a higher-priority milestone than exact class accuracy.
- From `Exp15` onward, generation runs should be judged as `prototype-guided validation`, not as full image-generation success.
- The most practical short-term route around generation collapse is `Stage 2 latent/readout`, not deeper pixel-space tuning.
- `TTA` and `SupCon` are high-priority low-cost improvements on the representation track.
- CLIP should first be used as an image embedding comparison target, not as text-vector supervision.
- The next cycle should optimize for `VS pretraining -> VI transfer readiness`, not image sharpness alone.
