# VS/VI EEG-to-Image Project

Last updated: 2026-06-17

This repository contains the current VS/VI EEG-to-image generation project.
The active path is within-subject `VS -> VI` transfer using an SD 1.5 LoRA generator conditioned on EEG/SupCon features.

## Current Achievement

The project has reached the first fully generative result that exceeds the non-generative Stage 2 retrieval baseline.

| Method | Subject | Setting | Dual-gallery DINO@1 | Note |
|--------|---------|---------|---------------------|------|
| Stage 2 retrieval | reference | non-generative | `0.3333` | nearest/readout baseline |
| Exp41 SD 1.5 LoRA | S18 | r=16, 8 tokens | `0.2870` | first strong SD prior result |
| Exp42-A SD 1.5 LoRA | S01 | r=16, 8 tokens | `0.3333` | matches retrieval baseline |
| Exp42-B Step4 SD 1.5 LoRA | S01 | r=32, 16 tokens | `0.3571` | current project best |
| Exp42-B Step5 augmentation | S18 | r=16, 16 tokens + target aug | internal `0.4464` | partial-test/internal metric; full-test dual-gallery still needed |

Main interpretation:

- SD 1.5 LoRA is the current best generator.
- S01 r=32, 16-token LoRA is the current best generative checkpoint.
- Generation now exceeds the Stage 2 retrieval reference for S01.
- LoRA rank is subject-dependent: S18 currently prefers r=16, while S01 benefits from r=32.
- VI fine-tuning should not force one global rank; include r=32 because VI EEG is expected to be weaker than VS EEG.
- Exp42-B Step5 S18 target augmentation finished. It should be treated as preliminary until evaluated with the same full-test dual-gallery protocol used for Exp41/42 comparisons.

## Experiment History Policy

Keep as much experiment history as possible when moving or continuing this project.

- `PROGRESS.md` is intentionally cumulative. Do not delete older Exp sections just because the active direction changed.
- New results should be appended or summarized at the top, while the original Exp notes remain available below for traceback.
- If an old result becomes obsolete, mark it as obsolete and explain why; do not remove it.
- Preserve failed runs, negative results, and abandoned branches because they explain why the current SD LoRA path was chosen.
- The current active baseline starts after `Exp42-B Step4`, but interpretation depends on the full chain from Exp13 collapse through Exp41/42 SD LoRA.

## Active Data

Use these repeated-session datasets for the current phase:

| Dataset | Path | Subjects | Total | Train | Val | Test | Note |
|---------|------|----------|-------|-------|-----|------|------|
| VS | `preproc_vs_re` | 21 | 9,207 | 7,335 | 837 | 1,035 | S01 session 9 skipped by current loader due anomalous `n_rep=108` |
| VI | `preproc_vi_re` | 20 | 9,315 | 7,452 | 882 | 981 | same repeated-session style as VS; S08 absent |

High-priority overlap subjects:

| Subject | VS trials | VI trials | Why important |
|---------|-----------|-----------|---------------|
| S01 | 1,215 | 1,215 | current best generator, r=32 |
| S18 | 1,080 | 1,080 | original LoRA benchmark, r=16 |
| S24 | 1,350 | 1,350 | largest VS/VI overlap subject |

## Important Files

Core documentation:

- `PROGRESS.md`: full experiment log and current status summary.
- `HUMAN_DIRECTIVE.md`: active experiment priorities and decision rules.
- `AGENT_LOG.md`: agent-side operational log for backup, verification, and handoff notes.
- `DATA_LOG.md`: dataset notes.

Core scripts:

- `dataset_vs_re.py`: repeated-session VS loader. It currently skips anomalous `n_rep > 20` VS files.
- `train_vs_re_dino.py`: EEG-DINO/SupCon representation training.
- `train_vs_re_lora_gen.py`: active SD 1.5 LoRA VS generator.
- `train_vi_latent_gen.py`: older VI latent generator baseline.
- `eval_exp39_multigallery.py`: full-test multi-gallery evaluation.

Current important checkpoint roots:

- `checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon`: best SupCon representation checkpoint.
- `checkpoints_vsre_lora_gen`: SD 1.5 LoRA generator checkpoints.
- `checkpoints_vsre_latent_gen/20260610_105755_ch32_ep300_supcon_frozen_ca`: older VAE-latent CA baseline.

## Environment

The working conda environment used in this project is named `eegdiff`.

Typical command prefix on Windows:

```powershell
conda run -n eegdiff --no-capture-output python <script>.py <args>
```

If `conda` is not visible in PowerShell, use the full Python path:

```powershell
C:\Users\Biocomputing\anaconda3\envs\eegdiff\python.exe <script>.py <args>
```

Main dependencies:

- PyTorch / torchvision
- diffusers
- peft
- scipy
- h5py
- numpy
- PIL / Pillow
- DINOv2 through torch hub cache

SD VAE model:

- `stabilityai/sd-vae-ft-mse`

If a new machine does not have cached Hugging Face / torch hub weights, the first run may need internet access.

## Current Next Steps

### 1. S24 SD LoRA VS generation

Run S24 with both r=16 and r=32 if compute allows.

```powershell
conda run -n eegdiff --no-capture-output python .\train_vs_re_lora_gen.py --subject_ids 24 --epochs 100 --lora_r 16 --n_eeg_tokens 16 *> .\logs\exp_s24_lora_r16_tok16.log

conda run -n eegdiff --no-capture-output python .\train_vs_re_lora_gen.py --subject_ids 24 --epochs 100 --lora_r 32 --n_eeg_tokens 16 *> .\logs\exp_s24_lora_r32_tok16.log
```

Decision rule:

- `S24 > S01`: session/data volume is a major driver.
- `S24 ~= S01`: SD LoRA may be near practical ceiling once data is sufficient.
- `S24 < S01`: subject-specific EEG signal quality matters more than data volume.

### 2. Exp43 SD LoRA VS-to-VI fine-tuning

Do not wait for Exp42-B Step5 augmentation. Start with the best current checkpoints.

Initial recommended setup:

- S01 init: Exp42-B Step4 r=32, 16-token checkpoint.
- S18 init: Exp42-B Step2 r=16, 16-token checkpoint.
- C0: VI scratch LoRA.
- C1: VS LoRA -> VI fine-tune with frozen EEG encoder.
- C2: staged encoder unfreeze if C1 does not improve by about epoch 50.

Success criteria:

- Minimum: `C1 > C0`.
- Meaningful: VI DINO@1 `> 0.20`.
- Collapse lower than VI scratch.

### 3. Exp42-B Step5 augmentation

S18 Step5 has been run once.

- Checkpoint: `checkpoints_vsre_lora_gen/20260617_124018_lora_r16_ep100`
- Log: `logs/exp42b_step5_aug_s18.log`
- Internal partial-test score: DINO@1 `0.4464`, entropy `1.960`, dominant `25.0%`, best epoch `95`
- Caveat: this is the training script's internal partial-test/prototype score, not the final full-test dual-gallery score.

Further augmentation should run only after S24 / Exp43 are underway.

Purpose:

- Secondary cleanup for heart/star/symbol confusion.
- Not the main bottleneck after S01 r=32 exceeded retrieval baseline.

## Current Non-Priorities

Do not spend primary compute on these unless the hypothesis changes:

- Exp31~40 scratch latent/pixel UNet path.
- Multi-image target training with visually inconsistent alternatives.
- CFG sweep without a real trained unconditional path.
- Single-template or stochastic sample-grid-only results as final metrics.

## Backup Notes

This project can be moved to another machine by copying this repository directory plus the data/checkpoint directories.

Current detected backup path on this machine:

- `E:\vsvi_project`

Earlier notes may mention `D:\vsvi_project`; Windows drive letters can change depending on mounted devices. Use whichever path actually contains `README.md`, `PROGRESS.md`, `preproc_vs_re`, and `preproc_vi_re`.

- `preproc_vs_re`
- `preproc_vi_re`
- `preproc_data_vi/images`
- `checkpoints_vsre_dino`
- `checkpoints_vsre_lora_gen`
- `logs`
- `PROGRESS.md`
- `HUMAN_DIRECTIVE.md`
- `AGENT_LOG.md`
- `README.md`

After copying, verify:

```powershell
conda run -n eegdiff --no-capture-output python .\dataset_vs_re.py .\preproc_vs_re
conda run -n eegdiff --no-capture-output python .\train_vs_re_lora_gen.py --subject_ids 1 --epochs 1 --lora_r 32 --n_eeg_tokens 16
```
