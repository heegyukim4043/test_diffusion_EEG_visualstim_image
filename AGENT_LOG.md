# Agent Log

Last updated: 2026-06-17

This file records agent-side operational actions so another environment or agent can understand what was changed, verified, and backed up.

Use this file for:

- documentation edits
- backup/sync actions
- metric interpretation caveats
- handoff notes between agents

Use `PROGRESS.md` for experiment results and scientific interpretation.
Use `HUMAN_DIRECTIVE.md` for active priorities and decision rules.

---

## 2026-06-17 Handoff / Backup Preparation

### Project Backup

Created and verified backup. The intended path was:

- `D:\vsvi_project`

Current detected path in the latest check:

- `E:\vsvi_project`

Note: Windows drive letters can change depending on mounted media. If `D:\vsvi_project` is missing, check `E:\vsvi_project` or search top-level drives for `vsvi_project`.

Backup includes the current project code, documentation, logs, data folders, and checkpoint folders required to continue the current VS/VI EEG-to-image work.

Key backup files:

- `D:\vsvi_project\README.md`
- `D:\vsvi_project\PROGRESS.md`
- `D:\vsvi_project\HUMAN_DIRECTIVE.md`
- `D:\vsvi_project\AGENT_LOG.md`

Key backup directories:

- `D:\vsvi_project\preproc_vs_re`
- `D:\vsvi_project\preproc_vi_re`
- `D:\vsvi_project\checkpoints_vsre_lora_gen`
- `D:\vsvi_project\checkpoints_vsre_dino`
- `D:\vsvi_project\logs`

Main backup log:

- `D:\vsvi_project_backup_20260617.log`

### Documentation Updated

Updated:

- `README.md`
- `PROGRESS.md`
- `HUMAN_DIRECTIVE.md`

Main content added:

- Current achievement summary after `Exp42-B Step4`.
- Current best result: S01 SD LoRA `r=32`, `16` EEG tokens, dual-gallery DINO@1 `0.3571`.
- Current S18 reference: SD LoRA `r=16`, `16` EEG tokens, dual-gallery DINO@1 `0.2963`.
- Next priority: S24 SD LoRA `r=16/r=32`.
- Next priority: `Exp43` SD LoRA-based VS -> VI fine-tuning.
- Experiment history preservation policy: do not delete old Exp records; mark obsolete results instead.

### Step 5 S18 Result Check

Checked:

- `logs\exp42b_step5_aug_s18.log`
- `checkpoints_vsre_lora_gen\20260617_124018_lora_r16_ep100`

Observed internal training-script result:

| Metric | Value |
|--------|-------|
| Subject | S18 |
| Config | LoRA `r=16`, `16` EEG tokens, class-preserving target augmentation |
| Best epoch | `95` |
| Internal DINO@1 | `0.4464` |
| Internal DINO@3 | `0.5714` |
| Internal DINO@5 | `0.6250` |
| Entropy | `1.960` |
| Dominant class ratio | `25.0%` |

Important caveat:

- This `0.4464` is from `train_vs_re_lora_gen.py` internal evaluation.
- That function uses the default partial-test setting (`n_samples=54`; actual sample count can exceed this by batch boundary).
- It is not directly comparable to the current final metric standard.
- The final standard remains full-test dual-gallery DINO retrieval.

Documentation decision:

- Step5 S18 is marked as completed and promising internally.
- It is not promoted above Step2/Step4 until matching full-test dual-gallery evaluation exists.
- Further augmentation is secondary after S24 and Exp43 start.

### Current Resume Point

Resume from:

- After `Exp42-B Step4`
- With `Exp42-B Step5 S18` logged as an auxiliary/preliminary augmentation result

Immediate next recommended compute:

1. `S24 SD LoRA VS generation`
   - run `r=16`
   - run `r=32` if compute allows

2. `Exp43 SD LoRA VS -> VI fine-tuning`
   - C0: VI scratch LoRA
   - C1: VS LoRA -> VI fine-tune with frozen EEG encoder
   - C2: staged encoder unfreeze if C1 does not improve by about epoch 50

Recommended initialization:

- S01: Exp42-B Step4 checkpoint, `r=32`, `16` tokens.
- S18: Exp42-B Step2 checkpoint, `r=16`, `16` tokens.

### Metric Policy

Use these labels clearly:

- `internal`: training-script partial/prototype metric.
- `single-gallery`: one target image per class.
- `dual-gallery`: original + alternative class image gallery.
- `full-test dual-gallery`: current final comparison standard.

Do not compare `internal` scores directly against `full-test dual-gallery` scores.

### Known Encoding Note

Some older sections of `PROGRESS.md` contain broken Korean text due prior encoding issues. Preserve them rather than deleting them. Add corrected summaries near the top when needed.
