# human_directive.md

Last Updated: 2026-06-29
Owner: H Kim
Status: active handoff for next computation

---

## Current human decision

CSV generation is deferred. Do **not** spend the next compute cycle on generating `results_lora_gen.csv` or updating `PROGRESS.md`.

The immediate next computation is the verified S24 Track-A SD1.5 LoRA run with the correct S24 SupCon encoder, continuing the rank ablation after the r=16 run produced a best checkpoint.

---

## Verified current state

The following Colab/Drive artifacts were observed by the human operator:

- Valid intermediate S24 r=16 checkpoint exists:
  - `/content/drive/MyDrive/vsvi_data/checkpoints_vsre_lora_gen/20260628_225108_lora_r16_ep100/subj24_lora_best.pt`
  - size approximately 63 MB
  - timestamp around `2026-06-29 00:37`
- Test-only checkpoint exists and must not be used for reporting:
  - `/content/drive/MyDrive/vsvi_data/checkpoints_vsre_lora_gen/20260628_224755_lora_r16_ep2/subj24_lora_best.pt`
- `results_lora_gen.csv` is not yet generated for the r=16 ep100 run.
- The missing CSV is accepted for now; evaluation-from-best can be done later.
- The old S24 checkpoints from `20260625_111012` and `20260626_073002` must remain invalid/provenance-uncertain unless their checkpoint metadata proves otherwise.

---

## Next computation to implement/run

Run the next S24 Track-A LoRA rank ablation:

```bash
python -u train_vs_re_lora_gen.py \
  --subject_ids 24 \
  --lora_r 32 \
  --n_eeg_tokens 16 \
  --epochs 100 \
  --batch_size 2 \
  --img_root preproc_data_vi/images \
  --supcon_ckpt checkpoints_vsre_dino/20260604_091352_ch32_merged_ep200_supcon \
  --ckpt_root /content/drive/MyDrive/vsvi_data/checkpoints_vsre_lora_gen \
  --fp16
```

A100 may support `--batch_size 4`, but prefer `2` if runtime stability is more important than speed. T4/L4 should use `--batch_size 2`; add `--grad_ckpt` only if memory pressure appears.

---

## Required implementation changes before running

### 1. Use Drive absolute checkpoint root

Do not rely on a fragile symlink for the primary output path. The effective `CKPT_ROOT` for Colab Track A must be:

```python
CKPT_ROOT = "/content/drive/MyDrive/vsvi_data/checkpoints_vsre_lora_gen"
```

The repo-local `checkpoints_vsre_lora_gen` symlink may exist for convenience, but the training command must pass the Drive absolute path to `--ckpt_root`.

### 2. Prevent duplicate background runs

Before launching a new training process, check for existing jobs:

```bash
pgrep -af train_vs_re_lora_gen
nvidia-smi
```

If an older duplicate process is running, stop it before launching the new rank-32 run. Do not run r=16 and r=32 simultaneously in the same Colab runtime.

### 3. Preserve logs and PID in Drive

The launch helper must write:

- log file under `/content/drive/MyDrive/vsvi_data/logs/`
- PID file under `/content/drive/MyDrive/vsvi_data/logs/`

Recommended names:

```text
s24_lora_r32_tok16_<timestamp>.log
s24_lora_r32_latest.pid
```

### 4. Patch the torchao/PEFT conflict in fresh Colab runtimes

Fresh Colab runtimes may contain `torchao==0.10.0`, while recent `peft` may require a newer torchao path and fail during LoRA injection. For this project, `torchao` is not required. The Colab setup cell should remove it before importing PEFT:

```bash
pip uninstall -y torchao
```

Then verify:

```bash
python -c "import importlib.util, peft; print(importlib.util.find_spec('torchao') is None); print(peft.__version__)"
```

The expected condition is `torchao` absent or PEFT patched so that `is_torchao_available()` returns `False`, not an ImportError.

### 5. Confirm early checkpoint creation

After epoch 1 finishes, verify that the new r=32 run creates:

```bash
find /content/drive/MyDrive/vsvi_data/checkpoints_vsre_lora_gen \
  -path '*lora_r32_ep100/subj24_lora_best.pt' -ls
```

If no best checkpoint appears after epoch 1, stop the run and fix the save path before continuing.

---

## Do not do now

- Do not generate `results_lora_gen.csv` yet.
- Do not run Colab cell `[8]` or update `PROGRESS.md` yet.
- Do not evaluate/report S24 r=16 or r=32 until the corresponding CSV exists.
- Do not use the `ep2` test checkpoint for any scientific comparison.
- Do not treat old S24 r=16/r=32 checkpoints as valid unless provenance confirms the correct SupCon path.

---

## Later, after both r=16 and r=32 are available

Generate CSVs from the best checkpoints, then update `PROGRESS.md` with:

- S24 r=16 DINO@1 / entropy / dominant / best epoch
- S24 r=32 DINO@1 / entropy / dominant / best epoch
- explicit checkpoint directory names
- interpretation against S01 and S18:
  - S01 best: r=32, DINO@1 0.3571
  - S18 best: r=16, DINO@1 0.2963
  - S24 result determines whether session count or subject-specific EEG quality dominates

---

## Acceptance criteria for this directive

The next compute implementation is acceptable only if all of the following are true:

1. The training command shows `--lora_r 32` and `--n_eeg_tokens 16`.
2. The command uses the correct S24 SupCon directory:
   `checkpoints_vsre_dino/20260604_091352_ch32_merged_ep200_supcon`.
3. The command writes checkpoints directly to:
   `/content/drive/MyDrive/vsvi_data/checkpoints_vsre_lora_gen`.
4. Only one `train_vs_re_lora_gen.py` process is active.
5. `nvidia-smi` shows the active process using the selected GPU.
6. A new `subj24_lora_best.pt` appears under a `lora_r32_ep100` directory after the first successful epoch.
7. No `PROGRESS.md` update is committed until `results_lora_gen.csv` exists.
