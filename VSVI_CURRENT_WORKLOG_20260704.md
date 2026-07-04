# VS/VI SD-LoRA Current Worklog

**Date:** 2026-07-04  
**Project:** EEG-to-image generation using SD1.5 LoRA with VS/VI datasets  
**Repository:** `heegyukim4043/test_diffusion_EEG_visualstim_image`

This file summarizes the current computation state, subject-level model availability, active runs, and the next execution queue.

---

## 1. Current active run

S01 Exp43 VI transfer is currently running on Colab A100.

```text
PID = 303092
LOG = /content/drive/MyDrive/vsvi_data/logs/exp43_s01_c0c1_r32_tok16_20260704_142634.log
GPU = NVIDIA A100-SXM4-40GB
Observed GPU memory = ~8.3 GB
```

Command currently running:

```bash
python3 -u train_exp43_vi_lora.py \
  --subject_ids 1 \
  --conditions c0,c1 \
  --data_root preproc_vi_re \
  --img_root preproc_data_vi/images \
  --supcon_ckpt checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon \
  --init_lora_ckpt /content/drive/MyDrive/vsvi_data/checkpoints_vsre_lora_gen/20260617_074245_lora_r32_ep100/subj01_lora_best.pt \
  --ckpt_root /content/drive/MyDrive/vsvi_data/checkpoints_exp43_vi_lora \
  --lora_r 32 \
  --n_eeg_tokens 16 \
  --epochs 100 \
  --batch_size 2 \
  --per_class_total 60 \
  --eval_n_samples 54 \
  --fp16
```

Observed normal startup log:

```text
[INFO] Device: cuda  Exp43 VI LoRA
[INFO] Loading DINO...
[INFO] Loading SD VAE...
[INFO] Class target latents: (9, 4, 64, 64)
[INFO] S01 train=432 val=54 test=54

======================================================================
  Exp43 C0  S01
  [Encoder] Loaded and frozen: checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon/subj01_best.pt
  Loading SD 1.5 UNet...
trainable params: 6,377,472 || all params: 865,898,436 || trainable%: 0.7365
  [Init] C0 scratch VI LoRA (no VS LoRA initialization)
```

The warnings about xFormers, HF_TOKEN, Flax deprecation, and AMP deprecation are not fatal for the current run.

Monitoring command:

```bash
pgrep -af 'train_exp43_vi_lora.py|train_vs_re_lora_gen.py' || echo "No training process"
nvidia-smi
tail -n 120 /content/drive/MyDrive/vsvi_data/logs/exp43_s01_c0c1_r32_tok16_20260704_142634.log
```

Completion/error check:

```bash
LOG=/content/drive/MyDrive/vsvi_data/logs/exp43_s01_c0c1_r32_tok16_20260704_142634.log
grep -nE "Done Exp43|Saved run summary|Traceback|RuntimeError|CUDA|Killed|OutOfMemory|Error" "$LOG" | tail -50
```

Do not launch another GPU training job while PID `303092` is active.

---

## 2. Implemented helper script

A VS model audit and missing-run helper was added.

```text
File: audit_launch_vs_lora_subjects.py
Commit: 84819b3dc943ac6da1cc2323fdfd2940cfcefb88
```

Purpose:

- Check whether each planned subject has a VS SD1.5 LoRA checkpoint.
- Report whether result CSV metrics are available.
- Detect whether VS data and SupCon checkpoints exist.
- Launch exactly one missing VS LoRA run at a time when data and SupCon are available.

Default target subject set:

```text
S24, S01, S02, S18, S28, S29, S09
```

Default planned rank policy:

```text
S24: r32
S01: r32
S02: r32
S18: r16
S28: r32
S29: r32
S09: r32
```

Audit command:

```bash
cd /content/vsvi_project
python -u audit_launch_vs_lora_subjects.py --save_csv
```

Audit CSV output:

```text
/content/drive/MyDrive/vsvi_data/audits/vs_lora_model_audit_*.csv
```

Launch one missing VS model only when status is `MISSING_CKPT`:

```bash
cd /content/vsvi_project
python -u audit_launch_vs_lora_subjects.py --launch_next_missing --save_csv
```

Force one subject, dry run first:

```bash
cd /content/vsvi_project
python -u audit_launch_vs_lora_subjects.py \
  --subject_ids 29 \
  --rank_policy r32 \
  --force_sid 29 \
  --save_csv \
  --dry_run
```

---

## 3. Latest VS model audit status

Latest confirmed audit output:

```text
order sid rank status        data   supcon top1         best_ep ckpt
---------------------------------------------------------------------
1     S24 r=32 COMPLETE      True   True   0.4629629630 98      $DRIVE/checkpoints_vsre_lora_gen/20260629_094904_lora_r32_ep100/subj24_lora_best.pt
2     S01 r=32 CKPT_ONLY     True   True   -            -       $DRIVE/checkpoints_vsre_lora_gen/20260617_074245_lora_r32_ep100/subj01_lora_best.pt
3     S02 r=32 BLOCKED_SUPCON True  False  -            -       
4     S18 r=16 CKPT_ONLY     True   True   -            -       $DRIVE/checkpoints_vsre_lora_gen/20260612_010728_lora_r16_ep100/subj18_lora_best.pt
5     S28 r=32 BLOCKED_DATA  False  False  -            -       
6     S29 r=32 BLOCKED_DATA  False  False  -            -       
7     S09 r=32 BLOCKED_SUPCON True  False  -            -       
```

Interpretation:

- `COMPLETE`: VS checkpoint and `results_lora_gen.csv` exist.
- `CKPT_ONLY`: VS checkpoint exists, but result CSV is missing.
- `BLOCKED_SUPCON`: VS data exists, but SupCon encoder checkpoint is missing.
- `BLOCKED_DATA`: VS data is missing.

---

## 4. Confirmed subject-level assets

### S24

Status: VS complete and Exp43 P0 complete.

VS r32 checkpoint:

```text
/content/drive/MyDrive/vsvi_data/checkpoints_vsre_lora_gen/20260629_094904_lora_r32_ep100/subj24_lora_best.pt
```

VS result:

```text
sid,best_ep,top1,top3,top5,dominant,entropy
24,98,0.46296296296296297,0.5555555555555556,0.6481481481481481,0.3148148148148148,1.669946131339928
```

Exp43 S24 summary:

```text
c0 top1 = 0.0925925926
c1 top1 = 0.1111111111
```

Interpretation: frozen VS-to-VI transfer was weak for S24. C1 only slightly exceeded C0 and remained at chance-level top-1.

### S01

Status: VS checkpoint preserved; Exp43 currently running.

VI data:

```text
preproc_vi_re/preproc_subj_01_1.npz ... preproc_subj_01_9.npz
```

SupCon:

```text
checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon/subj01_best.pt
```

VS r32 checkpoint:

```text
/content/drive/MyDrive/vsvi_data/checkpoints_vsre_lora_gen/20260617_074245_lora_r32_ep100/subj01_lora_best.pt
```

Current Exp43 command uses:

```text
lora_r = 32
n_eeg_tokens = 16
conditions = c0,c1
per_class_total = 60
```

### S18

Status: ready for Exp43 after S01 finishes.

VI data:

```text
preproc_vi_re/preproc_subj_18_1.npz ... preproc_subj_18_8.npz
```

SupCon:

```text
/content/vsvi_project/checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon/subj18_best.pt
```

VS r16 checkpoint preserved in Drive:

```text
/content/drive/MyDrive/vsvi_data/checkpoints_vsre_lora_gen/20260612_010728_lora_r16_ep100/subj18_lora_best.pt
```

S18 must be run with `--lora_r 16`, not r32, because the available VS initialization checkpoint is r16.

### S02

Status: VS data exists, but SupCon missing.

VS data:

```text
/content/drive/MyDrive/vsvi_data/preproc_vs_re/preproc_subj_02_1.npz
...
/content/drive/MyDrive/vsvi_data/preproc_vs_re/preproc_subj_02_9.npz
```

Needed before VS LoRA training:

```text
checkpoints_vsre_dino/.../subj02_best.pt
```

### S09

Status: VS data exists, but SupCon missing.

VS data:

```text
/content/drive/MyDrive/vsvi_data/preproc_vs_re/preproc_subj_09_1.npz
...
/content/drive/MyDrive/vsvi_data/preproc_vs_re/preproc_subj_09_4.npz
```

Needed before VS LoRA training:

```text
checkpoints_vsre_dino/.../subj09_best.pt
```

### S28 / S29

Status: VS data and SupCon both missing in current Drive/repo search.

Needed first:

```text
/content/drive/MyDrive/vsvi_data/preproc_vs_re/preproc_subj_28_*.npz
/content/drive/MyDrive/vsvi_data/preproc_vs_re/preproc_subj_29_*.npz
```

Then SupCon checkpoints:

```text
checkpoints_vsre_dino/.../subj28_best.pt
checkpoints_vsre_dino/.../subj29_best.pt
```

Then VS r32 LoRA training.

---

## 5. Why not just use `--subject_ids 1,18`?

Do not combine S01 and S18 in one Exp43 command with the current checkpoints.

Reason:

```text
S01 uses r32 VS LoRA initialization.
S18 uses r16 VS LoRA initialization.
```

A single command has only one `--lora_r` and one explicit `--init_lora_ckpt`. If `--subject_ids 1,18` is used with the S01 r32 checkpoint, S18 can be incorrectly initialized from the S01 r32 checkpoint. That would invalidate provenance.

Safe rule:

```text
Same lora_r + same SupCon root + subject-specific init auto-discovery available:
  multi-subject sequential command is acceptable.

Different lora_r or explicit subject-specific checkpoint:
  use separate commands or a queue launcher.
```

---

## 6. Next execution queue

### Step 1: finish current S01 Exp43

Wait until:

```bash
pgrep -af 'train_exp43_vi_lora.py|train_vs_re_lora_gen.py' || echo "No training process"
```

returns no active training process and the S01 log contains:

```text
Saved run summary CSV
```

### Step 2: run S18 Exp43 C0/C1

Use this only after S01 is complete.

```bash
cd /content/vsvi_project

LOG=/content/drive/MyDrive/vsvi_data/logs/exp43_s18_c0c1_r16_tok16_$(date +%Y%m%d_%H%M%S).log
PIDFILE=/content/drive/MyDrive/vsvi_data/logs/exp43_s18_c0c1_r16_tok16_latest.pid

nohup python -u train_exp43_vi_lora.py \
  --subject_ids 18 \
  --conditions c0,c1 \
  --data_root preproc_vi_re \
  --img_root preproc_data_vi/images \
  --supcon_ckpt /content/drive/MyDrive/vsvi_data/checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon \
  --init_lora_ckpt /content/drive/MyDrive/vsvi_data/checkpoints_vsre_lora_gen/20260612_010728_lora_r16_ep100/subj18_lora_best.pt \
  --ckpt_root /content/drive/MyDrive/vsvi_data/checkpoints_exp43_vi_lora \
  --lora_r 16 \
  --n_eeg_tokens 16 \
  --epochs 100 \
  --batch_size 2 \
  --per_class_total 60 \
  --eval_n_samples 54 \
  --fp16 \
  > "$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "Started S18 Exp43 C0/C1"
echo "PID=$(cat $PIDFILE)"
echo "LOG=$LOG"
```

### Step 3: create SupCon checkpoints for S02/S09

Required outputs:

```text
/content/drive/MyDrive/vsvi_data/checkpoints_vsre_dino/<timestamp>_ch32_merged_ep200_supcon/subj02_best.pt
/content/drive/MyDrive/vsvi_data/checkpoints_vsre_dino/<timestamp>_ch32_merged_ep200_supcon/subj09_best.pt
```

After SupCon exists, rerun:

```bash
cd /content/vsvi_project
python -u audit_launch_vs_lora_subjects.py --save_csv
```

Expected status should change from `BLOCKED_SUPCON` to `MISSING_CKPT` for S02/S09.

### Step 4: run S02/S09 VS r32 LoRA training

Once S02/S09 are `MISSING_CKPT`, use the audit launcher:

```bash
cd /content/vsvi_project
python -u audit_launch_vs_lora_subjects.py --launch_next_missing --save_csv
```

This launches only one subject at a time.

### Step 5: upload/create S28/S29 VS npz

Needed files:

```text
/content/drive/MyDrive/vsvi_data/preproc_vs_re/preproc_subj_28_*.npz
/content/drive/MyDrive/vsvi_data/preproc_vs_re/preproc_subj_29_*.npz
```

After upload, create SupCon for S28/S29, then run VS r32 LoRA training.

---

## 7. Important safety/provenance rules

1. Do not launch parallel GPU training jobs on the same Colab runtime unless explicitly intended.
2. Do not mix subjects with different LoRA ranks in one Exp43 command.
3. Do not initialize S18 from an S01 checkpoint.
4. Treat `CKPT_ONLY` as an existing model, not a missing model.
5. For `CKPT_ONLY` subjects, generate evaluation CSV later instead of retraining.
6. Store checkpoints under Drive, not only under `/content/vsvi_project`, because Colab runtime deletion removes repo-local files.
7. Keep final metrics/result CSVs on Drive unless explicitly pushing result summaries to GitHub.

---

## 8. Immediate commands to keep handy

Check active process:

```bash
pgrep -af 'train_exp43_vi_lora.py|train_vs_re_lora_gen.py' || echo "No training process"
nvidia-smi
```

Tail latest Exp43 log:

```bash
LOG=$(ls -t /content/drive/MyDrive/vsvi_data/logs/exp43_*.log | head -1)
echo "$LOG"
tail -n 120 "$LOG"
```

Check latest Exp43 summary:

```bash
find /content/drive/MyDrive/vsvi_data/checkpoints_exp43_vi_lora \
  -name "*exp43_vi_summary.csv" -ls | tail -10

CSV=$(ls -t /content/drive/MyDrive/vsvi_data/checkpoints_exp43_vi_lora/*exp43_vi_summary.csv | head -1)
echo "$CSV"
cat "$CSV"
```

Rerun VS audit:

```bash
cd /content/vsvi_project
python -u audit_launch_vs_lora_subjects.py --save_csv
```
