#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-preproc_vs_re}"
IMG_ROOT="${IMG_ROOT:-preproc_data_vi/images}"
SUPCON_CKPT="${SUPCON_CKPT:-checkpoints_vsre_dino/20260604_091352_ch32_merged_ep200_supcon}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints_vsre_lora_gen}"
LOG_DIR="${LOG_DIR:-logs}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
N_EEG_TOKENS="${N_EEG_TOKENS:-16}"
RANKS="${RANKS:-16 32}"

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
PREFLIGHT_LOG="$LOG_DIR/run_track_a_s24_preflight_${TS}.log"

printf '[INFO] Track A S24 runner
'
printf '[INFO] ROOT=%s
' "$ROOT"
printf '[INFO] DATA_ROOT=%s
' "$DATA_ROOT"
printf '[INFO] IMG_ROOT=%s
' "$IMG_ROOT"
printf '[INFO] SUPCON_CKPT=%s
' "$SUPCON_CKPT"
printf '[INFO] CKPT_ROOT=%s
' "$CKPT_ROOT"
printf '[INFO] EPOCHS=%s
' "$EPOCHS"
printf '[INFO] BATCH_SIZE=%s
' "$BATCH_SIZE"
printf '[INFO] N_EEG_TOKENS=%s
' "$N_EEG_TOKENS"
printf '[INFO] RANKS=%s
' "$RANKS"

python preflight_track_a.py   --subject_id 24   --img_root "$IMG_ROOT"   --supcon_ckpt "$SUPCON_CKPT"   --data_root "$DATA_ROOT"   --ckpt_root "$CKPT_ROOT"   --check_data 2>&1 | tee "$PREFLIGHT_LOG"

for rank in $RANKS; do
  TRAIN_LOG="$LOG_DIR/run_track_a_s24_r${rank}_bs${BATCH_SIZE}_${TS}.log"
  printf '
[INFO] Starting S24 r=%s (batch_size=%s)
' "$rank" "$BATCH_SIZE"
  python train_vs_re_lora_gen.py     --subject_ids 24     --epochs "$EPOCHS"     --lora_r "$rank"     --n_eeg_tokens "$N_EEG_TOKENS"     --batch_size "$BATCH_SIZE"     --img_root "$IMG_ROOT"     --supcon_ckpt "$SUPCON_CKPT"     --ckpt_root "$CKPT_ROOT" 2>&1 | tee "$TRAIN_LOG"
done

printf '
[INFO] S24 Track A run complete.
'
printf '[INFO] Checkpoints: %s/<timestamp>_lora_r16_ep%s, %s/<timestamp>_lora_r32_ep%s
' "$CKPT_ROOT" "$EPOCHS" "$CKPT_ROOT" "$EPOCHS"
