#!/bin/bash
# run_s24_lora_r32.sh
# S24 SD LoRA VS generation — r=32, 16 tokens, 100 epochs
# Run AFTER run_s24_lora_r16.sh completes successfully.
# Usage: bash run_s24_lora_r32.sh

set -e
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/track_a_s24_r32_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

echo "=== S24 LoRA r=32 preflight ==="
python preflight_track_a.py \
  --subject_id 24 \
  --img_root preproc_data_vi/images \
  --supcon_ckpt checkpoints_vsre_dino/20260604_091352_ch32_merged_ep200_supcon \
  --data_root preproc_vs_re \
  --ckpt_root checkpoints_vsre_lora_gen \
  --check_data
echo ""

echo "=== Starting S24 LoRA r=32 training ==="
echo "Log: $LOG_FILE"
echo ""

python train_vs_re_lora_gen.py \
  --subject_ids 24 \
  --lora_r 32 \
  --n_eeg_tokens 16 \
  --epochs 100 \
  --img_root preproc_data_vi/images \
  --supcon_ckpt checkpoints_vsre_dino/20260604_091352_ch32_merged_ep200_supcon \
  --ckpt_root checkpoints_vsre_lora_gen \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
if [ $EXIT_CODE -eq 0 ]; then
  echo "=== S24 r=32 training DONE. Log: $LOG_FILE ==="
  echo "Next: run dual-gallery eval and update PROGRESS.md"
else
  echo "=== S24 r=32 training FAILED (exit=$EXIT_CODE). Check $LOG_FILE ==="
fi
exit $EXIT_CODE
