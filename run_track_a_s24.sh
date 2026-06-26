#!/bin/bash
# Track A - S24 SD LoRA training (r=16, r=32)
# NPU 서버에서 실행: bash run_track_a_s24.sh
set -e

cd "$(dirname "$0")"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Track A S24 시작"

SUPCON_CKPT="checkpoints_vsre_dino/20260604_091352_ch32_merged_ep200_supcon"
IMG_ROOT="preproc_data_vi/images"
CKPT_ROOT="checkpoints_vsre_lora_gen"

# r=16
echo "[$(date '+%Y-%m-%d %H:%M:%S')] r=16 학습 시작"
python -u train_vs_re_lora_gen.py \
    --subject_ids 24 \
    --lora_r 16 \
    --n_eeg_tokens 16 \
    --epochs 100 \
    --batch_size 8 \
    --img_root "$IMG_ROOT" \
    --supcon_ckpt "$SUPCON_CKPT" \
    --ckpt_root "$CKPT_ROOT"

# r=32
echo "[$(date '+%Y-%m-%d %H:%M:%S')] r=32 학습 시작"
python -u train_vs_re_lora_gen.py \
    --subject_ids 24 \
    --lora_r 32 \
    --n_eeg_tokens 16 \
    --epochs 100 \
    --batch_size 8 \
    --img_root "$IMG_ROOT" \
    --supcon_ckpt "$SUPCON_CKPT" \
    --ckpt_root "$CKPT_ROOT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Track A S24 완료"
