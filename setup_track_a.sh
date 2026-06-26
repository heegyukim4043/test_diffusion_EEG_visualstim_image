#!/bin/bash
# setup_track_a.sh
# Track A 환경 설정: CUDA 서버 또는 Colab에서 실행
# Usage: bash setup_track_a.sh

set -e
echo "=== Track A Environment Setup ==="

# 1. Python/CUDA 확인
python -c "import torch; print(f'torch {torch.__version__}, CUDA={torch.cuda.is_available()}')"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — wrong runtime'"

# 2. 필수 패키지 설치
pip install -q peft diffusers accelerate transformers

# 3. 설치 확인
python -c "import peft; print(f'peft {peft.__version__}')"
python -c "import diffusers; print(f'diffusers {diffusers.__version__}')"

# 4. preflight
echo ""
echo "=== Running preflight ==="
python preflight_track_a.py \
  --subject_id 24 \
  --img_root preproc_data_vi/images \
  --supcon_ckpt checkpoints_vsre_dino/20260604_091352_ch32_merged_ep200_supcon \
  --data_root preproc_vs_re \
  --ckpt_root checkpoints_vsre_lora_gen \
  --check_data

echo ""
echo "=== Setup complete. Run run_s24_lora_r16.sh to start training. ==="
