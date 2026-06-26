# upload_npu — 서버 업로드 체크리스트 및 프로젝트 현황

Last updated: 2026-06-26  
기준 문서: README.md / HUMAN_DIRECTIVE.md / PROGRESS.md / vsvi_project_backup_20260617.log

---

## 1. 현재 프로젝트 상태

### 핵심 결과 요약

| 실험 | 피험자 | 설정 | Dual-gallery DINO@1 | 비고 |
|------|--------|------|---------------------|------|
| Stage 2 retrieval (비생성) | 참조 | non-generative | `0.3333` | 생성 모델 비교 기준선 |
| Exp41 SD 1.5 LoRA | S18 | r=16, 8tok | `0.2870` | 첫 SD prior 결과 |
| Exp42-A SD 1.5 LoRA | S01 | r=16, 8tok | `0.3333` | retrieval과 동등 |
| Exp42-B Step2 | S18 | r=16, 16tok | `0.2963` | S18 현재 최고 |
| **Exp42-B Step4** | **S01** | **r=32, 16tok** | **`0.3571`** | **프로젝트 최고 (retrieval 초과)** |
| Exp42-B Step5 (aug) | S18 | r=16, 16tok+aug | 내부 `0.4464` | full-test 아님, 참고용 |

### 현재 해석
- SD 1.5 LoRA가 현재 최적 생성 모델
- S01 r=32 16tok이 Stage 2 retrieval(0.3333)을 처음으로 초과한 완전 생성 결과
- LoRA rank는 피험자별로 다름: S18→r=16 최적, S01→r=32 최적
- VI fine-tuning 시 r=32 포함 필수 (VI EEG 신호가 더 약하기 때문)

### 최적 체크포인트
- **S18 best**: `checkpoints_vsre_lora_gen/20260612_010728_lora_r16_ep100` (r=16, 16tok)
- **S01 best**: `checkpoints_vsre_lora_gen/20260617_074245_lora_r32_ep100` (r=32, 16tok)
- **SupCon EEG 인코더**: `checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon`

---

## 2. 다음 실험 우선순위 (HUMAN_DIRECTIVE.md 기준)

### Priority 0 — S24 SD LoRA VS 생성

```bash
python train_vs_re_lora_gen.py --subject_ids 24 --epochs 100 --lora_r 16 --n_eeg_tokens 16
python train_vs_re_lora_gen.py --subject_ids 24 --epochs 100 --lora_r 32 --n_eeg_tokens 16
```

판단 기준:
- `S24 > S01`: 세션 수/데이터 양이 주요 요인
- `S24 ≈ S01`: SD LoRA가 데이터 충분 시 실질적 상한에 도달
- `S24 < S01`: 피험자별 EEG 신호 품질이 데이터 양보다 중요

### Priority 1 — Exp43 SD LoRA VI fine-tuning

필요 스크립트: `train_vi_lora_gen.py` (미작성 — 신규 작성 필요)

비교 조건:
- `C0`: VI scratch LoRA
- `C1`: VS LoRA → VI fine-tune (EEG 인코더 frozen)
- `C2`: VS LoRA → VI fine-tune (staged encoder unfreeze, C1이 ep50까지 개선 없으면 시작)

초기화 체크포인트:
- S01: Exp42-B Step4 r=32, 16tok
- S18: Exp42-B Step2 r=16, 16tok

성공 기준:
- 최소: `C1 > C0`
- 의미있음: VI DINO@1 `> 0.20`

### Priority 2 — Exp42-B Step5 추가 augmentation

S18 Step5 1회 완료. S24/Exp43 이후 여유 compute 시 추가.

### Priority 3 — Exp42-B Step6 staged encoder unfreeze

최하위 우선순위.

---

## 3. 서버 업로드 파일 목록

### 3-1. Python 스크립트 (소용량 — 전부 교체)

| 파일 | 필요 여부 | 비고 |
|------|-----------|------|
| `train_vs_re_lora_gen.py` | **필수** | S24 학습 메인 스크립트 |
| `train_vs_re_latent_gen.py` | **필수** | 위 스크립트가 import (make_schedule, sample_ddim 등) |
| `dataset_vs_re.py` | **필수** | VS 데이터 로더 |
| `model_eeg_dino.py` | **필수** | EEG 인코더 모델 |
| `model_128_eegonly_transformer.py` | **필수** | EEGEncoderV2 포함 |
| `train_crosssubj_dino.py` | **필수** | set_seed, compute_class_prototypes 등 |
| `eval_exp39_multigallery.py` | **필수** | dual-gallery 평가 |
| `eval_vs_re_exp25_vi_transfer.py` | **필수** | eval 스크립트에서 import |
| `eval_vs_re_stage2_readout.py` | 권장 | 비교용 |
| `train_vi_lora_gen.py` | Priority 1 | 미작성, Exp43용 신규 작성 필요 |

### 3-2. 체크포인트 (대용량 — 필수 우선)

| 경로 | 용도 | 우선순위 |
|------|------|----------|
| `checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon/` | S24 EEG 인코더 초기화 | **필수** |
| `checkpoints_vsre_lora_gen/20260617_074245_lora_r32_ep100/` | S01 best (Exp43 C1 시작점) | 권장 |
| `checkpoints_vsre_lora_gen/20260612_010728_lora_r16_ep100/` | S18 best (Exp43 C1 시작점) | 권장 |

### 3-3. 이미지 데이터 (소용량 — 필수)

| 경로 | 내용 |
|------|------|
| `preproc_data_vi/images/` | 클래스 이미지 01.png ~ 09.png (9장) |

없으면 학습 시작 불가.

### 3-4. 데이터 (이미 업로드 완료)

- `preproc_vs_re/` ✅
- `preproc_vi_re/` ✅

---

## 4. 서버에서 설치/다운로드해야 할 것

### Python 패키지

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install diffusers peft accelerate transformers huggingface_hub
pip install h5py scipy numpy Pillow tqdm kornia timm matplotlib scikit-learn
```

### 사전 학습 모델 (최초 실행 시 자동 다운로드 또는 수동)

| 모델 | 크기 | 다운로드 방법 |
|------|------|---------------|
| DINOv2 ViT-S14 | ~300MB | `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` |
| SD VAE | ~300MB | `AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")` |
| SD 1.5 UNet | ~3.4GB | `UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")` |

> **주의**: 서버의 `dinov2_vits14.pth`는 단독 가중치 파일임.  
> 학습 스크립트는 `~/.cache/torch/hub/facebookresearch_dinov2_main/` 구조로 로드하므로  
> 서버에서 `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` 1회 실행 필요.

---

## 5. 학습 실행 커맨드 (서버)

### S24 r=16 (Priority 0 - 먼저 실행)

```bash
python train_vs_re_lora_gen.py \
  --subject_ids 24 \
  --lora_r 16 \
  --n_eeg_tokens 16 \
  --epochs 100 \
  2>&1 | tee logs/exp_s24_lora_r16_tok16.log
```

### S24 r=32 (compute 허용 시)

```bash
python train_vs_re_lora_gen.py \
  --subject_ids 24 \
  --lora_r 32 \
  --n_eeg_tokens 16 \
  --epochs 100 \
  2>&1 | tee logs/exp_s24_lora_r32_tok16.log
```

### S24 full-test 평가

```bash
python eval_exp39_multigallery.py \
  --subject_ids 24 \
  --ckpt_dir checkpoints_vsre_lora_gen/<S24_체크포인트_폴더>
```

---

## 6. 평가 기준

- 공식 지표: **full-test dual-gallery DINO@1** (n=전체 test trial)
- 학습 중 내부 지표(partial-test)는 참고용만 — 최종 비교 불가
- 비교 기준선:
  - Stage 2 retrieval: `0.3333`
  - S01 Exp42-B Step4: `0.3571` (현재 프로젝트 최고)
  - S18 Exp42-B Step2: `0.2963`

---

## 7. 데이터 명세

| 항목 | 값 |
|------|-----|
| 채널 | 32 EEG only |
| 시간 윈도우 | 0~2 sec |
| 클래스 수 | 9 |
| S24 VS 세션 | 10세션 (최대) |
| S24 VI 세션 | 10세션 (최대) |
| 레이블 | 1~9 (클래스 레이블 conditioning 사용 안함) |

---

## 8. 주요 제약사항 (HUMAN_DIRECTIVE.md Constraints)

- 클래스 레이블을 생성 conditioning에 추가하지 말 것
- CFG sweep은 실제 unconditional path 없으면 하지 말 것
- stochastic sample-grid 결과만으로 final metric으로 보고하지 말 것
- multi-image target training 재시도 금지 (Exp40에서 성능 저하 확인)
- Exp31~40 scratch latent/pixel UNet path 반복 금지
