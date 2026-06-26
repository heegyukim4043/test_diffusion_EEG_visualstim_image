# Data Log — preproc_vs_re

> Last updated: 2026-04-11

---

## 1. 개요

| 항목 | 내용 |
|------|------|
| 경로 | `preproc_vs_re/` |
| 데이터 종류 | VS(Visual Stimulation) EEG, 피험자별 반복 측정 |
| 실험 패러다임 | 기존 VS 실험과 동일 |
| 파일 형식 | MATLAB v7.3 (HDF5) — `h5py`로 로드 필요 |
| 파일 명명 규칙 | `preproc_subj_{sid:02d}_{session}.mat` |
| 총 파일 수 | 54개 |
| 총 피험자 수 | 18명 |
| 총 trial 수 | 7,290 |

---

## 2. 파일 내부 구조

HDF5 키: `results/data`

```
data shape: (5, 3, 9, 4096, 40)  dtype=float32
```

| 차원 | 크기 | 의미 |
|------|------|------|
| dim0 | 5 | trials per class per block |
| dim1 | 3 | blocks per session |
| dim2 | 9 | classes (1~9) |
| dim3 | 4096 | time points — 1024 Hz × 4 sec (epoch: **-1 ~ +3 sec**) |
| dim4 | 40 | channels — 32 EEG + 8 EX (전체 사용) |

**파일당 trial 수**: 5 × 3 × 9 = **135 trials/session** (모든 파일 동일)

### Epoch 구간 상세

| 구간 | 샘플 index | 내용 |
|------|-----------|------|
| -1 ~ 0 sec | 0 ~ 1023 | 자극 전 (baseline) |
| 0 ~ +2 sec | 1024 ~ 3071 | 자극 후 2초 — **기존 preproc_for_gan_vs 동일 구간** |
| +2 ~ +3 sec | 3072 ~ 4095 | 자극 후 추가 1초 |

> 기존 데이터와 concat 시: `data[:, :, 1024:3072, :]` 슬라이싱으로 구간 맞춤

### 기타 메타데이터

| 키 | 내용 |
|----|------|
| `results/srate` | 1024 Hz |
| `results/ep_len` | [-1, 3] |
| `results/subj` | 피험자 번호 |
| `results/ch_info` | 채널 이름 40개 |
| `results/ica` | ICA 적용 여부 |

### 채널 목록 (40채널)

- **EEG 32ch**: FP1, AF3, F7, F3, FC1, FC5, T7, C3, CP1, CP5, P7, P3, Pz, PO3, O1, Oz, O2, PO4, P4, P8, CP6, CP2, C4, T8, FC6, FC2, F4, F8, AF4, FP2, Fz, Cz
- **External 8ch**: EX 1 ~ EX 8 — 학습 시 40ch 전체 사용

---

## 3. 피험자별 데이터 현황

| 피험자 | 세션 수 | 세션 목록 | 총 trial 수 |
|--------|---------|-----------|-------------|
| S01 | 8 | 1~8 | 1,080 |
| S02 | 5 | 1~5 | 675 |
| S03 | 1 | 1 | 135 |
| S04 | 2 | 1~2 | 270 |
| S05 | 1 | 1 | 135 |
| S09 | 2 | 1~2 | 270 |
| S10 | 2 | 1~2 | 270 |
| S11 | 1 | 1 | 135 |
| S16 | 1 | 1 | 135 |
| S18 | 8 | 1~8 | 1,080 |
| S19 | 1 | 1 | 135 |
| S20 | 1 | 1 | 135 |
| S21 | 1 | 1 | 135 |
| S23 | 1 | 1 | 135 |
| S24 | 9 | 1~9 | 1,215 |
| S28 | 4 | 1~4 | 540 |
| S29 | 5 | 1~5 | 675 |
| S35 | 1 | 1 | 135 |
| **합계** | **54** | | **7,290** |

### 세션 수 분포

| 세션 수 | 피험자 | 인원 |
|---------|--------|------|
| 1 | S03, S05, S11, S16, S19, S20, S21, S23, S35 | 9명 |
| 2 | S04, S09, S10 | 3명 |
| 4 | S28 | 1명 |
| 5 | S02, S29 | 2명 |
| 8 | S01, S18 | 2명 |
| 9 | S24 | 1명 |

> 피험자 ID 불연속: S06~S08, S12~S15, S17, S22, S25~S27, S30~S34 없음

---

## 4. 기존 데이터 (preproc_for_gan_vs) 와 비교

| 항목 | preproc_for_gan_vs | preproc_vs_re |
|------|-------------------|---------------|
| 실험 패러다임 | VS | VS (동일) |
| 피험자 수 | 20명 (S01~S20) | 18명 (불연속) |
| 파일 형식 | MATLAB v5 (scipy.io) | MATLAB v7.3 (h5py) |
| 데이터 구조 | X:(ch, time, trial), y:(trial,) | results/data:(5, 3, 9, 4096, 40) |
| 샘플링률 | — | 1024 Hz |
| epoch 구간 | 0 ~ +2 sec | -1 ~ +3 sec (전체), **0 ~ +2 sec 슬라이싱 가능** |
| 채널 수 | 32 | 40 (32 EEG + 8 EX) |
| trial/subj | 180 (20/class) | 135/session × n_sess |
| 반복 측정 | 없음 | 있음 (1~9세션) |
| 클래스 수 | 9 | 9 |

---

## 5. 데이터 활용 방향

### 기존 데이터와 병합 시 정렬 방법

```python
# (5, 3, 9, 4096, 40) → (trial, ch, time) 변환 + 0~2 sec 구간 추출
data = np.array(f["results/data"])           # (5, 3, 9, 4096, 40)
data = data[:, :, :, 1024:3072, :]           # 0~+2 sec 구간
data = data.transpose(2, 4, 3, 0, 1)         # (9, 40, 2048, 5, 3)
data = data.reshape(9, 40, 2048, -1)         # (9, 40, 2048, 15)  15 = 5*3
# label 생성: class axis(dim2)가 label
labels = np.repeat(np.arange(1, 10), 15)     # trial 수만큼 반복
```

### 세션 불균형 처리 옵션

| 옵션 | 방법 | 장점 | 단점 |
|------|------|------|------|
| A | 세션 단위 독립 처리 | 데이터 손실 없음, cross-session 평가 가능 | 피험자별 데이터 크기 상이 |
| B | 피험자별 전체 통합 | 구현 단순 | trial 수 불균형 심함 (135 vs 1215) |
| C | 최소 세션 수 기준 통일 (1세션=135 trial) | 균형 맞춤 | 데이터 손실 |

---

## 6. TODO

- [x] epoch 구간 확인: -1 ~ +3 sec
- [x] 채널 수 확인: 40ch 전체 사용
- [x] 실험 패러다임 확인: 기존 VS와 동일
- [ ] h5py 로더 구현 (기존 scipy.io 기반 코드와 통합)
- [ ] 세션 불균형 처리 방식 결정 (옵션 A/B/C)
- [ ] 기존 preproc_for_gan_vs와 피험자 overlap 확인 (S01~S20 중 일치하는 피험자)
- [ ] dim1=3 (blocks) 구조 확인 (block 간 휴식 있는지 여부)
