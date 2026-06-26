"""
train_vs_repa_allclass.py
─────────────────────────────────────────────────────────────────────────────
preproc_for_gan_vs 데이터셋 / 9-class (또는 그룹 3-class) / REPA 손실 포함
EEG→Image 확산 모델 학습 (VS 전용, subject-wise 모드)

▸ 모델   : model_128_eegonly_transformer_repa.EEGDiffusionModel128
▸ 손실   : ε-pred MSE + 재구성(L1) + SSIM + REPA(ResNet18 feature alignment)
▸ 데이터 : preproc_for_gan_vs/subj_XX.mat  (X: ch×T×trial, y: trial)
▸ 이미지 : preproc_data_vi/images/01.png ~ 09.png  (기본 경로, --img_root 변경 가능)
▸ 학습   : 피험자별 독립 모델 (subject-wise)
▸ 추론   : sample_vs_repa_allclass.py 사용

VS 데이터 특성
──────────────
- 경로   : preproc_for_gan_vs/subj_01.mat ~ subj_34.mat
- 구조   : X=(ch, 512, 360), y=(360,) — ch×time×trial
- fs     : 512 Hz → 1 sec epoch
- 클래스 : 1~9, trial당 40회

사용 예시
─────────
# 단일 피험자 (subj 03), 그룹 1~3
python train_vs_repa_allclass.py --subject_ids 3 --group_ids 1,2,3

# 범위 지정 (1~10번 피험자)
python train_vs_repa_allclass.py --subject_ids 1-10

# 전체 피험자, 그룹 1만
python train_vs_repa_allclass.py --subject_ids all --group_ids 1

# 9-class 전체 모드
python train_vs_repa_allclass.py --subject_ids 1 --group_ids 0

# 하이퍼파라미터 조정
python train_vs_repa_allclass.py --subject_ids 1-10 \\
    --epochs 200 --batch_size 32 --lr 1e-4 \\
    --lambda_percept 0.05 --lambda_rec 0.02 --lambda_ssim 0.05
"""

import os
import copy
import glob
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

from model_128_eegonly_transformer_repa import EEGDiffusionModel128


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def ema_update(model, ema_model, decay: float = 0.999):
    msd = model.state_dict()
    for name, param in ema_model.state_dict().items():
        if name in msd:
            param.copy_(param * decay + msd[name] * (1.0 - decay))


def parse_subject_ids(s: str, all_mat: list) -> list:
    """
    문자열을 피험자 ID 정수 리스트로 변환한다.
      'all'       → all_mat 에서 전체 번호 추출
      '1-10'      → [1, 2, ..., 10]
      '1,3,5'     → [1, 3, 5]
      '1-5,8,10'  → [1, 2, 3, 4, 5, 8, 10]
    """
    s = s.strip().lower()
    if s == "all":
        ids = []
        for p in all_mat:
            base = os.path.basename(p)
            num  = base.replace("subj_", "").replace(".mat", "")
            ids.append(int(num))
        return sorted(ids)

    ids = []
    for token in s.split(","):
        token = token.strip()
        if "-" in token:
            a, b = token.split("-", 1)
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(token))
    return ids


# 그룹별 클래스 정의
GROUP_CLASSES = {
    1: [1, 2, 3],
    2: [4, 5, 6],
    3: [7, 8, 9],
}


def parse_group_ids(s: str) -> list:
    """
    문자열을 그룹 ID 정수 리스트로 변환한다.
      'all'   → [1, 2, 3]
      '1-3'   → [1, 2, 3]
      '1,2'   → [1, 2]
      '2'     → [2]
    """
    s = s.strip().lower()
    if s == "all":
        return [1, 2, 3]
    ids = []
    for token in s.split(","):
        token = token.strip()
        if "-" in token:
            a, b = token.split("-", 1)
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(token))
    for gid in ids:
        if gid not in (1, 2, 3):
            raise ValueError(f"group_id는 1, 2, 3 중 하나여야 합니다. (입력: {gid})")
    return ids


def detect_eeg_channels(mat_path: str) -> int:
    """mat 파일에서 EEG 채널 수를 자동 감지한다."""
    try:
        mat = loadmat(mat_path)
        X = mat["X"]  # (ch, time, trial)
        return int(X.shape[0])
    except Exception as e:
        print(f"[WARN] eeg channel 자동 감지 실패: {e}. 기본값 32 사용.")
        return 32


def find_img_root(vs_root: str, fallback: str = "./preproc_data_vi/images") -> str:
    """
    VS 데이터 폴더 내에 images/ 가 있으면 그것을, 없으면 fallback 경로를 반환한다.
    """
    vs_img = os.path.join(vs_root, "images")
    if os.path.isdir(vs_img):
        return vs_img
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# EEG Augmentation
# ─────────────────────────────────────────────────────────────────────────────

class EEGAugment:
    """
    학습 시 EEG 배치 (B, ch, time) 에 적용하는 augmentation.

    augmentation 목록
    -----------------
    1. Gaussian noise   : EEG에 N(0, σ) 노이즈 추가
    2. Amplitude scale  : 신호 전체를 U(scale_min, scale_max) 로 스케일링
    3. Channel dropout  : 채널을 확률 p 로 0으로 만들기
    4. Time shift       : 시간 축을 최대 ±max_shift 샘플 순환 이동

    Parameters (args 에서 읽음)
    ---------------------------
    aug_noise_std   : float  (0이면 비활성)
    aug_scale_min   : float
    aug_scale_max   : float  (min==max==1이면 비활성)
    aug_ch_drop_p   : float  (0이면 비활성)
    aug_time_shift  : int    (0이면 비활성)
    """

    def __init__(self, args):
        self.noise_std  = getattr(args, "aug_noise_std",  0.02)
        self.scale_min  = getattr(args, "aug_scale_min",  0.8)
        self.scale_max  = getattr(args, "aug_scale_max",  1.2)
        self.ch_drop_p  = getattr(args, "aug_ch_drop_p",  0.1)
        self.time_shift = getattr(args, "aug_time_shift",  20)

    @torch.no_grad()
    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        """eeg: (B, ch, time)"""
        B, C, T = eeg.shape
        dev = eeg.device

        # 1. Gaussian noise
        if self.noise_std > 0:
            eeg = eeg + torch.randn_like(eeg) * self.noise_std

        # 2. Amplitude scaling (trial별 독립 스케일)
        if self.scale_min != 1.0 or self.scale_max != 1.0:
            scale = (torch.rand(B, 1, 1, device=dev)
                     * (self.scale_max - self.scale_min) + self.scale_min)
            eeg = eeg * scale

        # 3. Channel dropout (배치 내 각 샘플 독립)
        if self.ch_drop_p > 0:
            mask = (torch.rand(B, C, 1, device=dev) > self.ch_drop_p).float()
            eeg = eeg * mask

        # 4. Time shift (순환 이동, trial별 독립)
        if self.time_shift > 0:
            shifts = torch.randint(-self.time_shift, self.time_shift + 1, (B,))
            shifted = []
            for i in range(B):
                shifted.append(torch.roll(eeg[i], shifts[i].item(), dims=-1))
            eeg = torch.stack(shifted, dim=0)

        return eeg


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class EEGImageDatasetVS(Dataset):
    """
    VS EEG→Image 데이터셋 (preproc_for_gan_vs 포맷).
    9-class 전체 또는 3-class 그룹 모드 지원.

    Parameters
    ----------
    mat_paths  : list[str]
    img_root   : str   — 클래스 이미지 폴더 (01.png ~ 09.png)
    indices    : list[(subj_idx, trial_idx)]
    img_size   : int
    cls_list   : list[int] | None
        학습할 1-based 클래스 목록. None이면 1~9 전체.
    cls_min    : int
        그룹 내 0-based 라벨 계산 기준: label_0 = label_1 - cls_min.
    """

    def __init__(
        self,
        mat_paths: list,
        img_root: str,
        indices: list,
        img_size: int = 128,
        cls_list: list = None,
        cls_min: int = 1,
    ):
        super().__init__()
        self.img_root = img_root
        self.img_size = img_size
        self.indices = indices
        self.cls_min = cls_min
        load_classes = cls_list if cls_list is not None else list(range(1, 10))

        # EEG / label 데이터 로드
        self.eegs = []
        self.labels = []

        for mat_path in mat_paths:
            mat = loadmat(mat_path)
            X = mat["X"]             # (ch, time, trial)
            y = mat["y"].squeeze()   # (trial,)
            eeg_t = torch.from_numpy(X.astype(np.float32)).permute(2, 0, 1)
            self.eegs.append(eeg_t)
            self.labels.append(y.astype(np.int64))

        # 이미지 변환
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # 클래스 이미지 캐싱
        self._img_cache = {}
        for cls in load_classes:
            path = os.path.join(img_root, f"{cls:02d}.png")
            if os.path.isfile(path):
                self._img_cache[cls] = Image.open(path).convert("RGB")
            else:
                raise FileNotFoundError(f"클래스 이미지 없음: {path}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        subj_idx, trial_idx = self.indices[idx]
        eeg = self.eegs[subj_idx][trial_idx]
        label_1based = int(self.labels[subj_idx][trial_idx])
        label_0based = label_1based - self.cls_min

        img = self.transform(self._img_cache[label_1based])
        return eeg, img, label_0based


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 인덱스 분할 (stratified 8:1:1)
# ─────────────────────────────────────────────────────────────────────────────

def build_split_indices(
    mat_paths: list,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    class_filter: list = None,
):
    """
    각 피험자 × 클래스에 대해 stratified 8:1:1 분할 후
    (subj_idx, trial_idx) 형태의 리스트를 반환한다.

    class_filter : None이면 1~9 전체, 리스트 지정 시 해당 클래스만 포함.
    """
    classes = class_filter if class_filter is not None else list(range(1, 10))
    train_idx, val_idx, test_idx = [], [], []

    for s_idx, mat_path in enumerate(mat_paths):
        mat = loadmat(mat_path)
        y = mat["y"].squeeze().astype(np.int64)

        for cls in classes:
            cls_trials = np.where(y == cls)[0]
            if len(cls_trials) == 0:
                continue

            rng = np.random.RandomState(seed + s_idx * 100 + cls)
            rng.shuffle(cls_trials)

            n = len(cls_trials)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            for t in cls_trials[:n_train]:
                train_idx.append((s_idx, int(t)))
            for t in cls_trials[n_train:n_train + n_val]:
                val_idx.append((s_idx, int(t)))
            for t in cls_trials[n_train + n_val:]:
                test_idx.append((s_idx, int(t)))

    return train_idx, val_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────────
# 학습 함수 (단일 피험자 × 단일 그룹)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_subject_group(
    args, subject_id: int, group_id: int, device: torch.device, timestamp: str
):
    """
    단일 피험자(subject_id) × 단일 그룹(group_id)에 대해 독립적인 모델을 학습·저장한다.

    group_id : 1 → cls 1-3 | 2 → cls 4-6 | 3 → cls 7-9
               0 → 전체 9-class 모드
    """
    subj_str = f"{subject_id:02d}"
    mat_path = os.path.join(args.data_root, f"subj_{subj_str}.mat")
    if not os.path.isfile(mat_path):
        print(f"[WARN] 파일 없음, 건너뜀: {mat_path}")
        return None

    # 이미지 경로 결정 (VS 폴더 내 images/ 우선, 없으면 VI images/ 사용)
    img_root = args.img_root if args.img_root else find_img_root(
        args.data_root, fallback="./preproc_data_vi/images"
    )
    mat_paths = [mat_path]

    # ── 그룹 설정 ─────────────────────────────────────────────────────────────
    if group_id == 0:
        cls_list    = None
        num_classes = 9
        cls_min     = 1
        group_tag   = "9cls"
    else:
        cls_list    = GROUP_CLASSES[group_id]
        num_classes = 3
        cls_min     = cls_list[0]
        group_tag   = f"g{group_id}_cls{cls_list[0]}-{cls_list[-1]}"

    tag = f"[subj {subj_str}][{group_tag}]"
    print(f"\n{'='*60}")
    print(f"{tag} 학습 시작  |  device: {device}")
    print(f"{'='*60}")

    # ── EEG 채널 자동 감지 ───────────────────────────────────────────────────
    eeg_ch = detect_eeg_channels(mat_path)
    print(f"{tag} EEG 채널 수: {eeg_ch}")

    # ── 데이터 분할 ──────────────────────────────────────────────────────────
    train_idx, val_idx, test_idx = build_split_indices(
        mat_paths, seed=args.seed, class_filter=cls_list
    )
    print(
        f"{tag} 분할 완료 — "
        f"train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}"
    )
    if len(train_idx) == 0:
        print(f"[WARN] {tag}: 학습 데이터 없음. 건너뜀.")
        return None

    # ── 데이터로더 ────────────────────────────────────────────────────────────
    train_ds = EEGImageDatasetVS(
        mat_paths, img_root, train_idx, args.img_size,
        cls_list=cls_list, cls_min=cls_min,
    )
    val_ds = EEGImageDatasetVS(
        mat_paths, img_root, val_idx, args.img_size,
        cls_list=cls_list, cls_min=cls_min,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── 모델 ─────────────────────────────────────────────────────────────────
    ch_mult = [int(x) for x in args.ch_mult.split(",")]

    model = EEGDiffusionModel128(
        img_size=args.img_size,
        img_channels=3,
        eeg_channels=eeg_ch,
        num_classes=num_classes,
        num_timesteps=args.num_timesteps,
        base_channels=args.base_channels,
        ch_mult=ch_mult,
        time_dim=256,
        cond_dim=256,
        eeg_hidden_dim=256,
        cond_scale=2.0,
        n_res_blocks=args.n_res_blocks,
        lambda_rec=args.lambda_rec,
        lambda_ssim=args.lambda_ssim,
        lambda_percept=args.lambda_percept,
        percept_feat_dim=512,
        eeg_tf_heads=args.eeg_tf_heads,
        eeg_tf_layers=args.eeg_tf_layers,
        eeg_tf_dropout=args.eeg_tf_dropout,
    ).to(device)

    # EMA 복사본
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )

    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.05
    )

    # ── 저장 경로 ─────────────────────────────────────────────────────────────
    os.makedirs(args.ckpt_root, exist_ok=True)
    save_dir = os.path.join(
        args.ckpt_root,
        f"{timestamp}_vs{subj_str}_{group_tag}_{args.img_size}"
    )
    os.makedirs(save_dir, exist_ok=True)
    print(f"{tag} 체크포인트 → {save_dir}")

    # ── 메타 정보 ─────────────────────────────────────────────────────────────
    meta = dict(
        subject_id=subject_id,
        group_id=group_id,
        cls_list=cls_list,
        cls_min=cls_min,
        num_classes=num_classes,
        eeg_ch=eeg_ch,
        img_size=args.img_size,
        img_root=img_root,
        base_channels=args.base_channels,
        ch_mult=args.ch_mult,
        num_timesteps=args.num_timesteps,
        n_res_blocks=args.n_res_blocks,
        eeg_tf_heads=args.eeg_tf_heads,
        eeg_tf_layers=args.eeg_tf_layers,
        eeg_tf_dropout=args.eeg_tf_dropout,
        lambda_rec=args.lambda_rec,
        lambda_ssim=args.lambda_ssim,
        lambda_percept=args.lambda_percept,
    )

    # ── Augmenter ─────────────────────────────────────────────────────────────
    augmenter = EEGAugment(args)

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_train = []

        for step, (eeg, img, labels) in enumerate(train_loader):
            eeg    = eeg.to(device)
            img    = img.to(device)
            labels = labels.to(device)

            eeg = augmenter(eeg)   # EEG augmentation (train only)

            b = img.size(0)
            t = torch.randint(0, model.num_timesteps, (b,), device=device, dtype=torch.long)

            loss = model.p_losses(img, eeg, labels, t)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ema_update(model, ema_model, decay=args.ema_decay)

            epoch_train.append(loss.item())
            train_losses.append(loss.item())

            if step % args.log_interval == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"{tag}[Epoch {epoch:03d}/{args.epochs}] "
                    f"Step {step:04d}/{len(train_loader)} "
                    f"Loss {loss.item():.4f}  LR {lr_now:.2e}"
                )

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        epoch_val = []
        with torch.no_grad():
            for eeg, img, labels in val_loader:
                eeg    = eeg.to(device)
                img    = img.to(device)
                labels = labels.to(device)

                b = img.size(0)
                t = torch.randint(0, model.num_timesteps, (b,), device=device, dtype=torch.long)
                vloss = model.p_losses(img, eeg, labels, t)
                epoch_val.append(vloss.item())

        mean_train = float(np.mean(epoch_train)) if epoch_train else 0.0
        mean_val   = float(np.mean(epoch_val))   if epoch_val   else 0.0
        val_losses.append(mean_val)

        print(f"{tag}[Epoch {epoch:03d}] TrainLoss={mean_train:.4f}  ValLoss={mean_val:.4f}")

        # ── Best 체크포인트 (EMA만 저장 → 절반 크기) ────────────────────────
        if mean_val < best_val:
            best_val = mean_val
            best_ckpt_path = os.path.join(save_dir, "best.pt")
            torch.save(
                {"ema": ema_model.state_dict(), "config": meta},
                best_ckpt_path,
            )
            print(f"  ↳ {tag} Best 저장 (val={mean_val:.4f}) → {best_ckpt_path}")

        # ── 중간 체크포인트 (매 50에포크, EMA만) ─────────────────────────────
        if (epoch + 1) % 50 == 0:
            ep_path = os.path.join(save_dir, f"epoch_{epoch+1:04d}.pt")
            torch.save(
                {"ema": ema_model.state_dict(), "config": meta},
                ep_path,
            )
            print(f"  ↳ {tag} 중간 체크포인트 → {ep_path}")

    # ── 최종 체크포인트 (EMA만) ───────────────────────────────────────────────
    final_path = os.path.join(save_dir, "final.pt")
    torch.save(
        {"ema": ema_model.state_dict(), "config": meta},
        final_path,
    )
    print(f"{tag} 학습 완료. Final ckpt → {final_path}")

    # ── 손실 곡선 ─────────────────────────────────────────────────────────────
    np.save(os.path.join(save_dir, "train_loss.npy"), np.array(train_losses, dtype=np.float32))
    np.save(os.path.join(save_dir, "val_loss_epoch.npy"), np.array(val_losses, dtype=np.float32))

    plt.figure(figsize=(8, 4))
    plt.plot(val_losses, marker="o", label="val loss (per epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"VS Subj {subj_str} | {group_tag} REPA Val Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "val_loss_curve.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"{tag} 손실 곡선 → {fig_path}")

    return save_dir


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="preproc_for_gan_vs REPA EEG→Image 학습 (VS 전용, subject-wise)"
    )

    # 데이터
    parser.add_argument("--data_root", type=str, default="./preproc_for_gan_vs",
                        help="VS mat 파일 폴더 (preproc_for_gan_vs)")
    parser.add_argument("--img_root", type=str, default="",
                        help="클래스 이미지 폴더 (기본: data_root/images → ./preproc_data_vi/images)")
    parser.add_argument("--subject_ids", type=str, default="1",
                        help="예: '1' / '1-10' / '1,3,5' / '1-5,8,10' / 'all'")
    parser.add_argument("--img_size", type=int, default=128)

    # 학습
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # 모델 구조
    parser.add_argument("--num_timesteps", type=int, default=200)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--ch_mult", type=str, default="1,2,4,8",
                        help="콤마 구분 채널 배율, 예: '1,2,4,8'")
    parser.add_argument("--n_res_blocks", type=int, default=5)

    # EEG 인코더
    parser.add_argument("--eeg_tf_heads", type=int, default=4)
    parser.add_argument("--eeg_tf_layers", type=int, default=2)
    parser.add_argument("--eeg_tf_dropout", type=float, default=0.1)

    # 손실 가중치
    parser.add_argument("--lambda_rec",  type=float, default=0.02)
    parser.add_argument("--lambda_ssim", type=float, default=0.05)
    parser.add_argument("--lambda_percept", type=float, default=0.05)

    # EEG Augmentation
    parser.add_argument("--aug_noise_std",  type=float, default=0.02,
                        help="Gaussian noise std (0=비활성)")
    parser.add_argument("--aug_scale_min",  type=float, default=0.9,
                        help="Amplitude scale 최솟값")
    parser.add_argument("--aug_scale_max",  type=float, default=1.1,
                        help="Amplitude scale 최댓값")
    parser.add_argument("--aug_ch_drop_p",  type=float, default=0.1,
                        help="Channel dropout 확률 (0=비활성)")
    parser.add_argument("--aug_time_shift", type=int,   default=20,
                        help="Time shift 최대 샘플 수 (0=비활성)")

    # 로그 / 저장
    parser.add_argument("--ckpt_root", type=str, default="./checkpoints_vs_repa",
                        help="체크포인트 저장 루트 폴더")
    parser.add_argument("--log_interval", type=int, default=10)

    # 그룹
    parser.add_argument(
        "--group_ids", type=str, default="1,2,3",
        help=(
            "학습할 클래스 그룹. 예: '1' / '1,2,3' / 'all' / '0'(전체 9-class)\n"
            "  1 → cls 1-3\n  2 → cls 4-6\n  3 → cls 7-9\n  0 → 9-class 전체"
        ),
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # ── 피험자 목록 ──────────────────────────────────────────────────────────
    all_mat = sorted(glob.glob(os.path.join(args.data_root, "subj_*.mat")))
    if not all_mat:
        raise FileNotFoundError(
            f"mat 파일을 찾을 수 없습니다: {args.data_root}/subj_*.mat"
        )

    subject_list = parse_subject_ids(args.subject_ids, all_mat)

    # ── 그룹 목록 ─────────────────────────────────────────────────────────────
    gids_str = args.group_ids.strip()
    if gids_str == "0":
        group_list = [0]
    else:
        group_list = parse_group_ids(gids_str)

    total = len(subject_list) * len(group_list)
    print(f"[INFO] VS 데이터 경로: {args.data_root}")
    print(f"[INFO] 피험자: {subject_list}")
    print(f"[INFO] 그룹:   {group_list}  (0=9-class 전체)")
    print(f"[INFO] 총 {total}개 모델 학습 예정")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {}
    for sid in subject_list:
        for gid in group_list:
            key = (sid, gid)
            save_dir = train_one_subject_group(args, sid, gid, device, timestamp)
            results[key] = save_dir

    print("\n" + "="*60)
    print("[INFO] 전체 학습 완료 요약")
    print("="*60)
    for (sid, gid), d in results.items():
        gtag = f"g{gid}" if gid != 0 else "9cls"
        status = d if d else "건너뜀"
        print(f"  VS subj {sid:02d} [{gtag}] → {status}")


if __name__ == "__main__":
    main()
