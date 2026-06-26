"""
train_vs_cls_only.py
─────────────────────────────────────────────────────────────────────────────
preproc_for_gan_vs 데이터 / EEG Transformer → Classification 전용 학습
(UNet / Diffusion 없음)

▸ 모델   : model_eeg_transformer_cls_only.EEGTransformerClassifier (V2)
▸ 손실   : CrossEntropy (label smoothing) + Mixup
▸ 데이터 : preproc_for_gan_vs/subj_XX.mat  (X: ch×T×trial, y: trial)
▸ 학습   : 피험자별 독립 모델 (subject-wise), 3-class 그룹 또는 9-class 전체

개선 사항
─────────
  [아키텍처]
  - MultiScaleStem (k=7, 15, 31 병렬 Conv) → 단기·중기·장기 시간 패턴 포착
  - CLS Token → mean pooling보다 분류 특화
  - TransformerBlock × 4 + StochasticDepth → 과적합 억제
  - Transformer layers 2→4, heads 4→8

  [Augmentation]
  - Mixup (alpha=0.3): EEG 신호 보간 → 데이터 효율 향상
  - Frequency noise: FFT 진폭 perturbation → 주파수 도메인 다양성
  - 기존: Gaussian noise, Amplitude scale, Channel dropout, Time shift

  [학습 전략]
  - Linear Warmup + Cosine Decay LR 스케줄러
  - AdamW weight_decay=1e-3 (강화)

사용 예시
─────────
python train_vs_cls_only.py --subject_ids 1-20 --group_ids 1-3

# 강화된 설정
python train_vs_cls_only.py --subject_ids 1-20 --group_ids 1-3 \\
    --eeg_tf_layers 4 --eeg_tf_heads 8 --mixup_alpha 0.3 \\
    --stochastic_depth 0.1 --warmup_epochs 10
"""

import os
import math
import glob
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

from model_eeg_transformer_cls_only import EEGTransformerClassifier


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_subject_ids(s: str, all_mat: list) -> list:
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


GROUP_CLASSES = {
    1: [1, 2, 3],
    2: [4, 5, 6],
    3: [7, 8, 9],
}


def parse_group_ids(s: str) -> list:
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
    try:
        mat = loadmat(mat_path)
        return int(mat["X"].shape[0])
    except Exception as e:
        print(f"[WARN] eeg channel 자동 감지 실패: {e}. 기본값 32 사용.")
        return 32


# ─────────────────────────────────────────────────────────────────────────────
# LR 스케줄러: Linear Warmup + Cosine Decay
# ─────────────────────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int,
                                    eta_min_ratio: float = 0.05):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# EEG Augmentation
# ─────────────────────────────────────────────────────────────────────────────

class EEGAugment:
    """
    학습 시 EEG 배치 (B, ch, time) 에 적용하는 augmentation.

    augmentation 목록
    -----------------
    1. Gaussian noise      : N(0, σ) 노이즈 추가
    2. Amplitude scale     : U(scale_min, scale_max) 스케일링
    3. Channel dropout     : 채널을 확률 p 로 0
    4. Time shift          : ±max_shift 샘플 순환 이동
    5. Frequency noise     : FFT 진폭에 노이즈 추가 (주파수 도메인)
    """

    def __init__(self, args):
        self.noise_std    = getattr(args, "aug_noise_std",       0.02)
        self.scale_min    = getattr(args, "aug_scale_min",       0.9)
        self.scale_max    = getattr(args, "aug_scale_max",       1.1)
        self.ch_drop_p    = getattr(args, "aug_ch_drop_p",       0.1)
        self.time_shift   = getattr(args, "aug_time_shift",      20)
        self.freq_noise   = getattr(args, "aug_freq_noise_std",  0.05)

    @torch.no_grad()
    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        B, C, T = eeg.shape
        dev = eeg.device

        # 1. Gaussian noise
        if self.noise_std > 0:
            eeg = eeg + torch.randn_like(eeg) * self.noise_std

        # 2. Amplitude scaling
        if self.scale_min != 1.0 or self.scale_max != 1.0:
            scale = (torch.rand(B, 1, 1, device=dev)
                     * (self.scale_max - self.scale_min) + self.scale_min)
            eeg = eeg * scale

        # 3. Channel dropout
        if self.ch_drop_p > 0:
            mask = (torch.rand(B, C, 1, device=dev) > self.ch_drop_p).float()
            eeg  = eeg * mask

        # 4. Time shift
        if self.time_shift > 0:
            shifts  = torch.randint(-self.time_shift, self.time_shift + 1, (B,))
            shifted = [torch.roll(eeg[i], shifts[i].item(), dims=-1) for i in range(B)]
            eeg     = torch.stack(shifted, dim=0)

        # 5. Frequency domain noise (진폭 perturbation)
        if self.freq_noise > 0:
            X_f     = torch.fft.rfft(eeg, dim=-1)
            amp     = X_f.abs()
            noise   = torch.randn_like(amp) * self.freq_noise * amp.mean(dim=-1, keepdim=True)
            amp_new = (amp + noise).clamp(min=0)
            phase   = torch.angle(X_f)
            X_f_new = torch.polar(amp_new, phase)
            eeg     = torch.fft.irfft(X_f_new, n=T, dim=-1)

        return eeg


# ─────────────────────────────────────────────────────────────────────────────
# Mixup
# ─────────────────────────────────────────────────────────────────────────────

def mixup_batch(eeg: torch.Tensor, labels: torch.Tensor, alpha: float):
    """
    배치 내 두 샘플을 Beta(alpha, alpha)로 보간.

    Returns
    -------
    eeg_mix  : 혼합 EEG
    labels_a : 원본 라벨
    labels_b : 혼합 대상 라벨
    lam      : 혼합 비율 (0~1)
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(eeg.size(0), device=eeg.device)
    eeg_mix = lam * eeg + (1.0 - lam) * eeg[idx]
    return eeg_mix, labels, labels[idx], lam


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (EEG + label만, 이미지 불필요)
# ─────────────────────────────────────────────────────────────────────────────

class EEGClsDatasetVS(Dataset):
    def __init__(self, mat_paths: list, indices: list,
                 cls_list: list = None, cls_min: int = 1):
        super().__init__()
        self.indices = indices
        self.cls_min = cls_min

        self.eegs   = []
        self.labels = []
        for mat_path in mat_paths:
            mat = loadmat(mat_path)
            X = mat["X"]
            y = mat["y"].squeeze()
            self.eegs.append(
                torch.from_numpy(X.astype(np.float32)).permute(2, 0, 1)
            )
            self.labels.append(y.astype(np.int64))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        subj_idx, trial_idx = self.indices[idx]
        eeg          = self.eegs[subj_idx][trial_idx]
        label_1based = int(self.labels[subj_idx][trial_idx])
        return eeg, label_1based - self.cls_min


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 인덱스 분할 (stratified 8:1:1)
# ─────────────────────────────────────────────────────────────────────────────

def build_split_indices(mat_paths, seed=42, train_ratio=0.8, val_ratio=0.1,
                        class_filter=None):
    classes = class_filter if class_filter is not None else list(range(1, 10))
    train_idx, val_idx, test_idx = [], [], []
    for s_idx, mat_path in enumerate(mat_paths):
        mat = loadmat(mat_path)
        y   = mat["y"].squeeze().astype(np.int64)
        for cls in classes:
            cls_trials = np.where(y == cls)[0]
            if len(cls_trials) == 0:
                continue
            rng = np.random.RandomState(seed + s_idx * 100 + cls)
            rng.shuffle(cls_trials)
            n       = len(cls_trials)
            n_train = int(n * train_ratio)
            n_val   = int(n * val_ratio)
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

def train_one_subject_group(args, subject_id: int, group_id: int,
                            device: torch.device, timestamp: str):
    subj_str = f"{subject_id:02d}"
    mat_path = os.path.join(args.data_root, f"subj_{subj_str}.mat")
    if not os.path.isfile(mat_path):
        print(f"[WARN] 파일 없음, 건너뜀: {mat_path}")
        return None

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
    print(f"{tag} 분류 학습 시작  |  device: {device}")
    print(f"{'='*60}")

    eeg_ch = detect_eeg_channels(mat_path)
    print(f"{tag} EEG 채널 수: {eeg_ch}")

    # ── 데이터 분할 ──────────────────────────────────────────────────────────
    train_idx, val_idx, test_idx = build_split_indices(
        mat_paths, seed=args.seed, class_filter=cls_list
    )
    print(f"{tag} 분할 — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    if len(train_idx) == 0:
        print(f"[WARN] {tag}: 학습 데이터 없음. 건너뜀.")
        return None

    # ── 데이터로더 ────────────────────────────────────────────────────────────
    train_ds = EEGClsDatasetVS(mat_paths, train_idx, cls_list, cls_min)
    val_ds   = EEGClsDatasetVS(mat_paths, val_idx,   cls_list, cls_min)
    test_ds  = EEGClsDatasetVS(mat_paths, test_idx,  cls_list, cls_min)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── 모델 ─────────────────────────────────────────────────────────────────
    model = EEGTransformerClassifier(
        eeg_channels=eeg_ch,
        num_classes=num_classes,
        eeg_hidden_dim=args.eeg_hidden_dim,
        out_dim=args.out_dim,
        n_heads=args.eeg_tf_heads,
        n_layers=args.eeg_tf_layers,
        tf_dropout=args.eeg_tf_dropout,
        cls_dropout=args.cls_dropout,
        stochastic_depth=args.stochastic_depth,
    ).to(device)

    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_epochs * len(train_loader))
    scheduler   = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── 저장 경로 ─────────────────────────────────────────────────────────────
    save_dir = os.path.join(
        args.ckpt_root,
        f"{timestamp}_vs{subj_str}_{group_tag}_cls"
    )
    os.makedirs(save_dir, exist_ok=True)
    print(f"{tag} 체크포인트 → {save_dir}")

    meta = dict(
        subject_id=subject_id, group_id=group_id,
        cls_list=cls_list, cls_min=cls_min, num_classes=num_classes,
        eeg_ch=eeg_ch, eeg_hidden_dim=args.eeg_hidden_dim, out_dim=args.out_dim,
        eeg_tf_heads=args.eeg_tf_heads, eeg_tf_layers=args.eeg_tf_layers,
        eeg_tf_dropout=args.eeg_tf_dropout, cls_dropout=args.cls_dropout,
        stochastic_depth=args.stochastic_depth,
    )

    augmenter = EEGAugment(args)

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    train_accs, val_accs     = [], []
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        ep_loss, ep_acc = [], []

        for step, (eeg, labels) in enumerate(train_loader):
            eeg    = augmenter(eeg.to(device))
            labels = labels.to(device)

            # Mixup 적용
            if args.mixup_alpha > 0:
                eeg_mix, la, lb, lam = mixup_batch(eeg, labels, args.mixup_alpha)
                loss, acc = model.compute_loss_mixup(
                    eeg_mix, la, lb, lam, label_smoothing=args.label_smoothing
                )
            else:
                loss, acc = model.compute_loss(eeg, labels, label_smoothing=args.label_smoothing)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            ep_loss.append(loss.item())
            ep_acc.append(acc)

            if step % args.log_interval == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"{tag}[Epoch {epoch:03d}/{args.epochs}] "
                    f"Step {step:04d}/{len(train_loader)} "
                    f"Loss {loss.item():.4f}  Acc {acc:.3f}  LR {lr_now:.2e}"
                )

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        vl, va = [], []
        with torch.no_grad():
            for eeg, labels in val_loader:
                loss, acc = model.compute_loss(
                    eeg.to(device), labels.to(device), label_smoothing=0.0
                )
                vl.append(loss.item())
                va.append(acc)

        mean_train_loss = float(np.mean(ep_loss))
        mean_train_acc  = float(np.mean(ep_acc))
        mean_val_loss   = float(np.mean(vl)) if vl else 0.0
        mean_val_acc    = float(np.mean(va)) if va else 0.0

        train_losses.append(mean_train_loss)
        train_accs.append(mean_train_acc)
        val_losses.append(mean_val_loss)
        val_accs.append(mean_val_acc)

        print(
            f"{tag}[Epoch {epoch:03d}] "
            f"TrainLoss={mean_train_loss:.4f} TrainAcc={mean_train_acc:.3f}  "
            f"ValLoss={mean_val_loss:.4f} ValAcc={mean_val_acc:.3f}"
        )

        # ── Best 체크포인트 (val accuracy 기준) ──────────────────────────────
        if mean_val_acc > best_val_acc:
            best_val_acc = mean_val_acc
            torch.save({"model": model.state_dict(), "config": meta,
                        "epoch": epoch, "val_acc": best_val_acc},
                       os.path.join(save_dir, "best.pt"))
            print(f"  ↳ {tag} Best 저장 (val_acc={best_val_acc:.3f})")

        # ── 중간 체크포인트 (50에포크마다) ───────────────────────────────────
        if (epoch + 1) % 50 == 0:
            torch.save({"model": model.state_dict(), "config": meta, "epoch": epoch},
                       os.path.join(save_dir, f"epoch_{epoch+1:04d}.pt"))

    # ── 최종 체크포인트 ───────────────────────────────────────────────────────
    torch.save({"model": model.state_dict(), "config": meta},
               os.path.join(save_dir, "final.pt"))
    print(f"{tag} 학습 완료.")

    # ── Test 평가 ─────────────────────────────────────────────────────────────
    ckpt = torch.load(os.path.join(save_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ta = []
    with torch.no_grad():
        for eeg, labels in test_loader:
            _, acc = model.compute_loss(eeg.to(device), labels.to(device), label_smoothing=0.0)
            ta.append(acc)
    test_acc = float(np.mean(ta)) if ta else 0.0
    print(f"{tag} Test Accuracy (best ckpt): {test_acc:.4f}")
    np.save(os.path.join(save_dir, "test_acc.npy"), np.array([test_acc]))

    # ── 곡선 저장 ─────────────────────────────────────────────────────────────
    np.save(os.path.join(save_dir, "train_loss.npy"), np.array(train_losses, dtype=np.float32))
    np.save(os.path.join(save_dir, "val_loss.npy"),   np.array(val_losses,   dtype=np.float32))
    np.save(os.path.join(save_dir, "train_acc.npy"),  np.array(train_accs,   dtype=np.float32))
    np.save(os.path.join(save_dir, "val_acc.npy"),    np.array(val_accs,     dtype=np.float32))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, label="train"); axes[0].plot(val_losses, label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(train_accs, label="train"); axes[1].plot(val_accs, label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True)
    plt.suptitle(f"VS Subj {subj_str} | {group_tag} | TestAcc={test_acc:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curve.png"))
    plt.close()

    return save_dir


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VS EEG Transformer Classification 전용 학습 (V2)"
    )

    # 데이터
    parser.add_argument("--data_root",   type=str, default="./preproc_for_gan_vs")
    parser.add_argument("--subject_ids", type=str, default="1",
                        help="예: '1' / '1-10' / '1,3,5' / 'all'")
    parser.add_argument("--group_ids",   type=str, default="1,2,3",
                        help="1→cls1-3 / 2→cls4-6 / 3→cls7-9 / 0→9class 전체")

    # 학습
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_epochs",   type=int,   default=10,
                        help="LR warmup 에포크 수")
    parser.add_argument("--mixup_alpha",     type=float, default=0.3,
                        help="Mixup Beta 분포 alpha (0=비활성)")

    # 모델 구조
    parser.add_argument("--eeg_hidden_dim",   type=int,   default=256)
    parser.add_argument("--out_dim",          type=int,   default=256)
    parser.add_argument("--eeg_tf_heads",     type=int,   default=8)
    parser.add_argument("--eeg_tf_layers",    type=int,   default=4)
    parser.add_argument("--eeg_tf_dropout",   type=float, default=0.1)
    parser.add_argument("--cls_dropout",      type=float, default=0.3)
    parser.add_argument("--stochastic_depth", type=float, default=0.1,
                        help="StochasticDepth 최대 drop 비율 (0=비활성)")

    # EEG Augmentation
    parser.add_argument("--aug_noise_std",      type=float, default=0.02)
    parser.add_argument("--aug_scale_min",      type=float, default=0.9)
    parser.add_argument("--aug_scale_max",      type=float, default=1.1)
    parser.add_argument("--aug_ch_drop_p",      type=float, default=0.1)
    parser.add_argument("--aug_time_shift",     type=int,   default=20)
    parser.add_argument("--aug_freq_noise_std", type=float, default=0.05,
                        help="주파수 도메인 진폭 노이즈 std (0=비활성)")

    # 로그 / 저장
    parser.add_argument("--ckpt_root",    type=str, default="./checkpoints_vs_cls")
    parser.add_argument("--log_interval", type=int, default=10)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    all_mat = sorted(glob.glob(os.path.join(args.data_root, "subj_*.mat")))
    if not all_mat:
        raise FileNotFoundError(f"mat 파일 없음: {args.data_root}/subj_*.mat")

    subject_list = parse_subject_ids(args.subject_ids, all_mat)

    gids_str = args.group_ids.strip()
    if gids_str == "0":
        group_list = [0]
    else:
        group_list = parse_group_ids(gids_str)

    total = len(subject_list) * len(group_list)
    print(f"[INFO] 데이터 경로: {args.data_root}")
    print(f"[INFO] 피험자: {subject_list}")
    print(f"[INFO] 그룹:   {group_list}")
    print(f"[INFO] Transformer: layers={args.eeg_tf_layers}, heads={args.eeg_tf_heads}")
    print(f"[INFO] Mixup alpha={args.mixup_alpha}, StochasticDepth={args.stochastic_depth}")
    print(f"[INFO] Warmup={args.warmup_epochs}epoch, FreqNoise={args.aug_freq_noise_std}")
    print(f"[INFO] 총 {total}개 모델 학습 예정")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results   = {}

    for sid in subject_list:
        for gid in group_list:
            save_dir = train_one_subject_group(args, sid, gid, device, timestamp)
            results[(sid, gid)] = save_dir

    print("\n" + "="*60)
    print("[INFO] 전체 학습 완료 요약")
    print("="*60)
    for (sid, gid), d in results.items():
        gtag   = f"g{gid}" if gid != 0 else "9cls"
        status = d if d else "건너뜀"
        print(f"  VS subj {sid:02d} [{gtag}] → {status}")


if __name__ == "__main__":
    main()
