"""
train_vs_test_rest.py
─────────────────────────────────────────────────────────────────────────────
VS(Visual Stimulation) 데이터로 학습 → REST(Resting State) 데이터로 이미지 생성
★ subject-wise + group 모드: 피험자별 × 클래스 그룹별 독립 모델

  [학습] preproc_for_gan_vs/subj_XX.mat  + images/  (cls 1~9)
  [테스트] preproc_data_rest/subj_XX.mat  (X만 존재, label 없음)
           → 더미 라벨로 DDIM/DDPM 생성

그룹
────
  1 → cls 1-3   2 → cls 4-6   3 → cls 7-9   0 → 9-class 전체

REST 데이터 특성
────────────────
- 경로  : preproc_data_rest/subj_XX.mat
- 구조  : X=(ch, time, trial)  — y(라벨) 없음
- 생성  : 더미 라벨 0 사용 (그룹 내 첫 번째 클래스 기준)

사용 예시
─────────
# 피험자 1~15, 그룹 1,2,3 각각 학습 → REST 이미지 생성
python train_vs_test_rest.py --subjects 1-15 --group_ids 1,2,3

# 학습만 (REST 생성 생략)
python train_vs_test_rest.py --subjects 1-15 --group_ids 1,2,3 --skip_test

# 이미 학습된 VS 체크포인트로 REST 생성만
python train_vs_test_rest.py --subjects 1 --group_ids 1 \\
    --skip_train --ckpt_path ./checkpoints_vs2rest/<dir>/best.pt

# DDIM 샘플링
python train_vs_test_rest.py --subjects 1-15 --group_ids 1,2,3 \\
    --use_ddim --sample_steps 50 --guidance_scale 3.0

# 9-class 전체 모드
python train_vs_test_rest.py --subjects 1 --group_ids 0
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
# 상수 / 유틸
# ─────────────────────────────────────────────────────────────────────────────

GROUP_CLASSES = {
    1: [1, 2, 3],
    2: [4, 5, 6],
    3: [7, 8, 9],
}


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


def parse_subject_ids(s: str, data_root: str) -> list:
    s = s.strip().lower()
    if s == "all":
        paths = sorted(glob.glob(os.path.join(data_root, "subj_*.mat")))
        return [int(os.path.basename(p).replace("subj_", "").replace(".mat", ""))
                for p in paths]
    ids = []
    for token in s.split(","):
        token = token.strip()
        if "-" in token:
            a, b = token.split("-", 1)
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(token))
    return ids


def parse_group_ids(s: str) -> list:
    s = s.strip().lower()
    if s == "all":
        return [1, 2, 3]
    if s == "0":
        return [0]
    ids = []
    for token in s.split(","):
        token = token.strip()
        if "-" in token:
            a, b = token.split("-", 1)
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(token))
    for gid in ids:
        if gid not in (0, 1, 2, 3):
            raise ValueError(f"group_id는 0~3 중 하나여야 합니다. (입력: {gid})")
    return ids


def detect_eeg_channels(mat_path: str) -> int:
    try:
        return int(loadmat(mat_path)["X"].shape[0])
    except Exception as e:
        print(f"[WARN] EEG 채널 감지 실패: {e}. 기본값 32 사용.")
        return 32


# ─────────────────────────────────────────────────────────────────────────────
# EEG Augmentation
# ─────────────────────────────────────────────────────────────────────────────

class EEGAugment:
    """학습 시 EEG 배치 (B, ch, time) 에 적용하는 augmentation."""

    def __init__(self, args):
        self.noise_std  = getattr(args, "aug_noise_std",  0.02)
        self.scale_min  = getattr(args, "aug_scale_min",  0.8)
        self.scale_max  = getattr(args, "aug_scale_max",  1.2)
        self.ch_drop_p  = getattr(args, "aug_ch_drop_p",  0.1)
        self.time_shift = getattr(args, "aug_time_shift",  20)

    @torch.no_grad()
    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        B, C, T = eeg.shape
        dev = eeg.device
        if self.noise_std > 0:
            eeg = eeg + torch.randn_like(eeg) * self.noise_std
        if self.scale_min != 1.0 or self.scale_max != 1.0:
            scale = (torch.rand(B, 1, 1, device=dev)
                     * (self.scale_max - self.scale_min) + self.scale_min)
            eeg = eeg * scale
        if self.ch_drop_p > 0:
            mask = (torch.rand(B, C, 1, device=dev) > self.ch_drop_p).float()
            eeg = eeg * mask
        if self.time_shift > 0:
            shifts = torch.randint(-self.time_shift, self.time_shift + 1, (B,))
            eeg = torch.stack([torch.roll(eeg[i], shifts[i].item(), dims=-1)
                               for i in range(B)], dim=0)
        return eeg


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — VS 학습용 (라벨 있음)
# ─────────────────────────────────────────────────────────────────────────────

class EEGImageDatasetVS(Dataset):
    """VS 학습용 데이터셋. cls_list/cls_min 으로 그룹 모드 지원."""

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
        self.indices = indices
        self.cls_min = cls_min
        load_classes = cls_list if cls_list is not None else list(range(1, 10))

        self.eegs, self.labels = [], []
        for p in mat_paths:
            mat = loadmat(p)
            X = mat["X"]
            y = mat["y"].squeeze()
            self.eegs.append(torch.from_numpy(X.astype(np.float32)).permute(2, 0, 1))
            self.labels.append(y.astype(np.int64))

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self._img_cache = {}
        for cls in load_classes:
            p = os.path.join(img_root, f"{cls:02d}.png")
            if os.path.isfile(p):
                self._img_cache[cls] = Image.open(p).convert("RGB")
            else:
                raise FileNotFoundError(f"클래스 이미지 없음: {p}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s_idx, t_idx = self.indices[idx]
        eeg     = self.eegs[s_idx][t_idx]
        label_1 = int(self.labels[s_idx][t_idx])
        label_0 = label_1 - self.cls_min
        img     = self.transform(self._img_cache[label_1])
        return eeg, img, label_0


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — REST 추론용 (라벨 없음)
# ─────────────────────────────────────────────────────────────────────────────

class RestingEEGDataset(Dataset):
    """REST EEG 데이터셋. X=(ch, time, trial), y 없음."""

    def __init__(self, mat_path: str):
        super().__init__()
        mat = loadmat(mat_path)
        if "X" in mat:
            X = mat["X"]
        else:
            X = None
            for k, v in mat.items():
                if k.startswith("__"):
                    continue
                if isinstance(v, np.ndarray) and v.ndim == 3:
                    X = v
                    break
            if X is None:
                raise ValueError(f"3D EEG 배열을 찾을 수 없습니다: {mat_path}")

        # (ch, time, trial) → (trial, ch, time)
        self.eeg = torch.from_numpy(X.astype(np.float32)).permute(2, 0, 1)

    def __len__(self):
        return self.eeg.size(0)

    def __getitem__(self, idx):
        return self.eeg[idx], idx


# ─────────────────────────────────────────────────────────────────────────────
# 분할 함수 (VS 학습용)
# ─────────────────────────────────────────────────────────────────────────────

def build_split(mat_paths: list, seed: int = 42, class_filter: list = None):
    """stratified 8:1:1 분할."""
    classes = class_filter if class_filter is not None else list(range(1, 10))
    train_idx, val_idx, test_idx = [], [], []
    for s_idx, p in enumerate(mat_paths):
        y = loadmat(p)["y"].squeeze().astype(np.int64)
        for cls in classes:
            cls_trials = np.where(y == cls)[0]
            if len(cls_trials) == 0:
                continue
            rng = np.random.RandomState(seed + s_idx * 100 + cls)
            rng.shuffle(cls_trials)
            n = len(cls_trials)
            n_train = int(n * 0.8)
            n_val   = int(n * 0.1)
            for t in cls_trials[:n_train]:
                train_idx.append((s_idx, int(t)))
            for t in cls_trials[n_train:n_train + n_val]:
                val_idx.append((s_idx, int(t)))
            for t in cls_trials[n_train + n_val:]:
                test_idx.append((s_idx, int(t)))
    return train_idx, val_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────────
# 학습: VS 단일 피험자 × 단일 그룹
# ─────────────────────────────────────────────────────────────────────────────

def train_one_vs_group(
    args, subject_id: int, group_id: int, device: torch.device, timestamp: str
) -> str | None:
    subj_str = f"{subject_id:02d}"
    mat_path = os.path.join(args.vs_root, f"subj_{subj_str}.mat")
    if not os.path.isfile(mat_path):
        print(f"[WARN] VS 파일 없음, 건너뜀: {mat_path}")
        return None

    # 그룹 설정
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

    tag = f"[VS subj {subj_str}][{group_tag}]"
    print(f"\n{'='*65}")
    print(f"{tag} 학습 시작  |  device: {device}")
    print(f"{'='*65}")

    mat_paths = [mat_path]
    img_root  = os.path.join(args.vs_root, "images")
    if not os.path.isdir(img_root):
        img_root = os.path.join(args.vi_root, "images")
    eeg_ch = detect_eeg_channels(mat_path)
    print(f"{tag} EEG 채널: {eeg_ch}, 이미지 경로: {img_root}")

    train_idx, val_idx, _ = build_split(mat_paths, seed=args.seed, class_filter=cls_list)
    print(f"{tag} 분할 — train: {len(train_idx)}, val: {len(val_idx)}")
    if len(train_idx) == 0:
        print(f"[WARN] {tag}: 학습 데이터 없음. 건너뜀.")
        return None

    train_ds = EEGImageDatasetVS(mat_paths, img_root, train_idx, args.img_size,
                                 cls_list=cls_list, cls_min=cls_min)
    val_ds   = EEGImageDatasetVS(mat_paths, img_root, val_idx,   args.img_size,
                                 cls_list=cls_list, cls_min=cls_min)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

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

    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.05
    )

    save_dir = os.path.join(
        args.ckpt_root,
        f"{timestamp}_vs{subj_str}_{group_tag}_{args.img_size}"
    )
    os.makedirs(save_dir, exist_ok=True)
    print(f"{tag} 체크포인트 → {save_dir}")

    meta = dict(
        subject_id=subject_id,
        group_id=group_id,
        cls_list=cls_list,
        cls_min=cls_min,
        num_classes=num_classes,
        eeg_ch=eeg_ch,
        img_size=args.img_size,
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

    augmenter = EEGAugment(args)

    train_losses, val_losses = [], []
    best_val       = float("inf")
    best_ckpt_path = None

    for epoch in range(args.epochs):
        model.train()
        epoch_train = []

        for step, (eeg, img, label_0) in enumerate(train_loader):
            eeg     = eeg.to(device)
            img     = img.to(device)
            label_0 = label_0.to(device)

            eeg = augmenter(eeg)   # EEG augmentation (train only)

            b = img.size(0)
            t = torch.randint(0, model.num_timesteps, (b,), device=device, dtype=torch.long)
            loss = model.p_losses(img, eeg, label_0, t)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ema_update(model, ema_model, decay=args.ema_decay)

            epoch_train.append(loss.item())
            train_losses.append(loss.item())

            if step % args.log_interval == 0:
                print(
                    f"{tag}[Epoch {epoch:03d}/{args.epochs}] "
                    f"Step {step:04d}/{len(train_loader)}  "
                    f"Loss {loss.item():.4f}  LR {scheduler.get_last_lr()[0]:.2e}"
                )

        model.eval()
        epoch_val = []
        with torch.no_grad():
            for eeg, img, label_0 in val_loader:
                eeg     = eeg.to(device)
                img     = img.to(device)
                label_0 = label_0.to(device)
                b = img.size(0)
                t = torch.randint(0, model.num_timesteps, (b,), device=device, dtype=torch.long)
                epoch_val.append(model.p_losses(img, eeg, label_0, t).item())

        mean_train = float(np.mean(epoch_train)) if epoch_train else 0.0
        mean_val   = float(np.mean(epoch_val))   if epoch_val   else 0.0
        val_losses.append(mean_val)
        print(f"{tag}[Epoch {epoch:03d}] TrainLoss={mean_train:.4f}  ValLoss={mean_val:.4f}")

        # ── Best 체크포인트 (EMA만) ───────────────────────────────────────────
        if mean_val < best_val:
            best_val = mean_val
            best_ckpt_path = os.path.join(save_dir, "best.pt")
            torch.save({"ema": ema_model.state_dict(), "config": meta}, best_ckpt_path)
            print(f"  ↳ {tag} Best (val={mean_val:.4f}) → {best_ckpt_path}")

        # ── 중간 체크포인트 (매 50에포크, EMA만) ─────────────────────────────
        if (epoch + 1) % 50 == 0:
            torch.save(
                {"ema": ema_model.state_dict(), "config": meta},
                os.path.join(save_dir, f"epoch_{epoch+1:04d}.pt"),
            )

    # ── 최종 체크포인트 (EMA만) ───────────────────────────────────────────────
    final_path = os.path.join(save_dir, "final.pt")
    torch.save({"ema": ema_model.state_dict(), "config": meta}, final_path)
    print(f"{tag} 학습 완료. Final → {final_path}")

    np.save(os.path.join(save_dir, "train_loss.npy"), np.array(train_losses, dtype=np.float32))
    np.save(os.path.join(save_dir, "val_loss.npy"),   np.array(val_losses,   dtype=np.float32))
    plt.figure(figsize=(8, 4))
    plt.plot(val_losses, marker="o", label="val loss (VS)")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"VS subj{subj_str} | {group_tag} Val Loss")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_loss_curve.png"))
    plt.close()

    return best_ckpt_path or final_path


# ─────────────────────────────────────────────────────────────────────────────
# 테스트: REST 데이터로 이미지 생성 (라벨 없음 → 더미 라벨 0 사용)
# ─────────────────────────────────────────────────────────────────────────────

def test_one_rest_group(
    args, subject_id: int, group_id: int, ckpt_path: str, device: torch.device
):
    """VS 모델로 REST subj_{subject_id}.mat 에서 이미지 생성."""
    subj_str = f"{subject_id:02d}"
    mat_path = os.path.join(args.rest_root, f"subj_{subj_str}.mat")
    if not os.path.isfile(mat_path):
        print(f"[WARN] REST 파일 없음, 건너뜀: {mat_path}")
        return None

    print(f"\n{'='*65}")
    print(f"  [REST 생성] subj {subj_str}  |  ckpt: {ckpt_path}")
    print(f"{'='*65}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get("config", {})

    img_size      = cfg.get("img_size",      args.img_size)
    base_channels = cfg.get("base_channels", args.base_channels)
    n_ts          = cfg.get("num_timesteps", args.num_timesteps)
    n_res         = cfg.get("n_res_blocks",  args.n_res_blocks)
    ch_mult_str   = cfg.get("ch_mult",       args.ch_mult)
    eeg_ch        = cfg.get("eeg_ch",        0)
    eeg_heads     = cfg.get("eeg_tf_heads",  args.eeg_tf_heads)
    eeg_layers    = cfg.get("eeg_tf_layers", args.eeg_tf_layers)
    eeg_dropout   = cfg.get("eeg_tf_dropout",args.eeg_tf_dropout)
    num_classes   = cfg.get("num_classes",   3 if group_id != 0 else 9)
    cls_list      = cfg.get("cls_list",      None if group_id == 0 else GROUP_CLASSES[group_id])
    if cls_list is not None:
        cls_list = list(cls_list)

    ch_mult = ([int(x) for x in ch_mult_str.split(",")]
               if isinstance(ch_mult_str, str) else list(ch_mult_str))

    if eeg_ch == 0:
        eeg_ch = detect_eeg_channels(mat_path)

    group_tag = ("9cls" if group_id == 0
                 else f"g{group_id}_cls{cls_list[0]}-{cls_list[-1]}")

    print(f"[INFO] num_classes={num_classes}, group={group_tag}, EEG ch={eeg_ch}")

    model = EEGDiffusionModel128(
        img_size=img_size,
        img_channels=3,
        eeg_channels=eeg_ch,
        num_classes=num_classes,
        num_timesteps=n_ts,
        base_channels=base_channels,
        ch_mult=ch_mult,
        time_dim=256,
        cond_dim=256,
        eeg_hidden_dim=256,
        cond_scale=2.0,
        n_res_blocks=n_res,
        eeg_tf_heads=eeg_heads,
        eeg_tf_layers=eeg_layers,
        eeg_tf_dropout=eeg_dropout,
    ).to(device)

    # EMA 가중치 로드
    state = ckpt.get("ema", ckpt.get("model"))
    model.load_state_dict(state)
    model.eval()
    print("[INFO] 모델 로드 완료 (EMA 가중치 우선)")

    # REST 데이터셋 (라벨 없음)
    ds = RestingEEGDataset(mat_path)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    print(f"[INFO] REST trials: {len(ds)}")

    # 출력 폴더
    ckpt_dirname = os.path.basename(os.path.dirname(ckpt_path))
    out_dir = os.path.join(
        args.samples_root, ckpt_dirname,
        f"rest{subj_str}_{group_tag}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] 결과 저장 → {out_dir}")

    to_pil  = T.ToPILImage()
    denorm  = lambda x: (x.clamp(-1, 1) + 1.0) * 0.5
    mode    = "ddim" if args.use_ddim else "ddpm"
    total_gen = 0

    with torch.no_grad():
        for eeg, trial_idx in loader:
            eeg = eeg.to(device)
            b   = eeg.size(0)

            # REST는 라벨 없음 → 더미 라벨 0 (그룹 내 첫 번째 클래스)
            dummy_labels = torch.zeros(b, device=device, dtype=torch.long)

            if args.use_ddim:
                gen = model.sample_ddim(
                    eeg=eeg,
                    labels=dummy_labels,
                    num_steps=args.sample_steps,
                    guidance_scale=args.guidance_scale,
                    eta=args.ddim_eta,
                )
            else:
                gen = model.sample(
                    eeg=eeg,
                    labels=dummy_labels,
                    num_steps=args.sample_steps,
                    guidance_scale=args.guidance_scale,
                )

            gen_d = denorm(gen).cpu()

            for i in range(b):
                tidx = int(trial_idx[i].item())
                fname = f"rest{subj_str}_{group_tag}_trial{tidx:04d}_{mode}_GEN.png"
                to_pil(gen_d[i]).save(os.path.join(out_dir, fname))
                total_gen += 1

    print(f"[REST 생성 완료] {total_gen}개 이미지 저장")
    print(f"[결과 폴더] {out_dir}")
    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VS 학습 → REST 이미지 생성 (Subject-wise + Group)"
    )

    # 데이터 경로
    parser.add_argument("--vs_root",   type=str, default="./preproc_for_gan_vs",
                        help="VS mat + images 폴더")
    parser.add_argument("--vi_root",   type=str, default="./preproc_data_vi",
                        help="이미지 폴더 fallback (vs_root/images 없을 때)")
    parser.add_argument("--rest_root", type=str, default="./preproc_data_rest",
                        help="REST mat 폴더 (X만, 라벨 없음)")
    parser.add_argument("--subjects",  type=str, default="1",
                        help="피험자: '1' / '1-15' / '1,3,5' / 'all'")
    parser.add_argument("--group_ids", type=str, default="1,2,3",
                        help="그룹: '1,2,3' / 'all' / '0'(9-class 전체)")
    parser.add_argument("--img_size",  type=int, default=128)

    # 학습/테스트 제어
    parser.add_argument("--skip_train", action="store_true", default=False,
                        help="학습 생략 (--ckpt_path 직접 지정 필요)")
    parser.add_argument("--skip_test",  action="store_true", default=False,
                        help="REST 생성 생략 (학습만)")
    parser.add_argument("--ckpt_path",  type=str, default=None,
                        help="skip_train 시 사용할 체크포인트")

    # 학습 하이퍼파라미터
    parser.add_argument("--epochs",      type=int,   default=250)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--ema_decay",   type=float, default=0.999)

    # 모델 구조
    parser.add_argument("--num_timesteps", type=int,   default=200)
    parser.add_argument("--base_channels", type=int,   default=64)
    parser.add_argument("--ch_mult",       type=str,   default="1,2,4,4")
    parser.add_argument("--n_res_blocks",  type=int,   default=4)
    parser.add_argument("--eeg_tf_heads",  type=int,   default=4)
    parser.add_argument("--eeg_tf_layers", type=int,   default=2)
    parser.add_argument("--eeg_tf_dropout",type=float, default=0.1)

    # 손실 가중치
    parser.add_argument("--lambda_rec",  type=float, default=0.02)
    parser.add_argument("--lambda_ssim", type=float, default=0.05)
    parser.add_argument("--lambda_percept", type=float, default=0.05)

    # EEG Augmentation
    parser.add_argument("--aug_noise_std",  type=float, default=0.02)
    parser.add_argument("--aug_scale_min",  type=float, default=0.8)
    parser.add_argument("--aug_scale_max",  type=float, default=1.2)
    parser.add_argument("--aug_ch_drop_p",  type=float, default=0.1)
    parser.add_argument("--aug_time_shift", type=int,   default=20)

    # REST 생성 옵션
    parser.add_argument("--use_ddim",       action="store_true", default=False)
    parser.add_argument("--sample_steps",   type=int,   default=200)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--ddim_eta",       type=float, default=0.0)

    # 경로 / 기타
    parser.add_argument("--ckpt_root",    type=str, default="./checkpoints_vs2rest")
    parser.add_argument("--samples_root", type=str, default="./samples_vs2rest")
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--seed",         type=int, default=42)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    subject_list = parse_subject_ids(args.subjects, args.vs_root)
    group_list   = parse_group_ids(args.group_ids)

    total = len(subject_list) * len(group_list)
    print(f"[INFO] device       : {device}")
    print(f"[INFO] VS 경로      : {args.vs_root}")
    print(f"[INFO] REST 경로    : {args.rest_root}")
    print(f"[INFO] 피험자       : {subject_list}")
    print(f"[INFO] 그룹         : {group_list}  (0=9-class 전체)")
    print(f"[INFO] 총 {total}개 (피험자×그룹) 처리 예정")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results   = {}

    for sid in subject_list:
        for gid in group_list:
            key  = (sid, gid)
            gtag = f"g{gid}" if gid != 0 else "9cls"

            # ── 학습 (VS) ────────────────────────────────────────────────────
            if args.skip_train:
                if args.ckpt_path is None:
                    raise ValueError("--skip_train 사용 시 --ckpt_path 를 지정해야 합니다.")
                ckpt_path = args.ckpt_path
            else:
                ckpt_path = train_one_vs_group(args, sid, gid, device, timestamp)

            results[key] = {"ckpt": ckpt_path, "out": None}
            if ckpt_path is None:
                print(f"[SKIP] subj {sid:02d} [{gtag}]: 학습 실패 또는 데이터 없음.")
                continue

            # ── REST 이미지 생성 ─────────────────────────────────────────────
            if not args.skip_test:
                try:
                    out = test_one_rest_group(args, sid, gid, ckpt_path, device)
                    results[key]["out"] = out
                except Exception as e:
                    print(f"[ERROR] REST gen subj {sid:02d} [{gtag}]: {e}")

    # 요약
    print("\n" + "=" * 65)
    print("[INFO] 전체 완료 요약")
    print("=" * 65)
    for (sid, gid), r in results.items():
        gtag = f"g{gid}" if gid != 0 else "9cls"
        ckpt = r["ckpt"] or "건너뜀"
        out  = r["out"]  or "생성 없음"
        print(f"  subj {sid:02d} [{gtag}]")
        print(f"    ckpt → {ckpt}")
        print(f"    out  → {out}")


if __name__ == "__main__":
    main()
