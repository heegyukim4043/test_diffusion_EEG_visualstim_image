"""
train_vs_repa_cls.py
─────────────────────────────────────────────────────────────────────────────
preproc_for_gan_vs 데이터 / EEG→Image 생성 + Classification 동시 학습
(Perceptual Loss + CrossEntropy 결합)

▸ 모델   : model_128_eegonly_transformer_repa_cls.EEGDiffusionModel128Cls
▸ 손실   : ε-MSE + λ_rec·L1 + λ_ssim·(1-SSIM) + λ_percept·(1-cos) + λ_cls·CE
▸ 데이터 : preproc_for_gan_vs/subj_XX.mat  (X: ch×T×trial, y: trial)
▸ 이미지 : preproc_data_vi/images/01.png ~ 09.png
▸ 학습   : 피험자별 독립 모델 (subject-wise), 3-class 그룹 또는 9-class 전체

사용 예시
─────────
# 단일 피험자, 그룹 1~3
python train_vs_repa_cls.py --subject_ids 1 --group_ids 1,2,3

# 범위 지정
python train_vs_repa_cls.py --subject_ids 1-20 --group_ids 1-3

# 9-class 전체 모드
python train_vs_repa_cls.py --subject_ids 1 --group_ids 0
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

from model_128_eegonly_transformer_repa_cls import EEGDiffusionModel128Cls


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


def find_img_root(vs_root: str, fallback: str = "./preproc_data_vi/images") -> str:
    vs_img = os.path.join(vs_root, "images")
    if os.path.isdir(vs_img):
        return vs_img
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# EEG Augmentation
# ─────────────────────────────────────────────────────────────────────────────

class EEGAugment:
    def __init__(self, args):
        self.noise_std  = getattr(args, "aug_noise_std",  0.02)
        self.scale_min  = getattr(args, "aug_scale_min",  0.9)
        self.scale_max  = getattr(args, "aug_scale_max",  1.1)
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
            shifted = [torch.roll(eeg[i], shifts[i].item(), dims=-1) for i in range(B)]
            eeg = torch.stack(shifted, dim=0)
        return eeg


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class EEGImageDatasetVS(Dataset):
    def __init__(self, mat_paths, img_root, indices, img_size=128,
                 cls_list=None, cls_min=1):
        super().__init__()
        self.img_root = img_root
        self.img_size = img_size
        self.indices  = indices
        self.cls_min  = cls_min
        load_classes  = cls_list if cls_list is not None else list(range(1, 10))

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

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
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
        eeg          = self.eegs[subj_idx][trial_idx]
        label_1based = int(self.labels[subj_idx][trial_idx])
        label_0based = label_1based - self.cls_min
        img = self.transform(self._img_cache[label_1based])
        return eeg, img, label_0based


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

    img_root  = args.img_root if args.img_root else find_img_root(
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
    print(f"{tag} 생성+분류 학습 시작  |  device: {device}")
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
    train_ds = EEGImageDatasetVS(mat_paths, img_root, train_idx, args.img_size,
                                 cls_list=cls_list, cls_min=cls_min)
    val_ds   = EEGImageDatasetVS(mat_paths, img_root, val_idx,   args.img_size,
                                 cls_list=cls_list, cls_min=cls_min)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── 모델 ─────────────────────────────────────────────────────────────────
    ch_mult = [int(x) for x in args.ch_mult.split(",")]

    model = EEGDiffusionModel128Cls(
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
        lambda_cls=args.lambda_cls,
        cls_dropout=args.cls_dropout,
    ).to(device)

    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.05
    )

    # ── 저장 경로 ─────────────────────────────────────────────────────────────
    save_dir = os.path.join(
        args.ckpt_root,
        f"{timestamp}_vs{subj_str}_{group_tag}_{args.img_size}"
    )
    os.makedirs(save_dir, exist_ok=True)
    print(f"{tag} 체크포인트 → {save_dir}")

    meta = dict(
        subject_id=subject_id, group_id=group_id,
        cls_list=cls_list, cls_min=cls_min, num_classes=num_classes,
        eeg_ch=eeg_ch, img_size=args.img_size, img_root=img_root,
        base_channels=args.base_channels, ch_mult=args.ch_mult,
        num_timesteps=args.num_timesteps, n_res_blocks=args.n_res_blocks,
        eeg_tf_heads=args.eeg_tf_heads, eeg_tf_layers=args.eeg_tf_layers,
        eeg_tf_dropout=args.eeg_tf_dropout,
        lambda_rec=args.lambda_rec, lambda_ssim=args.lambda_ssim,
        lambda_percept=args.lambda_percept, lambda_cls=args.lambda_cls,
    )

    augmenter = EEGAugment(args)

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        ep_loss, ep_acc = [], []

        for step, (eeg, img, labels) in enumerate(train_loader):
            eeg    = augmenter(eeg.to(device))
            img    = img.to(device)
            labels = labels.to(device)

            b = img.size(0)
            t = torch.randint(0, model.num_timesteps, (b,), device=device, dtype=torch.long)

            loss, acc = model.p_losses(img, eeg, labels, t)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ema_update(model, ema_model, decay=args.ema_decay)

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
            for eeg, img, labels in val_loader:
                eeg    = eeg.to(device)
                img    = img.to(device)
                labels = labels.to(device)
                b = img.size(0)
                t = torch.randint(0, model.num_timesteps, (b,), device=device, dtype=torch.long)
                vloss, vacc = model.p_losses(img, eeg, labels, t)
                vl.append(vloss.item())
                va.append(vacc)

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

        # ── Best 체크포인트 (val loss 기준, EMA만 저장) ───────────────────────
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_path = os.path.join(save_dir, "best.pt")
            torch.save({"ema": ema_model.state_dict(), "config": meta,
                        "epoch": epoch, "val_loss": best_val_loss,
                        "val_acc": mean_val_acc}, best_path)
            print(f"  ↳ {tag} Best 저장 (val_loss={best_val_loss:.4f}, val_acc={mean_val_acc:.3f}) → {best_path}")

        # ── 중간 체크포인트 (50에포크마다, EMA만) ────────────────────────────
        if (epoch + 1) % 50 == 0:
            ep_path = os.path.join(save_dir, f"epoch_{epoch+1:04d}.pt")
            torch.save({"ema": ema_model.state_dict(), "config": meta,
                        "epoch": epoch}, ep_path)
            print(f"  ↳ {tag} 중간 체크포인트 → {ep_path}")

    # ── 최종 체크포인트 ───────────────────────────────────────────────────────
    final_path = os.path.join(save_dir, "final.pt")
    torch.save({"ema": ema_model.state_dict(), "config": meta}, final_path)
    print(f"{tag} 학습 완료. Final ckpt → {final_path}")

    # ── 손실/정확도 곡선 ──────────────────────────────────────────────────────
    np.save(os.path.join(save_dir, "train_loss.npy"), np.array(train_losses, dtype=np.float32))
    np.save(os.path.join(save_dir, "val_loss.npy"),   np.array(val_losses,   dtype=np.float32))
    np.save(os.path.join(save_dir, "train_acc.npy"),  np.array(train_accs,   dtype=np.float32))
    np.save(os.path.join(save_dir, "val_acc.npy"),    np.array(val_accs,     dtype=np.float32))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, label="train"); axes[0].plot(val_losses, label="val")
    axes[0].set_title("Total Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(train_accs, label="train"); axes[1].plot(val_accs, label="val")
    axes[1].set_title("Cls Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True)
    plt.suptitle(f"VS Subj {subj_str} | {group_tag} | Gen+Cls")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curve.png"))
    plt.close()

    return save_dir


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VS EEG→Image 생성 + Classification 동시 학습"
    )

    # 데이터
    parser.add_argument("--data_root",   type=str, default="./preproc_for_gan_vs")
    parser.add_argument("--img_root",    type=str, default="")
    parser.add_argument("--subject_ids", type=str, default="1",
                        help="예: '1' / '1-10' / '1,3,5' / 'all'")
    parser.add_argument("--img_size",    type=int, default=128)
    parser.add_argument("--group_ids",   type=str, default="1,2,3",
                        help="1→cls1-3 / 2→cls4-6 / 3→cls7-9 / 0→9class 전체")

    # 학습
    parser.add_argument("--epochs",      type=int,   default=250)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--ema_decay",   type=float, default=0.999)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--seed",        type=int,   default=42)

    # 모델 구조
    parser.add_argument("--num_timesteps",  type=int, default=200)
    parser.add_argument("--base_channels",  type=int, default=64)
    parser.add_argument("--ch_mult",        type=str, default="1,2,4,8")
    parser.add_argument("--n_res_blocks",   type=int, default=5)
    parser.add_argument("--eeg_tf_heads",   type=int, default=4)
    parser.add_argument("--eeg_tf_layers",  type=int, default=2)
    parser.add_argument("--eeg_tf_dropout", type=float, default=0.1)
    parser.add_argument("--cls_dropout",    type=float, default=0.3)

    # 손실 가중치
    parser.add_argument("--lambda_rec",     type=float, default=0.02)
    parser.add_argument("--lambda_ssim",    type=float, default=0.05)
    parser.add_argument("--lambda_percept", type=float, default=0.05)
    parser.add_argument("--lambda_cls",     type=float, default=0.5,
                        help="Classification 손실 가중치")

    # EEG Augmentation
    parser.add_argument("--aug_noise_std",  type=float, default=0.02)
    parser.add_argument("--aug_scale_min",  type=float, default=0.9)
    parser.add_argument("--aug_scale_max",  type=float, default=1.1)
    parser.add_argument("--aug_ch_drop_p",  type=float, default=0.1)
    parser.add_argument("--aug_time_shift", type=int,   default=20)

    # 로그 / 저장
    parser.add_argument("--ckpt_root",    type=str, default="./checkpoints_vs_repa_cls")
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
    print(f"[INFO] 총 {total}개 모델 학습 예정")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

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
