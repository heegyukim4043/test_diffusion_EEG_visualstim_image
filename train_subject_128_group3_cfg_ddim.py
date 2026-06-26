# train_subject_128_group3.py
import os
import argparse
import random
from datetime import datetime

import numpy as np
from scipy.io import loadmat
from PIL import Image

import copy


import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

from model_128_cfg_ddim import EEGDiffusionModel128


@torch.no_grad()
def ema_update(model, ema_model, decay: float = 0.999):
    """
    model: 학습 중인 원본 모델
    ema_model: EMA를 유지하는 모델 (gradient X)
    decay: 0.999 ~ 0.9999 정도
    """
    msd = model.state_dict()
    for name, param in ema_model.state_dict().items():
        if name in msd:
            param.copy_(param * decay + msd[name] * (1.0 - decay))

# ---------------------------------------------------------
# Dataset: 한 subject, 특정 class-range (예: 1~3)만 쓰는 128x128용
# ---------------------------------------------------------
class EEGImageDatasetGroup128(Dataset):
    def __init__(
        self,
        mat_path: str,
        img_root: str,
        indices,
        img_size: int = 128,
        cls_low: int = 1,
        cls_high: int = 3,
    ):
        super().__init__()
        self.mat_path = mat_path
        self.img_root = img_root
        self.indices = np.array(indices, dtype=np.int64)
        self.img_size = img_size
        self.cls_low = cls_low
        self.cls_high = cls_high

        mat = loadmat(mat_path)
        X = mat["X"]        # (ch, time, trial)
        y = mat["y"].squeeze()  # (trial,)

        # (trial, ch, time)
        self.eeg = torch.from_numpy(X).float().permute(2, 0, 1)
        self.labels = y.astype(np.int64)

        self.transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        trial_idx = int(self.indices[idx])

        eeg = self.eeg[trial_idx]  # (C, T)
        label_orig = int(self.labels[trial_idx])  # 1~9 중 하나

        # 그룹 내 로컬 라벨: (cls_low..cls_high) → 0..(high-low)
        local_label = label_orig - self.cls_low  # 예: 1→0, 2→1, 3→2

        # 원본 이미지는 여전히 global label을 사용해 매핑
        img_path = os.path.join(self.img_root, f"{label_orig:02d}.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)  # (3, H, W) in [-1,1]

        return eeg, img, local_label


# ---------------------------------------------------------
# Helper: 시드 고정
# ---------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# 1 subject, 1 class-group(예: 1~3) 학습
# ---------------------------------------------------------
def train_one_subject_group(
    args,
    subject_id: int,
    group_idx: int,
    cls_low: int,
    cls_high: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subj_str = f"{subject_id:02d}"

    print("=" * 80)
    print(f"[128-group] Training subject {subj_str}, class {cls_low}~{cls_high} on device: {device}")
    print("=" * 80)

    # -----------------------------------------------------
    # 1) 데이터 인덱스(split) 만들기: 해당 그룹 class만 필터 후 8:1:1
    # -----------------------------------------------------
    mat_path = os.path.join(args.data_root, f"subj_{subj_str}.mat")
    img_root = os.path.join(args.data_root, "images")

    mat = loadmat(mat_path)
    y = mat["y"].squeeze().astype(np.int64)  # (trial,)

    # 해당 그룹에 속하는 trial만 선택
    mask_group = (y >= cls_low) & (y <= cls_high)
    all_indices = np.where(mask_group)[0]
    n_total = len(all_indices)

    if n_total < 10:
        print(f"[Subj {subj_str}][Group {cls_low}-{cls_high}] "
              f"Too few trials ({n_total}). Skipping.")
        return

    rng = np.random.RandomState(args.seed + subject_id * 10 + group_idx)
    rng.shuffle(all_indices)

    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    train_idx = all_indices[:n_train]
    val_idx = all_indices[n_train:n_train + n_val]
    test_idx = all_indices[n_train + n_val:]

    print(f"[Subj {subj_str}][Group {cls_low}-{cls_high}] total={n_total}, "
          f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # -----------------------------------------------------
    # 2) Dataset / DataLoader
    # -----------------------------------------------------
    train_ds = EEGImageDatasetGroup128(
        mat_path, img_root, train_idx,
        img_size=args.img_size,
        cls_low=cls_low, cls_high=cls_high,
    )
    val_ds = EEGImageDatasetGroup128(
        mat_path, img_root, val_idx,
        img_size=args.img_size,
        cls_low=cls_low, cls_high=cls_high,
    )
    test_ds = EEGImageDatasetGroup128(
        mat_path, img_root, test_idx,
        img_size=args.img_size,
        cls_low=cls_low, cls_high=cls_high,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # -----------------------------------------------------
    # 3) 모델 생성 (num_classes=3, base_channels=64 권장)
    # -----------------------------------------------------
    ch_mult = [int(x) for x in args.ch_mult.split(",")]
    model = EEGDiffusionModel128(
        img_size=args.img_size,
        img_channels=3,
        eeg_channels=32,
        num_classes=3,
        num_timesteps=args.num_timesteps,
        base_channels=args.base_channels,
        ch_mult=ch_mult,           # ★ 여기서 전달
        time_dim=256,
        cond_dim=256,
        eeg_hidden_dim=256,
        cond_scale=2.0,
        n_res_blocks=args.n_res_blocks,
        lambda_rec=args.lambda_rec,  #
        cond_drop_prob=args.cond_drop_prob, # ← 연결
    ).to(device)

    # EMA 모델 생성 (학습하지 않고, weight만 누적)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()

    ema_decay = 0.999  # or 0.9995 등

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # -----------------------------------------------------
    # 4) 로그/체크포인트 경로
    # -----------------------------------------------------
    os.makedirs(args.ckpt_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        args.ckpt_root,
        f"{timestamp}_subj{subj_str}_g{group_idx+1}_cls{cls_low}-{cls_high}_{args.img_size}"
    )
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Subj {subj_str}][Group {cls_low}-{cls_high}] "
          f"Checkpoints & logs → {save_dir}")

    train_losses = []
    val_losses = []
    best_val = float("inf")
    best_ckpt_path = None

    total_steps = args.epochs * len(train_loader)
    if total_steps == 0:
        print(f"[Subj {subj_str}][Group {cls_low}-{cls_high}] "
              f"No training steps (empty loader). Skipping.")
        return

    # 20% 진행마다 중간 체크포인트
    ckpt_steps = [
        int(total_steps * r)
        for r in [0.2, 0.4, 0.6, 0.8]
    ]
    ckpt_steps = sorted(list(set([s for s in ckpt_steps if s > 0])))

    global_step = 0

    # -----------------------------------------------------
    # 5) 학습 루프
    # -----------------------------------------------------
    for epoch in range(args.epochs):
        model.train()
        epoch_train_losses = []

        for batch_idx, (eeg, img, labels) in enumerate(train_loader):
            eeg = eeg.to(device)              # (B, C, T)
            img = img.to(device)              # (B, 3, H, W)
            labels = labels.to(device)        # (B,)

            b = img.size(0)
            t = torch.randint(
                low=0,
                high=model.num_timesteps,
                size=(b,),
                device=device,
                dtype=torch.long,
            )

            loss = model.p_losses(img, eeg, labels, t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # === 여기! EMA 업데이트 ===
            ema_update(model, ema_model, decay=ema_decay)

            epoch_train_losses.append(loss.item())
            train_losses.append(loss.item())

            global_step += 1

            if batch_idx % args.log_interval == 0:
                print(
                    f"[Subj {subj_str}][Group {cls_low}-{cls_high}] "
                    f"Epoch {epoch} Step {batch_idx}/{len(train_loader)} "
                    f"Loss {loss.item():.4f}"
                )

            # 중간 checkpoint 저장
            if global_step in ckpt_steps:
                ckpt_path = os.path.join(
                    save_dir,
                    f"subj{subj_str}_g{group_idx+1}_step{global_step}.pt"
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": {
                            "subject_id": subject_id,
                            "group_idx": group_idx,
                            "cls_low": cls_low,
                            "cls_high": cls_high,
                            "img_size": args.img_size,
                            "base_channels": args.base_channels,
                            "num_timesteps": args.num_timesteps,
                            "n_res_blocks": args.n_res_blocks,
                        "cond_drop_prob": args.cond_drop_prob,
                        },
                    },
                    ckpt_path,
                )
                print(
                    f"[Subj {subj_str}][Group {cls_low}-{cls_high}] "
                    f"Saved checkpoint at step {global_step} → {ckpt_path}"
                )

        # epoch 종료 후 validation
        model.eval()
        val_epoch_losses = []
        with torch.no_grad():
            for eeg, img, labels in val_loader:
                eeg = eeg.to(device)
                img = img.to(device)
                labels = labels.to(device)

                b = img.size(0)
                t = torch.randint(
                    low=0,
                    high=model.num_timesteps,
                    size=(b,),
                    device=device,
                    dtype=torch.long,
                )
                vloss = model.p_losses(img, eeg, labels, t)
                val_epoch_losses.append(vloss.item())

        mean_train = float(np.mean(epoch_train_losses)) if epoch_train_losses else 0.0
        mean_val = float(np.mean(val_epoch_losses)) if val_epoch_losses else 0.0
        val_losses.append(mean_val)

        print(
            f"[Subj {subj_str}][Group {cls_low}-{cls_high}] "
            f"Epoch {epoch} TrainLoss {mean_train:.4f} ValLoss {mean_val:.4f}"
        )

        # best val 기준으로 best ckpt 저장
        if mean_val < best_val:
            best_val = mean_val
            best_ckpt_path = os.path.join(
                save_dir,
                f"subj{subj_str}_g{group_idx+1}_best.pt"
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),  # ← 추가
                    "config": {
                        "subject_id": subject_id,
                        "group_idx": group_idx,
                        "cls_low": cls_low,
                        "cls_high": cls_high,
                        "img_size": args.img_size,
                        "base_channels": args.base_channels,
                        "num_timesteps": args.num_timesteps,
                        "n_res_blocks": args.n_res_blocks,
                        "cond_drop_prob": args.cond_drop_prob,
                    },
                },
                best_ckpt_path,
            )
            print(
                f"[Subj {subj_str}][Group {cls_low}-{cls_high}] "
                f"Updated best checkpoint → {best_ckpt_path}"
            )

    # -----------------------------------------------------
    # 6) final checkpoint + loss curve 저장
    # -----------------------------------------------------
    final_ckpt_path = os.path.join(
        save_dir,
        f"subj{subj_str}_g{group_idx+1}_final.pt"
    )
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema_model.state_dict(),  # ← 추가
            "config": {
                "subject_id": subject_id,
                "group_idx": group_idx,
                "cls_low": cls_low,
                "cls_high": cls_high,
                "img_size": args.img_size,
                "base_channels": args.base_channels,
                "num_timesteps": args.num_timesteps,
                "n_res_blocks": args.n_res_blocks,
                        "cond_drop_prob": args.cond_drop_prob,
            },
        },
        final_ckpt_path,
    )
    print(
        f"[Subj {subj_str}][Group {cls_low}-{cls_high}] "
        f"Training finished. Final ckpt → {final_ckpt_path}"
    )

    # loss curve
    train_loss_path = os.path.join(save_dir, "train_loss.npy")
    val_loss_path = os.path.join(save_dir, "val_loss.npy")
    np.save(train_loss_path, np.array(train_losses, dtype=np.float32))
    np.save(val_loss_path, np.array(val_losses, dtype=np.float32))
    print(
        f"[Subj {subj_str}][Group {cls_low}-{cls_high}] "
        f"Saved loss curves → {train_loss_path}, {val_loss_path}"
    )

    # epoch 단위 val loss plot
    plt.figure()
    plt.plot(range(len(val_losses)), val_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.title(f"Subj {subj_str} Group {cls_low}-{cls_high} Val Loss")
    plt.grid(True)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "val_loss_epoch.png")
    plt.savefig(fig_path)
    plt.close()
    print(
        f"[Subj {subj_str}][Group {cls_low}-{cls_high}] "
        f"Saved val loss plot → {fig_path}"
    )


# ---------------------------------------------------------
# main: 여러 subject, 3개 그룹(1~3, 4~6, 7~9)을 순차 학습
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="EEG→Image diffusion, 128x128, per-subject, 3 class-groups (1-3,4-6,7-9)"
    )
    parser.add_argument("--data_root", type=str, default="./preproc_data")
    parser.add_argument("--subject_ids", type=str, default="1",
                        help="예: '1' 또는 '1,2,3'")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--base_channels", type=int, default=64,
                        help="체크포인트 구조와 맞추는 채널 수 (기존엔 64 사용)")
    parser.add_argument("--num_timesteps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_res_blocks", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_root", type=str, default="./checkpoints_subj128_group_cfg_ddim")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--lambda_rec", type=float, default=0.02)
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)

    # parser 부분
    parser.add_argument("--ch_mult", type=str, default="1,2,4,4",
                        help="콤마로 구분된 multiplier 리스트. 예: '1,2,4,8'")

    args = parser.parse_args()

    set_seed(args.seed)

    subject_ids = [int(s.strip()) for s in args.subject_ids.split(",") if s.strip()]

    # 3개의 class group 정의
    groups = [
        (1, 3),   # Group 1
        (4, 6),   # Group 2
        (7, 9),   # Group 3
    ]

    for sid in subject_ids:
        for g_idx, (lo, hi) in enumerate(groups):
            train_one_subject_group(args, sid, g_idx, lo, hi)


if __name__ == "__main__":
    main()
