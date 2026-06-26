import os
import argparse
import datetime
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from PIL import Image
import matplotlib.pyplot as plt

from model_128 import EEGDiffusionModel128


# ---------------------------------------------------------
# 1. Dataset: 9-class (label 1~9 → image 01.png~09.png)
# ---------------------------------------------------------
class EEGImageDataset9Cls(Dataset):
    """
    subj_XX.mat 안의 EEG (X, y) + preproc_data/images/NN.png 를 묶어서 반환.
    - X: (C, T, N_trials)
    - y: (N_trials,) or (1, N_trials)
    """

    def __init__(self, mat_path: str, img_dir: str, img_size: int = 128):
        super().__init__()
        self.mat_path = mat_path
        self.img_dir = img_dir
        self.img_size = img_size

        mat = loadmat(mat_path)
        X = mat["X"]  # (C, T, N)
        y = mat["y"].squeeze()  # (N,)

        # numpy → torch
        # X: (C, T, N) → 나중에 __getitem__에서 [ :, :, idx ] 로 슬라이스
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

        self.n_trials = self.X.shape[2]

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        # EEG: (C, T)
        eeg = self.X[:, :, idx]  # (C, T)
        label = int(self.y[idx].item())  # 1~9

        # 이미지 파일명: 01.png ~ 09.png
        img_name = f"{label:02d}.png"
        img_path = os.path.join(self.img_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
        img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

        return {
            "eeg": eeg,  # (C, T)
            "img": img,  # (3, H, W)
            "label": label,  # int, 1~9
        }


# ---------------------------------------------------------
# 2. 한 명의 subject에 대해 9-class 학습
# ---------------------------------------------------------
def train_one_subject_9cls(args, subject_id: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"[128-9cls] Training subject {subject_id:02d} (classes 1~9) on device: {device}")
    print("=" * 80)

    # ---------- 데이터셋 로딩 ----------
    mat_path = os.path.join(args.data_root, f"subj_{subject_id:02d}.mat")
    img_dir = os.path.join(args.data_root, "images")

    dataset = EEGImageDataset9Cls(mat_path, img_dir, img_size=args.img_size)
    n_total = len(dataset)

    # 8:1:1 split (train:val:test)
    indices = np.arange(n_total)
    rng = np.random.RandomState(args.seed + subject_id)
    rng.shuffle(indices)

    n_test = int(n_total * args.test_ratio)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)

    print(f"[Subj {subject_id:02d}] total={n_total}, train={len(train_ds)}, "
          f"val={len(val_ds)}, test={len(test_ds)}")

    # ---------- DataLoader ----------
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
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # EEG 채널 수 확인
    sample = dataset[0]
    eeg_channels = sample["eeg"].shape[0]
    print(f"[Subj {subject_id:02d}] EEG channels: {eeg_channels}, img size: {sample['img'].shape}")

    # ---------- 체크포인트 디렉터리 ----------
    os.makedirs(args.ckpt_root, exist_ok=True)
    time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{time_tag}_subj{subject_id:02d}_9cls_{args.img_size}"
    out_dir = os.path.join(args.ckpt_root, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Subj {subject_id:02d}] Checkpoints & logs → {out_dir}")

    # ---------- 모델 준비 ----------
    # ch_mult 파싱 (예: "1,2,2,4")
    ch_mult = tuple(int(x) for x in args.ch_mult.split(","))
    num_classes = 9  # 전체 클래스

    model = EEGDiffusionModel128(
        img_size=args.img_size,
        img_channels=3,
        eeg_channels=eeg_channels,
        num_classes=num_classes,
        num_timesteps=args.num_timesteps,
        base_channels=args.base_channels,
        ch_mult=ch_mult,
        n_res_blocks=args.n_res_blocks,
    ).to(device)

    # lambda_rec, cond_scale 를 __init__ 밖에서라도 강제 세팅 (안전)
    model.lambda_rec = getattr(args, "lambda_rec", 0.02)
    model.cond_scale = getattr(args, "cond_scale", 1.5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ---------- 학습 루프 설정 ----------
    train_losses = []
    val_losses = []

    total_steps = args.epochs * len(train_loader)
    if total_steps == 0:
        print(f"[Subj {subject_id:02d}] No training steps (empty train_loader).")
        return

    ckpt_steps = sorted(
        set(
            int(total_steps * r)
            for r in [0.2, 0.4, 0.6, 0.8]
        )
    )

    global_step = 0
    best_val = float("inf")
    best_ckpt_path = os.path.join(out_dir, f"subj{subject_id:02d}_best.pt")

    # ---------- Epoch Loop ----------
    for epoch in range(args.epochs):
        model.train()
        running = 0.0

        for step, batch in enumerate(train_loader):
            img = batch["img"].to(device)  # (B,3,H,W)
            eeg = batch["eeg"].to(device)  # (B,C,T)
            labels = batch["label"].to(device).long()  # (B,)

            # 0 ~ T-1 중 랜덤 timestep
            t = torch.randint(
                0, model.num_timesteps,
                (img.size(0),),
                device=device,
                dtype=torch.long
            )

            loss = model.p_losses(img, eeg, labels, t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1
            running += loss.item()

            if step % args.log_interval == 0:
                print(f"[Subj {subject_id:02d}] Epoch {epoch} Step {step}/{len(train_loader)} "
                      f"Loss {loss.item():.4f}")

            # 중간 체크포인트 저장
            if global_step in ckpt_steps:
                ckpt_path = os.path.join(out_dir, f"subj{subject_id:02d}_step{global_step}.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"[Subj {subject_id:02d}] Saved checkpoint step "
                      f"{global_step}/{total_steps} → {ckpt_path}")

        train_loss_epoch = running / len(train_loader)
        train_losses.append(train_loss_epoch)

        # ---------- Validation ----------
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["img"].to(device)
                eeg = batch["eeg"].to(device)
                labels = batch["label"].to(device).long()

                t = torch.randint(
                    0, model.num_timesteps,
                    (img.size(0),),
                    device=device,
                    dtype=torch.long
                )
                loss = model.p_losses(img, eeg, labels, t)
                val_running += loss.item()

        val_loss_epoch = val_running / len(val_loader)
        val_losses.append(val_loss_epoch)

        print(f"[Subj {subject_id:02d}] Epoch {epoch} TrainLoss {train_loss_epoch:.4f} "
              f"ValLoss {val_loss_epoch:.4f}")

        # best 모델 갱신
        if val_loss_epoch < best_val:
            best_val = val_loss_epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(f"[Subj {subject_id:02d}] Updated best checkpoint → {best_ckpt_path}")

    # ---------- Final checkpoint ----------
    final_ckpt_path = os.path.join(out_dir, f"subj{subject_id:02d}_final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": args.epochs,
            "global_step": global_step,
            "args": vars(args),
        },
        final_ckpt_path,
    )
    print(f"[Subj {subject_id:02d}] Training finished. Final ckpt → {final_ckpt_path}")

    # ---------- Loss history 저장 ----------
    train_loss_path = os.path.join(out_dir, "train_loss.npy")
    val_loss_path = os.path.join(out_dir, "val_loss.npy")
    np.save(train_loss_path, np.array(train_losses))
    np.save(val_loss_path, np.array(val_losses))
    print(f"[Subj {subject_id:02d}] Saved loss curves → {train_loss_path}, {val_loss_path}")

    # 간단한 loss plot
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    val_plot_path = os.path.join(out_dir, "val_loss_epoch.png")
    plt.savefig(val_plot_path)
    plt.close()
    print(f"[Subj {subject_id:02d}] Saved loss plot → {val_plot_path}")


# ---------------------------------------------------------
# 3. Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EEG→Image Diffusion (128x128, 9-class)")

    parser.add_argument("--data_root", type=str, default="./preproc_data",
                        help="Root dir containing subj_XX.mat and images/")
    parser.add_argument("--subject_ids", type=str, default="1",
                        help="Comma-separated subject IDs, e.g. '1,2,3'")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=128)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_timesteps", type=int, default=400)

    parser.add_argument("--base_channels", type=int, default=96,
                        help="Base channel count of UNet (e.g. 64, 96, 128)")
    parser.add_argument("--ch_mult", type=str, default="1,2,2,4",
                        help="Comma separated multipliers per UNet block, e.g. '1,2,2,4'")

    parser.add_argument("--lambda_rec", type=float, default=0.02,
                        help="weight of x0 reconstruction term in p_losses")
    parser.add_argument("--cond_scale", type=float, default=1.5,
                        help="scale for conditional embedding inside model (guidance-like)")

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--ckpt_root", type=str, default="./checkpoints_subj128_9cls",
                        help="Base directory for saving checkpoints/logs")

    parser.add_argument("--n_res_blocks", type=int, default=2,
                        help="number of ResNet blocks per UNet level")

    args = parser.parse_args()

    # 여러 subject 순회
    subject_ids = [int(s.strip()) for s in args.subject_ids.split(",") if s.strip()]

    for sid in subject_ids:
        train_one_subject_9cls(args, sid)


if __name__ == "__main__":
    main()

