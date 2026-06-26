# train_multi_subject_128.py
import argparse
import csv
import os
from datetime import datetime
from typing import List

import torch
from torch.utils.data import DataLoader, random_split

from multi_subject_dataset import MultiSubjectEEGImageDataset
from model_128 import EEGDiffusionModel128


def parse_subject_ids(s: str) -> List[int]:
    """문자열 '1-10' 또는 '1,2,3' 등을 [1,2,3,...] 리스트로 변환."""
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            a, b = int(a), int(b)
            out.extend(range(a, b + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


def train_multi_subject_128(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"[128] Multi-subject training on device: {device}")
    print("=" * 80)

    subject_ids = parse_subject_ids(args.train_subject_ids)
    print(f"[MultiSubj] Train subjects: {subject_ids}")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # ------------------------------------------------------------------
    # 1) Multi-subject dataset (split='train' → 각 subject 90% 데이터)
    #    그 안에서 다시 8:1 로 train / val 분할
    # ------------------------------------------------------------------
    base_trainval_ds = MultiSubjectEEGImageDataset(
        data_root=args.data_root,
        subject_ids=subject_ids,
        split="train",
        split_ratio=args.split_ratio,   # 예: 0.9
        img_size=args.img_size,
        seed=args.seed,
    )

    n_trainval = len(base_trainval_ds)
    if n_trainval == 0:
        raise RuntimeError("MultiSubject train+val 데이터가 0개입니다. 경로/subject_id 확인하세요.")

    n_train = int(n_trainval * (8.0 / 9.0))   # 전체 0.8
    n_val = n_trainval - n_train              # 전체 0.1
    print(f"[MultiSubj] total train+val={n_trainval}, train={n_train}, val={n_val}")

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_ds, val_ds = random_split(base_trainval_ds, [n_train, n_val], generator=g)

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

    # 샘플 하나 꺼내서 차원 확인
    eeg0, img0, label0 = train_ds[0]
    eeg_channels = eeg0.shape[0]
    print(f"[MultiSubj] EEG channels: {eeg_channels}, img size: {img0.shape[-2:]}")

    # ------------------------------------------------------------------
    # 2) 모델 구성
    # ------------------------------------------------------------------
    model = EEGDiffusionModel128(
        img_size=args.img_size,
        img_channels=3,
        eeg_channels=eeg_channels,
        num_classes=args.num_classes,
        num_timesteps=args.num_timesteps,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        cond_dim=args.cond_dim,
        eeg_hidden_dim=args.eeg_hidden_dim,
        cond_scale=args.cond_scale,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # 3) 로그/체크포인트 디렉토리
    # ------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    sid_str = "_".join(f"{s:02d}" for s in subject_ids)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_multi_s{sid_str}_128"
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[MultiSubj] Checkpoints & logs will be saved under: {run_dir}")

    num_batches = len(train_loader)
    total_steps = args.epochs * num_batches
    if total_steps == 0:
        raise RuntimeError("No training steps (batch 수가 0). batch_size/데이터 확인 필요.")

    ckpt_ratios = [0.2, 0.4, 0.6, 0.8]
    ckpt_steps = set(max(1, int(total_steps * r)) for r in ckpt_ratios)
    print(f"[MultiSubj] total_steps={total_steps}, ckpt_steps={sorted(ckpt_steps)}")

    # step-wise 기록
    history_steps = []
    history_losses = []
    history_epochs = []

    # epoch-wise 기록
    train_epoch_ids = []
    train_epoch_losses = []
    val_epoch_ids = []
    val_epoch_losses = []

    global_step = 0
    model.train()

    # ------------------------------------------------------------------
    # 4) 학습 루프
    # ------------------------------------------------------------------
    for epoch in range(args.epochs):
        epoch_loss_sum = 0.0
        epoch_count = 0

        for eeg, img, label in train_loader:
            eeg = eeg.to(device)                      # (B, C, T)
            img = img.to(device) * 2.0 - 1.0          # (B, 3, H, W) in [-1,1]
            label = label.to(device).long()           # (B,)

            b = img.size(0)
            t = torch.randint(
                0, args.num_timesteps, (b,), device=device, dtype=torch.long
            )

            loss = model.p_losses(img, eeg, label, t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            history_steps.append(global_step)
            history_losses.append(float(loss.item()))
            history_epochs.append(epoch)

            epoch_loss_sum += float(loss.item()) * b
            epoch_count += b

            if global_step % args.log_interval == 0:
                print(
                    f"[MultiSubj] Epoch {epoch} "
                    f"Step {global_step}/{total_steps} "
                    f"Train Loss {loss.item():.4f}"
                )

            if global_step in ckpt_steps:
                ckpt_path = os.path.join(
                    run_dir,
                    f"multi_s{sid_str}_step{global_step}.pt",
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                        "train_subject_ids": subject_ids,
                    },
                    ckpt_path,
                )
                print(
                    f"[MultiSubj] Saved checkpoint "
                    f"(progress {global_step}/{total_steps}) to {ckpt_path}"
                )

        # ---- epoch 평균 train loss ----
        avg_train_loss = epoch_loss_sum / max(1, epoch_count)
        train_epoch_ids.append(epoch)
        train_epoch_losses.append(avg_train_loss)
        print(f"[MultiSubj] Epoch {epoch} Train Avg Loss {avg_train_loss:.4f}")

        # ---- validation ----
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for eeg, img, label in val_loader:
                eeg = eeg.to(device)
                img = img.to(device) * 2.0 - 1.0
                label = label.to(device).long()

                b = img.size(0)
                t = torch.randint(
                    0, args.num_timesteps, (b,), device=device, dtype=torch.long
                )
                vloss = model.p_losses(img, eeg, label, t)
                val_loss_sum += float(vloss.item()) * b
                val_count += b

        avg_val_loss = val_loss_sum / max(1, val_count)
        val_epoch_ids.append(epoch)
        val_epoch_losses.append(avg_val_loss)
        print(f"[MultiSubj] Epoch {epoch} Validation Loss {avg_val_loss:.4f}")

        model.train()

    # ------------------------------------------------------------------
    # 5) 최종 체크포인트 저장
    # ------------------------------------------------------------------
    final_ckpt = os.path.join(run_dir, f"multi_s{sid_str}_final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": args.epochs - 1,
            "step": global_step,
            "train_subject_ids": subject_ids,
        },
        final_ckpt,
    )
    print(f"[MultiSubj] Training finished. Final checkpoint: {final_ckpt}")

    # ------------------------------------------------------------------
    # 6) loss CSV 및 plot 저장
    # ------------------------------------------------------------------
    csv_steps = os.path.join(run_dir, "train_loss_steps.csv")
    with open(csv_steps, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "loss"])
        for s, e, l in zip(history_steps, history_epochs, history_losses):
            w.writerow([s, e, l])
    print(f"[MultiSubj] Saved step-wise train loss to {csv_steps}")

    csv_train_epoch = os.path.join(run_dir, "train_loss_epoch.csv")
    with open(csv_train_epoch, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss"])
        for e, l in zip(train_epoch_ids, train_epoch_losses):
            w.writerow([e, l])
    print(f"[MultiSubj] Saved epoch-avg train loss to {csv_train_epoch}")

    csv_val = os.path.join(run_dir, "val_loss.csv")
    with open(csv_val, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "val_loss"])
        for e, l in zip(val_epoch_ids, val_epoch_losses):
            w.writerow([e, l])
    print(f"[MultiSubj] Saved val loss history to {csv_val}")

    # 간단한 plot (에러 나면 무시)
    try:
        import matplotlib.pyplot as plt

        # train
        fig, ax = plt.subplots()
        ax.plot(train_epoch_ids, train_epoch_losses, marker="o", label="train")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss")
        ax.set_title("Multi-subject Train Loss (128x128)")
        ax.legend()
        fig.tight_layout()
        png_train = os.path.join(run_dir, "train_loss.png")
        fig.savefig(png_train)
        plt.close(fig)
        print(f"[MultiSubj] Saved train loss plot to {png_train}")

        # val
        fig, ax = plt.subplots()
        ax.plot(val_epoch_ids, val_epoch_losses, marker="o", label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Loss")
        ax.set_title("Multi-subject Validation Loss (128x128)")
        ax.legend()
        fig.tight_layout()
        png_val = os.path.join(run_dir, "val_loss.png")
        fig.savefig(png_val)
        plt.close(fig)
        print(f"[MultiSubj] Saved val loss plot to {png_val}")
    except Exception as e:
        print(f"[MultiSubj] Could not save loss plots: {e}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./preproc_data")

    # 예: "1-10" 또는 "1,2,3,4,5"
    p.add_argument(
        "--train_subject_ids",
        type=str,
        default="1-10",
        help="예: '1-10' 또는 '1,2,3,4'",
    )

    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_timesteps", type=int, default=200)
    p.add_argument("--split_ratio", type=float, default=0.9)

    p.add_argument("--eeg_hidden_dim", type=int, default=256)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--cond_scale", type=float, default=2.0)
    p.add_argument("--num_classes", type=int, default=9)

    p.add_argument("--out_dir", type=str, default="./checkpoints_multi128")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=2025)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    train_multi_subject_128(args)
