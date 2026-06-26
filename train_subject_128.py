# train_subject_128.py
import argparse
import csv
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split

from dataset_subject import EEGImageSubjectDataset
from model_128 import EEGDiffusionModel128


def train_one_subject_128(args, subject_id: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"[128] Training subject {subject_id:02d} on device: {device}")
    print("=" * 80)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # --------------------------------------------------
    # 1) Dataset: 8:1:1 (train:val:test ~= 0.8:0.1:0.1)
    # --------------------------------------------------
    base_trainval_ds = EEGImageSubjectDataset(
        data_root=args.data_root,
        subject_id=subject_id,
        split="train",
        split_ratio=0.9,      # 90% (train+val)
        img_size=args.img_size,
        seed=args.seed,
    )
    base_test_ds = EEGImageSubjectDataset(
        data_root=args.data_root,
        subject_id=subject_id,
        split="test",
        split_ratio=0.9,      # 10% test
        img_size=args.img_size,
        seed=args.seed,
    )

    print(
        f"[Subj {subject_id:02d}] total trials ≈ "
        f"{len(base_trainval_ds) + len(base_test_ds)} "
        f"(train+val={len(base_trainval_ds)}, test={len(base_test_ds)})"
    )

    n_trainval = len(base_trainval_ds)
    n_train = int(n_trainval * (8.0 / 9.0))  # 전체 기준 0.8
    n_val = n_trainval - n_train             # 전체 기준 0.1

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_ds, val_ds = random_split(
        base_trainval_ds, [n_train, n_val], generator=g
    )

    print(
        f"[Subj {subject_id:02d}] final splits → "
        f"train={len(train_ds)}, val={len(val_ds)}, test={len(base_test_ds)}"
    )

    # --------------------------------------------------
    # 2) DataLoader
    # --------------------------------------------------
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

    # 샘플로 EEG 채널 수 확인
    sample_eeg, sample_img, sample_label = train_ds[0]
    eeg_channels = sample_eeg.shape[0]
    print(
        f"[Subj {subject_id:02d}] EEG channels: {eeg_channels}, "
        f"img size: {sample_img.shape[-2:]}"
    )

    # --------------------------------------------------
    # 3) Model & Optimizer
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 4) Run directory & checkpoints
    # --------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_subj{subject_id:02d}_128"
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"[Subj {subject_id:02d}] Checkpoints & logs will be saved under: {run_dir}")

    num_batches = len(train_loader)
    total_steps = args.epochs * num_batches
    if total_steps == 0:
        raise RuntimeError("No training steps (empty dataset).")

    ckpt_ratios = [0.2, 0.4, 0.6, 0.8]
    ckpt_steps = set(max(1, int(total_steps * r)) for r in ckpt_ratios)
    print(f"[Subj {subject_id:02d}] total_steps={total_steps}, ckpt_steps={sorted(ckpt_steps)}")

    global_step = 0

    # step-wise 기록
    history_steps = []
    history_losses = []
    history_epochs = []

    # epoch-wise 기록
    train_epoch_ids = []
    train_epoch_losses = []
    val_epoch_ids = []
    val_epoch_losses = []

    # --------------------------------------------------
    # 5) Training loop
    # --------------------------------------------------
    model.train()
    for epoch in range(args.epochs):
        epoch_loss_sum = 0.0
        epoch_count = 0

        for eeg, img, label in train_loader:
            eeg = eeg.to(device)       # (B, C, T)
            img = img.to(device)       # (B, 3, H, W) in [0,1]
            label = label.to(device)   # (B,)
            img = img * 2.0 - 1.0      # [-1,1]

            b = img.size(0)
            t = torch.randint(
                0, args.num_timesteps, (b,), device=device, dtype=torch.long
            )

            loss = model.p_losses(img, eeg, label, t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step_id = global_step + 1

            history_steps.append(step_id)
            history_losses.append(float(loss.item()))
            history_epochs.append(epoch)

            epoch_loss_sum += float(loss.item()) * b
            epoch_count += b

            if step_id % args.log_interval == 0:
                print(
                    f"[Subj {subject_id:02d}] "
                    f"Epoch {epoch} Step {step_id}/{total_steps} "
                    f"Train Loss {loss.item():.4f}"
                )

            if step_id in ckpt_steps:
                ckpt_path = os.path.join(
                    run_dir,
                    f"subj{subject_id:02d}_step{step_id}.pt",
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": step_id,
                    },
                    ckpt_path,
                )
                print(
                    f"[Subj {subject_id:02d}] Saved checkpoint "
                    f"(progress {step_id}/{total_steps}) to {ckpt_path}"
                )

            global_step += 1

        # ---- epoch 평균 train loss ----
        avg_train_loss = epoch_loss_sum / max(1, epoch_count)
        train_epoch_ids.append(epoch)
        train_epoch_losses.append(avg_train_loss)
        print(
            f"[Subj {subject_id:02d}] Epoch {epoch} Train Avg Loss {avg_train_loss:.4f}"
        )

        # ---- validation ----
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for eeg, img, label in val_loader:
                eeg = eeg.to(device)
                img = img.to(device)
                label = label.to(device)
                img = img * 2.0 - 1.0

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
        print(
            f"[Subj {subject_id:02d}] Epoch {epoch} Validation Loss {avg_val_loss:.4f}"
        )

        model.train()

    # --------------------------------------------------
    # 6) Final checkpoint
    # --------------------------------------------------
    final_ckpt = os.path.join(run_dir, f"subj{subject_id:02d}_final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": global_step,
        },
        final_ckpt,
    )
    print(f"[Subj {subject_id:02d}] Training finished. Final checkpoint: {final_ckpt}")

    # --------------------------------------------------
    # 7) CSV로 loss 기록 저장
    # --------------------------------------------------
    # step-wise
    csv_steps = os.path.join(run_dir, "train_loss_steps.csv")
    with open(csv_steps, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "loss"])
        for s, e, l in zip(history_steps, history_epochs, history_losses):
            w.writerow([s, e, l])
    print(f"[Subj {subject_id:02d}] Saved step-wise train loss to {csv_steps}")

    # epoch-wise train
    csv_train_epoch = os.path.join(run_dir, "train_loss_epoch.csv")
    with open(csv_train_epoch, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss"])
        for e, l in zip(train_epoch_ids, train_epoch_losses):
            w.writerow([e, l])
    print(f"[Subj {subject_id:02d}] Saved epoch-avg train loss to {csv_train_epoch}")

    # epoch-wise val
    csv_val = os.path.join(run_dir, "val_loss.csv")
    with open(csv_val, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "val_loss"])
        for e, l in zip(val_epoch_ids, val_epoch_losses):
            w.writerow([e, l])
    print(f"[Subj {subject_id:02d}] Saved val loss history to {csv_val}")

    # --------------------------------------------------
    # 8) Loss plot (epoch 기준)
    # --------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        # train
        fig, ax = plt.subplots()
        ax.plot(train_epoch_ids, train_epoch_losses, marker="o", label="train")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss")
        ax.set_title(f"Subject {subject_id:02d} Train Loss (128x128)")
        ax.legend()
        fig.tight_layout()
        png_train = os.path.join(run_dir, "train_loss.png")
        fig.savefig(png_train)
        plt.close(fig)
        print(f"[Subj {subject_id:02d}] Saved train loss plot to {png_train}")

        # val
        fig, ax = plt.subplots()
        ax.plot(val_epoch_ids, val_epoch_losses, marker="o", label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Loss")
        ax.set_title(f"Subject {subject_id:02d} Validation Loss (128x128)")
        ax.legend()
        fig.tight_layout()
        png_val = os.path.join(run_dir, "val_loss.png")
        fig.savefig(png_val)
        plt.close(fig)
        print(f"[Subj {subject_id:02d}] Saved val loss plot to {png_val}")

    except Exception as e:
        print(f"[Subj {subject_id:02d}] Could not save loss plots: {e}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./preproc_data")

    # 하나 또는 여러 subject
    p.add_argument("--subject_id", type=int, default=1)
    p.add_argument(
        "--subject_ids",
        type=str,
        default="",
        help="예: '1,2,3' 형태로 여러 subject를 순차 학습",
    )

    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_timesteps", type=int, default=200)

    p.add_argument("--eeg_hidden_dim", type=int, default=256)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--cond_scale", type=float, default=4.0)
    p.add_argument("--num_classes", type=int, default=9)

    p.add_argument("--out_dir", type=str, default="./checkpoints_subj128")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=2025)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.subject_ids.strip():
        subj_list = []
        for tok in args.subject_ids.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                subj_list.append(int(tok))
            except ValueError:
                print(f"Warning: cannot parse subject id '{tok}', skipping.")
        subj_list = sorted(set(subj_list))
        print(f"Training multiple subjects (128x128): {subj_list}")
        for sid in subj_list:
            train_one_subject_128(args, sid)
    else:
        train_one_subject_128(args, args.subject_id)
