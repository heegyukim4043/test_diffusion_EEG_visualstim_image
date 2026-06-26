# train_subject_segments.py
import argparse
import os
from datetime import datetime
import csv

import torch
from torch.utils.data import DataLoader

from dataset_subject_segments import EEGImageSubjectSegmentDataset
from model import EEGDiffusionModel


def train_one_subject_segments(args, subject_id: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"[SEG] Training subject {subject_id:02d} on device: {device}")
    print("=" * 80)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # ---------- Dataset & Dataloader ----------
    train_ds = EEGImageSubjectSegmentDataset(
        data_root=args.data_root,
        subject_id=subject_id,
        split="train",
        split_ratio=args.split_ratio,
        img_size=args.img_size,
        seed=args.seed,
        num_segments=args.num_segments,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # EEG 채널 수 자동 추론
    sample_eeg, sample_img, _ = train_ds[0]
    eeg_channels = sample_eeg.shape[0]
    segment_len = sample_eeg.shape[1]
    print(
        f"[Subj {subject_id:02d}] EEG channels: {eeg_channels}, "
        f"segment_len: {segment_len}, img size: {sample_img.shape[-2:]}"
    )

    # ---------- Model ----------
    model = EEGDiffusionModel(
        img_size=args.img_size,
        img_channels=3,
        eeg_channels=eeg_channels,
        eeg_hidden_dim=args.eeg_hidden_dim,
        time_dim=args.time_dim,
        base_channels=args.base_channels,
        num_timesteps=args.num_timesteps,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ---------- run 폴더 (날짜/시간 + subject + seg) ----------
    os.makedirs(args.out_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_subj{subject_id:02d}_seg"
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Subj {subject_id:02d}] Checkpoints & logs will be saved under: {run_dir}")

    global_step = 0
    history_steps = []
    history_losses = []
    history_epochs = []

    # ---------- Training Loop ----------
    model.train()
    for epoch in range(args.epochs):
        for eeg_seg, img, label in train_loader:
            eeg_seg = eeg_seg.to(device)  # (B, 32, segment_len)
            img = img.to(device)          # (B, 3, H, W)
            img = img * 2.0 - 1.0         # [0,1] -> [-1,1]

            b = img.size(0)
            t = torch.randint(
                0, args.num_timesteps, (b,), device=device
            ).long()

            loss = model.p_losses(img, eeg_seg, t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            history_steps.append(global_step)
            history_losses.append(float(loss.item()))
            history_epochs.append(epoch)

            if global_step % args.log_interval == 0:
                print(
                    f"[Subj {subject_id:02d}] Epoch {epoch} "
                    f"Step {global_step} Loss {loss.item():.4f}"
                )

            if global_step > 0 and global_step % args.ckpt_interval == 0:
                ckpt_path = os.path.join(
                    run_dir,
                    f"subj{subject_id:02d}_step{global_step}.pt"
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                    },
                    ckpt_path,
                )
                print(f"[Subj {subject_id:02d}] Saved checkpoint to {ckpt_path}")

            global_step += 1

    # ---------- 최종 체크포인트 ----------
    final_ckpt = os.path.join(
        run_dir,
        f"subj{subject_id:02d}_final.pt"
    )
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

    # ---------- loss CSV 저장 ----------
    csv_path = os.path.join(run_dir, "train_loss.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "loss"])
        for s, e, l in zip(history_steps, history_epochs, history_losses):
            writer.writerow([s, e, l])
    print(f"[Subj {subject_id:02d}] Saved loss history to {csv_path}")

    # ---------- loss plot 저장 ----------
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(history_steps, history_losses)
        ax.set_xlabel("Step")
        ax.set_ylabel("Train Loss")
        ax.set_title(f"Subject {subject_id:02d} Train Loss (SEG)")
        fig.tight_layout()

        png_path = os.path.join(run_dir, "train_loss.png")
        fig.savefig(png_path)
        plt.close(fig)

        print(f"[Subj {subject_id:02d}] Saved loss plot to {png_path}")
    except Exception as e:
        print(f"[Subj {subject_id:02d}] Could not save loss plot:", e)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./preproc_data")

    # 단일 subject
    p.add_argument("--subject_id", type=int, default=1)

    # 여러 subject: 예) --subject_ids 1,2,3
    p.add_argument(
        "--subject_ids",
        type=str,
        default="",
        help="Comma-separated subject IDs, e.g., '1,2,3'. "
             "If set, overrides --subject_id and trains all listed subjects sequentially.",
    )

    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_timesteps", type=int, default=200)
    p.add_argument("--eeg_hidden_dim", type=int, default=256)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--split_ratio", type=float, default=0.7)
    p.add_argument("--out_dir", type=str, default="./checkpoints_subj_seg")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--ckpt_interval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--num_segments", type=int, default=4)
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
        print(f"[SEG] Training multiple subjects: {subj_list}")
        for sid in subj_list:
            train_one_subject_segments(args, sid)
    else:
        train_one_subject_segments(args, args.subject_id)
