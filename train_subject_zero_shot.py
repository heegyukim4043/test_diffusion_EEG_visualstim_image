# train_subject_zero_shot.py
import argparse
import os
from datetime import datetime
import csv

import torch
from torch.utils.data import DataLoader, random_split

from dataset_subject import EEGImageSubjectDataset
from model import EEGDiffusionModel


class FilteredLabelDataset(torch.utils.data.Dataset):
    """
    base_dataset에서 특정 라벨만 포함(include)하거나 제외(exclude)하는 래퍼.
    base_dataset.__getitem__ -> (eeg, img, label) 구조를 가정.
    """
    def __init__(self, base_dataset, include_labels=None, exclude_labels=None):
        super().__init__()
        self.base = base_dataset
        self.indices = []

        for idx in range(len(base_dataset)):
            eeg, img, label = base_dataset[idx]
            lab = int(label)

            if include_labels is not None:
                if lab in include_labels:
                    self.indices.append(idx)
            elif exclude_labels is not None:
                if lab not in exclude_labels:
                    self.indices.append(idx)
            else:
                self.indices.append(idx)

        print(
            f"[FilteredLabelDataset] kept {len(self.indices)} / {len(base_dataset)} "
            f"samples (include={include_labels}, exclude={exclude_labels})"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


def train_one_subject_zero_shot(args, subject_id: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"[ZS] Training subject {subject_id:02d} on device: {device}")
    print("=" * 80)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    held_out = args.held_out_class
    print(f"[ZS] Held-out (pseudo-zero-shot) class = {held_out}")

    # ------------------------------------------------------------------
    # 1) 전체 trial을 8:1:1 (train:val:test) 로 만들기
    #    단, train/val에서는 held_out_class를 완전히 제외
    # ------------------------------------------------------------------

    # (train+val) 90% (모든 라벨 포함 상태)
    base_trainval_full = EEGImageSubjectDataset(
        data_root=args.data_root,
        subject_id=subject_id,
        split="train",
        split_ratio=0.9,
        img_size=args.img_size,
        seed=args.seed,
    )
    # test 10% (모든 라벨 포함 상태; 여기서는 정보 출력만)
    base_test_full = EEGImageSubjectDataset(
        data_root=args.data_root,
        subject_id=subject_id,
        split="test",
        split_ratio=0.9,
        img_size=args.img_size,
        seed=args.seed,
    )

    print(
        f"[Subj {subject_id:02d}] total trials ≈ "
        f"{len(base_trainval_full) + len(base_test_full)} "
        f"(train+val(full)={len(base_trainval_full)}, test(full)={len(base_test_full)})"
    )

    # train+val 풀셋에서 held_out_class 제외
    base_trainval_ds = FilteredLabelDataset(
        base_trainval_full,
        include_labels=None,
        exclude_labels=[held_out],
    )

    # (선택) test에서도 held_out을 제외한 분포가 궁금하면 아래처럼 볼 수 있지만
    # 학습에는 사용하지 않음. (held-out class는 test split에서만 사용 예정)
    _ = FilteredLabelDataset(
        base_test_full,
        include_labels=None,
        exclude_labels=[held_out],
    )

    # 이제 base_trainval_ds(=90% 중 non-held-out만)를 8:1 로 나누어
    # 전체 기준으로 대략 0.8(train) : 0.1(val) 비율이 되도록 함
    n_trainval = len(base_trainval_ds)
    n_train = int(n_trainval * (8.0 / 9.0))  # ≈ 전체 0.8
    n_val = n_trainval - n_train             # ≈ 전체 0.1

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_ds, val_ds = random_split(
        base_trainval_ds,
        [n_train, n_val],
        generator=g,
    )

    print(
        f"[Subj {subject_id:02d}] final (non-held-out) splits → "
        f"train={len(train_ds)}, val={len(val_ds)}, "
        f"test(full, incl. held-out)={len(base_test_full)}"
    )

    # ------------------------------------------------------------------
    # 2) DataLoader
    # ------------------------------------------------------------------
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

    # EEG 채널 수 자동 추론
    sample_eeg, sample_img, _ = train_ds[0]
    eeg_channels = sample_eeg.shape[0]
    print(
        f"[Subj {subject_id:02d}] EEG channels: {eeg_channels}, "
        f"img size: {sample_img.shape[-2:]}"
    )

    # ------------------------------------------------------------------
    # 3) Model & Optimizer
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 4) run 폴더 & checkpoint 스텝 (20% 단위)
    # ------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_subj{subject_id:02d}_noC{held_out}"
    )
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Subj {subject_id:02d}] Checkpoints & logs will be saved under: {run_dir}")

    num_batches = len(train_loader)
    total_steps = args.epochs * num_batches
    if total_steps == 0:
        raise RuntimeError("No training steps: dataset or loader is empty.")

    ckpt_ratios = [0.2, 0.4, 0.6, 0.8]
    ckpt_steps = set(max(1, int(total_steps * r)) for r in ckpt_ratios)
    print(f"[Subj {subject_id:02d}] total_steps={total_steps}, ckpt_steps={sorted(ckpt_steps)}")

    global_step = 0

    # step 단위 train loss 기록
    history_steps = []
    history_losses = []
    history_epochs = []

    # epoch 단위 train / val loss 기록
    train_epoch_ids = []
    train_epoch_losses = []
    val_history_epochs = []
    val_history_losses = []

    # ------------------------------------------------------------------
    # 5) Training Loop
    # ------------------------------------------------------------------
    model.train()
    for epoch in range(args.epochs):
        # epoch 평균 계산용
        epoch_loss_sum = 0.0
        epoch_count = 0

        for eeg, img, label in train_loader:
            eeg = eeg.to(device)                # (B, 32, T)
            img = img.to(device)                # (B, 3, H, W)
            img = img * 2.0 - 1.0               # [0,1] -> [-1,1]

            b = img.size(0)
            t = torch.randint(
                0, args.num_timesteps, (b,), device=device
            ).long()

            loss = model.p_losses(img, eeg, t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step_id = global_step + 1

            # step 단위 기록
            history_steps.append(step_id)
            history_losses.append(float(loss.item()))
            history_epochs.append(epoch)

            # epoch 평균용
            epoch_loss_sum += float(loss.item()) * b
            epoch_count += b

            if step_id % args.log_interval == 0:
                print(
                    f"[ZS Subj {subject_id:02d}] "
                    f"Epoch {epoch} Step {step_id}/{total_steps} "
                    f"Train Loss {loss.item():.4f}"
                )

            if step_id in ckpt_steps:
                ckpt_path = os.path.join(
                    run_dir,
                    f"subj{subject_id:02d}_step{step_id}.pt"
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": step_id,
                        "held_out_class": held_out,
                    },
                    ckpt_path,
                )
                print(
                    f"[ZS Subj {subject_id:02d}] Saved checkpoint "
                    f"(progress {step_id}/{total_steps}) to {ckpt_path}"
                )

            global_step += 1

        # ---- epoch 끝: train 평균 loss ----
        avg_train_loss = epoch_loss_sum / max(1, epoch_count)
        train_epoch_ids.append(epoch)
        train_epoch_losses.append(avg_train_loss)
        print(
            f"[ZS Subj {subject_id:02d}] Epoch {epoch} "
            f"Train Avg Loss {avg_train_loss:.4f}"
        )

        # ---- epoch 끝: validation loss ----
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for eeg, img, label in val_loader:
                eeg = eeg.to(device)
                img = img.to(device)
                img = img * 2.0 - 1.0

                b = img.size(0)
                t = torch.randint(
                    0, args.num_timesteps, (b,), device=device
                ).long()

                vloss = model.p_losses(img, eeg, t)
                val_loss_sum += float(vloss.item()) * b
                val_count += b

        avg_val_loss = val_loss_sum / max(1, val_count)
        val_history_epochs.append(epoch)
        val_history_losses.append(avg_val_loss)
        print(
            f"[ZS Subj {subject_id:02d}] Epoch {epoch} "
            f"Validation Loss {avg_val_loss:.4f}"
        )

        model.train()

    # ------------------------------------------------------------------
    # 6) 마지막 체크포인트
    # ------------------------------------------------------------------
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
            "held_out_class": held_out,
        },
        final_ckpt,
    )
    print(f"[ZS Subj {subject_id:02d}] Training finished. Final checkpoint: {final_ckpt}")

    # ------------------------------------------------------------------
    # 7) CSV 저장
    # ------------------------------------------------------------------
    # step 단위 train loss
    csv_path = os.path.join(run_dir, "train_loss_steps.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "loss"])
        for s, e, l in zip(history_steps, history_epochs, history_losses):
            writer.writerow([s, e, l])
    print(f"[ZS Subj {subject_id:02d}] Saved step-wise train loss to {csv_path}")

    # epoch 단위 val loss
    csv_val_path = os.path.join(run_dir, "val_loss.csv")
    with open(csv_val_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "val_loss"])
        for e, l in zip(val_history_epochs, val_history_losses):
            writer.writerow([e, l])
    print(f"[ZS Subj {subject_id:02d}] Saved val loss history to {csv_val_path}")

    # epoch 단위 train loss
    csv_train_epoch = os.path.join(run_dir, "train_loss_epoch.csv")
    with open(csv_train_epoch, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss"])
        for e, l in zip(train_epoch_ids, train_epoch_losses):
            writer.writerow([e, l])
    print(f"[ZS Subj {subject_id:02d}] Saved epoch-avg train loss to {csv_train_epoch}")

    # ------------------------------------------------------------------
    # 8) 그래프 저장 (train/val 모두 epoch 기준)
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        # (a) train loss vs epoch
        fig, ax = plt.subplots()
        ax.plot(train_epoch_ids, train_epoch_losses, marker="o", label="train")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss")
        ax.set_title(f"ZS Subject {subject_id:02d} Train Loss (no class {held_out})")
        ax.legend()
        fig.tight_layout()
        png_path = os.path.join(run_dir, "train_loss.png")
        fig.savefig(png_path)
        plt.close(fig)
        print(f"[ZS Subj {subject_id:02d}] Saved train loss plot to {png_path}")

        # (b) val loss vs epoch
        fig, ax = plt.subplots()
        ax.plot(val_history_epochs, val_history_losses, marker="o", label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Loss")
        ax.set_title(f"ZS Subject {subject_id:02d} Validation Loss (no class {held_out})")
        ax.legend()
        fig.tight_layout()
        png_val = os.path.join(run_dir, "val_loss.png")
        fig.savefig(png_val)
        plt.close(fig)
        print(f"[ZS Subj {subject_id:02d}] Saved val loss plot to {png_val}")

    except Exception as e:
        print(f"[ZS Subj {subject_id:02d}] Could not save loss plots:", e)


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
        help="Comma-separated subject IDs to train, e.g., '1,2,3'. "
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

    p.add_argument("--out_dir", type=str, default="./checkpoints_subj_zs")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=2025)

    # pseudo-zero-shot: 학습에서 제외할 라벨 (1~9)
    p.add_argument("--held_out_class", type=int, default=9)
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
        print(f"[ZS] Training multiple subjects: {subj_list}")
        for sid in subj_list:
            train_one_subject_zero_shot(args, sid)
    else:
        train_one_subject_zero_shot(args, args.subject_id)
