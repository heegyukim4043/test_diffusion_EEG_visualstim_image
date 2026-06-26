# train_loso_subject_128.py
import os
import argparse
import random
from datetime import datetime
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader

from multi_subject_dataset import MultiSubjectEEGImageDataset
from model_128 import EEGDiffusionModel128


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_subject_ids(data_root: str):
    mats = glob(os.path.join(data_root, "subj_*.mat"))
    ids = []
    for p in mats:
        name = os.path.basename(p)
        try:
            sid = int(name.replace("subj_", "").replace(".mat", ""))
            ids.append(sid)
        except ValueError:
            continue
    return sorted(list(set(ids)))


def ema_update(model, ema_model, decay: float = 0.999):
    msd = model.state_dict()
    for name, param in ema_model.state_dict().items():
        if name in msd:
            param.copy_(param * decay + msd[name] * (1.0 - decay))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.subject_ids:
        all_ids = [int(s.strip()) for s in args.subject_ids.split(",") if s.strip()]
    else:
        all_ids = discover_subject_ids(args.data_root)

    if args.heldout_subject_id not in all_ids:
        raise ValueError("heldout_subject_id not found in data_root")

    train_ids = [sid for sid in all_ids if sid != args.heldout_subject_id]

    print("=" * 80)
    print(f"[LOSO-128] Held-out subject: {args.heldout_subject_id:02d}")
    print(f"[LOSO-128] Train subjects: {train_ids}")
    print(f"[LOSO-128] device: {device}")
    print("=" * 80)

    set_seed(args.seed)

    train_ds = MultiSubjectEEGImageDataset(
        data_root=args.data_root,
        subject_ids=train_ids,
        split="train",
        split_ratio=args.split_ratio,
        img_size=args.img_size,
        seed=args.seed,
    )
    val_ds = MultiSubjectEEGImageDataset(
        data_root=args.data_root,
        subject_ids=train_ids,
        split="test",
        split_ratio=args.split_ratio,
        img_size=args.img_size,
        seed=args.seed,
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

    ch_mult = [int(x) for x in args.ch_mult.split(",")]
    model = EEGDiffusionModel128(
        img_size=args.img_size,
        img_channels=3,
        eeg_channels=32,
        num_classes=9,
        num_timesteps=args.num_timesteps,
        base_channels=args.base_channels,
        ch_mult=ch_mult,
        time_dim=256,
        cond_dim=256,
        eeg_hidden_dim=256,
        cond_scale=2.0,
        n_res_blocks=args.n_res_blocks,
        lambda_rec=args.lambda_rec,
    ).to(device)

    ema_model = None
    if args.use_ema:
        ema_model = type(model)(
            img_size=args.img_size,
            img_channels=3,
            eeg_channels=32,
            num_classes=9,
            num_timesteps=args.num_timesteps,
            base_channels=args.base_channels,
            ch_mult=ch_mult,
            time_dim=256,
            cond_dim=256,
            eeg_hidden_dim=256,
            cond_scale=2.0,
            n_res_blocks=args.n_res_blocks,
            lambda_rec=args.lambda_rec,
        ).to(device)
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.ckpt_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        args.ckpt_root,
        f"{timestamp}_loso_holdout{args.heldout_subject_id:02d}_{args.img_size}"
    )
    os.makedirs(save_dir, exist_ok=True)
    print(f"[LOSO-128] Checkpoints & logs -> {save_dir}")

    train_losses = []
    val_losses = []
    best_val = float("inf")
    best_ckpt_path = None

    total_steps = args.epochs * len(train_loader)
    if total_steps == 0:
        print("No training steps (empty loader).")
        return

    for epoch in range(args.epochs):
        model.train()
        epoch_train_losses = []

        for batch_idx, (eeg, img, labels) in enumerate(train_loader):
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

            loss = model.p_losses(img, eeg, labels, t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if ema_model is not None:
                ema_update(model, ema_model, decay=args.ema_decay)

            epoch_train_losses.append(loss.item())
            train_losses.append(loss.item())

            if batch_idx % args.log_interval == 0:
                print(
                    f"[LOSO-128] Epoch {epoch} Step {batch_idx}/{len(train_loader)} "
                    f"Loss {loss.item():.4f}"
                )

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
            f"[LOSO-128] Epoch {epoch} TrainLoss {mean_train:.4f} ValLoss {mean_val:.4f}"
        )

        if mean_val < best_val:
            best_val = mean_val
            best_ckpt_path = os.path.join(save_dir, "loso_best.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict() if ema_model is not None else None,
                    "config": {
                        "heldout_subject_id": args.heldout_subject_id,
                        "subject_ids": all_ids,
                        "img_size": args.img_size,
                        "base_channels": args.base_channels,
                        "num_timesteps": args.num_timesteps,
                        "n_res_blocks": args.n_res_blocks,
                        "ch_mult": args.ch_mult,
                        "lambda_rec": args.lambda_rec,
                    },
                },
                best_ckpt_path,
            )
            print(f"[LOSO-128] Updated best checkpoint -> {best_ckpt_path}")

    final_ckpt_path = os.path.join(save_dir, "loso_final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema_model.state_dict() if ema_model is not None else None,
            "config": {
                "heldout_subject_id": args.heldout_subject_id,
                "subject_ids": all_ids,
                "img_size": args.img_size,
                "base_channels": args.base_channels,
                "num_timesteps": args.num_timesteps,
                "n_res_blocks": args.n_res_blocks,
                "ch_mult": args.ch_mult,
                "lambda_rec": args.lambda_rec,
            },
        },
        final_ckpt_path,
    )
    print(f"[LOSO-128] Training finished. Final ckpt -> {final_ckpt_path}")

    np.save(os.path.join(save_dir, "train_loss.npy"), np.array(train_losses, dtype=np.float32))
    np.save(os.path.join(save_dir, "val_loss.npy"), np.array(val_losses, dtype=np.float32))


def build_parser():
    parser = argparse.ArgumentParser(
        description="LOSO training for EEG-to-image diffusion (128x128)."
    )
    parser.add_argument("--data_root", type=str, default="./preproc_data")
    parser.add_argument("--heldout_subject_id", type=int, required=True)
    parser.add_argument("--subject_ids", type=str, default="",
                        help="comma list; if empty, auto-discover in data_root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--ch_mult", type=str, default="1,2,4,8")
    parser.add_argument("--n_res_blocks", type=int, default=2)
    parser.add_argument("--num_timesteps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--ckpt_root", type=str, default="./checkpoints_loso128")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--lambda_rec", type=float, default=0.02)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
