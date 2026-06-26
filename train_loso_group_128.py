# train_loso_group_128.py
import os
import argparse
import random
from datetime import datetime
from glob import glob

import numpy as np
from scipy.io import loadmat
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

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


class LOSOGroupDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        subject_ids,
        split: str,
        split_ratio: float,
        img_size: int,
        seed: int,
        cls_low: int,
        cls_high: int,
    ):
        super().__init__()
        assert split in ("train", "val")

        self.subject_ids = sorted(list(subject_ids))
        self.cls_low = cls_low
        self.cls_high = cls_high

        self.eeg_list = []
        self.label_list = []
        self.transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.img_root = os.path.join(data_root, "images")

        for sid in self.subject_ids:
            mat_path = os.path.join(data_root, f"subj_{sid:02d}.mat")
            mat = loadmat(mat_path)
            X = mat["X"]  # (ch, time, trial)
            y = mat["y"].squeeze().astype(np.int64)  # (trial,)

            mask_group = (y >= cls_low) & (y <= cls_high)
            all_indices = np.where(mask_group)[0]
            if len(all_indices) == 0:
                continue

            rng = np.random.RandomState(seed + sid * 10 + cls_low)
            rng.shuffle(all_indices)

            n_train = int(len(all_indices) * split_ratio)
            if split == "train":
                use_idx = all_indices[:n_train]
            else:
                use_idx = all_indices[n_train:]

            eeg = torch.from_numpy(X).float().permute(2, 0, 1)  # (trial, ch, time)
            self.eeg_list.append(eeg[use_idx])
            self.label_list.append(y[use_idx])

        if self.eeg_list:
            self.eeg = torch.cat(self.eeg_list, dim=0)
            self.labels = np.concatenate(self.label_list, axis=0)
        else:
            self.eeg = torch.empty(0, 32, 1)
            self.labels = np.empty((0,), dtype=np.int64)

        print(
            f"[LOSO-Group] subjects={self.subject_ids}, split={split}, "
            f"classes={cls_low}-{cls_high}, total={len(self.labels)}"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = self.eeg[idx]
        label_global = int(self.labels[idx])
        label_local = label_global - self.cls_low

        img_path = os.path.join(self.img_root, f"{label_global:02d}.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return eeg, img, label_local


def train_one_group(args, train_ids, group_idx, cls_low, cls_high):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = LOSOGroupDataset(
        data_root=args.data_root,
        subject_ids=train_ids,
        split="train",
        split_ratio=args.split_ratio,
        img_size=args.img_size,
        seed=args.seed,
        cls_low=cls_low,
        cls_high=cls_high,
    )
    val_ds = LOSOGroupDataset(
        data_root=args.data_root,
        subject_ids=train_ids,
        split="val",
        split_ratio=args.split_ratio,
        img_size=args.img_size,
        seed=args.seed,
        cls_low=cls_low,
        cls_high=cls_high,
    )

    if len(train_ds) == 0:
        print(f"[LOSO-Group] No training data for classes {cls_low}-{cls_high}")
        return

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
        num_classes=3,
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
            num_classes=3,
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
        f"{timestamp}_loso_holdout{args.heldout_subject_id:02d}_g{group_idx+1}_cls{cls_low}-{cls_high}_{args.img_size}"
    )
    os.makedirs(save_dir, exist_ok=True)
    print(f"[LOSO-Group] Checkpoints & logs -> {save_dir}")

    train_losses = []
    val_losses = []
    best_val = float("inf")

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
                    f"[LOSO-Group] Epoch {epoch} Step {batch_idx}/{len(train_loader)} "
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
            f"[LOSO-Group] Epoch {epoch} TrainLoss {mean_train:.4f} ValLoss {mean_val:.4f}"
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
                        "subject_ids": args.subject_ids,
                        "group_idx": group_idx,
                        "cls_low": cls_low,
                        "cls_high": cls_high,
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
            print(f"[LOSO-Group] Updated best checkpoint -> {best_ckpt_path}")

    final_ckpt_path = os.path.join(save_dir, "loso_final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema_model.state_dict() if ema_model is not None else None,
            "config": {
                "heldout_subject_id": args.heldout_subject_id,
                "subject_ids": args.subject_ids,
                "group_idx": group_idx,
                "cls_low": cls_low,
                "cls_high": cls_high,
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
    print(f"[LOSO-Group] Training finished. Final ckpt -> {final_ckpt_path}")

    np.save(os.path.join(save_dir, "train_loss.npy"), np.array(train_losses, dtype=np.float32))
    np.save(os.path.join(save_dir, "val_loss.npy"), np.array(val_losses, dtype=np.float32))


def main(args):
    if args.subject_ids:
        all_ids = [int(s.strip()) for s in args.subject_ids.split(",") if s.strip()]
    else:
        all_ids = discover_subject_ids(args.data_root)

    if args.heldout_subject_id not in all_ids:
        raise ValueError("heldout_subject_id not found in data_root")

    train_ids = [sid for sid in all_ids if sid != args.heldout_subject_id]

    print("=" * 80)
    print(f"[LOSO-Group] Held-out subject: {args.heldout_subject_id:02d}")
    print(f"[LOSO-Group] Train subjects: {train_ids}")
    print("=" * 80)

    set_seed(args.seed)

    groups = [
        (1, 3),
        (4, 6),
        (7, 9),
    ]

    for g_idx, (cls_low, cls_high) in enumerate(groups):
        train_one_group(args, train_ids, g_idx, cls_low, cls_high)


def build_parser():
    parser = argparse.ArgumentParser(
        description="LOSO training for EEG-to-image diffusion with class groups (128x128)."
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
    parser.add_argument("--ckpt_root", type=str, default="./checkpoints_loso128_group")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--lambda_rec", type=float, default=0.02)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
