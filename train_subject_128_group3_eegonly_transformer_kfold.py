# train_subject_128_group3_eegonly_transformer_kfold.py
import os
import argparse
import random
from datetime import datetime

import numpy as np
from scipy.io import loadmat
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from model_128_eegonly_transformer import EEGDiffusionModel128


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ema_update(model, ema_model, decay: float = 0.999):
    msd = model.state_dict()
    for name, param in ema_model.state_dict().items():
        if name in msd:
            param.copy_(param * decay + msd[name] * (1.0 - decay))


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
        X = mat["X"]
        y = mat["y"].squeeze()

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
        eeg = self.eeg[trial_idx]
        label_global = int(self.labels[trial_idx])
        label_local = label_global - self.cls_low

        img_path = os.path.join(self.img_root, f"{label_global:02d}.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return eeg, img, label_local


def build_stratified_folds(y, cls_low, cls_high, k_folds, seed, subject_id, group_idx):
    folds = [([], []) for _ in range(k_folds)]
    for label in range(cls_low, cls_high + 1):
        class_indices = np.where(y == label)[0]
        if len(class_indices) == 0:
            continue
        rng = np.random.RandomState(seed + subject_id * 10 + group_idx + label)
        rng.shuffle(class_indices)
        splits = np.array_split(class_indices, k_folds)
        for k in range(k_folds):
            val_idx = splits[k]
            train_idx = np.concatenate([s for i, s in enumerate(splits) if i != k])
            folds[k][0].append(train_idx)
            folds[k][1].append(val_idx)

    fold_pairs = []
    for k in range(k_folds):
        train_parts, val_parts = folds[k]
        train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
        val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
        fold_pairs.append((train_idx, val_idx))
    return fold_pairs


def train_one_fold(args, fold_idx, train_idx, val_idx, mat_path, img_root, cls_low, cls_high):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subj_str = f"{args.subject_id:02d}"

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
        eeg_tf_heads=args.eeg_tf_heads,
        eeg_tf_layers=args.eeg_tf_layers,
        eeg_tf_dropout=args.eeg_tf_dropout,
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
            eeg_tf_heads=args.eeg_tf_heads,
            eeg_tf_layers=args.eeg_tf_layers,
            eeg_tf_dropout=args.eeg_tf_dropout,
        ).to(device)
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.ckpt_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        args.ckpt_root,
        f"{timestamp}_subj{subj_str}_g{args.group_id}_fold{fold_idx}_cls{cls_low}-{cls_high}_{args.img_size}"
    )
    os.makedirs(save_dir, exist_ok=True)
    print(f"[KFold] Checkpoints & logs -> {save_dir}")

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
                    f"[KFold] Fold {fold_idx} Epoch {epoch} "
                    f"Step {batch_idx}/{len(train_loader)} Loss {loss.item():.4f}"
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

        print(f"[KFold] Fold {fold_idx} Epoch {epoch} TrainLoss {mean_train:.4f} ValLoss {mean_val:.4f}")

        if mean_val < best_val:
            best_val = mean_val
            best_ckpt_path = os.path.join(save_dir, "best.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict() if ema_model is not None else None,
                    "config": {
                        "subject_id": args.subject_id,
                        "group_id": args.group_id,
                        "fold_idx": fold_idx,
                        "cls_low": cls_low,
                        "cls_high": cls_high,
                        "img_size": args.img_size,
                        "base_channels": args.base_channels,
                        "num_timesteps": args.num_timesteps,
                        "n_res_blocks": args.n_res_blocks,
                        "ch_mult": args.ch_mult,
                        "lambda_rec": args.lambda_rec,
                        "eeg_tf_heads": args.eeg_tf_heads,
                        "eeg_tf_layers": args.eeg_tf_layers,
                        "eeg_tf_dropout": args.eeg_tf_dropout,
                    },
                },
                best_ckpt_path,
            )
            print(f"[KFold] Updated best checkpoint -> {best_ckpt_path}")

    final_ckpt_path = os.path.join(save_dir, "final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema_model.state_dict() if ema_model is not None else None,
            "config": {
                "subject_id": args.subject_id,
                "group_id": args.group_id,
                "fold_idx": fold_idx,
                "cls_low": cls_low,
                "cls_high": cls_high,
                "img_size": args.img_size,
                "base_channels": args.base_channels,
                "num_timesteps": args.num_timesteps,
                "n_res_blocks": args.n_res_blocks,
                "ch_mult": args.ch_mult,
                "lambda_rec": args.lambda_rec,
                "eeg_tf_heads": args.eeg_tf_heads,
                "eeg_tf_layers": args.eeg_tf_layers,
                "eeg_tf_dropout": args.eeg_tf_dropout,
            },
        },
        final_ckpt_path,
    )
    print(f"[KFold] Fold {fold_idx} finished. Final ckpt -> {final_ckpt_path}")

    np.save(os.path.join(save_dir, "train_loss.npy"), np.array(train_losses, dtype=np.float32))
    np.save(os.path.join(save_dir, "val_loss.npy"), np.array(val_losses, dtype=np.float32))


def main(args):
    if args.group_id == 1:
        cls_low, cls_high = 1, 3
    elif args.group_id == 2:
        cls_low, cls_high = 4, 6
    elif args.group_id == 3:
        cls_low, cls_high = 7, 9
    else:
        raise ValueError("group_id must be 1, 2, or 3")

    mat_path = os.path.join(args.data_root, f"subj_{args.subject_id:02d}.mat")
    img_root = os.path.join(args.data_root, "images")

    mat = loadmat(mat_path)
    y = mat["y"].squeeze().astype(np.int64)

    folds = build_stratified_folds(
        y=y,
        cls_low=cls_low,
        cls_high=cls_high,
        k_folds=args.k_folds,
        seed=args.seed,
        subject_id=args.subject_id,
        group_idx=args.group_id - 1,
    )

    set_seed(args.seed)

    if args.fold_idx >= 0:
        train_idx, val_idx = folds[args.fold_idx]
        train_one_fold(args, args.fold_idx, train_idx, val_idx, mat_path, img_root, cls_low, cls_high)
    else:
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            train_one_fold(args, fold_idx, train_idx, val_idx, mat_path, img_root, cls_low, cls_high)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="K-fold training for EEG-only Transformer model (subject + group)."
    )
    parser.add_argument("--data_root", type=str, default="./preproc_data")
    parser.add_argument("--subject_id", type=int, required=True)
    parser.add_argument("--group_id", type=int, required=True)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--fold_idx", type=int, default=-1,
                        help="set 0..k-1 to run a single fold; -1 runs all folds")

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
    parser.add_argument("--ckpt_root", type=str, default="./checkpoints_subj128_group_eegonly_tf_kfold")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--lambda_rec", type=float, default=0.02)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--eeg_tf_heads", type=int, default=4)
    parser.add_argument("--eeg_tf_layers", type=int, default=2)
    parser.add_argument("--eeg_tf_dropout", type=float, default=0.1)

    args = parser.parse_args()
    main(args)
