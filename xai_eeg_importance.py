# xai_eeg_importance.py
import os
import argparse
import numpy as np
from scipy.io import loadmat
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt

from model_128_eegonly import EEGDiffusionModel128


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
        X = mat["X"]  # (ch, time, trial)
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
        label_global = int(self.labels[trial_idx])  # 1~9
        label_local = label_global - self.cls_low   # (cls_low..cls_high)->0,1,2

        img_path = os.path.join(self.img_root, f"{label_global:02d}.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)  # [-1,1]

        return eeg, img, label_local, label_global, trial_idx


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_group_test_indices(
    mat_path: str,
    subject_id: int,
    group_idx: int,
    cls_low: int,
    cls_high: int,
    seed: int,
):
    mat = loadmat(mat_path)
    y = mat["y"].squeeze().astype(np.int64)

    mask_group = (y >= cls_low) & (y <= cls_high)
    all_indices = np.where(mask_group)[0]
    n_total = len(all_indices)

    if n_total == 0:
        return np.array([], dtype=np.int64)

    rng = np.random.RandomState(seed + subject_id * 10 + group_idx)
    rng.shuffle(all_indices)

    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    train_idx = all_indices[:n_train]
    val_idx = all_indices[n_train:n_train + n_val]
    test_idx = all_indices[n_train + n_val:]

    return test_idx


def find_latest_group_ckpt_dir(ckpt_root, subject_id, group_idx,
                               cls_low, cls_high, img_size):
    subj_str = f"{subject_id:02d}"
    target_suffix = f"_subj{subj_str}_g{group_idx+1}_cls{cls_low}-{cls_high}_{img_size}"

    if not os.path.isdir(ckpt_root):
        return None

    cand_dirs = []
    for name in os.listdir(ckpt_root):
        full = os.path.join(ckpt_root, name)
        if not os.path.isdir(full):
            continue
        if name.endswith(target_suffix):
            cand_dirs.append(full)

    if not cand_dirs:
        return None

    cand_dirs.sort()
    return cand_dirs[-1]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subj_str = f"{args.subject_id:02d}"

    if args.group_id == 1:
        cls_low, cls_high = 1, 3
    elif args.group_id == 2:
        cls_low, cls_high = 4, 6
    elif args.group_id == 3:
        cls_low, cls_high = 7, 9
    else:
        raise ValueError("group_id must be 1, 2, or 3")

    print("=" * 80)
    print(
        f"[XAI] Subject {subj_str}, Group {args.group_id} "
        f"(classes {cls_low}~{cls_high}), device: {device}"
    )
    print("=" * 80)

    mat_path = os.path.join(args.data_root, f"subj_{subj_str}.mat")
    img_root = os.path.join(args.data_root, "images")

    set_seed(args.seed)
    test_idx = get_group_test_indices(
        mat_path,
        subject_id=args.subject_id,
        group_idx=args.group_id - 1,
        cls_low=cls_low,
        cls_high=cls_high,
        seed=args.seed,
    )
    if len(test_idx) == 0:
        print("No test trials found.")
        return

    test_ds = EEGImageDatasetGroup128(
        mat_path=mat_path,
        img_root=img_root,
        indices=test_idx,
        img_size=args.img_size,
        cls_low=cls_low,
        cls_high=cls_high,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None:
        ckpt_dir = find_latest_group_ckpt_dir(
            args.ckpt_root,
            args.subject_id,
            args.group_id - 1,
            cls_low,
            cls_high,
            args.img_size,
        )

    if ckpt_dir is None or (not os.path.isdir(ckpt_dir)):
        print("Checkpoint dir not found.")
        return

    best_path = os.path.join(ckpt_dir, f"subj{subj_str}_g{args.group_id}_best.pt")
    final_path = os.path.join(ckpt_dir, f"subj{subj_str}_g{args.group_id}_final.pt")

    if os.path.isfile(best_path):
        ckpt_path = best_path
    elif os.path.isfile(final_path):
        ckpt_path = final_path
    else:
        print("No best/final checkpoint found.")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})

    img_size = cfg.get("img_size", args.img_size)
    base_channels = cfg.get("base_channels", args.base_channels)
    num_timesteps = cfg.get("num_timesteps", args.num_timesteps)
    n_res_blocks = cfg.get("n_res_blocks", args.n_res_blocks)

    model = EEGDiffusionModel128(
        img_size=img_size,
        img_channels=3,
        eeg_channels=32,
        num_classes=3,
        num_timesteps=num_timesteps,
        base_channels=base_channels,
        time_dim=256,
        cond_dim=256,
        eeg_hidden_dim=256,
        cond_scale=2.0,
        n_res_blocks=n_res_blocks,
    ).to(device)

    use_ema = not args.no_ema
    state_dict = ckpt.get("ema", ckpt["model"]) if use_ema else ckpt["model"]
    model.load_state_dict(state_dict)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    total_grad = None
    total_count = 0

    for batch_idx, (eeg, img_gt, _, _, _) in enumerate(test_loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        eeg = eeg.to(device)
        img_gt = img_gt.to(device)

        eeg.requires_grad_(True)

        b = eeg.size(0)
        if args.timestep < 0:
            t = torch.randint(0, model.num_timesteps, size=(b,), device=device)
        else:
            t = torch.full((b,), args.timestep, device=device, dtype=torch.long)

        noise = torch.randn_like(img_gt)
        x_noisy = model.q_sample(img_gt, t, noise)

        t_emb = model.time_embed(t)
        cond = model.get_cond_emb_eeg_only(eeg)
        cond = cond * args.guidance_scale

        eps_pred = model.unet(x_noisy, t_emb, cond)

        sqrt_alpha_bar = model._extract(model.sqrt_alphas_cumprod, t, img_gt.shape)
        sqrt_one_minus = model._extract(model.sqrt_one_minus_alphas_cumprod, t, img_gt.shape)
        x0_pred = (x_noisy - sqrt_one_minus * eps_pred) / (sqrt_alpha_bar + 1e-8)
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        if args.loss == "l1":
            loss = (x0_pred - img_gt).abs().mean()
        else:
            loss = ((x0_pred - img_gt) ** 2).mean()

        loss.backward()

        grad = eeg.grad.detach().abs().cpu().numpy()  # (B, C, T)
        grad = grad.mean(axis=0)  # (C, T)

        if total_grad is None:
            total_grad = grad
        else:
            total_grad += grad
        total_count += 1

        if eeg.grad is not None:
            eeg.grad.zero_()

        print(f"Processed batch {batch_idx + 1}")

    if total_grad is None:
        print("No gradients computed.")
        return

    total_grad /= max(total_count, 1)

    channel_importance = total_grad.mean(axis=1)  # (C,)
    time_importance = total_grad.mean(axis=0)     # (T,)

    np.save(os.path.join(args.out_dir, "eeg_importance_heatmap.npy"), total_grad)
    np.save(os.path.join(args.out_dir, "eeg_importance_channels.npy"), channel_importance)
    np.save(os.path.join(args.out_dir, "eeg_importance_time.npy"), time_importance)

    np.savetxt(os.path.join(args.out_dir, "eeg_importance_channels.csv"), channel_importance, delimiter=",")
    np.savetxt(os.path.join(args.out_dir, "eeg_importance_time.csv"), time_importance, delimiter=",")

    # heatmap plot
    plt.figure(figsize=(10, 4))
    plt.imshow(total_grad, aspect="auto", cmap="viridis")
    plt.colorbar(label="|grad|")
    plt.xlabel("Time")
    plt.ylabel("Channel")
    plt.title("EEG Importance Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "eeg_importance_heatmap.png"))
    plt.close()

    print(f"Saved XAI outputs to: {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG channel/time importance for EEG-to-image (gradient saliency)."
    )
    parser.add_argument("--data_root", type=str, default="./preproc_data")
    parser.add_argument("--subject_id", type=int, required=True)
    parser.add_argument("--group_id", type=int, required=True)
    parser.add_argument("--img_size", type=int, default=128)

    parser.add_argument("--ckpt_root", type=str, default="./checkpoints_subj128_group")
    parser.add_argument("--ckpt_dir", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--num_timesteps", type=int, default=2000)
    parser.add_argument("--n_res_blocks", type=int, default=2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--timestep", type=int, default=100,
                        help="Fixed timestep for attribution; use -1 for random.")
    parser.add_argument("--loss", type=str, choices=["l1", "l2"], default="l1")
    parser.add_argument("--max_batches", type=int, default=5)
    parser.add_argument("--no_ema", action="store_true", default=False,
                        help="disable EMA weights")

    parser.add_argument("--out_dir", type=str, default="./xai_eeg_importance")

    args = parser.parse_args()
    main(args)
