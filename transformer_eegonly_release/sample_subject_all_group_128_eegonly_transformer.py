# sample_subject_all_group_128_eegonly_transformer.py
import os
import argparse

import numpy as np
from scipy.io import loadmat
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from model_128_eegonly_transformer import EEGDiffusionModel128


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

    test_idx_list = []
    for label in range(cls_low, cls_high + 1):
        class_indices = np.where(y == label)[0]
        if len(class_indices) == 0:
            continue

        rng = np.random.RandomState(seed + subject_id * 10 + group_idx + label)
        rng.shuffle(class_indices)

        n_train = int(len(class_indices) * 0.8)
        n_val = int(len(class_indices) * 0.1)
        test_idx_list.append(class_indices[n_train + n_val:])

    test_idx = np.concatenate(test_idx_list) if test_idx_list else np.array([], dtype=np.int64)
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
        f"[Sample-Group-EEGOnly-TF] Subject {subj_str}, Group {args.group_id} "
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

    best_path = os.path.join(
        ckpt_dir, f"subj{subj_str}_g{args.group_id}_best.pt"
    )
    final_path = os.path.join(
        ckpt_dir, f"subj{subj_str}_g{args.group_id}_final.pt"
    )

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
    ch_mult = cfg.get("ch_mult", args.ch_mult)
    if isinstance(ch_mult, str):
        ch_mult = [int(x) for x in ch_mult.split(",")]

    model = EEGDiffusionModel128(
        img_size=img_size,
        img_channels=3,
        eeg_channels=32,
        num_classes=3,
        num_timesteps=num_timesteps,
        base_channels=base_channels,
        ch_mult=ch_mult,
        time_dim=256,
        cond_dim=256,
        eeg_hidden_dim=256,
        cond_scale=2.0,
        n_res_blocks=n_res_blocks,
        eeg_tf_heads=args.eeg_tf_heads,
        eeg_tf_layers=args.eeg_tf_layers,
        eeg_tf_dropout=args.eeg_tf_dropout,
    ).to(device)

    state_dict = ckpt.get("ema", ckpt["model"])
    model.load_state_dict(state_dict)
    model.eval()

    os.makedirs(args.samples_root, exist_ok=True)
    ckpt_basename = os.path.basename(ckpt_dir.rstrip("/\\"))
    samples_dir = os.path.join(args.samples_root, ckpt_basename)
    os.makedirs(samples_dir, exist_ok=True)

    def denorm_img(x):
        return (x.clamp(-1, 1) + 1.0) * 0.5

    to_pil = T.ToPILImage()

    with torch.no_grad():
        global_sample_idx = 0
        for eeg, img_gt, label_local, label_global, trial_idx in test_loader:
            eeg = eeg.to(device)
            img_gt = img_gt.to(device)

            x_gen = model.sample(
                eeg=eeg,
                labels=label_local.to(device),
                num_steps=args.sample_steps,
                guidance_scale=args.guidance_scale,
            )

            x_gen_denorm = denorm_img(x_gen)
            img_gt_denorm = denorm_img(img_gt)

            for i in range(eeg.size(0)):
                g_label = int(label_global[i].item())
                t_idx = int(trial_idx[i].item())

                gen_pil = to_pil(x_gen_denorm[i].cpu())
                gt_pil = to_pil(img_gt_denorm[i].cpu())

                gen_name = (
                    f"subj{subj_str}_g{args.group_id}_trial{t_idx:03d}_"
                    f"label{g_label}_GEN.png"
                )
                gt_name = (
                    f"subj{subj_str}_g{args.group_id}_trial{t_idx:03d}_"
                    f"label{g_label}_GT.png"
                )

                gen_pil.save(os.path.join(samples_dir, gen_name))
                gt_pil.save(os.path.join(samples_dir, gt_name))
                global_sample_idx += 1

        print(f"Saved {global_sample_idx * 2} images (GEN+GT).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG-only + Transformer EEG encoder sampling, 128x128."
    )
    parser.add_argument("--data_root", type=str, default="./preproc_data")
    parser.add_argument("--subject_id", type=int, required=True)
    parser.add_argument("--group_id", type=int, required=True)
    parser.add_argument("--img_size", type=int, default=128)

    parser.add_argument("--ckpt_root", type=str,
                        default="./checkpoints_subj128_group_eegonly_tf")
    parser.add_argument("--ckpt_dir", type=str, default=None)

    parser.add_argument("--samples_root", type=str,
                        default="./samples_subj128_group_eegonly_tf")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--num_timesteps", type=int, default=2000)
    parser.add_argument("--n_res_blocks", type=int, default=2)
    parser.add_argument("--ch_mult", type=str, default="1,2,4,8")
    parser.add_argument("--sample_steps", type=int, default=200)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--eeg_tf_heads", type=int, default=4)
    parser.add_argument("--eeg_tf_layers", type=int, default=2)
    parser.add_argument("--eeg_tf_dropout", type=float, default=0.1)

    args = parser.parse_args()
    main(args)
