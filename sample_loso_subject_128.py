# sample_loso_subject_128.py
import os
import argparse
from glob import glob

import numpy as np
from scipy.io import loadmat
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from model_128 import EEGDiffusionModel128


class EEGImageSubjectAllDataset(Dataset):
    def __init__(self, mat_path, img_root, img_size=128):
        mat = loadmat(mat_path)
        X = mat["X"]  # (ch, time, trial)
        y = mat["y"].squeeze()

        self.eeg = torch.from_numpy(X).float().permute(2, 0, 1)
        self.labels = y.astype(np.int64)
        self.img_root = img_root

        self.transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = self.eeg[idx]
        label_global = int(self.labels[idx])
        img_path = os.path.join(self.img_root, f"{label_global:02d}.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return eeg, img, label_global, idx


def find_latest_loso_ckpt_dir(ckpt_root, heldout_subject_id, img_size):
    target_suffix = f"_loso_holdout{heldout_subject_id:02d}_{img_size}"
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
    subj_str = f"{args.heldout_subject_id:02d}"

    print("=" * 80)
    print(f"[LOSO-Sample] Held-out subject {subj_str}, device: {device}")
    print("=" * 80)

    mat_path = os.path.join(args.data_root, f"subj_{subj_str}.mat")
    img_root = os.path.join(args.data_root, "images")

    test_ds = EEGImageSubjectAllDataset(mat_path, img_root, img_size=args.img_size)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None:
        ckpt_dir = find_latest_loso_ckpt_dir(args.ckpt_root, args.heldout_subject_id, args.img_size)

    if ckpt_dir is None or (not os.path.isdir(ckpt_dir)):
        print("Checkpoint dir not found.")
        return

    best_path = os.path.join(ckpt_dir, "loso_best.pt")
    final_path = os.path.join(ckpt_dir, "loso_final.pt")

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
        num_classes=9,
        num_timesteps=num_timesteps,
        base_channels=base_channels,
        time_dim=256,
        cond_dim=256,
        eeg_hidden_dim=256,
        cond_scale=2.0,
        n_res_blocks=n_res_blocks,
        ch_mult=ch_mult,
    ).to(device)

    state_dict = ckpt.get("ema", ckpt["model"]) or ckpt["model"]
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
        for eeg, img_gt, label_global, trial_idx in test_loader:
            eeg = eeg.to(device)
            img_gt = img_gt.to(device)
            labels = label_global.to(device) - 1  # 0-based

            x_gen = model.sample(
                eeg=eeg,
                labels=labels,
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

                gen_name = f"holdout{subj_str}_trial{t_idx:03d}_label{g_label}_GEN.png"
                gt_name = f"holdout{subj_str}_trial{t_idx:03d}_label{g_label}_GT.png"

                gen_pil.save(os.path.join(samples_dir, gen_name))
                gt_pil.save(os.path.join(samples_dir, gt_name))

                global_sample_idx += 1

        print(f"[LOSO-Sample] Done. Total saved: {global_sample_idx * 2}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="LOSO sampling on held-out subject (128x128)."
    )
    parser.add_argument("--data_root", type=str, default="./preproc_data")
    parser.add_argument("--heldout_subject_id", type=int, required=True)
    parser.add_argument("--img_size", type=int, default=128)

    parser.add_argument("--ckpt_root", type=str, default="./checkpoints_loso128")
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--samples_root", type=str, default="./samples_loso128")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--num_timesteps", type=int, default=2000)
    parser.add_argument("--n_res_blocks", type=int, default=2)
    parser.add_argument("--ch_mult", type=str, default="1,2,4,8")
    parser.add_argument("--sample_steps", type=int, default=200)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
