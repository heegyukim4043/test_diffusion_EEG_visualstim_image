# sample_resting_eeg_128_eegonly_transformer.py
import os
import argparse

import numpy as np
from scipy.io import loadmat
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from model_128_eegonly_transformer import EEGDiffusionModel128


class RestingEEGDataset(Dataset):
    """
    Expected .mat format:
    - X: (ch, time, trial)
    """

    def __init__(self, mat_path: str):
        super().__init__()
        mat = loadmat(mat_path)
        if "X" in mat:
            X = mat["X"]
        else:
            # Fallback: pick first 3D numeric array
            X = None
            for k, v in mat.items():
                if k.startswith("__"):
                    continue
                if isinstance(v, np.ndarray) and v.ndim == 3:
                    X = v
                    break
            if X is None:
                raise ValueError(f"No 3D EEG array found in {mat_path}")

        self.eeg = torch.from_numpy(X).float().permute(2, 0, 1)  # (trial, ch, time)

    def __len__(self):
        return self.eeg.size(0)

    def __getitem__(self, idx):
        return self.eeg[idx], idx


def find_latest_group_ckpt_dir(ckpt_root, subject_id, group_idx, cls_low, cls_high, img_size):
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


def parse_subject_ids(subject_id, subject_ids):
    if subject_ids:
        s = subject_ids.strip()
        if "-" in s and "," not in s:
            a, b = s.split("-", 1)
            return list(range(int(a), int(b) + 1))
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    if subject_id is None:
        raise ValueError("Provide either --subject_id or --subject_ids")
    return [int(subject_id)]


def parse_group_ids(group_id, group_ids):
    if group_ids:
        s = group_ids.strip()
        if "-" in s and "," not in s:
            a, b = s.split("-", 1)
            ids = list(range(int(a), int(b) + 1))
        else:
            ids = [int(x.strip()) for x in s.split(",") if x.strip()]
    else:
        ids = [int(group_id)]

    for gid in ids:
        if gid not in (1, 2, 3):
            raise ValueError("group_id/group_ids must be in {1,2,3}")
    return ids


def run_one_subject_group(args, subject_id, group_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subj_str = f"{subject_id:02d}"

    if group_id == 1:
        cls_low, cls_high = 1, 3
    elif group_id == 2:
        cls_low, cls_high = 4, 6
    elif group_id == 3:
        cls_low, cls_high = 7, 9
    else:
        raise ValueError("group_id must be 1, 2, or 3")

    mat_path = os.path.join(args.data_root, f"subj_{subj_str}.mat")
    if not os.path.isfile(mat_path):
        print(f"[WARN] Resting EEG mat not found: {mat_path}. Skipping.")
        return

    ds = RestingEEGDataset(mat_path)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None:
        ckpt_dir = find_latest_group_ckpt_dir(
            args.ckpt_root,
            subject_id,
            group_id - 1,
            cls_low,
            cls_high,
            args.img_size,
        )

    if ckpt_dir is None or (not os.path.isdir(ckpt_dir)):
        print(f"[WARN] Checkpoint dir not found for subject {subj_str}. Skipping.")
        return

    best_path = os.path.join(ckpt_dir, f"subj{subj_str}_g{group_id}_best.pt")
    final_path = os.path.join(ckpt_dir, f"subj{subj_str}_g{group_id}_final.pt")

    if os.path.isfile(best_path):
        ckpt_path = best_path
    elif os.path.isfile(final_path):
        ckpt_path = final_path
    else:
        print(f"[WARN] No best/final checkpoint for subject {subj_str} in: {ckpt_dir}. Skipping.")
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
    out_dir = os.path.join(args.samples_root, f"{ckpt_basename}_rest")
    os.makedirs(out_dir, exist_ok=True)

    to_pil = T.ToPILImage()

    with torch.no_grad():
        saved = 0
        for eeg, trial_idx in loader:
            eeg = eeg.to(device)

            if args.use_ddim:
                x_gen = model.sample_ddim(
                    eeg=eeg,
                    num_steps=args.sample_steps,
                    guidance_scale=args.guidance_scale,
                    eta=args.ddim_eta,
                )
            else:
                dummy_labels = torch.zeros(eeg.size(0), device=device, dtype=torch.long)
                x_gen = model.sample(
                    eeg=eeg,
                    labels=dummy_labels,
                    num_steps=args.sample_steps,
                    guidance_scale=args.guidance_scale,
                )

            x_gen = (x_gen.clamp(-1, 1) + 1.0) * 0.5  # [0,1]

            for i in range(eeg.size(0)):
                t_idx = int(trial_idx[i].item())
                gen_pil = to_pil(x_gen[i].cpu())
                gen_name = f"subj{subj_str}_g{group_id}_rest_trial{t_idx:03d}_GEN.png"
                gen_pil.save(os.path.join(out_dir, gen_name))
                saved += 1

    print(f"Saved {saved} resting-GEN images to: {out_dir}")


def main(args):
    subject_list = parse_subject_ids(args.subject_id, args.subject_ids)
    group_list = parse_group_ids(args.group_id, args.group_ids)
    for sid in subject_list:
        for gid in group_list:
            run_one_subject_group(args, sid, gid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images from resting EEG (X only) using EEG-only Transformer checkpoints."
    )
    parser.add_argument("--data_root", type=str, default="./preproc_data_rest")
    parser.add_argument("--subject_id", type=int, default=None)
    parser.add_argument("--subject_ids", type=str, default="",
                        help="multi-subject input, e.g. '2-30' or '2,3,4'")
    parser.add_argument("--group_id", type=int, default=1)
    parser.add_argument("--group_ids", type=str, default="",
                        help="multi-group input, e.g. '1-3' or '1,2,3'")
    parser.add_argument("--img_size", type=int, default=128)

    parser.add_argument("--ckpt_root", type=str, default="./checkpoints_subj128_group_eegonly_tf")
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--samples_root", type=str, default="./samples_rest_eegonly_tf")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--num_timesteps", type=int, default=2000)
    parser.add_argument("--n_res_blocks", type=int, default=2)
    parser.add_argument("--ch_mult", type=str, default="1,2,4,8")
    parser.add_argument("--sample_steps", type=int, default=500)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--use_ddim", action="store_true", default=False)
    parser.add_argument("--ddim_eta", type=float, default=0.0)

    parser.add_argument("--eeg_tf_heads", type=int, default=4)
    parser.add_argument("--eeg_tf_layers", type=int, default=2)
    parser.add_argument("--eeg_tf_dropout", type=float, default=0.1)

    args = parser.parse_args()
    main(args)
