# sample_subject.py
import os
import glob
import argparse

import torch
from torchvision.utils import save_image

from dataset_subject import EEGImageSubjectDataset
from model import EEGDiffusionModel


@torch.no_grad()
def sample_ddim(model, eeg, num_steps=None, img_size=64, eta=0.0):
    device = next(model.parameters()).device
    B = eeg.size(0)
    num_timesteps = model.num_timesteps if num_steps is None else num_steps

    x = torch.randn(B, model.img_channels, img_size, img_size, device=device)

    alphas_cumprod = model.alphas_cumprod
    sqrt_alphas_cumprod = model.sqrt_alphas_cumprod
    sqrt_one_minus_alphas_cumprod = model.sqrt_one_minus_alphas_cumprod

    for i in reversed(range(num_timesteps)):
        t = torch.full((B,), i, device=device, dtype=torch.long)

        cond = model.get_cond(t, eeg)
        eps_theta = model.unet(x, cond)

        sqrt_alpha_t = sqrt_alphas_cumprod[i].view(1, 1, 1, 1)
        sqrt_one_minus_t = sqrt_one_minus_alphas_cumprod[i].view(1, 1, 1, 1)

        x0_pred = (x - sqrt_one_minus_t * eps_theta) / sqrt_alpha_t

        if i > 0:
            sqrt_alpha_prev = sqrt_alphas_cumprod[i - 1].view(1, 1, 1, 1)
            sqrt_one_minus_prev = torch.sqrt(
                1.0 - alphas_cumprod[i - 1]
            ).view(1, 1, 1, 1)

            if eta > 0.0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)

            x = sqrt_alpha_prev * x0_pred + sqrt_one_minus_prev * z
        else:
            x = x0_pred

    img = (x + 1.0) / 2.0
    img = torch.clamp(img, 0.0, 1.0)
    return img


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------- 1) subject dataset에서 EEG 하나 가져오기 ----------
    ds = EEGImageSubjectDataset(
        data_root=args.data_root,
        subject_id=args.subject_id,
        split="test",
        split_ratio=args.split_ratio,
        img_size=args.img_size,
        seed=args.seed,
    )

    assert 0 <= args.trial_index < len(ds), f"trial_index는 0 ~ {len(ds)-1} 사이여야 합니다."

    eeg, img_gt, label = ds[args.trial_index]
    print(f"Sampling for subject {args.subject_id:02d}, trial {args.trial_index}, label = {label}")

    eeg = eeg.unsqueeze(0).to(device)   # (1, 32, 512)

    # ---------- 2) 모델 생성 & 체크포인트 로드 ----------
    eeg_channels = eeg.shape[1]

    model = EEGDiffusionModel(
        img_size=args.img_size,
        img_channels=3,
        eeg_channels=eeg_channels,
        eeg_hidden_dim=args.eeg_hidden_dim,
        time_dim=args.time_dim,
        base_channels=args.base_channels,
        num_timesteps=args.num_timesteps,
    ).to(device)

    ckpt_path = args.ckpt_path
    if ckpt_path is None:
        # checkpoints_subj/*/subjXX_final.pt 중 가장 최근(run 폴더명 기준) 선택
        pattern = os.path.join(
            args.out_dir,
            "*",
            f"subj{args.subject_id:02d}_final.pt"
        )
        candidates = glob.glob(pattern)
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found matching pattern: {pattern}")
        candidates.sort()  # 폴더명이 YYYYMMDD_HHMMSS 형식이라 lex sort로 최신이 마지막
        ckpt_path = candidates[-1]

    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---------- 3) 샘플 결과 저장 폴더 (run_id 기준) ----------
    # ckpt_path: .../checkpoints_subj/<run_id>/subjXX_final.pt
    ckpt_dir = os.path.dirname(ckpt_path)
    run_id = os.path.basename(ckpt_dir)

    sample_root = args.sample_dir
    sample_dir = os.path.join(sample_root, run_id)
    os.makedirs(sample_dir, exist_ok=True)
    print(f"Samples will be saved under: {sample_dir}")

    # ---------- 4) EEG 조건으로 이미지 샘플링 ----------
    samples = sample_ddim(
        model,
        eeg,
        num_steps=args.num_timesteps,
        img_size=args.img_size,
        eta=0.0,
    )

    samples = samples.clamp(-1, 1)
    samples = (samples + 1.0) / 2.0  # [-1,1] -> [0,1]

    out_path = os.path.join(
        sample_dir,
        f"subj{args.subject_id:02d}_trial{args.trial_index:03d}_label{label}.png",
    )
    save_image(samples, out_path, nrow=1)
    print("Saved generated image to:", out_path)

    gt_out = os.path.join(
        sample_dir,
        f"subj{args.subject_id:02d}_trial{args.trial_index:03d}_label{label}_GT.png",
    )
    save_image(img_gt.unsqueeze(0), gt_out, nrow=1)
    print("Saved ground-truth image to:", gt_out)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./preproc_data")
    p.add_argument("--subject_id", type=int, default=1)
    p.add_argument("--trial_index", type=int, default=0)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--num_timesteps", type=int, default=200)
    p.add_argument("--eeg_hidden_dim", type=int, default=256)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--split_ratio", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--out_dir", type=str, default="./checkpoints_subj")
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--sample_dir", type=str, default="./samples_subj")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
