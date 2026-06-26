# sample_subject_all.py
import os
import glob
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset_subject import EEGImageSubjectDataset
from model import EEGDiffusionModel
from sample_subject import sample_ddim  # 이미 만든 단일 샘플러 재사용

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------- 1) test dataset & dataloader ----------
    test_ds = EEGImageSubjectDataset(
        data_root=args.data_root,
        subject_id=args.subject_id,
        split="test",
        split_ratio=args.split_ratio,
        img_size=args.img_size,
        seed=args.seed,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"[Subj {args.subject_id:02d}] #test trials = {len(test_ds)}")

    # ---------- 2) 모델 생성 & 체크포인트 로드 ----------
    # EEG 채널 수는 dataset에서 가져오기
    eeg_sample, img_sample, _ = test_ds[0]
    eeg_channels = eeg_sample.shape[0]

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
        # checkpoints_subj/*/subjXX_final.pt 중 가장 최근 run 사용
        pattern = os.path.join(
            args.out_dir,
            "*",
            f"subj{args.subject_id:02d}_final.pt"
        )
        candidates = glob.glob(pattern)
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found matching pattern: {pattern}")
        candidates.sort()  # run 폴더명이 YYYYMMDD_HHMMSS_subjXX 형식이라 lex sort로 최신이 마지막
        ckpt_path = candidates[-1]

    print(f"[Subj {args.subject_id:02d}] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---------- 3) 샘플 결과 저장 폴더 (run_id 기준) ----------
    ckpt_dir = os.path.dirname(ckpt_path)
    run_id = os.path.basename(ckpt_dir)  # 예: 20251207_150230_subj01

    sample_root = args.sample_dir
    sample_dir = os.path.join(sample_root, run_id)
    os.makedirs(sample_dir, exist_ok=True)
    print(f"[Subj {args.subject_id:02d}] Samples will be saved under: {sample_dir}")

    # ---------- 4) test 전체 trial에 대해 이미지 생성 ----------
    global_idx = 0
    model.to(device)

    for eeg, img_gt, label in tqdm(test_loader, desc=f"Sampling subj{args.subject_id:02d}"):
        eeg = eeg.to(device)  # (B, 32, 512)

        # EEG 조건으로 batch 샘플링
        samples = sample_ddim(
            model,
            eeg,
            num_steps=args.num_timesteps,
            img_size=args.img_size,
            eta=0.0,
        )

        # 각 trial별로 파일 저장
        batch_size = eeg.size(0)
        for b in range(batch_size):
            idx = global_idx + b
            lab = int(label[b])

            # 생성 이미지
            gen_name = f"subj{args.subject_id:02d}_testIdx{idx:03d}_label{lab}.png"
            gen_path = os.path.join(sample_dir, gen_name)
            save_image(samples[b:b+1], gen_path, nrow=1)

            # GT 이미지도 같이 저장 (옵션)
            gt_name = f"subj{args.subject_id:02d}_testIdx{idx:03d}_label{lab}_GT.png"
            gt_path = os.path.join(sample_dir, gt_name)
            save_image(img_gt[b:b+1], gt_path, nrow=1)

        global_idx += batch_size

    print(f"[Subj {args.subject_id:02d}] Done. Generated {global_idx} images.")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./preproc_data")
    p.add_argument("--subject_id", type=int, default=1)
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
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
