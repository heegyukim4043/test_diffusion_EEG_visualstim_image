sample_subject_segments_all.py
import os
import glob
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset_subject_segments import EEGImageSubjectSegmentDataset
from model import EEGDiffusionModel
from sample_subject import sample_ddim  # 기존 단일 샘플러 재사용

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------- 1) test dataset & dataloader (segment 기반) ----------
    test_ds = EEGImageSubjectSegmentDataset(
        data_root=args.data_root,
        subject_id=args.subject_id,
        split="test",
        split_ratio=args.split_ratio,
        img_size=args.img_size,
        seed=args.seed,
        num_segments=args.num_segments,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"[Subj {args.subject_id:02d}] #test segments = {len(test_ds)} "
          f"(num_segments={args.num_segments})")

    # ---------- 2) 모델 생성 & checkpoint 로드 ----------
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
        # segment 트레이닝용 out_dir에서 최신 subjXX_final.pt 선택
        pattern = os.path.join(
            args.out_dir,
            "*",
            f"subj{args.subject_id:02d}_final.pt"
        )
        candidates = glob.glob(pattern)
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found matching pattern: {pattern}")
        candidates.sort()
        ckpt_path = candidates[-1]

    print(f"[Subj {args.subject_id:02d}] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---------- 3) run_id 기반 sample 폴더 ----------
    ckpt_dir = os.path.dirname(ckpt_path)
    run_id = os.path.basename(ckpt_dir)  # 예: 20251207_153000_subj01_seg

    sample_root = args.sample_dir
    sample_dir = os.path.join(sample_root, run_id)
    os.makedirs(sample_dir, exist_ok=True)
    print(f"[Subj {args.subject_id:02d}] Samples will be saved under: {sample_dir}")

    # ---------- 4) test segment 전체에 대해 생성 ----------
    model.to(device)
    global_idx = 0

    for eeg_seg, img_gt, label in tqdm(
        test_loader,
        desc=f"Sampling subj{args.subject_id:02d}_seg"
    ):
        eeg_seg = eeg_seg.to(device)  # (B, 32, segment_len)

        samples = sample_ddim(
            model,
            eeg_seg,
            num_steps=args.num_timesteps,
            img_size=args.img_size,
            eta=0.0,
        )

        batch_size = eeg_seg.size(0)
        for b in range(batch_size):
            idx = global_idx + b
            lab = int(label[b])
            # trial_idx, seg_idx 얻기
            trial_idx, seg_idx = test_ds.get_trial_segment_info(idx)

            base_name = (
                f"subj{args.subject_id:02d}_"
                f"trial{trial_idx:03d}_seg{seg_idx}_label{lab}"
            )

            gen_path = os.path.join(sample_dir, base_name + ".png")
            gt_path = os.path.join(sample_dir, base_name + "_GT.png")

            save_image(samples[b:b+1], gen_path, nrow=1)
            save_image(img_gt[b:b+1], gt_path, nrow=1)

        global_idx += batch_size

    print(f"[Subj {args.subject_id:02d}] Done. Generated {global_idx} segment images.")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./preproc_data")
    p.add_argument("--subject_id", type=int, default=1)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--num_timesteps", type=int, default=200)
    p.add_argument("--eeg_hidden_dim", type=int, default=256)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--split_ratio", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--out_dir", type=str, default="./checkpoints_subj_seg")
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--sample_dir", type=str, default="./samples_subj_seg")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_segments", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
