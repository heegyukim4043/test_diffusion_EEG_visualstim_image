# sample_subject_all_128_9class.py
import os
import argparse
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from dataset_subject import EEGImageDataset
from model_128 import EEGDiffusionModel128


def find_latest_ckpt_dir(ckpt_root: str, subject_id: int, img_size: int = 128) -> str:
    """
    ckpt_root 아래에서 해당 subject_id와 img_size를 포함하는
    가장 최근(사전식으로 마지막) 디렉토리를 찾는다.
    예) 20260112_210417_subj16_128
    """
    if not os.path.isdir(ckpt_root):
        raise FileNotFoundError(f"Checkpoint root not found: {ckpt_root}")

    sid_tag = f"subj{subject_id:02d}_{img_size}"
    candidates = [
        d for d in os.listdir(ckpt_root)
        if os.path.isdir(os.path.join(ckpt_root, d)) and sid_tag in d
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint directory for subject {subject_id} under {ckpt_root}"
        )

    candidates.sort()
    latest_dir = os.path.join(ckpt_root, candidates[-1])
    return latest_dir


def find_ckpt_path(ckpt_dir: str, subject_id: int) -> str:
    """
    우선 best → 없으면 final → 없으면 에러
    """
    best = os.path.join(ckpt_dir, f"subj{subject_id:02d}_best.pt")
    final = os.path.join(ckpt_dir, f"subj{subject_id:02d}_final.pt")

    if os.path.isfile(best):
        return best
    if os.path.isfile(final):
        return final

    raise FileNotFoundError(
        f"Neither best nor final checkpoint found in {ckpt_dir} "
        f"(expected {os.path.basename(best)} or {os.path.basename(final)})"
    )


def build_test_loader(
    data_root: str,
    subject_id: int,
    img_size: int,
    batch_size: int,
    split_seed: int = 42,
):
    """
    한 subject 전체 trial을 불러와서
    8:1:1 (train:val:test) 분할을 재현하고,
    test subset에 대한 DataLoader를 만든다.
    """
    dataset = EEGImageDataset(
        data_root=data_root,
        subject_id=subject_id,
        img_size=img_size,
    )

    n_total = len(dataset)
    indices = np.arange(n_total)

    rng = np.random.RandomState(split_seed)
    rng.shuffle(indices)

    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"[Subj {subject_id:02d}] total={n_total}, train={len(train_idx)}, "
          f"val={len(val_idx)}, test={len(test_idx)}")

    test_subset = torch.utils.data.Subset(dataset, test_idx)
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return test_loader, test_idx, dataset


def main():
    parser = argparse.ArgumentParser(
        description="Sample all test trials (128x128, 9-class diffusion)"
    )

    parser.add_argument("--data_root", type=str, default="./preproc_data",
                        help="Root dir containing subj_XX.mat and images/")
    parser.add_argument("--subject_id", type=int, required=True,
                        help="Subject ID (e.g., 1, 2, 3, ...)")
    parser.add_argument("--img_size", type=int, default=128,
                        help="Image resolution (must match training)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for sampling")

    # UNet / diffusion 구조
    parser.add_argument("--base_channels", type=int, default=96,
                        help="Base channels of UNet (64, 96, 128, ...)")
    parser.add_argument("--ch_mult", type=str, default="1,2,2,4",
                        help="Channel multipliers per UNet level, e.g. '1,2,2,4'")
    parser.add_argument("--n_res_blocks", type=int, default=2,
                        help="Number of ResBlocks per UNet level")
    parser.add_argument("--num_timesteps", type=int, default=400,
                        help="Total diffusion timesteps used during training")

    # sampling 세부 설정
    parser.add_argument("--sample_steps", type=int, default=0,
                        help="Number of steps to use at sampling "
                             "(0 → use num_timesteps)")
    parser.add_argument("--guidance_scale", type=float, default=2.5,
                        help="Classifier-free guidance scale at sampling")

    # split 시드 (train과 맞추기)
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Seed for train/val/test split (must match training)")

    # checkpoint 경로
    parser.add_argument("--ckpt_root", type=str, default="./checkpoints_subj128_9class",
                        help="Root directory for 9-class checkpoints")
    parser.add_argument("--ckpt_dir", type=str, default="",
                        help="If set, use this checkpoint directory directly")
    # 출력 위치
    parser.add_argument("--out_root", type=str, default="./samples_subj128_9class",
                        help="Root directory to save generated samples")

    args = parser.parse_args()

    sid = args.subject_id

    # ----------------- device -----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"[Sample-9class] Subject {sid:02d}, device: {device}")
    print("=" * 80)

    # ----------------- 데이터 & test split -----------------
    test_loader, test_idx, dataset = build_test_loader(
        data_root=args.data_root,
        subject_id=sid,
        img_size=args.img_size,
        batch_size=args.batch_size,
        split_seed=args.split_seed,
    )
    print(f"[Subj {sid:02d}] number of test trials: {len(test_idx)}")

    # ----------------- 모델 구성 -----------------
    # 샘플 하나 꺼내서 EEG 채널 수 확인
    sample_eeg, sample_img, sample_label = dataset[0]
    eeg_channels = sample_eeg.shape[0]
    print(f"[Subj {sid:02d}] EEG channels: {eeg_channels}, img size: {sample_img.shape[-2:]}")

    ch_mult = tuple(int(x) for x in args.ch_mult.split(","))

    model = EEGDiffusionModel128(
        img_size=args.img_size,
        img_channels=3,
        eeg_channels=eeg_channels,
        num_classes=9,
        num_timesteps=args.num_timesteps,
        base_channels=args.base_channels,
        ch_mult=ch_mult,
        n_res_blocks=args.n_res_blocks,
    ).to(device)
    model.eval()

    # ----------------- checkpoint 로드 -----------------
    if args.ckpt_dir:
        ckpt_dir = args.ckpt_dir
    else:
        ckpt_dir = find_latest_ckpt_dir(args.ckpt_root, sid, args.img_size)

    ckpt_path = find_ckpt_path(ckpt_dir, sid)
    print(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    # 학습 스크립트에서 torch.save({"model": model.state_dict(), ...})로 저장했다면:
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        # 혹시 그냥 state_dict만 저장한 경우도 대비
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)

    # ----------------- 출력 디렉터리 -----------------
    exp_name = os.path.basename(os.path.normpath(ckpt_dir))
    out_dir = os.path.join(args.out_root, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Samples will be saved under: {out_dir}")

    # ----------------- 샘플링 루프 -----------------
    model.to(device)
    model.eval()

    # sample_steps 결정
    if args.sample_steps is None or args.sample_steps <= 0:
        sample_steps = None  # 모델 내부에서 num_timesteps 사용
    else:
        sample_steps = args.sample_steps

    global_trial_counter = 0
    with torch.no_grad():
        for batch_idx, (eeg, img_gt, labels) in enumerate(test_loader):
            eeg = eeg.to(device)
            img_gt = img_gt.to(device)
            labels = labels.to(device)

            # diffusion sampling
            x_gen = model.sample(
                eeg=eeg,
                labels=labels,
                num_steps=sample_steps,
                guidance_scale=args.guidance_scale,
            )  # (B,3,H,W), [-1,1]

            # 시각화를 위해 [0,1]로 스케일링
            x_gen_vis = (x_gen.clamp(-1.0, 1.0) + 1.0) / 2.0
            img_gt_vis = (img_gt.clamp(-1.0, 1.0) + 1.0) / 2.0

            bsz = x_gen.size(0)
            for b in range(bsz):
                trial_global_idx = test_idx[global_trial_counter]
                label_int = int(labels[b].item())

                gen_name = (
                    f"subj{sid:02d}_trial{trial_global_idx:03d}_label{label_int}_gen.png"
                )
                gt_name = (
                    f"subj{sid:02d}_trial{trial_global_idx:03d}_label{label_int}_GT.png"
                )

                gen_path = os.path.join(out_dir, gen_name)
                gt_path = os.path.join(out_dir, gt_name)

                vutils.save_image(x_gen_vis[b], gen_path)
                vutils.save_image(img_gt_vis[b], gt_path)

                global_trial_counter += 1

            print(f"[Subj {sid:02d}] batch {batch_idx+1}/{len(test_loader)} "
                  f"→ saved {bsz} pairs")

    print(f"[Subj {sid:02d}] Done. Total saved test trials: {global_trial_counter}")


if __name__ == "__main__":
    main()
