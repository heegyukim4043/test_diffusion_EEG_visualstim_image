# sample_subject_128.py
import argparse
import os

import torch
from torchvision.utils import save_image

from dataset_subject import EEGImageSubjectDataset
from model_128 import EEGDiffusionModel128


def find_latest_ckpt(run_root: str, subject_id: int):
    if not os.path.isdir(run_root):
        return None
    target = f"subj{subject_id:02d}_128"
    cand_dirs = []
    for d in os.listdir(run_root):
        if f"subj{subject_id:02d}_128" in d:
            cand_dirs.append(os.path.join(run_root, d))
    if not cand_dirs:
        return None
    latest_dir = sorted(cand_dirs)[-1]
    ckpt_path = os.path.join(latest_dir, f"subj{subject_id:02d}_final.pt")
    if os.path.isfile(ckpt_path):
        return latest_dir, ckpt_path
    return None


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # test split (10%)
    test_ds = EEGImageSubjectDataset(
        data_root=args.data_root,
        subject_id=args.subject_id,
        split="test",
        split_ratio=args.split_ratio,
        img_size=args.img_size,
        seed=args.seed,
    )

    print(f"[Subj {args.subject_id:02d}] test len:", len(test_ds))
    if args.trial_index >= len(test_ds):
        raise IndexError("trial_index가 test 데이터 길이를 초과했습니다.")

    eeg, img_gt, label = test_ds[args.trial_index]
    print(
        f"Sampling for subject {args.subject_id:02d}, "
        f"trial {args.trial_index}, label = {int(label)}"
    )

    eeg = eeg.unsqueeze(0).to(device)  # (1,C,T)
    img_gt = img_gt.unsqueeze(0)  # (1,3,H,W) in [0,1]
    label_t = torch.tensor([int(label)], device=device)

    # checkpoint 찾기
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
        run_dir = os.path.dirname(ckpt_path)
    else:
        found = find_latest_ckpt(args.ckpt_root, args.subject_id)
        if found is None:
            raise FileNotFoundError("checkpoint를 찾을 수 없습니다.")
        run_dir, ckpt_path = found

    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    eeg_channels = eeg.shape[1]
    model = EEGDiffusionModel128(
        img_size=args.img_size,
        img_channels=3,
        eeg_channels=eeg_channels,
        num_classes=9,
        num_timesteps=args.num_timesteps,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    os.makedirs(args.sample_root, exist_ok=True)
    run_name = os.path.basename(run_dir)
    out_dir = os.path.join(args.sample_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print("Samples will be saved under:", out_dir)

    with torch.no_grad():
        samples = model.sample(eeg, label_t, num_steps=args.num_timesteps)

    # [-1,1] -> [0,1]
    samples = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0

    gen_path = os.path.join(
        out_dir,
        f"subj{args.subject_id:02d}_trial{args.trial_index:03d}_label{int(label)}.png",
    )
    gt_path = os.path.join(
        out_dir,
        f"subj{args.subject_id:02d}_trial{args.trial_index:03d}_label{int(label)}_GT.png",
    )

    save_image(samples, gen_path, nrow=1)
    save_image(img_gt, gt_path, nrow=1)

    print("Saved generated image to:", gen_path)
    print("Saved ground-truth image to:", gt_path)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./preproc_data")
    p.add_argument("--subject_id", type=int, default=1)
    p.add_argument("--trial_index", type=int, default=0)
    p.add_argument("--split_ratio", type=float, default=0.9)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--num_timesteps", type=int, default=200)
    p.add_argument("--seed", type=int, default=2025)

    p.add_argument(
        "--ckpt_root",
        type=str,
        default="./checkpoints_subj128",
        help="자동으로 최신 checkpoint를 찾을 root 디렉토리",
    )
    p.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="직접 지정하고 싶으면 final pt 경로 입력",
    )
    p.add_argument(
        "--sample_root",
        type=str,
        default="./samples_subj128",
        help="생성 이미지 저장 루트",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
