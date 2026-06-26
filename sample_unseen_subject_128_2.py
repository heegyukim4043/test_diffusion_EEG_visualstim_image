# sample_unseen_subject_128.py
import argparse
import os

import torch
from torchvision.utils import save_image

from dataset_subject_2 import EEGImageSubjectDataset
from model_128_2 import EEGDiffusionModel128


def parse_subject_ids(s: str):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            a, b = int(a), int(b)
            out.extend(range(a, b + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


def find_latest_multi_ckpt(ckpt_root: str, train_subject_ids):
    if not os.path.isdir(ckpt_root):
        return None, None

    sid_str = "_".join(f"{s:02d}" for s in train_subject_ids)
    cand_dirs = [
        os.path.join(ckpt_root, d)
        for d in os.listdir(ckpt_root)
        if f"multi_s{sid_str}_128" in d
    ]
    if not cand_dirs:
        return None, None
    latest_dir = sorted(cand_dirs)[-1]
    ckpt_path = os.path.join(latest_dir, f"multi_s{sid_str}_final.pt")
    if not os.path.isfile(ckpt_path):
        return None, None
    return latest_dir, ckpt_path


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_subject_ids = parse_subject_ids(args.train_subject_ids)
    print(f"Using model trained on subjects: {train_subject_ids}")
    print(f"Generating for unseen subject: {args.test_subject_id:02d}")

    # 1) unseen subject 의 test split dataset
    test_ds = EEGImageSubjectDataset(
        data_root=args.data_root,
        subject_id=args.test_subject_id,
        split="test",
        split_ratio=args.split_ratio,
        img_size=args.img_size,
        seed=args.seed,
    )
    num_trials = len(test_ds)
    if num_trials == 0:
        raise RuntimeError("test split 이 비어 있습니다.")

    max_n = num_trials if args.max_samples < 0 else min(args.max_samples, num_trials)
    print(
        f"[Unseen Subj {args.test_subject_id:02d}] generating {max_n}/{num_trials} trials"
    )

    # 2) checkpoint 불러오기
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
        run_dir = os.path.dirname(ckpt_path)
    else:
        run_dir, ckpt_path = find_latest_multi_ckpt(args.ckpt_root, train_subject_ids)
        if ckpt_path is None:
            raise FileNotFoundError("multi-subject checkpoint 를 찾을 수 없습니다.")

    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    eeg0, img0, label0 = test_ds[0]
    eeg_channels = eeg0.shape[0]

    model = EEGDiffusionModel128(
        img_size=args.img_size,
        img_channels=3,
        eeg_channels=eeg_channels,
        num_classes=args.num_classes,
        num_timesteps=args.num_timesteps,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 3) 출력 디렉토리
    os.makedirs(args.sample_root, exist_ok=True)
    run_name = os.path.basename(run_dir)
    out_dir = os.path.join(
        args.sample_root,
        f"{run_name}_unseen_subj{args.test_subject_id:02d}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print("Samples will be saved under:", out_dir)

    # 4) 전체 test trial 에 대해 이미지 생성
    with torch.no_grad():
        for idx in range(max_n):
            eeg, img_gt, label = test_ds[idx]

            eeg_b = eeg.unsqueeze(0).to(device)    # (1,C_eeg,T)
            img_gt_b = img_gt.unsqueeze(0)         # (1,3,H,W) in [0,1]
            label_b = torch.tensor([int(label)], device=device, dtype=torch.long)

            samples = model.sample_guided(
                eeg_b,
                label_b,
                num_steps=args.num_timesteps,
                cfg_scale=args.cfg_scale,
            )
            samples = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0

            gen_path = os.path.join(
                out_dir,
                f"subj{args.test_subject_id:02d}_testIdx{idx:03d}_label{int(label)}.png",
            )
            gt_path = os.path.join(
                out_dir,
                f"subj{args.test_subject_id:02d}_testIdx{idx:03d}_label{int(label)}_GT.png",
            )

            save_image(samples, gen_path, nrow=1)
            save_image(img_gt_b, gt_path, nrow=1)

            if (idx + 1) % 10 == 0 or (idx + 1) == max_n:
                print(
                    f"[Unseen Subj {args.test_subject_id:02d}] "
                    f"generated {idx+1}/{max_n} (last label={int(label)})"
                )

    print(
        f"[Unseen Subj {args.test_subject_id:02d}] Done. Images saved in {out_dir}"
    )


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./preproc_data")

    p.add_argument(
        "--train_subject_ids",
        type=str,
        default="1-10",
        help="학습에 사용된 subject id들 (예: '1-10')",
    )
    p.add_argument("--test_subject_id", type=int, default=11)

    p.add_argument("--split_ratio", type=float, default=0.9)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--num_timesteps", type=int, default=200)
    p.add_argument("--num_classes", type=int, default=9)
    p.add_argument("--seed", type=int, default=2025)

    p.add_argument(
        "--ckpt_root",
        type=str,
        default="./checkpoints_multi128_2",
        help="train_multi_subject_128.py 출력 루트",
    )
    p.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="직접 checkpoint 지정시 사용",
    )
    p.add_argument(
        "--sample_root",
        type=str,
        default="./samples_unseen128",
        help="생성 이미지 저장 루트",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="<0 이면 전체 test trial 생성",
    )
    p.add_argument(
        "--cfg_scale",
        type=float,
        default=1.5,
        help="샘플링 시 조건(EEG+class+time) 강화 정도",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
