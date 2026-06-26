# sample_unseen_subject_128.py
import argparse
import os
from typing import List, Tuple

import torch
from torchvision.utils import save_image

from dataset_subject import EEGImageSubjectDataset
from model_128 import EEGDiffusionModel128


def parse_subject_ids(s: str) -> List[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            a, b = int(a), int(b)
            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(tok))
    out = sorted(set(out))
    return out


def find_latest_multi_ckpt(
    ckpt_root: str, train_subject_ids: List[int]
) -> Tuple[str, str]:
    """
    train_multi_subject_128.py 에서 저장한 run 디렉토리 중
    주어진 train_subject_ids 에 해당하는 가장 최신 run 을 찾는다.
    """
    if not os.path.isdir(ckpt_root):
        return None, None

    sid_str = "_".join(f"{s:02d}" for s in train_subject_ids)
    cand_dirs = []
    for d in os.listdir(ckpt_root):
        if f"multi_s{sid_str}_128" in d:
            cand_dirs.append(os.path.join(ckpt_root, d))
    if not cand_dirs:
        return None, None

    latest_dir = sorted(cand_dirs)[-1]
    ckpt_path = os.path.join(latest_dir, f"multi_s{sid_str}_final.pt")
    if os.path.isfile(ckpt_path):
        return latest_dir, ckpt_path
    return None, None


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_subject_ids = parse_subject_ids(args.train_subject_ids)
    print(f"Using multi-subject model trained on: {train_subject_ids}")
    print(f"Generating for unseen subject: {args.test_subject_id}")

    # --------------------------------------------------
    # 1) test subject (예: 11번) 의 test split dataset
    # --------------------------------------------------
    test_ds = EEGImageSubjectDataset(
        data_root=args.data_root,
        subject_id=args.test_subject_id,
        split="test",
        split_ratio=args.split_ratio,  # e.g. 0.9 → 마지막 10% 가 test
        img_size=args.img_size,
        seed=args.seed,
    )
    num_trials = len(test_ds)
    if num_trials == 0:
        raise RuntimeError("test split 이 비어 있습니다. split_ratio / subject_id 확인 필요.")

    print(
        f"[Unseen Subj {args.test_subject_id:02d}] number of test trials: {num_trials}"
    )

    max_n = num_trials if args.max_samples < 0 else min(args.max_samples, num_trials)
    print(
        f"[Unseen Subj {args.test_subject_id:02d}] generating {max_n} / {num_trials} test trials"
    )

    # --------------------------------------------------
    # 2) checkpoint 로 multi-subject 모델 로드
    # --------------------------------------------------
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
        run_dir = os.path.dirname(ckpt_path)
    else:
        run_dir, ckpt_path = find_latest_multi_ckpt(
            args.ckpt_root, train_subject_ids
        )
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

    # --------------------------------------------------
    # 3) 출력 디렉토리
    # --------------------------------------------------
    os.makedirs(args.sample_root, exist_ok=True)
    run_name = os.path.basename(run_dir)
    out_dir = os.path.join(
        args.sample_root,
        f"{run_name}_unseen_subj{args.test_subject_id:02d}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print("Samples will be saved under:", out_dir)

    # --------------------------------------------------
    # 4) unseen subject test trial 전체에 대해 생성
    # --------------------------------------------------
    with torch.no_grad():
        for idx in range(max_n):
            eeg, img_gt, label = test_ds[idx]
            eeg_b = eeg.unsqueeze(0).to(device)        # (1,C,T)
            img_gt_b = img_gt.unsqueeze(0)             # (1,3,H,W) in [0,1]
            label_b = torch.tensor([int(label)], device=device)

            samples = model.sample(
                eeg_b, label_b, num_steps=args.num_timesteps
            )
            samples = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0  # [0,1]

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

    # multi-subject 학습에 사용된 subject id (기본: 1~10)
    p.add_argument(
        "--train_subject_ids",
        type=str,
        default="1-10",
        help="예: '1-10' 또는 '1,2,3,4,5,6,7,8,9,10'",
    )

    # unseen test subject (기본: 11)
    p.add_argument("--test_subject_id", type=int, default=11)

    p.add_argument("--split_ratio", type=float, default=0.9)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--num_timesteps", type=int, default=200)
    p.add_argument("--num_classes", type=int, default=9)
    p.add_argument("--seed", type=int, default=2025)

    p.add_argument(
        "--ckpt_root",
        type=str,
        default="./checkpoints_multi128",
        help="train_multi_subject_128.py 가 저장한 루트",
    )
    p.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="직접 final checkpoint 경로 지정시 사용",
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
        help="생성할 test trial 수 제한 (<0 이면 전체)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
