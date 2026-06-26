# sample_subject_all_128.py
import argparse
import os

import torch
from torchvision.utils import save_image

from dataset_subject import EEGImageSubjectDataset
from model_128 import EEGDiffusionModel128


def find_latest_ckpt(run_root: str, subject_id: int):
    """
    train_subject_128.py 에서 만든 run 디렉토리 중
    해당 subject에 대한 최신 run 디렉토리를 찾고,
    그 안의 final checkpoint 경로를 리턴.
    """
    if not os.path.isdir(run_root):
        return None
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

    # ----------------------------------------------
    # 1) test split 구성 (train과 동일 split_ratio/seed 사용)
    # ----------------------------------------------
    test_ds = EEGImageSubjectDataset(
        data_root=args.data_root,
        subject_id=args.subject_id,
        split="test",
        split_ratio=args.split_ratio,  # 예: 0.9 → 마지막 10%가 test
        img_size=args.img_size,
        seed=args.seed,
    )

    num_trials = len(test_ds)
    print(f"[Subj {args.subject_id:02d}] number of test trials: {num_trials}")

    if num_trials == 0:
        raise RuntimeError("test split 이 비어 있습니다. split_ratio/subject_id 를 확인하세요.")

    # ----------------------------------------------
    # 2) checkpoint 불러오기
    # ----------------------------------------------
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

    # 모델 생성 (채널 수는 dataset에서 가져옴)
    eeg0, img0, label0 = test_ds[0]
    eeg_channels = eeg0.shape[0]

    model = EEGDiffusionModel128(
        img_size=args.img_size,
        eeg_channels=32,
        num_classes=9,
        num_timesteps=200,
        base_channels=64,  # ★ 체크포인트와 맞추는 핵심 포인트
        time_dim=256,
        cond_dim=256,
        eeg_hidden_dim=256,
        cond_scale=2.0,
        n_res_blocks=2,  # train에서 2로 썼다면 동일하게
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ----------------------------------------------
    # 3) 출력 디렉토리 설정
    # ----------------------------------------------
    os.makedirs(args.sample_root, exist_ok=True)
    run_name = os.path.basename(run_dir)
    out_dir = os.path.join(args.sample_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print("Samples will be saved under:", out_dir)

    # ----------------------------------------------
    # 4) 전체 test trial 루프 돌면서 생성
    # ----------------------------------------------
    max_n = num_trials if args.max_samples < 0 else min(args.max_samples, num_trials)
    print(f"[Subj {args.subject_id:02d}] generating {max_n} / {num_trials} test trials")

    with torch.no_grad():
        for idx in range(max_n):
            eeg, img_gt, label = test_ds[idx]

            eeg_b = eeg.unsqueeze(0).to(device)        # (1,C,T)
            img_gt_b = img_gt.unsqueeze(0)             # (1,3,H,W) in [0,1]
            label_b = torch.tensor([int(label)], device=device)

            # diffusion 샘플링 ([-1,1] 범위)
            samples = model.sample(eeg_b, label_b, num_steps=args.num_timesteps)

            # [-1,1] -> [0,1]
            samples = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0

            gen_path = os.path.join(
                out_dir,
                f"subj{args.subject_id:02d}_testIdx{idx:03d}_label{int(label)}.png",
            )
            gt_path = os.path.join(
                out_dir,
                f"subj{args.subject_id:02d}_testIdx{idx:03d}_label{int(label)}_GT.png",
            )

            save_image(samples, gen_path, nrow=1)
            save_image(img_gt_b, gt_path, nrow=1)

            if (idx + 1) % 10 == 0 or (idx + 1) == max_n:
                print(
                    f"[Subj {args.subject_id:02d}] "
                    f"generated {idx+1}/{max_n} (last label={int(label)})"
                )

    print(f"[Subj {args.subject_id:02d}] Done. Images saved in {out_dir}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./preproc_data")
    p.add_argument("--subject_id", type=int, default=1)
    p.add_argument("--split_ratio", type=float, default=0.9)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--num_timesteps", type=int, default=200)
    p.add_argument("--seed", type=int, default=2025)

    p.add_argument(
        "--ckpt_root",
        type=str,
        default="./checkpoints_subj128",
        help="train_subject_128.py 가 저장한 run 폴더 루트",
    )
    p.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="직접 final checkpoint 경로를 지정하고 싶을 때",
    )
    p.add_argument(
        "--sample_root",
        type=str,
        default="./samples_subj128",
        help="생성 이미지 저장 루트",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="생성 trial 수 제한. <0 이면 전체 test trial 사용",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
