# sample_subject_all_group_128.py
import os
import argparse
from datetime import datetime

import numpy as np
from scipy.io import loadmat
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from model_128_eegonly import EEGDiffusionModel128


# ---------------------------------------------------------
# Dataset: 한 subject + 특정 class range (예: 1~3) + test indices만
# ---------------------------------------------------------
class EEGImageDatasetGroup128(Dataset):
    def __init__(
        self,
        mat_path: str,
        img_root: str,
        indices,
        img_size: int = 128,
        cls_low: int = 1,
        cls_high: int = 3,
    ):
        super().__init__()
        self.mat_path = mat_path
        self.img_root = img_root
        self.indices = np.array(indices, dtype=np.int64)
        self.img_size = img_size
        self.cls_low = cls_low
        self.cls_high = cls_high

        mat = loadmat(mat_path)
        X = mat["X"]  # (ch, time, trial)
        y = mat["y"].squeeze()  # (trial,)

        # (trial, ch, time)
        self.eeg = torch.from_numpy(X).float().permute(2, 0, 1)
        self.labels = y.astype(np.int64)

        self.transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        trial_idx = int(self.indices[idx])

        eeg = self.eeg[trial_idx]  # (C, T)
        label_global = int(self.labels[trial_idx])  # 1~9
        label_local = label_global - self.cls_low   # (cls_low..cls_high)->0,1,2

        img_path = os.path.join(self.img_root,
                                f"{label_global:02d}.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)  # [-1,1] 범위

        # eeg, GT-img, 로컬/글로벌 라벨, 원 trial index
        return eeg, img, label_local, label_global, trial_idx


# ---------------------------------------------------------
# Helper: 시드 고정 (원하면 재현성 위해 사용 가능)
# ---------------------------------------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# train과 동일한 방식으로 그룹별 test index 복원
#  - train_subject_128_group3.py에서 사용한 로직과 동일
# ---------------------------------------------------------
def get_group_test_indices(
    mat_path: str,
    subject_id: int,
    group_idx: int,
    cls_low: int,
    cls_high: int,
    seed: int,
):
    """
    group_idx: 0,1,2  (→ Group1,2,3)
    cls_low, cls_high: (1,3), (4,6), (7,9)
    seed: train에서 사용한 args.seed와 동일해야 split이 동일하게 나옵니다.
    """
    mat = loadmat(mat_path)
    y = mat["y"].squeeze().astype(np.int64)  # (trial,)

    mask_group = (y >= cls_low) & (y <= cls_high)
    all_indices = np.where(mask_group)[0]
    n_total = len(all_indices)

    if n_total == 0:
        return np.array([], dtype=np.int64)

    # train_subject_128_group3.py와 완전히 동일한 seed 설계
    rng = np.random.RandomState(seed + subject_id * 10 + group_idx)
    rng.shuffle(all_indices)

    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    train_idx = all_indices[:n_train]
    val_idx = all_indices[n_train:n_train + n_val]
    test_idx = all_indices[n_train + n_val:]

    return test_idx


# ---------------------------------------------------------
# subject + group에 해당하는 최신 checkpoint 디렉토리 자동 탐색
# ---------------------------------------------------------
def find_latest_group_ckpt_dir(ckpt_root, subject_id, group_idx,
                               cls_low, cls_high, img_size):
    """
    ckpt_root 아래에서
    '..._subj07_g1_cls1-3_128' 형태 중 가장 최근(문자열 정렬 기준 마지막)을 선택
    """
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

    # 디렉토리명 앞에 timestamp가 있어서 문자열 정렬==시간순 정렬로 사용
    cand_dirs.sort()
    return cand_dirs[-1]


# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subj_str = f"{args.subject_id:02d}"

    # 그룹 매핑: group_id 1,2,3 → (1~3), (4~6), (7~9)
    if args.group_id == 1:
        cls_low, cls_high = 1, 3
    elif args.group_id == 2:
        cls_low, cls_high = 4, 6
    elif args.group_id == 3:
        cls_low, cls_high = 7, 9
    else:
        raise ValueError("group_id는 1, 2, 3 중 하나여야 합니다.")

    print("=" * 80)
    print(
        f"[Sample-Group] Subject {subj_str}, Group {args.group_id} "
        f"(classes {cls_low}~{cls_high}), device: {device}"
    )
    print("=" * 80)

    # -----------------------------------------------------
    # 1) test 인덱스 복원
    # -----------------------------------------------------
    mat_path = os.path.join(args.data_root, f"subj_{subj_str}.mat")
    img_root = os.path.join(args.data_root, "images")

    set_seed(args.seed)
    test_idx = get_group_test_indices(
        mat_path,
        subject_id=args.subject_id,
        group_idx=args.group_id - 1,
        cls_low=cls_low,
        cls_high=cls_high,
        seed=args.seed,
    )
    if len(test_idx) == 0:
        print(
            f"[Subj {subj_str}][Group {cls_low}-{cls_high}] "
            f"No test trials found. 종료."
        )
        return

    print(
        f"[Subj {subj_str}][Group {cls_low}-{cls_high}] "
        f"number of test trials: {len(test_idx)}"
    )

    # -----------------------------------------------------
    # 2) Dataset/DataLoader 구성
    # -----------------------------------------------------
    test_ds = EEGImageDatasetGroup128(
        mat_path=mat_path,
        img_root=img_root,
        indices=test_idx,
        img_size=args.img_size,
        cls_low=cls_low,
        cls_high=cls_high,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # -----------------------------------------------------
    # 3) checkpoint 디렉토리 및 ckpt 파일 탐색
    # -----------------------------------------------------
    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None:
        ckpt_dir = find_latest_group_ckpt_dir(
            args.ckpt_root,
            args.subject_id,
            args.group_id - 1,
            cls_low,
            cls_high,
            args.img_size,
        )

    if ckpt_dir is None or (not os.path.isdir(ckpt_dir)):
        print(
            f"[Subj {subj_str}][Group {cls_low}-{cls_high}] "
            f"Checkpoint dir not found. "
            f"ckpt_root={args.ckpt_root}"
        )
        return

    # best가 있으면 우선 사용, 없으면 final 사용
    best_path = os.path.join(
        ckpt_dir, f"subj{subj_str}_g{args.group_id}_best.pt"
    )
    final_path = os.path.join(
        ckpt_dir, f"subj{subj_str}_g{args.group_id}_final.pt"
    )

    if os.path.isfile(best_path):
        ckpt_path = best_path
    elif os.path.isfile(final_path):
        ckpt_path = final_path
    else:
        print(
            f"[Subj {subj_str}][Group {cls_low}-{cls_high}] "
            f"No best/final checkpoint found in {ckpt_dir}"
        )
        return

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})

    # -----------------------------------------------------
    # 4) 모델 구성 (config 있으면 맞춰서, 없으면 args로)
    # -----------------------------------------------------
    img_size = cfg.get("img_size", args.img_size)
    base_channels = cfg.get("base_channels", args.base_channels)
    num_timesteps = cfg.get("num_timesteps", args.num_timesteps)
    n_res_blocks = cfg.get("n_res_blocks", args.n_res_blocks)

    model = EEGDiffusionModel128(
        img_size=img_size,
        img_channels=3,
        eeg_channels=32,
        num_classes=3,             # 그룹 내 클래스 3개 (0~2)
        num_timesteps=num_timesteps,
        base_channels=base_channels,
        time_dim=256,
        cond_dim=256,
        eeg_hidden_dim=256,
        cond_scale=2.0,
        n_res_blocks=n_res_blocks,
    ).to(device)

    state_dict = ckpt.get("ema", ckpt["model"])
    model.load_state_dict(state_dict)
    model.eval()

    # -----------------------------------------------------
    # 5) 저장 폴더 설정
    # -----------------------------------------------------
    os.makedirs(args.samples_root, exist_ok=True)

    # ckpt_dir 이름을 그대로 samples 아래에 사용하면 관리가 편함
    ckpt_basename = os.path.basename(ckpt_dir.rstrip("/\\"))
    samples_dir = os.path.join(args.samples_root, ckpt_basename)
    os.makedirs(samples_dir, exist_ok=True)

    print(
        f"Samples will be saved under: {samples_dir}"
    )

    # -----------------------------------------------------
    # 6) test 전체에 대해 샘플 생성
    # -----------------------------------------------------
    # T.ToPILImage는 [-1,1] → [0,1] 범위로 들어가야 하므로 역정규화 필요
    def denorm_img(x):
        # x: (B,3,H,W), [-1,1] → [0,1]
        return (x.clamp(-1, 1) + 1.0) * 0.5

    to_pil = T.ToPILImage()

    with torch.no_grad():
        global_sample_idx = 0

        for batch_idx, (eeg, img_gt, label_local, label_global, trial_idx) in enumerate(test_loader):
            eeg = eeg.to(device)          # (B,C,T)
            img_gt = img_gt.to(device)    # (B,3,H,W)
            label_local = label_local.to(device)  # (B,)

            b = eeg.size(0)
            labels = label_local  # 로컬 라벨(0~2)을 그대로 사용

            # diffusion sampling
            x_gen = model.sample_eeg_only(
                eeg=eeg,
                #num_steps=args.sample_steps,
                num_steps=args.sample_steps,
                guidance_scale=args.guidance_scale,
            )  # (B,3,H,W), [-1,1]

            x_gen_denorm = denorm_img(x_gen)
            img_gt_denorm = denorm_img(img_gt)

            for i in range(b):
                g_label = int(label_global[i].item())
                t_idx = int(trial_idx[i].item())

                gen_pil = to_pil(x_gen_denorm[i].cpu())
                gt_pil = to_pil(img_gt_denorm[i].cpu())

                gen_name = (
                    f"subj{subj_str}_g{args.group_id}_trial{t_idx:03d}_"
                    f"label{g_label}_GEN.png"
                )
                gt_name = (
                    f"subj{subj_str}_g{args.group_id}_trial{t_idx:03d}_"
                    f"label{g_label}_GT.png"
                )

                gen_path = os.path.join(samples_dir, gen_name)
                gt_path = os.path.join(samples_dir, gt_name)

                gen_pil.save(gen_path)
                gt_pil.save(gt_path)

                global_sample_idx += 1

                if global_sample_idx % 10 == 0:
                    print(
                        f"[Subj {subj_str}][Group {cls_low}-{cls_high}] "
                        f"Saved {global_sample_idx} samples so far..."
                    )

    print(
        f"[Subj {subj_str}][Group {cls_low}-{cls_high}] "
        f"Done. Total saved samples: {global_sample_idx * 2} "
        f"(generated + GT)."
    )


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate images for ALL test trials, "
            "for a given subject & class group (1~3, 4~6, 7~9), 128x128."
        )
    )
    parser.add_argument("--data_root", type=str, default="./preproc_data")
    parser.add_argument("--subject_id", type=int, required=True,
                        help="예: 7")
    parser.add_argument("--group_id", type=int, required=True,
                        help="1: classes 1~3, 2: 4~6, 3: 7~9")
    parser.add_argument("--img_size", type=int, default=128)

    parser.add_argument("--ckpt_root", type=str,
                        default="./checkpoints_subj128_group",
                        help="train_subject_128_group3.py에서 쓴 ckpt_root")
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="특정 ckpt 디렉토리를 직접 지정하고 싶을 때 사용")

    parser.add_argument("--samples_root", type=str,
                        default="./samples_subj128_group_eegonly")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--num_timesteps", type=int, default=2000)
    parser.add_argument("--n_res_blocks", type=int, default=2)
    parser.add_argument("--sample_steps", type=int, default=200,
                        help="sampling 시 사용할 diffusion step 수 (<= num_timesteps)")
    parser.add_argument("--guidance_scale", type=float, default=2.5,
                        help="conditioning scale used at sampling")
    parser.add_argument("--seed", type=int, default=42,
                        help="train_subject_128_group3에서 사용한 seed와 동일해야 "
                             "train/val/test split이 동일하게 복원됩니다.")

    args = parser.parse_args()
    main(args)
