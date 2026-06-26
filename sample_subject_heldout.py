# sample_subject_heldout.py
import os
import glob
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset_subject import EEGImageSubjectDataset
from model import EEGDiffusionModel
from sample_subject import sample_ddim  # 기존 함수 재사용


class FilteredLabelDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, include_labels=None, exclude_labels=None):
        super().__init__()
        self.base = base_dataset
        self.indices = []

        for idx in range(len(base_dataset)):
            eeg, img, label = base_dataset[idx]
            lab = int(label)

            if include_labels is not None:
                if lab in include_labels:
                    self.indices.append(idx)
            elif exclude_labels is not None:
                if lab not in exclude_labels:
                    self.indices.append(idx)
            else:
                self.indices.append(idx)

        print(
            f"[FilteredLabelDataset] kept {len(self.indices)} / {len(base_dataset)} "
            f"samples (include={include_labels}, exclude={exclude_labels})"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    held_out = args.held_out_class
    print(f"[ZS] Generating for held-out class {held_out}")

    # ---------- test dataset에서 held-out class 샘플만 선택 ----------
    base_test_ds = EEGImageSubjectDataset(
        data_root=args.data_root,
        subject_id=args.subject_id,
        split="test",
        split_ratio=args.split_ratio,
        img_size=args.img_size,
        seed=args.seed,
    )

    test_ds = FilteredLabelDataset(
        base_test_ds,
        include_labels=[held_out],
        exclude_labels=None,
    )

    if len(test_ds) == 0:
        raise RuntimeError(
            f"No samples with label {held_out} found in test split."
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"[Subj {args.subject_id:02d}] #test samples with class {held_out} = {len(test_ds)}")

    # ---------- 모델 생성 & checkpoint 로드 ----------
    eeg_sample, img_sample, label_sample = test_ds[0]
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
        # zero-shot 학습용 디렉토리에서 subjXX_final.pt 검색
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

    # run_id → samples 저장 폴더 이름으로 사용
    ckpt_dir = os.path.dirname(ckpt_path)
    run_id = os.path.basename(ckpt_dir)  # 예: 20251207_XXXXXX_subj01_noC9

    sample_root = args.sample_dir
    sample_dir = os.path.join(sample_root, run_id + f"_holdoutC{held_out}")
    os.makedirs(sample_dir, exist_ok=True)
    print(f"[Subj {args.subject_id:02d}] Samples will be saved under: {sample_dir}")

    model.to(device)
    global_idx = 0

    for eeg, img_gt, label in test_loader:
        eeg = eeg.to(device)

        samples = sample_ddim(
            model,
            eeg,
            num_steps=args.num_timesteps,
            img_size=args.img_size,
            eta=0.0,
        )

        batch_size = eeg.size(0)
        for b in range(batch_size):
            idx = global_idx + b
            lab = int(label[b])  # 여기서는 lab == held_out 이어야 함

            base_name = (
                f"subj{args.subject_id:02d}_holdoutC{held_out}_"
                f"testIdx{idx:03d}_label{lab}"
            )

            gen_path = os.path.join(sample_dir, base_name + ".png")
            gt_path = os.path.join(sample_dir, base_name + "_GT.png")

            save_image(samples[b:b+1], gen_path, nrow=1)
            save_image(img_gt[b:b+1], gt_path, nrow=1)

        global_idx += batch_size

    print(
        f"[Subj {args.subject_id:02d}] Done. Generated {global_idx} images "
        f"for held-out class {held_out}."
    )


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

    p.add_argument("--out_dir", type=str, default="./checkpoints_subj_zs")
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--sample_dir", type=str, default="./samples_subj_zs")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--held_out_class", type=int, default=9)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
