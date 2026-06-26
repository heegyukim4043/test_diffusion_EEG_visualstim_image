# dataset_subject_segments.py
import os
from typing import List, Tuple

import numpy as np
import scipy.io as sio
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class EEGImageSubjectSegmentDataset(Dataset):
    """
    subj_XX.mat (X: ch x time x trial, y: trial labels)
    을 사용해서 time(512)을 num_segments개로 쪼개서
    각 segment(예: 32 x 128)를 하나의 샘플로 사용하는 Dataset.

    - split="train": trial 단위로 7:3 split 후 train trial들의 segment만 사용
    - split="test":  나머지 trial들의 segment만 사용
    """

    def __init__(
        self,
        data_root: str,
        subject_id: int,
        split: str = "train",
        split_ratio: float = 0.7,
        img_size: int = 64,
        seed: int = 2025,
        num_segments: int = 4,
    ):
        super().__init__()
        assert split in ["train", "test"]
        self.data_root = data_root
        self.subject_id = subject_id
        self.split = split
        self.split_ratio = split_ratio
        self.img_size = img_size
        self.seed = seed
        self.num_segments = num_segments

        # ---------- 1) MAT 파일 로드 ----------
        subj_name = f"subj_{subject_id:02d}.mat"
        mat_path = os.path.join(data_root, subj_name)
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"MAT file not found: {mat_path}")

        print(f"Loading: {mat_path}")
        mat = sio.loadmat(mat_path)

        X = mat["X"]  # (ch, time, trial)
        y = mat["y"].squeeze()  # (trial,)

        if X.ndim != 3:
            raise ValueError(f"X must be 3D (ch, time, trial), got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D (trial,), got shape {y.shape}")

        ch, time_len, num_trials = X.shape
        if time_len % num_segments != 0:
            raise ValueError(
                f"time_len({time_len}) must be divisible by num_segments({num_segments})"
            )

        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.ch = ch
        self.time_len = time_len
        self.num_trials = num_trials
        self.segment_len = time_len // num_segments

        print(
            f"Subject {subject_id:02d}: X shape={self.X.shape}, "
            f"num_trials={num_trials}, segment_len={self.segment_len}"
        )

        # ---------- 2) trial 단위 train/test split ----------
        rng = np.random.RandomState(seed)
        trial_indices = np.arange(num_trials)
        rng.shuffle(trial_indices)

        split_idx = int(num_trials * split_ratio)
        train_trials = trial_indices[:split_idx]
        test_trials = trial_indices[split_idx:]

        if split == "train":
            used_trials = train_trials
        else:
            used_trials = test_trials

        # ---------- 3) (trial_idx, segment_idx) 리스트 구성 ----------
        # segment_idx: 0,1,2,3 (num_segments=4일 때)
        self.samples: List[Tuple[int, int]] = []
        for t_idx in used_trials:
            for seg_idx in range(num_segments):
                self.samples.append((int(t_idx), int(seg_idx)))

        print(
            f"[{split}] using {len(used_trials)} trials -> "
            f"{len(self.samples)} segments (num_segments={num_segments})"
        )

        # ---------- 4) 이미지 transform ----------
        img_dir = os.path.join(data_root, "images")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image dir not found: {img_dir}")
        self.img_dir = img_dir

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        trial_idx, seg_idx = self.samples[idx]

        start = seg_idx * self.segment_len
        end = start + self.segment_len

        # EEG segment: (ch, segment_len)
        eeg_seg = self.X[:, start:end, trial_idx]  # (ch, segment_len)
        eeg_seg = torch.from_numpy(eeg_seg)  # float32

        # Label: trial label 그대로 사용
        label = int(self.y[trial_idx])  # 1~9

        # 이미지: label과 매칭되는 PNG
        img_name = f"{label:02d}.png"
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)  # (3, img_size, img_size)

        return eeg_seg, img, label

    # test 생성용에서 trial/segment 정보를 쓰고 싶을 때를 위해:
    def get_trial_segment_info(self, idx: int):
        """(trial_idx, seg_idx)를 반환 (샘플 이름 붙일 때 사용)"""
        return self.samples[idx]
