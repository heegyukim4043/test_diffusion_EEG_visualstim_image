# dataset_subject.py
import os
from typing import List, Tuple

import numpy as np
from scipy.io import loadmat
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class EEGImageSubjectDataset(Dataset):
    """
    하나의 subject (subj_XX.mat)에서 EEG trial과 대응하는 이미지를 불러오는 Dataset.

    디렉토리 구조 예:
      data_root/
        subj_01.mat
        subj_02.mat
        ...
        images/
          01.png
          02.png
          ...
          09.png

    .mat 구조:
      X: (ch, time, trial)  e.g. (32, 512, 540)
      y: (trial,)           label 1~9
    """

    def __init__(
        self,
        data_root: str,
        subject_id: int,
        split: str = "train",
        split_ratio: float = 0.9,
        img_size: int = 128,
        seed: int = 2025,
    ):
        super().__init__()
        assert split in ("train", "test")
        assert 0.0 < split_ratio < 1.0

        self.data_root = data_root
        self.subject_id = subject_id
        self.split = split
        self.split_ratio = split_ratio
        self.img_size = img_size

        mat_path = os.path.join(
            data_root, f"subj_{subject_id:02d}.mat"
        )
        if not os.path.isfile(mat_path):
            raise FileNotFoundError(f"Mat file not found: {mat_path}")

        mat = loadmat(mat_path, squeeze_me=True)
        X = mat["X"]  # (ch, time, trial)
        y = mat["y"]

        # Ensure shapes
        if X.ndim != 3:
            raise ValueError(f"Expected X ndim=3, got {X.ndim} in {mat_path}")
        ch, time_len, n_trial = X.shape

        # (trial, ch, time)
        X = np.transpose(X, (2, 0, 1)).astype(np.float32)
        y = np.array(y).reshape(-1).astype(np.int64)
        if y.shape[0] != n_trial:
            raise ValueError(f"y length {y.shape[0]} != n_trial {n_trial}")

        # Channel-wise z-score normalisation (전체 trial/time 기준)
        mean = X.mean(axis=(0, 2), keepdims=True)
        std = X.std(axis=(0, 2), keepdims=True) + 1e-6
        X = (X - mean) / std

        # PyTorch tensor
        self.eeg_all = torch.from_numpy(X)          # (N, C, T)
        self.labels_all = torch.from_numpy(y) - 1   # 0~8

        # trial index split (고정된 순서 기준)
        torch.manual_seed(seed)
        n_total = n_trial
        n_train = int(n_total * split_ratio)
        idx_all = torch.arange(n_total)

        if split == "train":
            self.indices = idx_all[:n_train]
        else:
            self.indices = idx_all[n_train:]
        self.indices = self.indices.tolist()

        # 이미지 로딩 transform
        self.img_root = os.path.join(data_root, "images")
        if not os.path.isdir(self.img_root):
            raise FileNotFoundError(f"Image folder not found: {self.img_root}")

        self.img_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),  # [0,1]
            ]
        )

        # 클래스별 원본 이미지 (01.png ~ 09.png) 미리 캐싱
        self.class_images = {}
        for c in range(9):
            fname = f"{c+1:02d}.png"
            path = os.path.join(self.img_root, fname)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Class image not found: {path}")
            self.class_images[c] = path

        print(
            f"[Subj {subject_id:02d}] split={split}, "
            f"indices={len(self.indices)}, EEG shape={self.eeg_all.shape}"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        trial_idx = self.indices[index]

        eeg = self.eeg_all[trial_idx]           # (C, T)
        label = int(self.labels_all[trial_idx]) # 0~8

        # 이미지 로딩 및 리사이즈
        img_path = self.class_images[label]
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)  # (3, H, W) in [0,1]

        return eeg, img, label


class MultiSubjectEEGImageDataset(Dataset):
    """
    여러 subject를 한 Dataset으로 합치는 래퍼.
    내부적으로 EEGImageSubjectDataset(subject_id=각각)를 들고 있고,
    (subject_idx, local_idx) 를 전역 index로 매핑한다.
    """

    def __init__(
        self,
        data_root: str,
        subject_ids: List[int],
        split: str = "train",
        split_ratio: float = 0.9,
        img_size: int = 128,
        seed: int = 2025,
    ):
        super().__init__()
        self.subject_ids = sorted(list(subject_ids))
        self.datasets: List[EEGImageSubjectDataset] = []
        self.index_map: List[Tuple[int, int]] = []

        for si, sid in enumerate(self.subject_ids):
            ds = EEGImageSubjectDataset(
                data_root=data_root,
                subject_id=sid,
                split=split,
                split_ratio=split_ratio,
                img_size=img_size,
                seed=seed,
            )
            self.datasets.append(ds)
            for local_idx in range(len(ds)):
                self.index_map.append((si, local_idx))

        total_len = len(self.index_map)
        print(
            f"[MultiSubjDataset] subjects={self.subject_ids}, "
            f"split={split}, total_len={total_len}"
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int):
        subj_idx, local_idx = self.index_map[index]
        return self.datasets[subj_idx][local_idx]
