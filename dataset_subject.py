# dataset_subject.py
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy.io import loadmat


class EEGImageSubjectDataset(Dataset):
    """
    한 명의 subject에 대해
    - subj_XX.mat 안의 X, y
    - images/01.png ~ 09.png
    를 연결해서 (EEG, Image, Label)을 반환하는 Dataset
    """

    def __init__(
        self,
        data_root: str,
        subject_id: int,
        split: str = "train",      # "train" 또는 "test"
        split_ratio: float = 0.7,  # 7:3 비율
        img_size: int = 64,
        seed: int = 42,
    ):
        assert split in ["train", "test"]
        self.data_root = Path(data_root)
        self.subject_id = subject_id
        self.split = split
        self.split_ratio = split_ratio

        # ---------- MAT 파일 로드 ----------
        mat_path = self.data_root / f"subj_{subject_id:02d}.mat"
        mat = loadmat(mat_path)

        X = mat["X"].astype("float32")         # (32, 512, 540) = ch x time x trial
        y = mat["y"].squeeze().astype("int64") # (540,) = label 1~9

        # trial first로 바꾸기: (540, 32, 512)
        X = np.transpose(X, (2, 0, 1))

        n_trials = X.shape[0]   # 540
        indices = np.arange(n_trials)

        # 재현 가능하게 섞기
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

        n_train = int(n_trials * split_ratio)

        if split == "train":
            use_idx = indices[:n_train]
        else:  # "test"
            use_idx = indices[n_train:]

        # 필요한 trial만 저장
        self.X = X[use_idx]       # (N_split, 32, 512)
        self.y = y[use_idx]       # (N_split,)

        # 이미지 transform
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),   # [0,1] 범위
        ])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # EEG
        eeg = torch.from_numpy(self.X[idx])    # (32, 512)
        label = int(self.y[idx])               # 1~9

        # label에 해당하는 이미지 파일: 01.png ~ 09.png
        img_path = self.data_root / "images" / f"{label:02d}.png"
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)          # (3, img_size, img_size)

        return eeg, img, label
