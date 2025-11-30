import os
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MyEEGDataset(Dataset):
    def __init__(self, root, split_key='stim', split='train', transform=None):
        self.root = root
        self.split_key = split_key
        self.split = split
        self.transform = transform  # 이미지 변환
        self.data, self.labels = [], []

        # train/test subject split
        if split == 'train':
            subjects = [f"subj_{i:02d}.mat" for i in range(1, 10)]
        else:
            subjects = [f"subj_{i:02d}.mat" for i in range(10, 20)]

        for s in subjects:
            mat = sio.loadmat(os.path.join(root, s))
            X, y = mat['X'], mat['y'].squeeze()

            X = np.transpose(X, (2, 0, 1))  # (N, 32, 512)
            X = np.tile(X, (1, 4, 1)) # (N, 32, 512)

            self.data.append(X)
            self.labels.append(y)

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = torch.tensor(self.data[idx], dtype=torch.float32)  # EEG는 그냥 Tensor
        # eeg = eeg.repeat(4, 1)
        label = int(self.labels[idx])

        # 이미지 로드
        img_path = f"./images/{label:02d}.png"  # 예: 01_air.png → 01.png
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return {'eeg': eeg, 'image': img, 'label': label}
