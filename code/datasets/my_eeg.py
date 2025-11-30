from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset


class MyEEGDataset(Dataset):
    def __init__(self, root, split_key='stim', split='train', transform=None, split_seed: int = 42):
        """EEG-image pair dataset that loads samples directly from ``.mat`` files.

        기존 코드에서는 ``images`` 폴더에서 PNG 이미지를 읽고, ``subj_XX.mat`` 파일의
        ``X``/``y`` 키만을 사용했습니다. 실제 데이터는 하나의 ``.mat`` 파일에 EEG와
        이미지 정보가 함께 들어있는 경우가 많아서 이를 자동으로 처리하도록 로더를
        확장했습니다.

        The loader now supports two layouts:

        1. Legacy subject-split files (``subj_XX.mat`` with ``X``/``y`` keys).
        2. A unified ``.mat`` file that contains EEG (``eeg``/``X``) and image
           information (``img`` array or ``p_name`` with image paths). Labels are
           read from ``y``/``label``/``class``/``obj``/``val`` in that order.
        """

        self.root = Path(root)
        self.split_key = split_key
        self.split = split
        self.transform = transform
        self.split_seed = split_seed

        self.data: List[np.ndarray] = []
        self.labels: List[int] = []
        self.images: List[Image.Image] = []

        # 1) Try legacy per-subject layout first for backward compatibility.
        subject_files = self._collect_subject_files()
        if subject_files:
            self._load_subject_files(subject_files)
        else:
            # 2) Fallback to a unified MAT file that bundles EEG + image info.
            mat_files = sorted(self.root.glob('*.mat'))
            if not mat_files:
                raise FileNotFoundError(f"No .mat files found under {self.root}")
            self._load_unified_file(mat_files[0])

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _collect_subject_files(self) -> List[Path]:
        if self.split == 'train':
            subjects = [self.root / f"subj_{i:02d}.mat" for i in range(1, 10)]
        else:
            subjects = [self.root / f"subj_{i:02d}.mat" for i in range(10, 20)]
        return [p for p in subjects if p.exists()]

    def _load_subject_files(self, subject_files: Iterable[Path]) -> None:
        for mat_path in subject_files:
            mat = sio.loadmat(mat_path)
            if 'X' not in mat or 'y' not in mat:
                continue
            X, y = mat['X'], mat['y'].squeeze()

            X = np.transpose(X, (2, 0, 1))  # (N, C, T)
            # Expand channels (e.g., 32 -> 128) to match the expected encoder
            if X.shape[1] < 128 and 128 % X.shape[1] == 0:
                repeat = 128 // X.shape[1]
                X = np.tile(X, (1, repeat, 1))

            self.data.append(X)
            self.labels.append(y)

        if not self.data:
            raise ValueError('No usable subject files were found.')

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        # Legacy layout uses external images; keep placeholder None for now.
        self.images = [None] * len(self.labels)

    def _load_unified_file(self, mat_path: Path) -> None:
        mat = sio.loadmat(mat_path)
        labels = self._extract_labels(mat)
        eeg = self._extract_eeg(mat, len(labels))
        images = self._extract_images(mat, len(labels))

        split_idx = self._split_indices(len(labels))
        if self.split == 'train':
            idx = split_idx[0]
        elif self.split == 'val':
            idx = split_idx[1]
        else:
            idx = split_idx[2]

        self.data = eeg[idx]
        self.labels = labels[idx]
        self.images = [images[i] for i in idx]

    def _split_indices(self, total: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.split_seed)
        order = np.arange(total)
        rng.shuffle(order)
        n_train = int(total * 0.7)
        n_val = int(total * 0.15)
        train_idx = order[:n_train]
        val_idx = order[n_train:n_train + n_val]
        test_idx = order[n_train + n_val:]
        return train_idx, val_idx, test_idx

    def _extract_labels(self, mat: dict) -> np.ndarray:
        for key in ['y', 'label', 'class', 'obj', 'val']:
            if key in mat:
                labels = np.asarray(mat[key]).squeeze()
                return labels.astype(int)
        raise KeyError('No label key (y/label/class/obj/val) found in MAT file.')

    def _extract_eeg(self, mat: dict, num_samples: int) -> np.ndarray:
        eeg_key = None
        for key in ['X', 'eeg']:
            if key in mat:
                eeg_key = key
                break
        if eeg_key is None:
            raise KeyError('No EEG key (X/eeg) found in MAT file.')

        eeg = np.asarray(mat[eeg_key])
        if eeg.ndim == 2:
            eeg = eeg[np.newaxis, ...]
        if eeg.shape[0] == num_samples:
            samples_first = eeg
        elif eeg.shape[-1] == num_samples:
            samples_first = np.transpose(eeg, (2, 0, 1))
        else:
            raise ValueError(f'EEG shape {eeg.shape} does not match sample count {num_samples}.')

        if samples_first.shape[1] < 128 and 128 % samples_first.shape[1] == 0:
            repeat = 128 // samples_first.shape[1]
            samples_first = np.tile(samples_first, (1, repeat, 1))

        return samples_first.astype(np.float32)

    def _extract_images(self, mat: dict, num_samples: int) -> List[Image.Image]:
        if 'img' in mat:
            raw = mat['img']
            if raw.ndim == 4:  # (N, H, W, C)
                imgs = [self._to_pil(raw[i]) for i in range(raw.shape[0])]
            else:  # cell array or list-like
                raw_list = np.atleast_1d(raw).squeeze()
                imgs = [self._to_pil(raw_list[i]) for i in range(raw_list.shape[0])]
        elif 'p_name' in mat:
            names = np.asarray(mat['p_name']).squeeze()
            imgs = [self._load_from_path(names[i]) for i in range(names.shape[0])]
        else:
            raise KeyError('No image key (img/p_name) found in MAT file.')

        if len(imgs) != num_samples:
            raise ValueError(f'Number of images ({len(imgs)}) does not match labels ({num_samples}).')
        return imgs

    def _to_pil(self, img_entry) -> Image.Image:
        arr = np.array(img_entry)
        if arr.ndim == 0:
            raise ValueError('Invalid image entry encountered in MAT file.')
        if arr.ndim == 1:
            arr = arr.reshape(int(np.sqrt(arr.size)), -1)
        if arr.ndim == 2:  # grayscale -> RGB
            arr = np.stack([arr] * 3, axis=-1)
        # Normalize dtype
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1) if arr.max() <= 1.0 else np.clip(arr, 0, 255)
            arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        return Image.fromarray(arr)

    def _load_from_path(self, path_entry) -> Image.Image:
        if isinstance(path_entry, (bytes, np.bytes_)):
            path_entry = path_entry.decode('utf-8')
        elif isinstance(path_entry, np.ndarray) and path_entry.dtype.kind in {'U', 'S', 'O'}:
            path_entry = str(path_entry.item())
        path = Path(path_entry)
        if not path.is_absolute():
            path = self.root / path
        return Image.open(path).convert('RGB')

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = torch.tensor(self.data[idx], dtype=torch.float32)
        label = int(self.labels[idx])

        img = self.images[idx]
        # For legacy subject layout we still need to read the image files
        if img is None:
            img_path = self.root / 'images' / f"{label:02d}.png"
            img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return {'eeg': eeg, 'image': img, 'label': label}
