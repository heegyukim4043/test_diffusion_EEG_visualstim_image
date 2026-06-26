# multi_subject_dataset.py
"""
여러 명의 subject를 하나의 Dataset으로 묶어서 사용하는 래퍼.

기존 dataset_subject.EEGImageSubjectDataset 를 그대로 재사용하고,
각 subject의 (split='train' 또는 'test') 데이터들을 단순히 이어붙여준다.
"""
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from dataset_subject import EEGImageSubjectDataset


class MultiSubjectEEGImageDataset(Dataset):
    """
    여러 subject를 합쳐서 하나의 dataset처럼 사용하는 클래스.

    - 내부적으로 subject별로 EEGImageSubjectDataset 인스턴스를 만들고,
      전체 인덱스를 (subj_dataset_idx, sample_idx) 로 매핑해서 관리한다.
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
        assert split in ("train", "test"), "split 은 'train' 또는 'test' 만 허용합니다."
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
            # 이 subject 의 각 샘플을 전역 인덱스에 매핑
            for local_idx in range(len(ds)):
                self.index_map.append((si, local_idx))

        print(
            f"[MultiSubject] subjects={self.subject_ids}, "
            f"split={split}, total_len={len(self.index_map)}"
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        subj_idx, local_idx = self.index_map[idx]
        return self.datasets[subj_idx][local_idx]
