# test_dataset_128.py
import torch
from torch.utils.data import DataLoader

from dataset_subject import EEGImageSubjectDataset


if __name__ == "__main__":
    data_root = "./preproc_data"
    subject_id = 1
    img_size = 128

    ds = EEGImageSubjectDataset(
        data_root=data_root,
        subject_id=subject_id,
        split="train",
        split_ratio=0.9,
        img_size=img_size,
        seed=2025,
    )

    print(f"Subject {subject_id:02d} train+val len:", len(ds))
    eeg, img, label = ds[0]
    print("Single sample EEG shape:", eeg.shape)   # (C,T)
    print("Single sample IMG shape:", img.shape)   # (3,H,W) in [0,1]
    print("Single sample label:", label)

    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    eeg_b, img_b, label_b = next(iter(loader))
    print("Batch EEG shape:", eeg_b.shape)
    print("Batch IMG shape:", img_b.shape)
    print("Batch labels:", label_b)
