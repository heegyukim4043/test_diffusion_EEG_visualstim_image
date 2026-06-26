# test_dataset_subject01.py
from torch.utils.data import DataLoader
from dataset_subject import EEGImageSubjectDataset

data_root = "./preproc_data"
subject_id = 1

train_ds = EEGImageSubjectDataset(
    data_root,
    subject_id=subject_id,
    split="train",
    split_ratio=0.7,
    img_size=64,
    seed=42,
)

test_ds = EEGImageSubjectDataset(
    data_root,
    subject_id=subject_id,
    split="test",
    split_ratio=0.7,
    img_size=64,
    seed=42,
)

print("Train len:", len(train_ds))
print("Test  len:", len(test_ds))

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

eeg_batch, img_batch, label_batch = next(iter(train_loader))

print("EEG batch shape :", eeg_batch.shape)  # 예상: (4, 32, 512)
print("IMG batch shape :", img_batch.shape)  # 예상: (4, 3, 64, 64)
print("Label batch :", label_batch)
