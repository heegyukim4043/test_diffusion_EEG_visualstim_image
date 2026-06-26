# test_train_step_128.py
import torch
from torch.utils.data import DataLoader

from dataset_subject import EEGImageSubjectDataset
from model_128 import EEGDiffusionModel128


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    eeg0, img0, label0 = ds[0]
    eeg_channels = eeg0.shape[0]

    model = EEGDiffusionModel128(
        img_size=img_size,
        img_channels=3,
        eeg_channels=eeg_channels,
        num_classes=9,
        num_timesteps=200,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    eeg_b, img_b, label_b = next(iter(loader))
    eeg_b = eeg_b.to(device)
    img_b = img_b.to(device) * 2.0 - 1.0
    label_b = label_b.to(device)

    b = img_b.size(0)
    t = torch.randint(0, 200, (b,), device=device, dtype=torch.long)

    loss = model.p_losses(img_b, eeg_b, label_b, t)
    print("Initial loss:", float(loss.item()))

    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

    print("One train step finished without error.")
