
import json, torch, numpy as np
import h5py
from pathlib import Path

root = Path(".")
with h5py.File(root/"preproc_data"/"subj_01.mat","r") as f:
    X = np.array(f["X"])
# print("raw subj_01 X shape:", X.shape)

import code.datasets.my_eeg as my
ds = my.MyEEGDataset(root='.', split_key='stim', split='train')
x = ds[0]["eeg"]; img = ds[0]["image"]; y = ds[0]["label"]
print("sample eeg:", tuple(x.shape), "image:", tuple(img.shape), "label:", int(y))
PY
