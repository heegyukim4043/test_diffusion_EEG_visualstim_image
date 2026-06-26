# check_subject_01.py
import os
import numpy as np
from scipy.io import loadmat

root = "./preproc_data"
subj_id = 1

mat_path = os.path.join(root, f"subj_{subj_id:02d}.mat")
print("Loading:", mat_path)

data = loadmat(mat_path)
print("Keys in mat file:", data.keys())

X = data["X"]        # (32, 512, 540) = ch x time x trial
y = data["y"].squeeze()  # (540,)  -> label 1~9
ch_name = data["ch_name"]   # (1,32) cell array
fs = data["fs"].item()      # scalar

# 계속 같은 스크립트 안에서

# (32, 512, 540) -> (540, 32, 512)
X_trials_first = np.transpose(X, (2, 0, 1))
print("X_trials_first shape:", X_trials_first.shape)
# -> (540, 32, 512)

# 나중에 torch로 바꿀 때
import torch
eeg_all = torch.from_numpy(X_trials_first).float()  # (540, 32, 512)
labels_all = torch.from_numpy(y.astype(np.int64))   # (540,)

n_trials = eeg_all.shape[0]   # 540
train_ratio = 0.7
n_train = int(n_trials * train_ratio)  # 378

# 간단하게 앞 70%를 train, 뒤 30%를 test 로 사용
train_idx = np.arange(0, n_train)
test_idx = np.arange(n_train, n_trials)

eeg_train = eeg_all[train_idx]      # (378, 32, 512)
y_train = labels_all[train_idx]     # (378,)
eeg_test = eeg_all[test_idx]        # (162, 32, 512)
y_test = labels_all[test_idx]       # (162,)

print("Train trials:", eeg_train.shape[0])
print("Test trials:", eeg_test.shape[0])
