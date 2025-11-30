# tools/prepare_my_eeg.py
import os, h5py, numpy as np, torch

ROOT = r"C:\Users\Biocomputing\Desktop\workspace_VIVS\DreamDiffusion-main\DreamDiffusion-main"
MAT  = os.path.join(ROOT, "preproc_data", "subj_01.mat")   # 여기에 .mat를 두세요
OUT_STD = os.path.join(ROOT, "preproc_data", "eeg_5_95_std.pth")
OUT_SPL = os.path.join(ROOT, "preproc_data", "block_splits_by_image_single.pth")

with h5py.File(MAT, "r") as f:
    X = np.array(f["X"])          # 기대: [N,C,T] 또는 [C,T,N]
    y = np.array(f["y"]).squeeze().astype(np.uint8)
    task = np.array(f["task"]).squeeze().astype(np.uint8)
    fs = float(np.array(f["fs"]))

# 모양 보정: [C,T,N]이면 [N,C,T]로 바꿈
if X.ndim == 3 and X.shape[0] in (16,32,64) and X.shape[2] > 10:
    X = np.transpose(X, (2,0,1)).astype("float32")
else:
    X = X.astype("float32")

N, C, T = X.shape
print(f"Loaded X: {X.shape}, y: {y.shape}, task: {task.shape}, fs={fs}")

# 채널별 5/95 퍼센타일 (train만으로 계산하는 게 이상적이지만 여기선 전체로 예시)
p05 = np.percentile(X,  5, axis=(0,2))
p95 = np.percentile(X, 95, axis=(0,2))
torch.save({"p05": torch.from_numpy(p05).float(),
            "p95": torch.from_numpy(p95).float()}, OUT_STD)
print("Saved:", OUT_STD)

# 재현가능 split (7:1:2)
rng = np.random.RandomState(42)
idx = rng.permutation(N)
ntr, nva = int(N*0.7), int(N*0.1)
splits = {
  "train": torch.from_numpy(idx[:ntr]).long(),
  "val":   torch.from_numpy(idx[ntr:ntr+nva]).long(),
  "test":  torch.from_numpy(idx[ntr+nva:]).long()
}
torch.save(splits, OUT_SPL)
print("Saved:", OUT_SPL)
