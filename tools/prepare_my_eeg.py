# tools/prepare_my_eeg.py
import os, h5py, numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # DreamDiffusion-main
MAT  = ROOT / "preproc_data" / "subj_01.mat"
OUT_STD = ROOT / "preproc_data" / "eeg_p05p95.pt"
OUT_SPL = ROOT / "preproc_data" / "splits.pt"

def main():
    assert MAT.exists(), f"not found: {MAT}"
    with h5py.File(MAT, "r") as f:
        X = np.array(f["X"])                   # [N,C,T] 또는 [C,T,N]
        y = np.array(f["y"]).squeeze().astype(np.uint8)
        task = np.array(f["task"]).squeeze().astype(np.uint8)  # 0=stim,1=imag
        fs = float(np.array(f["fs"]))
    # [C,T,N] -> [N,C,T]
    if X.shape[0] in (16,32,64) and X.shape[-1] > 10:
        X = np.transpose(X, (2,0,1)).astype("float32")
    else:
        X = X.astype("float32")
    N, C, T = X.shape
    print(f"Loaded X={X.shape}, y={y.shape}, task={task.shape}, fs={fs}")

    # 채널별 p05/p95 (여기선 전체로 계산; 실전은 train만으로 계산 권장)
    p05 = np.percentile(X,  5, axis=(0,2))
    p95 = np.percentile(X, 95, axis=(0,2))
    torch.save({"p05": torch.from_numpy(p05).float(),
                "p95": torch.from_numpy(p95).float()}, OUT_STD)
    print("Saved:", OUT_STD)

    # 재현가능 split (stim 전용/imag 전용도 따로 만듭니다)
    rng = np.random.RandomState(42)
    idx_all = rng.permutation(N)
    ntr, nva = int(N*0.7), int(N*0.1)

    # 전체
    splits = {
        "all": {
            "train": torch.from_numpy(idx_all[:ntr]).long(),
            "val":   torch.from_numpy(idx_all[ntr:ntr+nva]).long(),
            "test":  torch.from_numpy(idx_all[ntr+nva:]).long(),
        }
    }
    # stim 전용
    stim_idx = np.where(task==0)[0]
    rng.shuffle(stim_idx)
    ntr_s, nva_s = int(len(stim_idx)*0.7), int(len(stim_idx)*0.1)
    splits["stim"] = {
        "train": torch.from_numpy(stim_idx[:ntr_s]).long(),
        "val":   torch.from_numpy(stim_idx[ntr_s:ntr_s+nva_s]).long(),
        "test":  torch.from_numpy(stim_idx[ntr_s+nva_s:]).long(),
    }
    # imagery 전용
    imag_idx = np.where(task==1)[0]
    rng.shuffle(imag_idx)
    ntr_i, nva_i = int(len(imag_idx)*0.7), int(len(imag_idx)*0.1)
    splits["imag"] = {
        "train": torch.from_numpy(imag_idx[:ntr_i]).long(),
        "val":   torch.from_numpy(imag_idx[ntr_i:ntr_i+nva_i]).long(),
        "test":  torch.from_numpy(imag_idx[ntr_i+nva_i:]).long(),
    }

    torch.save(splits, OUT_SPL)
    print("Saved:", OUT_SPL)

if __name__ == "__main__":
    main()
