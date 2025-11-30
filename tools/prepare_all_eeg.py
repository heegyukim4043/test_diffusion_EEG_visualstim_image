# tools/prepare_all_eeg.py
# -*- coding: utf-8 -*-
"""
- preproc_data/subj_*.mat 를 모두 스캔
- 각 파일의 X:[N,C,T] 또는 [C,T,N]을 [N,C,T]로 통일
- 채널별 p05/p95 계산 → preproc_data/eeg_p05p95.pt
  (파일별 p05/p95를 샘플 수로 가중 평균하여 근사)
- 전체/자극/심상 split 생성 → preproc_data/splits.pt
"""
import os, glob, json
from pathlib import Path
import numpy as np
import torch, h5py

ROOT = Path(__file__).resolve().parents[1]
PDP  = ROOT / "preproc_data"
PDP.mkdir(exist_ok=True, parents=True)

def to_nct(x):
    x = np.asarray(x)
    if x.ndim != 3: raise ValueError(f"X must be 3D, got {x.shape}")
    C_like=(8,16,32,64,128); T_like=(128,256,512,1024,2048,4096)
    s0,s1,s2=x.shape
    if s0 in C_like and s1 in T_like:         # [C,T,N]
        x = np.transpose(x,(2,0,1))
    elif s1 in T_like and s2 in C_like:       # [N,T,C]
        x = np.transpose(x,(0,2,1))
    return x.astype('float32', copy=False)



def main():
    mats = sorted(glob.glob(str(PDP / "subj_*.mat")))
    assert mats, f"no subj_*.mat under {PDP}"
    print(f"[INFO] found {len(mats)} files")

    # p05/p95 (가중 평균 근사)
    sum_p05 = None; sum_p95 = None; sum_w = 0
    # 전역 인덱스 만들기(파일별 시작/길이 저장)
    starts=[]; lengths=[]; files=[]
    # task/label 전역 벡터
    all_task=[]; all_len=0

    for m in mats:
        with h5py.File(m, "r") as f:
            X = to_nct(f["X"])
            y = np.array(f["y"]).squeeze().astype(np.uint8)
            task = np.array(f["task"]).squeeze().astype(np.uint8) if "task" in f else None
        N,C,T = X.shape
        print(f"[{os.path.basename(m)}] X={X.shape}")
        # p05/p95
        p05 = np.percentile(X,  5, axis=(0,2))
        p95 = np.percentile(X, 95, axis=(0,2))
        w = N
        if sum_p05 is None:
            sum_p05 = p05*w; sum_p95 = p95*w
        else:
            sum_p05 += p05*w; sum_p95 += p95*w
        sum_w += w
        # index map
        starts.append(all_len); lengths.append(N); files.append(os.path.relpath(m, ROOT))
        # task vector
        if task is None:
            task = np.zeros(N, dtype=np.uint8)
        all_task.append(task)
        all_len += N

    p05 = (sum_p05/sum_w).astype("float32")
    p95 = (sum_p95/sum_w).astype("float32")
    torch.save({"p05": torch.from_numpy(p05), "p95": torch.from_numpy(p95)}, PDP / "eeg_p05p95.pt")
    print("[OK] saved", PDP / "eeg_p05p95.pt")

    # 전역 인덱스/스플릿 만들기
    task_vec = np.concatenate(all_task, axis=0) if len(all_task)>0 else np.zeros(all_len, np.uint8)
    rng = np.random.RandomState(42)
    idx_all = rng.permutation(all_len)

    def mk_split(indices):
        n = len(indices); ntr=int(n*0.7); nva=int(n*0.1)
        return {
            "train": torch.from_numpy(indices[:ntr].astype(np.int64)),
            "val":   torch.from_numpy(indices[ntr:ntr+nva].astype(np.int64)),
            "test":  torch.from_numpy(indices[ntr+nva:].astype(np.int64)),
        }

    splits = {"all": mk_split(idx_all)}
    stim_idx = np.where(task_vec==0)[0]
    imag_idx = np.where(task_vec==1)[0]
    if len(stim_idx)>0: splits["stim"]=mk_split(rng.permutation(stim_idx))
    if len(imag_idx)>0: splits["imag"]=mk_split(rng.permutation(imag_idx))

    meta = {
        "files": files,          # 프로젝트 루트 기준 상대경로
        "starts": starts,        # 각 파일의 글로발 시작 인덱스
        "lengths": lengths,      # 파일별 샘플 수
        "splits": {k:{s:v.tolist() for s,v in splits[k].items()} for k in splits.keys()},
    }
    torch.save(splits, PDP / "splits.pt")
    with open(PDP / "index.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[OK] saved", PDP / "splits.pt", "and index.json")

if __name__ == "__main__":
    main()
