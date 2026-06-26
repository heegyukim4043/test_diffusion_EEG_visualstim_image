"""
dataset_vs_re.py
────────────────────────────────────────────────────────────────────────────
preproc_vs_re/ 데이터 로더

파일 구조 (MATLAB v7.3 / HDF5):
  results/data : (n_rep, n_blk, 9, 4096, 40)  float32
    dim0 = n_rep : trials per class per block  (nominally 5, may vary in partial saves)
    dim1 = n_blk : blocks per session          (nominally 3, may vary in partial saves)
    dim2 = 9     : classes (1~9)
    dim3 = 4096  : time points  (1024 Hz × 4 sec,  epoch: -1 ~ +3 sec)
    dim4 = 40    : channels     (32 EEG + 8 EX)

NOTE (HUMAN_DIRECTIVE Priority 4):
  - Never assume fixed dim0=5 or dim1=3; read actual tensor shape.
  - File count is NOT equal to valid session count (some files are partial saves).
  - Session numbering may have gaps; scan all files, do not stop at first gap.

기본 변환:
  - 시간 구간 : 0 ~ +2 sec  → sample index [1024:3072]
  - 채널      : 32 EEG only → index [:32]   (default)
               또는 40 all  → 명시적으로 n_ch=40 지정

출력:
  eeg   : (ch, time)  float32  (정규화 전)
  label : 0-based class index  (0~8)

사용 예시:
  from dataset_vs_re import load_subject_vsre, VSReDataset
  eeg, labels = load_subject_vsre(data_root, sid=1, n_ch=32)
  ds = VSReDataset(data_root, subject_ids=[1,2,3], n_ch=32)
"""

import os
from collections import defaultdict
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# epoch 내 0~2 sec 구간 (1024 Hz 기준)
_T_START   = 1024   # 0 sec
_T_END     = 3072   # 2 sec  →  2048 samples
_BL_START  = 0      # -1 sec (baseline)
_BL_END    = 1024   # 0 sec
_N_CH_NPZ  = 32     # convert_vsre_to_npz.py 가 저장하는 채널 수


def _list_sessions(data_root: str, sid: int) -> List[int]:
    """sid에 해당하는 세션 번호 목록 반환.

    .npz 와 .mat 모두 스캔. npz가 있으면 mat 없어도 세션으로 인정.
    NOTE: 파일 번호가 연속적이지 않을 수 있음 (gap 허용).
    """
    sessions = set()
    prefix = f"preproc_subj_{sid:02d}_"
    for fname in sorted(os.listdir(data_root)):
        if fname.startswith(prefix) and fname.endswith((".mat", ".npz")):
            ext_len = 4  # ".mat" or ".npz"
            try:
                sess_num = int(fname[len(prefix):-ext_len])
                sessions.add(sess_num)
            except ValueError:
                continue
    return sorted(sessions)


def load_subject_vsre(
    data_root: str,
    sid: int,
    n_ch: int = 32,
    t_start: int = _T_START,
    t_end: int = _T_END,
    sessions: Optional[List[int]] = None,
    max_sessions: Optional[int] = None,
    baseline_correct: bool = False,
    ch_zscore: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    피험자 sid의 모든(또는 지정) 세션을 병합해 반환.

    Parameters
    ----------
    data_root   : preproc_vs_re 경로
    sid         : 피험자 번호 (1-indexed)
    n_ch        : 사용할 채널 수  (32 = EEG only,  40 = EEG+EX)
    t_start/end : 시간 슬라이스 인덱스  (default: 0~2 sec)
    sessions    : 특정 세션 목록. None이면 전체 세션 사용.
    max_sessions: 최대 세션 수 (session-capped 실험용)

    Returns
    -------
    eeg    : (n_trials, n_ch, time)  float32
    labels : (n_trials,)  int64,  0-based (0~8)
    """
    if sessions is None:
        sessions = _list_sessions(data_root, sid)
    if max_sessions is not None:
        sessions = sessions[:max_sessions]
    if len(sessions) == 0:
        raise FileNotFoundError(f"No session files found for subject {sid} in {data_root}")

    eeg_list, lbl_list = [], []
    effective_sessions = 0

    for sess in sessions:
        npz_path = os.path.join(data_root, f"preproc_subj_{sid:02d}_{sess}.npz")
        mat_path = os.path.join(data_root, f"preproc_subj_{sid:02d}_{sess}.mat")

        # ── .npz 우선 로드 (fast path) ─────────────────────────────────────
        _npz_ok = (
            os.path.isfile(npz_path)
            and n_ch == _N_CH_NPZ           # npz는 32ch 고정
            and t_start == _T_START         # 같은 시간 슬라이스여야 함
            and t_end == _T_END
            and not baseline_correct        # baseline 데이터는 npz에 없음
        )
        if _npz_ok:
            try:
                z = np.load(npz_path)
                arr    = z["eeg"].astype(np.float32)   # (n_trials, 32, 2048)
                labels = z["labels"].astype(np.int64)
                print(f"  [dataset] S{sid:02d} sess{sess}: npz  shape={arr.shape}", flush=True)
            except Exception as e:
                print(f"  [dataset] S{sid:02d} sess{sess}: npz load failed ({e}), fallback to mat", flush=True)
                _npz_ok = False

        # ── .mat 폴백 ──────────────────────────────────────────────────────
        if not _npz_ok:
            if not os.path.isfile(mat_path):
                print(f"  [dataset] SKIP S{sid:02d} sess{sess}: neither .npz nor .mat found", flush=True)
                continue
            with h5py.File(mat_path, "r") as f:
                raw = f["results/data"]
                # Read actual shape — do NOT assume (5, 3, 9, 4096, 40)
                data = np.array(raw)                         # (n_rep, n_blk, n_cls, 4096, n_all_ch)

            if data.ndim != 5:
                print(f"  [dataset] SKIP S{sid:02d} sess{sess}: unexpected ndim={data.ndim}", flush=True)
                continue

            n_rep, n_blk, n_cls, n_time, n_all_ch = data.shape
            if n_rep == 0 or n_blk == 0 or n_cls == 0:
                print(f"  [dataset] SKIP S{sid:02d} sess{sess}: empty session shape={data.shape}", flush=True)
                continue
            # Sanity: n_rep should be small (nominally 5). Flag and skip if anomalously large.
            if n_rep > 20:
                print(f"  [dataset] SKIP S{sid:02d} sess{sess}: anomalous n_rep={n_rep} (expected <=20), shape={data.shape}", flush=True)
                continue

            # Baseline correction: subtract -1~0 sec mean from 0~2 sec signal
            if baseline_correct and _BL_END <= t_start:
                bl = data[:, :, :, _BL_START:_BL_END, :n_ch]   # (n_rep, n_blk, n_cls, 1024, ch)
                bl_mean = bl.mean(axis=3, keepdims=True)          # (n_rep, n_blk, n_cls, 1, ch)
                sig = data[:, :, :, t_start:t_end, :n_ch] - bl_mean
            else:
                sig = data[:, :, :, t_start:t_end, :n_ch]

            data = sig                                           # (n_rep, n_blk, n_cls, T, ch)
            _, _, _, T, ch = data.shape

            data = data.transpose(2, 4, 3, 0, 1)             # (n_cls, ch, T, n_rep, n_blk)
            data = data.reshape(n_cls, ch, T, -1)             # (n_cls, ch, T, n_rep*n_blk)
            data = data.transpose(0, 3, 1, 2)                # (n_cls, n_rep*n_blk, ch, T)
            data = data.reshape(-1, ch, T)                   # (n_cls*n_rep*n_blk, ch, T)

            n_trials_per_cls = n_rep * n_blk
            labels = np.repeat(np.arange(n_cls, dtype=np.int64), n_trials_per_cls)

            arr = data.astype(np.float32)   # (n_trials, ch, T)

        # Channel-wise z-score per trial
        if ch_zscore:
            mean = arr.mean(axis=2, keepdims=True)
            std  = arr.std(axis=2, keepdims=True).clip(min=1e-6)
            arr  = (arr - mean) / std

        eeg_list.append(arr)
        lbl_list.append(labels)
        effective_sessions += 1

    if len(eeg_list) == 0:
        raise FileNotFoundError(
            f"No valid session data found for subject {sid} in {data_root}"
        )

    eeg    = np.concatenate(eeg_list, axis=0)   # (total_trials, ch, T)
    labels = np.concatenate(lbl_list, axis=0)   # (total_trials,)
    return eeg, labels, effective_sessions


def available_subjects(data_root: str) -> List[int]:
    """데이터 루트에서 파일(.mat 또는 .npz)이 존재하는 피험자 ID 목록 반환."""
    sids = set()
    for fname in os.listdir(data_root):
        if fname.startswith("preproc_subj_") and fname.endswith((".mat", ".npz")):
            parts = fname.split("_")
            try:
                sids.add(int(parts[2]))
            except (IndexError, ValueError):
                pass
    return sorted(sids)


def session_counts(data_root: str) -> dict:
    """피험자별 세션 수 반환."""
    sids = available_subjects(data_root)
    return {sid: len(_list_sessions(data_root, sid)) for sid in sids}


# ── Dataset ──────────────────────────────────────────────────────────────────
class VSReDataset(Dataset):
    """
    preproc_vs_re 다중 피험자 Dataset.

    Parameters
    ----------
    data_root    : preproc_vs_re 경로
    subject_ids  : 학습에 포함할 피험자 목록
    subj_map     : {sid: embedding_index}  (없으면 자동 생성)
    n_ch         : 채널 수 (32 or 40)
    split        : "train" | "val" | "test"  (80:10:10 split per class per subject)
    seed         : shuffle seed
    max_sessions : 피험자당 최대 세션 수 (None = 전체)
    """

    def __init__(
        self,
        data_root: str,
        subject_ids: List[int],
        subj_map: Optional[dict] = None,
        n_ch: int = 32,
        split: str = "train",
        seed: int = 42,
        max_sessions: Optional[int] = None,
        baseline_correct: bool = False,
        ch_zscore: bool = False,
    ):
        if subj_map is None:
            subj_map = {sid: i for i, sid in enumerate(subject_ids)}

        self.samples = []   # (eeg_tensor, subj_idx, label_0based)

        for sid in subject_ids:
            eeg, labels, eff_sess = load_subject_vsre(
                data_root, sid, n_ch=n_ch, max_sessions=max_sessions,
                baseline_correct=baseline_correct, ch_zscore=ch_zscore,
            )
            si = subj_map[sid]
            if split == "train":
                # Report effective counts once (on train split construction)
                print(
                    f"  [dataset] S{sid:02d}  trials={len(labels)}  "
                    f"eff_sessions={eff_sess}",
                    flush=True,
                )
            # class별 train/val/test split
            for cls in range(9):
                idx = np.where(labels == cls)[0]
                if len(idx) == 0:
                    continue
                rng = np.random.RandomState(seed + si * 100 + cls)
                rng.shuffle(idx)
                n = len(idx)
                n_train = int(n * 0.8)
                n_val   = int(n * 0.1)
                if split == "train":
                    sel = idx[:n_train]
                elif split == "val":
                    sel = idx[n_train:n_train + n_val]
                else:
                    sel = idx[n_train + n_val:]
                for i in sel:
                    t = torch.from_numpy(eeg[i])   # (ch, T)
                    self.samples.append((t, si, int(labels[i])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate_fn(batch):
    eeg  = torch.stack([b[0] for b in batch])
    subj = torch.tensor([b[1] for b in batch], dtype=torch.long)
    lbl  = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return eeg, subj, lbl


# ── 빠른 확인용 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else "./preproc_vs_re"

    sids = available_subjects(data_root)
    sc   = session_counts(data_root)
    print(f"Subjects ({len(sids)}): {sids}")
    print(f"Sessions: {sc}")

    # 첫 피험자 로드 테스트
    sid = sids[0]
    sid = sids[0]
    eeg, labels, eff_sess = load_subject_vsre(data_root, sid, n_ch=32)
    print(f"\nS{sid:02d}  eeg={eeg.shape}  labels={labels.shape}  eff_sessions={eff_sess}")
    print(f"  label dist: { {c: int((labels==c).sum()) for c in range(9)} }")
    print(f"  eeg  mean={eeg.mean():.4f}  std={eeg.std():.4f}")

    # Dataset 테스트
    ds = VSReDataset(data_root, subject_ids=sids[:3], split="train")
    ds = VSReDataset(data_root, subject_ids=sids[:3], split="train")
    print(f"\nVSReDataset (3 subjects, train): {len(ds)} samples")
    eeg0, s0, l0 = ds[0]
    print(f"  sample[0]: eeg={eeg0.shape}  subj={s0}  label={l0}")
