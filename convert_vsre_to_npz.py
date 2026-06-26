"""
convert_vsre_to_npz.py
──────────────────────────────────────────────────────────────
preproc_vs_re/*.mat → preproc_vs_re/*.npz  (float16, compressed)

동일한 전처리(시간 슬라이스, 채널 선택)를 적용해 npz로 저장.
dataset_vs_re.py가 .npz를 우선 읽으므로 변환 후에는 .mat 없이도 동작.

Usage:
    python convert_vsre_to_npz.py --subject_ids 24
    python convert_vsre_to_npz.py --subject_ids 1,18,24
    python convert_vsre_to_npz.py --subject_ids all --data_root preproc_vs_re

Output:
    preproc_vs_re/preproc_subj_24_1.npz  (float16, compressed)
    preproc_vs_re/preproc_subj_24_2.npz  ...
    ...
"""

import argparse
import os
import sys

import h5py
import numpy as np

_T_START = 1024   # 0 sec  (1024 Hz * 1 sec offset)
_T_END   = 3072   # 2 sec
_N_CH    = 32


def list_sessions(data_root: str, sid: int):
    sessions = []
    for fname in os.listdir(data_root):
        prefix = f"preproc_subj_{sid:02d}_"
        if fname.startswith(prefix) and fname.endswith(".mat"):
            try:
                sessions.append(int(fname[len(prefix):-4]))
            except ValueError:
                continue
    return sorted(sessions)


def convert_session(data_root: str, sid: int, sess: int, overwrite: bool = False):
    mat_path = os.path.join(data_root, f"preproc_subj_{sid:02d}_{sess}.mat")
    npz_path = os.path.join(data_root, f"preproc_subj_{sid:02d}_{sess}.npz")

    if not os.path.isfile(mat_path):
        print(f"  SKIP: {mat_path} not found")
        return None

    if os.path.isfile(npz_path) and not overwrite:
        sz = os.path.getsize(npz_path) / 1e6
        print(f"  EXISTS: {npz_path} ({sz:.1f} MB) -- use --overwrite to redo")
        return npz_path

    print(f"  Converting S{sid:02d} sess{sess}: {mat_path} ...", end=" ", flush=True)

    with h5py.File(mat_path, "r") as f:
        data = np.array(f["results/data"])   # (n_rep, n_blk, n_cls, 4096, n_all_ch)

    if data.ndim != 5:
        print(f"SKIP (ndim={data.ndim})")
        return None

    n_rep, n_blk, n_cls, n_time, n_all_ch = data.shape
    if n_rep == 0 or n_blk == 0 or n_cls == 0 or n_rep > 20:
        print(f"SKIP (shape={data.shape})")
        return None

    # same slice as dataset_vs_re.py
    sig = data[:, :, :, _T_START:_T_END, :_N_CH]   # (n_rep, n_blk, n_cls, 2048, 32)

    # reshape to (n_trials, 32, 2048)
    sig = sig.transpose(2, 4, 3, 0, 1)              # (n_cls, ch, T, n_rep, n_blk)
    sig = sig.reshape(n_cls, _N_CH, _T_END - _T_START, -1)
    sig = sig.transpose(0, 3, 1, 2)                 # (n_cls, n_rep*n_blk, ch, T)
    sig = sig.reshape(-1, _N_CH, _T_END - _T_START) # (n_trials, ch, T)

    n_trials_per_cls = n_rep * n_blk
    labels = np.repeat(np.arange(n_cls, dtype=np.int64), n_trials_per_cls)

    eeg = sig.astype(np.float16)   # float16 for size

    np.savez_compressed(npz_path, eeg=eeg, labels=labels,
                        sid=np.int32(sid), sess=np.int32(sess),
                        t_start=np.int32(_T_START), t_end=np.int32(_T_END),
                        n_ch=np.int32(_N_CH))

    sz_mb = os.path.getsize(npz_path) / 1e6
    print(f"OK  shape={eeg.shape} dtype=float16  {sz_mb:.1f} MB")
    return npz_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    default="preproc_vs_re")
    parser.add_argument("--subject_ids",  default="24",
                        help="comma-separated, or 'all'")
    parser.add_argument("--overwrite",    action="store_true")
    args = parser.parse_args()

    # parse subject list
    if args.subject_ids.strip().lower() == "all":
        # find all subjects present
        sids = set()
        for fname in os.listdir(args.data_root):
            if fname.startswith("preproc_subj_") and fname.endswith(".mat"):
                try:
                    sids.add(int(fname.split("_")[2]))
                except (IndexError, ValueError):
                    pass
        sids = sorted(sids)
    else:
        sids = [int(x) for x in args.subject_ids.split(",")]

    total_mb = 0.0
    for sid in sids:
        sessions = list_sessions(args.data_root, sid)
        if not sessions:
            print(f"S{sid:02d}: no .mat files found in {args.data_root}")
            continue
        print(f"\nS{sid:02d}: {len(sessions)} sessions {sessions}")
        for sess in sessions:
            p = convert_session(args.data_root, sid, sess, args.overwrite)
            if p and os.path.isfile(p):
                total_mb += os.path.getsize(p) / 1e6

    print(f"\n=== Done. Total .npz size: {total_mb:.1f} MB ===")
    if total_mb < 100:
        print("  -> GitHub 업로드 가능 (100MB 미만)")
    else:
        print(f"  -> {total_mb:.0f} MB. 개별 파일은 100MB 미만이면 업로드 가능.")


if __name__ == "__main__":
    main()
