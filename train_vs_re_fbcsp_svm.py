"""
train_vs_re_fbcsp_svm.py

Subject-wise EEG classification on preproc_vs_re using:
- Filter Bank CSP
- SVM

Bands:
- delta: 1-4 Hz
- theta: 4-8 Hz
- alpha: 8-13 Hz
- beta : 13-30 Hz
- gamma: 30-45 Hz

Current scope:
- preproc_vs_re
- 0~2 sec window only
- 32 EEG channels by default
- subject-wise classification

Usage examples:
  python train_vs_re_fbcsp_svm.py --subject_ids all
  python train_vs_re_fbcsp_svm.py --subject_ids 1,2,18 --csp_components 2
"""

import argparse
import csv
import os
from datetime import datetime

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset_vs_re import available_subjects, load_subject_vsre, session_counts


FS = 1024
BANDS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
]


def set_seed(seed: int):
    np.random.seed(seed)


def bandpass_filter_trials(x: np.ndarray, fs: int, low: float, high: float, order: int = 4) -> np.ndarray:
    """
    x: (n_trials, n_channels, n_times)
    """
    nyq = fs * 0.5
    low = max(low / nyq, 1e-6)
    high = min(high / nyq, 0.999)
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def stratified_split(labels: np.ndarray, seed: int = 42):
    """
    Match current subject-wise split logic used elsewhere:
    train = floor(0.8n), val = floor(0.1n), test = rest, per class.
    """
    train_idx, val_idx, test_idx = [], [], []
    for cls in sorted(np.unique(labels)):
        idx = np.where(labels == cls)[0]
        rng = np.random.RandomState(seed + int(cls))
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:n_train + n_val])
        test_idx.append(idx[n_train + n_val:])
    return (
        np.concatenate(train_idx),
        np.concatenate(val_idx),
        np.concatenate(test_idx),
    )


def compute_csp_filters(x: np.ndarray, y_bin: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Binary CSP.
    x: (n_trials, n_channels, n_times)
    y_bin: (n_trials,) in {0,1}
    returns W: (2*n_components, n_channels)
    """
    eps = 1e-8
    classes = [0, 1]
    covs = []
    for cls in classes:
        cls_x = x[y_bin == cls]
        if len(cls_x) == 0:
            raise ValueError("Binary CSP received empty class.")
        cls_cov = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
        for trial in cls_x:
            c = trial @ trial.T
            c = c / (np.trace(c) + eps)
            cls_cov += c
        cls_cov /= len(cls_x)
        covs.append(cls_cov)

    c0, c1 = covs
    composite = c0 + c1
    evals, evecs = np.linalg.eigh(composite)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    whitening = np.diag(1.0 / np.sqrt(evals + eps)) @ evecs.T

    s0 = whitening @ c0 @ whitening.T
    evals0, b = np.linalg.eigh(s0)
    order0 = np.argsort(evals0)[::-1]
    b = b[:, order0]
    w = b.T @ whitening

    picks = np.r_[0:n_components, -n_components:0]
    return w[picks].astype(np.float32)


class FBCSPExtractor:
    def __init__(self, bands, n_classes: int = 9, n_components: int = 2, fs: int = FS):
        self.bands = bands
        self.n_classes = n_classes
        self.n_components = n_components
        self.fs = fs
        self.filters_ = {}

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.filters_ = {}
        for band_name, low, high in self.bands:
            xb = bandpass_filter_trials(x_train, self.fs, low, high)
            self.filters_[band_name] = []
            for cls in range(self.n_classes):
                y_bin = (y_train == cls).astype(np.int64)
                w = compute_csp_filters(xb, y_bin, n_components=self.n_components)
                self.filters_[band_name].append(w)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        eps = 1e-8
        feats = []
        for band_name, low, high in self.bands:
            xb = bandpass_filter_trials(x, self.fs, low, high)
            band_feats = []
            for w in self.filters_[band_name]:
                z = np.einsum("kc,nct->nkt", w, xb)  # (n_trials, 2m, T)
                var = np.var(z, axis=-1)
                logvar = np.log(var / (np.sum(var, axis=1, keepdims=True) + eps) + eps)
                band_feats.append(logvar)
            feats.append(np.concatenate(band_feats, axis=1))
        return np.concatenate(feats, axis=1).astype(np.float32)

    def fit_transform(self, x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        self.fit(x_train, y_train)
        return self.transform(x_train)


def parse_subject_ids(spec: str, all_sids):
    if spec == "all":
        return all_sids
    subject_ids = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            subject_ids.extend(range(int(a), int(b) + 1))
        else:
            subject_ids.append(int(tok))
    return [s for s in subject_ids if s in all_sids]


def train_subject(sid: int, args, save_dir: str):
    loaded = load_subject_vsre(
        args.data_root,
        sid,
        n_ch=args.n_ch,
        max_sessions=args.max_sessions,
    )
    if len(loaded) == 3:
        x, y, effective_sessions = loaded
    else:
        x, y = loaded
        effective_sessions = int(len(x) // 135)
    train_idx, val_idx, test_idx = stratified_split(y, seed=args.seed)

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    extractor = FBCSPExtractor(
        bands=BANDS,
        n_classes=9,
        n_components=args.csp_components,
        fs=FS,
    )

    feat_train = extractor.fit_transform(x_train, y_train)
    feat_val = extractor.transform(x_val)
    feat_test = extractor.transform(x_test)

    clf = make_pipeline(
        StandardScaler(),
        SVC(
            C=args.C,
            kernel=args.kernel,
            gamma=args.gamma,
            decision_function_shape="ovr",
        ),
    )
    clf.fit(feat_train, y_train)

    pred_val = clf.predict(feat_val) if len(feat_val) > 0 else np.array([], dtype=y.dtype)
    pred_test = clf.predict(feat_test)

    val_acc = accuracy_score(y_val, pred_val) if len(y_val) > 0 else float("nan")
    test_acc = accuracy_score(y_test, pred_test)

    subj_dir = os.path.join(save_dir, f"subj{sid:02d}")
    os.makedirs(subj_dir, exist_ok=True)

    cm = confusion_matrix(y_test, pred_test, labels=np.arange(9))
    cm_path = os.path.join(subj_dir, "confusion_matrix.csv")
    np.savetxt(cm_path, cm, fmt="%d", delimiter=",")

    return {
        "sid": sid,
        "sessions": int(effective_sessions),
        "train": int(len(train_idx)),
        "val": int(len(val_idx)),
        "test": int(len(test_idx)),
        "val_acc": float(val_acc),
        "test_acc": float(test_acc),
        "random": 1.0 / 9.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./preproc_vs_re")
    parser.add_argument("--subject_ids", type=str, default="all")
    parser.add_argument("--n_ch", type=int, default=32)
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csp_components", type=int, default=2)
    parser.add_argument("--kernel", type=str, default="rbf", choices=["linear", "rbf"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--out_root", type=str, default="./checkpoints_vsre_fbcsp_svm")
    args = parser.parse_args()

    set_seed(args.seed)
    all_sids = available_subjects(args.data_root)
    subject_ids = parse_subject_ids(args.subject_ids, all_sids)
    sc = session_counts(args.data_root)

    print(f"[INFO] Subject-wise FBCSP+SVM")
    print(f"[INFO] Subjects ({len(subject_ids)}): {subject_ids}")
    print(f"[INFO] Session counts: {sc}")
    print(f"[INFO] Bands: {[b[0] for b in BANDS]}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sess_tag = f"cap{args.max_sessions}" if args.max_sessions else "merged"
    save_dir = os.path.join(
        args.out_root,
        f"{ts}_ch{args.n_ch}_{sess_tag}_csp{args.csp_components}_{args.kernel}",
    )
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    print("\n=======================================================")
    print("  Subject-wise FBCSP + SVM results [9cls]")
    print("   Subj   Sess   Train   Val   Test    ValAcc   TestAcc")
    print(" ------------------------------------------------------")
    for sid in subject_ids:
        r = train_subject(sid, args, save_dir)
        rows.append(r)
        print(
            f"  S{sid:02d}   {r['sessions']:>4}   {r['train']:>5}  {r['val']:>4}  "
            f"{r['test']:>5}   {r['val_acc']:.4f}   {r['test_acc']:.4f}"
        )

    mean_val = float(np.mean([r["val_acc"] for r in rows])) if rows else 0.0
    mean_test = float(np.mean([r["test_acc"] for r in rows])) if rows else 0.0
    print(" ------------------------------------------------------")
    print(f"   Mean                              {mean_val:.4f}   {mean_test:.4f}")
    print(f" Random                              {1.0/9.0:.4f}   {1.0/9.0:.4f}")

    out_csv = os.path.join(save_dir, "results_fbcsp_svm.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "sessions", "train", "val", "test", "val_acc", "test_acc", "random"])
        for r in rows:
            w.writerow([
                f"S{r['sid']:02d}",
                r["sessions"],
                r["train"],
                r["val"],
                r["test"],
                round(r["val_acc"], 6),
                round(r["test_acc"], 6),
                round(r["random"], 6),
            ])

    print(f"\n[INFO] Saved: {save_dir}")
    print(f"[INFO] CSV: {out_csv}")


if __name__ == "__main__":
    main()
