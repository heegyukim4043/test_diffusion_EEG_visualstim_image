"""Riemannian / tangent-space baseline for VI EEG decoding.

This script is intentionally independent from the Transformer encoder track.
It uses the same VSReDataset train/val/test split, but replaces deep feature
learning with covariance features:

    EEG trial -> covariance SPD matrix -> tangent-space vector -> linear model

The main purpose is to test whether a strong small-data EEG baseline can beat
or match raw+TF Transformer VI-only decoding.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from dataset_vs_re import VSReDataset, session_counts


N_CLASSES = 9
DEFAULT_BANDS = (
    (4.0, 7.0),    # theta
    (8.0, 13.0),   # alpha
    (14.0, 20.0),  # low beta
    (21.0, 30.0),  # high beta
    (31.0, 45.0),  # low gamma
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", required=True, type=int)
    parser.add_argument("--vi_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_ch", type=int, default=32)
    parser.add_argument("--sampling_rate", type=float, default=1024.0)
    parser.add_argument("--feature", choices=("cov", "filterbank"), default="filterbank")
    parser.add_argument("--classifier", choices=("logreg", "linsvm", "lda"), default="logreg")
    parser.add_argument("--shrinkage", type=float, default=0.10)
    parser.add_argument("--logreg_c", type=float, default=1.0)
    parser.add_argument("--svm_c", type=float, default=1.0)
    parser.add_argument("--crop_start_sample", type=int, default=0)
    parser.add_argument("--crop_end_sample", type=int, default=0, help="0 means use trial end")
    parser.add_argument("--ch_zscore", action="store_true")
    parser.add_argument("--n_permutations", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(root: str, sid: int, split: str, seed: int, n_ch: int) -> tuple[np.ndarray, np.ndarray]:
    dataset = VSReDataset(root, [sid], {sid: 0}, n_ch, split, seed)
    xs, ys = [], []
    removed = 0
    for eeg, _, label in dataset.samples:
        if not torch.isfinite(eeg).all().item():
            removed += 1
            continue
        xs.append(eeg.detach().cpu().numpy().astype(np.float64, copy=False))
        ys.append(int(label))
    if not xs:
        raise RuntimeError(f"S{sid:02d} {split} split is empty after non-finite filtering")
    if removed:
        print(f"[WARN] S{sid:02d} {split}: removed {removed} non-finite trials", flush=True)
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64)


def crop_and_normalize(X: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    start = max(0, args.crop_start_sample)
    end = X.shape[-1] if args.crop_end_sample <= 0 else min(X.shape[-1], args.crop_end_sample)
    if end <= start:
        raise ValueError(f"Invalid crop [{start}:{end}] for trial length {X.shape[-1]}")
    X = X[:, :, start:end].copy()
    if args.ch_zscore:
        mean = X.mean(axis=-1, keepdims=True)
        std = X.std(axis=-1, keepdims=True)
        X = (X - mean) / np.maximum(std, 1e-8)
    return X


def bandpass(X: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    nyq = fs / 2.0
    high = min(high, nyq - 1.0)
    if not (0.0 < low < high):
        raise ValueError(f"Invalid band {low}-{high} Hz for fs={fs}")
    sos = signal.butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, X, axis=-1)


def regularized_covariances(X: np.ndarray, shrinkage: float) -> np.ndarray:
    """Return trial covariance matrices with trace-normalized diagonal shrinkage."""
    n, c, t = X.shape
    covs = np.empty((n, c, c), dtype=np.float64)
    eye = np.eye(c, dtype=np.float64)
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
    for i in range(n):
        x = X[i] - X[i].mean(axis=1, keepdims=True)
        cov = (x @ x.T) / max(t - 1, 1)
        cov = (cov + cov.T) * 0.5
        scale = np.trace(cov) / c
        cov = (1.0 - shrinkage) * cov + shrinkage * scale * eye
        covs[i] = (cov + cov.T) * 0.5
    return covs


def spd_eigh(matrix: np.ndarray, eps: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh((matrix + matrix.T) * 0.5)
    vals = np.maximum(vals, eps)
    return vals, vecs


def spd_log(matrix: np.ndarray) -> np.ndarray:
    vals, vecs = spd_eigh(matrix)
    return (vecs * np.log(vals)) @ vecs.T


def spd_expm(matrix: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((matrix + matrix.T) * 0.5)
    return (vecs * np.exp(vals)) @ vecs.T


def log_euclidean_mean(covs: np.ndarray) -> np.ndarray:
    logs = np.stack([spd_log(cov) for cov in covs], axis=0)
    return spd_expm(logs.mean(axis=0))


def fit_tangent_reference(covs: np.ndarray) -> dict:
    ref = log_euclidean_mean(covs)
    vals, vecs = spd_eigh(ref)
    inv_sqrt = (vecs * (vals ** -0.5)) @ vecs.T
    return {"ref": ref, "inv_sqrt": inv_sqrt}


def tangent_project(covs: np.ndarray, state: dict) -> np.ndarray:
    inv_sqrt = state["inv_sqrt"]
    c = covs.shape[1]
    tri = np.triu_indices(c)
    features = np.empty((len(covs), c * (c + 1) // 2), dtype=np.float64)
    offdiag = tri[0] != tri[1]
    for i, cov in enumerate(covs):
        whitened = inv_sqrt @ cov @ inv_sqrt
        logm = spd_log(whitened)
        vec = logm[tri].copy()
        vec[offdiag] *= math.sqrt(2.0)
        features[i] = vec
    return features


def make_feature_blocks(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, args: argparse.Namespace):
    if args.feature == "cov":
        bands = [(0.0, 0.0, "broadband")]
    else:
        bands = [(lo, hi, f"{lo:g}-{hi:g}Hz") for lo, hi in DEFAULT_BANDS]

    train_blocks, val_blocks, test_blocks, diagnostics = [], [], [], []
    for low, high, name in bands:
        if args.feature == "cov":
            Xt, Xv, Xs = X_train, X_val, X_test
        else:
            Xt = bandpass(X_train, low, high, args.sampling_rate)
            Xv = bandpass(X_val, low, high, args.sampling_rate)
            Xs = bandpass(X_test, low, high, args.sampling_rate)

        cov_train = regularized_covariances(Xt, args.shrinkage)
        cov_val = regularized_covariances(Xv, args.shrinkage)
        cov_test = regularized_covariances(Xs, args.shrinkage)

        tangent_state = fit_tangent_reference(cov_train)
        train_blocks.append(tangent_project(cov_train, tangent_state))
        val_blocks.append(tangent_project(cov_val, tangent_state))
        test_blocks.append(tangent_project(cov_test, tangent_state))

        vals = np.linalg.eigvalsh(tangent_state["ref"])
        diagnostics.append({
            "band": name,
            "feature_dim": int(train_blocks[-1].shape[1]),
            "ref_condition_number": float(vals.max() / max(vals.min(), 1e-12)),
        })

    return (
        np.concatenate(train_blocks, axis=1),
        np.concatenate(val_blocks, axis=1),
        np.concatenate(test_blocks, axis=1),
        diagnostics,
    )


def build_classifier(args: argparse.Namespace):
    if args.classifier == "logreg":
        model = LogisticRegression(
            C=args.logreg_c,
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="auto",
            random_state=args.seed,
        )
    elif args.classifier == "linsvm":
        model = LinearSVC(
            C=args.svm_c,
            class_weight="balanced",
            max_iter=20000,
            random_state=args.seed,
            dual="auto",
        )
    elif args.classifier == "lda":
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    else:
        raise ValueError(args.classifier)
    return make_pipeline(StandardScaler(), model)


def decision_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
    else:
        raise RuntimeError("Classifier has neither decision_function nor predict_proba")
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    return np.asarray(scores, dtype=np.float64)


def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray) -> tuple[dict, list[dict], np.ndarray]:
    pred = scores.argmax(axis=1)
    confusion = confusion_matrix(y_true, pred, labels=np.arange(N_CLASSES))
    recalls = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    order = np.argsort(scores, axis=1)[:, ::-1]
    top1 = float((order[:, :1] == y_true[:, None]).any(axis=1).mean())
    top3 = float((order[:, :3] == y_true[:, None]).any(axis=1).mean())
    top5 = float((order[:, :5] == y_true[:, None]).any(axis=1).mean())
    counts = confusion.sum(axis=0)
    probs = counts / max(counts.sum(), 1)
    nonzero = probs[probs > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum()) if len(nonzero) else 0.0
    dominant = int(counts.argmax()) if counts.sum() else 0

    rows = []
    for i, truth in enumerate(y_true):
        truth = int(truth)
        predicted = int(pred[i])
        other = np.concatenate([scores[i, :truth], scores[i, truth + 1 :]])
        rows.append({
            "sample_index": i,
            "true_label": truth,
            "pred_label": predicted,
            "correct": int(truth == predicted),
            "true_score": float(scores[i, truth]),
            "true_margin": float(scores[i, truth] - other.max()),
        })

    return {
        "n": int(len(y_true)),
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "balanced_accuracy": float(recalls.mean()),
        "mean_true_margin": float(np.mean([row["true_margin"] for row in rows])),
        "normalized_entropy": entropy / math.log(N_CLASSES),
        "dominant_label": dominant,
        "dominant_ratio": float(counts[dominant] / max(counts.sum(), 1)),
        "prediction_counts": counts.tolist(),
    }, rows, confusion


def permutation_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    if args.n_permutations <= 0:
        return {}
    rng = np.random.default_rng(args.seed + 1009)
    observed = None
    perm_bac = []
    for i in range(args.n_permutations + 1):
        y_fit = y_train if i == 0 else rng.permutation(y_train)
        model = build_classifier(args)
        model.fit(X_train, y_fit)
        scores = decision_scores(model, X_test)
        metric, _, _ = metrics_from_scores(y_test, scores)
        if i == 0:
            observed = metric["balanced_accuracy"]
        else:
            perm_bac.append(metric["balanced_accuracy"])
    perm_bac = np.asarray(perm_bac, dtype=np.float64)
    p = float((1 + np.sum(perm_bac >= observed)) / (len(perm_bac) + 1))
    return {
        "n_permutations": int(args.n_permutations),
        "observed_bac": float(observed),
        "permutation_bac_mean": float(perm_bac.mean()),
        "permutation_bac_std": float(perm_bac.std(ddof=1)) if len(perm_bac) > 1 else 0.0,
        "permutation_p_ge_observed": p,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite to rerun")

    print(
        f"[INFO] S{args.subject_id:02d} feature={args.feature} classifier={args.classifier} "
        f"seed={args.seed}",
        flush=True,
    )
    X_train, y_train = load_split(args.vi_root, args.subject_id, "train", args.seed, args.n_ch)
    X_val, y_val = load_split(args.vi_root, args.subject_id, "val", args.seed, args.n_ch)
    X_test, y_test = load_split(args.vi_root, args.subject_id, "test", args.seed, args.n_ch)
    X_train = crop_and_normalize(X_train, args)
    X_val = crop_and_normalize(X_val, args)
    X_test = crop_and_normalize(X_test, args)

    F_train, F_val, F_test, diagnostics = make_feature_blocks(X_train, X_val, X_test, args)
    model = build_classifier(args)
    model.fit(F_train, y_train)

    val_scores = decision_scores(model, F_val)
    test_scores = decision_scores(model, F_test)
    val_metrics, val_rows, val_confusion = metrics_from_scores(y_val, val_scores)
    test_metrics, test_rows, test_confusion = metrics_from_scores(y_test, test_scores)
    perm = permutation_test(F_train, y_train, F_test, y_test, args)

    counts = session_counts(args.vi_root)
    metrics = {
        "subject": args.subject_id,
        "stage": "vi_riemannian",
        "seed": args.seed,
        "feature": args.feature,
        "classifier": args.classifier,
        "n_sessions": counts.get(args.subject_id),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "feature_dim": int(F_train.shape[1]),
        "config": vars(args),
        "feature_diagnostics": diagnostics,
        "validation": val_metrics,
        "test": test_metrics,
        "permutation_test": perm,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    write_csv(out_dir / "validation_predictions.csv", val_rows)
    write_csv(out_dir / "test_predictions.csv", test_rows)
    np.savetxt(out_dir / "validation_confusion.csv", val_confusion, delimiter=",", fmt="%d")
    np.savetxt(out_dir / "test_confusion.csv", test_confusion, delimiter=",", fmt="%d")

    print(
        f"[DONE] val_BAC={val_metrics['balanced_accuracy']:.4f} "
        f"test_BAC={test_metrics['balanced_accuracy']:.4f} "
        f"test@3={test_metrics['top3']:.4f} test@5={test_metrics['top5']:.4f} "
        f"dom={test_metrics['dominant_ratio']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
