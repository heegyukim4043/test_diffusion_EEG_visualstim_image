"""Leakage-controlled Euclidean Alignment ablation for VS-to-VI transfer.

The alignment matrix for an inductive domain is fitted on its *training split
only* and then kept fixed for validation/test.  Two zero-shot variants are kept
separate:

``zero_shot_strict``
    Uses only the VS training reference.  No VI sample is used for adaptation.

``zero_shot_calibrated``
    Uses the unlabeled VI training split to estimate a VI covariance reference.
    This is unsupervised VI calibration, not pure zero-shot.

The script reuses the raw classification architecture and optimization from
``train_vi_tf_representation_ablation.py`` so that EA is the only intended
change from the existing raw baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_vs_re import VSReDataset, collate_fn, session_counts
from train_vi_tf_representation_ablation import (
    RepresentationEncoder,
    evaluate,
    filter_nonfinite,
    safe_torch_load,
    set_seed,
    train_model,
    write_outputs,
)


STAGES = (
    "vs_pretrain",
    "zero_shot_strict",
    "zero_shot_calibrated",
    "vi_only",
    "vs_to_vi",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--subject_id", required=True, type=int)
    parser.add_argument("--vs_root", required=True)
    parser.add_argument("--vi_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--init_ckpt", default="")
    parser.add_argument("--source_alignment", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_ch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--w_supcon", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--shrinkage", type=float, default=0.05)
    parser.add_argument("--eigen_eps", type=float, default=1e-6)
    parser.add_argument("--alignment_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    # Required by the shared RepresentationEncoder.  The EA pilot is raw-only,
    # but recording these values makes checkpoints self-describing.
    parser.set_defaults(
        representation="raw",
        sampling_rate=1024.0,
        n_fft=256,
        hop_length=64,
    )
    return parser.parse_args()


def build_datasets(root: str, sid: int, args: argparse.Namespace, domain: str):
    subject_map = {sid: 0}
    datasets = {
        split: VSReDataset(
            root,
            [sid],
            subject_map,
            args.n_ch,
            split,
            args.seed,
        )
        for split in ("train", "val", "test")
    }
    removed = {
        split: filter_nonfinite(dataset, f"{domain}_{split}")
        for split, dataset in datasets.items()
    }
    for split, dataset in datasets.items():
        if not dataset.samples:
            raise RuntimeError(f"S{sid:02d} {domain} {split} split is empty")
    return datasets, removed


def fit_ea_reference(
    samples: list[tuple[torch.Tensor, int, int]],
    shrinkage: float,
    eigen_eps: float,
    batch_size: int,
) -> tuple[torch.Tensor, dict]:
    """Fit R^(-1/2) using training trials only."""
    if not 0.0 <= shrinkage < 1.0:
        raise ValueError("shrinkage must satisfy 0 <= shrinkage < 1")
    if not samples:
        raise ValueError("Cannot fit EA on an empty training split")

    channels = samples[0][0].shape[0]
    covariance_sum = torch.zeros(channels, channels, dtype=torch.float64)
    n_trials = 0
    for start in range(0, len(samples), batch_size):
        batch = torch.stack([item[0] for item in samples[start : start + batch_size]]).float()
        batch = batch - batch.mean(dim=-1, keepdim=True)
        time = batch.shape[-1]
        covariance_sum += torch.bmm(batch, batch.transpose(1, 2)).sum(dim=0).double() / time
        n_trials += batch.shape[0]

    covariance = covariance_sum / n_trials
    scale = torch.trace(covariance) / channels
    covariance_regularized = (
        (1.0 - shrinkage) * covariance
        + shrinkage * scale * torch.eye(channels, dtype=torch.float64)
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_regularized)
    floor = max(float(eigen_eps * scale), np.finfo(np.float64).eps)
    eigenvalues_clipped = eigenvalues.clamp_min(floor)
    reference = (
        eigenvectors
        @ torch.diag(eigenvalues_clipped.rsqrt())
        @ eigenvectors.T
    ).float()
    diagnostics = {
        "fit_trials": n_trials,
        "channels": channels,
        "shrinkage": shrinkage,
        "eigen_eps": eigen_eps,
        "covariance_scale": float(scale),
        "minimum_eigenvalue_raw": float(eigenvalues.min()),
        "minimum_eigenvalue_used": float(eigenvalues_clipped.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "condition_number_regularized": float(
            eigenvalues_clipped.max() / eigenvalues_clipped.min()
        ),
    }
    return reference, diagnostics


def apply_ea_to_dataset(
    dataset: VSReDataset,
    reference: torch.Tensor,
    batch_size: int,
) -> None:
    """Materialize aligned trials once so training epochs do not redo EA."""
    updated = []
    for start in range(0, len(dataset.samples), batch_size):
        items = dataset.samples[start : start + batch_size]
        batch = torch.stack([item[0] for item in items]).float()
        batch = batch - batch.mean(dim=-1, keepdim=True)
        aligned = torch.matmul(reference.unsqueeze(0), batch)
        for index, (_, subject, label) in enumerate(items):
            updated.append((aligned[index].contiguous(), subject, label))
    dataset.samples = updated


def covariance_error_after_alignment(
    samples: list[tuple[torch.Tensor, int, int]],
    max_trials: int = 256,
) -> float:
    chosen = samples[:max_trials]
    channels = chosen[0][0].shape[0]
    covariance = torch.zeros(channels, channels, dtype=torch.float64)
    for eeg, _, _ in chosen:
        centered = eeg.double() - eeg.double().mean(dim=-1, keepdim=True)
        covariance += centered @ centered.T / centered.shape[-1]
    covariance /= len(chosen)
    identity = torch.eye(channels, dtype=torch.float64)
    return float(torch.linalg.norm(covariance - identity) / torch.linalg.norm(identity))


def make_loaders(datasets: dict[str, VSReDataset], args: argparse.Namespace):
    generator = torch.Generator().manual_seed(args.seed)
    return {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            generator=generator if split == "train" else None,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets.items()
    }


def resolve_alignment(
    args: argparse.Namespace,
    datasets: dict[str, VSReDataset],
) -> tuple[torch.Tensor, dict, str]:
    if args.stage == "zero_shot_strict":
        if not args.source_alignment:
            raise ValueError("--source_alignment is required for strict zero-shot")
        path = Path(args.source_alignment)
        if not path.is_file():
            raise FileNotFoundError(path)
        reference = torch.from_numpy(np.load(path).astype(np.float32))
        return reference, {"loaded_from": str(path.resolve())}, "vs_train"

    # VS pretraining fits VS train.  All remaining stages fit VI train; the
    # calibrated zero-shot stage must be described as unlabeled VI calibration.
    reference, diagnostics = fit_ea_reference(
        datasets["train"].samples,
        args.shrinkage,
        args.eigen_eps,
        args.alignment_batch_size,
    )
    fit_domain = "vs_train" if args.stage == "vs_pretrain" else "vi_train_unlabeled"
    return reference, diagnostics, fit_domain


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.out_dir)
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite to rerun")

    domain = "vs" if args.stage == "vs_pretrain" else "vi"
    root = args.vs_root if domain == "vs" else args.vi_root
    counts = session_counts(root)
    if args.subject_id not in counts:
        raise FileNotFoundError(f"S{args.subject_id:02d} missing from {root}")

    datasets, removed = build_datasets(root, args.subject_id, args, domain)
    reference, alignment_diagnostics, alignment_fit_domain = resolve_alignment(args, datasets)
    for dataset in datasets.values():
        apply_ea_to_dataset(dataset, reference, args.alignment_batch_size)
    identity_error = covariance_error_after_alignment(datasets["train"].samples)
    loaders = make_loaders(datasets, args)

    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "alignment_matrix.npy", reference.numpy())
    with (output / "alignment_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **alignment_diagnostics,
                "fit_domain": alignment_fit_domain,
                "train_covariance_identity_error_after_alignment": identity_error,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[INFO] stage={args.stage} subject=S{args.subject_id:02d} domain={domain} "
        f"sessions={counts[args.subject_id]} fit={alignment_fit_domain} device={device}",
        flush=True,
    )
    print(
        f"[EA] train-only fit; post-alignment identity error={identity_error:.6f}",
        flush=True,
    )

    model = RepresentationEncoder(args).to(device)
    initialized_from_vs = args.stage in (
        "zero_shot_strict",
        "zero_shot_calibrated",
        "vs_to_vi",
    )
    if initialized_from_vs:
        if not args.init_ckpt:
            raise ValueError(f"--init_ckpt is required for {args.stage}")
        checkpoint_path = Path(args.init_ckpt)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = safe_torch_load(checkpoint_path, "cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        print(f"[INIT] Loaded EA-VS checkpoint: {checkpoint_path}", flush=True)

    train_stage = args.stage in ("vs_pretrain", "vi_only", "vs_to_vi")
    history = []
    best_epoch = 0
    if train_stage:
        best_epoch, history = train_model(
            model,
            loaders,
            args,
            device,
            output / "encoder_best.pt",
        )

    test_metrics, rows, confusion = evaluate(model, loaders["test"], device)
    metrics = {
        "subject": args.subject_id,
        "stage": args.stage,
        "representation": "raw_ea",
        "domain": domain,
        "seed": args.seed,
        "n_sessions": counts[args.subject_id],
        "n_train": len(datasets["train"]),
        "n_val": len(datasets["val"]),
        "nonfinite_removed": removed,
        "initialized_from_vs": initialized_from_vs,
        "initial_checkpoint": str(Path(args.init_ckpt).resolve()) if args.init_ckpt else None,
        "alignment_fit_domain": alignment_fit_domain,
        "alignment_is_transductive": False,
        "uses_unlabeled_vi_calibration": args.stage == "zero_shot_calibrated",
        "alignment_diagnostics": alignment_diagnostics,
        "train_covariance_identity_error_after_alignment": identity_error,
        "best_epoch": best_epoch,
        **test_metrics,
        "config": vars(args),
    }
    write_outputs(output, metrics, rows, confusion, history)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"[Saved] {output}", flush=True)


if __name__ == "__main__":
    main()
