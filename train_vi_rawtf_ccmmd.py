"""Joint raw+TF VS/VI training with class-conditional MMD.

This experiment isolates two effects after a matching raw+TF VS checkpoint:

* ``replay``: joint VI training and VS replay with no alignment loss.
* ``ccmmd``: the identical objective plus class-conditional multi-kernel MMD.

For a development subject, ``ccmmd`` trains each prespecified lambda, selects
using VI validation balanced accuracy (Top-3 tie-break), and evaluates the VI
test split only once with the selected checkpoint.  Confirmatory subjects use
one fixed lambda selected on the development subject.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from dataset_vs_re import collate_fn, session_counts
from train_vi_tf_representation_ablation import (
    N_CLASSES,
    EEGWaveAugment,
    RepresentationEncoder,
    amp_context,
    build_loaders,
    evaluate,
    safe_torch_load,
    set_seed,
    supervised_contrastive_loss,
    write_outputs,
)


MODES = ("replay", "ccmmd")
MMD_SCALES = (0.5, 1.0, 2.0, 4.0)


def parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(sorted(set(float(value.strip()) for value in raw.split(",") if value.strip())))
    if not values:
        raise ValueError("At least one MMD lambda is required")
    if any(value <= 0 for value in values):
        raise ValueError("CC-MMD lambdas must be positive; replay is the lambda=0 control")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--subject_id", required=True, type=int)
    parser.add_argument("--vs_root", required=True)
    parser.add_argument("--vi_root", required=True)
    parser.add_argument("--vs_ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_candidates", default="0.01,0.05,0.1")
    parser.add_argument(
        "--fixed_lambda",
        type=float,
        default=-1.0,
        help="Positive value for confirmatory evaluation; negative tunes on validation.",
    )
    parser.add_argument("--vs_replay_weight", type=float, default=0.5)
    parser.add_argument("--samples_per_class", type=int, default=2)
    parser.add_argument("--sampling_rate", type=float, default=1024.0)
    parser.add_argument("--n_ch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--w_supcon", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--n_fft", type=int, default=256)
    parser.add_argument("--hop_length", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    # RepresentationEncoder consumes this field.  It is intentionally fixed.
    args.representation = "raw_tf"
    return args


class ClassBalancedBatchSampler(Sampler[list[int]]):
    """Deterministic class-balanced batches with reshuffling and wraparound."""

    def __init__(self, dataset, samples_per_class: int, seed: int) -> None:
        if samples_per_class < 2:
            raise ValueError("samples_per_class must be >=2 for within-domain SupCon")
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.epoch = 0
        self.by_class = {
            cls: [index for index, sample in enumerate(dataset.samples) if int(sample[2]) == cls]
            for cls in range(N_CLASSES)
        }
        missing = [cls for cls, indices in self.by_class.items() if not indices]
        if missing:
            raise RuntimeError(f"Balanced sampler is missing classes: {missing}")
        self.batch_size = N_CLASSES * samples_per_class
        self.n_batches = max(1, math.ceil(len(dataset) / self.batch_size))

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch * 1009)
        self.epoch += 1
        queues = {}
        positions = {}
        for cls, indices in self.by_class.items():
            queues[cls] = np.asarray(indices, dtype=np.int64)[rng.permutation(len(indices))]
            positions[cls] = 0

        for _ in range(self.n_batches):
            batch = []
            for cls in range(N_CLASSES):
                for _ in range(self.samples_per_class):
                    if positions[cls] >= len(queues[cls]):
                        indices = np.asarray(self.by_class[cls], dtype=np.int64)
                        queues[cls] = indices[rng.permutation(len(indices))]
                        positions[cls] = 0
                    batch.append(int(queues[cls][positions[cls]]))
                    positions[cls] += 1
            rng.shuffle(batch)
            yield batch


def balanced_loader(dataset, args: argparse.Namespace, seed_offset: int) -> DataLoader:
    sampler = ClassBalancedBatchSampler(
        dataset,
        samples_per_class=args.samples_per_class,
        seed=args.seed + seed_offset,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def next_batch(loader: DataLoader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def multi_kernel_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Biased non-negative RBF MMD with a detached median bandwidth."""
    x = F.normalize(x.float(), dim=1)
    y = F.normalize(y.float(), dim=1)
    combined = torch.cat((x, y), dim=0)
    distances = torch.cdist(combined, combined, p=2).square()
    positive = distances.detach()[distances.detach() > 1e-12]
    bandwidth = positive.median() if positive.numel() else distances.new_tensor(1.0)
    bandwidth = bandwidth.clamp_min(1e-4)
    kernel = sum(
        torch.exp(-distances / (2.0 * bandwidth * scale))
        for scale in MMD_SCALES
    ) / len(MMD_SCALES)
    n_x = x.size(0)
    return (
        kernel[:n_x, :n_x].mean()
        + kernel[n_x:, n_x:].mean()
        - 2.0 * kernel[:n_x, n_x:].mean()
    ).clamp_min(0.0)


def class_conditional_mmd(
    vs_latent: torch.Tensor,
    vs_labels: torch.Tensor,
    vi_latent: torch.Tensor,
    vi_labels: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for cls in range(N_CLASSES):
        vs_mask = vs_labels.eq(cls)
        vi_mask = vi_labels.eq(cls)
        if vs_mask.any() and vi_mask.any():
            losses.append(multi_kernel_mmd(vs_latent[vs_mask], vi_latent[vi_mask]))
    if not losses:
        return (vs_latent.sum() + vi_latent.sum()) * 0.0
    return torch.stack(losses).mean()


def load_vs_initialized_model(args: argparse.Namespace, device: torch.device) -> RepresentationEncoder:
    checkpoint_path = Path(args.vs_ckpt)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = safe_torch_load(checkpoint_path, "cpu")
    config = checkpoint.get("config", {})
    if config.get("representation") != "raw_tf":
        raise RuntimeError(
            f"Expected raw_tf VS checkpoint, found {config.get('representation')!r}: {checkpoint_path}"
        )
    model = RepresentationEncoder(args).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def write_history(path: Path, history: list[dict]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train_candidate(
    args: argparse.Namespace,
    mmd_lambda: float,
    vs_dataset,
    vi_dataset,
    vi_val_loader: DataLoader,
    device: torch.device,
    candidate_dir: Path,
) -> dict:
    # Reset all stochastic state so lambda candidates differ only in the MMD weight.
    set_seed(args.seed)
    model = load_vs_initialized_model(args, device)
    vs_loader = balanced_loader(vs_dataset, args, seed_offset=17)
    vi_loader = balanced_loader(vi_dataset, args, seed_offset=29)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")
    augmenter = EEGWaveAugment()
    checkpoint_path = candidate_dir / "encoder_best.pt"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    best_score = (-1.0, -1.0)
    best_epoch = 0
    best_validation = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        vs_iterator = iter(vs_loader)
        vi_iterator = iter(vi_loader)
        steps = max(len(vs_loader), len(vi_loader))
        totals = {name: 0.0 for name in ("loss", "vi_ce", "vs_ce", "supcon", "mmd")}

        for _ in range(steps):
            (vs_eeg, _, vs_labels), vs_iterator = next_batch(vs_loader, vs_iterator)
            (vi_eeg, _, vi_labels), vi_iterator = next_batch(vi_loader, vi_iterator)
            vs_eeg = vs_eeg.to(device, non_blocking=True)
            vi_eeg = vi_eeg.to(device, non_blocking=True)
            vs_labels = vs_labels.to(device, non_blocking=True)
            vi_labels = vi_labels.to(device, non_blocking=True)
            if not args.no_aug:
                vs_eeg = augmenter(vs_eeg)
                vi_eeg = augmenter(vi_eeg)

            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, args.fp16):
                combined_eeg = torch.cat((vi_eeg, vs_eeg), dim=0)
                combined_logits, combined_latent = model(combined_eeg)
                n_vi = vi_eeg.size(0)
                vi_logits, vs_logits = combined_logits[:n_vi], combined_logits[n_vi:]
                vi_latent, vs_latent = combined_latent[:n_vi], combined_latent[n_vi:]
                vi_ce = F.cross_entropy(
                    vi_logits, vi_labels, label_smoothing=args.label_smoothing
                )
                vs_ce = F.cross_entropy(
                    vs_logits, vs_labels, label_smoothing=args.label_smoothing
                )
                vi_supcon = supervised_contrastive_loss(
                    vi_latent, vi_labels, args.temperature
                )
                vs_supcon = supervised_contrastive_loss(
                    vs_latent, vs_labels, args.temperature
                )
                supcon = vi_supcon + args.vs_replay_weight * vs_supcon
                mmd = class_conditional_mmd(vs_latent, vs_labels, vi_latent, vi_labels)
                loss = (
                    vi_ce
                    + args.vs_replay_weight * vs_ce
                    + args.w_supcon * supcon
                    + mmd_lambda * mmd
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach())
            totals["vi_ce"] += float(vi_ce.detach())
            totals["vs_ce"] += float(vs_ce.detach())
            totals["supcon"] += float(supcon.detach())
            totals["mmd"] += float(mmd.detach())
        scheduler.step()

        validation, _, _ = evaluate(model, vi_val_loader, device)
        score = (validation["balanced_accuracy"], validation["top3"])
        row = {
            "epoch": epoch,
            **{f"train_{name}": value / steps for name, value in totals.items()},
            "val_balanced_accuracy": validation["balanced_accuracy"],
            "val_top1": validation["top1"],
            "val_top3": validation["top3"],
            "val_top5": validation["top5"],
            "val_normalized_entropy": validation["normalized_entropy"],
            "val_dominant_ratio": validation["dominant_ratio"],
        }
        history.append(row)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_validation = validation
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": vars(args),
                    "mode": args.mode,
                    "mmd_lambda": mmd_lambda,
                    "best_epoch": best_epoch,
                    "best_validation": validation,
                },
                checkpoint_path,
            )
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  lambda={mmd_lambda:g} epoch={epoch:03d} "
                f"loss={row['train_loss']:.4f} mmd={row['train_mmd']:.4f} "
                f"VI-val BAC={validation['balanced_accuracy']:.4f} "
                f"@3={validation['top3']:.4f}",
                flush=True,
            )
        if args.patience > 0 and epoch - best_epoch >= args.patience:
            print(
                f"[Early stop] lambda={mmd_lambda:g} epoch={epoch} best={best_epoch}",
                flush=True,
            )
            break

    write_history(candidate_dir / "history.csv", history)
    if best_validation is None:
        raise RuntimeError(f"No validation checkpoint produced for lambda={mmd_lambda}")
    return {
        "lambda": mmd_lambda,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "checkpoint": str(checkpoint_path.resolve()),
        "history": history,
    }


def lambda_tag(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.out_dir)
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite to rerun")
    if args.vs_replay_weight < 0:
        raise ValueError("vs_replay_weight must be non-negative")

    vs_counts = session_counts(args.vs_root)
    vi_counts = session_counts(args.vi_root)
    sid = args.subject_id
    if sid not in vs_counts or sid not in vi_counts:
        raise FileNotFoundError(f"S{sid:02d} must exist in both VS and VI roots")

    vs_datasets, _, vs_removed = build_loaders(args.vs_root, sid, args, "vs")
    vi_datasets, vi_loaders, vi_removed = build_loaders(args.vi_root, sid, args, "vi")
    if args.mode == "replay":
        lambdas = (0.0,)
    elif args.fixed_lambda > 0:
        lambdas = (float(args.fixed_lambda),)
    else:
        lambdas = parse_float_list(args.lambda_candidates)

    print(
        f"[INFO] S{sid:02d} mode={args.mode} raw_tf device={device} "
        f"lambdas={list(lambdas)}",
        flush=True,
    )
    print(
        f"[INFO] class-balanced domain batch={N_CLASSES * args.samples_per_class}; "
        f"VI selection only, test evaluated after lambda selection",
        flush=True,
    )

    candidates = []
    for mmd_lambda in lambdas:
        candidate_dir = output / "candidates" / f"lambda_{lambda_tag(mmd_lambda)}"
        candidates.append(
            train_candidate(
                args,
                mmd_lambda,
                vs_datasets["train"],
                vi_datasets["train"],
                vi_loaders["val"],
                device,
                candidate_dir,
            )
        )

    # Candidates are ordered by lambda; max keeps the smaller lambda on an exact tie.
    selected = max(
        candidates,
        key=lambda item: (
            item["best_validation"]["balanced_accuracy"],
            item["best_validation"]["top3"],
            -item["lambda"],
        ),
    )
    selected_checkpoint = Path(selected["checkpoint"])
    output.mkdir(parents=True, exist_ok=True)
    final_checkpoint = output / "encoder_best.pt"
    shutil.copyfile(selected_checkpoint, final_checkpoint)

    model = RepresentationEncoder(args).to(device)
    checkpoint = safe_torch_load(final_checkpoint, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    test_metrics, rows, confusion = evaluate(model, vi_loaders["test"], device)
    candidate_validation = [
        {
            "lambda": item["lambda"],
            "best_epoch": item["best_epoch"],
            "balanced_accuracy": item["best_validation"]["balanced_accuracy"],
            "top3": item["best_validation"]["top3"],
            "top5": item["best_validation"]["top5"],
            "dominant_ratio": item["best_validation"]["dominant_ratio"],
        }
        for item in candidates
    ]
    metrics = {
        "subject": sid,
        "stage": args.mode,
        "mode": args.mode,
        "representation": "raw_tf",
        "seed": args.seed,
        "vs_sessions": vs_counts[sid],
        "vi_sessions": vi_counts[sid],
        "n_vs_train": len(vs_datasets["train"]),
        "n_vi_train": len(vi_datasets["train"]),
        "n_vi_val": len(vi_datasets["val"]),
        "nonfinite_removed_vs": vs_removed,
        "nonfinite_removed_vi": vi_removed,
        "initialized_from_vs": True,
        "initial_checkpoint": str(Path(args.vs_ckpt).resolve()),
        "selection_domain": "VI validation",
        "selection_primary": "balanced_accuracy",
        "selection_tie_break": "top3_then_smaller_lambda",
        "selected_lambda": selected["lambda"],
        "best_epoch": selected["best_epoch"],
        "best_validation": selected["best_validation"],
        "candidate_validation": candidate_validation,
        "test_evaluations": 1,
        **test_metrics,
        "config": vars(args),
    }
    write_outputs(output, metrics, rows, confusion, selected["history"])
    with (output / "validation_candidates.json").open("w", encoding="utf-8") as handle:
        json.dump(candidate_validation, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"[Saved] {output}", flush=True)


if __name__ == "__main__":
    main()
