"""Encoder-only raw/time-frequency ablation for VS-to-VI transfer.

Each representation is evaluated with the same four-stage protocol:

1. ``vs_pretrain``: train on the subject's VS train split.
2. ``zero_shot``: load the VS checkpoint and evaluate the VI test split.
3. ``vi_only``: train from scratch on the VI train split.
4. ``vs_to_vi``: initialize from the matching VS checkpoint and fine-tune on VI.

The script deliberately excludes DINO, Stable Diffusion, and LoRA.  Its purpose
is to test whether the EEG input representation itself improves nine-way class
decoding and, specifically, the benefit of VS initialization over VI-only.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_vs_re import VSReDataset, collate_fn, session_counts
from model_128_eegonly_transformer import EEGEncoderV2


N_CLASSES = 9
STAGES = ("vs_pretrain", "zero_shot", "vi_only", "vs_to_vi")
REPRESENTATIONS = ("raw", "tf", "raw_tf")
DEFAULT_BANDS = ((4.0, 7.0), (8.0, 13.0), (14.0, 20.0), (21.0, 30.0), (31.0, 45.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--representation", required=True, choices=REPRESENTATIONS)
    parser.add_argument("--subject_id", required=True, type=int)
    parser.add_argument("--vs_root", required=True)
    parser.add_argument("--vi_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--init_ckpt", default="")
    parser.add_argument("--seed", type=int, default=42)
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
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def filter_nonfinite(dataset: VSReDataset, tag: str) -> int:
    before = len(dataset.samples)
    dataset.samples = [sample for sample in dataset.samples if torch.isfinite(sample[0]).all().item()]
    removed = before - len(dataset.samples)
    if removed:
        print(f"[WARN] {tag}: removed {removed}/{before} non-finite trials", flush=True)
    return removed


def build_loaders(root: str, sid: int, args: argparse.Namespace, domain: str):
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
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
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
    return datasets, loaders, removed


class EEGWaveAugment:
    """Small waveform augmentation shared by all representation conditions."""

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        output = eeg.clone()
        if random.random() < 0.5:
            scale = torch.empty(output.size(0), 1, 1, device=output.device).uniform_(0.9, 1.1)
            output = output * scale
        if random.random() < 0.5:
            std = output.std(dim=-1, keepdim=True).clamp_min(1e-6)
            output = output + torch.randn_like(output) * std * 0.02
        if random.random() < 0.3:
            shift = random.randint(-25, 25)
            output = torch.roll(output, shift, dims=-1)
        if random.random() < 0.2:
            keep = (torch.rand(output.size(0), output.size(1), 1, device=output.device) > 0.05)
            output = output * keep
        return output


class TimeFrequencyBackbone(nn.Module):
    """STFT band-power encoder preserving channel, band, and time axes."""

    def __init__(
        self,
        eeg_channels: int,
        out_dim: int,
        sampling_rate: float,
        n_fft: int,
        hop_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_fft <= 0 or hop_length <= 0:
            raise ValueError("n_fft and hop_length must be positive")
        self.eeg_channels = eeg_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

        frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / sampling_rate)
        weights = []
        for low, high in DEFAULT_BANDS:
            mask = ((frequencies >= low) & (frequencies <= high)).float()
            if mask.sum() == 0:
                raise ValueError(
                    f"No FFT bins for band {low:g}-{high:g} Hz; "
                    f"sampling_rate={sampling_rate}, n_fft={n_fft}"
                )
            weights.append(mask / mask.sum())
        self.register_buffer("band_weights", torch.stack(weights), persistent=True)

        self.net = nn.Sequential(
            nn.Conv2d(eeg_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((len(DEFAULT_BANDS), 4)),
            nn.Flatten(),
            nn.Linear(128 * len(DEFAULT_BANDS) * 4, out_dim),
            nn.LayerNorm(out_dim),
        )

    def band_power(self, eeg: torch.Tensor) -> torch.Tensor:
        batch, channels, time = eeg.shape
        if channels != self.eeg_channels:
            raise ValueError(f"Expected {self.eeg_channels} channels, got {channels}")
        # CUDA STFT does not support all half-precision paths.  Feature
        # extraction remains float32; downstream convolutions may use AMP.
        device_type = eeg.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            x = eeg.float()
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
            spectrum = torch.stft(
                x.reshape(batch * channels, time),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window.float(),
                center=True,
                return_complex=True,
            )
            power = spectrum.abs().square().reshape(batch, channels, -1, spectrum.size(-1))
            bands = torch.einsum("bcft,kf->bckt", power, self.band_weights.float())
            bands = torch.log1p(bands)
            bands = (bands - bands.mean(dim=-1, keepdim=True)) / (
                bands.std(dim=-1, keepdim=True) + 1e-6
            )
        return bands

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.net(self.band_power(eeg))


class RepresentationEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        representation = args.representation
        self.representation = representation

        if representation == "raw_tf":
            raw_out = args.latent_dim // 2
            tf_out = args.latent_dim - raw_out
        else:
            raw_out = args.latent_dim
            tf_out = args.latent_dim

        self.raw = None
        if representation in ("raw", "raw_tf"):
            raw_hidden = args.hidden_dim if representation == "raw" else max(128, args.hidden_dim // 2)
            self.raw = EEGEncoderV2(
                eeg_channels=args.n_ch,
                eeg_hidden_dim=raw_hidden,
                out_dim=raw_out,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                dropout=args.dropout,
            )

        self.tf = None
        if representation in ("tf", "raw_tf"):
            self.tf = TimeFrequencyBackbone(
                eeg_channels=args.n_ch,
                out_dim=tf_out,
                sampling_rate=args.sampling_rate,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                dropout=args.dropout,
            )

        self.fusion = nn.Sequential(
            nn.Linear(args.latent_dim, args.latent_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.LayerNorm(args.latent_dim),
        )
        self.classifier = nn.Linear(args.latent_dim, N_CLASSES)

    def encode(self, eeg: torch.Tensor) -> torch.Tensor:
        features = []
        if self.raw is not None:
            features.append(self.raw(eeg))
        if self.tf is not None:
            features.append(self.tf(eeg))
        latent = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        return F.normalize(self.fusion(latent), dim=1)

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(eeg)
        return self.classifier(latent), latent


def supervised_contrastive_loss(
    latent: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if latent.size(0) < 2:
        return latent.sum() * 0.0
    similarity = latent @ latent.T / temperature
    eye = torch.eye(latent.size(0), dtype=torch.bool, device=latent.device)
    positives = labels[:, None].eq(labels[None, :]) & ~eye
    valid = positives.any(dim=1)
    if not valid.any():
        return latent.sum() * 0.0
    similarity = similarity.masked_fill(eye, -torch.inf)
    log_denominator = torch.logsumexp(similarity, dim=1)
    positive_logits = similarity.masked_fill(~positives, -torch.inf)
    log_numerator = torch.logsumexp(positive_logits, dim=1)
    return -(log_numerator[valid] - log_denominator[valid]).mean()


def prediction_metrics(confusion: np.ndarray, rows: list[dict], topk_correct: dict[int, int]) -> dict:
    total = len(rows)
    recalls = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    prediction_counts = confusion.sum(axis=0)
    probabilities = prediction_counts / max(prediction_counts.sum(), 1)
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum()) if len(nonzero) else 0.0
    dominant = int(prediction_counts.argmax()) if prediction_counts.sum() else 0
    return {
        "n_test": total,
        "top1": topk_correct[1] / max(total, 1),
        "top3": topk_correct[3] / max(total, 1),
        "top5": topk_correct[5] / max(total, 1),
        "balanced_accuracy": float(recalls.mean()),
        "mean_true_margin": float(np.mean([row["true_margin"] for row in rows])) if rows else 0.0,
        "normalized_entropy": entropy / math.log(N_CLASSES),
        "dominant_label": dominant,
        "dominant_ratio": float(prediction_counts[dominant] / max(total, 1)),
        "prediction_counts": prediction_counts.tolist(),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, list[dict], np.ndarray]:
    model.eval()
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    rows = []
    correct = {1: 0, 3: 0, 5: 0}
    sample_index = 0
    for eeg, _, labels in loader:
        eeg = eeg.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        logits, _ = model(eeg)
        predictions = logits.argmax(dim=1)
        for k in correct:
            indices = logits.topk(k, dim=1).indices
            correct[k] += indices.eq(labels_device[:, None]).any(dim=1).sum().item()
        logits_cpu = logits.float().cpu()
        predictions_cpu = predictions.cpu()
        for index in range(len(labels)):
            truth = int(labels[index])
            predicted = int(predictions_cpu[index])
            confusion[truth, predicted] += 1
            other = torch.cat((logits_cpu[index, :truth], logits_cpu[index, truth + 1 :]))
            true_score = float(logits_cpu[index, truth])
            rows.append({
                "sample_index": sample_index,
                "true_label": truth,
                "pred_label": predicted,
                "correct": int(truth == predicted),
                "true_score": true_score,
                "true_margin": true_score - float(other.max()),
            })
            sample_index += 1
    return prediction_metrics(confusion, rows, correct), rows, confusion


def amp_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def train_model(
    model: RepresentationEncoder,
    loaders: dict[str, DataLoader],
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[int, list[dict]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")
    augmenter = EEGWaveAugment()
    best_score = (-1.0, -1.0, -1.0)
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for eeg, _, labels in loaders["train"]:
            eeg = eeg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if not args.no_aug:
                eeg = augmenter(eeg)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, args.fp16):
                logits, latent = model(eeg)
                loss_ce = F.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=args.label_smoothing,
                )
                loss_supcon = supervised_contrastive_loss(latent, labels, args.temperature)
                loss = loss_ce + args.w_supcon * loss_supcon
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * eeg.size(0)
            total_seen += eeg.size(0)
        scheduler.step()

        validation, _, _ = evaluate(model, loaders["val"], device)
        score = (
            validation["balanced_accuracy"],
            validation["top3"],
            validation["top5"],
        )
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(total_seen, 1),
            "val_balanced_accuracy": validation["balanced_accuracy"],
            "val_top1": validation["top1"],
            "val_top3": validation["top3"],
            "val_top5": validation["top5"],
            "val_normalized_entropy": validation["normalized_entropy"],
            "val_dominant_ratio": validation["dominant_ratio"],
        })
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "config": vars(args),
                "best_epoch": best_epoch,
                "best_validation": validation,
            }, checkpoint_path)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  epoch={epoch:03d} loss={history[-1]['train_loss']:.4f} "
                f"val_BAC={validation['balanced_accuracy']:.4f} "
                f"val@3={validation['top3']:.4f} dom={validation['dominant_ratio']:.3f}",
                flush=True,
            )
        if args.patience > 0 and epoch - best_epoch >= args.patience:
            print(f"[Early stop] epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    checkpoint = safe_torch_load(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return best_epoch, history


def write_outputs(
    out_dir: Path,
    metrics: dict,
    rows: list[dict],
    confusion: np.ndarray,
    history: list[dict],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["sample_index"])
        writer.writeheader()
        writer.writerows(rows)
    np.savetxt(out_dir / "confusion.csv", confusion, delimiter=",", fmt="%d")
    if history:
        with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)


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

    datasets, loaders, removed = build_loaders(root, args.subject_id, args, domain)
    print(
        f"[INFO] stage={args.stage} representation={args.representation} "
        f"subject=S{args.subject_id:02d} domain={domain} sessions={counts[args.subject_id]} "
        f"device={device}",
        flush=True,
    )
    print(
        f"[INFO] split train={len(datasets['train'])} val={len(datasets['val'])} "
        f"test={len(datasets['test'])}",
        flush=True,
    )

    model = RepresentationEncoder(args).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    initialized_from_vs = args.stage in ("zero_shot", "vs_to_vi")
    if initialized_from_vs:
        if not args.init_ckpt:
            raise ValueError(f"--init_ckpt is required for {args.stage}")
        checkpoint_path = Path(args.init_ckpt)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = safe_torch_load(checkpoint_path, "cpu")
        checkpoint_config = checkpoint.get("config", {})
        checkpoint_representation = checkpoint_config.get("representation")
        if checkpoint_representation != args.representation:
            raise RuntimeError(
                f"Representation mismatch: checkpoint={checkpoint_representation}, "
                f"requested={args.representation}"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        print(f"[INIT] Loaded matching VS checkpoint: {checkpoint_path}", flush=True)

    history: list[dict] = []
    best_epoch = 0
    trained_checkpoint = output / "encoder_best.pt"
    if args.stage not in ("zero_shot",):
        output.mkdir(parents=True, exist_ok=True)
        best_epoch, history = train_model(
            model,
            loaders,
            args,
            device,
            trained_checkpoint,
        )

    test_metrics, rows, confusion = evaluate(model, loaders["test"], device)
    metrics = {
        "subject": args.subject_id,
        "stage": args.stage,
        "representation": args.representation,
        "domain": domain,
        "seed": args.seed,
        "sampling_rate": args.sampling_rate,
        "n_sessions": counts[args.subject_id],
        "n_train": len(datasets["train"]),
        "n_val": len(datasets["val"]),
        "nonfinite_removed": removed,
        "parameter_count": parameter_count,
        "initialized_from_vs": initialized_from_vs,
        "initial_checkpoint": str(Path(args.init_ckpt).resolve()) if args.init_ckpt else None,
        "best_epoch": best_epoch,
        **test_metrics,
        "config": vars(args),
    }
    write_outputs(output, metrics, rows, confusion, history)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"[Saved] {output}", flush=True)


if __name__ == "__main__":
    main()
