"""Locate the class-information bottleneck before Exp43 image generation.

This diagnostic does not load Stable Diffusion, the VAE, or DINO.  It compares
class separability at two points using the fixed Exp43 VI train/test split:

1. the frozen SupCon encoder output (``eeg_lat``), and
2. the SD cross-attention tokens produced by the trained ``cond_proj``.

It also evaluates the encoder's VS-trained auxiliary classifier on VI test EEG.
Nearest-centroid results use VI *train* labels and are diagnostic probes, not
end-to-end test results.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import train_exp43_vi_lora as exp43
from train_vs_re_latent_gen import build_eeg_encoder
from train_vs_re_lora_gen import EEGConditionProjector


CLASS_NAMES = (
    "airplane",
    "cup",
    "tree",
    "digit1",
    "digit3",
    "digit5",
    "heart",
    "star",
    "triangle",
)
N_CLASSES = len(CLASS_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Exp43 C0/C1 checkpoint")
    parser.add_argument(
        "--supcon_ckpt", required=True, help="directory containing subjNN_best.pt"
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--subject_id", type=int, required=True)
    parser.add_argument("--n_eeg_tokens", type=int, default=16)
    parser.add_argument("--per_class_total", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--out_root",
        default="/content/drive/MyDrive/vsvi_data/conditioning_diagnostic",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_file(path: str | Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def infer_condition(path: str) -> str:
    lowered = path.lower()
    if "exp43_c1" in lowered or "_c1_" in lowered:
        return "c1"
    if "exp43_c0" in lowered or "_c0_" in lowered:
        return "c0"
    return "unknown"


def load_models(args: argparse.Namespace, device: torch.device):
    encoder_path = require_file(
        Path(args.supcon_ckpt) / f"subj{args.subject_id:02d}_best.pt",
        "SupCon checkpoint",
    )
    encoder_payload = torch.load(
        encoder_path, map_location=device, weights_only=False
    )
    encoder_key = "eeg_enc" if "eeg_enc" in encoder_payload else "model"
    encoder = build_eeg_encoder(
        32,
        384,
        SimpleNamespace(eeg_occipital_ids="auto"),
        device,
    )
    encoder.load_state_dict(encoder_payload[encoder_key], strict=True)
    encoder.eval()

    generator_path = require_file(args.ckpt, "Exp43 checkpoint")
    generator_payload = torch.load(
        generator_path, map_location="cpu", weights_only=False
    )
    if "cond_proj" not in generator_payload:
        raise KeyError("Exp43 checkpoint does not contain cond_proj")
    cond_proj = EEGConditionProjector(
        eeg_dim=512,
        sd_dim=768,
        n_tokens=args.n_eeg_tokens,
        deep=False,
    ).to(device)
    cond_proj.load_state_dict(generator_payload["cond_proj"], strict=True)
    cond_proj.eval()

    for model in (encoder, cond_proj):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    print(f"[Encoder] {encoder_path} (key={encoder_key})", flush=True)
    print(f"[Condition projector] {generator_path}", flush=True)
    return encoder, cond_proj, encoder_path, generator_path


@torch.no_grad()
def encode_dataset(dataset, encoder, cond_proj, batch_size: int, device):
    latent_parts = []
    token_parts = []
    logit_parts = []
    label_parts = []
    subject_index = torch.zeros(1, dtype=torch.long, device=device)

    for start in range(0, len(dataset), batch_size):
        batch = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        eeg = torch.stack([item[0] for item in batch]).to(device)
        labels = torch.tensor([int(item[2]) for item in batch], dtype=torch.long)
        subject_ids = subject_index.expand(eeg.size(0))
        latent = encoder.encode_eeg(eeg, subject_ids)
        tokens = cond_proj(latent)
        logits = encoder.aux_cls_head(latent)
        latent_parts.append(latent.cpu())
        token_parts.append(tokens.cpu())
        logit_parts.append(logits.cpu())
        label_parts.append(labels)

    return {
        "latent": torch.cat(latent_parts),
        "tokens": torch.cat(token_parts),
        "aux_logits": torch.cat(logit_parts),
        "labels": torch.cat(label_parts),
    }


def topk_metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict:
    order = scores.argsort(dim=1, descending=True)
    predictions = order[:, 0]
    recalls = []
    for label in range(N_CLASSES):
        mask = labels == label
        recalls.append(float((predictions[mask] == label).float().mean().item()))
    return {
        "top1": float((predictions == labels).float().mean().item()),
        "top3": float((order[:, :3] == labels[:, None]).any(dim=1).float().mean().item()),
        "top5": float((order[:, :5] == labels[:, None]).any(dim=1).float().mean().item()),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class_recall": {
            CLASS_NAMES[label]: recalls[label] for label in range(N_CLASSES)
        },
    }


def normalize_rows(features: torch.Tensor) -> torch.Tensor:
    if features.ndim > 2:
        features = features.flatten(1)
    return F.normalize(features.float(), dim=1)


def class_centroids(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    centroids = []
    for label in range(N_CLASSES):
        selected = features[labels == label]
        if selected.numel() == 0:
            raise RuntimeError(f"no samples for class {label}")
        centroids.append(F.normalize(selected.mean(dim=0), dim=0))
    return torch.stack(centroids)


def pairwise_separation(features: torch.Tensor, labels: torch.Tensor) -> dict:
    similarities = features @ features.T
    n = len(labels)
    diagonal = torch.eye(n, dtype=torch.bool)
    same = labels[:, None].eq(labels[None, :]) & ~diagonal
    different = ~labels[:, None].eq(labels[None, :])
    within = float(similarities[same].mean().item())
    between = float(similarities[different].mean().item())
    return {
        "mean_within_class_cosine": within,
        "mean_between_class_cosine": between,
        "within_minus_between_cosine": within - between,
    }


def representation_metrics(
    train_raw: torch.Tensor,
    train_labels: torch.Tensor,
    test_raw: torch.Tensor,
    test_labels: torch.Tensor,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    train_features = normalize_rows(train_raw)
    test_features = normalize_rows(test_raw)
    centroids = class_centroids(train_features, train_labels)
    scores = test_features @ centroids.T
    metrics = topk_metrics(scores, test_labels)

    true_scores = scores.gather(1, test_labels[:, None]).squeeze(1)
    other_scores = scores.clone()
    other_scores[torch.arange(len(test_labels)), test_labels] = -float("inf")
    metrics["mean_true_centroid_similarity"] = float(true_scores.mean().item())
    metrics["mean_true_centroid_margin"] = float(
        (true_scores - other_scores.max(dim=1).values).mean().item()
    )
    metrics.update(pairwise_separation(test_features, test_labels))

    centroid_similarity = centroids @ centroids.T
    off_diagonal = ~torch.eye(N_CLASSES, dtype=torch.bool)
    metrics["mean_offdiag_centroid_cosine"] = float(
        centroid_similarity[off_diagonal].mean().item()
    )
    metrics["max_offdiag_centroid_cosine"] = float(
        centroid_similarity[off_diagonal].max().item()
    )

    train_flat = train_raw.flatten(1).float()
    train_norm = train_flat.norm(dim=1).mean()
    global_mean = train_flat.mean(dim=0)
    metrics["global_mean_norm_ratio"] = float(
        (global_mean.norm() / train_norm.clamp_min(1e-12)).item()
    )
    global_direction = F.normalize(global_mean, dim=0)
    metrics["mean_cosine_to_global_mean"] = float(
        (normalize_rows(train_flat) @ global_direction).mean().item()
    )
    return metrics, scores, centroid_similarity


def projector_common_component_metrics(
    cond_proj, latent: torch.Tensor, tokens: torch.Tensor, device: torch.device
) -> dict:
    with torch.no_grad():
        zero_tokens = cond_proj(torch.zeros((1, latent.shape[1]), device=device)).cpu()
    token_flat = tokens.flatten(1).float()
    zero_flat = zero_tokens.flatten(1).float()
    zero_direction = F.normalize(zero_flat, dim=1)
    cosine = normalize_rows(token_flat) @ zero_direction.T
    delta = token_flat - zero_flat
    return {
        "mean_cosine_to_projected_zero_latent": float(cosine.mean().item()),
        "mean_delta_from_projected_zero_ratio": float(
            (delta.norm(dim=1) / token_flat.norm(dim=1).clamp_min(1e-12)).mean().item()
        ),
    }


def save_centroid_heatmaps(path: Path, latent_matrix, token_matrix) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, matrix, title in zip(
        axes,
        (latent_matrix, token_matrix),
        ("EEG latent train-centroid cosine", "Condition-token train-centroid cosine"),
    ):
        image = ax.imshow(matrix.numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(title)
        ax.set_xticks(range(N_CLASSES), CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(N_CLASSES), CLASS_NAMES)
        for row in range(N_CLASSES):
            for column in range(N_CLASSES):
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_prediction_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    condition = infer_condition(args.ckpt)
    run_dir = (
        Path(args.out_root)
        / f"S{args.subject_id:02d}"
        / condition
        / "condition_space"
    )
    metrics_path = run_dir / "condition_space_metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite to rerun")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] device={device} S{args.subject_id:02d} {condition}", flush=True)
    encoder, cond_proj, encoder_path, generator_path = load_models(args, device)
    per_class_total = args.per_class_total if args.per_class_total > 0 else None
    train_dataset = exp43.SubjectClassLimitedDataset(
        data_root=args.data_root,
        sid=args.subject_id,
        split="train",
        seed=args.split_seed,
        per_class_total=per_class_total,
    )
    test_dataset = exp43.SubjectClassLimitedDataset(
        data_root=args.data_root,
        sid=args.subject_id,
        split="test",
        seed=args.split_seed,
        per_class_total=per_class_total,
    )
    train = encode_dataset(train_dataset, encoder, cond_proj, args.batch_size, device)
    test = encode_dataset(test_dataset, encoder, cond_proj, args.batch_size, device)

    latent_metrics, latent_scores, latent_matrix = representation_metrics(
        train["latent"], train["labels"], test["latent"], test["labels"]
    )
    token_metrics, token_scores, token_matrix = representation_metrics(
        train["tokens"], train["labels"], test["tokens"], test["labels"]
    )
    token_metrics.update(
        projector_common_component_metrics(
            cond_proj, test["latent"], test["tokens"], device
        )
    )
    aux_metrics = topk_metrics(test["aux_logits"], test["labels"])

    latent_predictions = latent_scores.argmax(dim=1)
    token_predictions = token_scores.argmax(dim=1)
    aux_predictions = test["aux_logits"].argmax(dim=1)
    prediction_rows = []
    for index in range(len(test["labels"])):
        true_label = int(test["labels"][index].item())
        prediction_rows.append(
            {
                "test_index": index,
                "true_label": true_label + 1,
                "true_name": CLASS_NAMES[true_label],
                "aux_pred_label": int(aux_predictions[index].item()) + 1,
                "aux_pred_name": CLASS_NAMES[int(aux_predictions[index].item())],
                "latent_centroid_pred_label": int(latent_predictions[index].item()) + 1,
                "latent_centroid_pred_name": CLASS_NAMES[int(latent_predictions[index].item())],
                "token_centroid_pred_label": int(token_predictions[index].item()) + 1,
                "token_centroid_pred_name": CLASS_NAMES[int(token_predictions[index].item())],
            }
        )

    metrics = {
        "subject": args.subject_id,
        "condition": condition,
        "n_train": len(train_dataset),
        "n_test": len(test_dataset),
        "split_seed": args.split_seed,
        "per_class_total": args.per_class_total,
        "checkpoint": str(generator_path),
        "supcon_checkpoint": str(encoder_path),
        "aux_head_vs_to_vi_zero_shot": aux_metrics,
        "vi_train_nearest_centroid": {
            "eeg_latent": latent_metrics,
            "condition_tokens": token_metrics,
        },
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    write_prediction_csv(run_dir / "test_predictions.csv", prediction_rows)
    save_centroid_heatmaps(
        run_dir / "class_centroid_cosine_heatmaps.png",
        latent_matrix,
        token_matrix,
    )

    compact = {
        "aux_top1": aux_metrics["top1"],
        "latent_centroid_top1": latent_metrics["top1"],
        "token_centroid_top1": token_metrics["top1"],
        "latent_within_minus_between": latent_metrics[
            "within_minus_between_cosine"
        ],
        "token_within_minus_between": token_metrics[
            "within_minus_between_cosine"
        ],
        "latent_global_mean_norm_ratio": latent_metrics[
            "global_mean_norm_ratio"
        ],
        "token_global_mean_norm_ratio": token_metrics["global_mean_norm_ratio"],
        "token_cosine_to_projected_zero": token_metrics[
            "mean_cosine_to_projected_zero_latent"
        ],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    print(f"[Saved] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
