"""Train a small VI adapter into a frozen subject-specific VS condition space.

The source SupCon encoder, its auxiliary classifier, and the original VS LoRA
condition projector stay frozen.  Only a zero-initialized residual MLP is
trained on VI train trials.  VI validation selects the checkpoint; VI test is
evaluated once after selection.

The adapter is supervised by three fixed VS teachers:

* VS EEG latent class centroids,
* VS SD-condition-token class centroids, and
* the VS-trained auxiliary classifier.

Stable Diffusion is not loaded by this script.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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


class ResidualVIAdapter(nn.Module):
    """Zero-initialized latent correction; the initial model is identity."""

    def __init__(self, dim: int = 512, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return F.normalize(latent + self.net(latent), dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vs_lora_ckpt", required=True)
    parser.add_argument(
        "--supcon_ckpt", required=True, help="directory containing subjNN_best.pt"
    )
    parser.add_argument("--vs_root", required=True)
    parser.add_argument("--vi_root", required=True)
    parser.add_argument("--subject_id", type=int, required=True)
    parser.add_argument("--n_eeg_tokens", type=int, default=16)
    parser.add_argument("--vi_per_class_total", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--adapter_hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--w_latent", type=float, default=1.0)
    parser.add_argument("--w_token", type=float, default=1.0)
    parser.add_argument("--w_aux", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_root",
        default="/content/drive/MyDrive/vsvi_data/vi_to_vs_adapter",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_file(path: str | Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_frozen_source(args: argparse.Namespace, device: torch.device):
    encoder_path = require_file(
        Path(args.supcon_ckpt) / f"subj{args.subject_id:02d}_best.pt",
        "SupCon checkpoint",
    )
    payload = torch.load(encoder_path, map_location=device, weights_only=False)
    key = "eeg_enc" if "eeg_enc" in payload else "model"
    encoder = build_eeg_encoder(
        32,
        384,
        SimpleNamespace(eeg_occipital_ids="auto"),
        device,
    )
    encoder.load_state_dict(payload[key], strict=True)
    encoder.eval()

    lora_path = require_file(args.vs_lora_ckpt, "VS LoRA checkpoint")
    lora_payload = torch.load(lora_path, map_location="cpu", weights_only=False)
    if "cond_proj" not in lora_payload:
        raise KeyError("VS LoRA checkpoint does not contain cond_proj")
    cond_proj = EEGConditionProjector(
        eeg_dim=512,
        sd_dim=768,
        n_tokens=args.n_eeg_tokens,
        deep=False,
    ).to(device)
    cond_proj.load_state_dict(lora_payload["cond_proj"], strict=True)
    cond_proj.eval()

    for model in (encoder, cond_proj):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    print(f"[Frozen VS encoder] {encoder_path} (key={key})", flush=True)
    print(f"[Frozen VS condition projector] {lora_path}", flush=True)
    return encoder, cond_proj, encoder_path, lora_path


def make_dataset(root: str, sid: int, split: str, seed: int, cap):
    return exp43.SubjectClassLimitedDataset(
        data_root=root,
        sid=sid,
        split=split,
        seed=seed,
        per_class_total=cap,
    )


@torch.no_grad()
def encode_dataset(dataset, encoder, batch_size: int, device: torch.device):
    latent_parts = []
    label_parts = []
    subject_index = torch.zeros(1, dtype=torch.long, device=device)
    for start in range(0, len(dataset), batch_size):
        batch = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        eeg = torch.stack([item[0] for item in batch]).to(device)
        labels = torch.tensor([int(item[2]) for item in batch], dtype=torch.long)
        latent = encoder.encode_eeg(eeg, subject_index.expand(eeg.size(0)))
        latent_parts.append(latent.cpu())
        label_parts.append(labels)
    return torch.cat(latent_parts), torch.cat(label_parts)


def normalized_centroids(features: torch.Tensor, labels: torch.Tensor):
    features = F.normalize(features.float(), dim=1)
    centroids = []
    for label in range(N_CLASSES):
        selected = features[labels == label]
        if selected.numel() == 0:
            raise RuntimeError(f"source train has no samples for class {label}")
        centroids.append(F.normalize(selected.mean(dim=0), dim=0))
    return torch.stack(centroids)


@torch.no_grad()
def build_source_centroids(vs_latent, vs_labels, cond_proj, device):
    latent_centroids = normalized_centroids(vs_latent, vs_labels)
    token_parts = []
    for start in range(0, len(vs_latent), 128):
        latent = vs_latent[start : start + 128].to(device)
        token_parts.append(cond_proj(latent).flatten(1).cpu())
    token_features = torch.cat(token_parts)
    token_centroids = normalized_centroids(token_features, vs_labels)
    return latent_centroids, token_centroids


def classification_metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict:
    order = scores.argsort(dim=1, descending=True)
    predictions = order[:, 0]
    recalls = []
    for label in range(N_CLASSES):
        mask = labels == label
        recalls.append(float((predictions[mask] == label).float().mean().item()))
    return {
        "top1": float((predictions == labels).float().mean().item()),
        "top3": float((order[:, :3] == labels[:, None]).any(1).float().mean().item()),
        "top5": float((order[:, :5] == labels[:, None]).any(1).float().mean().item()),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class_recall": {
            CLASS_NAMES[label]: recalls[label] for label in range(N_CLASSES)
        },
    }


def pairwise_separation(features: torch.Tensor, labels: torch.Tensor) -> float:
    features = F.normalize(features.float(), dim=1)
    similarities = features @ features.T
    diagonal = torch.eye(len(labels), dtype=torch.bool)
    same = labels[:, None].eq(labels[None, :]) & ~diagonal
    different = ~labels[:, None].eq(labels[None, :])
    return float(
        (similarities[same].mean() - similarities[different].mean()).item()
    )


def model_outputs(adapter, latent, cond_proj, aux_head, lat_cent, tok_cent, device):
    adapted_parts = []
    token_parts = []
    aux_parts = []
    for start in range(0, len(latent), 128):
        raw = latent[start : start + 128].to(device)
        adapted = adapter(raw)
        tokens = F.normalize(cond_proj(adapted).flatten(1), dim=1)
        adapted_parts.append(adapted.cpu())
        token_parts.append(tokens.cpu())
        aux_parts.append(aux_head(adapted).cpu())
    adapted = torch.cat(adapted_parts)
    tokens = torch.cat(token_parts)
    aux_logits = torch.cat(aux_parts)
    latent_scores = F.normalize(adapted, dim=1) @ lat_cent.T
    token_scores = F.normalize(tokens, dim=1) @ tok_cent.T
    return adapted, tokens, aux_logits, latent_scores, token_scores


@torch.no_grad()
def evaluate(adapter, latent, labels, cond_proj, aux_head, lat_cent, tok_cent, device):
    adapter.eval()
    adapted, tokens, aux_logits, latent_scores, token_scores = model_outputs(
        adapter, latent, cond_proj, aux_head, lat_cent, tok_cent, device
    )
    return {
        "aux_head": classification_metrics(aux_logits, labels),
        "vs_latent_centroid": classification_metrics(latent_scores, labels),
        "vs_token_centroid": classification_metrics(token_scores, labels),
        "latent_within_minus_between": pairwise_separation(adapted, labels),
        "token_within_minus_between": pairwise_separation(tokens, labels),
        "mean_cosine_raw_to_adapted": float(
            F.cosine_similarity(F.normalize(latent, dim=1), adapted, dim=1)
            .mean()
            .item()
        ),
    }, (adapted, tokens, aux_logits, latent_scores, token_scores)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.out_root) / f"S{args.subject_id:02d}" / f"seed{args.seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite to rerun")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] device={device} S{args.subject_id:02d} seed={args.seed}")

    encoder, cond_proj, encoder_path, lora_path = load_frozen_source(args, device)
    vs_train = make_dataset(args.vs_root, args.subject_id, "train", args.seed, None)
    vi_cap = args.vi_per_class_total if args.vi_per_class_total > 0 else None
    vi_train = make_dataset(args.vi_root, args.subject_id, "train", args.seed, vi_cap)
    vi_val = make_dataset(args.vi_root, args.subject_id, "val", args.seed, vi_cap)
    vi_test = make_dataset(args.vi_root, args.subject_id, "test", args.seed, vi_cap)

    print("[INFO] Encoding fixed datasets...", flush=True)
    vs_latent, vs_labels = encode_dataset(
        vs_train, encoder, args.batch_size, device
    )
    vi_train_latent, vi_train_labels = encode_dataset(
        vi_train, encoder, args.batch_size, device
    )
    vi_val_latent, vi_val_labels = encode_dataset(
        vi_val, encoder, args.batch_size, device
    )
    vi_test_latent, vi_test_labels = encode_dataset(
        vi_test, encoder, args.batch_size, device
    )
    latent_centroids, token_centroids = build_source_centroids(
        vs_latent, vs_labels, cond_proj, device
    )
    latent_centroids = latent_centroids.to(device)
    token_centroids = token_centroids.to(device)
    aux_head = copy.deepcopy(encoder.aux_cls_head).to(device).eval()
    for parameter in aux_head.parameters():
        parameter.requires_grad_(False)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    adapter = ResidualVIAdapter(
        dim=512, hidden=args.adapter_hidden, dropout=args.dropout
    ).to(device)
    raw_adapter = ResidualVIAdapter(
        dim=512, hidden=args.adapter_hidden, dropout=args.dropout
    ).to(device)
    raw_adapter.load_state_dict(adapter.state_dict())
    raw_metrics, _ = evaluate(
        raw_adapter,
        vi_test_latent,
        vi_test_labels,
        cond_proj,
        aux_head,
        latent_centroids.cpu(),
        token_centroids.cpu(),
        device,
    )

    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(vi_train_latent, vi_train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        loss_sum = 0.0
        count = 0
        for raw_cpu, labels_cpu in train_loader:
            raw = raw_cpu.to(device)
            labels = labels_cpu.to(device)
            adapted = adapter(raw)
            token_features = F.normalize(cond_proj(adapted).flatten(1), dim=1)
            latent_logits = adapted @ latent_centroids.T / args.temperature
            token_logits = token_features @ token_centroids.T / args.temperature
            aux_logits = aux_head(adapted)
            loss_latent = F.cross_entropy(latent_logits, labels)
            loss_token = F.cross_entropy(token_logits, labels)
            loss_aux = F.cross_entropy(aux_logits, labels)
            loss = (
                args.w_latent * loss_latent
                + args.w_token * loss_token
                + args.w_aux * loss_aux
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
            count += len(labels)

        val_metrics, _ = evaluate(
            adapter,
            vi_val_latent,
            vi_val_labels,
            cond_proj,
            aux_head,
            latent_centroids.cpu(),
            token_centroids.cpu(),
            device,
        )
        score = val_metrics["vs_token_centroid"]["balanced_accuracy"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(count, 1),
                "val_aux_top1": val_metrics["aux_head"]["top1"],
                "val_latent_top1": val_metrics["vs_latent_centroid"]["top1"],
                "val_token_top1": val_metrics["vs_token_centroid"]["top1"],
                "val_token_balanced_accuracy": score,
            }
        )
        if score > best_score + 1e-12:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(adapter.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[Epoch {epoch:03d}] loss={history[-1]['train_loss']:.4f} "
                f"val_aux={history[-1]['val_aux_top1']:.4f} "
                f"val_lat={history[-1]['val_latent_top1']:.4f} "
                f"val_tok={history[-1]['val_token_top1']:.4f} "
                f"best={best_score:.4f}@{best_epoch}",
                flush=True,
            )
        if stale >= args.patience:
            print(f"[Early stop] epoch={epoch} best={best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("adapter training did not produce a checkpoint")
    adapter.load_state_dict(best_state)
    adapted_metrics, outputs = evaluate(
        adapter,
        vi_test_latent,
        vi_test_labels,
        cond_proj,
        aux_head,
        latent_centroids.cpu(),
        token_centroids.cpu(),
        device,
    )
    _, _, aux_logits, latent_scores, token_scores = outputs

    checkpoint = {
        "adapter": {key: value.cpu() for key, value in adapter.state_dict().items()},
        "best_epoch": best_epoch,
        "best_val_token_balanced_accuracy": best_score,
        "config": vars(args),
        "source_latent_centroids": latent_centroids.cpu(),
        "source_token_centroids": token_centroids.cpu(),
    }
    torch.save(checkpoint, run_dir / "adapter_best.pt")
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    rows = []
    raw_lat_scores = F.normalize(vi_test_latent, dim=1) @ latent_centroids.cpu().T
    with torch.no_grad():
        raw_tokens = []
        for start in range(0, len(vi_test_latent), 128):
            raw_tokens.append(
                F.normalize(
                    cond_proj(vi_test_latent[start : start + 128].to(device)).flatten(1),
                    dim=1,
                ).cpu()
            )
        raw_token_scores = torch.cat(raw_tokens) @ token_centroids.cpu().T
    for index, label_tensor in enumerate(vi_test_labels):
        label = int(label_tensor.item())
        rows.append(
            {
                "test_index": index,
                "true_label": label + 1,
                "true_name": CLASS_NAMES[label],
                "raw_latent_pred": int(raw_lat_scores[index].argmax().item()) + 1,
                "raw_token_pred": int(raw_token_scores[index].argmax().item()) + 1,
                "adapted_aux_pred": int(aux_logits[index].argmax().item()) + 1,
                "adapted_latent_pred": int(latent_scores[index].argmax().item()) + 1,
                "adapted_token_pred": int(token_scores[index].argmax().item()) + 1,
            }
        )
    with (run_dir / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "subject": args.subject_id,
        "seed": args.seed,
        "n_vs_train": len(vs_train),
        "n_vi_train": len(vi_train),
        "n_vi_val": len(vi_val),
        "n_vi_test": len(vi_test),
        "best_epoch": best_epoch,
        "best_val_token_balanced_accuracy": best_score,
        "source_encoder": str(encoder_path),
        "source_vs_lora": str(lora_path),
        "raw_vs_pipeline_on_vi": raw_metrics,
        "adapted_vi_to_vs": adapted_metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    compact = {
        "best_epoch": best_epoch,
        "raw_aux_top1": raw_metrics["aux_head"]["top1"],
        "adapted_aux_top1": adapted_metrics["aux_head"]["top1"],
        "raw_latent_top1": raw_metrics["vs_latent_centroid"]["top1"],
        "adapted_latent_top1": adapted_metrics["vs_latent_centroid"]["top1"],
        "raw_token_top1": raw_metrics["vs_token_centroid"]["top1"],
        "adapted_token_top1": adapted_metrics["vs_token_centroid"]["top1"],
        "mean_cosine_raw_to_adapted": adapted_metrics[
            "mean_cosine_raw_to_adapted"
        ],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    print(f"[Saved] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
