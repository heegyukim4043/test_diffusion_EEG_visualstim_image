"""Joint VS+VI class-aligned adaptation for the VS-to-VI transfer study.

This is a representation-stage experiment.  Stable Diffusion is deliberately
not loaded.  The subject-specific VS SupCon encoder and the original VS LoRA
condition projector define the source space.  We then adapt VI into that space
while preserving VS performance:

* a zero-initialized residual adapter is applied to VI EEG only;
* only the last encoder transformer block, encoder output head, and fusion MLP
  are trainable;
* the original VS condition projector and auxiliary classifier stay frozen;
* every optimization batch contains the same nine classes from both domains;
* VI validation token balanced accuracy selects the checkpoint;
* VI and VS test splits are evaluated only after checkpoint selection.

The predeclared primary comparison is against the already-computed VI-only and
naive full-finetuning baselines.  Do not choose hyperparameters from test data.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_vi_to_vs_adapter import (
    CLASS_NAMES,
    N_CLASSES,
    build_source_centroids,
    classification_metrics,
    encode_dataset,
    load_frozen_source,
    make_dataset,
    pairwise_separation,
)


class ResidualVIInputAdapter(nn.Module):
    """Identity-initialized channel/temporal correction in raw EEG space."""

    def __init__(self, channels: int = 32, temporal_kernel: int = 15):
        super().__init__()
        if temporal_kernel % 2 != 1:
            raise ValueError("temporal_kernel must be odd")
        self.channel_delta = nn.Conv1d(channels, channels, 1, bias=False)
        self.temporal_delta = nn.Conv1d(
            channels,
            channels,
            temporal_kernel,
            padding=temporal_kernel // 2,
            groups=channels,
            bias=False,
        )
        nn.init.zeros_(self.channel_delta.weight)
        nn.init.zeros_(self.temporal_delta.weight)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return eeg + self.channel_delta(eeg) + self.temporal_delta(eeg)


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
    parser.add_argument("--samples_per_class", type=int, default=1)
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=0,
        help="0 chooses ceil(max(train sizes)/(9*samples_per_class))",
    )
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr_adapter", type=float, default=1e-3)
    parser.add_argument("--lr_encoder", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--temporal_kernel", type=int, default=15)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--w_vi_latent", type=float, default=1.0)
    parser.add_argument("--w_vi_token", type=float, default=1.0)
    parser.add_argument("--w_vi_aux", type=float, default=0.5)
    parser.add_argument("--w_cross_supcon", type=float, default=0.5)
    parser.add_argument("--w_source_preserve", type=float, default=0.5)
    parser.add_argument("--w_source_class", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--out_root",
        default="/content/drive/MyDrive/vsvi_data/joint_vs_vi_class_aligned",
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


def configure_adapted_encoder(model: nn.Module):
    """Freeze source model, then open only its late representation layers."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    encoder = model.eeg_encoder
    if not hasattr(encoder, "transformer") or not hasattr(encoder, "fc"):
        raise TypeError(
            "This predeclared experiment requires the V2/transformer EEG encoder"
        )
    if len(encoder.transformer.layers) == 0:
        raise RuntimeError("EEG transformer has no layers")

    train_modules = [
        encoder.transformer.layers[-1],
        encoder.fc,
        model.fusion_mlp,
    ]
    for module in train_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    # The source auxiliary decision boundary is a fixed anchor.
    for parameter in model.aux_cls_head.parameters():
        parameter.requires_grad_(False)
    return train_modules


def set_adaptation_train_mode(model: nn.Module, modules, adapter: nn.Module) -> None:
    # Keep frozen stem/early blocks deterministic (especially their dropout).
    model.eval()
    adapter.train()
    for module in modules:
        module.train()


def class_index_pools(dataset) -> list[np.ndarray]:
    pools: list[list[int]] = [[] for _ in range(N_CLASSES)]
    for index in range(len(dataset)):
        label = int(dataset[index][2])
        pools[label].append(index)
    result = [np.asarray(pool, dtype=np.int64) for pool in pools]
    missing = [CLASS_NAMES[i] for i, pool in enumerate(result) if len(pool) == 0]
    if missing:
        raise RuntimeError(f"training dataset has empty classes: {missing}")
    return result


def paired_balanced_batch(
    vs_dataset,
    vi_dataset,
    vs_pools,
    vi_pools,
    samples_per_class: int,
    rng: np.random.RandomState,
):
    vs_indices: list[int] = []
    vi_indices: list[int] = []
    labels: list[int] = []
    for label in range(N_CLASSES):
        for _ in range(samples_per_class):
            vs_indices.append(int(rng.choice(vs_pools[label])))
            vi_indices.append(int(rng.choice(vi_pools[label])))
            labels.append(label)
    order = rng.permutation(len(labels))
    vs_eeg = torch.stack([vs_dataset[vs_indices[i]][0] for i in order])
    vi_eeg = torch.stack([vi_dataset[vi_indices[i]][0] for i in order])
    y = torch.tensor([labels[i] for i in order], dtype=torch.long)
    return vs_eeg, vi_eeg, y


def token_features(cond_proj: nn.Module, latent: torch.Tensor) -> torch.Tensor:
    return F.normalize(cond_proj(latent).flatten(1), dim=1)


def paired_cross_domain_supcon(
    vs_latent: torch.Tensor,
    vi_latent: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """SupCon over both domains; same-class VS/VI samples are positives."""
    features = F.normalize(torch.cat([vs_latent, vi_latent], dim=0), dim=1)
    all_labels = torch.cat([labels, labels], dim=0)
    count = features.size(0)
    similarity = features @ features.T / temperature
    eye = torch.eye(count, device=features.device, dtype=torch.bool)
    positive = all_labels[:, None].eq(all_labels[None, :]) & ~eye
    logits = similarity.masked_fill(eye, float("-inf"))
    log_prob = similarity - torch.logsumexp(logits, dim=1, keepdim=True)
    n_positive = positive.sum(dim=1)
    valid = n_positive > 0
    per_anchor = -(log_prob.masked_fill(~positive, 0.0).sum(1)) / n_positive.clamp(1)
    return per_anchor[valid].mean()


@torch.no_grad()
def evaluate_domain(
    dataset,
    model: nn.Module,
    input_adapter: nn.Module | None,
    cond_proj: nn.Module,
    latent_centroids: torch.Tensor,
    token_centroids: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict:
    model.eval()
    cond_proj.eval()
    if input_adapter is not None:
        input_adapter.eval()
    latent_parts = []
    token_parts = []
    aux_parts = []
    label_parts = []
    subject_index = torch.zeros(1, dtype=torch.long, device=device)
    for start in range(0, len(dataset), batch_size):
        items = [
            dataset[index]
            for index in range(start, min(start + batch_size, len(dataset)))
        ]
        eeg = torch.stack([item[0] for item in items]).to(device)
        labels = torch.tensor([int(item[2]) for item in items], dtype=torch.long)
        if input_adapter is not None:
            eeg = input_adapter(eeg)
        latent = model.encode_eeg(eeg, subject_index.expand(eeg.size(0)))
        tokens = token_features(cond_proj, latent)
        latent_parts.append(latent.cpu())
        token_parts.append(tokens.cpu())
        aux_parts.append(model.aux_cls_head(latent).cpu())
        label_parts.append(labels)

    latent = torch.cat(latent_parts)
    tokens = torch.cat(token_parts)
    aux_logits = torch.cat(aux_parts)
    labels = torch.cat(label_parts)
    latent_scores = F.normalize(latent, dim=1) @ latent_centroids.T
    token_scores = F.normalize(tokens, dim=1) @ token_centroids.T
    return {
        "n": int(len(labels)),
        "aux_head": classification_metrics(aux_logits, labels),
        "vs_latent_centroid": classification_metrics(latent_scores, labels),
        "vs_token_centroid": classification_metrics(token_scores, labels),
        "latent_within_minus_between": pairwise_separation(latent, labels),
        "token_within_minus_between": pairwise_separation(tokens, labels),
    }


def flat_history_row(epoch: int, loss_parts: dict, val_metrics: dict) -> dict:
    return {
        "epoch": epoch,
        **{f"train_{key}": value for key, value in loss_parts.items()},
        "val_aux_bac": val_metrics["aux_head"]["balanced_accuracy"],
        "val_latent_bac": val_metrics["vs_latent_centroid"]["balanced_accuracy"],
        "val_token_bac": val_metrics["vs_token_centroid"]["balanced_accuracy"],
        "val_latent_separation": val_metrics["latent_within_minus_between"],
        "val_token_separation": val_metrics["token_within_minus_between"],
    }


def cpu_state_dict(module: nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def save_history(path: Path, history: list[dict]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def test_delta(adapted: dict, raw: dict) -> dict:
    result = {}
    for head in ("aux_head", "vs_latent_centroid", "vs_token_centroid"):
        result[head] = {
            metric: adapted[head][metric] - raw[head][metric]
            for metric in ("top1", "top3", "top5", "balanced_accuracy")
        }
    result["latent_within_minus_between"] = (
        adapted["latent_within_minus_between"]
        - raw["latent_within_minus_between"]
    )
    result["token_within_minus_between"] = (
        adapted["token_within_minus_between"]
        - raw["token_within_minus_between"]
    )
    return result


def main() -> None:
    args = parse_args()
    positive = (
        args.samples_per_class,
        args.eval_batch_size,
        args.epochs,
        args.patience,
        args.lr_adapter,
        args.lr_encoder,
        args.temperature,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("batch/epoch/patience/lr/temperature arguments must be positive")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.out_root) / f"S{args.subject_id:02d}" / f"seed{args.seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite to rerun")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] device={device} S{args.subject_id:02d} seed={args.seed}")

    source_encoder, cond_proj, encoder_path, lora_path = load_frozen_source(
        args, device
    )
    teacher = source_encoder
    adapted_encoder = copy.deepcopy(source_encoder).to(device)
    train_modules = configure_adapted_encoder(adapted_encoder)
    input_adapter = ResidualVIInputAdapter(
        channels=32, temporal_kernel=args.temporal_kernel
    ).to(device)

    vi_cap = args.vi_per_class_total if args.vi_per_class_total > 0 else None
    vs_train = make_dataset(args.vs_root, args.subject_id, "train", args.seed, None)
    vs_test = make_dataset(args.vs_root, args.subject_id, "test", args.seed, None)
    vi_train = make_dataset(args.vi_root, args.subject_id, "train", args.seed, vi_cap)
    vi_val = make_dataset(args.vi_root, args.subject_id, "val", args.seed, vi_cap)
    vi_test = make_dataset(args.vi_root, args.subject_id, "test", args.seed, vi_cap)

    print("[INFO] Building fixed VS source centroids...", flush=True)
    vs_train_latent, vs_train_labels = encode_dataset(
        vs_train, teacher, args.eval_batch_size, device
    )
    latent_centroids, token_centroids = build_source_centroids(
        vs_train_latent, vs_train_labels, cond_proj, device
    )
    latent_centroids = latent_centroids.to(device)
    token_centroids = token_centroids.to(device)

    vs_pools = class_index_pools(vs_train)
    vi_pools = class_index_pools(vi_train)
    paired_batch_size = N_CLASSES * args.samples_per_class
    steps_per_epoch = args.steps_per_epoch or math.ceil(
        max(len(vs_train), len(vi_train)) / paired_batch_size
    )
    print(
        f"[INFO] paired batch={paired_batch_size} per domain, "
        f"steps/epoch={steps_per_epoch}",
        flush=True,
    )

    encoder_parameters = [
        parameter for parameter in adapted_encoder.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": input_adapter.parameters(), "lr": args.lr_adapter},
            {"params": encoder_parameters, "lr": args.lr_encoder},
        ],
        weight_decay=args.weight_decay,
    )
    use_fp16 = bool(args.fp16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    rng = np.random.RandomState(args.seed)
    subject_index = torch.zeros(paired_batch_size, dtype=torch.long, device=device)

    best_key = (-float("inf"), -float("inf"))
    best_epoch = 0
    best_encoder_state = None
    best_adapter_state = None
    history: list[dict] = []
    stale = 0

    for epoch in range(1, args.epochs + 1):
        set_adaptation_train_mode(
            adapted_encoder, train_modules, input_adapter
        )
        totals = {
            "loss": 0.0,
            "vi_latent": 0.0,
            "vi_token": 0.0,
            "vi_aux": 0.0,
            "cross_supcon": 0.0,
            "source_preserve": 0.0,
            "source_class": 0.0,
        }
        for _ in range(steps_per_epoch):
            vs_cpu, vi_cpu, labels_cpu = paired_balanced_batch(
                vs_train,
                vi_train,
                vs_pools,
                vi_pools,
                args.samples_per_class,
                rng,
            )
            vs_eeg = vs_cpu.to(device)
            vi_eeg = vi_cpu.to(device)
            labels = labels_cpu.to(device)
            with torch.no_grad():
                teacher_vs = teacher.encode_eeg(vs_eeg, subject_index)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_fp16):
                current_vs = adapted_encoder.encode_eeg(vs_eeg, subject_index)
                current_vi = adapted_encoder.encode_eeg(
                    input_adapter(vi_eeg), subject_index
                )
                vs_tokens = token_features(cond_proj, current_vs)
                vi_tokens = token_features(cond_proj, current_vi)

                vi_latent_loss = F.cross_entropy(
                    current_vi @ latent_centroids.T / args.temperature, labels
                )
                vi_token_loss = F.cross_entropy(
                    vi_tokens @ token_centroids.T / args.temperature, labels
                )
                vi_aux_loss = F.cross_entropy(
                    adapted_encoder.aux_cls_head(current_vi), labels
                )
                cross_loss = paired_cross_domain_supcon(
                    current_vs, current_vi, labels, args.temperature
                )
                preserve_loss = (
                    1.0 - F.cosine_similarity(current_vs, teacher_vs, dim=1)
                ).mean()
                source_class_loss = (
                    F.cross_entropy(
                        current_vs @ latent_centroids.T / args.temperature, labels
                    )
                    + F.cross_entropy(
                        vs_tokens @ token_centroids.T / args.temperature, labels
                    )
                    + F.cross_entropy(
                        adapted_encoder.aux_cls_head(current_vs), labels
                    )
                ) / 3.0
                loss = (
                    args.w_vi_latent * vi_latent_loss
                    + args.w_vi_token * vi_token_loss
                    + args.w_vi_aux * vi_aux_loss
                    + args.w_cross_supcon * cross_loss
                    + args.w_source_preserve * preserve_loss
                    + args.w_source_class * source_class_loss
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(input_adapter.parameters()) + encoder_parameters,
                args.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            batch_values = {
                "loss": loss,
                "vi_latent": vi_latent_loss,
                "vi_token": vi_token_loss,
                "vi_aux": vi_aux_loss,
                "cross_supcon": cross_loss,
                "source_preserve": preserve_loss,
                "source_class": source_class_loss,
            }
            for key, value in batch_values.items():
                totals[key] += float(value.detach().item())

        means = {key: value / steps_per_epoch for key, value in totals.items()}
        val_metrics = evaluate_domain(
            vi_val,
            adapted_encoder,
            input_adapter,
            cond_proj,
            latent_centroids.cpu(),
            token_centroids.cpu(),
            args.eval_batch_size,
            device,
        )
        row = flat_history_row(epoch, means, val_metrics)
        history.append(row)
        current_key = (
            val_metrics["vs_token_centroid"]["balanced_accuracy"],
            val_metrics["vs_latent_centroid"]["balanced_accuracy"],
        )
        if current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            best_encoder_state = cpu_state_dict(adapted_encoder)
            best_adapter_state = cpu_state_dict(input_adapter)
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % 5 == 0 or current_key == best_key:
            print(
                f"Ep {epoch:03d} loss={means['loss']:.4f} "
                f"VI-val latentBAC={current_key[1]:.4f} "
                f"tokenBAC={current_key[0]:.4f} best={best_epoch}",
                flush=True,
            )
        if stale >= args.patience:
            print(f"[Early stop] epoch={epoch} best={best_epoch}", flush=True)
            break

    if best_encoder_state is None or best_adapter_state is None:
        raise RuntimeError("training produced no checkpoint")
    adapted_encoder.load_state_dict(best_encoder_state, strict=True)
    input_adapter.load_state_dict(best_adapter_state, strict=True)

    # The only test evaluation occurs here, after validation-based selection.
    print("[INFO] Final one-time VI/VS test evaluation...", flush=True)
    raw_vi = evaluate_domain(
        vi_test,
        teacher,
        None,
        cond_proj,
        latent_centroids.cpu(),
        token_centroids.cpu(),
        args.eval_batch_size,
        device,
    )
    adapted_vi = evaluate_domain(
        vi_test,
        adapted_encoder,
        input_adapter,
        cond_proj,
        latent_centroids.cpu(),
        token_centroids.cpu(),
        args.eval_batch_size,
        device,
    )
    raw_vs = evaluate_domain(
        vs_test,
        teacher,
        None,
        cond_proj,
        latent_centroids.cpu(),
        token_centroids.cpu(),
        args.eval_batch_size,
        device,
    )
    adapted_vs = evaluate_domain(
        vs_test,
        adapted_encoder,
        None,
        cond_proj,
        latent_centroids.cpu(),
        token_centroids.cpu(),
        args.eval_batch_size,
        device,
    )

    trainable_names = [
        name
        for name, parameter in adapted_encoder.named_parameters()
        if parameter.requires_grad
    ]
    checkpoint = {
        "encoder": best_encoder_state,
        "vi_input_adapter": best_adapter_state,
        "best_epoch": best_epoch,
        "best_vi_val_token_bac": best_key[0],
        "best_vi_val_latent_bac": best_key[1],
        "source_encoder_checkpoint": str(encoder_path),
        "source_vs_lora_checkpoint": str(lora_path),
        "trainable_encoder_parameters": trainable_names,
        "args": vars(args),
    }
    torch.save(checkpoint, run_dir / "best.pt")
    save_history(run_dir / "history.csv", history)

    metrics = {
        "subject": args.subject_id,
        "seed": args.seed,
        "selection": {
            "primary": "VI validation VS-token-centroid balanced accuracy",
            "tie_break": "VI validation VS-latent-centroid balanced accuracy",
            "best_epoch": best_epoch,
            "best_vi_val_token_bac": best_key[0],
            "best_vi_val_latent_bac": best_key[1],
        },
        "raw": {"vi_test": raw_vi, "vs_test": raw_vs},
        "adapted": {"vi_test": adapted_vi, "vs_test": adapted_vs},
        "delta_adapted_minus_raw": {
            "vi_test": test_delta(adapted_vi, raw_vi),
            "vs_test": test_delta(adapted_vs, raw_vs),
        },
        "source_encoder_checkpoint": str(encoder_path),
        "source_vs_lora_checkpoint": str(lora_path),
        "steps_per_epoch": steps_per_epoch,
        "trainable_encoder_parameters": trainable_names,
        "args": vars(args),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print(json.dumps(metrics["selection"], indent=2), flush=True)
    print(
        "VI token BAC raw/adapted: "
        f"{raw_vi['vs_token_centroid']['balanced_accuracy']:.4f} / "
        f"{adapted_vi['vs_token_centroid']['balanced_accuracy']:.4f}",
        flush=True,
    )
    print(
        "VS token BAC raw/adapted: "
        f"{raw_vs['vs_token_centroid']['balanced_accuracy']:.4f} / "
        f"{adapted_vs['vs_token_centroid']['balanced_accuracy']:.4f}",
        flush=True,
    )
    print(f"[Saved] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
