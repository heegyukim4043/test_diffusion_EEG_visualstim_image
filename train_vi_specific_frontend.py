"""Session-held-out VI-specific front-end adaptation into a frozen VS space.

The source subject-specific VS SupCon encoder and the original VS LoRA
condition projector define the semantic space.  Their high-level components
stay frozen.  Only a VI-specific copy of the V2 encoder front-end is trained:

    channel gate -> multi-scale temporal stem -> temporal projection

Training batches contain the same number of trials for all nine classes and
prefer different VI sessions within each class.  Class-mean latent and token
prototypes are aligned to fixed VS class prototypes.  Validation uses one
complete VI session and the outer test uses another complete VI session.

The outer test is evaluated once after validation checkpoint selection.  The
saved Hungarian-aligned accuracy is diagnostic only and must not be reported
as decoding performance because it derives a mapping from test labels.
Stable Diffusion is not loaded.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset

from dataset_vs_re import load_subject_vsre
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vs_lora_ckpt", required=True)
    parser.add_argument(
        "--supcon_ckpt", required=True, help="directory containing subjNN_best.pt"
    )
    parser.add_argument("--vs_root", required=True)
    parser.add_argument("--vi_root", required=True)
    parser.add_argument("--subject_id", type=int, required=True)
    parser.add_argument("--test_session", type=int, required=True)
    parser.add_argument(
        "--val_session",
        type=int,
        default=0,
        help="0 selects the next available VI session cyclically",
    )
    parser.add_argument("--n_eeg_tokens", type=int, default=16)
    parser.add_argument("--samples_per_class", type=int, default=4)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--prototype_margin", type=float, default=0.10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--w_individual_latent", type=float, default=0.25)
    parser.add_argument("--w_individual_token", type=float, default=0.25)
    parser.add_argument("--w_individual_aux", type=float, default=0.25)
    parser.add_argument("--w_prototype_latent", type=float, default=1.0)
    parser.add_argument("--w_prototype_token", type=float, default=1.0)
    parser.add_argument("--w_consistency", type=float, default=0.25)
    parser.add_argument("--w_margin", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--out_root",
        default="/content/drive/MyDrive/vsvi_data/vi_specific_frontend_loso",
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


def discover_sessions(root: str | Path, sid: int) -> list[int]:
    root = Path(root)
    pattern = re.compile(rf"^preproc_subj_{sid:02d}_(\d+)\.(?:npz|mat)$")
    sessions = set()
    if root.is_dir():
        for path in root.iterdir():
            match = pattern.match(path.name)
            if match:
                sessions.add(int(match.group(1)))
    if not sessions:
        raise FileNotFoundError(f"No S{sid:02d} session files in {root}")
    return sorted(sessions)


def choose_validation_session(sessions: list[int], test_session: int) -> int:
    index = sessions.index(test_session)
    for offset in range(1, len(sessions)):
        candidate = sessions[(index + offset) % len(sessions)]
        if candidate != test_session:
            return candidate
    raise RuntimeError("At least three VI sessions are required")


class SessionEEGDataset(Dataset):
    """EEG trials retaining their session identity; non-finite trials drop."""

    def __init__(self, root: str, sid: int, sessions: list[int], label: str):
        self.samples: list[tuple[torch.Tensor, int, int]] = []
        self.sessions = list(sessions)
        self.class_counts = {class_id: 0 for class_id in range(N_CLASSES)}
        self.session_counts: dict[int, int] = {}
        for session in sessions:
            eeg, labels, effective = load_subject_vsre(
                root, sid, n_ch=32, sessions=[session]
            )
            if effective != 1:
                raise RuntimeError(
                    f"S{sid:02d} session {session} did not load exactly once"
                )
            finite = np.isfinite(eeg).all(axis=(1, 2))
            dropped = int((~finite).sum())
            if dropped:
                print(
                    f"[WARN] {label} session {session}: dropped {dropped} non-finite trials",
                    flush=True,
                )
            eeg = eeg[finite]
            labels = labels[finite]
            self.session_counts[session] = int(len(labels))
            for trial, class_id in zip(eeg, labels):
                class_id = int(class_id)
                self.samples.append(
                    (torch.from_numpy(trial), class_id, int(session))
                )
                self.class_counts[class_id] += 1

        missing = [
            CLASS_NAMES[class_id]
            for class_id, count in self.class_counts.items()
            if count == 0
        ]
        if missing:
            raise RuntimeError(f"{label} has empty classes: {missing}")
        print(
            f"[dataset:{label}] sessions={sessions} n={len(self.samples)} "
            f"class_counts={self.class_counts}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


class VISpecificFrontend(nn.Module):
    """Trainable copy of the source V2 encoder's low-level feature extractor."""

    def __init__(self, source_eeg_encoder: nn.Module):
        super().__init__()
        required = ("channel_gate", "stem", "proj", "transformer", "fc")
        missing = [name for name in required if not hasattr(source_eeg_encoder, name)]
        if missing:
            raise TypeError(f"V2 EEG encoder required; missing {missing}")
        self.channel_gate = copy.deepcopy(source_eeg_encoder.channel_gate)
        self.stem = copy.deepcopy(source_eeg_encoder.stem)
        self.proj = copy.deepcopy(source_eeg_encoder.proj)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        eeg = (eeg - eeg.mean(-1, keepdim=True)) / (
            eeg.std(-1, keepdim=True) + 1e-6
        )
        hidden = self.channel_gate(eeg)
        hidden = self.stem(hidden)
        return self.proj(hidden)


def encode_with_vi_frontend(
    source_model: nn.Module,
    vi_frontend: VISpecificFrontend,
    eeg: torch.Tensor,
    subject_ids: torch.Tensor,
) -> torch.Tensor:
    """VI front-end followed by the completely frozen source high-level path."""
    encoder = source_model.eeg_encoder
    hidden = vi_frontend(eeg).transpose(1, 2)
    position = encoder._sinusoidal_pos_embed(
        hidden.size(1), hidden.size(2), hidden.device
    )
    hidden = hidden + position
    hidden = encoder.transformer(hidden).mean(1)
    eeg_features = encoder.fc(hidden)
    subject_features = source_model.subject_emb(subject_ids)
    fused = torch.cat([eeg_features, subject_features], dim=1)
    return F.normalize(source_model.fusion_mlp(fused), dim=1)


def token_features(cond_proj: nn.Module, latent: torch.Tensor) -> torch.Tensor:
    return F.normalize(cond_proj(latent).flatten(1), dim=1)


def build_class_session_pools(dataset: SessionEEGDataset):
    pools: dict[int, dict[int, np.ndarray]] = {
        class_id: {} for class_id in range(N_CLASSES)
    }
    temporary: dict[int, dict[int, list[int]]] = {
        class_id: {} for class_id in range(N_CLASSES)
    }
    for index, (_, class_id, session) in enumerate(dataset.samples):
        temporary[class_id].setdefault(session, []).append(index)
    for class_id in range(N_CLASSES):
        for session, indices in temporary[class_id].items():
            pools[class_id][session] = np.asarray(indices, dtype=np.int64)
        if not pools[class_id]:
            raise RuntimeError(f"No train pools for class {CLASS_NAMES[class_id]}")
    return pools


def balanced_multisession_batch(
    dataset: SessionEEGDataset,
    pools,
    samples_per_class: int,
    rng: np.random.RandomState,
):
    indices: list[int] = []
    labels: list[int] = []
    sessions: list[int] = []
    for class_id in range(N_CLASSES):
        available_sessions = sorted(pools[class_id])
        replace_sessions = len(available_sessions) < samples_per_class
        selected_sessions = rng.choice(
            available_sessions,
            size=samples_per_class,
            replace=replace_sessions,
        )
        for session in selected_sessions:
            pool = pools[class_id][int(session)]
            indices.append(int(rng.choice(pool)))
            labels.append(class_id)
            sessions.append(int(session))
    order = rng.permutation(len(indices))
    eeg = torch.stack([dataset[indices[i]][0] for i in order])
    class_labels = torch.tensor([labels[i] for i in order], dtype=torch.long)
    session_labels = torch.tensor([sessions[i] for i in order], dtype=torch.long)
    return eeg, class_labels, session_labels


def class_prototypes(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    prototypes = []
    for class_id in range(N_CLASSES):
        selected = features[labels == class_id]
        if selected.size(0) == 0:
            raise RuntimeError(f"batch missing class {class_id}")
        prototypes.append(F.normalize(selected.mean(0), dim=0))
    return torch.stack(prototypes)


def within_class_consistency(
    features: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor
) -> torch.Tensor:
    target = prototypes[labels]
    return (1.0 - F.cosine_similarity(features, target, dim=1)).mean()


def target_margin_loss(
    scores: torch.Tensor, targets: torch.Tensor, margin: float
) -> torch.Tensor:
    correct = scores.gather(1, targets[:, None]).squeeze(1)
    mask = F.one_hot(targets, num_classes=N_CLASSES).bool()
    strongest_wrong = scores.masked_fill(mask, float("-inf")).max(dim=1).values
    return F.relu(margin - correct + strongest_wrong).mean()


def confusion_matrix(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long)
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        matrix[int(truth), int(prediction)] += 1
    return matrix


def centroid_diagnostics(
    features: torch.Tensor,
    labels: torch.Tensor,
    source_centroids: torch.Tensor,
) -> dict:
    prototypes = class_prototypes(F.normalize(features, dim=1), labels)
    similarity = prototypes @ source_centroids.T
    diagonal = similarity.diagonal()
    mask = torch.eye(N_CLASSES, dtype=torch.bool)
    best_wrong = similarity.masked_fill(mask, float("-inf")).max(1).values
    predicted_source = similarity.argmax(1)
    return {
        "similarity_matrix": similarity.tolist(),
        "diagonal_mean": float(diagonal.mean().item()),
        "best_wrong_mean": float(best_wrong.mean().item()),
        "diagonal_margin_mean": float((diagonal - best_wrong).mean().item()),
        "prototype_top1": float(
            (predicted_source == torch.arange(N_CLASSES)).float().mean().item()
        ),
        "prototype_assignment": {
            CLASS_NAMES[class_id]: CLASS_NAMES[int(predicted_source[class_id])]
            for class_id in range(N_CLASSES)
        },
    }


def prediction_diagnostics(scores: torch.Tensor, labels: torch.Tensor) -> dict:
    predictions = scores.argmax(1)
    matrix = confusion_matrix(predictions, labels)
    row_indices, column_indices = linear_sum_assignment(-matrix.numpy())
    hungarian_correct = int(matrix[row_indices, column_indices].sum().item())
    mapping = {
        CLASS_NAMES[int(column)]: CLASS_NAMES[int(row)]
        for row, column in zip(row_indices, column_indices)
    }
    return {
        "confusion_matrix_rows_true_cols_pred": matrix.tolist(),
        "prediction_counts": {
            CLASS_NAMES[class_id]: int((predictions == class_id).sum().item())
            for class_id in range(N_CLASSES)
        },
        "hungarian_aligned_accuracy_diagnostic_only": (
            hungarian_correct / max(len(labels), 1)
        ),
        "hungarian_pred_to_true_mapping_diagnostic_only": mapping,
    }


@torch.no_grad()
def evaluate_dataset(
    dataset: SessionEEGDataset,
    source_model: nn.Module,
    vi_frontend: VISpecificFrontend | None,
    cond_proj: nn.Module,
    latent_centroids: torch.Tensor,
    token_centroids: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict:
    source_model.eval()
    cond_proj.eval()
    if vi_frontend is not None:
        vi_frontend.eval()
    latent_parts = []
    token_parts = []
    aux_parts = []
    label_parts = []
    session_parts = []
    subject_index = torch.zeros(1, dtype=torch.long, device=device)
    for start in range(0, len(dataset), batch_size):
        items = [
            dataset[index]
            for index in range(start, min(start + batch_size, len(dataset)))
        ]
        eeg = torch.stack([item[0] for item in items]).to(device)
        labels = torch.tensor([int(item[1]) for item in items], dtype=torch.long)
        sessions = torch.tensor([int(item[2]) for item in items], dtype=torch.long)
        subject_ids = subject_index.expand(eeg.size(0))
        if vi_frontend is None:
            latent = source_model.encode_eeg(eeg, subject_ids)
        else:
            latent = encode_with_vi_frontend(
                source_model, vi_frontend, eeg, subject_ids
            )
        tokens = token_features(cond_proj, latent)
        latent_parts.append(latent.cpu())
        token_parts.append(tokens.cpu())
        aux_parts.append(source_model.aux_cls_head(latent).cpu())
        label_parts.append(labels)
        session_parts.append(sessions)

    latent = torch.cat(latent_parts)
    tokens = torch.cat(token_parts)
    aux_logits = torch.cat(aux_parts)
    labels = torch.cat(label_parts)
    sessions = torch.cat(session_parts)
    latent_scores = F.normalize(latent, dim=1) @ latent_centroids.T
    token_scores = F.normalize(tokens, dim=1) @ token_centroids.T
    return {
        "n": int(len(labels)),
        "sessions": sorted(set(int(value) for value in sessions.tolist())),
        "aux_head": classification_metrics(aux_logits, labels),
        "vs_latent_centroid": classification_metrics(latent_scores, labels),
        "vs_token_centroid": classification_metrics(token_scores, labels),
        "latent_within_minus_between": pairwise_separation(latent, labels),
        "token_within_minus_between": pairwise_separation(tokens, labels),
        "latent_centroid_diagnostics": centroid_diagnostics(
            latent, labels, latent_centroids
        ),
        "token_centroid_diagnostics": centroid_diagnostics(
            tokens, labels, token_centroids
        ),
        "latent_prediction_diagnostics": prediction_diagnostics(
            latent_scores, labels
        ),
        "token_prediction_diagnostics": prediction_diagnostics(token_scores, labels),
    }


def cpu_state_dict(module: nn.Module) -> dict:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def save_history(path: Path, history: list[dict]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    positive = (
        args.samples_per_class,
        args.eval_batch_size,
        args.epochs,
        args.patience,
        args.lr,
        args.temperature,
        args.grad_clip,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("batch/epoch/patience/lr/temperature arguments must be positive")
    if args.samples_per_class < 2:
        raise ValueError("samples_per_class must be at least 2 for prototypes")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sessions = discover_sessions(args.vi_root, args.subject_id)
    if len(sessions) < 3:
        raise RuntimeError("Session-held-out training requires at least three sessions")
    if args.test_session not in sessions:
        raise ValueError(f"test session {args.test_session} not in {sessions}")
    val_session = args.val_session or choose_validation_session(
        sessions, args.test_session
    )
    if val_session not in sessions or val_session == args.test_session:
        raise ValueError("val_session must exist and differ from test_session")
    train_sessions = [
        session
        for session in sessions
        if session not in {args.test_session, val_session}
    ]

    run_dir = (
        Path(args.out_root)
        / f"S{args.subject_id:02d}"
        / f"test_sess{args.test_session:02d}_val_sess{val_session:02d}"
        / f"seed{args.seed}"
    )
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite to rerun")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] device={device} S{args.subject_id:02d} "
        f"train={train_sessions} val={val_session} test={args.test_session}",
        flush=True,
    )

    source_model, cond_proj, encoder_path, lora_path = load_frozen_source(
        args, device
    )
    for module in (source_model, cond_proj):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    vi_frontend = VISpecificFrontend(source_model.eeg_encoder).to(device)
    initial_frontend_state = cpu_state_dict(vi_frontend)
    vi_train = SessionEEGDataset(
        args.vi_root, args.subject_id, train_sessions, "vi_train_sessions"
    )
    vi_val = SessionEEGDataset(
        args.vi_root, args.subject_id, [val_session], "vi_validation_session"
    )
    vi_test = SessionEEGDataset(
        args.vi_root, args.subject_id, [args.test_session], "vi_test_session"
    )

    # Fixed VS prototypes use the same source-training construction as prior work.
    vs_train = make_dataset(args.vs_root, args.subject_id, "train", args.seed, None)
    print("[INFO] Building frozen VS latent/token prototypes...", flush=True)
    vs_latent, vs_labels = encode_dataset(
        vs_train, source_model, args.eval_batch_size, device
    )
    latent_centroids, token_centroids = build_source_centroids(
        vs_latent, vs_labels, cond_proj, device
    )
    latent_centroids = latent_centroids.to(device)
    token_centroids = token_centroids.to(device)

    pools = build_class_session_pools(vi_train)
    batch_size = N_CLASSES * args.samples_per_class
    steps_per_epoch = args.steps_per_epoch or math.ceil(len(vi_train) / batch_size)
    print(
        f"[INFO] VI batch={batch_size} ({args.samples_per_class}/class), "
        f"steps/epoch={steps_per_epoch}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        vi_frontend.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    use_fp16 = bool(args.fp16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    rng = np.random.RandomState(args.seed)
    subject_ids = torch.zeros(
        args.samples_per_class, dtype=torch.long, device=device
    )

    best_key = (-float("inf"), -float("inf"))
    best_epoch = 0
    best_state = None
    stale = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        source_model.eval()
        cond_proj.eval()
        vi_frontend.train()
        totals = {
            "loss": 0.0,
            "individual_latent": 0.0,
            "individual_token": 0.0,
            "individual_aux": 0.0,
            "prototype_latent": 0.0,
            "prototype_token": 0.0,
            "consistency": 0.0,
            "margin": 0.0,
        }
        for _ in range(steps_per_epoch):
            eeg_cpu, labels_cpu, _ = balanced_multisession_batch(
                vi_train, pools, args.samples_per_class, rng
            )
            labels = labels_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            step_values = {name: 0.0 for name in totals}
            for class_id in range(N_CLASSES):
                class_mask = labels == class_id
                class_eeg = eeg_cpu[class_mask.cpu()].to(device)
                class_labels = labels[class_mask]
                class_target = torch.tensor(
                    [class_id], dtype=torch.long, device=device
                )
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    latent = encode_with_vi_frontend(
                        source_model, vi_frontend, class_eeg, subject_ids
                    )
                    tokens = token_features(cond_proj, latent)
                    latent_logits = (
                        latent @ latent_centroids.T / args.temperature
                    )
                    token_logits = tokens @ token_centroids.T / args.temperature
                    individual_latent = F.cross_entropy(
                        latent_logits, class_labels
                    )
                    individual_token = F.cross_entropy(
                        token_logits, class_labels
                    )
                    individual_aux = F.cross_entropy(
                        source_model.aux_cls_head(latent), class_labels
                    )

                    latent_proto = F.normalize(latent.mean(0), dim=0)[None, :]
                    token_proto = F.normalize(tokens.mean(0), dim=0)[None, :]
                    latent_proto_scores = latent_proto @ latent_centroids.T
                    token_proto_scores = token_proto @ token_centroids.T
                    prototype_latent = F.cross_entropy(
                        latent_proto_scores / args.temperature, class_target
                    )
                    prototype_token = F.cross_entropy(
                        token_proto_scores / args.temperature, class_target
                    )
                    consistency = 0.5 * (
                        (
                            1.0
                            - F.cosine_similarity(
                                latent, latent_proto.expand_as(latent), dim=1
                            )
                        ).mean()
                        + (
                            1.0
                            - F.cosine_similarity(
                                tokens, token_proto.expand_as(tokens), dim=1
                            )
                        ).mean()
                    )
                    margin_loss = 0.5 * (
                        target_margin_loss(
                            latent_proto_scores,
                            class_target,
                            args.prototype_margin,
                        )
                        + target_margin_loss(
                            token_proto_scores,
                            class_target,
                            args.prototype_margin,
                        )
                    )
                    loss = (
                        args.w_individual_latent * individual_latent
                        + args.w_individual_token * individual_token
                        + args.w_individual_aux * individual_aux
                        + args.w_prototype_latent * prototype_latent
                        + args.w_prototype_token * prototype_token
                        + args.w_consistency * consistency
                        + args.w_margin * margin_loss
                    )
                    scaled_class_loss = loss / N_CLASSES

                scaler.scale(scaled_class_loss).backward()
                class_values = {
                    "loss": loss,
                    "individual_latent": individual_latent,
                    "individual_token": individual_token,
                    "individual_aux": individual_aux,
                    "prototype_latent": prototype_latent,
                    "prototype_token": prototype_token,
                    "consistency": consistency,
                    "margin": margin_loss,
                }
                for name, value in class_values.items():
                    step_values[name] += float(value.detach().item()) / N_CLASSES

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                vi_frontend.parameters(), args.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            for name, value in step_values.items():
                totals[name] += value

        means = {name: value / steps_per_epoch for name, value in totals.items()}
        val_metrics = evaluate_dataset(
            vi_val,
            source_model,
            vi_frontend,
            cond_proj,
            latent_centroids.cpu(),
            token_centroids.cpu(),
            args.eval_batch_size,
            device,
        )
        current_key = (
            val_metrics["vs_token_centroid"]["balanced_accuracy"],
            val_metrics["vs_latent_centroid"]["balanced_accuracy"],
        )
        row = {
            "epoch": epoch,
            **{f"train_{name}": value for name, value in means.items()},
            "val_aux_bac": val_metrics["aux_head"]["balanced_accuracy"],
            "val_latent_bac": current_key[1],
            "val_token_bac": current_key[0],
            "val_latent_separation": val_metrics[
                "latent_within_minus_between"
            ],
            "val_token_separation": val_metrics["token_within_minus_between"],
            "val_latent_centroid_margin": val_metrics[
                "latent_centroid_diagnostics"
            ]["diagonal_margin_mean"],
            "val_token_centroid_margin": val_metrics[
                "token_centroid_diagnostics"
            ]["diagonal_margin_mean"],
        }
        history.append(row)
        if current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            best_state = cpu_state_dict(vi_frontend)
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0 or stale == 0:
            print(
                f"Ep {epoch:03d} loss={means['loss']:.4f} "
                f"val latentBAC={current_key[1]:.4f} "
                f"tokenBAC={current_key[0]:.4f} best={best_epoch}",
                flush=True,
            )
        if stale >= args.patience:
            print(f"[Early stop] epoch={epoch} best={best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    vi_frontend.load_state_dict(best_state, strict=True)
    raw_frontend = VISpecificFrontend(source_model.eeg_encoder).to(device)
    raw_frontend.load_state_dict(initial_frontend_state, strict=True)

    print("[INFO] Final one-time held-out VI test evaluation...", flush=True)
    raw_test = evaluate_dataset(
        vi_test,
        source_model,
        raw_frontend,
        cond_proj,
        latent_centroids.cpu(),
        token_centroids.cpu(),
        args.eval_batch_size,
        device,
    )
    adapted_test = evaluate_dataset(
        vi_test,
        source_model,
        vi_frontend,
        cond_proj,
        latent_centroids.cpu(),
        token_centroids.cpu(),
        args.eval_batch_size,
        device,
    )
    final_val = evaluate_dataset(
        vi_val,
        source_model,
        vi_frontend,
        cond_proj,
        latent_centroids.cpu(),
        token_centroids.cpu(),
        args.eval_batch_size,
        device,
    )

    checkpoint = {
        "vi_frontend": best_state,
        "best_epoch": best_epoch,
        "best_vi_val_token_bac": best_key[0],
        "best_vi_val_latent_bac": best_key[1],
        "train_sessions": train_sessions,
        "val_session": val_session,
        "test_session": args.test_session,
        "source_encoder_checkpoint": str(encoder_path),
        "source_vs_lora_checkpoint": str(lora_path),
        "args": vars(args),
    }
    torch.save(checkpoint, run_dir / "best.pt")
    save_history(run_dir / "history.csv", history)
    metrics = {
        "subject": args.subject_id,
        "seed": args.seed,
        "protocol": "outer VI session held out; next session validation",
        "train_sessions": train_sessions,
        "val_session": val_session,
        "test_session": args.test_session,
        "selection": {
            "primary": "VI validation VS-token-centroid balanced accuracy",
            "tie_break": "VI validation VS-latent-centroid balanced accuracy",
            "best_epoch": best_epoch,
            "best_vi_val_token_bac": best_key[0],
            "best_vi_val_latent_bac": best_key[1],
        },
        "validation": final_val,
        "raw_test": raw_test,
        "adapted_test": adapted_test,
        "delta_test": {
            "aux_bac": (
                adapted_test["aux_head"]["balanced_accuracy"]
                - raw_test["aux_head"]["balanced_accuracy"]
            ),
            "latent_bac": (
                adapted_test["vs_latent_centroid"]["balanced_accuracy"]
                - raw_test["vs_latent_centroid"]["balanced_accuracy"]
            ),
            "token_bac": (
                adapted_test["vs_token_centroid"]["balanced_accuracy"]
                - raw_test["vs_token_centroid"]["balanced_accuracy"]
            ),
            "latent_separation": (
                adapted_test["latent_within_minus_between"]
                - raw_test["latent_within_minus_between"]
            ),
            "token_separation": (
                adapted_test["token_within_minus_between"]
                - raw_test["token_within_minus_between"]
            ),
        },
        "source_path": "exactly frozen; VS output is unchanged by construction",
        "source_encoder_checkpoint": str(encoder_path),
        "source_vs_lora_checkpoint": str(lora_path),
        "steps_per_epoch": steps_per_epoch,
        "args": vars(args),
        "reporting_warning": (
            "Hungarian-aligned accuracy is diagnostic only and is not a valid "
            "decoding result."
        ),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print(json.dumps(metrics["selection"], indent=2), flush=True)
    print(
        "Held-out token BAC raw/adapted: "
        f"{raw_test['vs_token_centroid']['balanced_accuracy']:.4f} / "
        f"{adapted_test['vs_token_centroid']['balanced_accuracy']:.4f}",
        flush=True,
    )
    print(
        "Held-out token Hungarian diagnostic raw/adapted: "
        f"{raw_test['token_prediction_diagnostics']['hungarian_aligned_accuracy_diagnostic_only']:.4f} / "
        f"{adapted_test['token_prediction_diagnostics']['hungarian_aligned_accuracy_diagnostic_only']:.4f}",
        flush=True,
    )
    print(f"[Saved] {run_dir}", flush=True)


if __name__ == "__main__":
    main()

