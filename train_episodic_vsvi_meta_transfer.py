"""Episodic multi-subject VS-to-VI meta-transfer with target-VI LOSO.

For each target subject, the model is trained only on the remaining subjects.
Every training episode uses VS trials as support and VI trials as query from
the same source subject.  A subject-ID-free shared EEG encoder is optimized so
that VI queries are classified by VS support prototypes.  Target VI is not
loaded until source-subject validation has selected the checkpoint.

The target subject's VS data are allowed at evaluation to construct its nine
class prototypes.  Thus this is target-VI-held-out LOSO rather than strict
all-modality subject LOSO.  Stable Diffusion and DINO are not loaded.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

from model_128_eegonly_transformer import EEGEncoderV2
from train_crosssubj_dino import EEGAugment
from train_vi_specific_frontend import (
    CLASS_NAMES,
    N_CLASSES,
    SessionEEGDataset,
    build_class_session_pools,
    discover_sessions,
)


@dataclass
class SourceEpisodeData:
    sid: int
    subject_index: int
    vs: SessionEEGDataset
    vi_train: SessionEEGDataset
    vi_val: SessionEEGDataset
    vs_pools: dict
    vi_pools: dict
    vs_sessions: list[int]
    vi_train_sessions: list[int]
    vi_val_session: int


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float):
        ctx.scale = float(scale)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.scale * grad_output, None


class SharedMetaEncoder(nn.Module):
    """Subject-ID-free shared V2 EEG encoder and diagnostic class head."""

    def __init__(
        self,
        n_source_subjects: int,
        hidden_dim: int = 128,
        eeg_out: int = 256,
        latent_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.backbone = EEGEncoderV2(
            eeg_channels=32,
            eeg_hidden_dim=hidden_dim,
            out_dim=eeg_out,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            occipital_indices=None,
        )
        self.projector = nn.Sequential(
            nn.LayerNorm(eeg_out),
            nn.Linear(eeg_out, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.class_head = nn.Linear(latent_dim, N_CLASSES)
        self.domain_head = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, 2)
        )
        self.subject_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_source_subjects),
        )
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(temperature), dtype=torch.float32)
        )

    def encode(self, eeg: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(self.backbone(eeg)), dim=1)

    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.03, 0.30)

    def adversarial_logits(
        self, latent: torch.Tensor, grl_scale: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reversed_latent = GradientReverse.apply(latent, grl_scale)
        return (
            self.domain_head(reversed_latent),
            self.subject_head(reversed_latent),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_subject", required=True, type=int)
    parser.add_argument("--subjects", default="1,2,9,18,24,28,29")
    parser.add_argument("--vs_root", required=True)
    parser.add_argument("--vi_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=2)
    parser.add_argument("--steps_per_epoch", type=int, default=24)
    parser.add_argument("--subjects_per_step", type=int, default=2)
    parser.add_argument("--support_per_class", type=int, default=1)
    parser.add_argument("--query_per_class", type=int, default=1)
    parser.add_argument("--eval_support_per_class", type=int, default=15)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--eeg_out", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--w_episode", type=float, default=1.0)
    parser.add_argument("--w_alignment", type=float, default=0.5)
    parser.add_argument("--w_supcon", type=float, default=0.25)
    parser.add_argument("--w_aux", type=float, default=0.25)
    parser.add_argument("--w_domain", type=float, default=0.10)
    parser.add_argument("--w_subject", type=float, default=0.10)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--out_root",
        default=(
            "/content/drive/MyDrive/vsvi_data/"
            "episodic_vsvi_meta_transfer"
        ),
    )
    return parser.parse_args()


def parse_subjects(text: str) -> list[int]:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if len(values) != len(set(values)):
        raise ValueError("subjects contains duplicates")
    return values


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_validation_dataset(args, sid: int, sessions: list[int]):
    errors = []
    for session in sorted(sessions, reverse=True):
        try:
            dataset = SessionEEGDataset(
                args.vi_root, sid, [session], f"S{sid:02d}_vi_val"
            )
            if min(dataset.class_counts.values()) < 2:
                raise RuntimeError("fewer than two finite trials in a class")
            return session, dataset
        except Exception as error:
            errors.append(f"session {session}: {error}")
    raise RuntimeError(
        f"No valid VI validation session for S{sid:02d}: "
        + " | ".join(errors)
    )


def prepare_source(args, sid: int, subject_index: int) -> SourceEpisodeData:
    vs_sessions = discover_sessions(args.vs_root, sid)
    vi_sessions = discover_sessions(args.vi_root, sid)
    if len(vi_sessions) < 2:
        raise RuntimeError(f"S{sid:02d} needs at least two VI sessions")
    vi_val_session, vi_val = choose_validation_dataset(args, sid, vi_sessions)
    vi_train_sessions = [s for s in vi_sessions if s != vi_val_session]
    vs = SessionEEGDataset(
        args.vs_root, sid, vs_sessions, f"S{sid:02d}_vs_support"
    )
    vi_train = SessionEEGDataset(
        args.vi_root, sid, vi_train_sessions, f"S{sid:02d}_vi_train"
    )
    return SourceEpisodeData(
        sid=sid,
        subject_index=subject_index,
        vs=vs,
        vi_train=vi_train,
        vi_val=vi_val,
        vs_pools=build_class_session_pools(vs),
        vi_pools=build_class_session_pools(vi_train),
        vs_sessions=vs_sessions,
        vi_train_sessions=vi_train_sessions,
        vi_val_session=vi_val_session,
    )


def sample_class_session_indices(
    pools: dict,
    samples_per_class: int,
    rng: np.random.RandomState,
) -> list[int]:
    indices = []
    for class_id in range(N_CLASSES):
        session_pools = pools[class_id]
        sessions = sorted(session_pools)
        if not sessions:
            raise RuntimeError(f"class {class_id} has no session pools")
        chosen_sessions = rng.choice(
            sessions,
            size=samples_per_class,
            replace=len(sessions) < samples_per_class,
        )
        for session in chosen_sessions:
            indices.append(
                int(rng.choice(session_pools[int(session)]))
            )
    return indices


def gather(dataset, indices: list[int]):
    eeg = torch.stack([dataset[index][0] for index in indices])
    labels = torch.tensor(
        [int(dataset[index][1]) for index in indices], dtype=torch.long
    )
    return eeg, labels


def class_prototypes(latent: torch.Tensor, labels: torch.Tensor):
    prototypes = []
    for class_id in range(N_CLASSES):
        selected = latent[labels == class_id]
        if selected.numel() == 0:
            raise RuntimeError(f"prototype batch missing class {class_id}")
        prototypes.append(F.normalize(selected.mean(0), dim=0))
    return torch.stack(prototypes)


def supervised_contrastive_loss(
    latent: torch.Tensor, labels: torch.Tensor, temperature: float = 0.10
):
    similarity = latent @ latent.T / temperature
    eye = torch.eye(len(latent), device=latent.device, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None, :]) & ~eye
    counts = positives.sum(1)
    valid = counts > 0
    logits = similarity.masked_fill(eye, float("-inf"))
    log_prob = similarity - torch.logsumexp(logits, dim=1, keepdim=True)
    loss = -(log_prob.masked_fill(~positives, 0.0).sum(1)) / counts.clamp(min=1)
    return loss[valid].mean()


def grl_schedule(epoch: int, epochs: int) -> float:
    progress = min(max(epoch / max(epochs, 1), 0.0), 1.0)
    return float(2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def make_meta_batch(
    spaces: dict[int, SourceEpisodeData],
    selected_subjects: list[int],
    args,
    rng: np.random.RandomState,
):
    eeg_parts = []
    class_parts = []
    domain_parts = []
    subject_parts = []
    ranges = []
    offset = 0
    for sid in selected_subjects:
        space = spaces[sid]
        support_indices = sample_class_session_indices(
            space.vs_pools, args.support_per_class, rng
        )
        query_indices = sample_class_session_indices(
            space.vi_pools, args.query_per_class, rng
        )
        support_eeg, support_labels = gather(space.vs, support_indices)
        query_eeg, query_labels = gather(space.vi_train, query_indices)
        n_support = len(support_labels)
        n_query = len(query_labels)
        eeg_parts.extend([support_eeg, query_eeg])
        class_parts.extend([support_labels, query_labels])
        domain_parts.extend(
            [torch.zeros(n_support, dtype=torch.long),
             torch.ones(n_query, dtype=torch.long)]
        )
        subject_parts.append(
            torch.full(
                (n_support + n_query,),
                space.subject_index,
                dtype=torch.long,
            )
        )
        ranges.append(
            (sid, offset, offset + n_support, offset + n_support + n_query)
        )
        offset += n_support + n_query
    return (
        torch.cat(eeg_parts),
        torch.cat(class_parts),
        torch.cat(domain_parts),
        torch.cat(subject_parts),
        ranges,
    )


def meta_training_loss(
    model: SharedMetaEncoder,
    eeg: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    subject_labels: torch.Tensor,
    ranges: list[tuple[int, int, int, int]],
    args,
    grl_scale: float,
):
    latent = model.encode(eeg)
    episodic_losses = []
    alignment_losses = []
    episode_correct = 0
    episode_total = 0
    for _, start, support_end, query_end in ranges:
        support_latent = latent[start:support_end]
        query_latent = latent[support_end:query_end]
        support_labels = labels[start:support_end]
        query_labels = labels[support_end:query_end]
        support_proto = class_prototypes(support_latent, support_labels)
        query_proto = class_prototypes(query_latent, query_labels)
        logits = query_latent @ support_proto.T / model.temperature()
        episodic_losses.append(F.cross_entropy(logits, query_labels))
        alignment_losses.append(
            (1.0 - (support_proto * query_proto).sum(1)).mean()
        )
        episode_correct += int((logits.argmax(1) == query_labels).sum())
        episode_total += len(query_labels)

    episode_loss = torch.stack(episodic_losses).mean()
    alignment_loss = torch.stack(alignment_losses).mean()
    supcon_loss = supervised_contrastive_loss(latent, labels)
    auxiliary_loss = F.cross_entropy(model.class_head(latent), labels)
    domain_logits, subject_logits = model.adversarial_logits(latent, grl_scale)
    domain_loss = F.cross_entropy(domain_logits, domains)
    subject_loss = F.cross_entropy(subject_logits, subject_labels)
    total = (
        args.w_episode * episode_loss
        + args.w_alignment * alignment_loss
        + args.w_supcon * supcon_loss
        + args.w_aux * auxiliary_loss
        + args.w_domain * domain_loss
        + args.w_subject * subject_loss
    )
    parts = {
        "loss": total,
        "episode": episode_loss,
        "alignment": alignment_loss,
        "supcon": supcon_loss,
        "aux": auxiliary_loss,
        "domain": domain_loss,
        "subject": subject_loss,
        "episode_accuracy": episode_correct / max(episode_total, 1),
    }
    return total, parts


@torch.no_grad()
def encode_dataset(
    model: SharedMetaEncoder,
    dataset,
    batch_size: int,
    device: torch.device,
):
    model.eval()
    latent_parts = []
    label_parts = []
    session_parts = []
    for start in range(0, len(dataset), batch_size):
        items = [
            dataset[index]
            for index in range(start, min(start + batch_size, len(dataset)))
        ]
        eeg = torch.stack([item[0] for item in items]).to(device)
        latent_parts.append(model.encode(eeg).cpu())
        label_parts.append(
            torch.tensor([int(item[1]) for item in items], dtype=torch.long)
        )
        session_parts.append(
            torch.tensor([int(item[2]) for item in items], dtype=torch.long)
        )
    return (
        torch.cat(latent_parts),
        torch.cat(label_parts),
        torch.cat(session_parts),
    )


def deterministic_support_subset(
    dataset: SessionEEGDataset,
    per_class: int,
    seed: int,
):
    selected = []
    for class_id in range(N_CLASSES):
        indices = [
            index
            for index, (_, label, _) in enumerate(dataset.samples)
            if int(label) == class_id
        ]
        rng = np.random.RandomState(seed + class_id)
        rng.shuffle(indices)
        if per_class > 0:
            indices = indices[: min(per_class, len(indices))]
        selected.extend(indices)
    return Subset(dataset, selected)


def classification_metrics(scores: torch.Tensor, labels: torch.Tensor):
    order = scores.argsort(1, descending=True)
    predictions = order[:, 0]
    recalls = []
    confusion = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long)
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        confusion[int(truth), int(prediction)] += 1
    for class_id in range(N_CLASSES):
        mask = labels == class_id
        recalls.append(float((predictions[mask] == class_id).float().mean()))
    counts = torch.bincount(predictions, minlength=N_CLASSES).float()
    probabilities = counts / counts.sum().clamp(min=1)
    nonzero = probabilities > 0
    entropy = float(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
    true_scores = scores.gather(1, labels[:, None]).squeeze(1)
    wrong = scores.masked_fill(F.one_hot(labels, N_CLASSES).bool(), float("-inf"))
    margins = true_scores - wrong.max(1).values
    return {
        "n": int(len(labels)),
        "top1": float((order[:, :1] == labels[:, None]).any(1).float().mean()),
        "top3": float((order[:, :3] == labels[:, None]).any(1).float().mean()),
        "top5": float((order[:, :5] == labels[:, None]).any(1).float().mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "mean_true_similarity": float(true_scores.mean()),
        "mean_true_margin": float(margins.mean()),
        "normalized_prediction_entropy": entropy / math.log(N_CLASSES),
        "dominant_ratio": float(counts.max() / counts.sum().clamp(min=1)),
        "prediction_counts": {
            CLASS_NAMES[i]: int(counts[i]) for i in range(N_CLASSES)
        },
        "confusion_matrix_rows_true_cols_pred": confusion.tolist(),
    }


@torch.no_grad()
def evaluate_vs_support_vi_query(
    model: SharedMetaEncoder,
    vs_support,
    vi_query,
    args,
    device: torch.device,
):
    support_latent, support_labels, _ = encode_dataset(
        model, vs_support, args.eval_batch_size, device
    )
    query_latent, query_labels, query_sessions = encode_dataset(
        model, vi_query, args.eval_batch_size, device
    )
    prototypes = class_prototypes(support_latent, support_labels)
    scores = query_latent @ prototypes.T
    metrics = classification_metrics(scores, query_labels)
    metrics["support_n"] = int(len(support_labels))
    metrics["support_class_counts"] = {
        CLASS_NAMES[class_id]: int((support_labels == class_id).sum())
        for class_id in range(N_CLASSES)
    }
    return metrics, scores, query_labels, query_sessions


def validation_metrics(model, spaces, args, device):
    per_subject = {}
    for sid, space in spaces.items():
        support = deterministic_support_subset(
            space.vs,
            args.eval_support_per_class,
            args.seed + sid * 1000,
        )
        metrics, _, _, _ = evaluate_vs_support_vi_query(
            model, support, space.vi_val, args, device
        )
        per_subject[sid] = metrics
    mean_bac = float(
        np.mean([value["balanced_accuracy"] for value in per_subject.values()])
    )
    mean_top3 = float(
        np.mean([value["top3"] for value in per_subject.values()])
    )
    return mean_bac, mean_top3, per_subject


def cpu_state_dict(module: nn.Module):
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def save_history(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def atomic_torch_save(payload: dict, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def audit(args, subjects, run_dir):
    rows = []
    for sid in subjects:
        vs_sessions = discover_sessions(args.vs_root, sid)
        vi_sessions = discover_sessions(args.vi_root, sid)
        rows.append(
            {
                "subject": sid,
                "role": "target" if sid == args.target_subject else "source",
                "vs_sessions": vs_sessions,
                "vi_sessions": vi_sessions,
            }
        )
    with (run_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    for row in rows:
        print(
            f"S{row['subject']:02d} {row['role']:>6} "
            f"VS={len(row['vs_sessions'])} VI={len(row['vi_sessions'])}"
        )
    print(f"[Saved] {run_dir / 'audit.json'}")


def main():
    args = parse_args()
    subjects = parse_subjects(args.subjects)
    if args.target_subject not in subjects:
        raise ValueError("target_subject must be included in subjects")
    if len(subjects) < 3:
        raise ValueError("at least three subjects are required")
    if min(args.support_per_class, args.query_per_class) < 1:
        raise ValueError("support/query per class must be positive")
    if args.subjects_per_step < 2:
        raise ValueError("subjects_per_step must be at least two")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_subjects = [sid for sid in subjects if sid != args.target_subject]
    run_dir = (
        Path(args.out_root)
        / f"target_S{args.target_subject:02d}"
        / f"seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    progress_path = run_dir / "progress.pt"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite to rerun")

    print(
        f"[INFO] target-VI held out S{args.target_subject:02d}; "
        f"sources={source_subjects}; device={device}",
        flush=True,
    )
    if args.audit_only:
        audit(args, subjects, run_dir)
        return

    spaces = {}
    for subject_index, sid in enumerate(source_subjects):
        print(f"\n[Prepare source S{sid:02d}]", flush=True)
        spaces[sid] = prepare_source(args, sid, subject_index)

    model = SharedMetaEncoder(
        n_source_subjects=len(source_subjects),
        hidden_dim=args.hidden_dim,
        eeg_out=args.eeg_out,
        latent_dim=args.latent_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01
    )
    use_fp16 = bool(args.fp16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    augmenter = EEGAugment(
        noise_std=0.03,
        scale_range=(0.9, 1.1),
        ch_drop_prob=0.05,
        max_shift=16,
        freq_noise_std=0.0,
        p_noise=0.4,
        p_scale=0.4,
        p_drop=0.2,
        p_shift=0.2,
        p_freq=0.0,
    )
    rng = np.random.RandomState(args.seed)
    best_key = (-float("inf"), -float("inf"))
    best_epoch = 0
    best_state = None
    stale_evaluations = 0
    history = []
    start_epoch = 1

    if args.resume and progress_path.is_file():
        progress = torch.load(
            progress_path, map_location=device, weights_only=False
        )
        if progress["target_subject"] != args.target_subject:
            raise RuntimeError("progress target subject mismatch")
        if progress["source_subjects"] != source_subjects:
            raise RuntimeError("progress source subjects mismatch")
        model.load_state_dict(progress["model"], strict=True)
        optimizer.load_state_dict(progress["optimizer"])
        scheduler.load_state_dict(progress["scheduler"])
        scaler.load_state_dict(progress["scaler"])
        best_key = tuple(float(value) for value in progress["best_key"])
        best_epoch = int(progress["best_epoch"])
        best_state = {
            name: value.detach().cpu()
            for name, value in progress["best_state"].items()
        }
        stale_evaluations = int(progress["stale_evaluations"])
        history = list(progress["history"])
        rng.set_state(progress["numpy_rng_state"])
        torch.set_rng_state(progress["torch_rng_state"].cpu())
        if device.type == "cuda" and progress.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(progress["cuda_rng_states"])
        start_epoch = int(progress["epoch"]) + 1
        print(f"[RESUME] epoch={start_epoch} best={best_epoch}", flush=True)
    elif args.resume:
        print("[RESUME] no progress.pt; starting at epoch 1", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "episode": 0.0,
            "alignment": 0.0,
            "supcon": 0.0,
            "aux": 0.0,
            "domain": 0.0,
            "subject": 0.0,
            "episode_accuracy": 0.0,
        }
        for _ in range(args.steps_per_epoch):
            replace = len(source_subjects) < args.subjects_per_step
            selected = rng.choice(
                source_subjects,
                size=args.subjects_per_step,
                replace=replace,
            ).tolist()
            eeg, labels, domains, subject_labels, ranges = make_meta_batch(
                spaces, selected, args, rng
            )
            eeg = augmenter(eeg.to(device))
            labels = labels.to(device)
            domains = domains.to(device)
            subject_labels = subject_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_fp16):
                loss, parts = meta_training_loss(
                    model,
                    eeg,
                    labels,
                    domains,
                    subject_labels,
                    ranges,
                    args,
                    grl_schedule(epoch, args.epochs),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            for name in totals:
                value = parts[name]
                totals[name] += (
                    float(value.detach())
                    if isinstance(value, torch.Tensor)
                    else float(value)
                )
        scheduler.step()
        means = {
            name: value / args.steps_per_epoch
            for name, value in totals.items()
        }
        should_evaluate = epoch == 1 or epoch % args.eval_interval == 0
        row = {
            "epoch": epoch,
            **{f"train_{name}": value for name, value in means.items()},
            "val_mean_bac": "",
            "val_mean_top3": "",
        }
        if should_evaluate:
            mean_bac, mean_top3, val_by_subject = validation_metrics(
                model, spaces, args, device
            )
            row["val_mean_bac"] = mean_bac
            row["val_mean_top3"] = mean_top3
            current_key = (mean_bac, mean_top3)
            if current_key > best_key:
                best_key = current_key
                best_epoch = epoch
                best_state = cpu_state_dict(model)
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            print(
                f"Ep {epoch:03d} loss={means['loss']:.4f} "
                f"episodeAcc={means['episode_accuracy']:.4f} "
                f"valMeanBAC={mean_bac:.4f} top3={mean_top3:.4f} "
                f"best={best_epoch}",
                flush=True,
            )
        history.append(row)
        save_history(run_dir / "history.csv", history)
        progress = {
            "epoch": epoch,
            "target_subject": args.target_subject,
            "source_subjects": source_subjects,
            "model": cpu_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_key": best_key,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "stale_evaluations": stale_evaluations,
            "history": history,
            "numpy_rng_state": rng.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else None
            ),
        }
        atomic_torch_save(progress, progress_path)
        if should_evaluate and stale_evaluations >= args.patience:
            print(f"[Early stop] epoch={epoch} best={best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("training produced no selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    final_val_bac, final_val_top3, final_val_subjects = validation_metrics(
        model, spaces, args, device
    )

    del spaces
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # The target VI contents are first loaded here, after model selection.
    target_vs_sessions = discover_sessions(args.vs_root, args.target_subject)
    target_vi_sessions = discover_sessions(args.vi_root, args.target_subject)
    target_vs = SessionEEGDataset(
        args.vs_root,
        args.target_subject,
        target_vs_sessions,
        f"S{args.target_subject:02d}_target_vs_support",
    )
    target_vi = SessionEEGDataset(
        args.vi_root,
        args.target_subject,
        target_vi_sessions,
        f"S{args.target_subject:02d}_target_vi_query",
    )
    target_support = deterministic_support_subset(
        target_vs, 0, args.seed + args.target_subject * 1000
    )
    target_metrics, scores, labels, sessions = evaluate_vs_support_vi_query(
        model, target_support, target_vi, args, device
    )
    class_head_scores = []
    with torch.no_grad():
        latent, _, _ = encode_dataset(
            model, target_vi, args.eval_batch_size, device
        )
        for start in range(0, len(latent), args.eval_batch_size):
            class_head_scores.append(
                model.class_head(
                    latent[start : start + args.eval_batch_size].to(device)
                ).cpu()
            )
    class_head_metrics = classification_metrics(
        torch.cat(class_head_scores), labels
    )
    per_session = {}
    for session in sorted(set(int(value) for value in sessions.tolist())):
        mask = sessions == session
        per_session[str(session)] = classification_metrics(
            scores[mask], labels[mask]
        )

    checkpoint = {
        "model": best_state,
        "target_subject": args.target_subject,
        "source_subjects": source_subjects,
        "best_epoch": best_epoch,
        "best_validation_bac": best_key[0],
        "args": vars(args),
    }
    torch.save(checkpoint, run_dir / "best.pt")
    metrics = {
        "protocol": (
            "episodic source-subject VS support to VI query; target VI held "
            "out; target VS support allowed"
        ),
        "target_subject": args.target_subject,
        "source_subjects": source_subjects,
        "selection": {
            "primary": "mean source held-out VI-session balanced accuracy",
            "tie_break": "mean source held-out VI-session top-3",
            "best_epoch": best_epoch,
            "best_validation_bac": best_key[0],
            "best_validation_top3": best_key[1],
        },
        "final_source_validation": {
            "mean_bac": final_val_bac,
            "mean_top3": final_val_top3,
            "per_subject": {
                str(sid): value for sid, value in final_val_subjects.items()
            },
        },
        "target_vs_sessions": target_vs_sessions,
        "target_vi_sessions": target_vi_sessions,
        "target_vs_prototype_metrics": target_metrics,
        "target_class_head_diagnostic": class_head_metrics,
        "per_target_vi_session": per_session,
        "chance": 1.0 / N_CLASSES,
        "target_vi_loaded_after_selection": True,
        "reporting_warning": (
            "Primary inference uses target VS prototypes. The target class "
            "head is diagnostic only because it was trained on source subjects."
        ),
        "args": vars(args),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metrics["selection"], indent=2), flush=True)
    print(
        "Target VS-prototype -> VI: "
        f"BAC={target_metrics['balanced_accuracy']:.4f} "
        f"top3={target_metrics['top3']:.4f} "
        f"top5={target_metrics['top5']:.4f}",
        flush=True,
    )
    print(f"[Saved] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
