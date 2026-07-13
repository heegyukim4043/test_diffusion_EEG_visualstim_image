"""Multi-subject universal VS-to-VI signal adapter with target-VI LOSO.

Protocol
--------
For one target subject, no target VI trial is used for training or checkpoint
selection.  A single raw-signal adapter is trained on the other subjects.  For
each training subject, its frozen subject-specific VS SupCon encoder, frozen
VS LoRA condition projector, and VS class centroids provide the teacher space.
The same adapter must therefore learn a subject-shared VI-to-VS correction.

Within every training subject, the last valid VI session is validation and all
remaining VI sessions are training.  Each optimization example contains four
same-class VI trials sampled from different sessions.  Selection uses the mean
validation token balanced accuracy across training subjects.  Only after
selection are the target subject's VI sessions loaded and evaluated.

The target subject's VS encoder/projector and VS trials are allowed: this is a
target-VI-held-out transfer protocol, not a strict all-modality subject LOSO.
Stable Diffusion is not loaded and every VS model remains exactly frozen.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

from train_exp43_vi_lora import find_init_lora_ckpt
from train_vi_specific_frontend import (
    SessionEEGDataset,
    build_class_session_pools,
    centroid_diagnostics,
    discover_sessions,
    prediction_diagnostics,
    target_margin_loss,
    token_features,
)
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


@dataclass
class SubjectSpace:
    sid: int
    encoder: nn.Module
    cond_proj: nn.Module
    latent_centroids: torch.Tensor
    token_centroids: torch.Tensor
    vi_train: SessionEEGDataset
    vi_val: SessionEEGDataset
    train_pools: dict
    train_sessions: list[int]
    val_session: int
    encoder_checkpoint: str
    lora_checkpoint: str


class UniversalVISignalAdapter(nn.Module):
    """Identity-initialized multi-scale temporal/channel signal correction."""

    def __init__(self, channels: int = 32):
        super().__init__()
        kernels = (7, 31, 127)
        self.depthwise = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=channels,
                    bias=False,
                )
                for kernel in kernels
            ]
        )
        self.mix = nn.Conv1d(channels * len(kernels), channels, 1, bias=True)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    @staticmethod
    def normalize(eeg: torch.Tensor) -> torch.Tensor:
        return (eeg - eeg.mean(-1, keepdim=True)) / (
            eeg.std(-1, keepdim=True) + 1e-6
        )

    def forward_with_delta(self, eeg: torch.Tensor):
        normalized = self.normalize(eeg)
        branches = [F.gelu(layer(normalized)) for layer in self.depthwise]
        delta = self.mix(torch.cat(branches, dim=1))
        return normalized + delta, delta

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        adapted, _ = self.forward_with_delta(eeg)
        return adapted


def parse_csv_ints(text: str) -> list[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError("empty subject list")
    return values


def parse_roots(text: str) -> list[str]:
    return [token.strip() for token in text.split(",") if token.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_subject", type=int, required=True)
    parser.add_argument("--subjects", default="1,2,9,18,24,28,29")
    parser.add_argument("--vs_root", required=True)
    parser.add_argument("--vi_root", required=True)
    parser.add_argument(
        "--supcon_roots",
        default=(
            "/content/drive/MyDrive/vsvi_data/checkpoints_vsre_dino,"
            "/content/vsvi_project/checkpoints_vsre_dino"
        ),
    )
    parser.add_argument(
        "--vs_lora_roots",
        default=(
            "/content/drive/MyDrive/vsvi_data/checkpoints_vsre_lora_gen,"
            "/content/vsvi_project/checkpoints_vsre_lora_gen"
        ),
    )
    parser.add_argument("--n_eeg_tokens", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--samples_per_class", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--prototype_margin", type=float, default=0.10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--w_individual_latent", type=float, default=0.25)
    parser.add_argument("--w_individual_token", type=float, default=0.25)
    parser.add_argument("--w_individual_aux", type=float, default=0.25)
    parser.add_argument("--w_prototype_latent", type=float, default=1.0)
    parser.add_argument("--w_prototype_token", type=float, default=1.0)
    parser.add_argument("--w_consistency", type=float, default=0.25)
    parser.add_argument("--w_margin", type=float, default=0.50)
    parser.add_argument("--w_identity", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--audit_only", action="store_true")
    parser.add_argument(
        "--out_root",
        default=(
            "/content/drive/MyDrive/vsvi_data/"
            "multisubject_universal_vsvi_adapter"
        ),
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


def checkpoint_timestamp(path: Path) -> int:
    match = re.search(r"(20\d{6})_(\d{6})", str(path.parent))
    return int("".join(match.groups())) if match else 0


def resolve_supcon_checkpoint(roots: list[str], sid: int) -> Path:
    candidates = []
    for root_text in roots:
        root = Path(root_text)
        if root.exists():
            candidates.extend(root.rglob(f"subj{sid:02d}_best.pt"))
    unique = {str(path.resolve()): path for path in candidates}
    candidates = list(unique.values())
    if not candidates:
        raise FileNotFoundError(f"No SupCon checkpoint for S{sid:02d}")

    def key(path: Path):
        drive_priority = int("/drive/" in str(path).replace("\\", "/"))
        return checkpoint_timestamp(path), drive_priority, str(path)

    selected = max(candidates, key=key)
    print(f"[Resolve SupCon S{sid:02d}] {selected}", flush=True)
    if len(candidates) > 1:
        for candidate in sorted(candidates, key=key, reverse=True):
            print(f"  candidate: {candidate}", flush=True)
    return selected


def resolve_lora_checkpoint(
    roots: list[str], sid: int, lora_r: int
) -> Path:
    selected = find_init_lora_ckpt(
        roots=roots, sid=sid, lora_r=lora_r, explicit=None
    )
    return Path(selected)


def resolve_all_checkpoints(args, subjects: list[int]) -> dict[int, dict[str, str]]:
    supcon_roots = parse_roots(args.supcon_roots)
    lora_roots = parse_roots(args.vs_lora_roots)
    resolved = {}
    for sid in subjects:
        supcon = resolve_supcon_checkpoint(supcon_roots, sid)
        lora = resolve_lora_checkpoint(lora_roots, sid, args.lora_r)
        resolved[sid] = {"supcon": str(supcon), "vs_lora": str(lora)}
        print(f"[Resolve LoRA S{sid:02d}] {lora}", flush=True)
    return resolved


def load_subject_source(
    sid: int,
    checkpoints: dict[str, str],
    n_eeg_tokens: int,
    device: torch.device,
):
    namespace = SimpleNamespace(
        subject_id=sid,
        supcon_ckpt=str(Path(checkpoints["supcon"]).parent),
        vs_lora_ckpt=checkpoints["vs_lora"],
        n_eeg_tokens=n_eeg_tokens,
    )
    encoder, cond_proj, encoder_path, lora_path = load_frozen_source(
        namespace, device
    )
    return encoder, cond_proj, encoder_path, lora_path


def build_vs_centroids(
    args,
    sid: int,
    encoder: nn.Module,
    cond_proj: nn.Module,
    device: torch.device,
):
    dataset = make_dataset(args.vs_root, sid, "train", args.seed, None)
    latent, labels = encode_dataset(
        dataset, encoder, args.eval_batch_size, device
    )
    latent_centroids, token_centroids = build_source_centroids(
        latent, labels, cond_proj, device
    )
    del dataset, latent, labels
    gc.collect()
    return latent_centroids.to(device), token_centroids.to(device)


def choose_validation_dataset(args, sid: int, sessions: list[int]):
    errors = []
    for candidate in sorted(sessions, reverse=True):
        try:
            dataset = SessionEEGDataset(
                args.vi_root, sid, [candidate], f"S{sid:02d}_vi_val"
            )
            if min(dataset.class_counts.values()) < 2:
                raise RuntimeError("fewer than two finite trials in a class")
            return candidate, dataset
        except Exception as error:
            errors.append(f"session {candidate}: {error}")
    raise RuntimeError(
        f"No valid validation session for S{sid:02d}: " + " | ".join(errors)
    )


def prepare_training_subject(
    args,
    sid: int,
    checkpoints: dict[str, str],
    device: torch.device,
) -> SubjectSpace:
    encoder, cond_proj, encoder_path, lora_path = load_subject_source(
        sid, checkpoints, args.n_eeg_tokens, device
    )
    latent_centroids, token_centroids = build_vs_centroids(
        args, sid, encoder, cond_proj, device
    )
    sessions = discover_sessions(args.vi_root, sid)
    if len(sessions) < 2:
        raise RuntimeError(f"S{sid:02d} needs at least two VI sessions")
    val_session, vi_val = choose_validation_dataset(args, sid, sessions)
    train_sessions = [session for session in sessions if session != val_session]
    vi_train = SessionEEGDataset(
        args.vi_root, sid, train_sessions, f"S{sid:02d}_vi_train"
    )
    pools = build_class_session_pools(vi_train)
    return SubjectSpace(
        sid=sid,
        encoder=encoder,
        cond_proj=cond_proj,
        latent_centroids=latent_centroids,
        token_centroids=token_centroids,
        vi_train=vi_train,
        vi_val=vi_val,
        train_pools=pools,
        train_sessions=train_sessions,
        val_session=val_session,
        encoder_checkpoint=str(encoder_path),
        lora_checkpoint=str(lora_path),
    )


def sample_subject_class(
    space: SubjectSpace,
    class_id: int,
    samples_per_class: int,
    rng: np.random.RandomState,
):
    session_pools = space.train_pools[class_id]
    sessions = sorted(session_pools)
    chosen = rng.choice(
        sessions,
        size=samples_per_class,
        replace=len(sessions) < samples_per_class,
    )
    indices = [
        int(rng.choice(session_pools[int(session)])) for session in chosen
    ]
    eeg = torch.stack([space.vi_train[index][0] for index in indices])
    labels = torch.full((samples_per_class,), class_id, dtype=torch.long)
    return eeg, labels


def encode_subject(
    encoder: nn.Module,
    cond_proj: nn.Module,
    adapter: UniversalVISignalAdapter | None,
    eeg: torch.Tensor,
):
    subject_ids = torch.zeros(len(eeg), dtype=torch.long, device=eeg.device)
    if adapter is not None:
        eeg = adapter(eeg)
    latent = encoder.encode_eeg(eeg, subject_ids)
    tokens = token_features(cond_proj, latent)
    return latent, tokens


@torch.no_grad()
def evaluate_subject(
    dataset,
    space,
    adapter,
    batch_size: int,
    device: torch.device,
):
    space.encoder.eval()
    space.cond_proj.eval()
    if adapter is not None:
        adapter.eval()
    latent_parts = []
    token_parts = []
    aux_parts = []
    label_parts = []
    for start in range(0, len(dataset), batch_size):
        items = [
            dataset[index]
            for index in range(start, min(start + batch_size, len(dataset)))
        ]
        eeg = torch.stack([item[0] for item in items]).to(device)
        labels = torch.tensor([int(item[1]) for item in items], dtype=torch.long)
        latent, tokens = encode_subject(
            space.encoder, space.cond_proj, adapter, eeg
        )
        latent_parts.append(latent.cpu())
        token_parts.append(tokens.cpu())
        aux_parts.append(space.encoder.aux_cls_head(latent).cpu())
        label_parts.append(labels)

    latent = torch.cat(latent_parts)
    tokens = torch.cat(token_parts)
    aux_logits = torch.cat(aux_parts)
    labels = torch.cat(label_parts)
    latent_centroids = space.latent_centroids.cpu()
    token_centroids = space.token_centroids.cpu()
    latent_scores = F.normalize(latent, dim=1) @ latent_centroids.T
    token_scores = F.normalize(tokens, dim=1) @ token_centroids.T
    return {
        "n": int(len(labels)),
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
        "token_prediction_diagnostics": prediction_diagnostics(
            token_scores, labels
        ),
    }


def mean_validation_metrics(spaces, adapter, args, device):
    per_subject = {}
    for sid, space in spaces.items():
        metrics = evaluate_subject(
            space.vi_val,
            space,
            adapter,
            args.eval_batch_size,
            device,
        )
        per_subject[sid] = metrics
    mean_token = float(
        np.mean(
            [
                metrics["vs_token_centroid"]["balanced_accuracy"]
                for metrics in per_subject.values()
            ]
        )
    )
    mean_latent = float(
        np.mean(
            [
                metrics["vs_latent_centroid"]["balanced_accuracy"]
                for metrics in per_subject.values()
            ]
        )
    )
    return mean_token, mean_latent, per_subject


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


def aggregate_target_sessions(target_dataset, space, adapter, args, device):
    session_to_indices = {}
    for index, (_, _, session) in enumerate(target_dataset.samples):
        session_to_indices.setdefault(int(session), []).append(index)
    results = {}
    for session, indices in sorted(session_to_indices.items()):
        subset = Subset(target_dataset, indices)
        try:
            raw = evaluate_subject(
                subset, space, None, args.eval_batch_size, device
            )
            adapted = evaluate_subject(
                subset, space, adapter, args.eval_batch_size, device
            )
            results[str(session)] = {"raw": raw, "adapted": adapted}
        except Exception as error:
            results[str(session)] = {"error": str(error)}
    return results


def main() -> None:
    args = parse_args()
    subjects = parse_csv_ints(args.subjects)
    if args.target_subject not in subjects:
        raise ValueError("target_subject must be included in subjects")
    if len(subjects) < 3:
        raise ValueError("at least three subjects are required")
    if args.samples_per_class < 2:
        raise ValueError("samples_per_class must be at least two")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_subjects = [sid for sid in subjects if sid != args.target_subject]
    run_dir = (
        Path(args.out_root)
        / f"target_S{args.target_subject:02d}"
        / f"seed{args.seed}"
    )
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite to rerun")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] target-VI held out S{args.target_subject:02d}; "
        f"train subjects={train_subjects}; device={device}",
        flush=True,
    )
    checkpoints = resolve_all_checkpoints(args, subjects)
    with (run_dir / "resolved_checkpoints.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(checkpoints, handle, indent=2)
    if args.audit_only:
        print(f"[AUDIT ONLY] Saved {run_dir / 'resolved_checkpoints.json'}")
        return

    spaces = {}
    for sid in train_subjects:
        print(f"\n[Prepare training subject S{sid:02d}]", flush=True)
        spaces[sid] = prepare_training_subject(
            args, sid, checkpoints[sid], device
        )

    adapter = UniversalVISignalAdapter().to(device)
    n_trainable = sum(
        parameter.numel()
        for parameter in adapter.parameters()
        if parameter.requires_grad
    )
    print(f"[INFO] universal adapter trainable={n_trainable:,}", flush=True)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    use_fp16 = bool(args.fp16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    rng = np.random.RandomState(args.seed)

    best_key = (-float("inf"), -float("inf"))
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    subject_class_pairs = [
        (sid, class_id)
        for sid in train_subjects
        for class_id in range(N_CLASSES)
    ]

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        order = rng.permutation(len(subject_class_pairs))
        totals = {
            "loss": 0.0,
            "individual_latent": 0.0,
            "individual_token": 0.0,
            "individual_aux": 0.0,
            "prototype_latent": 0.0,
            "prototype_token": 0.0,
            "consistency": 0.0,
            "margin": 0.0,
            "identity": 0.0,
        }
        for pair_index in order:
            sid, class_id = subject_class_pairs[int(pair_index)]
            space = spaces[sid]
            eeg_cpu, labels_cpu = sample_subject_class(
                space, class_id, args.samples_per_class, rng
            )
            eeg = eeg_cpu.to(device)
            labels = labels_cpu.to(device)
            target = torch.tensor([class_id], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_fp16):
                adapted_eeg, delta = adapter.forward_with_delta(eeg)
                subject_ids = torch.zeros(
                    len(eeg), dtype=torch.long, device=device
                )
                latent = space.encoder.encode_eeg(adapted_eeg, subject_ids)
                tokens = token_features(space.cond_proj, latent)
                latent_logits = (
                    latent @ space.latent_centroids.T / args.temperature
                )
                token_logits = (
                    tokens @ space.token_centroids.T / args.temperature
                )
                individual_latent = F.cross_entropy(latent_logits, labels)
                individual_token = F.cross_entropy(token_logits, labels)
                individual_aux = F.cross_entropy(
                    space.encoder.aux_cls_head(latent), labels
                )
                latent_proto = F.normalize(latent.mean(0), dim=0)[None, :]
                token_proto = F.normalize(tokens.mean(0), dim=0)[None, :]
                latent_scores = latent_proto @ space.latent_centroids.T
                token_scores = token_proto @ space.token_centroids.T
                prototype_latent = F.cross_entropy(
                    latent_scores / args.temperature, target
                )
                prototype_token = F.cross_entropy(
                    token_scores / args.temperature, target
                )
                consistency = 0.5 * (
                    (1.0 - F.cosine_similarity(
                        latent, latent_proto.expand_as(latent), dim=1
                    )).mean()
                    + (1.0 - F.cosine_similarity(
                        tokens, token_proto.expand_as(tokens), dim=1
                    )).mean()
                )
                margin_loss = 0.5 * (
                    target_margin_loss(
                        latent_scores, target, args.prototype_margin
                    )
                    + target_margin_loss(
                        token_scores, target, args.prototype_margin
                    )
                )
                identity = delta.square().mean()
                loss = (
                    args.w_individual_latent * individual_latent
                    + args.w_individual_token * individual_token
                    + args.w_individual_aux * individual_aux
                    + args.w_prototype_latent * prototype_latent
                    + args.w_prototype_token * prototype_token
                    + args.w_consistency * consistency
                    + args.w_margin * margin_loss
                    + args.w_identity * identity
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            values = {
                "loss": loss,
                "individual_latent": individual_latent,
                "individual_token": individual_token,
                "individual_aux": individual_aux,
                "prototype_latent": prototype_latent,
                "prototype_token": prototype_token,
                "consistency": consistency,
                "margin": margin_loss,
                "identity": identity,
            }
            for name, value in values.items():
                totals[name] += float(value.detach().item())

        pair_count = len(subject_class_pairs)
        means = {name: value / pair_count for name, value in totals.items()}
        mean_token, mean_latent, val_by_subject = mean_validation_metrics(
            spaces, adapter, args, device
        )
        row = {
            "epoch": epoch,
            **{f"train_{name}": value for name, value in means.items()},
            "val_mean_token_bac": mean_token,
            "val_mean_latent_bac": mean_latent,
        }
        for sid, metrics in val_by_subject.items():
            row[f"val_S{sid:02d}_token_bac"] = metrics[
                "vs_token_centroid"
            ]["balanced_accuracy"]
        history.append(row)
        current_key = (mean_token, mean_latent)
        if current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            best_state = cpu_state_dict(adapter)
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0 or stale == 0:
            print(
                f"Ep {epoch:03d} loss={means['loss']:.4f} "
                f"valMean latentBAC={mean_latent:.4f} "
                f"tokenBAC={mean_token:.4f} best={best_epoch}",
                flush=True,
            )
        if stale >= args.patience:
            print(f"[Early stop] epoch={epoch} best={best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training produced no adapter checkpoint")
    adapter.load_state_dict(best_state, strict=True)
    final_val_token, final_val_latent, final_val_by_subject = (
        mean_validation_metrics(spaces, adapter, args, device)
    )

    # Release training subject models/data before the untouched target VI load.
    del spaces
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"\n[Final target evaluation S{args.target_subject:02d}]", flush=True
    )
    target_encoder, target_proj, encoder_path, lora_path = load_subject_source(
        args.target_subject,
        checkpoints[args.target_subject],
        args.n_eeg_tokens,
        device,
    )
    target_latent_centroids, target_token_centroids = build_vs_centroids(
        args,
        args.target_subject,
        target_encoder,
        target_proj,
        device,
    )
    target_sessions = discover_sessions(args.vi_root, args.target_subject)
    target_vi = SessionEEGDataset(
        args.vi_root,
        args.target_subject,
        target_sessions,
        f"S{args.target_subject:02d}_target_vi_all",
    )
    target_space = SimpleNamespace(
        encoder=target_encoder,
        cond_proj=target_proj,
        latent_centroids=target_latent_centroids,
        token_centroids=target_token_centroids,
    )
    raw_target = evaluate_subject(
        target_vi, target_space, None, args.eval_batch_size, device
    )
    adapted_target = evaluate_subject(
        target_vi, target_space, adapter, args.eval_batch_size, device
    )
    per_session = aggregate_target_sessions(
        target_vi, target_space, adapter, args, device
    )

    checkpoint = {
        "adapter": best_state,
        "target_subject": args.target_subject,
        "train_subjects": train_subjects,
        "best_epoch": best_epoch,
        "best_mean_val_token_bac": best_key[0],
        "best_mean_val_latent_bac": best_key[1],
        "resolved_checkpoints": checkpoints,
        "args": vars(args),
    }
    torch.save(checkpoint, run_dir / "best.pt")
    save_history(run_dir / "history.csv", history)
    metrics = {
        "protocol": "target VI subject held out; target VS model allowed",
        "target_subject": args.target_subject,
        "target_sessions": target_sessions,
        "train_subjects": train_subjects,
        "selection": {
            "primary": "mean training-subject VI-validation token BAC",
            "tie_break": "mean training-subject VI-validation latent BAC",
            "best_epoch": best_epoch,
            "best_mean_val_token_bac": best_key[0],
            "best_mean_val_latent_bac": best_key[1],
        },
        "final_validation": {
            "mean_token_bac": final_val_token,
            "mean_latent_bac": final_val_latent,
            "per_subject": {
                str(sid): value for sid, value in final_val_by_subject.items()
            },
        },
        "raw_target": raw_target,
        "adapted_target": adapted_target,
        "delta_target": {
            "latent_bac": (
                adapted_target["vs_latent_centroid"]["balanced_accuracy"]
                - raw_target["vs_latent_centroid"]["balanced_accuracy"]
            ),
            "token_bac": (
                adapted_target["vs_token_centroid"]["balanced_accuracy"]
                - raw_target["vs_token_centroid"]["balanced_accuracy"]
            ),
            "latent_separation": (
                adapted_target["latent_within_minus_between"]
                - raw_target["latent_within_minus_between"]
            ),
            "token_separation": (
                adapted_target["token_within_minus_between"]
                - raw_target["token_within_minus_between"]
            ),
        },
        "per_target_session": per_session,
        "target_encoder_checkpoint": str(encoder_path),
        "target_lora_checkpoint": str(lora_path),
        "resolved_checkpoints": checkpoints,
        "reporting_warning": (
            "Hungarian-aligned accuracy is diagnostic only. The target VS "
            "model is allowed, so this is target-VI LOSO, not strict subject LOSO."
        ),
        "args": vars(args),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print(json.dumps(metrics["selection"], indent=2), flush=True)
    print(
        "Target token BAC raw/adapted: "
        f"{raw_target['vs_token_centroid']['balanced_accuracy']:.4f} / "
        f"{adapted_target['vs_token_centroid']['balanced_accuracy']:.4f}",
        flush=True,
    )
    print(f"[Saved] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
