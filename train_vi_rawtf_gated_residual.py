"""VI-primary gated residual transfer from a frozen VS raw+TF encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_vs_re import session_counts
from train_vi_tf_representation_ablation import (
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


SELECTION_DECIMALS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", required=True, type=int)
    parser.add_argument("--vi_root", required=True)
    parser.add_argument("--vi_ckpt", required=True)
    parser.add_argument("--vs_ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling_rate", type=float, default=1024.0)
    parser.add_argument("--n_ch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--residual_only_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--student_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--w_supcon", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--gate_penalty", type=float, default=1e-3)
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
    args.representation = "raw_tf"
    return args


def load_matching(model: RepresentationEncoder, path: Path, label: str) -> dict:
    checkpoint = safe_torch_load(path, "cpu")
    representation = checkpoint.get("config", {}).get("representation")
    if representation != "raw_tf":
        raise RuntimeError(f"{label} checkpoint is {representation!r}, expected 'raw_tf'")
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint


class GatedResidualTransfer(nn.Module):
    def __init__(
        self,
        student: RepresentationEncoder,
        teacher: RepresentationEncoder,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.adapter = nn.Linear(latent_dim, latent_dim)
        with torch.no_grad():
            self.adapter.weight.copy_(torch.eye(latent_dim))
            self.adapter.bias.zero_()
        self.gate_logit = nn.Parameter(torch.zeros(()))
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()

    @property
    def gate(self) -> torch.Tensor:
        return torch.tanh(self.gate_logit)

    def forward_features(
        self, eeg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vi_latent = self.student.encode(eeg)
        with torch.no_grad():
            vs_latent = self.teacher.encode(eeg)
        residual = self.gate * self.adapter(vs_latent)
        fused = F.normalize(vi_latent + residual, dim=1)
        return fused, vi_latent, vs_latent, residual

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fused, _, _, _ = self.forward_features(eeg)
        return self.student.classifier(fused), fused


def configure_phase(model: GatedResidualTransfer, phase: str) -> None:
    for parameter in model.teacher.parameters():
        parameter.requires_grad_(False)
    if phase == "residual_only":
        for parameter in model.student.parameters():
            parameter.requires_grad_(False)
    else:
        for parameter in model.student.parameters():
            parameter.requires_grad_(True)
    for parameter in model.adapter.parameters():
        parameter.requires_grad_(True)
    model.gate_logit.requires_grad_(True)


def set_train_mode(model: GatedResidualTransfer, phase: str) -> None:
    model.eval()
    model.adapter.train()
    if phase == "joint":
        model.student.train()
    model.teacher.eval()


def build_optimizer(
    model: GatedResidualTransfer, phase: str, args: argparse.Namespace
) -> torch.optim.Optimizer:
    groups = [
        {"params": model.adapter.parameters(), "lr": args.lr, "name": "adapter"},
        {"params": [model.gate_logit], "lr": args.lr, "name": "gate"},
    ]
    if phase == "joint":
        groups.extend([
            {
                "params": model.student.classifier.parameters(),
                "lr": args.lr,
                "name": "classifier",
            },
            {
                "params": [
                    parameter
                    for name, parameter in model.student.named_parameters()
                    if not name.startswith("classifier.")
                ],
                "lr": args.lr * args.student_lr_ratio,
                "name": "student_backbone",
            },
        ])
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def score(metrics: dict) -> tuple[float, float, float]:
    return tuple(
        round(float(metrics[key]), SELECTION_DECIMALS)
        for key in ("balanced_accuracy", "top3", "top5")
    )


@torch.no_grad()
def feature_diagnostics(model, loader, device: torch.device) -> dict:
    model.eval()
    cosine_sum = residual_sum = 0.0
    n = 0
    for eeg, _, _ in loader:
        eeg = eeg.to(device, non_blocking=True)
        _, vi_latent, vs_latent, residual = model.forward_features(eeg)
        batch = eeg.size(0)
        cosine_sum += float(F.cosine_similarity(vi_latent, vs_latent).sum())
        residual_sum += float(residual.norm(dim=1).sum())
        n += batch
    return {
        "gate_value": float(model.gate.detach().cpu()),
        "mean_vi_vs_cosine": cosine_sum / max(n, 1),
        "mean_residual_norm": residual_sum / max(n, 1),
    }


def save_candidate(
    path: Path,
    model: GatedResidualTransfer,
    args: argparse.Namespace,
    epoch: int,
    phase: str,
    validation: dict,
) -> None:
    torch.save({
        "model": model.state_dict(),
        "config": vars(args),
        "best_epoch": epoch,
        "best_phase": phase,
        "best_validation": validation,
        "gate_value": float(model.gate.detach().cpu()),
    }, path)


def train(model, loaders, args, device, checkpoint_path):
    # Epoch 0 is the untouched VI-only checkpoint and is a valid candidate.
    initial_validation, _, _ = evaluate(model, loaders["val"], device)
    best_score = score(initial_validation)
    best_epoch = 0
    best_phase = "vi_only_epoch0"
    best_validation = initial_validation
    save_candidate(
        checkpoint_path, model, args, best_epoch, best_phase, best_validation
    )
    print(
        f"[Epoch 0 VI-only] val_BAC={initial_validation['balanced_accuracy']:.4f} "
        f"@3={initial_validation['top3']:.4f}",
        flush=True,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")
    augmenter = EEGWaveAugment()
    history = [{
        "epoch": 0,
        "phase": best_phase,
        "train_loss": "",
        "train_ce": "",
        "train_supcon": "",
        "gate_value": 0.0,
        **{f"val_{key}": initial_validation[key] for key in (
            "balanced_accuracy", "top1", "top3", "top5",
            "normalized_entropy", "dominant_ratio",
        )},
    }]
    current_phase = ""
    optimizer = None
    joint_start = args.residual_only_epochs + 1

    for epoch in range(1, args.epochs + 1):
        phase = "residual_only" if epoch <= args.residual_only_epochs else "joint"
        if phase != current_phase:
            configure_phase(model, phase)
            optimizer = build_optimizer(model, phase, args)
            current_phase = phase
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[Phase] epoch={epoch} {phase} trainable={trainable:,}", flush=True)
        assert optimizer is not None
        set_train_mode(model, phase)
        total_loss = total_ce = total_supcon = 0.0
        total_seen = 0
        for eeg, _, labels in loaders["train"]:
            eeg = eeg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if not args.no_aug:
                eeg = augmenter(eeg)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, args.fp16):
                logits, fused = model(eeg)
                loss_ce = F.cross_entropy(
                    logits, labels, label_smoothing=args.label_smoothing
                )
                loss_supcon = supervised_contrastive_loss(
                    fused, labels, args.temperature
                )
                loss = (
                    loss_ce
                    + args.w_supcon * loss_supcon
                    + args.gate_penalty * model.gate.square()
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            batch = eeg.size(0)
            total_seen += batch
            total_loss += float(loss.detach()) * batch
            total_ce += float(loss_ce.detach()) * batch
            total_supcon += float(loss_supcon.detach()) * batch

        validation, _, _ = evaluate(model, loaders["val"], device)
        current_score = score(validation)
        row = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": total_loss / max(total_seen, 1),
            "train_ce": total_ce / max(total_seen, 1),
            "train_supcon": total_supcon / max(total_seen, 1),
            "gate_value": float(model.gate.detach().cpu()),
            **{f"val_{key}": validation[key] for key in (
                "balanced_accuracy", "top1", "top3", "top5",
                "normalized_entropy", "dominant_ratio",
            )},
        }
        history.append(row)
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_phase = phase
            best_validation = validation
            save_candidate(
                checkpoint_path, model, args, best_epoch, best_phase, best_validation
            )
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  epoch={epoch:03d} phase={phase:13s} "
                f"loss={row['train_loss']:.4f} gate={row['gate_value']:+.4f} "
                f"val_BAC={validation['balanced_accuracy']:.4f} "
                f"@3={validation['top3']:.4f}",
                flush=True,
            )
        patience_anchor = max(best_epoch, joint_start - 1)
        if (
            args.patience > 0
            and epoch >= joint_start
            and epoch - patience_anchor >= args.patience
        ):
            print(f"[Early stop] epoch={epoch} best={best_epoch}", flush=True)
            break

    checkpoint = safe_torch_load(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return initial_validation, best_epoch, best_phase, best_validation, history


def main() -> None:
    args = parse_args()
    if not 0 < args.residual_only_epochs < args.epochs:
        raise ValueError("residual_only_epochs must be between 1 and epochs-1")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.out_dir)
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite to rerun")
    vi_path, vs_path = Path(args.vi_ckpt), Path(args.vs_ckpt)
    if not vi_path.is_file() or not vs_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: VI={vi_path}, VS={vs_path}")
    counts = session_counts(args.vi_root)
    sid = args.subject_id
    if counts.get(sid, 0) < 2:
        raise RuntimeError(f"S{sid:02d} requires multi-session VI data")
    datasets, loaders, removed = build_loaders(args.vi_root, sid, args, "vi")
    student = RepresentationEncoder(args)
    teacher = RepresentationEncoder(args)
    load_matching(student, vi_path, "VI")
    load_matching(teacher, vs_path, "VS")
    model = GatedResidualTransfer(student, teacher, args.latent_dim).to(device)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "encoder_best.pt"
    print(
        f"[INFO] S{sid:02d} VI-primary gated VS residual device={device} "
        f"train={len(datasets['train'])} val={len(datasets['val'])} "
        f"test={len(datasets['test'])}",
        flush=True,
    )
    initial_validation, best_epoch, best_phase, best_validation, history = train(
        model, loaders, args, device, checkpoint_path
    )
    test_metrics, rows, confusion = evaluate(model, loaders["test"], device)
    diagnostics = feature_diagnostics(model, loaders["test"], device)
    metrics = {
        "subject": sid,
        "stage": "gated_residual_vs_to_vi",
        "representation": "raw_tf",
        "seed": args.seed,
        "n_sessions": counts[sid],
        "n_train": len(datasets["train"]),
        "n_val": len(datasets["val"]),
        "nonfinite_removed": removed,
        "vi_checkpoint": str(vi_path.resolve()),
        "vs_checkpoint": str(vs_path.resolve()),
        "teacher_frozen": True,
        "gate_initialized_zero": True,
        "adapter_initialized_identity": True,
        "epoch0_is_vi_only": True,
        "initial_validation": initial_validation,
        "selection_domain": "VI validation",
        "selection_order": "BAC_then_Top3_then_Top5",
        "selection_round_decimals": SELECTION_DECIMALS,
        "best_epoch": best_epoch,
        "best_phase": best_phase,
        "best_validation": best_validation,
        "test_evaluations": 1,
        **test_metrics,
        **diagnostics,
        "config": vars(args),
    }
    write_outputs(output, metrics, rows, confusion, history)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"[Saved] {output}", flush=True)


if __name__ == "__main__":
    main()
