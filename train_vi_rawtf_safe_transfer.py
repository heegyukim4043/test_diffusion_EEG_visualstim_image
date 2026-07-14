"""Transfer-safe raw+TF VS-to-VI fine-tuning.

The VS classifier is discarded.  Training then follows a fixed schedule:
head only, fusion plus head, and finally the complete backbone with
discriminative learning rates.  Model selection uses VI validation only and
the VI test split is evaluated once after loading the selected checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from dataset_vs_re import session_counts


SELECTION_DECIMALS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", required=True, type=int)
    parser.add_argument("--vi_root", required=True)
    parser.add_argument("--vs_ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling_rate", type=float, default=1024.0)
    parser.add_argument("--n_ch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fusion_lr_ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--backbone_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--head_epochs", type=int, default=10)
    parser.add_argument("--fusion_epochs", type=int, default=20)
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
    args.representation = "raw_tf"
    return args


def phase_for_epoch(epoch: int, args: argparse.Namespace) -> str:
    if epoch <= args.head_epochs:
        return "head"
    if epoch <= args.head_epochs + args.fusion_epochs:
        return "fusion"
    return "full"


def configure_phase(model: RepresentationEncoder, phase: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    if phase in ("fusion", "full"):
        for parameter in model.fusion.parameters():
            parameter.requires_grad_(True)
    if phase == "full":
        for parameter in model.parameters():
            parameter.requires_grad_(True)


def set_training_mode(model: RepresentationEncoder, phase: str) -> None:
    # Keep frozen dropout paths deterministic during the first two phases.
    model.eval()
    model.classifier.train()
    if phase in ("fusion", "full"):
        model.fusion.train()
    if phase == "full":
        model.train()


def build_optimizer(
    model: RepresentationEncoder,
    phase: str,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    groups = [{"params": model.classifier.parameters(), "lr": args.lr, "name": "head"}]
    if phase in ("fusion", "full"):
        groups.append({
            "params": model.fusion.parameters(),
            "lr": args.lr * args.fusion_lr_ratio,
            "name": "fusion",
        })
    if phase == "full":
        backbone = []
        for module in (model.raw, model.tf):
            if module is not None:
                backbone.extend(module.parameters())
        groups.append({
            "params": backbone,
            "lr": args.lr * args.backbone_lr_ratio,
            "name": "backbone",
        })
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def load_vs_backbone(
    model: RepresentationEncoder,
    checkpoint_path: Path,
) -> dict:
    checkpoint = safe_torch_load(checkpoint_path, "cpu")
    config = checkpoint.get("config", {})
    if config.get("representation") != "raw_tf":
        raise RuntimeError(
            f"Expected raw_tf VS checkpoint, got {config.get('representation')!r}"
        )
    state = {
        name: value
        for name, value in checkpoint["model"].items()
        if not name.startswith("classifier.")
    }
    incompatible = model.load_state_dict(state, strict=False)
    expected_missing = {"classifier.weight", "classifier.bias"}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint keys: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    # Explicitly document and enforce a new VI decision head.
    model.classifier.reset_parameters()
    return checkpoint


def rounded_score(metrics: dict) -> tuple[float, float, float]:
    return tuple(
        round(float(metrics[key]), SELECTION_DECIMALS)
        for key in ("balanced_accuracy", "top3", "top5")
    )


def train(
    model: RepresentationEncoder,
    loaders,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[int, str, dict, list[dict]]:
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")
    augmenter = EEGWaveAugment()
    best_score = (-1.0, -1.0, -1.0)
    best_epoch = 0
    best_phase = ""
    best_validation = None
    history = []
    current_phase = ""
    optimizer = None
    full_start = args.head_epochs + args.fusion_epochs + 1

    for epoch in range(1, args.epochs + 1):
        phase = phase_for_epoch(epoch, args)
        if phase != current_phase:
            configure_phase(model, phase)
            optimizer = build_optimizer(model, phase, args)
            current_phase = phase
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[Phase] epoch={epoch} phase={phase} trainable={trainable:,}", flush=True)
        assert optimizer is not None
        set_training_mode(model, phase)
        total_loss = total_ce = total_supcon = 0.0
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
                    logits, labels, label_smoothing=args.label_smoothing
                )
                loss_supcon = supervised_contrastive_loss(latent, labels, args.temperature)
                loss = loss_ce + args.w_supcon * loss_supcon
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
        score = rounded_score(validation)
        row = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": total_loss / max(total_seen, 1),
            "train_ce": total_ce / max(total_seen, 1),
            "train_supcon": total_supcon / max(total_seen, 1),
            "head_lr": args.lr,
            "fusion_lr": args.lr * args.fusion_lr_ratio if phase != "head" else 0.0,
            "backbone_lr": args.lr * args.backbone_lr_ratio if phase == "full" else 0.0,
            **{f"val_{key}": validation[key] for key in (
                "balanced_accuracy", "top1", "top3", "top5",
                "normalized_entropy", "dominant_ratio",
            )},
        }
        history.append(row)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_phase = phase
            best_validation = validation
            torch.save({
                "model": model.state_dict(),
                "config": vars(args),
                "transfer_protocol": "reinit_head_gradual_unfreeze",
                "best_epoch": best_epoch,
                "best_phase": best_phase,
                "best_validation": best_validation,
            }, checkpoint_path)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  epoch={epoch:03d} phase={phase:6s} loss={row['train_loss']:.4f} "
                f"val_BAC={validation['balanced_accuracy']:.4f} "
                f"@3={validation['top3']:.4f} dom={validation['dominant_ratio']:.3f}",
                flush=True,
            )
        patience_anchor = max(best_epoch, full_start - 1)
        if (
            args.patience > 0
            and epoch >= full_start
            and epoch - patience_anchor >= args.patience
        ):
            print(f"[Early stop] epoch={epoch} best={best_epoch}", flush=True)
            break

    if best_validation is None:
        raise RuntimeError("No validation checkpoint was produced")
    checkpoint = safe_torch_load(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return best_epoch, best_phase, best_validation, history


def main() -> None:
    args = parse_args()
    if args.head_epochs < 1 or args.fusion_epochs < 1:
        raise ValueError("head_epochs and fusion_epochs must be positive")
    if args.head_epochs + args.fusion_epochs >= args.epochs:
        raise ValueError("Schedule must leave at least one full-unfreeze epoch")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.out_dir)
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite to rerun")
    checkpoint_path = Path(args.vs_ckpt)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    counts = session_counts(args.vi_root)
    sid = args.subject_id
    if counts.get(sid, 0) < 2:
        raise RuntimeError(f"S{sid:02d} requires multi-session VI data")
    datasets, loaders, removed = build_loaders(args.vi_root, sid, args, "vi")
    model = RepresentationEncoder(args).to(device)
    load_vs_backbone(model, checkpoint_path)
    output.mkdir(parents=True, exist_ok=True)
    trained_checkpoint = output / "encoder_best.pt"
    print(
        f"[INFO] S{sid:02d} raw_tf safe VS->VI device={device} "
        f"train={len(datasets['train'])} val={len(datasets['val'])} "
        f"test={len(datasets['test'])}",
        flush=True,
    )
    best_epoch, best_phase, best_validation, history = train(
        model, loaders, args, device, trained_checkpoint
    )
    test_metrics, rows, confusion = evaluate(model, loaders["test"], device)
    metrics = {
        "subject": sid,
        "stage": "safe_vs_to_vi",
        "representation": "raw_tf",
        "domain": "vi",
        "seed": args.seed,
        "n_sessions": counts[sid],
        "n_train": len(datasets["train"]),
        "n_val": len(datasets["val"]),
        "n_test": len(datasets["test"]),
        "nonfinite_removed": removed,
        "initialized_from_vs": True,
        "initial_checkpoint": str(checkpoint_path.resolve()),
        "classifier_reinitialized": True,
        "transfer_protocol": "reinit_head_gradual_unfreeze",
        "phase_schedule": {
            "head_only": [1, args.head_epochs],
            "fusion_and_head": [args.head_epochs + 1, args.head_epochs + args.fusion_epochs],
            "full": [args.head_epochs + args.fusion_epochs + 1, args.epochs],
        },
        "selection_domain": "VI validation",
        "selection_order": "BAC_then_Top3_then_Top5",
        "selection_round_decimals": SELECTION_DECIMALS,
        "best_epoch": best_epoch,
        "best_phase": best_phase,
        "best_validation": best_validation,
        "test_evaluations": 1,
        **test_metrics,
        "config": vars(args),
    }
    write_outputs(output, metrics, rows, confusion, history)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"[Saved] {output}", flush=True)


if __name__ == "__main__":
    main()
