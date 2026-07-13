"""Direct VI EEG classification transfer experiment on current NPZ data.

The three prespecified conditions are:

* ``zero_shot``: evaluate the frozen VS-trained encoder on VI test trials.
* ``vi_only``: train the same encoder architecture from random initialization on VI.
* ``vs_to_vi``: initialize from the VS checkpoint and fine-tune all weights on VI.

This script deliberately does not involve Stable Diffusion or LoRA.  It measures
EEG-to-class retrieval against the same nine DINO class prototypes used by the
VS SupCon encoder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_vs_re import VSReDataset, collate_fn, session_counts
from model_eeg_dino import EEGDINORegressor, DINO_DIM, LATENT_DIM, load_dino_encoder
from train_crosssubj_dino import EEGAugment, compute_class_prototypes, set_seed


N_CLASSES = 9
CLASS_IDS = list(range(1, 10))
MODES = ("zero_shot", "vi_only", "vs_to_vi")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--subject_id", required=True, type=int)
    p.add_argument("--vi_root", required=True)
    p.add_argument("--img_root", required=True)
    p.add_argument("--vs_ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_ch", type=int, default=32)
    p.add_argument("--dino_model", default="dinov2_vits14")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--loss_type", choices=("supcon", "supcon_proto"), default="supcon")
    p.add_argument("--w_supcon", type=float, default=1.0)
    p.add_argument("--w_proto", type=float, default=1.0)
    p.add_argument("--w_aux", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=0,
                   help="0 disables early stopping; checkpoint selection still uses VI validation")
    p.add_argument("--no_aug", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def checkpoint_config(checkpoint: dict) -> dict:
    cfg = checkpoint.get("config", {})
    if isinstance(cfg, argparse.Namespace):
        return vars(cfg)
    return dict(cfg)


def parse_occipital(raw: object) -> list[int] | None:
    text = str(raw if raw is not None else "auto").strip().lower()
    if text == "auto":
        return None
    if text == "none":
        return []
    return [int(x) for x in text.split(",") if x.strip()]


def build_model(cfg: dict, dino_feat_dim: int, n_ch: int, device: torch.device) -> EEGDINORegressor:
    """Build the exact architecture recorded in the VS checkpoint."""
    return EEGDINORegressor(
        eeg_channels=int(cfg.get("n_ch", n_ch)),
        n_subjects=1,
        dino_feat_dim=dino_feat_dim,
        latent_dim=LATENT_DIM,
        eeg_hidden=int(cfg.get("eeg_hidden", 256)),
        eeg_out=int(cfg.get("eeg_out", 256)),
        subj_emb_dim=int(cfg.get("subj_emb_dim", 32)),
        n_heads=int(cfg.get("n_heads", 4)),
        n_layers=int(cfg.get("n_layers", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        temperature=float(cfg.get("temperature", 0.1)),
        encoder_type=str(cfg.get("encoder_type", "v2")),
        n_classes=N_CLASSES,
        eeg_occipital_indices=parse_occipital(cfg.get("eeg_occipital_ids", "auto")),
    ).to(device)


def make_loaders(args: argparse.Namespace):
    sid = args.subject_id
    subj_map = {sid: 0}
    datasets = {
        split: VSReDataset(
            args.vi_root,
            [sid],
            subj_map,
            args.n_ch,
            split,
            args.seed,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            collate_fn=collate_fn,
        )
        for split, ds in datasets.items()
    }
    return datasets, loaders


@torch.no_grad()
def evaluate(model, loader, proto_dino, device):
    model.eval()
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    rows = []
    correct = {1: 0, 3: 0, 5: 0}
    sample_index = 0

    for eeg, subj, lbl in loader:
        eeg = eeg.to(device)
        subj = subj.to(device)
        lbl_dev = lbl.to(device)
        logits, _, _ = model.predict(eeg, subj, proto_dino)
        pred = logits.argmax(dim=1)

        for k in correct:
            topk = logits.topk(min(k, N_CLASSES), dim=1).indices
            correct[k] += topk.eq(lbl_dev.unsqueeze(1)).any(1).sum().item()

        logits_cpu = logits.detach().float().cpu()
        pred_cpu = pred.cpu()
        for i in range(len(lbl)):
            truth = int(lbl[i])
            guess = int(pred_cpu[i])
            cm[truth, guess] += 1
            true_score = float(logits_cpu[i, truth])
            best_other = float(torch.cat((logits_cpu[i, :truth], logits_cpu[i, truth + 1:])).max())
            rows.append({
                "sample_index": sample_index,
                "true_label": truth,
                "pred_label": guess,
                "correct": int(truth == guess),
                "true_score": true_score,
                "true_margin": true_score - best_other,
            })
            sample_index += 1

    total = len(rows)
    recalls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    metrics = {
        "n_test": total,
        "top1": correct[1] / max(total, 1),
        "top3": correct[3] / max(total, 1),
        "top5": correct[5] / max(total, 1),
        "balanced_accuracy": float(recalls.mean()),
        "mean_true_score": float(np.mean([r["true_score"] for r in rows])) if rows else 0.0,
        "mean_true_margin": float(np.mean([r["true_margin"] for r in rows])) if rows else 0.0,
    }
    return metrics, rows, cm


def training_loss(model, eeg, subj, lbl, proto_dino, args):
    target = proto_dino[lbl]
    eeg_lat = model.encode_eeg(eeg, subj)
    img_lat = model.encode_img(target)
    temperature = model.log_temp.exp().clamp(0.01, 1.0).item()
    loss_supcon = EEGDINORegressor.supcon_loss(
        eeg_lat, img_lat, lbl, temperature=temperature
    )
    loss_aux = F.cross_entropy(model.aux_cls_head(eeg_lat), lbl)
    if args.loss_type == "supcon_proto":
        proto_lat = model.encode_img(proto_dino)
        loss_proto = F.cross_entropy(eeg_lat @ proto_lat.T / temperature, lbl)
    else:
        loss_proto = torch.zeros((), device=eeg.device)
    return (
        args.w_supcon * loss_supcon
        + args.w_proto * loss_proto
        + args.w_aux * loss_aux
    )


def train(model, loaders, proto_dino, args, device, checkpoint_path: Path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warmup = max(1, args.epochs // 10)

    def lr_lambda(epoch):
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / max(1, args.epochs - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    augmenter = EEGAugment(
        noise_std=0.05,
        scale_range=(0.8, 1.2),
        ch_drop_prob=0.1,
        max_shift=25,
        freq_noise_std=0.02,
        p_noise=0.5,
        p_scale=0.5,
        p_drop=0.0 if args.no_aug else 0.3,
        p_shift=0.0 if args.no_aug else 0.3,
        p_freq=0.0,
    )

    best_score = (-1.0, -1.0, -1.0)
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        for eeg, subj, lbl in loaders["train"]:
            eeg = augmenter(eeg.to(device))
            subj = subj.to(device)
            lbl = lbl.to(device)
            loss = training_loss(model, eeg, subj, lbl, proto_dino, args)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * eeg.size(0)
            n_seen += eeg.size(0)
        scheduler.step()

        val, _, _ = evaluate(model, loaders["val"], proto_dino, device)
        score = (val["top1"], val["top3"], val["top5"])
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / max(n_seen, 1),
            "val_top1": val["top1"],
            "val_top3": val["top3"],
            "val_top5": val["top5"],
        })
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "subject_id": args.subject_id,
                "mode": args.mode,
                "config": vars(args),
                "best_epoch": best_epoch,
                "best_val": dict(zip(("top1", "top3", "top5"), score)),
            }, checkpoint_path)

        if epoch == 1 or epoch % max(1, args.epochs // 5) == 0:
            print(
                f"  ep={epoch:03d} loss={history[-1]['train_loss']:.4f} "
                f"val@1={score[0]:.4f} @3={score[1]:.4f} @5={score[2]:.4f}",
                flush=True,
            )
        if args.patience > 0 and epoch - best_epoch >= args.patience:
            print(f"  early stop ep={epoch} best_ep={best_epoch}", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return best_epoch, best_score, history


def write_outputs(out_dir: Path, metrics: dict, rows: list[dict], cm: np.ndarray, history: list[dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["sample_index"])
        writer.writeheader()
        writer.writerows(rows)
    np.savetxt(out_dir / "confusion.csv", cm, fmt="%d", delimiter=",")
    if history:
        with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; use --overwrite to replace")

    vs_ckpt_path = Path(args.vs_ckpt)
    if not vs_ckpt_path.is_file():
        raise FileNotFoundError(vs_ckpt_path)
    vi_sessions = session_counts(args.vi_root)
    if args.subject_id not in vi_sessions:
        raise FileNotFoundError(f"S{args.subject_id:02d} VI data missing in {args.vi_root}")

    print(
        f"[INFO] mode={args.mode} subject=S{args.subject_id:02d} "
        f"VI_sessions={vi_sessions[args.subject_id]} device={device}",
        flush=True,
    )
    checkpoint = torch.load(vs_ckpt_path, map_location="cpu", weights_only=False)
    vs_cfg = checkpoint_config(checkpoint)

    print(f"[INFO] Loading DINO {args.dino_model}", flush=True)
    dino = load_dino_encoder(args.dino_model, device)
    proto_dino = compute_class_prototypes(dino, args.img_root, CLASS_IDS, device)
    dino_feat_dim = DINO_DIM[args.dino_model]
    datasets, loaders = make_loaders(args)
    print(
        f"[INFO] VI split train={len(datasets['train'])} val={len(datasets['val'])} "
        f"test={len(datasets['test'])}",
        flush=True,
    )

    # All conditions use the architecture recorded in the same VS checkpoint.
    model = build_model(vs_cfg, dino_feat_dim, args.n_ch, device)
    initialized_from_vs = args.mode in ("zero_shot", "vs_to_vi")
    if initialized_from_vs:
        model.load_state_dict(checkpoint["model"], strict=True)
        print(f"[INFO] Loaded VS encoder: {vs_ckpt_path}", flush=True)
    else:
        print("[INFO] Random encoder initialization (VI-only)", flush=True)

    history = []
    best_epoch = 0
    best_val = None
    trained_checkpoint = out_dir / "encoder_best.pt"
    if args.mode != "zero_shot":
        out_dir.mkdir(parents=True, exist_ok=True)
        best_epoch, score, history = train(
            model, loaders, proto_dino, args, device, trained_checkpoint
        )
        best_val = {"top1": score[0], "top3": score[1], "top5": score[2]}

    test_metrics, rows, cm = evaluate(model, loaders["test"], proto_dino, device)
    metrics = {
        "subject": args.subject_id,
        "mode": args.mode,
        "seed": args.seed,
        "vi_sessions": vi_sessions[args.subject_id],
        "n_train": len(datasets["train"]),
        "n_val": len(datasets["val"]),
        **test_metrics,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "vs_checkpoint": str(vs_ckpt_path.resolve()),
        "initialized_from_vs": initialized_from_vs,
        "encoder_frozen": args.mode == "zero_shot",
        "architecture_config": {
            key: vs_cfg.get(key)
            for key in (
                "n_ch", "encoder_type", "eeg_hidden", "eeg_out", "subj_emb_dim",
                "n_heads", "n_layers", "dropout", "temperature", "eeg_occipital_ids",
            )
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "wd": args.wd,
            "loss_type": args.loss_type,
            "w_supcon": args.w_supcon,
            "w_proto": args.w_proto,
            "w_aux": args.w_aux,
            "patience": args.patience,
        },
    }
    write_outputs(out_dir, metrics, rows, cm, history)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"[Saved] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
