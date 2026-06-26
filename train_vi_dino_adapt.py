"""
Exp28-B: VI adaptation after failed VS->VI zero-shot transfer.

Runs three comparable VI adaptation modes on the same subject/split:
  - scratch: random-init EEG-DINO model trained on VI
  - finetune: initialize from VS SupCon checkpoint, fine-tune all weights on VI
  - linear: freeze VS encoder, train a linear classifier on VI latents

Default target subjects are S01/S02/S18 because they have VS checkpoints and
prior zero-shot VI results.

Usage:
  python train_vi_dino_adapt.py --mode all --subject_ids 1,2,18
  python train_vi_dino_adapt.py --mode finetune --epochs 100 --patience 20
  python train_vi_dino_adapt.py --mode linear --linear_epochs 20
"""

import argparse
import csv
import datetime
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_vs_re_exp25_vi_transfer import VIDataset, collate_vi, load_vs_model
from model_eeg_dino import EEGDINORegressor, load_dino_encoder, DINO_DIM, LATENT_DIM
from train_crosssubj_dino import set_seed, EEGAugment, compute_class_prototypes


N_CLASSES = 9
CLS_LIST = list(range(1, 10))


def parse_subject_ids(raw):
    ids = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(tok))
    return ids


def topk_from_logits(logits, labels, k_list=(1, 3, 5)):
    out = {}
    for k in k_list:
        topk = logits.topk(min(k, logits.size(1)), dim=1).indices
        out[k] = topk.eq(labels.unsqueeze(1)).any(1).sum().item()
    return out


@torch.no_grad()
def evaluate_model(model, loader, proto_dino, device):
    correct = {1: 0, 3: 0, 5: 0}
    total = 0
    subj = torch.zeros(1, dtype=torch.long, device=device)
    model.eval()
    for eeg, lbl in loader:
        eeg = eeg.to(device)
        lbl = lbl.to(device)
        logits, _, _ = model.predict(eeg, subj.expand(eeg.size(0)), proto_dino)
        got = topk_from_logits(logits, lbl)
        for k in correct:
            correct[k] += got[k]
        total += eeg.size(0)
    return {k: correct[k] / max(total, 1) for k in correct}


@torch.no_grad()
def evaluate_linear(model, head, loader, device):
    correct = {1: 0, 3: 0, 5: 0}
    total = 0
    subj = torch.zeros(1, dtype=torch.long, device=device)
    model.eval()
    head.eval()
    for eeg, lbl in loader:
        eeg = eeg.to(device)
        lbl = lbl.to(device)
        lat = model.encode_eeg(eeg, subj.expand(eeg.size(0)))
        logits = head(lat)
        got = topk_from_logits(logits, lbl)
        for k in correct:
            correct[k] += got[k]
        total += eeg.size(0)
    return {k: correct[k] / max(total, 1) for k in correct}


def build_scratch_model(args, dino_feat_dim, device):
    raw_occ = args.eeg_occipital_ids.strip().lower()
    if raw_occ == "auto":
        occ_idx = None
    elif raw_occ == "none":
        occ_idx = []
    else:
        occ_idx = [int(x) for x in raw_occ.split(",") if x.strip()]

    return EEGDINORegressor(
        eeg_channels=args.n_ch,
        n_subjects=1,
        dino_feat_dim=dino_feat_dim,
        latent_dim=LATENT_DIM,
        eeg_hidden=args.eeg_hidden,
        eeg_out=args.eeg_out,
        subj_emb_dim=args.subj_emb_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        temperature=args.temperature,
        encoder_type=args.encoder_type,
        n_classes=N_CLASSES,
        eeg_occipital_indices=occ_idx,
    ).to(device)


def make_loaders(args, sid):
    train_ds = VIDataset(args.vi_root, sid, args.n_ch, "train", args.seed)
    val_ds = VIDataset(args.vi_root, sid, args.n_ch, "val", args.seed)
    test_ds = VIDataset(args.vi_root, sid, args.n_ch, "test", args.seed)
    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_vi),
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_vi),
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_vi),
        {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds), "time": train_ds.n_time},
    )


def train_dino_model(mode, sid, model, loaders, proto_dino, args, device, save_dir):
    train_loader, val_loader, test_loader, sizes = loaders
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    augmenter = EEGAugment(
        noise_std=0.03, scale_range=(0.9, 1.1),
        ch_drop_prob=0.05, max_shift=10, freq_noise_std=0.0,
        p_noise=0.5, p_scale=0.5, p_drop=0.2, p_shift=0.2, p_freq=0.0,
    )

    best_val = -1.0
    best_epoch = 0
    best_path = os.path.join(save_dir, f"subj{sid:02d}_{mode}_best.pt")
    subj = torch.zeros(1, dtype=torch.long, device=device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for eeg, lbl in train_loader:
            eeg = augmenter(eeg.to(device))
            lbl = lbl.to(device)
            subj_b = subj.expand(eeg.size(0))
            tgt = proto_dino[lbl]

            eeg_lat = model.encode_eeg(eeg, subj_b)
            img_lat = model.encode_img(tgt)
            temp = model.log_temp.exp().clamp(0.01, 1.0).item()
            loss_sc = EEGDINORegressor.supcon_loss(eeg_lat, img_lat, lbl, temperature=temp)
            proto_lat = model.encode_img(proto_dino)
            loss_proto = F.cross_entropy(eeg_lat @ proto_lat.T / temp, lbl)
            loss_aux = F.cross_entropy(model.aux_cls_head(eeg_lat), lbl)
            loss = args.w_supcon * loss_sc + args.w_proto * loss_proto + args.w_aux * loss_aux

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val = evaluate_model(model, val_loader, proto_dino, device)
        if val[1] >= best_val:
            best_val = val[1]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "sid": sid, "mode": mode, "config": vars(args)}, best_path)

        if epoch - best_epoch >= args.patience:
            break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test = evaluate_model(model, test_loader, proto_dino, device)
    return {
        "subject": f"S{sid:02d}",
        "mode": mode,
        "best_epoch": best_epoch,
        "val_top1": best_val,
        "top1": test[1],
        "top3": test[3],
        "top5": test[5],
        **sizes,
    }


def train_linear_probe(sid, model, loaders, args, device, save_dir):
    train_loader, val_loader, test_loader, sizes = loaders
    for p in model.parameters():
        p.requires_grad = False
    head = nn.Linear(LATENT_DIM, N_CLASSES).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.linear_lr, weight_decay=args.wd)
    subj = torch.zeros(1, dtype=torch.long, device=device)
    best_val = -1.0
    best_epoch = 0
    best_path = os.path.join(save_dir, f"subj{sid:02d}_linear_best.pt")

    for epoch in range(1, args.linear_epochs + 1):
        model.eval()
        head.train()
        for eeg, lbl in train_loader:
            eeg = eeg.to(device)
            lbl = lbl.to(device)
            with torch.no_grad():
                lat = model.encode_eeg(eeg, subj.expand(eeg.size(0)))
            loss = F.cross_entropy(head(lat), lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val = evaluate_linear(model, head, val_loader, device)
        if val[1] >= best_val:
            best_val = val[1]
            best_epoch = epoch
            torch.save({"head": head.state_dict(), "sid": sid, "mode": "linear", "config": vars(args)}, best_path)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    head.load_state_dict(ckpt["head"])
    test = evaluate_linear(model, head, test_loader, device)
    return {
        "subject": f"S{sid:02d}",
        "mode": "linear",
        "best_epoch": best_epoch,
        "val_top1": best_val,
        "top1": test[1],
        "top3": test[3],
        "top5": test[5],
        **sizes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scratch", "finetune", "linear", "all"], default="all")
    parser.add_argument("--ckpt_dir", default="./checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon")
    parser.add_argument("--vi_root", default="./preproc_data_vi")
    parser.add_argument("--img_root", default="./preproc_data_vi/images")
    parser.add_argument("--subject_ids", default="1,2,18")
    parser.add_argument("--n_ch", type=int, default=32)
    parser.add_argument("--dino_model", default="dinov2_vits14")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--linear_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--linear_lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--encoder_type", default="v2", choices=["transformer", "conv", "v2"])
    parser.add_argument("--eeg_occipital_ids", default="auto")
    parser.add_argument("--eeg_hidden", type=int, default=256)
    parser.add_argument("--eeg_out", type=int, default=256)
    parser.add_argument("--subj_emb_dim", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--w_supcon", type=float, default=1.0)
    parser.add_argument("--w_proto", type=float, default=1.0)
    parser.add_argument("--w_aux", type=float, default=0.5)
    parser.add_argument("--out_root", default="./checkpoints_vi_adapt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject_ids = parse_subject_ids(args.subject_ids)
    modes = ["scratch", "finetune", "linear"] if args.mode == "all" else [args.mode]

    dino = load_dino_encoder(args.dino_model, device)
    dino_feat_dim = DINO_DIM[args.dino_model]
    proto_dino = compute_class_prototypes(dino, args.img_root, CLS_LIST, device)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.out_root, f"{ts}_exp28b_{args.mode}")
    os.makedirs(save_dir, exist_ok=True)
    results = []

    for sid in subject_ids:
        loaders = make_loaders(args, sid)
        print(f"\nS{sid:02d}: VI train={loaders[3]['train']} val={loaders[3]['val']} test={loaders[3]['test']}")

        if "scratch" in modes:
            model = build_scratch_model(args, dino_feat_dim, device)
            results.append(train_dino_model("scratch", sid, model, loaders, proto_dino, args, device, save_dir))

        if "finetune" in modes or "linear" in modes:
            ckpt_path = os.path.join(args.ckpt_dir, f"subj{sid:02d}_best.pt")
            if not os.path.isfile(ckpt_path):
                print(f"[SKIP] S{sid:02d}: VS checkpoint not found for finetune/linear")
                continue
            if "finetune" in modes:
                model_ft, _ = load_vs_model(ckpt_path, dino_feat_dim, args.n_ch, device)
                results.append(train_dino_model("finetune", sid, model_ft, loaders, proto_dino, args, device, save_dir))
            if "linear" in modes:
                model_lin, _ = load_vs_model(ckpt_path, dino_feat_dim, args.n_ch, device)
                results.append(train_linear_probe(sid, model_lin, loaders, args, device, save_dir))

    out_csv = os.path.join(save_dir, "results_vi_adapt.csv")
    fields = ["subject", "mode", "best_epoch", "val_top1", "top1", "top3", "top5", "train", "val", "test", "time"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("\nSummary")
    for row in results:
        print(
            f"{row['subject']} {row['mode']:>8} "
            f"T1={row['top1']:.4f} T3={row['top3']:.4f} T5={row['top5']:.4f} "
            f"best_ep={row['best_epoch']}"
        )
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
