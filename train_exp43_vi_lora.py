#!/usr/bin/env python3
"""
train_exp43_vi_lora.py
────────────────────────────────────────────────────────────────────────────
Exp43: VI LoRA scratch/transfer fine-tuning for EEG-conditioned SD1.5 generation.

Conditions:
  C0 = VI scratch LoRA with frozen subject SupCon EEG encoder
  C1 = VS LoRA -> VI fine-tune with frozen subject SupCon EEG encoder

Results are written only to the selected checkpoint root, which should normally be
a Google Drive path. This script does not update PROGRESS.md and does not push to
GitHub.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_vs_re import collate_fn, load_subject_vsre
from model_eeg_dino import DINO_DIM, LATENT_DIM
from train_crosssubj_dino import EEGAugment, compute_class_prototypes, set_seed
from train_vs_re_latent_gen import build_eeg_encoder, make_schedule
from train_vs_re_lora_gen import (
    CLS_LIST,
    EEGConditionProjector,
    ddpm_q_sample,
    encode_class_images_512,
    encode_class_images_512_aug,
    evaluate_lora,
    load_sd15_unet_lora,
)


@dataclass
class InitCandidate:
    path: str
    dirname: str
    top1: Optional[float]
    mtime: float


class SubjectClassLimitedDataset(Dataset):
    """Single-subject VI/VS dataset with optional per-class total cap before split.

    `per_class_total=60` gives 48 train / 6 val / 6 test per class under the
    same 80/10/10 split used in VSReDataset, i.e. 432 train trials over 9 classes.
    Set `per_class_total=0` or None to use all available trials.
    """

    def __init__(
        self,
        data_root: str,
        sid: int,
        n_ch: int = 32,
        split: str = "train",
        seed: int = 42,
        per_class_total: Optional[int] = 60,
        baseline_correct: bool = False,
        ch_zscore: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train|val|test, got {split}")
        if per_class_total is not None and per_class_total <= 0:
            per_class_total = None

        eeg, labels, eff_sess = load_subject_vsre(
            data_root,
            sid,
            n_ch=n_ch,
            baseline_correct=baseline_correct,
            ch_zscore=ch_zscore,
        )

        self.samples = []
        self.sid = sid
        self.split = split
        self.eff_sessions = eff_sess
        self.per_class_total = per_class_total
        self.class_counts: dict[int, int] = {}

        for cls in range(9):
            idx = np.where(labels == cls)[0]
            if len(idx) == 0:
                continue
            rng = np.random.RandomState(seed + sid * 100 + cls)
            rng.shuffle(idx)
            if per_class_total is not None:
                idx = idx[: min(per_class_total, len(idx))]
            n = len(idx)
            n_train = int(n * 0.8)
            n_val = int(n * 0.1)
            if split == "train":
                sel = idx[:n_train]
            elif split == "val":
                sel = idx[n_train : n_train + n_val]
            else:
                sel = idx[n_train + n_val :]
            self.class_counts[cls] = len(sel)
            for i in sel:
                self.samples.append((torch.from_numpy(eeg[i]), 0, int(labels[i])))

        print(
            f"  [dataset:{split}] S{sid:02d} samples={len(self.samples)} "
            f"eff_sessions={eff_sess} per_class_total={per_class_total or 'all'} "
            f"class_counts={self.class_counts}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        return self.samples[i]


def parse_subject_ids(text: str) -> list[int]:
    out: list[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def parse_conditions(text: str) -> list[str]:
    conds = []
    for c in text.split(","):
        c = c.strip().lower()
        if not c:
            continue
        if c not in {"c0", "c1"}:
            raise ValueError(f"Unknown condition '{c}'. Use c0,c1.")
        conds.append(c)
    return conds


def read_top1_from_csv(csv_path: str, sid: int) -> Optional[float]:
    if not os.path.isfile(csv_path):
        return None
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if int(row.get("sid", -1)) == sid:
                    return float(row["top1"])
    except Exception:
        return None
    return None


def find_init_lora_ckpt(
    roots: Iterable[str],
    sid: int,
    lora_r: int,
    explicit: Optional[str] = None,
) -> str:
    """Resolve a VS LoRA checkpoint path for C1.

    If explicit is provided, it may be either a checkpoint file or a directory
    containing subjXX_lora_best.pt. Otherwise roots are searched recursively.
    Candidates with a results_lora_gen.csv score are preferred by highest top1;
    unscored candidates fall back to newest mtime.
    """
    if explicit and explicit.lower() != "auto":
        p = Path(explicit)
        if p.is_dir():
            p = p / f"subj{sid:02d}_lora_best.pt"
        if not p.is_file():
            raise FileNotFoundError(f"Explicit init checkpoint not found: {p}")
        return str(p)

    candidates: list[InitCandidate] = []
    for root in roots:
        if not root:
            continue
        root_path = Path(root)
        if not root_path.exists():
            continue
        pattern = str(root_path / "**" / f"subj{sid:02d}_lora_best.pt")
        for ckpt in glob.glob(pattern, recursive=True):
            dirname = os.path.basename(os.path.dirname(ckpt))
            if f"lora_r{lora_r}" not in dirname:
                continue
            csv_path = os.path.join(os.path.dirname(ckpt), "results_lora_gen.csv")
            top1 = read_top1_from_csv(csv_path, sid)
            candidates.append(
                InitCandidate(
                    path=ckpt,
                    dirname=dirname,
                    top1=top1,
                    mtime=os.path.getmtime(ckpt),
                )
            )

    if not candidates:
        roots_str = ", ".join(str(r) for r in roots)
        raise FileNotFoundError(
            f"No VS LoRA init checkpoint found for S{sid:02d}, r={lora_r}. "
            f"Searched roots: {roots_str}"
        )

    scored = [c for c in candidates if c.top1 is not None]
    if scored:
        scored.sort(key=lambda c: (c.top1 if c.top1 is not None else -1.0, c.mtime), reverse=True)
        best = scored[0]
    else:
        candidates.sort(key=lambda c: c.mtime, reverse=True)
        best = candidates[0]

    print(
        f"  [Init] Selected VS checkpoint: {best.path} "
        f"(dirname={best.dirname}, top1={best.top1})",
        flush=True,
    )
    return best.path


def load_supcon_encoder(args, sid: int, dino_feat_dim: int, device: torch.device):
    eeg_enc = build_eeg_encoder(
        args.n_ch,
        dino_feat_dim,
        type("a", (), {"eeg_occipital_ids": "auto"})(),
        device,
    )
    ckpt_path = os.path.join(args.supcon_ckpt, f"subj{sid:02d}_best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"SupCon checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "eeg_enc" in ckpt:
        eeg_enc.load_state_dict(ckpt["eeg_enc"])
    elif "model" in ckpt:
        eeg_enc.load_state_dict(ckpt["model"])
    else:
        raise KeyError(f"SupCon checkpoint has neither 'eeg_enc' nor 'model': {ckpt_path}")
    eeg_enc.eval()
    for p in eeg_enc.parameters():
        p.requires_grad_(False)
    print(f"  [Encoder] Loaded and frozen: {ckpt_path}", flush=True)
    return eeg_enc, ckpt_path


def copy_lora_state(unet, cond_proj, init_ckpt_path: str, device: torch.device) -> dict:
    ckpt = torch.load(init_ckpt_path, map_location=device, weights_only=False)
    if "cond_proj" not in ckpt or "unet_lora" not in ckpt:
        raise KeyError(f"Not a LoRA generator checkpoint: {init_ckpt_path}")

    cond_proj.load_state_dict(ckpt["cond_proj"])
    loaded = 0
    missing = 0
    src = ckpt["unet_lora"]
    for name, param in unet.named_parameters():
        if name in src:
            param.data.copy_(src[name].to(device))
            loaded += 1
    for name in src:
        if not any(name == n for n, _ in unet.named_parameters()):
            missing += 1
    print(f"  [Init] Loaded LoRA tensors={loaded}, unmatched_source_tensors={missing}", flush=True)
    return ckpt


def save_best_checkpoint(
    path: str,
    unet,
    cond_proj,
    sid: int,
    condition: str,
    best_ep: int,
    best_loss: float,
    args,
    supcon_path: str,
    init_ckpt_path: Optional[str],
) -> None:
    torch.save(
        {
            "unet_lora": {
                k: v.detach().cpu()
                for k, v in unet.named_parameters()
                if v.requires_grad
            },
            "cond_proj": {k: v.detach().cpu() for k, v in cond_proj.state_dict().items()},
            "sid": sid,
            "best_ep": best_ep,
            "best_loss": best_loss,
            "provenance": {
                "experiment": "Exp43_VI_LoRA",
                "condition": condition,
                "data_root": args.data_root,
                "img_root": args.img_root,
                "supcon_ckpt_dir": args.supcon_ckpt,
                "supcon_ckpt_path": supcon_path,
                "init_lora_ckpt": init_ckpt_path,
                "encoder_source": "supcon_frozen",
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "n_eeg_tokens": args.n_eeg_tokens,
                "per_class_total": args.per_class_total,
                "epochs": args.epochs,
                "seed": args.seed,
            },
        },
        path,
    )


def train_condition(
    sid: int,
    condition: str,
    args,
    device: torch.device,
    vae,
    dino,
    proto_dino,
    dino_feat_dim: int,
    acp,
    train_loader,
    test_loader,
    cls_latents,
    cls_latents_aug,
    save_dir: str,
) -> dict:
    print(f"\n{'=' * 70}\n  Exp43 {condition.upper()}  S{sid:02d}", flush=True)

    eeg_enc, supcon_path = load_supcon_encoder(args, sid, dino_feat_dim, device)

    unet = load_sd15_unet_lora(args.lora_r, args.lora_alpha).to(device)
    if args.grad_ckpt:
        unet.enable_gradient_checkpointing()
        print("  [UNet] gradient checkpointing enabled", flush=True)

    cond_proj = EEGConditionProjector(
        eeg_dim=LATENT_DIM,
        sd_dim=768,
        n_tokens=args.n_eeg_tokens,
        deep=args.deep_proj,
    ).to(device)

    init_ckpt_path: Optional[str] = None
    if condition == "c1":
        init_ckpt_path = find_init_lora_ckpt(
            args.vs_ckpt_roots,
            sid=sid,
            lora_r=args.lora_r,
            explicit=args.init_lora_ckpt,
        )
        init_meta = copy_lora_state(unet, cond_proj, init_ckpt_path, device)
        prov = init_meta.get("provenance", {})
        if prov:
            print(f"  [Init] source provenance={prov}", flush=True)
    else:
        print("  [Init] C0 scratch VI LoRA (no VS LoRA initialization)", flush=True)

    train_params = [p for p in unet.parameters() if p.requires_grad] + list(cond_proj.parameters())
    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.wd)

    augmenter = EEGAugment(
        noise_std=0.03,
        scale_range=(0.9, 1.1),
        ch_drop_prob=0.05,
        max_shift=10,
        freq_noise_std=0.0,
        p_noise=0.5,
        p_scale=0.5,
        p_drop=0.2,
        p_shift=0.2,
        p_freq=0.0,
    )

    use_fp16 = bool(args.fp16)
    scaler = GradScaler() if use_fp16 else None
    subj_t = torch.zeros(1, dtype=torch.long, device=device)

    best_loss = float("inf")
    best_ep = 0
    best_path = os.path.join(save_dir, f"subj{sid:02d}_exp43_{condition}_lora_best.pt")

    print("    Ep    TrLoss", flush=True)
    for epoch in range(1, args.epochs + 1):
        unet.train()
        cond_proj.train()
        tr_loss = 0.0

        for eeg, _, lbl in train_loader:
            eeg = augmenter(eeg.to(device))
            lbl = lbl.to(device)
            with torch.no_grad():
                eeg_lat = eeg_enc.encode_eeg(eeg, subj_t.expand(eeg.size(0)))
            cond_tokens = cond_proj(eeg_lat)

            if args.augment_targets and cls_latents_aug is not None:
                x0_list = []
                for li in lbl:
                    views = cls_latents_aug[li.item()]
                    aug_idx = torch.randint(views.size(0), (1,)).item()
                    x0_list.append(views[aug_idx])
                x0 = torch.stack(x0_list)
            else:
                x0 = cls_latents[lbl]

            t = torch.randint(0, args.num_timesteps, (eeg.size(0),), device=device)
            xt, noise = ddpm_q_sample(x0, t, acp)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_fp16):
                noise_pred = unet(xt, t, encoder_hidden_states=cond_tokens).sample
                loss = F.mse_loss(noise_pred, noise)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(train_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(train_params, 1.0)
                optimizer.step()

            tr_loss += loss.item()

        tr_loss /= max(len(train_loader), 1)
        print(f"  {epoch:5d}  {tr_loss:.5f}", flush=True)

        if tr_loss < best_loss:
            best_loss = tr_loss
            best_ep = epoch
            save_best_checkpoint(
                best_path,
                unet,
                cond_proj,
                sid,
                condition,
                best_ep,
                best_loss,
                args,
                supcon_path,
                init_ckpt_path,
            )

        if epoch - best_ep >= args.patience:
            print(f"  Early stop ep={epoch} (best={best_ep})", flush=True)
            break

    # Load best and evaluate.
    best = torch.load(best_path, map_location=device, weights_only=False)
    cond_proj.load_state_dict(best["cond_proj"])
    for name, param in unet.named_parameters():
        if name in best["unet_lora"]:
            param.data.copy_(best["unet_lora"][name].to(device))

    diag = evaluate_lora(
        unet,
        cond_proj,
        eeg_enc,
        test_loader,
        vae,
        dino,
        proto_dino,
        acp,
        args.num_timesteps,
        device,
        n_samples=args.eval_n_samples,
    )

    result = {
        "sid": sid,
        "condition": condition,
        "best_ep": best_ep,
        "best_loss": best_loss,
        "top1": diag["top1"],
        "top3": diag["top3"],
        "top5": diag["top5"],
        "dominant": diag["dominant"],
        "entropy": diag["entropy"],
        "per_class_total": args.per_class_total,
        "init_lora_ckpt": init_ckpt_path or "",
        "save_dir": save_dir,
    }
    print(
        f"  [Done Exp43 {condition.upper()} S{sid:02d}] "
        f"DINO@1={diag['top1']:.4f} @3={diag['top3']:.4f} @5={diag['top5']:.4f} "
        f"entropy={diag['entropy']:.3f} dominant={diag['dominant']*100:.1f}% "
        f"best_ep={best_ep}",
        flush=True,
    )

    del unet, cond_proj, eeg_enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./preproc_vi_re")
    parser.add_argument("--img_root", default="./preproc_data_vi/images")
    parser.add_argument("--subject_ids", default="24")
    parser.add_argument("--conditions", default="c0,c1", help="Comma list: c0,c1")
    parser.add_argument("--n_ch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--n_eeg_tokens", type=int, default=16)
    parser.add_argument("--per_class_total", type=int, default=60,
                        help="Cap trials per class before split. Use 0 for all trials.")
    parser.add_argument("--eval_n_samples", type=int, default=54)
    parser.add_argument("--deep_proj", action="store_true")
    parser.add_argument("--augment_targets", action="store_true")
    parser.add_argument("--supcon_ckpt", default="checkpoints_vsre_dino/20260604_091352_ch32_merged_ep200_supcon")
    parser.add_argument("--vs_ckpt_roots", nargs="*", default=[
        "/content/drive/MyDrive/vsvi_data/checkpoints_vsre_lora_gen",
        "checkpoints_vsre_lora_gen",
    ])
    parser.add_argument("--init_lora_ckpt", default="auto",
                        help="C1 init checkpoint file/dir, or 'auto' to search --vs_ckpt_roots.")
    parser.add_argument("--dino_model", default="dinov2_vits14")
    parser.add_argument("--ckpt_root", default="/content/drive/MyDrive/vsvi_data/checkpoints_exp43_vi_lora")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--grad_ckpt", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.per_class_total <= 0:
        args.per_class_total = 0

    subject_ids = parse_subject_ids(args.subject_ids)
    conditions = parse_conditions(args.conditions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}  Exp43 VI LoRA", flush=True)
    if device.type != "cuda":
        print("[WARN] CUDA is unavailable. This run will be extremely slow.", flush=True)

    # Preflight.
    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"data_root not found: {args.data_root}")
    for sid in subject_ids:
        if not glob.glob(os.path.join(args.data_root, f"preproc_subj_{sid:02d}_*.npz")) and not glob.glob(
            os.path.join(args.data_root, f"preproc_subj_{sid:02d}_*.mat")
        ):
            raise FileNotFoundError(f"No VI data files for S{sid:02d} in {args.data_root}")
        supcon_path = os.path.join(args.supcon_ckpt, f"subj{sid:02d}_best.pt")
        if not os.path.isfile(supcon_path):
            raise FileNotFoundError(f"Missing SupCon checkpoint for S{sid:02d}: {supcon_path}")

    os.makedirs(args.ckpt_root, exist_ok=True)

    # DINO.
    hub_dir = os.path.expanduser("~/.cache/torch/hub")
    dino_local = os.path.join(hub_dir, "facebookresearch_dinov2_main")
    print("[INFO] Loading DINO...", flush=True)
    if os.path.isdir(dino_local):
        dino = torch.hub.load(dino_local, args.dino_model, source="local", verbose=False)
    else:
        print("[INFO] local DINO cache not found, downloading facebookresearch/dinov2...", flush=True)
        dino = torch.hub.load("facebookresearch/dinov2", args.dino_model, verbose=False)
    dino = dino.to(device).eval()
    for p in dino.parameters():
        p.requires_grad_(False)
    dino_feat_dim = DINO_DIM[args.dino_model]
    proto_dino = compute_class_prototypes(dino, args.img_root, CLS_LIST, device)

    # VAE.
    print("[INFO] Loading SD VAE...", flush=True)
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    _, _, acp = make_schedule(args.num_timesteps, device)

    # Targets.
    if args.augment_targets:
        cls_latents_aug = encode_class_images_512_aug(vae, args.img_root, CLS_LIST, device, n_aug=4)
        cls_latents = torch.stack([x[0] for x in cls_latents_aug])
        print("[INFO] Class target latents: augmented 4 views/class", flush=True)
    else:
        cls_latents = encode_class_images_512(vae, args.img_root, CLS_LIST, device)
        cls_latents_aug = None
        print(f"[INFO] Class target latents: {tuple(cls_latents.shape)}", flush=True)

    all_results: list[dict] = []
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    percls_tag = f"percls{args.per_class_total}" if args.per_class_total else "full"

    for sid in subject_ids:
        train_ds = SubjectClassLimitedDataset(
            args.data_root,
            sid,
            n_ch=args.n_ch,
            split="train",
            seed=args.seed,
            per_class_total=args.per_class_total,
        )
        val_ds = SubjectClassLimitedDataset(
            args.data_root,
            sid,
            n_ch=args.n_ch,
            split="val",
            seed=args.seed,
            per_class_total=args.per_class_total,
        )
        test_ds = SubjectClassLimitedDataset(
            args.data_root,
            sid,
            n_ch=args.n_ch,
            split="test",
            seed=args.seed,
            per_class_total=args.per_class_total,
        )
        print(f"[INFO] S{sid:02d} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)
        if len(train_ds) == 0 or len(test_ds) == 0:
            raise RuntimeError(f"Empty train/test dataset for S{sid:02d}")

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        for condition in conditions:
            save_dir = os.path.join(
                args.ckpt_root,
                f"{ts}_exp43_vi_{condition}_s{sid:02d}_lora_r{args.lora_r}_tok{args.n_eeg_tokens}_ep{args.epochs}_{percls_tag}",
            )
            os.makedirs(save_dir, exist_ok=True)
            result = train_condition(
                sid,
                condition,
                args,
                device,
                vae,
                dino,
                proto_dino,
                dino_feat_dim,
                acp,
                train_loader,
                test_loader,
                cls_latents,
                cls_latents_aug,
                save_dir,
            )
            all_results.append(result)

            out_csv = os.path.join(save_dir, "results_exp43_vi_lora.csv")
            with open(out_csv, "w", newline="") as f:
                fieldnames = [
                    "sid",
                    "condition",
                    "best_ep",
                    "best_loss",
                    "top1",
                    "top3",
                    "top5",
                    "dominant",
                    "entropy",
                    "per_class_total",
                    "init_lora_ckpt",
                    "save_dir",
                ]
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerow(result)
            print(f"[INFO] Saved condition CSV: {out_csv}", flush=True)

    # One run-level summary in Drive only.
    summary_csv = os.path.join(args.ckpt_root, f"{ts}_exp43_vi_summary.csv")
    if all_results:
        with open(summary_csv, "w", newline="") as f:
            fieldnames = [
                "sid",
                "condition",
                "best_ep",
                "best_loss",
                "top1",
                "top3",
                "top5",
                "dominant",
                "entropy",
                "per_class_total",
                "init_lora_ckpt",
                "save_dir",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_results)
        print(f"[INFO] Saved run summary CSV: {summary_csv}", flush=True)

    print("\nSummary:", flush=True)
    for r in all_results:
        print(
            f"  S{int(r['sid']):02d} {str(r['condition']).upper()}: "
            f"DINO@1={float(r['top1']):.4f} @3={float(r['top3']):.4f} "
            f"dominant={float(r['dominant'])*100:.1f}% entropy={float(r['entropy']):.3f} "
            f"best_ep={int(r['best_ep'])}",
            flush=True,
        )


if __name__ == "__main__":
    main()
