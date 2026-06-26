"""
train_vi_latent_gen.py
────────────────────────────────────────────────────────────────────────────
Exp38: Gen-WS-03 — same-subject VI generation via VAE latent diffusion.

Comparisons on S18 VI data:
  C0: random-init encoder + random-init UNetCA (VI scratch)
  C1: frozen VS-pretrained encoder (Exp37 S18) + fine-tune UNetCA on VI
  C2 (optional): staged unfreeze after 50 epochs

Usage:
  python train_vi_latent_gen.py --subject_ids 18
  python train_vi_latent_gen.py --subject_ids 18,1 --epochs 100
"""

import argparse, csv, datetime, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_vs_re_exp25_vi_transfer import VIDataset
from model_eeg_dino import EEGDINORegressor, DINO_DIM, LATENT_DIM
from train_crosssubj_dino import set_seed, EEGAugment, compute_class_prototypes
from train_vs_re_latent_gen import (
    LatentUNetCA, make_schedule, sample_ddim,
    encode_class_images, VAE_SCALE, IMG_TRANSFORM,
    build_eeg_encoder, load_supcon_encoder,
)

N_CLASSES = 9
CLS_LIST  = list(range(1, 10))


def collate_vi(batch):
    eeg = torch.stack([b[0] for b in batch])
    lbl = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return eeg, lbl


@torch.no_grad()
def collapse_diagnostics_vi(unet, eeg_encoder, test_loader, vae, dino,
                              proto_dino, acp, num_timesteps, device):
    dino_tf = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    subj = torch.zeros(1, dtype=torch.long, device=device)
    pred_classes, true_classes = [], []
    unet.eval(); eeg_encoder.eval()

    from collections import Counter
    for eeg, lbl in test_loader:
        eeg = eeg.to(device)
        eeg_lat = eeg_encoder.encode_eeg(eeg, subj.expand(eeg.size(0)))
        gen = sample_ddim(unet, eeg_lat, acp, num_timesteps, steps=50, device=device)
        decoded = vae.decode(gen / VAE_SCALE).sample.clamp(-1,1)
        decoded = (decoded + 1) / 2
        imgs = torch.stack([dino_tf(x) for x in decoded])
        feats = F.normalize(dino(imgs), dim=-1)
        sims  = feats @ F.normalize(proto_dino, dim=-1).T
        preds = sims.argmax(dim=-1).cpu().tolist()
        pred_classes.extend(preds)
        true_classes.extend(lbl.tolist())

    n = len(pred_classes)
    top1 = sum(p==t for p,t in zip(pred_classes,true_classes)) / n
    top3 = sum(t in pred_classes[i*1:(i*1)+1] for i, t in enumerate(true_classes)) / n  # approx via top-k
    # Compute top-k properly
    subj_t = torch.zeros(1, dtype=torch.long, device=device)
    correct = {1:0, 3:0, 5:0}
    total = 0
    with torch.no_grad():
        for eeg, lbl in test_loader:
            eeg = eeg.to(device)
            eeg_lat = eeg_encoder.encode_eeg(eeg, subj_t.expand(eeg.size(0)))
            gen = sample_ddim(unet, eeg_lat, acp, num_timesteps, steps=50, device=device)
            decoded = vae.decode(gen / VAE_SCALE).sample.clamp(-1,1)
            decoded = (decoded + 1) / 2
            imgs = torch.stack([dino_tf(x) for x in decoded])
            feats = F.normalize(dino(imgs), dim=-1)
            sims  = feats @ F.normalize(proto_dino, dim=-1).T
            for k in correct:
                topk = sims.topk(min(k, 9), dim=1).indices
                correct[k] += topk.eq(lbl.to(device).unsqueeze(1)).any(1).sum().item()
            total += eeg.size(0)

    cnt = Counter(pred_classes)
    dominant_frac = cnt.most_common(1)[0][1] / n
    counts = np.array(list(cnt.values()), dtype=float)
    probs  = counts / counts.sum()
    entropy = float(-np.sum(probs * np.log(probs + 1e-9)))

    return {
        "top1": correct[1]/max(total,1),
        "top3": correct[3]/max(total,1),
        "top5": correct[5]/max(total,1),
        "dominant": dominant_frac,
        "entropy": entropy,
    }


def train_vi_mode(mode, sid, args, vae, dino, proto_dino, dino_feat_dim,
                   acp, device, save_dir, vs_ckpt_dir=""):
    print(f"\n--- S{sid:02d} mode={mode} ---", flush=True)

    train_ds = VIDataset(args.vi_root, sid, args.n_ch, "train", args.seed)
    val_ds   = VIDataset(args.vi_root, sid, args.n_ch, "val",   args.seed)
    test_ds  = VIDataset(args.vi_root, sid, args.n_ch, "test",  args.seed)

    print(f"  VI n_time={train_ds.n_time}  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, collate_fn=collate_vi)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_vi)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_vi)

    cls_latents = encode_class_images(vae, args.img_root, CLS_LIST, device)

    # EEG encoder
    eeg_enc = build_eeg_encoder(args.n_ch, dino_feat_dim,
                                 type('a', (), {'eeg_occipital_ids': 'auto'})(), device)
    if mode in ("C1", "C2") and vs_ckpt_dir:
        ckpt_path = os.path.join(vs_ckpt_dir, f"subj{sid:02d}_best.pt")
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            # Exp37 latent-gen format: {"unet":..., "eeg_enc":..., ...}
            if "eeg_enc" in ckpt:
                eeg_enc.load_state_dict(ckpt["eeg_enc"], strict=True)
                print(f"  [Encoder] Loaded eeg_enc from Exp37 checkpoint: {ckpt_path}", flush=True)
            # SupCon dino format: {"model": EEGDINORegressor_state, ...}
            elif "model" in ckpt:
                eeg_enc.load_state_dict(ckpt["model"], strict=True)
                print(f"  [Encoder] Loaded model from SupCon checkpoint: {ckpt_path}", flush=True)
            else:
                print(f"  [WARN] Unknown checkpoint format, keys: {list(ckpt.keys())}", flush=True)
        else:
            print(f"  [WARN] VS checkpoint not found: {ckpt_path}, using random init", flush=True)
    if mode == "C1":
        for p in eeg_enc.parameters():
            p.requires_grad_(False)
        print(f"  [Encoder] Frozen (C1)", flush=True)

    unet = LatentUNetCA(in_ch=4, base_ch=args.base_ch,
                        ch_mult=tuple(int(x) for x in args.ch_mult.split(",")),
                        time_dim=256, eeg_dim=LATENT_DIM,
                        n_heads=4, n_eeg_tokens=8).to(device)

    params = list(unet.parameters())
    if mode != "C1":
        params += list(eeg_enc.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    augmenter = EEGAugment(noise_std=0.03, scale_range=(0.9,1.1), ch_drop_prob=0.05,
                            max_shift=5, freq_noise_std=0.0,
                            p_noise=0.5, p_scale=0.5, p_drop=0.2, p_shift=0.2, p_freq=0.0)

    best_val = float("inf")
    best_ep  = 0
    patience  = args.patience
    best_path = os.path.join(save_dir, f"subj{sid:02d}_{mode}_best.pt")
    subj_t = torch.zeros(1, dtype=torch.long, device=device)

    print(f"    Ep    TrLoss   ValLoss", flush=True)
    for epoch in range(1, args.epochs + 1):
        # Staged unfreeze for C2: unfreeze encoder after 50 epochs
        if mode == "C2" and epoch == 51:
            for p in eeg_enc.parameters():
                p.requires_grad_(True)
            optimizer = torch.optim.AdamW(
                list(unet.parameters()) + list(eeg_enc.parameters()),
                lr=args.lr * 0.1, weight_decay=args.wd
            )
            print(f"  [C2] Encoder unfrozen at epoch 51", flush=True)

        unet.train(); eeg_enc.train()
        tr_loss = 0.0
        for eeg, lbl in train_loader:
            eeg = augmenter(eeg.to(device))
            lbl = lbl.to(device)
            eeg_lat = eeg_enc.encode_eeg(eeg, subj_t.expand(eeg.size(0)))
            x0  = cls_latents[lbl]
            t   = torch.randint(0, args.num_timesteps, (eeg.size(0),), device=device)
            xt, noise = _q_sample(x0, t, acp)
            pred = unet(xt, t, eeg_lat)
            loss = F.mse_loss(pred, noise)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= max(len(train_loader), 1)

        unet.eval(); eeg_enc.eval()
        val_loss = 0.0
        with torch.no_grad():
            for eeg, lbl in val_loader:
                eeg = eeg.to(device); lbl = lbl.to(device)
                eeg_lat = eeg_enc.encode_eeg(eeg, subj_t.expand(eeg.size(0)))
                x0  = cls_latents[lbl]
                t   = torch.randint(0, args.num_timesteps, (eeg.size(0),), device=device)
                xt, noise = _q_sample(x0, t, acp)
                val_loss += F.mse_loss(unet(xt, t, eeg_lat), noise).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val:
            best_val = val_loss; best_ep = epoch
            torch.save({"unet": unet.state_dict(), "eeg_enc": eeg_enc.state_dict(),
                        "sid": sid, "mode": mode}, best_path)

        if epoch - best_ep >= patience:
            print(f"  Early stop at ep {epoch} (best_ep={best_ep})", flush=True)
            break
        if epoch % 20 == 0:
            print(f"  {epoch:5d}  {tr_loss:.5f}  {val_loss:.5f}", flush=True)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    unet.load_state_dict(ckpt["unet"]); eeg_enc.load_state_dict(ckpt["eeg_enc"])
    diag = collapse_diagnostics_vi(unet, eeg_enc, test_loader, vae, dino, proto_dino,
                                    acp, args.num_timesteps, device)
    print(f"  [Done S{sid:02d} {mode}]  DINO@1={diag['top1']:.4f} @3={diag['top3']:.4f} @5={diag['top5']:.4f}  entropy={diag['entropy']:.3f}  dominant={diag['dominant']*100:.1f}%  best_ep={best_ep}", flush=True)
    return {"sid": sid, "mode": mode, "best_ep": best_ep, **diag}


def _q_sample(x0, t, acp, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    s = acp[t][:, None, None, None].sqrt()
    r = (1 - acp[t])[:, None, None, None].sqrt()
    return s * x0 + r * noise, noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vi_root",      default="./preproc_data_vi")
    parser.add_argument("--img_root",     default="./preproc_data_vi/images")
    parser.add_argument("--subject_ids",  default="18")
    parser.add_argument("--n_ch",         type=int,   default=32)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--patience",     type=int,   default=25)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--wd",           type=float, default=1e-4)
    parser.add_argument("--num_timesteps",type=int,   default=200)
    parser.add_argument("--base_ch",      type=int,   default=128)
    parser.add_argument("--ch_mult",      type=str,   default="1,2,4")
    parser.add_argument("--vs_ckpt_dir",  default="checkpoints_vsre_latent_gen/20260610_105755_ch32_ep300_supcon_frozen_ca",
                        help="Exp37 checkpoint dir for C1/C2 initialization")
    parser.add_argument("--dino_model",   default="dinov2_vits14")
    parser.add_argument("--modes",        default="C0,C1",
                        help="Comma-separated: C0=scratch, C1=frozen VS init, C2=staged unfreeze")
    parser.add_argument("--ckpt_root",    default="./checkpoints_vi_latent_gen")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}  VI latent generation (SD VAE + CA)", flush=True)

    subject_ids = []
    for tok in args.subject_ids.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-"); subject_ids.extend(range(int(a), int(b)+1))
        else:
            subject_ids.append(int(tok))
    modes = [m.strip() for m in args.modes.split(",")]

    print("[INFO] Loading SD VAE...", flush=True)
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    for p in vae.parameters(): p.requires_grad_(False)

    _hub_dir = os.path.expanduser("~/.cache/torch/hub")
    print(f"[INFO] Loading DINO...", flush=True)
    dino = torch.hub.load(os.path.join(_hub_dir, "facebookresearch_dinov2_main"),
                          args.dino_model, source="local", verbose=False)
    dino = dino.to(device).eval()
    for p in dino.parameters(): p.requires_grad_(False)

    dino_feat_dim = DINO_DIM[args.dino_model]
    proto_dino = compute_class_prototypes(dino, args.img_root, CLS_LIST, device)
    _, _, acp = make_schedule(args.num_timesteps, device)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.ckpt_root, f"{ts}_vi_ep{args.epochs}")
    os.makedirs(save_dir, exist_ok=True)

    all_results = []
    for sid in subject_ids:
        for mode in modes:
            r = train_vi_mode(mode, sid, args, vae, dino, proto_dino, dino_feat_dim,
                               acp, device, save_dir, args.vs_ckpt_dir)
            if r:
                all_results.append(r)

    print("\nSummary:")
    for r in all_results:
        print(f"  S{r['sid']:02d} {r['mode']:>3}: DINO@1={r['top1']:.4f} @3={r['top3']:.4f} @5={r['top5']:.4f}  dominant={r['dominant']*100:.1f}%  entropy={r['entropy']:.3f}")

    out_csv = os.path.join(save_dir, "results_vi_latent_gen.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sid","mode","best_ep","top1","top3","top5","dominant","entropy"])
        w.writeheader(); w.writerows(all_results)
    print(f"[INFO] Saved: {save_dir}", flush=True)


if __name__ == "__main__":
    main()
