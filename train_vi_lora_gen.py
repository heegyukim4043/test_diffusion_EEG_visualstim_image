"""
train_vi_lora_gen.py
────────────────────────────────────────────────────────────────────────────
Exp43: SD 1.5 LoRA VS→VI fine-tuning.

Uses preproc_vi_re (repeated-session VI data, same HDF5 format as preproc_vs_re).

Comparison modes:
  C0: VI scratch LoRA  (random-init UNet LoRA + cond_proj + EEG encoder)
  C1: VS LoRA → VI fine-tune with frozen EEG encoder
  C2: VS LoRA → VI fine-tune with staged encoder unfreeze at epoch 51

Checkpoint format loaded from VS LoRA run:
  {unet_lora: {name: param}, cond_proj: state_dict, sid: int}
SupCon encoder loaded from:
  checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon/subj{sid:02d}_best.pt

Usage:
  python train_vi_lora_gen.py --subject_ids 18,1 --modes C0,C1 --epochs 100
  python train_vi_lora_gen.py --subject_ids 1 --modes C1 --lora_r 32 --n_eeg_tokens 16 \\
      --vs_lora_ckpt checkpoints_vsre_lora_gen/20260617_074245_lora_r32_ep100
"""

import argparse, csv, datetime, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_vs_re import VSReDataset, collate_fn, available_subjects
from model_eeg_dino import EEGDINORegressor, DINO_DIM, LATENT_DIM
from train_crosssubj_dino import set_seed, EEGAugment, compute_class_prototypes
from train_vs_re_latent_gen import make_schedule, build_eeg_encoder, VAE_SCALE
from train_vs_re_lora_gen import (
    load_sd15_unet_lora, EEGConditionProjector,
    ddpm_q_sample, sample_sd_ddim,
    encode_class_images_512, IMG_TRANSFORM_512, DINO_EVAL_TF,
)

N_CLASSES = 9
CLS_LIST  = list(range(1, 10))


@torch.no_grad()
def evaluate_vi_lora(unet, cond_proj, eeg_enc, test_loader, vae, dino,
                      proto_dino, acp, T, device):
    """Full-test DDIM evaluation for VI generation."""
    subj = torch.zeros(1, dtype=torch.long, device=device)
    correct = {1: 0, 3: 0, 5: 0}
    pred_classes, true_classes = [], []
    total = 0
    unet.eval(); cond_proj.eval(); eeg_enc.eval()

    for eeg, _, lbl in test_loader:
        eeg = eeg.to(device); lbl = lbl.to(device)
        eeg_lat = eeg_enc.encode_eeg(eeg, subj.expand(eeg.size(0)))
        gen_lat = sample_sd_ddim(unet, cond_proj, eeg_lat, acp, T,
                                  steps=30, device=device)
        decoded = vae.decode(gen_lat / VAE_SCALE).sample.clamp(-1, 1)
        decoded = (decoded + 1) / 2
        imgs = torch.stack([DINO_EVAL_TF(x) for x in decoded])
        feats = F.normalize(dino(imgs), dim=-1)
        sims  = feats @ F.normalize(proto_dino, dim=-1).T
        preds = sims.argmax(dim=-1).cpu().tolist()
        pred_classes.extend(preds)
        true_classes.extend(lbl.cpu().tolist())
        for k in correct:
            topk = sims.topk(min(k, 9), dim=1).indices
            correct[k] += topk.eq(lbl.unsqueeze(1)).any(1).sum().item()
        total += eeg.size(0)

    cnt = Counter(pred_classes)
    dominant = cnt.most_common(1)[0][1] / max(len(pred_classes), 1)
    counts = np.array(list(cnt.values()), dtype=float)
    entropy = float(-np.sum((counts / counts.sum()) * np.log(counts / counts.sum() + 1e-9)))
    return {
        "top1": correct[1] / max(total, 1),
        "top3": correct[3] / max(total, 1),
        "top5": correct[5] / max(total, 1),
        "dominant": dominant,
        "entropy": entropy,
    }


def train_vi_mode(mode, sid, args, vae, dino, proto_dino, dino_feat_dim,
                   acp, device, save_dir):
    print(f"\n{'='*55}\n  S{sid:02d} mode={mode}", flush=True)
    set_seed(args.seed)

    # VI data (preproc_vi_re, same HDF5 structure as VS)
    train_ds = VSReDataset(args.vi_root, [sid], n_ch=args.n_ch, split="train", seed=args.seed)
    val_ds   = VSReDataset(args.vi_root, [sid], n_ch=args.n_ch, split="val",   seed=args.seed)
    test_ds  = VSReDataset(args.vi_root, [sid], n_ch=args.n_ch, split="test",  seed=args.seed)
    if len(train_ds) == 0:
        print(f"  [SKIP] No VI data for S{sid:02d}", flush=True)
        return None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=0, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                               num_workers=0, collate_fn=collate_fn)
    print(f"  VI train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}", flush=True)

    cls_latents = encode_class_images_512(vae, args.img_root, CLS_LIST, device)

    # EEG encoder — always build fresh, then optionally load weights
    eeg_enc = build_eeg_encoder(args.n_ch, dino_feat_dim,
                                 type('a', (), {'eeg_occipital_ids': 'auto'})(), device)

    if mode in ("C1", "C2") and args.supcon_ckpt:
        enc_path = os.path.join(args.supcon_ckpt, f"subj{sid:02d}_best.pt")
        if os.path.isfile(enc_path):
            ckpt_enc = torch.load(enc_path, map_location=device, weights_only=False)
            key = "eeg_enc" if "eeg_enc" in ckpt_enc else "model"
            eeg_enc.load_state_dict(ckpt_enc[key])
            print(f"  [Encoder] Loaded SupCon weights from {enc_path}", flush=True)
        else:
            print(f"  [WARN] SupCon ckpt not found: {enc_path}, using random init", flush=True)

    # SD 1.5 UNet with LoRA
    unet = load_sd15_unet_lora(args.lora_r, args.lora_alpha).to(device)

    # EEG conditioning projector
    cond_proj = EEGConditionProjector(
        eeg_dim=LATENT_DIM, sd_dim=768, n_tokens=args.n_eeg_tokens,
        deep=getattr(args, 'deep_proj', False)
    ).to(device)

    # For C1/C2: load VS LoRA weights (UNet LoRA + cond_proj)
    if mode in ("C1", "C2") and args.vs_lora_ckpt:
        lora_path = os.path.join(args.vs_lora_ckpt, f"subj{sid:02d}_lora_best.pt")
        if os.path.isfile(lora_path):
            ckpt_lora = torch.load(lora_path, map_location=device, weights_only=False)
            # Load LoRA params into unet
            unet_state = dict(unet.named_parameters())
            for name, param in ckpt_lora["unet_lora"].items():
                if name in unet_state and unet_state[name].requires_grad:
                    unet_state[name].data.copy_(param)
            # Load cond_proj
            cond_proj.load_state_dict(ckpt_lora["cond_proj"])
            print(f"  [LoRA+Proj] Loaded VS LoRA weights from {lora_path}", flush=True)
        else:
            print(f"  [WARN] VS LoRA ckpt not found: {lora_path}, starting from random LoRA", flush=True)

    # Freeze/unfreeze EEG encoder
    if mode == "C1":
        for p in eeg_enc.parameters():
            p.requires_grad_(False)
        print(f"  [Encoder] Frozen (C1)", flush=True)
    elif mode == "C0":
        # C0 = scratch: use random encoder, allow training
        for p in eeg_enc.parameters():
            p.requires_grad_(True)
        print(f"  [Encoder] Random init, trainable (C0)", flush=True)
    elif mode == "C2":
        # C2: start frozen like C1, unfreeze after epoch 50
        for p in eeg_enc.parameters():
            p.requires_grad_(False)
        print(f"  [Encoder] Frozen initially (C2, unfreeze at ep 51)", flush=True)

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    train_params = lora_params + list(cond_proj.parameters())
    if mode in ("C0",):
        train_params += list(eeg_enc.parameters())
    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.wd)

    augmenter = EEGAugment(noise_std=0.03, scale_range=(0.9, 1.1), ch_drop_prob=0.05,
                            max_shift=5, freq_noise_std=0.0,
                            p_noise=0.5, p_scale=0.5, p_drop=0.2, p_shift=0.2, p_freq=0.0)

    best_val = float("inf")
    best_ep  = 0
    subj_t   = torch.zeros(1, dtype=torch.long, device=device)
    best_path = os.path.join(save_dir, f"subj{sid:02d}_{mode}_best.pt")

    print(f"    Ep    TrLoss   ValLoss", flush=True)
    for epoch in range(1, args.epochs + 1):

        # C2: staged unfreeze at epoch 51
        if mode == "C2" and epoch == 51:
            for p in eeg_enc.parameters():
                p.requires_grad_(True)
            enc_params = list(eeg_enc.parameters())
            optimizer = torch.optim.AdamW(
                lora_params + list(cond_proj.parameters()) + enc_params,
                lr=args.lr, weight_decay=args.wd
            )
            # Use a much smaller LR for the encoder
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * 0.01 if pg['params'] is enc_params else args.lr
            print(f"  [C2] Encoder unfrozen at ep 51, enc_lr={args.lr*0.01:.2e}", flush=True)

        unet.train(); cond_proj.train()
        if mode in ("C0",): eeg_enc.train()

        tr_loss = 0.0
        for eeg, _, lbl in train_loader:
            eeg = augmenter(eeg.to(device)); lbl = lbl.to(device)
            with torch.set_grad_enabled(mode not in ("C1",) or epoch > 50):
                eeg_lat = eeg_enc.encode_eeg(eeg, subj_t.expand(eeg.size(0)))
            if mode == "C1":
                eeg_lat = eeg_lat.detach()
            cond_tokens = cond_proj(eeg_lat)
            x0 = cls_latents[lbl]
            t  = torch.randint(0, args.num_timesteps, (eeg.size(0),), device=device)
            xt, noise = ddpm_q_sample(x0, t, acp)
            noise_pred = unet(xt, t, encoder_hidden_states=cond_tokens).sample
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(unet.parameters()) + list(cond_proj.parameters()), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= max(len(train_loader), 1)

        # Validation loss
        unet.eval(); cond_proj.eval(); eeg_enc.eval()
        val_loss = 0.0
        with torch.no_grad():
            for eeg, _, lbl in val_loader:
                eeg = eeg.to(device); lbl = lbl.to(device)
                eeg_lat = eeg_enc.encode_eeg(eeg, subj_t.expand(eeg.size(0)))
                cond_tokens = cond_proj(eeg_lat)
                x0 = cls_latents[lbl]
                t  = torch.randint(0, args.num_timesteps, (eeg.size(0),), device=device)
                xt, noise = ddpm_q_sample(x0, t, acp)
                val_loss += F.mse_loss(unet(xt, t, encoder_hidden_states=cond_tokens).sample, noise).item()
        val_loss /= max(len(val_loader), 1)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  {epoch:5d}  {tr_loss:.5f}  {val_loss:.5f}", flush=True)

        if val_loss < best_val:
            best_val = val_loss; best_ep = epoch
            torch.save({
                "unet_lora": {k: v for k, v in unet.named_parameters() if v.requires_grad},
                "cond_proj": cond_proj.state_dict(),
                "eeg_enc":   eeg_enc.state_dict(),
                "sid": sid, "mode": mode,
            }, best_path)

        if epoch - best_ep >= args.patience:
            print(f"  Early stop ep={epoch} (best={best_ep})", flush=True)
            break

    # Reload best checkpoint and evaluate
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    cond_proj.load_state_dict(ckpt["cond_proj"])
    eeg_enc.load_state_dict(ckpt["eeg_enc"])
    for name, param in unet.named_parameters():
        if name in ckpt["unet_lora"]:
            param.data.copy_(ckpt["unet_lora"][name])

    diag = evaluate_vi_lora(unet, cond_proj, eeg_enc, test_loader, vae, dino,
                              proto_dino, acp, args.num_timesteps, device)
    print(f"  [Done S{sid:02d} {mode}]  DINO@1={diag['top1']:.4f} @3={diag['top3']:.4f} "
          f"@5={diag['top5']:.4f}  entropy={diag['entropy']:.3f}  "
          f"dominant={diag['dominant']*100:.1f}%  best_ep={best_ep}", flush=True)
    return {"sid": sid, "mode": mode, "best_ep": best_ep, **diag}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vi_root",       default="./preproc_vi_re",
                        help="preproc_vi_re directory (repeated-session VI data)")
    parser.add_argument("--img_root",      default="./preproc_data_vi/images")
    parser.add_argument("--subject_ids",   default="18,1")
    parser.add_argument("--n_ch",          type=int,   default=32)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--patience",      type=int,   default=30)
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--wd",            type=float, default=1e-4)
    parser.add_argument("--num_timesteps", type=int,   default=1000)
    parser.add_argument("--lora_r",        type=int,   default=16)
    parser.add_argument("--lora_alpha",    type=int,   default=32)
    parser.add_argument("--n_eeg_tokens",  type=int,   default=16)
    parser.add_argument("--deep_proj",     action="store_true")
    parser.add_argument("--modes",         default="C0,C1",
                        help="Comma-separated: C0=scratch, C1=frozen VS init, C2=staged unfreeze")
    parser.add_argument("--supcon_ckpt",
                        default="checkpoints_vsre_dino/20260530_095045_ch32_merged_ep200_supcon",
                        help="SupCon encoder checkpoint dir for C1/C2 EEG encoder init")
    parser.add_argument("--vs_lora_ckpt",
                        default="",
                        help="VS LoRA checkpoint dir (subj{sid}_lora_best.pt) for C1/C2 init")
    parser.add_argument("--dino_model",    default="dinov2_vits14")
    parser.add_argument("--ckpt_root",     default="./checkpoints_vi_lora_gen")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}  Exp43 VI SD LoRA generator", flush=True)

    subject_ids = []
    for tok in args.subject_ids.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-"); subject_ids.extend(range(int(a), int(b)+1))
        else:
            subject_ids.append(int(tok))
    modes = [m.strip() for m in args.modes.split(",")]

    _hub_dir = os.path.expanduser("~/.cache/torch/hub")
    print("[INFO] Loading DINO...", flush=True)
    dino = torch.hub.load(os.path.join(_hub_dir, "facebookresearch_dinov2_main"),
                          args.dino_model, source="local", verbose=False)
    dino = dino.to(device).eval()
    for p in dino.parameters(): p.requires_grad_(False)
    dino_feat_dim = DINO_DIM[args.dino_model]
    proto_dino = compute_class_prototypes(dino, args.img_root, CLS_LIST, device)

    print("[INFO] Loading SD VAE...", flush=True)
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    for p in vae.parameters(): p.requires_grad_(False)

    _, _, acp = make_schedule(args.num_timesteps, device)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"r{args.lora_r}_tok{args.n_eeg_tokens}"
    save_dir = os.path.join(args.ckpt_root, f"{ts}_vi_lora_{tag}_ep{args.epochs}")
    os.makedirs(save_dir, exist_ok=True)

    all_results = []
    for sid in subject_ids:
        for mode in modes:
            r = train_vi_mode(mode, sid, args, vae, dino, proto_dino, dino_feat_dim,
                               acp, device, save_dir)
            if r:
                all_results.append(r)

    print("\nSummary:")
    for r in all_results:
        print(f"  S{r['sid']:02d} {r['mode']:>3}: DINO@1={r['top1']:.4f} @3={r['top3']:.4f} "
              f"dominant={r['dominant']*100:.1f}%  entropy={r['entropy']:.3f}")

    out_csv = os.path.join(save_dir, "results_vi_lora_gen.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sid","mode","best_ep","top1","top3","top5","dominant","entropy"])
        w.writeheader(); w.writerows(all_results)
    print(f"[INFO] Saved: {save_dir}", flush=True)


if __name__ == "__main__":
    main()
