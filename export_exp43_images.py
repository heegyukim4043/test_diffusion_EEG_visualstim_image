"""
export_exp43_images.py
======================
Exp43 LoRA checkpoint에서 생성 이미지를 뽑아 PNG grid로 저장.
학습 코드(train_vs_re_lora_gen.py)의 검증된 생성 로직을 재사용. 재학습 없음.

사용법:
  python export_exp43_images.py \
    --ckpt <.../subjNN_exp43_c0_lora_best.pt> \
    --supcon_ckpt <.../checkpoints_vsre_dino/RUN> \
    --subject_id 24 --lora_r 32 --per_class_total 150 \
    --out_dir /content/drive/MyDrive/vsvi_data/gen_images
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from types import SimpleNamespace

from train_vs_re_latent_gen import build_eeg_encoder, make_schedule
from train_vs_re_lora_gen import (
    CLS_LIST,
    EEGConditionProjector,
    load_sd15_unet_lora,
    sample_sd_ddim,
    VAE_SCALE,
    DINO_EVAL_TF,
)
import train_exp43_vi_lora as exp43

CLASS_NAMES = {1:"airplane",2:"cup",3:"tree",4:"digit1",5:"digit3",
               6:"digit5",7:"heart",8:"star",9:"triangle"}


@torch.no_grad()
def build_proto_dino(img_root, dino, device):
    """class 이미지를 DINO로 인코딩해 proto_dino (9, feat_dim) 생성."""
    protos = []
    for c in CLS_LIST:
        img = Image.open(os.path.join(img_root, f"{c:02d}.png")).convert("RGB")
        x = DINO_EVAL_TF(img).unsqueeze(0).to(device)
        feat = F.normalize(dino(x), dim=-1)
        protos.append(feat)
    return torch.cat(protos, dim=0)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--supcon_ckpt", required=True)
    p.add_argument("--data_root", default="/content/vsvi_project/preproc_vi_re")
    p.add_argument("--img_root", default="/content/vsvi_project/preproc_data_vi/images")
    p.add_argument("--subject_id", type=int, required=True)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--n_eeg_tokens", type=int, default=16)
    p.add_argument("--per_class_total", type=int, default=135)
    p.add_argument("--samples_per_class", type=int, default=3)
    p.add_argument("--ddim_steps", type=int, default=30)
    p.add_argument("--out_dir", default="/content/drive/MyDrive/vsvi_data/gen_images")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    condition = "c1" if "_c1_" in args.ckpt else "c0"

    # DINO
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    for pr in dino.parameters():
        pr.requires_grad = False
    dino_feat_dim = 384

    # VAE
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae"
    ).to(device).eval()
    for pr in vae.parameters():
        pr.requires_grad = False

    proto_dino = build_proto_dino(args.img_root, dino, device)

    # EEG encoder
    enc_args = SimpleNamespace(eeg_occipital_ids="auto")
    eeg_enc = build_eeg_encoder(32, dino_feat_dim, enc_args, device)
    supcon_path = os.path.join(args.supcon_ckpt, f"subj{args.subject_id:02d}_best.pt")
    ckpt_enc = torch.load(supcon_path, map_location=device, weights_only=False)
    key = "eeg_enc" if "eeg_enc" in ckpt_enc else "model"
    eeg_enc.load_state_dict(ckpt_enc[key])
    eeg_enc.eval()
    print(f"[Encoder] {supcon_path} (key={key})")

    # SD UNet + LoRA
    unet = load_sd15_unet_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    unet = unet.to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    for name, param in unet.named_parameters():
        if name in ckpt["unet_lora"]:
            param.data.copy_(ckpt["unet_lora"][name].to(device))
    unet.eval()
    print(f"[UNet+LoRA] {args.ckpt}")

    # cond_proj
    cond_proj = EEGConditionProjector(
        eeg_dim=512, sd_dim=768, n_tokens=args.n_eeg_tokens, deep=False
    ).to(device).eval()
    cond_proj.load_state_dict(ckpt["cond_proj"])

    acp = make_schedule(1000, device)

    # VI test dataset
    test_ds = exp43.SubjectClassLimitedDataset(
        data_root=args.data_root,
        sid=args.subject_id,
        split="test",
        per_class_total=args.per_class_total,
        img_root=args.img_root,
    )

    by_class = {c: [] for c in range(9)}
    for i in range(len(test_ds)):
        eeg, _, lbl = test_ds[i]
        c = int(lbl)
        if len(by_class[c]) < args.samples_per_class:
            by_class[c].append(eeg)

    subj = torch.zeros(1, dtype=torch.long, device=device)
    generated = {c: [] for c in range(9)}
    for c in range(9):
        for eeg in by_class[c]:
            eeg = eeg.unsqueeze(0).to(device)
            eeg_lat = eeg_enc.encode_eeg(eeg, subj.expand(1))
            gen_lat = sample_sd_ddim(
                unet, cond_proj, eeg_lat, acp, 1000,
                steps=args.ddim_steps, device=device
            )
            decoded = vae.decode(gen_lat / VAE_SCALE).sample.clamp(-1, 1)
            decoded = (decoded + 1) / 2
            img = (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            generated[c].append(Image.fromarray(img))
        print(f"  class {c+1} ({CLASS_NAMES[c+1]}): {len(generated[c])} imgs")

    spc = args.samples_per_class
    fig, axes = plt.subplots(spc, 9, figsize=(9 * 1.5, spc * 1.5))
    for c in range(9):
        for s in range(spc):
            ax = axes[s, c] if spc > 1 else axes[c]
            if s < len(generated[c]):
                ax.imshow(generated[c][s])
            ax.axis("off")
            if s == 0:
                ax.set_title(CLASS_NAMES[c + 1], fontsize=9)
    fig.suptitle(f"S{args.subject_id:02d} Exp43 {condition}", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(
        args.out_dir, f"S{args.subject_id:02d}_exp43_{condition}_grid.png"
    )
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
