import argparse
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataset_vs_re import VSReDataset
from model_128_eegonly_transformer_repa import EEGDiffusionModel128


IMG_TRANSFORM = T.Compose([
    T.Resize(128),
    T.CenterCrop(128),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


class GenDataset(Dataset):
    def __init__(self, vsre_ds, img_root, n_classes=9):
        self.base = vsre_ds
        self.class_imgs = []
        for c in range(n_classes):
            p = os.path.join(img_root, f"{c+1:02d}.png")
            img = IMG_TRANSFORM(Image.open(p).convert("RGB"))
            self.class_imgs.append(img)
        self.class_imgs = torch.stack(self.class_imgs)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        eeg, subj, lbl = self.base[i]
        img = self.class_imgs[lbl]
        return eeg, lbl, img


def gen_collate(batch):
    eeg = torch.stack([b[0] for b in batch])
    lbl = torch.tensor([b[1] for b in batch], dtype=torch.long)
    img = torch.stack([b[2] for b in batch])
    return eeg, lbl, img


def tensor_to_uint8(x):
    x = (x.clamp(-1, 1) + 1.0) * 0.5
    x = x.mul(255.0).round().byte().cpu().permute(1, 2, 0).numpy()
    return x


def save_triplet(save_path: Path, gen, tgt, lbl, idx):
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    axes[0].imshow(tensor_to_uint8(tgt))
    axes[0].set_title(f"target cls {lbl+1}")
    axes[0].axis("off")
    axes[1].imshow(tensor_to_uint8(gen))
    axes[1].set_title(f"gen #{idx:03d}")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def save_grid(save_path: Path, imgs, labels, title):
    n = len(imgs)
    ncols = 6
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i, (img, lbl) in enumerate(zip(imgs, labels)):
        axes[i].imshow(tensor_to_uint8(img))
        axes[i].set_title(f"c{lbl+1}", fontsize=7)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def build_model_from_config(cfg, state_dict, device):
    occ = cfg.get("eeg_occipital_indices", None)
    encoder_version = cfg.get("encoder_version")
    if encoder_version is None:
        # Older checkpoints may not store encoder_version. Infer V2 from state-dict keys.
        if any(k.startswith("eeg_encoder.channel_gate.") for k in state_dict):
            encoder_version = "v2"
        else:
            encoder_version = "v1"
    model = EEGDiffusionModel128(
        eeg_channels=cfg.get("eeg_channels", cfg.get("n_ch", 32)),
        num_classes=cfg.get("num_classes", 9),
        num_timesteps=cfg.get("num_timesteps", 200),
        base_channels=cfg.get("base_channels", 64),
        ch_mult=tuple(int(x) for x in str(cfg.get("ch_mult", "1,2,4,4")).split(",")),
        lambda_percept=cfg.get("lambda_percept", 0.1),
        lambda_rec=cfg.get("lambda_rec", 0.01),
        lambda_ssim=cfg.get("lambda_ssim", 0.05),
        lambda_lpips=cfg.get("lambda_lpips", 0.0),
        beta_schedule=cfg.get("beta_schedule", "linear"),
        encoder_version=encoder_version,
        eeg_stem_filters=cfg.get("eeg_stem_filters", 32),
        eeg_occipital_indices=occ,
    ).to(device)
    return model


def export_subject(run_dir: Path, sid: int, ckpt_path: Path, args, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    state_dict = ckpt["ema_model"]
    model = build_model_from_config(cfg, state_dict, device)
    model.load_state_dict(state_dict)
    model.eval()

    subj_map = {sid: 0}
    base_test = VSReDataset(
        args.data_root,
        [sid],
        subj_map,
        cfg.get("n_ch", 32),
        "test",
        cfg.get("seed", 42),
        cfg.get("max_sessions", None),
    )
    test_ds = GenDataset(base_test, args.img_root)
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=gen_collate,
    )

    export_dir = run_dir / f"subj{sid:02d}" / "test_generations"
    export_dir.mkdir(parents=True, exist_ok=True)

    all_gen = []
    all_lbl = []
    offset = 0
    for eeg, lbl, tgt in loader:
        eeg = eeg.to(device)
        with torch.no_grad():
            gen = model.sample_ddim(
                eeg,
                num_steps=args.eval_ddim_steps,
                guidance_scale=args.guidance_scale,
                eta=args.eta,
            )
        for i in range(gen.size(0)):
            save_triplet(
                export_dir / f"test_{offset+i:03d}_cls{lbl[i].item()+1}.png",
                gen[i],
                tgt[i],
                int(lbl[i].item()),
                offset + i,
            )
            all_gen.append(gen[i].cpu())
            all_lbl.append(int(lbl[i].item()))
        offset += gen.size(0)

    save_grid(export_dir / "all_generated_grid.png", all_gen, all_lbl, f"S{sid:02d} generated")
    save_grid(export_dir / "all_targets_grid.png", [test_ds[i][2] for i in range(len(test_ds))], all_lbl, f"S{sid:02d} targets")
    return len(test_ds), export_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data_root", default="./preproc_vs_re")
    parser.add_argument("--img_root", default="./preproc_data_vi/images")
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_ddim_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--eta", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)

    ckpts = sorted(run_dir.glob("subj*/best.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No best.pt found under {run_dir}")

    if args.subjects == "all":
        subject_ids = [int(p.parent.name.replace("subj", "")) for p in ckpts]
    else:
        subject_ids = []
        for tok in args.subjects.split(","):
            tok = tok.strip()
            if "-" in tok:
                a, b = tok.split("-")
                subject_ids.extend(range(int(a), int(b) + 1))
            else:
                subject_ids.append(int(tok))

    for sid in subject_ids:
        ckpt_path = run_dir / f"subj{sid:02d}" / "best.pt"
        if not ckpt_path.exists():
            print(f"[SKIP] S{sid:02d}: checkpoint not found")
            continue
        n_test, export_dir = export_subject(run_dir, sid, ckpt_path, args, device)
        print(f"[DONE] S{sid:02d}: exported {n_test} test generations -> {export_dir}")


if __name__ == "__main__":
    main()
