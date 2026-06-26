# eval_generated_metrics_per_image.py
import argparse
import csv
import os
from glob import glob

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms


def _pair_images(samples_dir):
    gen_files = sorted(glob(os.path.join(samples_dir, "*_GEN.png")))
    pairs = []
    for gen_path in gen_files:
        gt_path = gen_path.replace("_GEN.png", "_GT.png")
        if os.path.isfile(gt_path):
            pairs.append((gen_path, gt_path))
    return pairs


def _load_image(path, img_size):
    img = Image.open(path).convert("RGB")
    if img_size is not None:
        img = img.resize((img_size, img_size), Image.BILINEAR)
    img = transforms.ToTensor()(img)  # [0,1]
    return img


def _gaussian_window(win_size, channel, device, sigma=1.5):
    coords = torch.arange(win_size, device=device).float() - win_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]
    window = window.expand(channel, 1, win_size, win_size).contiguous()
    return window


def _ssim(img1, img2, data_range=1.0, k1=0.01, k2=0.03, win_size=11, eps=1e-8):
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    channel = img1.size(1)
    window = _gaussian_window(win_size, channel, img1.device)

    mu1 = F.conv2d(img1, window, padding=win_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=win_size // 2, groups=channel)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=win_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=win_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=win_size // 2, groups=channel) - mu12

    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + eps
    )
    return ssim_map.mean()


def main():
    parser = argparse.ArgumentParser(
        description="Per-image metrics (SSIM/LPIPS) for GEN vs GT pairs."
    )
    parser.add_argument("--root_dir", type=str, required=True,
                        help="root folder containing g1/g2/g3 subfolders")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_lpips", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    lpips_model = None
    if not args.no_lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net="alex").to(device)
        except Exception as exc:
            print(f"LPIPS not available: {exc}")

    all_pairs = []
    for subdir in sorted(glob(os.path.join(args.root_dir, "*"))):
        if not os.path.isdir(subdir):
            continue
        all_pairs.extend(_pair_images(subdir))

    if not all_pairs:
        print("No GEN/GT pairs found.")
        return

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gen_path", "gt_path", "ssim", "lpips"])

        for gen_path, gt_path in all_pairs:
            gen = _load_image(gen_path, args.img_size)
            gt = _load_image(gt_path, args.img_size)

            gen_b = gen.unsqueeze(0).to(device)
            gt_b = gt.unsqueeze(0).to(device)
            ssim_val = float(_ssim(gen_b, gt_b).item())

            lpips_val = ""
            if lpips_model is not None:
                with torch.no_grad():
                    g = gen_b * 2 - 1
                    t = gt_b * 2 - 1
                    lpips_val = float(lpips_model(g, t).item())

            writer.writerow([gen_path, gt_path, f"{ssim_val:.6f}", lpips_val])

    print(f"Saved {args.out_csv}")


if __name__ == "__main__":
    main()
