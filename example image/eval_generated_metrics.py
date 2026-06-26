# eval_generated_metrics.py
import argparse
import csv
import os
import re
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


def _extract_subject_group(samples_dir):
    base = os.path.basename(os.path.normpath(samples_dir))
    m = re.search(r"_subj(?P<subj>\\d+)_g(?P<grp>\\d+)", base)
    if not m:
        return "", ""
    return m.group("subj"), m.group("grp")


def _load_image(path, img_size):
    img = Image.open(path).convert("RGB")
    if img_size is not None:
        img = img.resize((img_size, img_size), Image.BILINEAR)
    img = transforms.ToTensor()(img)  # [0,1]
    return img


def _ssim(img1, img2, data_range=1.0, k1=0.01, k2=0.03, win_size=11, eps=1e-8):
    # Simple SSIM over RGB with Gaussian window
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


def _gaussian_window(win_size, channel, device, sigma=1.5):
    coords = torch.arange(win_size, device=device).float() - win_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]
    window = window.expand(channel, 1, win_size, win_size).contiguous()
    return window


def _compute_fid_stats(model, images, device, batch_size=16):
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i:i + batch_size]).to(device)
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
            feat = model(batch).squeeze(-1).squeeze(-1)
            feats.append(feat.cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    from scipy import linalg

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated images with SSIM/LPIPS/FID."
    )
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=None,
                        help="Resize for metrics; None keeps original size.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_lpips", action="store_true", default=False)
    parser.add_argument("--no_fid", action="store_true", default=False)
    parser.add_argument("--print_per_image", action="store_true", default=False,
                        help="Print per-image SSIM/LPIPS to stdout.")
    parser.add_argument("--out_csv", type=str, default=None,
                        help="optional csv output path")
    args = parser.parse_args()

    pairs = _pair_images(args.samples_dir)
    if not pairs:
        print("No *_GEN.png/*_GT.png pairs found.")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    gen_imgs = []
    gt_imgs = []
    ssim_scores = []
    lpips_scores = []
    lpips_available = False

    for gen_path, gt_path in pairs:
        gen = _load_image(gen_path, args.img_size)
        gt = _load_image(gt_path, args.img_size)
        gen_imgs.append(gen)
        gt_imgs.append(gt)

        gen_b = gen.unsqueeze(0).to(device)
        gt_b = gt.unsqueeze(0).to(device)
        ssim_scores.append(float(_ssim(gen_b, gt_b).item()))

    print(f"SSIM (mean): {np.mean(ssim_scores):.4f}")

    if not args.no_lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net="alex").to(device)
            lpips_available = True
            with torch.no_grad():
                for gen, gt in zip(gen_imgs, gt_imgs):
                    g = gen.unsqueeze(0).to(device) * 2 - 1
                    t = gt.unsqueeze(0).to(device) * 2 - 1
                    lp = lpips_model(g, t).item()
                    lpips_scores.append(lp)
            print(f"LPIPS (mean): {np.mean(lpips_scores):.4f}")
        except Exception as exc:
            print(f"LPIPS not available: {exc}")

    fid_value = None
    if not args.no_fid:
        try:
            from torchvision.models import inception_v3
            inception = inception_v3(pretrained=True, transform_input=False)
            inception.fc = torch.nn.Identity()
            inception.eval().to(device)

            mu_g, sigma_g = _compute_fid_stats(inception, gen_imgs, device, args.batch_size)
            mu_t, sigma_t = _compute_fid_stats(inception, gt_imgs, device, args.batch_size)
            fid_value = _frechet_distance(mu_g, sigma_g, mu_t, sigma_t)
            print(f"FID: {fid_value:.4f}")
        except Exception as exc:
            print(f"FID not available: {exc}")

    if args.out_csv is None:
        args.out_csv = os.path.join(args.samples_dir, "metrics.csv")

    subj_id, group_id = _extract_subject_group(args.samples_dir)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "group_id", "gen_path", "gt_path", "ssim", "lpips"])
        for i, (gen_path, gt_path) in enumerate(pairs):
            lp = lpips_scores[i] if lpips_available else ""
            writer.writerow([subj_id, group_id, gen_path, gt_path, f"{ssim_scores[i]:.6f}", lp])

        writer.writerow([])
        writer.writerow(["metric", "value"])
        writer.writerow(["ssim_mean", f"{np.mean(ssim_scores):.6f}"])
        if lpips_available:
            writer.writerow(["lpips_mean", f"{np.mean(lpips_scores):.6f}"])
        if fid_value is not None:
            writer.writerow(["fid", f"{fid_value:.6f}"])
    print(f"Saved metrics CSV: {args.out_csv}")

    if args.print_per_image:
        for i, (gen_path, gt_path) in enumerate(pairs):
            lp = lpips_scores[i] if lpips_available else None
            lp_str = f"{lp:.6f}" if lp is not None else "N/A"
            print(f"[{i:04d}] SSIM {ssim_scores[i]:.6f} | LPIPS {lp_str} | GEN {gen_path} | GT {gt_path}")


if __name__ == "__main__":
    main()
