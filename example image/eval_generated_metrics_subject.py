# eval_generated_metrics_subject.py
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


def _extract_subject_group(samples_dir):
    base = os.path.basename(os.path.normpath(samples_dir))
    subj = ""
    grp = ""
    for token in base.split("_"):
        if token.startswith("subj"):
            subj = token.replace("subj", "")
        if token.startswith("g"):
            grp = token.replace("g", "")
    return subj, grp


def _extract_label(gen_path):
    name = os.path.basename(gen_path)
    parts = name.lower().split("_")
    for part in parts:
        if part.startswith("label"):
            return part.replace("label", "").replace("gen.png", "")
    return ""


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

    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)


def main():
    parser = argparse.ArgumentParser(
        description="Subject-level metrics by label for GEN/GT pairs."
    )
    parser.add_argument("--root_dir", type=str, required=True,
                        help="root folder containing subject subfolders")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_lpips", action="store_true", default=False)
    parser.add_argument("--no_fid", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    lpips_model = None
    if not args.no_lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net="alex").to(device)
        except Exception as exc:
            print(f"LPIPS not available: {exc}")

    inception = None
    if not args.no_fid:
        try:
            from torchvision.models import inception_v3
            inception = inception_v3(pretrained=True, transform_input=False)
            inception.fc = torch.nn.Identity()
            inception.eval().to(device)
        except Exception as exc:
            print(f"FID not available: {exc}")

    os.makedirs(args.out_dir, exist_ok=True)

    for subdir in sorted(glob(os.path.join(args.root_dir, "*"))):
        if not os.path.isdir(subdir):
            continue

        subj_id, group_id = _extract_subject_group(subdir)
        if not subj_id:
            continue

        pairs = _pair_images(subdir)
        if not pairs:
            continue

        by_label = {}
        for gen_path, gt_path in pairs:
            label = _extract_label(gen_path)
            if not label:
                continue
            by_label.setdefault(label, {"gen": [], "gt": [], "ssim": [], "lpips": []})

            gen = _load_image(gen_path, args.img_size)
            gt = _load_image(gt_path, args.img_size)
            by_label[label]["gen"].append(gen)
            by_label[label]["gt"].append(gt)

            gen_b = gen.unsqueeze(0).to(device)
            gt_b = gt.unsqueeze(0).to(device)
            by_label[label]["ssim"].append(float(_ssim(gen_b, gt_b).item()))

            if lpips_model is not None:
                with torch.no_grad():
                    g = gen_b * 2 - 1
                    t = gt_b * 2 - 1
                    lp = lpips_model(g, t).item()
                    by_label[label]["lpips"].append(lp)

        group_suffix = f"_g{group_id}" if group_id else ""
        out_csv = os.path.join(args.out_dir, f"metrics_subj{subj_id}{group_suffix}.csv")
        labels_sorted = sorted(by_label.keys(), key=lambda x: int(x))

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            header = ["subject_id", "group_id"]
            for label in labels_sorted:
                header += [f"label{label}_ssim", f"label{label}_lpips", f"label{label}_fid"]
            writer.writerow(header)

            row = [subj_id, group_id]
            for label in labels_sorted:
                ssim_mean = np.mean(by_label[label]["ssim"]) if by_label[label]["ssim"] else ""
                lpips_mean = np.mean(by_label[label]["lpips"]) if by_label[label]["lpips"] else ""

                fid_val = ""
                if inception is not None and by_label[label]["gen"] and by_label[label]["gt"]:
                    mu_g, sigma_g = _compute_fid_stats(
                        inception, by_label[label]["gen"], device, args.batch_size
                    )
                    mu_t, sigma_t = _compute_fid_stats(
                        inception, by_label[label]["gt"], device, args.batch_size
                    )
                    fid_val = _frechet_distance(mu_g, sigma_g, mu_t, sigma_t)

                row += [
                    f"{ssim_mean:.6f}" if ssim_mean != "" else "",
                    f"{lpips_mean:.6f}" if lpips_mean != "" else "",
                    f"{fid_val:.6f}" if fid_val != "" else "",
                ]

            writer.writerow(row)

        print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
