"""Diagnose whether an Exp43 generator uses its EEG conditioning.

For every selected VI test trial, generate three images from exactly the same
initial diffusion noise:

* correct: conditioning from the trial's EEG
* shuffled: conditioning from a test EEG belonging to the next class
* zero: all-zero SD conditioning tokens (after the EEG projector)

This is a diagnostic, not a replacement for the full-test evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import eval_exp43_full_test as full
import train_exp43_vi_lora as exp43
from train_vs_re_latent_gen import VAE_SCALE, make_schedule
from train_vs_re_lora_gen import DINO_EVAL_TF


CONDITIONS = ("correct", "shuffled", "zero")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="Exp43 C0 or C1 checkpoint")
    p.add_argument("--supcon_ckpt", required=True, help="directory with subjNN_best.pt")
    p.add_argument("--data_root", required=True)
    p.add_argument("--img_root", required=True)
    p.add_argument("--subject_id", type=int, required=True)
    p.add_argument("--condition", choices=("auto", "c0", "c1"), default="auto")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--n_eeg_tokens", type=int, default=16)
    p.add_argument("--per_class_total", type=int, default=0)
    p.add_argument("--samples_per_class", type=int, default=1)
    p.add_argument("--ddim_steps", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260711)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument(
        "--out_root",
        default="/content/drive/MyDrive/vsvi_data/conditioning_diagnostic",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


@torch.no_grad()
def sample_from_tokens(
    unet,
    cond_tokens: torch.Tensor,
    initial_noise: torch.Tensor,
    acp: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Project DDIM sampler with caller-controlled tokens and noise."""
    total_steps = 1000
    seq = list(range(0, total_steps, total_steps // steps))
    x = initial_noise.clone()
    for i in reversed(range(len(seq))):
        timestep = torch.full(
            (x.size(0),), seq[i], dtype=torch.long, device=x.device
        )
        noise_pred = unet(x, timestep, encoder_hidden_states=cond_tokens).sample
        alpha = acp[seq[i]]
        alpha_prev = (
            acp[seq[i - 1]]
            if i > 0
            else torch.tensor(1.0, device=x.device, dtype=acp.dtype)
        )
        x0 = (x - (1 - alpha).sqrt() * noise_pred) / alpha.sqrt()
        x0 = x0.clamp(-1, 1)
        x = alpha_prev.sqrt() * x0 + (1 - alpha_prev).sqrt() * noise_pred
    return x


def select_trials(dataset, samples_per_class: int):
    by_class: dict[int, list[int]] = {i: [] for i in range(full.N_CLASSES)}
    for index in range(len(dataset)):
        label = int(dataset[index][2])
        if label not in by_class:
            raise ValueError(f"unexpected zero-based class label: {label}")
        by_class[label].append(index)

    missing = [label for label, indices in by_class.items() if not indices]
    if missing:
        raise RuntimeError(f"test split is missing classes: {missing}")

    selected = []
    shuffled_source = {}
    for label in range(full.N_CLASSES):
        count = len(by_class[label]) if samples_per_class <= 0 else samples_per_class
        if count > len(by_class[label]):
            raise ValueError(
                f"class {label + 1} has {len(by_class[label])} trials, requested {count}"
            )
        next_label = (label + 1) % full.N_CLASSES
        for rank, index in enumerate(by_class[label][:count]):
            selected.append(index)
            shuffled_source[index] = by_class[next_label][rank % len(by_class[next_label])]
    return selected, shuffled_source


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.permute(1, 2, 0).detach().cpu().numpy() * 255
    ).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def aggregate(rows: list[dict], features: dict[str, list[torch.Tensor]]) -> dict:
    result = {}
    correct_predictions = {
        int(row["test_index"]): int(row["pred_label"])
        for row in rows
        if row["conditioning"] == "correct"
    }
    correct_features = features["correct"]

    for condition in CONDITIONS:
        selected = [row for row in rows if row["conditioning"] == condition]
        recalls = []
        for label in range(1, full.N_CLASSES + 1):
            class_rows = [row for row in selected if int(row["true_label"]) == label]
            recalls.append(np.mean([int(row["correct"]) for row in class_rows]))
        predictions = [int(row["pred_label"]) for row in selected]
        counts = np.array(
            [predictions.count(label) for label in range(1, full.N_CLASSES + 1)],
            dtype=float,
        )
        probabilities = counts / counts.sum()
        entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
        dominant_label, dominant_count = Counter(predictions).most_common(1)[0]
        condition_result = {
            "n": len(selected),
            "top1": float(np.mean([int(row["correct"]) for row in selected])),
            "top3": float(np.mean([int(row["top3_correct"]) for row in selected])),
            "top5": float(np.mean([int(row["top5_correct"]) for row in selected])),
            "balanced_accuracy": float(np.mean(recalls)),
            "normalized_entropy": entropy / math.log(full.N_CLASSES),
            "dominant_label": dominant_label,
            "dominant_name": full.CLASS_NAMES[dominant_label],
            "dominant_ratio": dominant_count / len(selected),
            "prediction_counts": {
                full.CLASS_NAMES[label]: int(counts[label - 1])
                for label in range(1, full.N_CLASSES + 1)
            },
        }
        if condition != "correct":
            condition_result["prediction_change_rate_vs_correct"] = float(
                np.mean(
                    [
                        int(int(row["pred_label"]) != correct_predictions[int(row["test_index"])])
                        for row in selected
                    ]
                )
            )
            cosine = [
                float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
                for a, b in zip(correct_features, features[condition])
            ]
            condition_result["mean_dino_cosine_vs_correct"] = float(np.mean(cosine))
            condition_result["mean_dino_distance_vs_correct"] = float(
                np.mean([1.0 - value for value in cosine])
            )
        result[condition] = condition_result
    return result


def save_grid(path: Path, selected_indices: list[int], rows: list[dict], images):
    lookup = {
        (int(row["test_index"]), row["conditioning"]): row for row in rows
    }
    n_rows = len(selected_indices)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, max(3, n_rows * 3)))
    axes = np.asarray(axes).reshape(n_rows, 3)
    for row_index, test_index in enumerate(selected_indices):
        for column, condition in enumerate(CONDITIONS):
            row = lookup[(test_index, condition)]
            ax = axes[row_index, column]
            ax.imshow(images[(test_index, condition)])
            ax.axis("off")
            if row_index == 0:
                ax.set_title(condition)
            if column == 0:
                ax.set_ylabel(
                    f'{row["true_name"]}\ntrial {test_index}', rotation=0, labelpad=45
                )
            ax.text(
                0.02,
                0.02,
                f'pred={row["pred_name"]}',
                transform=ax.transAxes,
                fontsize=8,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 2},
            )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.ddim_steps <= 0 or 1000 // args.ddim_steps <= 0:
        raise ValueError("--ddim_steps must be between 1 and 1000")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    condition_name = full.resolve_condition(args.ckpt, args.condition)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.out_root) / f"S{args.subject_id:02d}" / condition_name
    metrics_path = run_dir / "conditioning_metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite to rerun")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] device={device} S{args.subject_id:02d} {condition_name}")
    print("[INFO] Loading DINO...", flush=True)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dino = dino.to(device).eval()
    for parameter in dino.parameters():
        parameter.requires_grad_(False)
    prototypes = full.build_dino_prototypes(args.img_root, dino, device)

    print("[INFO] Loading SD VAE...", flush=True)
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae"
    ).to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    encoder, encoder_path = full.load_encoder(args, device)
    unet, cond_proj, checkpoint_path = full.load_generator(args, device)
    _, _, acp = make_schedule(1000, device)

    per_class_total = args.per_class_total if args.per_class_total > 0 else None
    dataset = exp43.SubjectClassLimitedDataset(
        data_root=args.data_root,
        sid=args.subject_id,
        split="test",
        seed=args.split_seed,
        per_class_total=per_class_total,
    )
    selected_indices, shuffled_source = select_trials(dataset, args.samples_per_class)
    print(
        f"[INFO] selected={len(selected_indices)} "
        f"({args.samples_per_class} per class)",
        flush=True,
    )

    subject_index = torch.zeros(1, dtype=torch.long, device=device)
    rows: list[dict] = []
    images = {}
    features: dict[str, list[torch.Tensor]] = {name: [] for name in CONDITIONS}

    for start in range(0, len(selected_indices), args.batch_size):
        batch_indices = selected_indices[start : start + args.batch_size]
        correct_batch = [dataset[index] for index in batch_indices]
        shuffled_indices = [shuffled_source[index] for index in batch_indices]
        shuffled_batch = [dataset[index] for index in shuffled_indices]

        eeg = torch.stack([item[0] for item in correct_batch]).to(device)
        shuffled_eeg = torch.stack([item[0] for item in shuffled_batch]).to(device)
        true_zero = torch.tensor(
            [int(item[2]) for item in correct_batch], device=device
        )
        shuffled_zero = torch.tensor(
            [int(item[2]) for item in shuffled_batch], device=device
        )
        if torch.any(true_zero == shuffled_zero):
            raise RuntimeError("shuffled conditioning contains a same-class trial")

        subject_ids = subject_index.expand(eeg.size(0))
        correct_latent = encoder.encode_eeg(eeg, subject_ids)
        shuffled_latent = encoder.encode_eeg(shuffled_eeg, subject_ids)
        correct_tokens = cond_proj(correct_latent)
        shuffled_tokens = cond_proj(shuffled_latent)
        tokens = {
            "correct": correct_tokens,
            "shuffled": shuffled_tokens,
            "zero": torch.zeros_like(correct_tokens),
        }
        seeds = [args.seed + index for index in batch_indices]
        initial_noise = full.deterministic_noise(seeds, device)

        for condition in CONDITIONS:
            generated_latent = sample_from_tokens(
                unet, tokens[condition], initial_noise, acp, args.ddim_steps
            )
            decoded = vae.decode(generated_latent / VAE_SCALE).sample.clamp(-1, 1)
            decoded = (decoded + 1) / 2
            dino_input = torch.stack([DINO_EVAL_TF(image) for image in decoded])
            batch_features = F.normalize(dino(dino_input), dim=-1)
            similarities = batch_features @ prototypes.T
            top5 = similarities.topk(5, dim=1).indices

            for offset, test_index in enumerate(batch_indices):
                true_index = int(true_zero[offset].item())
                pred_index = int(top5[offset, 0].item())
                source_index = (
                    true_index
                    if condition == "correct"
                    else int(shuffled_zero[offset].item())
                    if condition == "shuffled"
                    else -1
                )
                pil = tensor_to_pil(decoded[offset])
                filename = f"test{test_index:04d}_{condition}_seed{seeds[offset]}.png"
                pil.save(run_dir / filename)
                images[(test_index, condition)] = pil
                features[condition].append(batch_features[offset].detach().cpu())
                rows.append(
                    {
                        "subject": args.subject_id,
                        "model_condition": condition_name,
                        "test_index": test_index,
                        "conditioning": condition,
                        "conditioning_source_label": source_index + 1 if source_index >= 0 else 0,
                        "true_label": true_index + 1,
                        "true_name": full.CLASS_NAMES[true_index + 1],
                        "seed": seeds[offset],
                        "pred_label": pred_index + 1,
                        "pred_name": full.CLASS_NAMES[pred_index + 1],
                        "correct": int(pred_index == true_index),
                        "top3_correct": int(true_index in top5[offset, :3].tolist()),
                        "top5_correct": int(true_index in top5[offset, :5].tolist()),
                    }
                )
        print(f"[Generate] {min(start + args.batch_size, len(selected_indices))}/{len(selected_indices)}", flush=True)

    metrics = {
        "subject": args.subject_id,
        "model_condition": condition_name,
        "samples_per_class": args.samples_per_class,
        "n_trials": len(selected_indices),
        "seed": args.seed,
        "split_seed": args.split_seed,
        "ddim_steps": args.ddim_steps,
        "checkpoint": str(checkpoint_path),
        "supcon_checkpoint": str(encoder_path),
        "conditions": aggregate(rows, features),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with (run_dir / "conditioning_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_grid(
        run_dir / f"S{args.subject_id:02d}_{condition_name}_conditioning_grid.png",
        selected_indices,
        rows,
        images,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"[Saved] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
