"""Generate and evaluate every Exp43 test trial.

The script is designed for Colab and writes all durable outputs to Drive:

* one PNG for every test trial
* a manifest CSV with deterministic seed, prediction, and similarities
* aggregate DINO Top-k/collapse metrics
* per-class recall and a confusion matrix
* a full qualitative grid grouped by the true class

C0 and C1 are paired fairly when invoked with the same ``--seed`` and
``--split_seed``: test order and initial diffusion noise are identical.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import train_exp43_vi_lora as exp43
from train_vs_re_latent_gen import VAE_SCALE, build_eeg_encoder, make_schedule
from train_vs_re_lora_gen import (
    CLS_LIST,
    DINO_EVAL_TF,
    EEGConditionProjector,
    load_sd15_unet_lora,
)


CLASS_NAMES = {
    1: "airplane",
    2: "cup",
    3: "tree",
    4: "digit1",
    5: "digit3",
    6: "digit5",
    7: "heart",
    8: "star",
    9: "triangle",
}
N_CLASSES = len(CLASS_NAMES)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="Exp43 C0 or C1 best checkpoint")
    p.add_argument(
        "--supcon_ckpt",
        required=True,
        help="directory containing subjNN_best.pt (not the checkpoint file)",
    )
    p.add_argument("--data_root", default="/content/vsvi_project/preproc_vi_re")
    p.add_argument("--img_root", default="/content/vsvi_project/preproc_data_vi/images")
    p.add_argument("--subject_id", type=int, required=True)
    p.add_argument("--condition", choices=("auto", "c0", "c1"), default="auto")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--n_eeg_tokens", type=int, default=16)
    p.add_argument(
        "--per_class_total",
        type=int,
        default=0,
        help="same cap used for training; 0 uses all available trials",
    )
    p.add_argument("--ddim_steps", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260711, help="diffusion noise seed")
    p.add_argument("--split_seed", type=int, default=42, help="dataset split seed")
    p.add_argument(
        "--out_root",
        default="/content/drive/MyDrive/vsvi_data/gen_images_full",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve_condition(path: str, requested: str) -> str:
    if requested != "auto":
        return requested
    name = path.lower()
    if "_c0_" in name or "exp43_c0" in name:
        return "c0"
    if "_c1_" in name or "exp43_c1" in name:
        return "c1"
    raise ValueError("Cannot infer C0/C1 from checkpoint path; pass --condition")


def require_file(path: str | Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


@torch.no_grad()
def build_dino_prototypes(img_root: str, dino, device: torch.device) -> torch.Tensor:
    to_tensor = T.ToTensor()
    protos = []
    for cls_id in CLS_LIST:
        path = require_file(Path(img_root) / f"{cls_id:02d}.png", "class image")
        image = Image.open(path).convert("RGB")
        tensor = DINO_EVAL_TF(to_tensor(image)).unsqueeze(0).to(device)
        protos.append(F.normalize(dino(tensor), dim=-1))
    return torch.cat(protos, dim=0)


def load_encoder(args: argparse.Namespace, device: torch.device):
    encoder = build_eeg_encoder(
        32,
        384,
        SimpleNamespace(eeg_occipital_ids="auto"),
        device,
    )
    path = require_file(
        Path(args.supcon_ckpt) / f"subj{args.subject_id:02d}_best.pt",
        "SupCon checkpoint",
    )
    payload = torch.load(path, map_location=device, weights_only=False)
    key = "eeg_enc" if "eeg_enc" in payload else "model"
    encoder.load_state_dict(payload[key], strict=True)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    print(f"[Encoder] {path} (key={key})", flush=True)
    return encoder, path


def load_generator(args: argparse.Namespace, device: torch.device):
    checkpoint_path = require_file(args.ckpt, "Exp43 checkpoint")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "unet_lora" not in payload or "cond_proj" not in payload:
        raise KeyError("checkpoint must contain unet_lora and cond_proj")

    unet = load_sd15_unet_lora(args.lora_r, args.lora_alpha).to(device)
    source = payload["unet_lora"]
    loaded = 0
    shape_errors = []
    with torch.no_grad():
        for name, param in unet.named_parameters():
            if name not in source:
                continue
            value = source[name]
            if tuple(value.shape) != tuple(param.shape):
                shape_errors.append((name, tuple(value.shape), tuple(param.shape)))
                continue
            param.copy_(value.to(device=device, dtype=param.dtype))
            loaded += 1
    if shape_errors:
        raise ValueError(f"LoRA tensor shape mismatch: {shape_errors[:3]}")
    if loaded == 0:
        raise RuntimeError("No LoRA tensors matched the current UNet; check --lora_r")
    unet.eval()

    cond_proj = EEGConditionProjector(
        eeg_dim=512,
        sd_dim=768,
        n_tokens=args.n_eeg_tokens,
        deep=False,
    ).to(device)
    cond_proj.load_state_dict(payload["cond_proj"], strict=True)
    cond_proj.eval()
    print(f"[UNet+LoRA] {checkpoint_path} tensors={loaded}", flush=True)
    return unet, cond_proj, checkpoint_path


def deterministic_noise(seeds: list[int], device: torch.device) -> torch.Tensor:
    tensors = []
    for seed in seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        tensors.append(torch.randn((4, 64, 64), generator=generator))
    return torch.stack(tensors).to(device)


@torch.no_grad()
def sample_from_noise(
    unet,
    cond_proj,
    eeg_lat: torch.Tensor,
    initial_noise: torch.Tensor,
    acp: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """The project DDIM sampler with caller-controlled initial noise."""
    total_steps = 1000
    cond_tokens = cond_proj(eeg_lat)
    seq = list(range(0, total_steps, total_steps // steps))
    x = initial_noise
    for i in reversed(range(len(seq))):
        t_value = seq[i]
        timestep = torch.full(
            (x.size(0),), t_value, dtype=torch.long, device=x.device
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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_grid(
    output_path: Path,
    records: list[dict],
    images: dict[int, Image.Image],
    subject_id: int,
    condition: str,
) -> None:
    grouped = {label: [] for label in range(1, N_CLASSES + 1)}
    for record in records:
        grouped[record["true_label"]].append(record)
    rows = max((len(items) for items in grouped.values()), default=1)
    fig, axes = plt.subplots(
        rows,
        N_CLASSES,
        figsize=(N_CLASSES * 1.6, max(rows, 1) * 1.65),
        squeeze=False,
    )
    for col, label in enumerate(range(1, N_CLASSES + 1)):
        items = grouped[label]
        for row in range(rows):
            axis = axes[row, col]
            axis.axis("off")
            if row < len(items):
                record = items[row]
                axis.imshow(images[record["test_index"]])
                color = "green" if record["correct"] else "red"
                axis.set_title(
                    f"{CLASS_NAMES[label]}\npred={record['pred_name']}",
                    fontsize=7,
                    color=color,
                )
            elif row == 0:
                axis.set_title(CLASS_NAMES[label], fontsize=8)
    fig.suptitle(f"S{subject_id:02d} Exp43 {condition} — full test", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.ddim_steps <= 0 or 1000 // args.ddim_steps <= 0:
        raise ValueError("--ddim_steps must be between 1 and 1000")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    condition = resolve_condition(args.ckpt, args.condition)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device} subject=S{args.subject_id:02d} condition={condition}")

    run_dir = Path(args.out_root) / f"S{args.subject_id:02d}" / condition
    image_root = run_dir / "images"
    manifest_path = run_dir / "manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{manifest_path} already exists; use --overwrite to replace this run"
        )
    image_root.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading DINO...", flush=True)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dino = dino.to(device).eval()
    for param in dino.parameters():
        param.requires_grad_(False)
    prototypes = build_dino_prototypes(args.img_root, dino, device)

    print("[INFO] Loading SD VAE...", flush=True)
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae"
    ).to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    encoder, encoder_path = load_encoder(args, device)
    unet, cond_proj, checkpoint_path = load_generator(args, device)
    _, _, acp = make_schedule(1000, device)

    per_class_total = args.per_class_total if args.per_class_total > 0 else None
    dataset = exp43.SubjectClassLimitedDataset(
        data_root=args.data_root,
        sid=args.subject_id,
        split="test",
        seed=args.split_seed,
        per_class_total=per_class_total,
    )
    if len(dataset) == 0:
        raise RuntimeError("test dataset is empty")
    print(
        f"[INFO] full test n={len(dataset)} per_class_total={per_class_total}",
        flush=True,
    )

    records: list[dict] = []
    grid_images: dict[int, Image.Image] = {}
    subject_index = torch.zeros(1, dtype=torch.long, device=device)

    for start in range(0, len(dataset), args.batch_size):
        stop = min(start + args.batch_size, len(dataset))
        batch = [dataset[index] for index in range(start, stop)]
        eeg = torch.stack([item[0] for item in batch]).to(device)
        true_zero = torch.tensor([int(item[2]) for item in batch], device=device)
        test_indices = list(range(start, stop))
        seeds = [args.seed + index for index in test_indices]

        eeg_lat = encoder.encode_eeg(eeg, subject_index.expand(eeg.size(0)))
        initial_noise = deterministic_noise(seeds, device)
        generated_latent = sample_from_noise(
            unet,
            cond_proj,
            eeg_lat,
            initial_noise,
            acp,
            args.ddim_steps,
        )
        decoded = vae.decode(generated_latent / VAE_SCALE).sample.clamp(-1, 1)
        decoded = (decoded + 1) / 2
        dino_input = torch.stack([DINO_EVAL_TF(image) for image in decoded])
        features = F.normalize(dino(dino_input), dim=-1)
        similarities = features @ prototypes.T
        top5 = similarities.topk(5, dim=1).indices

        for offset, test_index in enumerate(test_indices):
            true_index = int(true_zero[offset].item())
            pred_index = int(top5[offset, 0].item())
            true_similarity = float(similarities[offset, true_index].item())
            pred_similarity = float(similarities[offset, pred_index].item())
            other = torch.cat(
                [similarities[offset, :true_index], similarities[offset, true_index + 1 :]]
            )
            true_margin = true_similarity - float(other.max().item())

            true_label = true_index + 1
            pred_label = pred_index + 1
            class_dir = image_root / f"class{true_label:02d}_{CLASS_NAMES[true_label]}"
            class_dir.mkdir(parents=True, exist_ok=True)
            filename = f"test{test_index:04d}_seed{seeds[offset]}.png"
            image_path = class_dir / filename

            array = (
                decoded[offset].permute(1, 2, 0).detach().cpu().numpy() * 255
            ).round().clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(array)
            pil_image.save(image_path)
            grid_images[test_index] = pil_image.copy()

            record = {
                "subject": args.subject_id,
                "condition": condition,
                "test_index": test_index,
                "true_label": true_label,
                "true_name": CLASS_NAMES[true_label],
                "seed": seeds[offset],
                "image_path": str(image_path),
                "pred_label": pred_label,
                "pred_name": CLASS_NAMES[pred_label],
                "correct": int(pred_index == true_index),
                "top3_correct": int(true_index in top5[offset, :3].tolist()),
                "top5_correct": int(true_index in top5[offset, :5].tolist()),
                "true_similarity": true_similarity,
                "pred_similarity": pred_similarity,
                "true_margin": true_margin,
            }
            records.append(record)

        print(f"[Generate] {stop}/{len(dataset)}", flush=True)

    fields = list(records[0].keys())
    write_csv(manifest_path, records, fields)

    predictions = [int(row["pred_label"]) for row in records]
    counts = np.array(
        [predictions.count(label) for label in range(1, N_CLASSES + 1)], dtype=float
    )
    probabilities = counts / counts.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
    dominant_label, dominant_count = Counter(predictions).most_common(1)[0]

    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    per_class_rows = []
    for row in records:
        confusion[int(row["true_label"]) - 1, int(row["pred_label"]) - 1] += 1
    for label in range(1, N_CLASSES + 1):
        selected = [row for row in records if int(row["true_label"]) == label]
        correct = sum(int(row["correct"]) for row in selected)
        per_class_rows.append(
            {
                "label": label,
                "class_name": CLASS_NAMES[label],
                "support": len(selected),
                "correct": correct,
                "recall": correct / max(len(selected), 1),
            }
        )

    metrics = {
        "subject": args.subject_id,
        "condition": condition,
        "n_test": len(records),
        "top1": float(np.mean([row["correct"] for row in records])),
        "top3": float(np.mean([row["top3_correct"] for row in records])),
        "top5": float(np.mean([row["top5_correct"] for row in records])),
        "balanced_accuracy": float(np.mean([row["recall"] for row in per_class_rows])),
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(N_CLASSES),
        "dominant_label": dominant_label,
        "dominant_name": CLASS_NAMES[dominant_label],
        "dominant_ratio": dominant_count / len(records),
        "mean_true_similarity": float(
            np.mean([row["true_similarity"] for row in records])
        ),
        "mean_true_margin": float(np.mean([row["true_margin"] for row in records])),
        "seed": args.seed,
        "split_seed": args.split_seed,
        "per_class_total": args.per_class_total,
        "ddim_steps": args.ddim_steps,
        "checkpoint": str(checkpoint_path),
        "supcon_checkpoint": str(encoder_path),
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    write_csv(run_dir / "per_class_metrics.csv", per_class_rows, list(per_class_rows[0]))

    confusion_rows = []
    for true_index in range(N_CLASSES):
        row = {"true_class": CLASS_NAMES[true_index + 1]}
        for pred_index in range(N_CLASSES):
            row[f"pred_{CLASS_NAMES[pred_index + 1]}"] = int(
                confusion[true_index, pred_index]
            )
        confusion_rows.append(row)
    write_csv(run_dir / "confusion_matrix.csv", confusion_rows, list(confusion_rows[0]))
    save_grid(
        run_dir / f"S{args.subject_id:02d}_exp43_{condition}_full_grid.png",
        records,
        grid_images,
        args.subject_id,
        condition,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"[Saved] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
