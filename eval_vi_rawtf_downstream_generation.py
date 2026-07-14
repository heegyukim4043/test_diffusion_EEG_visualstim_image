"""Downstream image-generation utility of three raw+TF VI encoders.

The SD1.5 UNet and its existing LoRA weights are frozen.  For each pre-specified
encoder (VI-only, full VS-to-VI, gated residual), an identically initialized
256-to-SD-token bridge is trained on the same VI train split and selected by
deterministic VI validation diffusion loss.  Correct, shuffled, and zero
conditioning are then generated with identical test trials and initial noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from dataset_vs_re import collate_fn
from train_vi_rawtf_gated_residual import GatedResidualTransfer
from train_vi_tf_representation_ablation import (
    EEGWaveAugment,
    RepresentationEncoder,
    build_loaders,
    safe_torch_load,
    set_seed,
)
from train_vs_re_latent_gen import VAE_SCALE, make_schedule
from train_vs_re_lora_gen import (
    CLS_LIST,
    DINO_EVAL_TF,
    EEGConditionProjector,
    encode_class_images_512,
    load_sd15_unet_lora,
)


ENCODERS = ("vi_only", "full_vs_to_vi", "gated_residual")
CONDITIONS = ("correct", "shuffled", "zero")
CLASS_NAMES = (
    "airplane", "cup", "tree", "digit1", "digit3", "digit5",
    "heart", "star", "triangle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", type=int, default=24)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--img_root", required=True)
    parser.add_argument("--vi_only_ckpt", required=True)
    parser.add_argument("--full_vs_to_vi_ckpt", required=True)
    parser.add_argument("--gated_ckpt", required=True)
    parser.add_argument("--generator_ckpt", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--n_eeg_tokens", type=int, default=16)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--ddim_steps", type=int, default=30)
    parser.add_argument("--samples_per_class", type=int, default=2)
    parser.add_argument("--generations_per_trial", type=int, default=1)
    parser.add_argument("--sampling_rate", type=float, default=1024.0)
    parser.add_argument("--n_ch", type=int, default=32)
    parser.add_argument("--n_fft", type=int, default=256)
    parser.add_argument("--hop_length", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--audit_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.representation = "raw_tf"
    args.num_workers = 0
    args.no_aug = False
    return args


def require(path: str | Path, label: str) -> Path:
    result = Path(path)
    if not result.is_file():
        raise FileNotFoundError(f"{label} not found: {result}")
    return result


def audit(args: argparse.Namespace) -> dict:
    checks = {
        "VI data": Path(args.data_root).is_dir(),
        "class images": all((Path(args.img_root) / f"{i:02d}.png").is_file() for i in CLS_LIST),
        "VI-only checkpoint": Path(args.vi_only_ckpt).is_file(),
        "full VS-to-VI checkpoint": Path(args.full_vs_to_vi_ckpt).is_file(),
        "gated checkpoint": Path(args.gated_ckpt).is_file(),
        "generator checkpoint": Path(args.generator_ckpt).is_file(),
    }
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    if not all(checks.values()):
        raise RuntimeError("Audit failed")
    return checks


def rawtf_model(args, checkpoint_path: Path, device: torch.device):
    model = RepresentationEncoder(args)
    payload = safe_torch_load(checkpoint_path, "cpu")
    if payload.get("config", {}).get("representation") != "raw_tf":
        raise RuntimeError(f"Not a raw_tf checkpoint: {checkpoint_path}")
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def gated_model(args, checkpoint_path: Path, device: torch.device):
    payload = safe_torch_load(checkpoint_path, "cpu")
    student = RepresentationEncoder(args)
    teacher = RepresentationEncoder(args)
    model = GatedResidualTransfer(student, teacher, args.latent_dim)
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def encode(model, name: str, eeg: torch.Tensor) -> torch.Tensor:
    if name == "gated_residual":
        return model.forward_features(eeg)[0]
    return model.encode(eeg)


def load_encoders(args, device):
    return {
        "vi_only": rawtf_model(args, require(args.vi_only_ckpt, "VI-only checkpoint"), device),
        "full_vs_to_vi": rawtf_model(
            args, require(args.full_vs_to_vi_ckpt, "full VS-to-VI checkpoint"), device
        ),
        "gated_residual": gated_model(
            args, require(args.gated_ckpt, "gated checkpoint"), device
        ),
    }


def load_frozen_unet(args, device):
    payload = torch.load(
        require(args.generator_ckpt, "generator checkpoint"),
        map_location="cpu",
        weights_only=False,
    )
    if "unet_lora" not in payload:
        raise KeyError("generator checkpoint lacks unet_lora")
    unet = load_sd15_unet_lora(args.lora_r, args.lora_alpha).to(device)
    source = payload["unet_lora"]
    loaded = 0
    with torch.no_grad():
        for name, parameter in unet.named_parameters():
            if name in source:
                value = source[name]
                if tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(f"LoRA shape mismatch for {name}")
                parameter.copy_(value.to(device=device, dtype=parameter.dtype))
                loaded += 1
    if loaded == 0:
        raise RuntimeError("No LoRA tensor matched; check lora_r/lora_alpha")
    for parameter in unet.parameters():
        parameter.requires_grad_(False)
    unet.eval()
    print(f"[Frozen UNet+LoRA] tensors={loaded}", flush=True)
    return unet


def q_sample(x0, t, acp, noise):
    signal = acp[t][:, None, None, None].sqrt()
    residual = (1 - acp[t])[:, None, None, None].sqrt()
    return signal * x0 + residual * noise


@torch.no_grad()
def deterministic_validation_loss(
    encoder_name, encoder, projector, unet, loader, class_latents, acp, args, device
):
    projector.eval()
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 99173)
    losses = []
    for eeg, _, labels in loader:
        eeg = eeg.to(device)
        labels = labels.to(device)
        latent = encode(encoder, encoder_name, eeg)
        tokens = projector(latent)
        x0 = class_latents[labels]
        t = torch.randint(
            0, args.num_timesteps, (eeg.size(0),), generator=generator
        ).to(device)
        noise = torch.randn(x0.shape, generator=generator).to(device=device, dtype=x0.dtype)
        xt = q_sample(x0, t, acp, noise)
        prediction = unet(xt, t, encoder_hidden_states=tokens).sample
        losses.append(F.mse_loss(prediction.float(), noise.float(), reduction="sum").item())
    denominator = len(loader.dataset) * int(np.prod(class_latents.shape[1:]))
    return sum(losses) / max(denominator, 1)


def train_bridge(
    name, encoder, unet, class_latents, acp, args, device, output: Path
):
    set_seed(args.seed)
    _, loaders, _ = build_loaders(args.data_root, args.subject_id, args, "vi")
    projector = EEGConditionProjector(
        eeg_dim=args.latent_dim,
        sd_dim=768,
        n_tokens=args.n_eeg_tokens,
        deep=False,
    ).to(device)
    optimizer = torch.optim.AdamW(
        projector.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")
    augmenter = EEGWaveAugment()
    best_loss = float("inf")
    best_epoch = 0
    history = []
    path = output / name / "bridge_best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        projector.train()
        total = 0.0
        seen = 0
        for eeg, _, labels in loaders["train"]:
            eeg = augmenter(eeg.to(device))
            labels = labels.to(device)
            with torch.no_grad():
                latent = encode(encoder, name, eeg)
                x0 = class_latents[labels]
                t = torch.randint(0, args.num_timesteps, (eeg.size(0),), device=device)
                noise = torch.randn_like(x0)
                xt = q_sample(x0, t, acp, noise)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=args.fp16 and device.type == "cuda",
            ):
                tokens = projector(latent)
                prediction = unet(xt, t, encoder_hidden_states=tokens).sample
                loss = F.mse_loss(prediction.float(), noise.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * eeg.size(0)
            seen += eeg.size(0)

        validation_loss = deterministic_validation_loss(
            name, encoder, projector, unet, loaders["val"],
            class_latents, acp, args, device
        )
        row = {
            "epoch": epoch,
            "train_loss": total / max(seen, 1),
            "validation_diffusion_loss": validation_loss,
        }
        history.append(row)
        print(
            f"[{name}] epoch={epoch:03d} train={row['train_loss']:.6f} "
            f"val={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss - 1e-12:
            best_loss = validation_loss
            best_epoch = epoch
            torch.save({
                "projector": projector.state_dict(),
                "encoder_condition": name,
                "best_epoch": best_epoch,
                "best_validation_diffusion_loss": best_loss,
                "config": vars(args),
            }, path)
        if args.patience > 0 and epoch - best_epoch >= args.patience:
            print(f"[{name}] early stop epoch={epoch} best={best_epoch}", flush=True)
            break

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    projector.load_state_dict(checkpoint["projector"], strict=True)
    with (path.parent / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return projector.eval(), checkpoint


def select_trials(dataset, per_class: int):
    by_class = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        by_class[int(sample[2])].append(index)
    selected = []
    for label in range(9):
        if len(by_class[label]) < per_class:
            raise RuntimeError(f"Class {label} has fewer than {per_class} test trials")
        selected.extend(by_class[label][:per_class])
    shuffled = {}
    for label in range(9):
        target = (label + 1) % 9
        for rank, index in enumerate(by_class[label][:per_class]):
            shuffled[index] = by_class[target][rank % len(by_class[target])]
    return selected, shuffled


def fixed_noise(keys, device):
    tensors = []
    for key in keys:
        generator = torch.Generator(device="cpu").manual_seed(int(key))
        tensors.append(torch.randn((4, 64, 64), generator=generator))
    return torch.stack(tensors).to(device)


@torch.no_grad()
def sample_tokens(unet, tokens, noise, acp, args):
    sequence = list(range(0, args.num_timesteps, args.num_timesteps // args.ddim_steps))
    x = noise
    for index in reversed(range(len(sequence))):
        timestep = torch.full(
            (x.size(0),), sequence[index], dtype=torch.long, device=x.device
        )
        prediction = unet(x, timestep, encoder_hidden_states=tokens).sample
        alpha = acp[sequence[index]]
        previous = (
            acp[sequence[index - 1]]
            if index > 0
            else torch.tensor(1.0, device=x.device, dtype=acp.dtype)
        )
        x0 = (x - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
        x = previous.sqrt() * x0.clamp(-1, 1) + (1 - previous).sqrt() * prediction
    return x


@torch.no_grad()
def dino_prototypes(img_root, dino, device):
    to_tensor = T.ToTensor()
    features = []
    for class_id in CLS_LIST:
        image = Image.open(Path(img_root) / f"{class_id:02d}.png").convert("RGB")
        tensor = DINO_EVAL_TF(to_tensor(image)).unsqueeze(0).to(device)
        features.append(F.normalize(dino(tensor), dim=-1))
    return torch.cat(features)


def aggregate(rows, features, prototypes):
    result = {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["conditioning"] == condition]
        predictions = [row["pred_label"] for row in subset]
        counts = np.bincount(predictions, minlength=9)
        probabilities = counts / max(counts.sum(), 1)
        nonzero = probabilities[probabilities > 0]
        result[condition] = {
            "n_images": len(subset),
            "top1": float(np.mean([row["correct"] for row in subset])),
            "top3": float(np.mean([row["top3_correct"] for row in subset])),
            "top5": float(np.mean([row["top5_correct"] for row in subset])),
            "normalized_entropy": float(-(nonzero * np.log(nonzero)).sum() / math.log(9)),
            "dominant_ratio": float(counts.max() / max(counts.sum(), 1)),
            "prediction_counts": counts.tolist(),
        }
    correct = torch.stack(features["correct"])
    for condition in ("shuffled", "zero"):
        other = torch.stack(features[condition])
        result[condition]["mean_dino_cosine_vs_correct"] = float(
            F.cosine_similarity(correct, other).mean()
        )
        correct_pred = [r["pred_label"] for r in rows if r["conditioning"] == "correct"]
        other_pred = [r["pred_label"] for r in rows if r["conditioning"] == condition]
        result[condition]["prediction_change_rate_vs_correct"] = float(
            np.mean(np.asarray(correct_pred) != np.asarray(other_pred))
        )
    return result


@torch.no_grad()
def evaluate_generation(
    name, encoder, projector, unet, vae, dino, prototypes,
    test_dataset, acp, args, device, output
):
    selected, shuffled = select_trials(test_dataset, args.samples_per_class)
    rows = []
    features = {condition: [] for condition in CONDITIONS}
    image_dir = output / name / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(selected), args.batch_size):
        indices = selected[start : start + args.batch_size]
        eeg = torch.stack([test_dataset[i][0] for i in indices]).to(device)
        shuffled_eeg = torch.stack([test_dataset[shuffled[i]][0] for i in indices]).to(device)
        labels = torch.tensor([int(test_dataset[i][2]) for i in indices], device=device)
        correct_latent = encode(encoder, name, eeg)
        shuffled_latent = encode(encoder, name, shuffled_eeg)
        token_sets = {
            "correct": projector(correct_latent),
            "shuffled": projector(shuffled_latent),
        }
        token_sets["zero"] = torch.zeros_like(token_sets["correct"])
        for generation in range(args.generations_per_trial):
            seeds = [args.seed * 100000 + i * 100 + generation for i in indices]
            initial_noise = fixed_noise(seeds, device)
            for condition in CONDITIONS:
                latent = sample_tokens(
                    unet, token_sets[condition], initial_noise.clone(), acp, args
                )
                decoded = vae.decode(latent / VAE_SCALE).sample.clamp(-1, 1)
                decoded = (decoded + 1) / 2
                dino_input = torch.stack([DINO_EVAL_TF(image) for image in decoded])
                batch_features = F.normalize(dino(dino_input), dim=-1)
                similarities = batch_features @ prototypes.T
                top5 = similarities.topk(5, dim=1).indices
                for offset, test_index in enumerate(indices):
                    truth = int(labels[offset])
                    prediction = int(top5[offset, 0])
                    filename = (
                        f"test{test_index:04d}_g{generation}_{condition}_seed{seeds[offset]}.png"
                    )
                    to_pil_image(decoded[offset].cpu()).save(image_dir / filename)
                    features[condition].append(batch_features[offset].cpu())
                    rows.append({
                        "encoder_condition": name,
                        "test_index": test_index,
                        "generation": generation,
                        "conditioning": condition,
                        "true_label": truth,
                        "pred_label": prediction,
                        "correct": int(prediction == truth),
                        "top3_correct": int(truth in top5[offset, :3].tolist()),
                        "top5_correct": int(truth in top5[offset, :5].tolist()),
                        "seed": seeds[offset],
                        "filename": filename,
                    })
        print(f"[{name}] generated {min(start + args.batch_size, len(selected))}/{len(selected)} trials")
    with (output / name / "generation_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return aggregate(rows, features, prototypes)


def main() -> None:
    args = parse_args()
    audit(args)
    if args.audit_only:
        return
    if args.ddim_steps <= 0 or args.num_timesteps % args.ddim_steps == 0 and args.ddim_steps > args.num_timesteps:
        raise ValueError("Invalid ddim_steps")
    output = Path(args.out_root) / f"seed{args.seed}" / f"S{args.subject_id:02d}"
    metrics_path = output / "generation_comparison.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite to rerun")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}", flush=True)

    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae"
    ).to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    class_latents = encode_class_images_512(
        vae, args.img_root, CLS_LIST, device
    )
    unet = load_frozen_unet(args, device)
    _, _, acp = make_schedule(args.num_timesteps, device)
    encoders = load_encoders(args, device)
    datasets, _, removed = build_loaders(
        args.data_root, args.subject_id, args, "vi"
    )

    print("[INFO] Loading DINO", flush=True)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    for parameter in dino.parameters():
        parameter.requires_grad_(False)
    prototypes = dino_prototypes(args.img_root, dino, device)

    results = {}
    for name in ENCODERS:
        projector, checkpoint = train_bridge(
            name, encoders[name], unet, class_latents, acp, args, device, output
        )
        generation = evaluate_generation(
            name, encoders[name], projector, unet, vae, dino, prototypes,
            datasets["test"], acp, args, device, output
        )
        results[name] = {
            "best_epoch": checkpoint["best_epoch"],
            "best_validation_diffusion_loss": checkpoint[
                "best_validation_diffusion_loss"
            ],
            "generation": generation,
            "test_evaluations": 1,
        }

    metrics = {
        "subject": args.subject_id,
        "seed": args.seed,
        "protocol": "frozen_shared_UNet_LoRA_equal_encoder_specific_bridges",
        "generator_checkpoint": str(Path(args.generator_ckpt).resolve()),
        "test_trials_per_class": args.samples_per_class,
        "generations_per_trial": args.generations_per_trial,
        "ddim_steps": args.ddim_steps,
        "nonfinite_removed": removed,
        "results": results,
        "config": vars(args),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"[Saved] {output}", flush=True)


if __name__ == "__main__":
    main()
