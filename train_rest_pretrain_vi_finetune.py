"""
train_rest_pretrain_vi_finetune.py
─────────────────────────────────────────────────────────────────────────────
2단계 학습 파이프라인

  [Stage 1]  preproc_data_rest  →  EEG 인코더 자기지도 사전학습
             라벨·이미지 없이 EEG만 사용 (Masked Temporal Reconstruction)
             저장: {ckpt_root}/pretrain/encoder_best.pt

  [Stage 2]  preproc_data_vi    →  REPA 확산 모델 파인튜닝
             사전학습된 인코더를 초기값으로 사용
             저장: {ckpt_root}/finetune/{timestamp}_subj{tag}_rest2vi/

  [테스트]   sample_vi_repa_allclass.py --ckpt_path <Stage2 best.pt> 사용

사용 예시
─────────
# Stage 1+2 모두 실행 (1~10번 피험자 rest로 사전학습, 같은 번호 vi로 파인튜닝)
python train_rest_pretrain_vi_finetune.py \\
    --rest_subjects 1-10 --vi_subjects 1-10

# 사전학습 건너뛰기 (이미 encoder_best.pt 있을 때)
python train_rest_pretrain_vi_finetune.py \\
    --vi_subjects 1-10 --skip_pretrain \\
    --pretrain_ckpt ./checkpoints_rest2vi/pretrain/encoder_best.pt

# 전체 피험자
python train_rest_pretrain_vi_finetune.py \\
    --rest_subjects all --vi_subjects all \\
    --pretrain_epochs 30 --finetune_epochs 100
"""

import os
import copy
import glob
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

from model_128_eegonly_transformer import EEGEncoderTransformer
from model_128_eegonly_transformer_repa import EEGDiffusionModel128


# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def ema_update(model, ema_model, decay: float = 0.999):
    msd = model.state_dict()
    for name, param in ema_model.state_dict().items():
        if name in msd:
            param.copy_(param * decay + msd[name] * (1.0 - decay))


def parse_subject_ids(s: str, data_root: str) -> list:
    """
    'all'       → data_root 의 subj_*.mat 전체
    '1-10'      → [1, 2, ..., 10]
    '1,3,5'     → [1, 3, 5]
    '1-5,8,10'  → [1, 2, 3, 4, 5, 8, 10]
    """
    s = s.strip().lower()
    if s == "all":
        paths = sorted(glob.glob(os.path.join(data_root, "subj_*.mat")))
        return [int(os.path.basename(p).replace("subj_", "").replace(".mat", ""))
                for p in paths]
    ids = []
    for token in s.split(","):
        token = token.strip()
        if "-" in token:
            a, b = token.split("-", 1)
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(token))
    return ids


def detect_eeg_channels(mat_path: str) -> int:
    try:
        mat = loadmat(mat_path)
        return int(mat["X"].shape[0])
    except Exception as e:
        print(f"[WARN] EEG 채널 감지 실패: {e}. 기본값 32 사용.")
        return 32


def detect_eeg_timepoints(mat_path: str) -> int:
    try:
        mat = loadmat(mat_path)
        return int(mat["X"].shape[1])
    except Exception as e:
        print(f"[WARN] EEG timepoint 감지 실패: {e}. 기본값 256 사용.")
        return 256


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 구성요소: 데이터셋 / 디코더 / Masked AE
# ─────────────────────────────────────────────────────────────────────────────

class RestingEEGDataset(Dataset):
    """
    preproc_data_rest: X(ch, time, trial) 만 있음.
    라벨·이미지 없음.
    """
    def __init__(self, mat_paths: list):
        segs = []
        for p in mat_paths:
            mat = loadmat(p)
            X = mat["X"]  # (ch, time, trial)
            eeg = torch.from_numpy(X.astype(np.float32)).permute(2, 0, 1)  # (trial, ch, T)
            segs.append(eeg)
        self.eeg = torch.cat(segs, dim=0)  # (N, ch, T)

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return self.eeg[idx]  # (ch, T)


class EEGTemporalDecoder(nn.Module):
    """
    EEG 인코더 잠재 벡터 (B, latent_dim) → (B, eeg_channels, T) 재구성.
    Stage 1 전용 경량 MLP 디코더.
    """
    def __init__(self, latent_dim: int, eeg_channels: int, T: int):
        super().__init__()
        self.eeg_channels = eeg_channels
        self.T = T
        hidden = latent_dim * 4
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, eeg_channels * T),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(z.size(0), self.eeg_channels, self.T)


class MaskedEEGAutoencoder(nn.Module):
    """
    Stage 1: Masked Temporal Reconstruction
    - 전체 시간 길이를 patch_size 단위로 나눔
    - mask_ratio 비율의 패치를 0으로 마스킹
    - 인코더 → 디코더로 마스킹된 위치만 복원
    """
    def __init__(
        self,
        encoder: EEGEncoderTransformer,
        decoder: EEGTemporalDecoder,
        mask_ratio: float = 0.5,
        patch_size: int = 16,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor):
        """
        x: (B, ch, T)
        returns: (loss, x_recon, mask_indices)
        """
        B, C, T = x.shape
        patch_size = self.patch_size

        # 패치 단위 마스킹
        n_patches = T // patch_size
        n_mask = max(1, int(n_patches * self.mask_ratio))

        # 각 배치마다 동일한 마스크 패턴 (배치 독립 마스킹도 가능하나 단순화)
        perm = torch.randperm(n_patches, device=x.device)
        masked_patches = perm[:n_mask]                          # 마스킹할 패치 인덱스

        # 마스킹된 위치의 전체 시간 인덱스 수집
        mask_time_idx = []
        for pi in masked_patches:
            start = int(pi.item()) * patch_size
            mask_time_idx.extend(range(start, start + patch_size))
        mask_time_idx = torch.tensor(mask_time_idx, device=x.device, dtype=torch.long)

        # 입력 마스킹 (해당 시간 위치를 0으로)
        x_masked = x.clone()
        x_masked[:, :, mask_time_idx] = 0.0

        # 인코더 → 잠재 벡터
        z = self.encoder(x_masked)          # (B, latent_dim)

        # 디코더 → 전체 시계열 복원
        x_recon = self.decoder(z)           # (B, ch, T)

        # 마스킹된 위치에 대해서만 MSE 계산
        loss = F.mse_loss(
            x_recon[:, :, mask_time_idx],
            x[:, :, mask_time_idx],
        )
        return loss, x_recon, mask_time_idx


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: vi 데이터셋 (9-class)
# ─────────────────────────────────────────────────────────────────────────────

class EEGImageDataset9Class(Dataset):
    def __init__(self, mat_paths: list, img_root: str, indices: list, img_size: int = 128):
        super().__init__()
        self.img_root = img_root
        self.indices = indices

        self.eegs, self.labels = [], []
        for p in mat_paths:
            mat = loadmat(p)
            X = mat["X"]
            y = mat["y"].squeeze()
            self.eegs.append(torch.from_numpy(X.astype(np.float32)).permute(2, 0, 1))
            self.labels.append(y.astype(np.int64))

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self._img_cache = {}
        for cls in range(1, 10):
            p = os.path.join(img_root, f"{cls:02d}.png")
            if os.path.isfile(p):
                self._img_cache[cls] = Image.open(p).convert("RGB")
            else:
                raise FileNotFoundError(f"클래스 이미지 없음: {p}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s_idx, t_idx = self.indices[idx]
        eeg = self.eegs[s_idx][t_idx]
        label_1 = int(self.labels[s_idx][t_idx])
        img = self.transform(self._img_cache[label_1])
        return eeg, img, label_1 - 1  # 0-based label


def build_vi_split(mat_paths: list, seed: int = 42):
    train_idx, val_idx, test_idx = [], [], []
    for s_idx, p in enumerate(mat_paths):
        mat = loadmat(p)
        y = mat["y"].squeeze().astype(np.int64)
        for cls in range(1, 10):
            cls_trials = np.where(y == cls)[0]
            if len(cls_trials) == 0:
                continue
            rng = np.random.RandomState(seed + s_idx * 100 + cls)
            rng.shuffle(cls_trials)
            n = len(cls_trials)
            n_train = int(n * 0.8)
            n_val   = int(n * 0.1)
            for t in cls_trials[:n_train]:
                train_idx.append((s_idx, int(t)))
            for t in cls_trials[n_train:n_train + n_val]:
                val_idx.append((s_idx, int(t)))
            for t in cls_trials[n_train + n_val:]:
                test_idx.append((s_idx, int(t)))
    return train_idx, val_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: 사전학습
# ─────────────────────────────────────────────────────────────────────────────

def run_pretrain(args, device) -> str:
    """
    rest EEG로 EEGEncoderTransformer를 자기지도 학습.
    Returns: 저장된 인코더 체크포인트 경로
    """
    print("\n" + "=" * 70)
    print("  [Stage 1] REST EEG 자기지도 사전학습 (Masked Temporal Reconstruction)")
    print("=" * 70)

    # 피험자 목록
    rest_ids = parse_subject_ids(args.rest_subjects, args.rest_root)
    rest_paths = []
    for sid in rest_ids:
        p = os.path.join(args.rest_root, f"subj_{sid:02d}.mat")
        if os.path.isfile(p):
            rest_paths.append(p)
        else:
            print(f"[SKIP] rest subj_{sid:02d}.mat 없음")

    if not rest_paths:
        raise FileNotFoundError("사용 가능한 rest mat 파일이 없습니다.")

    print(f"[INFO] rest 피험자 수: {len(rest_paths)}")

    eeg_ch = detect_eeg_channels(rest_paths[0])
    T_len  = detect_eeg_timepoints(rest_paths[0])
    print(f"[INFO] EEG ch={eeg_ch}, T={T_len}")

    # 데이터로더
    ds = RestingEEGDataset(rest_paths)
    loader = DataLoader(
        ds, batch_size=args.pretrain_batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    print(f"[INFO] rest trial 수: {len(ds)}, 배치 수: {len(loader)}")

    # 모델
    encoder = EEGEncoderTransformer(
        eeg_channels=eeg_ch,
        eeg_hidden_dim=args.eeg_hidden_dim,
        out_dim=args.cond_dim,
        n_heads=args.eeg_tf_heads,
        n_layers=args.eeg_tf_layers,
        dropout=args.eeg_tf_dropout,
    ).to(device)

    decoder = EEGTemporalDecoder(
        latent_dim=args.cond_dim,
        eeg_channels=eeg_ch,
        T=T_len,
    ).to(device)

    mae = MaskedEEGAutoencoder(
        encoder=encoder,
        decoder=decoder,
        mask_ratio=args.mask_ratio,
        patch_size=args.patch_size,
    ).to(device)

    optimizer = torch.optim.AdamW(mae.parameters(), lr=args.pretrain_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.pretrain_epochs * len(loader), eta_min=args.pretrain_lr * 0.05
    )

    # 저장 경로
    pretrain_dir = os.path.join(args.ckpt_root, "pretrain")
    os.makedirs(pretrain_dir, exist_ok=True)
    best_ckpt = os.path.join(pretrain_dir, "encoder_best.pt")

    losses = []
    best_loss = float("inf")

    for epoch in range(args.pretrain_epochs):
        mae.train()
        epoch_losses = []

        for step, eeg in enumerate(loader):
            eeg = eeg.to(device)
            loss, _, _ = mae(eeg)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            losses.append(loss.item())

            if step % args.log_interval == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  [Pretrain Epoch {epoch:03d}/{args.pretrain_epochs}] "
                    f"Step {step:04d}/{len(loader)}  Loss {loss.item():.4f}  LR {lr_now:.2e}"
                )

        mean_loss = float(np.mean(epoch_losses))
        print(f"  [Pretrain Epoch {epoch:03d}] MeanLoss={mean_loss:.4f}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            # pos_embed 초기화를 위해 더미 패스 1회 실행 (이미 실행됨)
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "eeg_ch": eeg_ch,
                    "T_len": T_len,
                    "cond_dim": args.cond_dim,
                    "eeg_hidden_dim": args.eeg_hidden_dim,
                    "eeg_tf_heads": args.eeg_tf_heads,
                    "eeg_tf_layers": args.eeg_tf_layers,
                    "eeg_tf_dropout": args.eeg_tf_dropout,
                },
                best_ckpt,
            )
            print(f"  ↳ Pretrain best 저장 (loss={mean_loss:.4f}) → {best_ckpt}")

    # 손실 곡선
    np.save(os.path.join(pretrain_dir, "pretrain_loss.npy"), np.array(losses, dtype=np.float32))
    plt.figure(figsize=(7, 3))
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.title("Stage 1: Masked EEG Reconstruction Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(pretrain_dir, "pretrain_loss.png"))
    plt.close()

    print(f"[Stage 1 완료] 인코더 체크포인트 → {best_ckpt}")
    return best_ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: vi 파인튜닝
# ─────────────────────────────────────────────────────────────────────────────

def run_finetune(args, pretrain_ckpt: str, device) -> str:
    """
    vi EEG+이미지로 REPA 확산 모델 파인튜닝.
    pretrain_ckpt: Stage 1에서 저장된 encoder_best.pt 경로 (없으면 무작위 초기화)
    Returns: 저장 디렉터리 경로
    """
    print("\n" + "=" * 70)
    print("  [Stage 2] VI EEG → Image REPA 파인튜닝")
    print("=" * 70)

    # 피험자 목록
    vi_ids = parse_subject_ids(args.vi_subjects, args.vi_root)
    vi_paths = []
    for sid in vi_ids:
        p = os.path.join(args.vi_root, f"subj_{sid:02d}.mat")
        if os.path.isfile(p):
            vi_paths.append(p)
        else:
            print(f"[SKIP] vi subj_{sid:02d}.mat 없음")

    if not vi_paths:
        raise FileNotFoundError("사용 가능한 vi mat 파일이 없습니다.")

    img_root = os.path.join(args.vi_root, "images")
    print(f"[INFO] vi 피험자 수: {len(vi_paths)}")

    # EEG 채널 감지 (vi 데이터 기준)
    eeg_ch = detect_eeg_channels(vi_paths[0])
    print(f"[INFO] EEG ch={eeg_ch}")

    # 데이터 분할
    train_idx, val_idx, _ = build_vi_split(vi_paths, seed=args.seed)
    print(f"[INFO] vi 분할 — train: {len(train_idx)}, val: {len(val_idx)}")

    train_ds = EEGImageDataset9Class(vi_paths, img_root, train_idx, args.img_size)
    val_ds   = EEGImageDataset9Class(vi_paths, img_root, val_idx,   args.img_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.finetune_batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.finetune_batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # 모델 생성
    ch_mult = [int(x) for x in args.ch_mult.split(",")]
    model = EEGDiffusionModel128(
        img_size=args.img_size,
        img_channels=3,
        eeg_channels=eeg_ch,
        num_classes=9,
        num_timesteps=args.num_timesteps,
        base_channels=args.base_channels,
        ch_mult=ch_mult,
        time_dim=256,
        cond_dim=args.cond_dim,
        eeg_hidden_dim=args.eeg_hidden_dim,
        cond_scale=2.0,
        n_res_blocks=args.n_res_blocks,
        lambda_rec=args.lambda_rec,
        lambda_ssim=args.lambda_ssim,
        lambda_repa=args.lambda_repa,
        repa_feat_dim=512,
        eeg_tf_heads=args.eeg_tf_heads,
        eeg_tf_layers=args.eeg_tf_layers,
        eeg_tf_dropout=args.eeg_tf_dropout,
    ).to(device)

    # 사전학습 인코더 가중치 로드
    if pretrain_ckpt and os.path.isfile(pretrain_ckpt):
        print(f"[INFO] 사전학습 인코더 로드: {pretrain_ckpt}")
        pt = torch.load(pretrain_ckpt, map_location=device)
        enc_state = pt["encoder"]
        missing, unexpected = model.eeg_encoder.load_state_dict(enc_state, strict=False)
        if missing:
            print(f"[WARN] 인코더 누락 키: {missing}")
        if unexpected:
            print(f"[WARN] 인코더 예상치 못한 키: {unexpected}")
        print("[INFO] 인코더 가중치 적용 완료")
    else:
        print("[INFO] 사전학습 없이 무작위 초기화")

    # EMA
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=1e-4)
    total_steps = args.finetune_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.finetune_lr * 0.05
    )

    # 저장 경로
    vi_tag = f"{vi_ids[0]:02d}-{vi_ids[-1]:02d}" if len(vi_ids) > 1 else f"{vi_ids[0]:02d}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        args.ckpt_root, "finetune",
        f"{timestamp}_subj{vi_tag}_rest2vi_{args.img_size}",
    )
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] 파인튜닝 체크포인트 → {save_dir}")

    meta = dict(
        vi_subjects=args.vi_subjects,
        rest_subjects=args.rest_subjects,
        pretrain_ckpt=pretrain_ckpt,
        eeg_ch=eeg_ch,
        img_size=args.img_size,
        base_channels=args.base_channels,
        ch_mult=args.ch_mult,
        num_timesteps=args.num_timesteps,
        n_res_blocks=args.n_res_blocks,
        num_classes=9,
        cond_dim=args.cond_dim,
        eeg_hidden_dim=args.eeg_hidden_dim,
        eeg_tf_heads=args.eeg_tf_heads,
        eeg_tf_layers=args.eeg_tf_layers,
        eeg_tf_dropout=args.eeg_tf_dropout,
        lambda_rec=args.lambda_rec,
        lambda_ssim=args.lambda_ssim,
        lambda_repa=args.lambda_repa,
    )

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(args.finetune_epochs):
        model.train()
        epoch_train = []

        for step, (eeg, img, labels) in enumerate(train_loader):
            eeg    = eeg.to(device)
            img    = img.to(device)
            labels = labels.to(device)

            b = img.size(0)
            t = torch.randint(0, model.num_timesteps, (b,), device=device, dtype=torch.long)
            loss = model.p_losses(img, eeg, labels, t)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ema_update(model, ema_model, decay=args.ema_decay)

            epoch_train.append(loss.item())
            train_losses.append(loss.item())

            if step % args.log_interval == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  [Finetune Epoch {epoch:03d}/{args.finetune_epochs}] "
                    f"Step {step:04d}/{len(train_loader)}  Loss {loss.item():.4f}  LR {lr_now:.2e}"
                )

        # Validation
        model.eval()
        epoch_val = []
        with torch.no_grad():
            for eeg, img, labels in val_loader:
                eeg    = eeg.to(device)
                img    = img.to(device)
                labels = labels.to(device)
                b = img.size(0)
                t = torch.randint(0, model.num_timesteps, (b,), device=device, dtype=torch.long)
                epoch_val.append(model.p_losses(img, eeg, labels, t).item())

        mean_train = float(np.mean(epoch_train)) if epoch_train else 0.0
        mean_val   = float(np.mean(epoch_val))   if epoch_val   else 0.0
        val_losses.append(mean_val)

        print(
            f"  [Finetune Epoch {epoch:03d}] "
            f"TrainLoss={mean_train:.4f}  ValLoss={mean_val:.4f}"
        )

        if mean_val < best_val:
            best_val = mean_val
            best_path = os.path.join(save_dir, "best.pt")
            torch.save(
                {"model": model.state_dict(), "ema": ema_model.state_dict(), "config": meta},
                best_path,
            )
            print(f"  ↳ Best 체크포인트 (val={mean_val:.4f}) → {best_path}")

        if (epoch + 1) % 25 == 0:
            ep_path = os.path.join(save_dir, f"epoch_{epoch+1:04d}.pt")
            torch.save(
                {"model": model.state_dict(), "ema": ema_model.state_dict(), "config": meta},
                ep_path,
            )

    # 최종 저장
    final_path = os.path.join(save_dir, "final.pt")
    torch.save(
        {"model": model.state_dict(), "ema": ema_model.state_dict(), "config": meta},
        final_path,
    )
    print(f"[Stage 2 완료] Final → {final_path}")

    # 손실 곡선
    np.save(os.path.join(save_dir, "train_loss.npy"), np.array(train_losses, dtype=np.float32))
    np.save(os.path.join(save_dir, "val_loss.npy"),   np.array(val_losses,   dtype=np.float32))
    plt.figure(figsize=(8, 4))
    plt.plot(val_losses, marker="o", label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Stage 2 Finetune Val Loss (subj {vi_tag})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_loss_curve.png"))
    plt.close()

    return save_dir


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="REST 사전학습 → VI 파인튜닝 (2-Stage EEG-to-Image)"
    )

    # ── 데이터 경로 ──────────────────────────────────────────────────────────
    parser.add_argument("--rest_root",    type=str, default="./preproc_data_rest",
                        help="resting state mat 폴더")
    parser.add_argument("--vi_root",      type=str, default="./preproc_data_vi",
                        help="visual imagery mat + images 폴더")
    parser.add_argument("--rest_subjects",type=str, default="1",
                        help="Stage 1 피험자: '1-10' / '1,3,5' / 'all'")
    parser.add_argument("--vi_subjects",  type=str, default="1",
                        help="Stage 2 피험자: '1-10' / '1,3,5' / 'all'")
    parser.add_argument("--img_size",     type=int, default=128)

    # ── Stage 1 (사전학습) ───────────────────────────────────────────────────
    parser.add_argument("--skip_pretrain",   action="store_true", default=False,
                        help="Stage 1 생략 (--pretrain_ckpt 로 직접 지정)")
    parser.add_argument("--pretrain_ckpt",   type=str, default=None,
                        help="기존 사전학습 체크포인트 경로 (skip_pretrain 시 필수)")
    parser.add_argument("--pretrain_epochs", type=int, default=30)
    parser.add_argument("--pretrain_batch",  type=int, default=32)
    parser.add_argument("--pretrain_lr",     type=float, default=1e-3)
    parser.add_argument("--mask_ratio",      type=float, default=0.5,
                        help="마스킹 비율 (0.0~1.0)")
    parser.add_argument("--patch_size",      type=int, default=16,
                        help="마스킹 패치 크기 (시간 축 단위)")

    # ── Stage 2 (파인튜닝) ───────────────────────────────────────────────────
    parser.add_argument("--finetune_epochs", type=int, default=100)
    parser.add_argument("--finetune_batch",  type=int, default=16)
    parser.add_argument("--finetune_lr",     type=float, default=1e-4)
    parser.add_argument("--ema_decay",       type=float, default=0.999)

    # ── 모델 공통 ────────────────────────────────────────────────────────────
    parser.add_argument("--num_timesteps", type=int,   default=200)
    parser.add_argument("--base_channels", type=int,   default=64)
    parser.add_argument("--ch_mult",       type=str,   default="1,2,4,4")
    parser.add_argument("--n_res_blocks",  type=int,   default=2)
    parser.add_argument("--cond_dim",      type=int,   default=256)
    parser.add_argument("--eeg_hidden_dim",type=int,   default=256)
    parser.add_argument("--eeg_tf_heads",  type=int,   default=4)
    parser.add_argument("--eeg_tf_layers", type=int,   default=2)
    parser.add_argument("--eeg_tf_dropout",type=float, default=0.1)

    # ── 손실 가중치 ──────────────────────────────────────────────────────────
    parser.add_argument("--lambda_rec",   type=float, default=0.02)
    parser.add_argument("--lambda_ssim",  type=float, default=0.05)
    parser.add_argument("--lambda_repa",  type=float, default=0.05)

    # ── 기타 ─────────────────────────────────────────────────────────────────
    parser.add_argument("--ckpt_root",   type=str, default="./checkpoints_rest2vi")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval",type=int, default=10)
    parser.add_argument("--seed",        type=int, default=42)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    print(f"[INFO] device: {device}")
    print(f"[INFO] rest_subjects: {args.rest_subjects}")
    print(f"[INFO] vi_subjects  : {args.vi_subjects}")

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    if args.skip_pretrain:
        pretrain_ckpt = args.pretrain_ckpt
        if pretrain_ckpt is None:
            # 기본 경로에서 탐색
            pretrain_ckpt = os.path.join(args.ckpt_root, "pretrain", "encoder_best.pt")
        if not os.path.isfile(pretrain_ckpt):
            print(f"[WARN] 사전학습 체크포인트 없음: {pretrain_ckpt}. 무작위 초기화로 진행.")
            pretrain_ckpt = None
        else:
            print(f"[INFO] 사전학습 체크포인트 사용: {pretrain_ckpt}")
    else:
        pretrain_ckpt = run_pretrain(args, device)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    save_dir = run_finetune(args, pretrain_ckpt, device)

    print("\n" + "=" * 70)
    print("  전체 학습 완료!")
    print(f"  파인튜닝 결과: {save_dir}")
    print()
    print("  [테스트 방법]")
    print(f"  python sample_vi_repa_allclass.py \\")
    print(f"      --subject_ids {args.vi_subjects} \\")
    print(f"      --ckpt_path {save_dir}/best.pt \\")
    print(f"      --use_ddim --sample_steps 50 --split test")
    print("=" * 70)


if __name__ == "__main__":
    main()
