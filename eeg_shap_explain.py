# eeg_shap_explain.py
import os
import glob
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_subject import EEGImageSubjectDataset
from model import EEGDiffusionModel

import matplotlib.pyplot as plt

try:
    import shap
except ImportError as e:
    raise ImportError(
        "shap 패키지가 필요합니다. 먼저\n\n"
        "    pip install shap\n\n"
        "으로 설치해 주세요."
    ) from e


class EEGClassifier(nn.Module):
    """
    EEGDiffusionModel 안의 eeg_encoder 위에
    간단한 Linear head를 올린 classifier.
    - 입력: eeg (B, C, T)
    - 출력: logits (B, num_classes)
    """

    def __init__(self, eeg_encoder: nn.Module, feat_dim: int = 256, num_classes: int = 9):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, eeg):
        # eeg: (B, C, T)
        feat = self.eeg_encoder(eeg)  # (B, feat_dim)
        logits = self.head(feat)
        return logits


def train_classifier(
    classifier: EEGClassifier,
    train_ds: EEGImageSubjectDataset,
    device,
    epochs: int = 5,
    batch_size: int = 64,
    num_workers: int = 4,
    lr: float = 1e-3,
):
    """
    EEGClassifier를 label(1~9)을 예측하도록 간단히 학습.
    encoder는 freeze하고 head만 학습하는 구조로 설계.
    """
    classifier.to(device)
    classifier.train()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # encoder는 freeze (가중치는 고정, 하지만 입력에 대한 gradient는 흘러감)
    for p in classifier.eeg_encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(classifier.head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        total = 0

        for eeg, img, label in train_loader:
            eeg = eeg.to(device)
            # 라벨이 1~9라고 가정 → 0~8로 변환
            y = (label.to(device) - 1).long()

            optimizer.zero_grad(set_to_none=True)
            logits = classifier(eeg)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * eeg.size(0)
            _, preds = torch.max(logits, dim=1)
            running_correct += (preds == y).sum().item()
            total += eeg.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_correct / total
        print(f"[CLS] Epoch {epoch} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

    print("[CLS] Classifier training finished.")


def build_and_load_diffusion_model(args, subject_id, device):
    """
    EEGDiffusionModel을 만들고 subjXX_final.pt를 로드한 뒤 반환.
    """
    # dummy dataset에서 EEG 채널 수 추론
    tmp_ds = EEGImageSubjectDataset(
        data_root=args.data_root,
        subject_id=subject_id,
        split="train",
        split_ratio=args.split_ratio,
        img_size=args.img_size,
        seed=args.seed,
    )
    eeg_sample, img_sample, _ = tmp_ds[0]
    eeg_channels = eeg_sample.shape[0]
    print(
        f"[Subj {subject_id:02d}] Building diffusion model: "
        f"eeg_channels={eeg_channels}, img_size={args.img_size}"
    )

    model = EEGDiffusionModel(
        img_size=args.img_size,
        img_channels=3,
        eeg_channels=eeg_channels,
        eeg_hidden_dim=args.eeg_hidden_dim,
        time_dim=args.time_dim,
        base_channels=args.base_channels,
        num_timesteps=args.num_timesteps,
    ).to(device)

    # checkpoint 탐색
    ckpt_path = args.ckpt_path
    if ckpt_path is None:
        pattern = os.path.join(
            args.ckpt_out_dir,
            "*",
            f"subj{subject_id:02d}_final.pt"
        )
        candidates = glob.glob(pattern)
        if not candidates:
            raise FileNotFoundError(f"No diffusion checkpoint found matching: {pattern}")
        candidates.sort()
        ckpt_path = candidates[-1]

    print(f"[Subj {subject_id:02d}] Loading diffusion checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # run_id를 shap 결과 저장용으로 재사용
    ckpt_dir = os.path.dirname(ckpt_path)
    run_id = os.path.basename(ckpt_dir)

    return model, run_id


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    subject_id = args.subject_id

    # ---------- 1) diffusion 모델 + eeg_encoder 로드 ----------
    diffusion_model, run_id = build_and_load_diffusion_model(args, subject_id, device)

    # EEGDiffusionModel 안에 eeg_encoder가 있다고 가정
    if not hasattr(diffusion_model, "eeg_encoder"):
        raise AttributeError(
            "EEGDiffusionModel 안에 'eeg_encoder' 속성이 없습니다. "
            "model.py에서 eeg_encoder를 멤버 변수로 노출해야 합니다."
        )
    eeg_encoder = diffusion_model.eeg_encoder

    # ---------- 2) classifier 래퍼 구성 ----------
    num_classes = args.num_classes  # 보통 9
    classifier = EEGClassifier(
        eeg_encoder=eeg_encoder,
        feat_dim=args.eeg_hidden_dim,
        num_classes=num_classes,
    )

    # ---------- 3) dataset 준비 ----------
    train_ds = EEGImageSubjectDataset(
        data_root=args.data_root,
        subject_id=subject_id,
        split="train",
        split_ratio=args.split_ratio,
        img_size=args.img_size,
        seed=args.seed,
    )
    test_ds = EEGImageSubjectDataset(
        data_root=args.data_root,
        subject_id=subject_id,
        split="test",
        split_ratio=args.split_ratio,
        img_size=args.img_size,
        seed=args.seed,
    )
    print(
        f"[Subj {subject_id:02d}] #train trials={len(train_ds)}, "
        f"#test trials={len(test_ds)}"
    )

    # ---------- 4) classifier 간단 학습 ----------
    train_classifier(
        classifier,
        train_ds,
        device,
        epochs=args.cls_epochs,
        batch_size=args.cls_batch_size,
        num_workers=args.num_workers,
        lr=args.cls_lr,
    )

    classifier.eval()

    # ---------- 5) SHAP용 background / explain 샘플 준비 ----------
    # background: train에서 일부 EEG 샘플
    n_bg = min(args.n_background, len(train_ds))
    bg_indices = np.random.RandomState(args.seed).choice(
        len(train_ds),
        size=n_bg,
        replace=False,
    )
    bg_eegs = []
    for i in bg_indices:
        eeg, img, label = train_ds[int(i)]
        bg_eegs.append(eeg.unsqueeze(0))
    background = torch.cat(bg_eegs, dim=0).to(device)  # (N_bg, C, T)
    print(f"[SHAP] background shape: {background.shape}")

    # explain 대상으로 test에서 EEG 샘플 선택
    n_exp = min(args.n_explain, len(test_ds))
    exp_indices = np.random.RandomState(args.seed + 1).choice(
        len(test_ds),
        size=n_exp,
        replace=False,
    )
    exp_eegs = []
    exp_labels = []
    for i in exp_indices:
        eeg, img, label = test_ds[int(i)]
        exp_eegs.append(eeg.unsqueeze(0))
        exp_labels.append(int(label))
    exp_eegs = torch.cat(exp_eegs, dim=0).to(device)  # (N_exp, C, T)
    exp_labels = np.array(exp_labels)
    print(f"[SHAP] explain shape: {exp_eegs.shape}")

    # ---------- 6) SHAP GradientExplainer ----------
    # classifier: (B, C, T) -> (B, num_classes)
    print("[SHAP] Building GradientExplainer...")
    explainer = shap.GradientExplainer(classifier, background)

    # shap_values: list length = num_classes, each (N_exp, C, T)
    print("[SHAP] Computing shap_values (this may take some time)...")
    shap_values = explainer.shap_values(exp_eegs)

    # shap는 numpy로 반환되는 경우가 많음
    # shap_values[i] -> class i에 대한 shap 값, shape: (N_exp, C, T)

    # ---------- 7) 결과 저장 ----------
    save_root = args.save_dir
    os.makedirs(save_root, exist_ok=True)
    shap_dir = os.path.join(save_root, run_id + f"_subj{subject_id:02d}")
    os.makedirs(shap_dir, exist_ok=True)
    print(f"[SHAP] Saving SHAP results under: {shap_dir}")

    # (1) shap_values 전체 저장 (class별로)
    for c in range(len(shap_values)):
        sv = np.array(shap_values[c])  # (N_exp, C, T)
        out_path = os.path.join(shap_dir, f"shap_values_class{c}.npy")
        np.save(out_path, sv)
        print(f"  - Saved {out_path} with shape {sv.shape}")

    # (2) 인덱스/라벨 정보 저장
    np.save(os.path.join(shap_dir, "exp_indices.npy"), exp_indices)
    np.save(os.path.join(shap_dir, "exp_labels.npy"), exp_labels)
    np.save(os.path.join(shap_dir, "bg_indices.npy"), bg_indices)

    # (3) 예시 heatmap (채널 × 시간) 저장 (몇 개 샘플만)
    num_plot = min(args.n_plot, exp_eegs.shape[0])
    time_len = exp_eegs.shape[-1]
    channels = exp_eegs.shape[1]

    # class_index: -1이면 "예측 클래스" 기준으로 그림
    class_index = args.class_index

    with torch.no_grad():
        logits = classifier(exp_eegs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()  # 0~8

    for i in range(num_plot):
        if class_index >= 0:
            c_idx = class_index
        else:
            c_idx = int(preds[i])  # 해당 샘플의 예측 클래스

        sv = np.array(shap_values[c_idx][i])  # (C, T)

        vmax = np.max(np.abs(sv)) + 1e-8

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            sv,
            aspect="auto",
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
            origin="lower",
        )
        ax.set_xlabel("Time (sample index)")
        ax.set_ylabel("Channel index")
        ax.set_title(
            f"Subj {subject_id:02d} | Sample {i} | "
            f"Pred class={preds[i]+1} | Explained class={c_idx+1}"
        )
        plt.colorbar(im, ax=ax, label="SHAP value")
        fig.tight_layout()

        out_png = os.path.join(
            shap_dir,
            f"shap_heatmap_sample{i}_class{c_idx+1}.png",
        )
        fig.savefig(out_png)
        plt.close(fig)
        print(f"  - Saved heatmap: {out_png}")

    print("[SHAP] Done.")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./preproc_data")
    p.add_argument("--subject_id", type=int, default=1)

    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--num_timesteps", type=int, default=200)
    p.add_argument("--eeg_hidden_dim", type=int, default=256)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--split_ratio", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=2025)

    # diffusion checkpoint
    p.add_argument(
        "--ckpt_out_dir",
        type=str,
        default="./checkpoints_subj",
        help="diffusion 체크포인트들이 들어있는 상위 폴더 (train_subject.py에서 out_dir)",
    )
    p.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="특정 checkpoint 경로를 직접 지정하고 싶으면 사용. "
             "None이면 ckpt_out_dir/*/subjXX_final.pt 중 최신을 사용.",
    )

    # classifier 설정
    p.add_argument("--num_classes", type=int, default=9)
    p.add_argument("--cls_epochs", type=int, default=5)
    p.add_argument("--cls_batch_size", type=int, default=64)
    p.add_argument("--cls_lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)

    # SHAP 설정
    p.add_argument("--n_background", type=int, default=50)
    p.add_argument("--n_explain", type=int, default=20)
    p.add_argument(
        "--class_index",
        type=int,
        default=-1,
        help="설명할 클래스 인덱스(0~num_classes-1). "
             "-1이면 각 샘플의 예측된 클래스를 기준으로 설명.",
    )
    p.add_argument("--n_plot", type=int, default=5)

    p.add_argument(
        "--save_dir",
        type=str,
        default="./shap_results",
        help="SHAP 결과 저장 상위 폴더",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
