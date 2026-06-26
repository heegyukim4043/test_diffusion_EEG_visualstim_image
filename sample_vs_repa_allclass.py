"""
sample_vs_repa_allclass.py
─────────────────────────────────────────────────────────────────────────────
train_vs_repa_allclass.py 로 학습한 모델에서 이미지를 생성한다.
(VS 데이터로 학습한 모델을 VS 데이터로 추론)

▸ 체크포인트의 EMA 가중치를 우선 사용 (없으면 model 가중치)
▸ DDPM / DDIM 두 가지 샘플링 지원
▸ 생성 이미지와 GT 이미지를 나란히 저장 (선택)

사용 예시
─────────
# 단일 피험자 (DDIM)
python sample_vs_repa_allclass.py --subject_ids 1 --use_ddim --sample_steps 50

# 범위 지정
python sample_vs_repa_allclass.py --subject_ids 1-10 --use_ddim --sample_steps 50

# 전체 피험자, 그룹 1만
python sample_vs_repa_allclass.py --subject_ids all --group_ids 1

# 9-class 전체 모드 (group_ids=0 으로 학습한 경우)
python sample_vs_repa_allclass.py --subject_ids 1 --group_ids 0

# test set만 생성
python sample_vs_repa_allclass.py --subject_ids 1-10 --split test

# 체크포인트 직접 지정
python sample_vs_repa_allclass.py --subject_ids 1 \\
    --ckpt_path ./checkpoints_vs_repa/<dir>/best.pt \\
    --use_ddim --sample_steps 50 --guidance_scale 3.0

# 전체 trial 생성 (GT 저장 없이)
python sample_vs_repa_allclass.py --subject_ids all --split all --no_gt
"""

import os
import glob
import random
import argparse

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

from model_128_eegonly_transformer_repa import EEGDiffusionModel128


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def denorm(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1]"""
    return (x.clamp(-1, 1) + 1.0) * 0.5


GROUP_CLASSES = {
    1: [1, 2, 3],
    2: [4, 5, 6],
    3: [7, 8, 9],
}


def parse_group_ids(s: str) -> list:
    s = s.strip().lower()
    if s == "all":
        return [1, 2, 3]
    ids = []
    for token in s.split(","):
        token = token.strip()
        if "-" in token:
            a, b = token.split("-", 1)
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(token))
    for gid in ids:
        if gid not in (1, 2, 3):
            raise ValueError(f"group_id는 1, 2, 3 중 하나여야 합니다. (입력: {gid})")
    return ids


def parse_subject_ids(s: str, data_root: str) -> list:
    s = s.strip().lower()
    if s == "all":
        import glob as _glob
        paths = sorted(_glob.glob(os.path.join(data_root, "subj_*.mat")))
        ids = []
        for p in paths:
            base = os.path.basename(p)
            num  = base.replace("subj_", "").replace(".mat", "")
            ids.append(int(num))
        return ids

    ids = []
    for token in s.split(","):
        token = token.strip()
        if "-" in token:
            a, b = token.split("-", 1)
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(token))
    return ids


def find_best_ckpt(ckpt_root: str, subj_tag: str, group_tag: str = "") -> str | None:
    """
    ckpt_root 하위에서 subj 태그 (+ 선택적 group 태그)가 포함된
    가장 최신 디렉터리의 best.pt 를 찾는다.
    VS 형식: *vs{subj_tag}*{group_tag}*
    """
    if not os.path.isdir(ckpt_root):
        return None

    # VS 형식 우선 탐색 (*vs01*g1_cls1-3* 등)
    pattern_vs = f"*vs{subj_tag}*{group_tag}*" if group_tag else f"*vs{subj_tag}*"
    cands = sorted(glob.glob(os.path.join(ckpt_root, pattern_vs)))
    for d in reversed(cands):
        if not os.path.isdir(d):
            continue
        bp = os.path.join(d, "best.pt")
        if os.path.isfile(bp):
            return bp
        fp = os.path.join(d, "final.pt")
        if os.path.isfile(fp):
            return fp
    return None


def find_img_root(vs_root: str, fallback: str = "./preproc_data_vi/images") -> str:
    vs_img = os.path.join(vs_root, "images")
    if os.path.isdir(vs_img):
        return vs_img
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (추론 전용)
# ─────────────────────────────────────────────────────────────────────────────

class EEGInferenceDatasetVS(Dataset):
    """VS mat 파일의 특정 trial 집합에 대한 EEG + (선택적) GT 이미지 로드."""

    def __init__(
        self,
        mat_path: str,
        img_root: str,
        indices: list,
        img_size: int = 128,
        load_gt: bool = True,
        cls_list: list = None,
    ):
        super().__init__()
        self.indices = np.array(indices, dtype=np.int64)
        self.img_size = img_size
        self.load_gt = load_gt

        mat = loadmat(mat_path)
        X = mat["X"]           # (ch, time, trial)
        y = mat["y"].squeeze() # (trial,)

        self.eeg    = torch.from_numpy(X.astype(np.float32)).permute(2, 0, 1)
        self.labels = y.astype(np.int64)

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # 클래스 이미지 캐시
        load_classes = cls_list if cls_list is not None else list(range(1, 10))
        self._img_cache = {}
        if load_gt:
            for cls in load_classes:
                p = os.path.join(img_root, f"{cls:02d}.png")
                if os.path.isfile(p):
                    self._img_cache[cls] = Image.open(p).convert("RGB")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t_idx        = int(self.indices[idx])
        eeg          = self.eeg[t_idx]
        label_1based = int(self.labels[t_idx])
        label_0based = label_1based - 1  # 0~8 (전체 표기용)

        if self.load_gt and label_1based in self._img_cache:
            gt_img = self.transform(self._img_cache[label_1based])
        else:
            gt_img = torch.zeros(3, self.img_size, self.img_size)

        return eeg, gt_img, label_0based, label_1based, t_idx


# ─────────────────────────────────────────────────────────────────────────────
# 분할 인덱스 재현 (학습 때와 동일한 시드)
# ─────────────────────────────────────────────────────────────────────────────

def get_split_indices(mat_path: str, split: str, seed: int = 42, class_filter: list = None):
    """
    학습 때와 동일한 시드로 split 인덱스를 재현한다.
    class_filter: None이면 전체 9-class, 리스트 지정 시 해당 클래스만.
    """
    mat = loadmat(mat_path)
    y   = mat["y"].squeeze().astype(np.int64)

    classes = class_filter if class_filter is not None else list(range(1, 10))

    if split == "all":
        if class_filter is None:
            return list(range(len(y)))
        return [int(i) for i in np.where(np.isin(y, classes))[0]]

    train_idx, val_idx, test_idx = [], [], []
    for cls in classes:
        cls_trials = np.where(y == cls)[0]
        if len(cls_trials) == 0:
            continue
        rng = np.random.RandomState(seed + cls)
        rng.shuffle(cls_trials)
        n  = len(cls_trials)
        n_train = int(n * 0.8)
        n_val   = int(n * 0.1)
        train_idx.extend(cls_trials[:n_train].tolist())
        val_idx.extend(cls_trials[n_train:n_train + n_val].tolist())
        test_idx.extend(cls_trials[n_train + n_val:].tolist())

    return {"train": train_idx, "val": val_idx, "test": test_idx}[split]


# ─────────────────────────────────────────────────────────────────────────────
# 샘플링 함수
# ─────────────────────────────────────────────────────────────────────────────

def sample(args, subject_id: int, group_id: int = 0):
    """
    group_id : 0      → 9-class 전체 모델
               1,2,3  → 그룹 모델 (cls 1-3 / 4-6 / 7-9)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subj_str = f"{subject_id:02d}"
    mat_path = os.path.join(args.data_root, f"subj_{subj_str}.mat")
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"mat 파일 없음: {mat_path}")

    # 그룹 설정
    if group_id == 0:
        cls_list  = None
        cls_min   = 1
        group_tag = "9cls"
    else:
        cls_list  = GROUP_CLASSES[group_id]
        cls_min   = cls_list[0]
        group_tag = f"g{group_id}_cls{cls_list[0]}-{cls_list[-1]}"

    # ── 체크포인트 로드 ───────────────────────────────────────────────────────
    ckpt_path = args.ckpt_path
    if ckpt_path is None:
        ckpt_path = find_best_ckpt(args.ckpt_root, subj_tag=subj_str, group_tag=group_tag)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"체크포인트를 찾을 수 없습니다. --ckpt_path 로 직접 지정하거나 "
                f"--ckpt_root 를 확인하세요.\n"
                f"  탐색: *vs{subj_str}*{group_tag}*"
            )

    print(f"[INFO] 체크포인트: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get("config", {})

    # ── 모델 재구성 ───────────────────────────────────────────────────────────
    img_size     = cfg.get("img_size",      args.img_size)
    base_channels= cfg.get("base_channels", args.base_channels)
    n_ts         = cfg.get("num_timesteps", args.num_timesteps)
    n_res        = cfg.get("n_res_blocks",  args.n_res_blocks)
    ch_mult_str  = cfg.get("ch_mult",       args.ch_mult)
    eeg_ch       = cfg.get("eeg_ch",        args.eeg_ch)
    eeg_heads    = cfg.get("eeg_tf_heads",  args.eeg_tf_heads)
    eeg_layers   = cfg.get("eeg_tf_layers", args.eeg_tf_layers)
    eeg_dropout  = cfg.get("eeg_tf_dropout",args.eeg_tf_dropout)

    num_classes  = cfg.get("num_classes", 3 if group_id != 0 else 9)
    cfg_cls_min  = cfg.get("cls_min", cls_min)
    cfg_cls_list = cfg.get("cls_list", cls_list)
    if cfg_cls_list is not None:
        cfg_cls_list = list(cfg_cls_list)

    # 이미지 경로: cfg에 저장된 img_root 우선
    img_root = cfg.get("img_root", "")
    if not img_root or not os.path.isdir(img_root):
        img_root = args.img_root if args.img_root else find_img_root(
            args.data_root, fallback="./preproc_data_vi/images"
        )

    if isinstance(ch_mult_str, str):
        ch_mult = [int(x) for x in ch_mult_str.split(",")]
    else:
        ch_mult = list(ch_mult_str)

    if eeg_ch == 0:
        _mat = loadmat(mat_path)
        eeg_ch = int(_mat["X"].shape[0])

    print(
        f"[INFO] EEG 채널: {eeg_ch}, img_size: {img_size}, "
        f"num_timesteps: {n_ts}, num_classes: {num_classes}"
    )

    model = EEGDiffusionModel128(
        img_size=img_size,
        img_channels=3,
        eeg_channels=eeg_ch,
        num_classes=num_classes,
        num_timesteps=n_ts,
        base_channels=base_channels,
        ch_mult=ch_mult,
        time_dim=256,
        cond_dim=256,
        eeg_hidden_dim=256,
        cond_scale=2.0,
        n_res_blocks=n_res,
        eeg_tf_heads=eeg_heads,
        eeg_tf_layers=eeg_layers,
        eeg_tf_dropout=eeg_dropout,
    ).to(device)

    state = ckpt["ema"] if "ema" in ckpt else ckpt["model"]
    model.load_state_dict(state)
    model.eval()
    print("[INFO] 모델 로드 완료 (EMA 가중치 우선 사용)")

    # ── trial 인덱스 ──────────────────────────────────────────────────────────
    indices = get_split_indices(
        mat_path, split=args.split, seed=args.seed, class_filter=cfg_cls_list
    )
    if not indices:
        print("[WARN] 생성할 trial이 없습니다.")
        return

    print(f"[INFO] split={args.split}, trials={len(indices)}, group={group_tag}")

    ds = EEGInferenceDatasetVS(
        mat_path, img_root, indices,
        img_size=img_size, load_gt=(not args.no_gt), cls_list=cfg_cls_list,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # ── 출력 폴더 ────────────────────────────────────────────────────────────
    ckpt_dir  = os.path.dirname(ckpt_path)
    ckpt_name = os.path.basename(ckpt_dir)
    out_dir   = os.path.join(args.samples_root, ckpt_name, f"vs{subj_str}_{group_tag}_{args.split}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] 저장 폴더 → {out_dir}")

    to_pil = T.ToPILImage()
    total_gen = 0

    with torch.no_grad():
        for eeg, gt_img, label_0, label_1, trial_idx in loader:
            eeg = eeg.to(device)

            # 그룹 내 0-based 라벨
            label_group = (label_1 - cfg_cls_min).to(device)

            if args.use_ddim:
                gen = model.sample_ddim(
                    eeg=eeg,
                    labels=label_group,
                    num_steps=args.sample_steps,
                    guidance_scale=args.guidance_scale,
                    eta=args.ddim_eta,
                )
            else:
                gen = model.sample(
                    eeg=eeg,
                    labels=label_group,
                    num_steps=args.sample_steps,
                    guidance_scale=args.guidance_scale,
                )

            gen_denorm = denorm(gen).cpu()
            gt_denorm  = denorm(gt_img).cpu()

            for i in range(eeg.size(0)):
                lbl  = int(label_1[i].item())
                tidx = int(trial_idx[i].item())
                mode = "ddim" if args.use_ddim else "ddpm"

                gen_name = f"vs{subj_str}_trial{tidx:04d}_cls{lbl:02d}_{mode}_GEN.png"
                to_pil(gen_denorm[i]).save(os.path.join(out_dir, gen_name))

                if not args.no_gt:
                    gt_name = f"vs{subj_str}_trial{tidx:04d}_cls{lbl:02d}_GT.png"
                    to_pil(gt_denorm[i]).save(os.path.join(out_dir, gt_name))

                total_gen += 1

    suffix = "(GEN only)" if args.no_gt else "(GEN + GT)"
    print(f"[INFO] 생성 완료 — {total_gen}개 이미지 저장 {suffix}")
    print(f"[INFO] 결과 폴더: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VS REPA EEG→Image 생성 (preproc_for_gan_vs 학습 모델 추론)"
    )

    # 데이터
    parser.add_argument("--data_root",   type=str, default="./preproc_for_gan_vs",
                        help="VS mat 파일 폴더")
    parser.add_argument("--img_root",    type=str, default="",
                        help="클래스 이미지 폴더 (기본: data_root/images → preproc_data_vi/images)")
    parser.add_argument("--subject_ids", type=str, required=True,
                        help="예: '1' / '1-10' / '1,3,5' / 'all'")
    parser.add_argument("--split",       type=str, default="test",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--img_size",    type=int, default=128)

    # 체크포인트
    parser.add_argument("--ckpt_root",  type=str, default="./checkpoints_vs_repa",
                        help="자동 탐색할 루트 폴더 (train_vs_repa_allclass.py 출력)")
    parser.add_argument("--ckpt_path",  type=str, default=None,
                        help="체크포인트 파일 직접 지정 (우선 적용)")

    # 모델 구조 (cfg 없을 때 fallback)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--ch_mult",       type=str, default="1,2,4,4")
    parser.add_argument("--num_timesteps", type=int, default=200)
    parser.add_argument("--n_res_blocks",  type=int, default=2)
    parser.add_argument("--eeg_ch",        type=int, default=0,
                        help="EEG 채널 수 (0이면 mat에서 자동 감지)")
    parser.add_argument("--eeg_tf_heads",  type=int, default=4)
    parser.add_argument("--eeg_tf_layers", type=int, default=2)
    parser.add_argument("--eeg_tf_dropout",type=float, default=0.1)

    # 샘플링
    parser.add_argument("--use_ddim",       action="store_true", default=False)
    parser.add_argument("--sample_steps",   type=int, default=200)
    parser.add_argument("--guidance_scale", type=float, default=1.5,
                        help="학습 conditioning 강도=1.5(기본). 높일수록 클래스 선명도 증가(권장: 1.5~2.5)")
    parser.add_argument("--ddim_eta",       type=float, default=0.0)

    # 출력
    parser.add_argument("--samples_root", type=str, default="./samples_vs_repa")
    parser.add_argument("--no_gt",        action="store_true", default=False)
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--seed",         type=int, default=42)

    # 그룹
    parser.add_argument(
        "--group_ids", type=str, default="1,2,3",
        help="생성할 클래스 그룹. 예: '1' / '1,2,3' / 'all' / '0'(전체 9-class)"
    )

    args = parser.parse_args()
    set_seed(args.seed)

    subject_list = parse_subject_ids(args.subject_ids, args.data_root)

    gids_str = args.group_ids.strip()
    if gids_str == "0":
        group_list = [0]
    else:
        group_list = parse_group_ids(gids_str)

    print(f"[INFO] VS 데이터 경로: {args.data_root}")
    print(f"[INFO] 생성 대상 피험자: {subject_list}")
    print(f"[INFO] 그룹: {group_list}  (0=9-class 전체)")

    for sid in subject_list:
        mat_check = os.path.join(args.data_root, f"subj_{sid:02d}.mat")
        if not os.path.isfile(mat_check):
            print(f"[SKIP] subj_{sid:02d}.mat 없음 — 건너뜀")
            continue
        for gid in group_list:
            gtag = f"g{gid}" if gid != 0 else "9cls"
            print(f"\n{'='*60}")
            print(f"  VS Subject {sid:02d}  [{gtag}]")
            print(f"{'='*60}")
            try:
                sample(args, subject_id=sid, group_id=gid)
            except Exception as e:
                print(f"[ERROR] vs subj {sid:02d} [{gtag}]: {e}")


if __name__ == "__main__":
    main()
