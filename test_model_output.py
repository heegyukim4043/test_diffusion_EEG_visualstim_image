import os
import sys
import argparse
import yaml
import torch
from einops import rearrange

import glob
import numpy as np
import scipy.io as sio

from PIL import Image

# ---------------------------------------------------------
# 1) 패키지 경로 설정: code/dc_ldm 을 import 할 수 있게 해줌
# ---------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(ROOT, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# 이제 dc_ldm import 가능
from dc_ldm.util import instantiate_from_config

# (옵션) 이미지 저장용 헬퍼
def save_grid(grid_np, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = Image.fromarray(grid_np.astype("uint8"))
    img.save(out_path)
    print(f"[info] saved: {out_path}")


@torch.no_grad()
def load_model(config_path, ckpt_path, device="cuda"):
    print("[info] building model from config...")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = instantiate_from_config(cfg["model"])
    model = model.to(device)

    print(f"[info] loading checkpoint: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")

    # v1-5-pruned.ckpt 형식 & 우리 학습 체크포인트 둘 다 커버
    if "state_dict" in sd:
        state = sd["state_dict"]
    elif "model_state_dict" in sd:
        state = sd["model_state_dict"]
    else:
        state = sd

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[info] loaded with {len(missing)} missing and {len(unexpected)} unexpected keys")
    if missing:
        print("  missing:", missing[:10])
    if unexpected:
        print("  unexpected:", unexpected[:10])

    model.eval()
    return model


def load_test_data(root_path):
    """
    root_path: ./preproc_data
    - subj_XX.mat 안에 'eeg' 또는 'X' 만 사용
    - 이미지는 ../images/01.png ~ 09.png 를 순환하면서 GT로 사용
    """

    root_path = os.path.abspath(root_path)
    proj_root = os.path.dirname(root_path)       # .../DreamDiffusion-main/DreamDiffusion-main
    image_root = os.path.join(proj_root, "images")

    # ---- 이미지 9장 로드 (01.png ~ 09.png) ----
    base_imgs = []
    for cls in range(1, 10):
        img_path = os.path.join(image_root, f"{cls:02d}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"image not found: {img_path}")
        img = Image.open(img_path).convert("RGB").resize((512, 512))
        img_np = np.array(img).astype(np.float32)
        # [0,255] -> [-1,1]
        img_np = img_np / 127.5 - 1.0          # (H,W,3)
        img_np = np.transpose(img_np, (2, 0, 1))  # (3,H,W)
        base_imgs.append(img_np)

    eeg_list = []
    img_list = []

    mat_files = sorted(glob.glob(os.path.join(root_path, "subj_*.mat")))
    if len(mat_files) == 0:
        raise FileNotFoundError(f"no subj_*.mat in {root_path}")

    for mf in mat_files:
        print(f"[info] loading {mf}")
        mat = sio.loadmat(mf)

        # ---- EEG 가져오기 ----
        if "eeg" in mat:
            eeg = mat["eeg"]
        elif "X" in mat:
            eeg = mat["X"]
        else:
            raise KeyError("mat file must contain 'eeg' or 'X'")

        eeg = np.asarray(eeg)
        if eeg.ndim != 3:
            raise ValueError(f"eeg must be 3D, got {eeg.shape}")

        # 축 정렬: (B, 512, F) 형태로 맞춤
        if eeg.shape[1] == 512:
            pass
        elif eeg.shape[0] == 512:
            eeg = np.transpose(eeg, (1, 0, 2))
        elif eeg.shape[2] == 512:
            eeg = np.transpose(eeg, (0, 2, 1))
        else:
            raise ValueError(f"512-length time axis not found in {eeg.shape}")

        # 이제 eeg.shape = (B_i, 512, F)
        B_i = eeg.shape[0]
        eeg_list.append(eeg)

        # 이 subject의 샘플 수만큼 이미지를 01~09 순환하면서 붙임
        for n in range(B_i):
            img_list.append(base_imgs[len(img_list) % len(base_imgs)])

    eeg_arr = np.concatenate(eeg_list, axis=0).astype(np.float32)   # (N,512,F)
    img_arr = np.stack(img_list, axis=0).astype(np.float32)         # (N,3,512,512)

    eeg_t = torch.from_numpy(eeg_arr)
    img_t = torch.from_numpy(img_arr)

    print("[info] final eeg shape :", eeg_t.shape)
    print("[info] final image shape:", img_t.shape)

    return {
        "eeg": eeg_t,
        "image": img_t,
    }



@torch.no_grad()
def main(
    config,
    ckpt,
    root_path,
    batch_size=4,
    num_samples=1,
    ddim_steps=25,
    outdir="./eeg_ldm_infer_out",
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(config, ckpt, device=device)

    # 테스트 데이터 로드
    data = load_test_data(root_path)
    # generate() 안에서 .to(self.device) 를 호출하므로 여기선 CPU여도 OKs

    # -------------------------------------------------
    # 모델 generate 호출
    # (ddpm.LatentDiffusion.generate 정의에 맞춤)
    # -------------------------------------------------
    print(f"[info] start generation: num_samples={num_samples}, ddim_steps={ddim_steps}")
    grid, all_samples, _ = model.generate(
        data,
        num_samples=num_samples,
        ddim_steps=ddim_steps,
        HW=None,
        limit=None,
        state=None,
    )

    # grid (하나의 큰 이미지) 저장
    os.makedirs(outdir, exist_ok=True)
    grid_path = os.path.join(outdir, "grid.png")
    save_grid(grid, grid_path)

    # 개별 샘플도 저장 (gt + 생성)
    # all_samples: shape (N, num_samples+1, 3, H, W), 첫 번째가 GT
    for i, imgs in enumerate(all_samples):
        gt = imgs[0]
        gen = imgs[1]  # num_samples=1 이라면 하나뿐

        gt_img = Image.fromarray(rearrange(gt, "c h w -> h w c").astype("uint8"))
        gen_img = Image.fromarray(rearrange(gen, "c h w -> h w c").astype("uint8"))

        gt_path = os.path.join(outdir, f"sample_{i:04d}_gt.png")
        gen_path = os.path.join(outdir, f"sample_{i:04d}_gen.png")
        gt_img.save(gt_path)
        gen_img.save(gen_path)

    print("[done] inference finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--ddim_steps", type=int, default=25)
    parser.add_argument("--outdir", type=str, default="./eeg_ldm_infer_out")
    args = parser.parse_args()

    main(
        config=args.config,
        ckpt=args.ckpt,
        root_path=args.root_path,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        ddim_steps=args.ddim_steps,
        outdir=args.outdir,
    )
