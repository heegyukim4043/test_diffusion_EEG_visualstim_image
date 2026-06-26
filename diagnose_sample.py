"""
diagnose_sample.py
체크포인트가 실제로 올바르게 로드되고 EEG conditioning이 작동하는지 진단.
"""
import os, sys
import numpy as np
import torch
import torchvision.transforms as T
from scipy.io import loadmat
from PIL import Image

# ── 설정 ─────────────────────────────────────────────────────────────
CKPT_ROOT  = "./checkpoints_vs_repa"
DATA_ROOT  = "./preproc_for_gan_vs"
IMG_ROOT   = "./preproc_data_vi/images"
SUBJECT_ID = 1
GROUP_ID   = 1          # g1 = cls 1-3
CLS_MIN    = 1
NUM_CLASSES= 3

import glob
dirs = sorted(glob.glob(os.path.join(CKPT_ROOT, f"*vs{SUBJECT_ID:02d}_g{GROUP_ID}*")))
if not dirs:
    print(f"[ERROR] 체크포인트 없음: *vs{SUBJECT_ID:02d}_g{GROUP_ID}*")
    sys.exit(1)
ckpt_path = os.path.join(dirs[-1], "best.pt")
print(f"[CKPT] {ckpt_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {device}")

# ── 1. 체크포인트 키 확인 ─────────────────────────────────────────────
ckpt = torch.load(ckpt_path, map_location="cpu")
print(f"\n[1] 체크포인트 키: {list(ckpt.keys())}")
cfg = ckpt.get("config", {})
print(f"    config: {cfg}")

# ── 2. 모델 로드 ──────────────────────────────────────────────────────
from model_128_eegonly_transformer_repa import EEGDiffusionModel128

img_size      = cfg.get("img_size", 128)
base_channels = cfg.get("base_channels", 64)
n_ts          = cfg.get("num_timesteps", 200)
n_res         = cfg.get("n_res_blocks", 2)
ch_mult_str   = cfg.get("ch_mult", "1,2,4,4")
eeg_heads     = cfg.get("eeg_tf_heads", 4)
eeg_layers    = cfg.get("eeg_tf_layers", 2)
eeg_dropout   = cfg.get("eeg_tf_dropout", 0.1)

mat = loadmat(os.path.join(DATA_ROOT, f"subj_{SUBJECT_ID:02d}.mat"))
eeg_ch = int(mat["X"].shape[0])
ch_mult = [int(x) for x in ch_mult_str.split(",")]

model = EEGDiffusionModel128(
    img_size=img_size, img_channels=3, eeg_channels=eeg_ch,
    num_classes=NUM_CLASSES, num_timesteps=n_ts,
    base_channels=base_channels, ch_mult=ch_mult,
    time_dim=256, cond_dim=256, eeg_hidden_dim=256, cond_scale=2.0,
    n_res_blocks=n_res, eeg_tf_heads=eeg_heads, eeg_tf_layers=eeg_layers,
    eeg_tf_dropout=eeg_dropout,
).to(device)

state_key = "ema" if "ema" in ckpt else "model"
model.load_state_dict(ckpt[state_key])
model.eval()
print(f"\n[2] 모델 로드 완료 (키: {state_key})")
n_params = sum(p.numel() for p in model.parameters())
print(f"    파라미터 수: {n_params:,}")

# ── 3. EEG 로드 및 통계 ──────────────────────────────────────────────
X = mat["X"]   # (ch, time, trial)
y = mat["y"].squeeze().astype(np.int64)
cls1_trials = np.where(y == 1)[0]
cls2_trials = np.where(y == 2)[0]

eeg_all = torch.from_numpy(X.astype(np.float32)).permute(2, 0, 1)  # (trial, ch, time)
print(f"\n[3] EEG 통계")
print(f"    shape: {eeg_all.shape}")
print(f"    mean={eeg_all.mean():.4f}, std={eeg_all.std():.4f}")
print(f"    min={eeg_all.min():.4f}, max={eeg_all.max():.4f}")

# ── 4. EEG embedding 다양성 확인 ─────────────────────────────────────
with torch.no_grad():
    eeg_batch = eeg_all[:6].to(device)
    labels    = torch.zeros(6, dtype=torch.long, device=device)
    emb = model.get_cond_emb_eeg_only(eeg_batch)
    print(f"\n[4] EEG embedding 통계")
    print(f"    shape: {emb.shape}")
    print(f"    mean={emb.mean():.4f}, std={emb.std():.4f}")
    print(f"    각 샘플 L2 norm: {emb.norm(dim=1).cpu().numpy().round(3)}")

    # cls1 vs cls2 임베딩 유사도
    if len(cls1_trials) >= 2 and len(cls2_trials) >= 2:
        e1 = model.get_cond_emb_eeg_only(eeg_all[cls1_trials[:3]].to(device))
        e2 = model.get_cond_emb_eeg_only(eeg_all[cls2_trials[:3]].to(device))
        cos_within = torch.nn.functional.cosine_similarity(e1[0:1], e1[1:2]).item()
        cos_cross  = torch.nn.functional.cosine_similarity(e1[0:1], e2[0:1]).item()
        print(f"    같은 클래스 cos 유사도: {cos_within:.4f}")
        print(f"    다른 클래스 cos 유사도: {cos_cross:.4f}")
        print(f"    → 두 값이 비슷하면 EEG encoder가 클래스 구분 못하는 것")

# ── 5. 단일 샘플 생성 (t=5 짧은 DDIM) ──────────────────────────────
print(f"\n[5] 빠른 샘플 생성 (DDIM 10스텝)")
with torch.no_grad():
    eeg1   = eeg_all[cls1_trials[0]:cls1_trials[0]+1].to(device)
    lbl1   = torch.tensor([0], device=device)
    gen    = model.sample_ddim(eeg1, labels=lbl1, num_steps=10, guidance_scale=1.5)
    gen_np = ((gen.squeeze().permute(1,2,0).cpu().numpy() + 1) * 127.5).clip(0,255).astype(np.uint8)
    out = Image.fromarray(gen_np)
    out.save("./diagnose_gen_cls1.png")
    print(f"    저장: ./diagnose_gen_cls1.png")
    print(f"    pixel 통계: mean={gen_np.mean():.1f}, std={gen_np.std():.1f}")
    print(f"    → std가 낮으면(<30) 단색/흐림, 높으면(>80) 노이즈")

# ── 6. zero conditioning 비교 ────────────────────────────────────────
print(f"\n[6] Zero conditioning으로 생성 (EEG 무시)")
with torch.no_grad():
    zero_cond = torch.zeros(1, 256, device=device)
    x_t = torch.randn(1, 3, img_size, img_size, device=device)
    for i in reversed(np.linspace(0, n_ts-1, 10, dtype=int)):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        t_emb = model.time_embed(t)
        eps = model.unet(x_t, t_emb, zero_cond)
        ab = model.alphas_cumprod[i]
        x0 = (x_t - (1-ab).sqrt() * eps) / (ab.sqrt() + 1e-8)
        x0 = x0.clamp(-1,1)
        ab_prev = model.alphas_cumprod[max(i-1,0)] if i > 0 else torch.tensor(1.0, device=device)
        x_t = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    gen0_np = ((x_t.squeeze().permute(1,2,0).cpu().numpy()+1)*127.5).clip(0,255).astype(np.uint8)
    Image.fromarray(gen0_np).save("./diagnose_gen_zero.png")
    print(f"    저장: ./diagnose_gen_zero.png")
    print(f"    pixel 통계: mean={gen0_np.mean():.1f}, std={gen0_np.std():.1f}")

print("\n[완료] diagnose_gen_cls1.png 과 diagnose_gen_zero.png 를 비교해주세요.")
print("      두 이미지가 비슷하면 → EEG conditioning이 실제로 작동 안 하는 것")
print("      두 이미지가 다르면 → EEG conditioning 작동, 품질 문제만 있는 것")
