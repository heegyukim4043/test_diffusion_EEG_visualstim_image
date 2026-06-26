"""
train_crosssubj_dino.py
─────────────────────────────────────────────────────────────────────────────
Stage 1: EEG → DINO 512-dim shared latent, cross-subject 학습

9-class 전체로 학습 (그룹 분리 없음)
  → 갤러리 = 9개 이미지 → top-1/3/5 모두 의미 있음
  → 3-class 분리 시 loss=ln(3) 고착 문제 해결

두 가지 학습 모드
─────────────────
  within : 20명 전체 학습, 동일 피험자 held-out trial 테스트
  loso   : leave-one-subject-out (20회, cross-subject 일반화)

사용 예시
─────────
  # 9-class, within
  python train_crosssubj_dino.py --mode within

  # 9-class, LOSO
  python train_crosssubj_dino.py --mode loso

  # 특정 그룹만 (3-class, 비교 실험용)
  python train_crosssubj_dino.py --mode within --group_id 1
"""

import os, math, random, argparse, datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

from model_eeg_dino import EEGDINORegressor, load_dino_encoder, DINO_DIM, LATENT_DIM


# ── 재현성 ────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


GROUP_CLASSES = {0: list(range(1,10)), 1:[1,2,3], 2:[4,5,6], 3:[7,8,9]}


# ── EEG Augmentation ──────────────────────────────────────────────────────
class EEGAugment:
    """
    On-the-fly EEG augmentation (학습 시 배치마다 랜덤 적용).

    적용 순서 (각각 독립적 확률):
      1. Gaussian noise    : SNR 저하 시뮬레이션
      2. Amplitude scaling : 피험자 간 진폭 차이 시뮬레이션
      3. Channel dropout   : 전극 불량 시뮬레이션
      4. Time shift        : 자극 반응 latency 개인차 시뮬레이션
      5. Freq-domain noise : 주파수 대역별 노이즈 (FFT 진폭 perturbation)
    """

    def __init__(
        self,
        noise_std:      float = 0.05,   # Gaussian noise 강도 (신호 std 대비)
        scale_range:    tuple = (0.8, 1.2),
        ch_drop_prob:   float = 0.1,    # 채널별 dropout 확률
        max_shift:      int   = 25,     # 최대 time shift (samples, ~50ms @512Hz)
        freq_noise_std: float = 0.02,   # FFT 진폭 noise 강도
        p_noise:  float = 0.5,
        p_scale:  float = 0.5,
        p_drop:   float = 0.3,
        p_shift:  float = 0.3,
        p_freq:   float = 0.3,
    ):
        self.noise_std      = noise_std
        self.scale_range    = scale_range
        self.ch_drop_prob   = ch_drop_prob
        self.max_shift      = max_shift
        self.freq_noise_std = freq_noise_std
        self.p_noise  = p_noise
        self.p_scale  = p_scale
        self.p_drop   = p_drop
        self.p_shift  = p_shift
        self.p_freq   = p_freq

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        """eeg: (B, ch, T)"""
        x = eeg.clone()

        # 1. Gaussian noise
        if random.random() < self.p_noise:
            sig_std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
            x = x + torch.randn_like(x) * sig_std * self.noise_std

        # 2. Amplitude scaling (per-sample)
        if random.random() < self.p_scale:
            lo, hi = self.scale_range
            scale = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(lo, hi)
            x = x * scale

        # 3. Channel dropout (같은 배치 내 채널별 독립)
        if random.random() < self.p_drop:
            mask = (torch.rand(x.size(0), x.size(1), 1, device=x.device)
                    > self.ch_drop_prob).float()
            x = x * mask

        # 4. Time shift (circular)
        if random.random() < self.p_shift:
            shift = random.randint(-self.max_shift, self.max_shift)
            if shift != 0:
                x = torch.roll(x, shift, dims=-1)

        # 5. Frequency-domain noise (FFT 진폭 perturbation)
        # CPU에서 실행 → nvrtc JIT 컴파일 오류 회피
        if random.random() < self.p_freq:
            orig_device = x.device
            xc  = x.cpu()
            xf  = torch.fft.rfft(xc, dim=-1)
            amp = xf.abs()
            noise = torch.randn_like(amp) * amp * self.freq_noise_std
            xf  = xf + noise * torch.exp(
                1j * torch.rand_like(amp) * 2 * math.pi
            )
            x = torch.fft.irfft(xf, n=eeg.size(-1), dim=-1).to(orig_device)

        return x


dino_transform = T.Compose([
    T.Resize(224), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────
class EEGDataset(Dataset):
    def __init__(self, samples):
        # samples: [(eeg_tensor, subj_idx, label_0based, label_1based)]
        self.samples = samples

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collate_fn(batch):
    eeg   = torch.stack([b[0] for b in batch])
    subj  = torch.tensor([b[1] for b in batch], dtype=torch.long)
    lbl0  = torch.tensor([b[2] for b in batch], dtype=torch.long)
    lbl1  = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return eeg, subj, lbl0, lbl1


def build_dataset(data_root, subject_ids, cls_list, split, subj_map, seed=42):
    """subject_ids 의 EEG를 split에 맞게 로드."""
    cls_min  = cls_list[0]
    samples  = []
    for sid in subject_ids:
        mat_path = os.path.join(data_root, f"subj_{sid:02d}.mat")
        if not os.path.isfile(mat_path): continue
        mat = loadmat(mat_path)
        X   = mat["X"]
        y   = mat["y"].squeeze().astype(np.int64)
        eeg = torch.from_numpy(X.astype(np.float32)).permute(2,0,1)
        si  = subj_map[sid]
        for cls in cls_list:
            trials = np.where(y == cls)[0]
            if len(trials) == 0: continue
            rng = np.random.RandomState(seed + si * 100 + cls)
            rng.shuffle(trials)
            n       = len(trials)
            n_train = int(n * 0.8)
            n_val   = int(n * 0.1)
            if split == "train": sel = trials[:n_train]
            elif split == "val": sel = trials[n_train:n_train+n_val]
            else:                sel = trials[n_train+n_val:]
            for t in sel:
                samples.append((eeg[t], si, cls - cls_min, int(cls)))
    return EEGDataset(samples)


# ── DINO feature 계산 ────────────────────────────────────────────────────
@torch.no_grad()
def compute_class_prototypes(dino, img_root, cls_list, device):
    """각 클래스 이미지의 DINO feature. Returns (n_cls, dino_dim)."""
    import torch.nn.functional as F
    feats = []
    for cls in cls_list:
        p = os.path.join(img_root, f"{cls:02d}.png")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"이미지 없음: {p}")
        img = dino_transform(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        f   = F.normalize(dino(img), dim=1)
        feats.append(f)
    return torch.cat(feats, dim=0)   # (n_cls, dino_dim)


# ── top-k retrieval ────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_retrieval(model, loader, proto_dino, device, k_list=(1,3,5)):
    correct = {k: 0 for k in k_list}
    total   = 0
    for eeg, subj, lbl0, _ in loader:
        eeg   = eeg.to(device)
        subj  = subj.to(device)
        lbl0  = lbl0.to(device)
        logits, _, _ = model.predict(eeg, subj, proto_dino)
        for k in k_list:
            topk = logits.topk(min(k, logits.size(1)), dim=1).indices
            correct[k] += topk.eq(lbl0.unsqueeze(1)).any(1).sum().item()
        total += eeg.size(0)
    return {k: correct[k]/total for k in k_list} if total > 0 else {k: 0.0 for k in k_list}


# ── 단일 학습 실행 ────────────────────────────────────────────────────────
def run_training(
    args, train_subjs, val_subjs, test_subjs,
    subj_map, n_subjects, cls_list, dino, proto_dino, dino_feat_dim,
    eeg_ch, device, save_dir, tag
):
    cls_min   = cls_list[0]
    n_classes = len(cls_list)

    train_ds = build_dataset(args.data_root, train_subjs, cls_list, "train", subj_map, args.seed)
    val_ds   = build_dataset(args.data_root, val_subjs,   cls_list, "val",   subj_map, args.seed)
    test_ds  = build_dataset(args.data_root, test_subjs,  cls_list, "test",  subj_map, args.seed)

    if len(train_ds) == 0:
        print(f"  [SKIP] {tag}: train 데이터 없음")
        return None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    model = EEGDINORegressor(
        eeg_channels=eeg_ch, n_subjects=n_subjects,
        dino_feat_dim=dino_feat_dim, latent_dim=LATENT_DIM,
        eeg_hidden=args.eeg_hidden, eeg_out=args.eeg_out,
        subj_emb_dim=args.subj_emb_dim,
        n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout, temperature=args.temperature,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Warmup + Cosine decay
    warmup_epochs = max(1, args.epochs // 10)
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return ep / warmup_epochs          # linear warmup
        progress = (ep - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    augmenter = EEGAugment(
        noise_std=0.05, scale_range=(0.8,1.2),
        ch_drop_prob=0.1, max_shift=25, freq_noise_std=0.02,
        p_noise=0.0 if args.no_aug else 0.5,
        p_scale=0.0 if args.no_aug else 0.5,
        p_drop =0.0 if args.no_aug else 0.3,
        p_shift=0.0 if args.no_aug else 0.3,
        p_freq =0.0 if args.no_aug else 0.3,
    )

    best_val_top1 = 0.0
    ckpt_path     = os.path.join(save_dir, f"{tag}_best.pt")

    print(f"\n  {tag}  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"  {'Ep':>4}  {'Loss':>7}  {'Lcos':>6}  {'Lce':>6}  {'TrAcc':>6}  {'ValT1':>6}  {'ValT3':>6}  {'ValT5':>6}  {'temp':>6}")
    print(f"  {'-'*62}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = ep_correct = ep_total = 0

        ep_lcos = ep_lce = 0.0
        for eeg, subj, lbl0, _ in train_loader:
            eeg  = augmenter(eeg.to(device))   # augmentation 적용
            subj = subj.to(device)
            lbl0 = lbl0.to(device)
            tgt  = proto_dino[lbl0]   # (B, dino_dim)

            loss, lc, lp, li, acc = model.compute_loss(
                eeg, subj, tgt, proto_dino, lbl0,
                use_infonce=(args.use_infonce and not args.no_infonce),
                w_cos=args.w_cos,
                w_proto=args.w_proto,
                w_infonce=args.w_infonce,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            b = eeg.size(0)
            ep_loss    += loss.item() * b
            ep_lcos    += lc.item() * b
            ep_lce     += lp.item() * b
            ep_correct += int((acc * b).round())
            ep_total   += b

        scheduler.step()
        tr_loss = ep_loss / ep_total
        tr_lcos = ep_lcos / ep_total
        tr_lce  = ep_lce  / ep_total
        tr_acc  = ep_correct / ep_total
        cur_temp = model.log_temp.exp().item()

        model.eval()
        val_k = evaluate_retrieval(model, val_loader, proto_dino, device, k_list=(1,3,5))

        if val_k[1] >= best_val_top1:
            best_val_top1 = val_k[1]
            torch.save({
                "model": model.state_dict(),
                "config": {
                    "eeg_ch": eeg_ch, "n_subjects": n_subjects,
                    "dino_feat_dim": dino_feat_dim,
                    "dino_model": args.dino_model,
                    "eeg_hidden": args.eeg_hidden, "eeg_out": args.eeg_out,
                    "subj_emb_dim": args.subj_emb_dim,
                    "n_heads": args.n_heads, "n_layers": args.n_layers,
                    "dropout": args.dropout, "temperature": args.temperature,
                    "group_id": args.group_id, "cls_list": cls_list,
                    "subj_map": subj_map,
                },
            }, ckpt_path)

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"  {epoch:>4}  {tr_loss:>7.4f}  {tr_lcos:>6.3f}  {tr_lce:>6.3f}"
                  f"  {tr_acc:>6.3f}  {val_k[1]:>6.3f}  {val_k[3]:>6.3f}  {val_k[5]:>6.3f}"
                  f"  {cur_temp:>6.3f}")

    # Test
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    test_k = evaluate_retrieval(model, test_loader, proto_dino, device, k_list=(1,3,5))
    print(f"\n  [Test] Top-1={test_k[1]:.4f}  Top-3={test_k[3]:.4f}  Top-5={test_k[5]:.4f}"
          f"  (random={1/n_classes:.4f})")
    return test_k


# ── main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   type=str,   default="./preproc_for_gan_vs")
    parser.add_argument("--img_root",    type=str,   default="")
    parser.add_argument("--subject_ids", type=str,   default="1-20")
    parser.add_argument("--group_id",    type=int,   default=0, choices=[0,1,2,3],
                        help="0=9-class 전체(기본), 1=cls1-3, 2=cls4-6, 3=cls7-9")
    parser.add_argument("--mode",        type=str,   default="within",
                        choices=["within","loso"],
                        help="within: 전체 학습+held-out 평가 / loso: leave-one-subject-out")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--dino_model",  type=str,   default="dinov2_vits14",
                        choices=list(DINO_DIM.keys()))
    parser.add_argument("--eeg_hidden",  type=int,   default=256)
    parser.add_argument("--eeg_out",     type=int,   default=256)
    parser.add_argument("--subj_emb_dim",type=int,   default=64)
    parser.add_argument("--n_heads",     type=int,   default=4)
    parser.add_argument("--n_layers",    type=int,   default=4)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--wd",          type=float, default=1e-4)
    parser.add_argument("--lambda_align",type=float, default=1.0)
    parser.add_argument("--use_infonce",  action="store_true", default=True)
    parser.add_argument("--no_infonce",   action="store_true", default=False,
                        help="InfoNCE 비활성화 (cosine+proto만)")
    parser.add_argument("--w_cos",        type=float, default=1.0)
    parser.add_argument("--w_proto",      type=float, default=1.0)
    parser.add_argument("--w_infonce",    type=float, default=1.0)
    parser.add_argument("--no_aug",       action="store_true", default=False,
                        help="EEG augmentation 비활성화")
    parser.add_argument("--ckpt_root",    type=str,   default="./checkpoints_dino")

    args  = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}  Mode: {args.mode}")

    cls_list  = GROUP_CLASSES[args.group_id]
    group_tag = "9cls" if args.group_id == 0 else f"g{args.group_id}_cls{cls_list[0]}-{cls_list[-1]}"

    # subject_ids 파싱
    subject_ids = []
    for tok in args.subject_ids.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            subject_ids.extend(range(int(a), int(b)+1))
        else:
            subject_ids.append(int(tok))
    subject_ids = [s for s in subject_ids
                   if os.path.isfile(os.path.join(args.data_root, f"subj_{s:02d}.mat"))]
    n_subjects  = len(subject_ids)
    subj_map    = {sid: i for i, sid in enumerate(subject_ids)}
    print(f"[INFO] 그룹: {group_tag}  피험자: {n_subjects}명")

    # EEG 채널 감지
    eeg_ch = int(loadmat(
        os.path.join(args.data_root, f"subj_{subject_ids[0]:02d}.mat")
    )["X"].shape[0])

    # DINO & prototypes
    print(f"[INFO] DINO 로드: {args.dino_model}")
    dino          = load_dino_encoder(args.dino_model, device)
    dino_feat_dim = DINO_DIM[args.dino_model]
    img_root      = args.img_root or "./preproc_data_vi/images"
    proto_dino    = compute_class_prototypes(dino, img_root, cls_list, device)
    print(f"  proto shape: {proto_dino.shape}  (n_cls × {dino_feat_dim})")

    # 저장 경로
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.ckpt_root, f"{ts}_{args.mode}_{group_tag}")
    os.makedirs(save_dir, exist_ok=True)

    # ── Within-subject 모드 ───────────────────────────────────────────────
    if args.mode == "within":
        print(f"\n[Within-subject]  전체 {n_subjects}명 학습, held-out trial 평가")
        result = run_training(
            args,
            train_subjs=subject_ids, val_subjs=subject_ids, test_subjs=subject_ids,
            subj_map=subj_map, n_subjects=n_subjects,
            cls_list=cls_list, dino=dino, proto_dino=proto_dino,
            dino_feat_dim=dino_feat_dim, eeg_ch=eeg_ch,
            device=device, save_dir=save_dir, tag="within",
        )
        if result:
            import csv
            with open(os.path.join(save_dir, "results.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["mode","group","top1","top3","top5","random"])
                w.writerow(["within", group_tag,
                            round(result[1],4), round(result[3],4), round(result[5],4),
                            round(1/len(cls_list),4)])

    # ── LOSO 모드 ────────────────────────────────────────────────────────
    else:
        print(f"\n[LOSO]  {n_subjects}회 반복 (각 피험자 1명씩 test)")
        loso_results = {}

        for test_sid in subject_ids:
            train_sids = [s for s in subject_ids if s != test_sid]
            # LOSO: test subject의 subject embedding은 학습 중 본 적 없음
            # → 가장 가까운 embedding 사용 or zero로 처리
            # 여기서는 test subject를 n_subjects번째 slot에 배치
            loso_map      = {sid: i for i, sid in enumerate(train_sids)}
            loso_map[test_sid] = len(train_sids)   # 추가 slot
            loso_n_subj   = n_subjects              # 동일 크기 유지

            tag = f"loso_test{test_sid:02d}"
            result = run_training(
                args,
                train_subjs=train_sids, val_subjs=train_sids, test_subjs=[test_sid],
                subj_map=loso_map, n_subjects=loso_n_subj,
                cls_list=cls_list, dino=dino, proto_dino=proto_dino,
                dino_feat_dim=dino_feat_dim, eeg_ch=eeg_ch,
                device=device, save_dir=save_dir, tag=tag,
            )
            loso_results[test_sid] = result

        # 요약 출력 및 CSV 저장
        print(f"\n{'='*55}")
        print(f"  LOSO 결과 요약  [{group_tag}]")
        print(f"  {'Subj':>6}  {'Top-1':>7}  {'Top-3':>7}  {'Top-5':>7}")
        print(f"  {'-'*35}")
        import csv
        rows = []
        for sid, r in loso_results.items():
            if r is None: continue
            print(f"  S{sid:02d}     {r[1]:>7.4f}  {r[3]:>7.4f}  {r[5]:>7.4f}")
            rows.append([f"S{sid:02d}", round(r[1],4), round(r[3],4), round(r[5],4)])

        t1s = [r[1] for r in loso_results.values() if r]
        t3s = [r[3] for r in loso_results.values() if r]
        t5s = [r[5] for r in loso_results.values() if r]
        print(f"  {'-'*35}")
        print(f"  {'Mean':>6}  {np.mean(t1s):>7.4f}  {np.mean(t3s):>7.4f}  {np.mean(t5s):>7.4f}")
        print(f"  Random: {1/len(cls_list):.4f}")

        with open(os.path.join(save_dir, "results_loso.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subject","top1","top3","top5"])
            w.writerows(rows)
            w.writerow(["Mean",round(np.mean(t1s),4),round(np.mean(t3s),4),round(np.mean(t5s),4)])
            w.writerow(["Random",round(1/len(cls_list),4),"",""])

    print(f"\n[INFO] 저장 완료: {save_dir}")


if __name__ == "__main__":
    main()
