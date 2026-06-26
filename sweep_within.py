"""
sweep_within.py
───────────────────────────────────────────────────────────────────
within-subject 모드에서 하이퍼파라미터 sweep (순차 실행).

실험 그룹:
  A. w_infonce sweep  : 0.5 / 1.0 / 2.0   (w_proto=1.0, aug=default)
  B. w_proto  sweep   : 1.0 / 1.5 / 2.0   (w_infonce=1.0, aug=default)
  C. augmentation     : default / weak / noisy3off
       weak      : ch_drop_prob·p_drop·p_shift·p_freq 각 0.5배
       noisy3off : channel-dropout, time-shift, freq-noise 끔 (noise+scale만)

결과는 sweep_results.csv 에 저장.
사용:
  python sweep_within.py
  python sweep_within.py --epochs 100   # 빠른 프리뷰
"""

import os, sys, csv, argparse, datetime, math, random, copy
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

# ── 경로 추가 ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from model_eeg_dino import EEGDINORegressor, load_dino_encoder, DINO_DIM, LATENT_DIM

# train_crosssubj_dino 에서 재사용할 함수들
from train_crosssubj_dino import (
    set_seed, GROUP_CLASSES, EEGAugment, dino_transform,
    EEGDataset, collate_fn, build_dataset,
    compute_class_prototypes, evaluate_retrieval,
)


# ── 실험 정의 ──────────────────────────────────────────────────────────────
def make_experiments():
    base = dict(
        w_cos=1.0, w_proto=1.0, w_infonce=1.0,
        aug="default",
    )

    exps = []

    # A. w_infonce sweep
    for wi in [0.5, 1.0, 2.0]:
        e = copy.copy(base)
        e["name"]      = f"infonce_{wi}"
        e["group"]     = "A_infonce"
        e["w_infonce"] = wi
        exps.append(e)

    # B. w_proto sweep
    for wp in [1.0, 1.5, 2.0]:
        if wp == 1.0:          # 1.0 은 A 와 동일 → 재사용 태그만
            e = copy.copy(base)
            e["name"]    = f"proto_{wp}"
            e["group"]   = "B_proto"
            e["w_proto"] = wp
        else:
            e = copy.copy(base)
            e["name"]    = f"proto_{wp}"
            e["group"]   = "B_proto"
            e["w_proto"] = wp
        exps.append(e)

    # C. augmentation ablation
    for aug_tag in ["default", "weak", "noisy3off"]:
        e = copy.copy(base)
        e["name"] = f"aug_{aug_tag}"
        e["group"] = "C_aug"
        e["aug"]   = aug_tag
        exps.append(e)

    return exps


def build_augmenter(aug_tag):
    if aug_tag == "default":
        return EEGAugment(
            noise_std=0.05, scale_range=(0.8,1.2),
            ch_drop_prob=0.1, max_shift=25, freq_noise_std=0.02,
            p_noise=0.5, p_scale=0.5, p_drop=0.3, p_shift=0.3, p_freq=0.3,
        )
    elif aug_tag == "weak":
        # 각 확률 절반, 강도도 절반
        return EEGAugment(
            noise_std=0.025, scale_range=(0.9,1.1),
            ch_drop_prob=0.05, max_shift=12, freq_noise_std=0.01,
            p_noise=0.25, p_scale=0.25, p_drop=0.15, p_shift=0.15, p_freq=0.15,
        )
    elif aug_tag == "noisy3off":
        # channel-dropout / time-shift / freq-noise 끔 → noise + scale 만
        return EEGAugment(
            noise_std=0.05, scale_range=(0.8,1.2),
            ch_drop_prob=0.1, max_shift=25, freq_noise_std=0.02,
            p_noise=0.5, p_scale=0.5, p_drop=0.0, p_shift=0.0, p_freq=0.0,
        )
    else:
        raise ValueError(f"Unknown aug_tag: {aug_tag}")


# ── 단일 학습 실행 ────────────────────────────────────────────────────────
def run_one(cfg, args, subject_ids, subj_map, n_subjects,
            cls_list, dino, proto_dino, dino_feat_dim, eeg_ch, device):
    n_classes = len(cls_list)

    train_ds = build_dataset(args.data_root, subject_ids, cls_list, "train", subj_map, args.seed)
    val_ds   = build_dataset(args.data_root, subject_ids, cls_list, "val",   subj_map, args.seed)
    test_ds  = build_dataset(args.data_root, subject_ids, cls_list, "test",  subj_map, args.seed)

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
    warmup_epochs = max(1, args.epochs // 10)
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return ep / warmup_epochs
        progress = (ep - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    augmenter  = build_augmenter(cfg["aug"])
    best_val1  = 0.0
    ckpt_path  = os.path.join(args.save_dir, f"{cfg['name']}_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = ep_total = 0
        for eeg, subj, lbl0, _ in train_loader:
            eeg  = augmenter(eeg.to(device))
            subj = subj.to(device)
            lbl0 = lbl0.to(device)
            tgt  = proto_dino[lbl0]

            loss, lc, lp, li, acc = model.compute_loss(
                eeg, subj, tgt, proto_dino, lbl0,
                use_infonce=True,
                w_cos=cfg["w_cos"],
                w_proto=cfg["w_proto"],
                w_infonce=cfg["w_infonce"],
            )
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * eeg.size(0)
            ep_total += eeg.size(0)

        scheduler.step()

        model.eval()
        val_k = evaluate_retrieval(model, val_loader, proto_dino, device, k_list=(1,3,5))
        if val_k[1] >= best_val1:
            best_val1 = val_k[1]
            torch.save({"model": model.state_dict()}, ckpt_path)

        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"    ep{epoch:>4}  loss={ep_loss/ep_total:.4f}  "
                  f"val@1={val_k[1]:.4f}  val@3={val_k[3]:.4f}", flush=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    test_k = evaluate_retrieval(model, test_loader, proto_dino, device, k_list=(1,3,5))
    return test_k


# ── main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    type=str,   default="./preproc_for_gan_vs")
    parser.add_argument("--img_root",     type=str,   default="")
    parser.add_argument("--subject_ids",  type=str,   default="1-20")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--dino_model",   type=str,   default="dinov2_vits14")
    parser.add_argument("--eeg_hidden",   type=int,   default=256)
    parser.add_argument("--eeg_out",      type=int,   default=256)
    parser.add_argument("--subj_emb_dim", type=int,   default=128,
                        help="128 권장 (within Top-1 최고)")
    parser.add_argument("--n_heads",      type=int,   default=4)
    parser.add_argument("--n_layers",     type=int,   default=4)
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--temperature",  type=float, default=0.1)
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--wd",           type=float, default=1e-4)
    parser.add_argument("--out_csv",      type=str,   default="sweep_results.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}  subj_emb_dim={args.subj_emb_dim}  epochs={args.epochs}")

    cls_list = GROUP_CLASSES[0]   # 9-class

    # subject_ids 파싱
    subject_ids = []
    for tok in args.subject_ids.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-"); subject_ids.extend(range(int(a), int(b)+1))
        else:
            subject_ids.append(int(tok))
    subject_ids = [s for s in subject_ids
                   if os.path.isfile(os.path.join(args.data_root, f"subj_{s:02d}.mat"))]
    n_subjects  = len(subject_ids)
    subj_map    = {sid: i for i, sid in enumerate(subject_ids)}
    print(f"[INFO] 피험자: {n_subjects}명  클래스: {len(cls_list)}개")

    eeg_ch = int(loadmat(
        os.path.join(args.data_root, f"subj_{subject_ids[0]:02d}.mat")
    )["X"].shape[0])

    print(f"[INFO] DINO 로드: {args.dino_model}")
    dino          = load_dino_encoder(args.dino_model, device)
    dino_feat_dim = DINO_DIM[args.dino_model]
    img_root      = args.img_root or "./preproc_data_vi/images"
    proto_dino    = compute_class_prototypes(dino, img_root, cls_list, device)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_dir = f"./checkpoints_sweep/{ts}"
    os.makedirs(args.save_dir, exist_ok=True)

    experiments = make_experiments()
    rows = []

    print(f"\n{'='*65}")
    print(f"  총 {len(experiments)}개 실험  (each {args.epochs} epochs)")
    print(f"{'='*65}")

    for i, cfg in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}]  {cfg['name']}  "
              f"(group={cfg['group']}, w_infonce={cfg['w_infonce']}, "
              f"w_proto={cfg['w_proto']}, aug={cfg['aug']})")
        set_seed(args.seed)
        res = run_one(cfg, args, subject_ids, subj_map, n_subjects,
                      cls_list, dino, proto_dino, dino_feat_dim, eeg_ch, device)
        print(f"  → Test  Top-1={res[1]:.4f}  Top-3={res[3]:.4f}  Top-5={res[5]:.4f}")
        rows.append({
            "name":      cfg["name"],
            "group":     cfg["group"],
            "w_infonce": cfg["w_infonce"],
            "w_proto":   cfg["w_proto"],
            "aug":       cfg["aug"],
            "top1":      round(res[1], 4),
            "top3":      round(res[3], 4),
            "top5":      round(res[5], 4),
            "random":    round(1/len(cls_list), 4),
        })

    # CSV 저장
    out_path = os.path.join(args.save_dir, args.out_csv)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # 요약 출력
    print(f"\n{'='*65}")
    print(f"  Sweep 결과 요약 (Top-1 기준 정렬)")
    print(f"  {'Name':<20}  {'Group':<12}  {'Top-1':>7}  {'Top-3':>7}  {'Top-5':>7}")
    print(f"  {'-'*58}")
    for r in sorted(rows, key=lambda x: -x["top1"]):
        print(f"  {r['name']:<20}  {r['group']:<12}  {r['top1']:>7.4f}  {r['top3']:>7.4f}  {r['top5']:>7.4f}")
    print(f"  Random: {1/len(cls_list):.4f}")
    print(f"\n[INFO] CSV 저장: {out_path}")


if __name__ == "__main__":
    main()
