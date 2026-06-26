"""
run_all_experiments.py
══════════════════════════════════════════════════════════════════════════════
5일 연산 계획 전체 실행 스크립트 (순차 실행, 재시작 가능)

Phase 1  (Day 1-2) : LOSO  subj_emb_dim = 32, 128  (seed=42)
                     → LOSO Top-1 기준 best_subj_emb 자동 결정
Phase 2  (Day 2)   : within  w_infonce = 0.5, 1.0, 2.0
Phase 3  (Day 3)   : within  w_proto   = 0.5, 1.0, 2.0
Phase 4  (Day 3-4) : within  aug       = full / none / noise_scale_only
                     → 위 3 Phase 에서 within Top-1 기준 top-2 config 자동 선택
Phase 5  (Day 4)   : within  top-2 configs × seed 42,52,62  (재현성)
Phase 6  (Day 5)   : LOSO   top-2 configs × seed 42,52,62  (최종 확정)

결과 저장
─────────
  master_results/
    master_results.csv          ← 전체 실험 통합 (exp_name, mode, ...)
    phase1_loso_summary.csv
    phase2_infonce_within.csv
    phase3_proto_within.csv
    phase4_aug_within.csv
    phase5_repro_within.csv
    phase6_repro_loso.csv
    final_summary.csv           ← 최종 요약 테이블

재시작:
  실험마다 체크포인트가 있으면 skip.
  python run_all_experiments.py --resume

빠른 테스트 (epochs=30):
  python run_all_experiments.py --epochs 30 --loso_epochs 30
"""

import os, sys, csv, math, copy, random, argparse, datetime, json, io

# Windows cp949 환경에서 UTF-8 출력 강제
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_eeg_dino import EEGDINORegressor, load_dino_encoder, DINO_DIM, LATENT_DIM
from train_crosssubj_dino import (
    set_seed, GROUP_CLASSES, EEGAugment, dino_transform,
    collate_fn, build_dataset,
    compute_class_prototypes, evaluate_retrieval,
)


# ══════════════════════════════════════════════════════════════════════════════
# Augmentation factory
# ══════════════════════════════════════════════════════════════════════════════
def make_augmenter(aug_tag: str) -> EEGAugment:
    if aug_tag == "full":
        return EEGAugment(
            noise_std=0.05, scale_range=(0.8, 1.2),
            ch_drop_prob=0.1, max_shift=25, freq_noise_std=0.02,
            p_noise=0.5, p_scale=0.5, p_drop=0.3, p_shift=0.3, p_freq=0.3,
        )
    elif aug_tag == "none":
        return EEGAugment(
            noise_std=0.0, scale_range=(1.0, 1.0),
            ch_drop_prob=0.0, max_shift=0, freq_noise_std=0.0,
            p_noise=0.0, p_scale=0.0, p_drop=0.0, p_shift=0.0, p_freq=0.0,
        )
    elif aug_tag == "noise_scale_only":
        # channel dropout / time shift / freq noise OFF → noise + amplitude scale 만
        return EEGAugment(
            noise_std=0.05, scale_range=(0.8, 1.2),
            ch_drop_prob=0.0, max_shift=0, freq_noise_std=0.0,
            p_noise=0.5, p_scale=0.5, p_drop=0.0, p_shift=0.0, p_freq=0.0,
        )
    else:
        raise ValueError(f"Unknown aug_tag: {aug_tag}")


# ══════════════════════════════════════════════════════════════════════════════
# Single-run trainer  (within 또는 LOSO 1 fold)
# ══════════════════════════════════════════════════════════════════════════════
def train_one_fold(
    *,
    train_subjs, val_subjs, test_subjs,
    subj_map, n_subjects, cls_list,
    dino, proto_dino, dino_feat_dim, eeg_ch,
    device, ckpt_path, resume,
    # hyper-params
    subj_emb_dim, w_cos, w_proto, w_infonce, aug_tag, seed,
    eeg_hidden, eeg_out, n_heads, n_layers, dropout, temperature,
    epochs, batch_size, lr, wd,
    data_root,
    verbose=True,
):
    """학습 + 평가. test Top-k dict 반환."""
    if resume and os.path.isfile(ckpt_path + ".done"):
        info = json.load(open(ckpt_path + ".done"))
        if verbose:
            print(f"    [SKIP] already done → Top-1={info['top1']:.4f}")
        return info

    set_seed(seed)
    n_classes = len(cls_list)

    train_ds = build_dataset(data_root, train_subjs, cls_list, "train", subj_map, seed)
    val_ds   = build_dataset(data_root, val_subjs,   cls_list, "val",   subj_map, seed)
    test_ds  = build_dataset(data_root, test_subjs,  cls_list, "test",  subj_map, seed)

    if len(train_ds) == 0:
        return None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    model = EEGDINORegressor(
        eeg_channels=eeg_ch, n_subjects=n_subjects,
        dino_feat_dim=dino_feat_dim, latent_dim=LATENT_DIM,
        eeg_hidden=eeg_hidden, eeg_out=eeg_out,
        subj_emb_dim=subj_emb_dim,
        n_heads=n_heads, n_layers=n_layers,
        dropout=dropout, temperature=temperature,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = max(1, epochs // 10)
    def lr_lambda(ep):
        if ep < warmup:
            return ep / warmup
        p = (ep - warmup) / max(1, epochs - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    augmenter  = make_augmenter(aug_tag)
    best_val1  = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = ep_total = 0
        for eeg, subj, lbl0, _ in train_loader:
            eeg  = augmenter(eeg.to(device))
            subj = subj.to(device)
            lbl0 = lbl0.to(device)
            tgt  = proto_dino[lbl0]

            loss, *_ = model.compute_loss(
                eeg, subj, tgt, proto_dino, lbl0,
                use_infonce=True,
                w_cos=w_cos, w_proto=w_proto, w_infonce=w_infonce,
            )
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss  += loss.item() * eeg.size(0)
            ep_total += eeg.size(0)

        scheduler.step()
        model.eval()
        val_k = evaluate_retrieval(model, val_loader, proto_dino, device)

        if val_k[1] >= best_val1:
            best_val1 = val_k[1]
            torch.save({"model": model.state_dict()}, ckpt_path)

        if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs):
            print(f"      ep{epoch:>4}  loss={ep_loss/ep_total:.4f}  "
                  f"val@1={val_k[1]:.4f}  val@3={val_k[3]:.4f}", flush=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    test_k = evaluate_retrieval(model, test_loader, proto_dino, device)

    result = {"top1": test_k[1], "top3": test_k[3], "top5": test_k[5]}
    json.dump(result, open(ckpt_path + ".done", "w"))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# within 실험
# ══════════════════════════════════════════════════════════════════════════════
def run_within(exp_name, subject_ids, subj_map, cls_list,
               dino, proto_dino, dino_feat_dim, eeg_ch,
               device, save_dir, resume, cfg, data_root):
    n_subjects = len(subject_ids)
    ckpt_path  = os.path.join(save_dir, f"{exp_name}_best.pt")
    print(f"  [within] {exp_name}", flush=True)
    r = train_one_fold(
        train_subjs=subject_ids, val_subjs=subject_ids, test_subjs=subject_ids,
        subj_map=subj_map, n_subjects=n_subjects, cls_list=cls_list,
        dino=dino, proto_dino=proto_dino, dino_feat_dim=dino_feat_dim,
        eeg_ch=eeg_ch, device=device, ckpt_path=ckpt_path, resume=resume,
        data_root=data_root, **cfg,
    )
    if r:
        print(f"  → Top-1={r['top1']:.4f}  Top-3={r['top3']:.4f}  "
              f"Top-5={r['top5']:.4f}", flush=True)
    return r


# ══════════════════════════════════════════════════════════════════════════════
# LOSO 실험
# ══════════════════════════════════════════════════════════════════════════════
def run_loso(exp_name, subject_ids, cls_list,
             dino, proto_dino, dino_feat_dim, eeg_ch,
             device, save_dir, resume, cfg, data_root):
    per_subj = {}
    for test_sid in subject_ids:
        train_sids = [s for s in subject_ids if s != test_sid]
        loso_map   = {sid: i for i, sid in enumerate(train_sids)}
        loso_map[test_sid] = len(train_sids)
        n_subj     = len(subject_ids)

        fold_name = f"{exp_name}_loso{test_sid:02d}"
        ckpt_path = os.path.join(save_dir, f"{fold_name}_best.pt")
        print(f"  [loso] {fold_name}", flush=True)

        r = train_one_fold(
            train_subjs=train_sids, val_subjs=train_sids, test_subjs=[test_sid],
            subj_map=loso_map, n_subjects=n_subj, cls_list=cls_list,
            dino=dino, proto_dino=proto_dino, dino_feat_dim=dino_feat_dim,
            eeg_ch=eeg_ch, device=device, ckpt_path=ckpt_path, resume=resume,
            data_root=data_root, **cfg,
        )
        per_subj[test_sid] = r
        if r:
            print(f"  → S{test_sid:02d}  Top-1={r['top1']:.4f}  "
                  f"Top-3={r['top3']:.4f}  Top-5={r['top5']:.4f}", flush=True)

    valid = [r for r in per_subj.values() if r]
    if not valid:
        return None
    mean = {
        "top1": np.mean([r["top1"] for r in valid]),
        "top3": np.mean([r["top3"] for r in valid]),
        "top5": np.mean([r["top5"] for r in valid]),
        "per_subj": per_subj,
    }
    print(f"  LOSO mean  Top-1={mean['top1']:.4f}  Top-3={mean['top3']:.4f}  "
          f"Top-5={mean['top5']:.4f}", flush=True)

    # per-subject CSV
    subj_csv = os.path.join(save_dir, f"{exp_name}_loso_per_subj.csv")
    with open(subj_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "top1", "top3", "top5"])
        for sid, r in per_subj.items():
            if r:
                w.writerow([f"S{sid:02d}", r["top1"], r["top3"], r["top5"]])
        w.writerow(["Mean", mean["top1"], mean["top3"], mean["top5"]])
        w.writerow(["Random", round(1/len(cls_list), 4), "", ""])
    return mean


# ══════════════════════════════════════════════════════════════════════════════
# Master results writer
# ══════════════════════════════════════════════════════════════════════════════
class MasterLog:
    FIELDS = [
        "exp_name", "phase", "mode", "subj_emb_dim",
        "w_cos", "w_proto", "w_infonce", "aug", "seed",
        "top1", "top3", "top5", "random",
    ]

    def __init__(self, path):
        self.path = path
        if not os.path.isfile(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, self.FIELDS).writeheader()

    def write(self, row: dict):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, self.FIELDS).writerow(row)

    def read_all(self):
        with open(self.path) as f:
            return list(csv.DictReader(f))


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   type=str,   default="./preproc_for_gan_vs")
    parser.add_argument("--img_root",    type=str,   default="")
    parser.add_argument("--subject_ids", type=str,   default="1-20")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--dino_model",  type=str,   default="dinov2_vits14")
    parser.add_argument("--eeg_hidden",  type=int,   default=256)
    parser.add_argument("--eeg_out",     type=int,   default=256)
    parser.add_argument("--n_heads",     type=int,   default=4)
    parser.add_argument("--n_layers",    type=int,   default=4)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--epochs",      type=int,   default=200,
                        help="within 학습 epoch 수")
    parser.add_argument("--loso_epochs", type=int,   default=None,
                        help="LOSO epoch 수 (기본=epochs와 동일)")
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--wd",          type=float, default=1e-4)
    parser.add_argument("--out_dir",     type=str,   default="./master_results")
    parser.add_argument("--resume",      action="store_true", default=False,
                        help="체크포인트 있으면 skip")
    parser.add_argument("--repro_seeds", type=str,   default="42,52,62",
                        help="재현성 실험 seed 목록")
    args = parser.parse_args()

    if args.loso_epochs is None:
        args.loso_epochs = args.epochs

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] within epochs={args.epochs}  LOSO epochs={args.loso_epochs}")

    os.makedirs(args.out_dir, exist_ok=True)
    master_csv = os.path.join(args.out_dir, "master_results.csv")
    log = MasterLog(master_csv)

    # ── subject 파싱 ─────────────────────────────────────────────────────────
    subject_ids = []
    for tok in args.subject_ids.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-"); subject_ids.extend(range(int(a), int(b)+1))
        else:
            subject_ids.append(int(tok))
    subject_ids = [s for s in subject_ids
                   if os.path.isfile(os.path.join(args.data_root, f"subj_{s:02d}.mat"))]
    subj_map    = {sid: i for i, sid in enumerate(subject_ids)}
    n_subjects  = len(subject_ids)
    cls_list    = GROUP_CLASSES[0]   # 9-class
    n_classes   = len(cls_list)
    random_acc  = round(1 / n_classes, 4)
    repro_seeds = [int(s) for s in args.repro_seeds.split(",")]

    print(f"[INFO] 피험자 {n_subjects}명  클래스 {n_classes}개  random={random_acc:.4f}")

    eeg_ch = int(loadmat(
        os.path.join(args.data_root, f"subj_{subject_ids[0]:02d}.mat")
    )["X"].shape[0])

    img_root = args.img_root or "./preproc_data_vi/images"
    print(f"[INFO] DINO 로드: {args.dino_model}")
    dino          = load_dino_encoder(args.dino_model, device)
    dino_feat_dim = DINO_DIM[args.dino_model]
    proto_dino    = compute_class_prototypes(dino, img_root, cls_list, device)
    print(f"  proto shape: {proto_dino.shape}")

    # 공통 cfg builder
    def base_cfg(**overrides):
        cfg = dict(
            subj_emb_dim=64, w_cos=1.0, w_proto=1.0, w_infonce=1.0,
            aug_tag="full", seed=args.seed,
            eeg_hidden=args.eeg_hidden, eeg_out=args.eeg_out,
            n_heads=args.n_heads, n_layers=args.n_layers,
            dropout=args.dropout, temperature=args.temperature,
            batch_size=args.batch_size, lr=args.lr, wd=args.wd,
        )
        cfg.update(overrides)
        return cfg

    def log_row(exp_name, phase, mode, cfg, result):
        if result is None:
            return
        log.write({
            "exp_name":    exp_name,
            "phase":       phase,
            "mode":        mode,
            "subj_emb_dim":cfg.get("subj_emb_dim"),
            "w_cos":       cfg.get("w_cos"),
            "w_proto":     cfg.get("w_proto"),
            "w_infonce":   cfg.get("w_infonce"),
            "aug":         cfg.get("aug_tag"),
            "seed":        cfg.get("seed"),
            "top1":        round(result["top1"], 4),
            "top3":        round(result["top3"], 4),
            "top5":        round(result["top5"], 4),
            "random":      random_acc,
        })

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1: LOSO  subj_emb_dim = 32, 128
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 1 — LOSO  subj_emb_dim = 32, 128  (seed=42)")
    print(f"{'='*65}")
    phase1_dir = os.path.join(args.out_dir, "phase1_loso_subj_emb")
    os.makedirs(phase1_dir, exist_ok=True)

    phase1_results = {}
    for sdim in [32, 128]:
        cfg = base_cfg(subj_emb_dim=sdim, epochs=args.loso_epochs)
        exp = f"loso_sdim{sdim}_s42"
        r   = run_loso(exp, subject_ids, cls_list,
                       dino, proto_dino, dino_feat_dim, eeg_ch,
                       device, phase1_dir, args.resume, cfg, args.data_root)
        phase1_results[sdim] = r
        log_row(exp, "phase1", "loso", cfg, r)

    # best subj_emb_dim 결정
    best_sdim = max(phase1_results, key=lambda d: phase1_results[d]["top1"]
                    if phase1_results[d] else -1)
    print(f"\n  ★ best subj_emb_dim = {best_sdim}  "
          f"(LOSO Top-1={phase1_results[best_sdim]['top1']:.4f})\n")

    # Phase 1 summary CSV
    with open(os.path.join(phase1_dir, "phase1_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subj_emb_dim", "loso_top1", "loso_top3", "loso_top5", "random", "best"])
        for sdim in [32, 128]:
            r = phase1_results[sdim]
            if r:
                w.writerow([sdim, r["top1"], r["top3"], r["top5"],
                             random_acc, "★" if sdim == best_sdim else ""])

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2: within  w_infonce sweep  [0.5, 1.0, 2.0]
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 2 — within  w_infonce sweep  [0.5, 1.0, 2.0]")
    print(f"  subj_emb_dim = {best_sdim}")
    print(f"{'='*65}")
    phase2_dir = os.path.join(args.out_dir, "phase2_infonce")
    os.makedirs(phase2_dir, exist_ok=True)

    phase2_results = {}
    for wi in [0.5, 1.0, 2.0]:
        cfg = base_cfg(subj_emb_dim=best_sdim, w_infonce=wi, epochs=args.epochs)
        exp = f"within_sdim{best_sdim}_wi{wi}_s42"
        r   = run_within(exp, subject_ids, subj_map, cls_list,
                         dino, proto_dino, dino_feat_dim, eeg_ch,
                         device, phase2_dir, args.resume, cfg, args.data_root)
        phase2_results[wi] = r
        log_row(exp, "phase2", "within", cfg, r)

    best_wi = max(phase2_results, key=lambda k: phase2_results[k]["top1"]
                  if phase2_results[k] else -1)
    print(f"\n  ★ best w_infonce = {best_wi}  "
          f"(Top-1={phase2_results[best_wi]['top1']:.4f})\n")

    with open(os.path.join(phase2_dir, "phase2_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["w_infonce", "top1", "top3", "top5", "random", "best"])
        for wi in [0.5, 1.0, 2.0]:
            r = phase2_results[wi]
            if r:
                w.writerow([wi, r["top1"], r["top3"], r["top5"],
                             random_acc, "★" if wi == best_wi else ""])

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3: within  w_proto sweep  [0.5, 1.0, 2.0]
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 3 — within  w_proto sweep  [0.5, 1.0, 2.0]")
    print(f"  subj_emb_dim={best_sdim}  w_infonce={best_wi}")
    print(f"{'='*65}")
    phase3_dir = os.path.join(args.out_dir, "phase3_proto")
    os.makedirs(phase3_dir, exist_ok=True)

    phase3_results = {}
    for wp in [0.5, 1.0, 2.0]:
        cfg = base_cfg(subj_emb_dim=best_sdim, w_infonce=best_wi,
                       w_proto=wp, epochs=args.epochs)
        exp = f"within_sdim{best_sdim}_wi{best_wi}_wp{wp}_s42"
        r   = run_within(exp, subject_ids, subj_map, cls_list,
                         dino, proto_dino, dino_feat_dim, eeg_ch,
                         device, phase3_dir, args.resume, cfg, args.data_root)
        phase3_results[wp] = r
        log_row(exp, "phase3", "within", cfg, r)

    best_wp = max(phase3_results, key=lambda k: phase3_results[k]["top1"]
                  if phase3_results[k] else -1)
    print(f"\n  ★ best w_proto = {best_wp}  "
          f"(Top-1={phase3_results[best_wp]['top1']:.4f})\n")

    with open(os.path.join(phase3_dir, "phase3_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["w_proto", "top1", "top3", "top5", "random", "best"])
        for wp in [0.5, 1.0, 2.0]:
            r = phase3_results[wp]
            if r:
                w.writerow([wp, r["top1"], r["top3"], r["top5"],
                             random_acc, "★" if wp == best_wp else ""])

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 4: within  augmentation ablation
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 4 — within  augmentation ablation")
    print(f"  subj_emb_dim={best_sdim}  w_infonce={best_wi}  w_proto={best_wp}")
    print(f"{'='*65}")
    phase4_dir = os.path.join(args.out_dir, "phase4_aug")
    os.makedirs(phase4_dir, exist_ok=True)

    aug_tags = ["full", "none", "noise_scale_only"]
    phase4_results = {}
    for aug_tag in aug_tags:
        cfg = base_cfg(subj_emb_dim=best_sdim, w_infonce=best_wi,
                       w_proto=best_wp, aug_tag=aug_tag, epochs=args.epochs)
        exp = f"within_sdim{best_sdim}_wi{best_wi}_wp{best_wp}_aug{aug_tag}_s42"
        r   = run_within(exp, subject_ids, subj_map, cls_list,
                         dino, proto_dino, dino_feat_dim, eeg_ch,
                         device, phase4_dir, args.resume, cfg, args.data_root)
        phase4_results[aug_tag] = r
        log_row(exp, "phase4", "within", cfg, r)

    best_aug = max(phase4_results, key=lambda k: phase4_results[k]["top1"]
                   if phase4_results[k] else -1)
    print(f"\n  ★ best aug = {best_aug}  "
          f"(Top-1={phase4_results[best_aug]['top1']:.4f})\n")

    with open(os.path.join(phase4_dir, "phase4_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aug", "top1", "top3", "top5", "random", "best"])
        for aug_tag in aug_tags:
            r = phase4_results[aug_tag]
            if r:
                w.writerow([aug_tag, r["top1"], r["top3"], r["top5"],
                             random_acc, "★" if aug_tag == best_aug else ""])

    # ── Top-2 configs 선택 (Phase 2-4 통합) ─────────────────────────────────
    all_within = []
    for wi in [0.5, 1.0, 2.0]:
        r = phase2_results.get(wi)
        if r:
            all_within.append({"w_infonce": wi, "w_proto": 1.0,   "aug_tag": "full",
                                "top1": r["top1"], "src": f"p2_wi{wi}"})
    for wp in [0.5, 1.0, 2.0]:
        r = phase3_results.get(wp)
        if r:
            all_within.append({"w_infonce": best_wi, "w_proto": wp, "aug_tag": "full",
                                "top1": r["top1"], "src": f"p3_wp{wp}"})
    for aug_tag in aug_tags:
        r = phase4_results.get(aug_tag)
        if r:
            all_within.append({"w_infonce": best_wi, "w_proto": best_wp, "aug_tag": aug_tag,
                                "top1": r["top1"], "src": f"p4_{aug_tag}"})

    all_within.sort(key=lambda x: -x["top1"])
    # 동일 config 중복 제거 (w_infonce, w_proto, aug_tag 기준)
    seen = set()
    top2 = []
    for item in all_within:
        key = (item["w_infonce"], item["w_proto"], item["aug_tag"])
        if key not in seen:
            seen.add(key)
            top2.append(item)
        if len(top2) == 2:
            break

    print(f"\n  ★ Top-2 configs 선택:")
    for i, c in enumerate(top2, 1):
        print(f"    #{i}: {c['src']}  "
              f"w_infonce={c['w_infonce']}  w_proto={c['w_proto']}  "
              f"aug={c['aug_tag']}  → within Top-1={c['top1']:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 5: within  top-2 configs × seeds  (재현성)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 5 — within  top-2 configs × seeds {repro_seeds}")
    print(f"{'='*65}")
    phase5_dir = os.path.join(args.out_dir, "phase5_repro_within")
    os.makedirs(phase5_dir, exist_ok=True)

    phase5_results = []
    for ci, tc in enumerate(top2, 1):
        for seed in repro_seeds:
            cfg = base_cfg(
                subj_emb_dim=best_sdim,
                w_infonce=tc["w_infonce"], w_proto=tc["w_proto"],
                aug_tag=tc["aug_tag"], seed=seed, epochs=args.epochs,
            )
            exp = (f"repro_within_cfg{ci}_sdim{best_sdim}"
                   f"_wi{tc['w_infonce']}_wp{tc['w_proto']}"
                   f"_aug{tc['aug_tag']}_s{seed}")
            r   = run_within(exp, subject_ids, subj_map, cls_list,
                             dino, proto_dino, dino_feat_dim, eeg_ch,
                             device, phase5_dir, args.resume, cfg, args.data_root)
            phase5_results.append({"exp": exp, "cfg_idx": ci, "seed": seed, "result": r})
            log_row(exp, "phase5", "within", cfg, r)

    with open(os.path.join(phase5_dir, "phase5_repro_within.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cfg_idx", "seed", "top1", "top3", "top5"])
        for item in phase5_results:
            r = item["result"]
            if r:
                w.writerow([item["cfg_idx"], item["seed"],
                             r["top1"], r["top3"], r["top5"]])

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 6: LOSO  top-2 configs × seeds  (최종 확정)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 6 — LOSO  top-2 configs × seeds {repro_seeds}")
    print(f"{'='*65}")
    phase6_dir = os.path.join(args.out_dir, "phase6_repro_loso")
    os.makedirs(phase6_dir, exist_ok=True)

    phase6_results = []
    for ci, tc in enumerate(top2, 1):
        for seed in repro_seeds:
            cfg = base_cfg(
                subj_emb_dim=best_sdim,
                w_infonce=tc["w_infonce"], w_proto=tc["w_proto"],
                aug_tag=tc["aug_tag"], seed=seed, epochs=args.loso_epochs,
            )
            exp = (f"repro_loso_cfg{ci}_sdim{best_sdim}"
                   f"_wi{tc['w_infonce']}_wp{tc['w_proto']}"
                   f"_aug{tc['aug_tag']}_s{seed}")
            r   = run_loso(exp, subject_ids, cls_list,
                           dino, proto_dino, dino_feat_dim, eeg_ch,
                           device, phase6_dir, args.resume, cfg, args.data_root)
            phase6_results.append({"exp": exp, "cfg_idx": ci, "seed": seed, "result": r})
            log_row(exp, "phase6", "loso", cfg, r)

    with open(os.path.join(phase6_dir, "phase6_repro_loso.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cfg_idx", "seed", "loso_top1", "loso_top3", "loso_top5"])
        for item in phase6_results:
            r = item["result"]
            if r:
                w.writerow([item["cfg_idx"], item["seed"],
                             r["top1"], r["top3"], r["top5"]])

    # ══════════════════════════════════════════════════════════════════════════
    # 최종 요약
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  전체 실험 완료  →  {master_csv}")
    print(f"{'='*65}")

    rows = log.read_all()
    print(f"\n  {'exp_name':<55}  {'mode':<6}  {'top1':>6}  {'top3':>6}  {'top5':>6}")
    print(f"  {'-'*85}")
    for r in sorted(rows, key=lambda x: -float(x["top1"])):
        print(f"  {r['exp_name']:<55}  {r['mode']:<6}  "
              f"{float(r['top1']):>6.4f}  {float(r['top3']):>6.4f}  {float(r['top5']):>6.4f}")

    print(f"\n  Random baseline: {random_acc:.4f}")
    print(f"[DONE]  master_results.csv → {master_csv}")


if __name__ == "__main__":
    main()
