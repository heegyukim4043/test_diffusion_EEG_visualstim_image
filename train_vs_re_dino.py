"""
train_vs_re_dino.py
────────────────────────────────────────────────────────────────────────────
preproc_vs_re 데이터 기반  Subject-wise DINO alignment  (Stage 1)

전략: subject-wise first
  - 피험자별 독립 모델 학습 (per-subject)
  - 세션 병합 (session-merged) 기본값
  - EEG 32ch, 0~2 sec 구간 사용

평가:
  - Top-1 / Top-3 / Top-5 retrieval  (DINO 9개 클래스 이미지 갤러리)
  - Per-subject confusion matrix

사용 예시:
  # 전체 피험자
  python train_vs_re_dino.py

  # 특정 피험자
  python train_vs_re_dino.py --subject_ids 1,2,18

  # session-capped (피험자당 최대 2세션)
  python train_vs_re_dino.py --max_sessions 2

  # 채널 40개 ablation
  python train_vs_re_dino.py --n_ch 40
"""

import os, sys, csv, math, random, argparse, datetime, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_vs_re import (
    VSReDataset, collate_fn, available_subjects,
    load_subject_vsre, session_counts,
)
from model_eeg_dino import EEGDINORegressor, load_dino_encoder, DINO_DIM, LATENT_DIM
from train_crosssubj_dino import (
    set_seed, EEGAugment, dino_transform,
    compute_class_prototypes,
)

N_CLASSES  = 9
CLS_LIST   = list(range(1, 10))   # 1~9 (1-based, for image file names)


# ── confusion matrix (sklearn 없어도 동작) ───────────────────────────────────
def compute_confusion(model, loader, proto_dino, device, n_cls=9):
    cm = np.zeros((n_cls, n_cls), dtype=int)
    model.eval()
    with torch.no_grad():
        for eeg, subj, lbl in loader:
            eeg  = eeg.to(device)
            subj = subj.to(device)
            logits, _, _ = model.predict(eeg, subj, proto_dino)
            pred = logits.argmax(dim=1).cpu().numpy()
            gt   = lbl.numpy()
            for g, p in zip(gt, pred):
                cm[g, p] += 1
    return cm


def save_confusion_csv(cm, path, cls_names=None):
    n = cm.shape[0]
    if cls_names is None:
        cls_names = [str(i+1) for i in range(n)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + cls_names)
        for i in range(n):
            row_total = cm[i].sum()
            normed = [f"{cm[i,j]/max(row_total,1):.3f}" for j in range(n)]
            w.writerow([cls_names[i]] + normed)


@torch.no_grad()
def evaluate_retrieval_vsre(model, loader, proto_dino, device, k_list=(1, 3, 5)):
    correct = {k: 0 for k in k_list}
    total = 0
    model.eval()
    for eeg, subj, lbl in loader:
        eeg = eeg.to(device)
        subj = subj.to(device)
        lbl = lbl.to(device)
        logits, _, _ = model.predict(eeg, subj, proto_dino)
        for k in k_list:
            topk = logits.topk(min(k, logits.size(1)), dim=1).indices
            correct[k] += topk.eq(lbl.unsqueeze(1)).any(1).sum().item()
        total += eeg.size(0)
    if total == 0:
        return {k: 0.0 for k in k_list}
    return {k: correct[k] / total for k in k_list}


# ── single-subject training ──────────────────────────────────────────────────
def train_subject(
    sid, data_root, img_root, args,
    dino, proto_dino, dino_feat_dim, device, save_dir,
):
    set_seed(args.seed)

    subj_map = {sid: 0}   # single subject → embedding index 0
    n_subjects = 1

    bl = args.baseline_correct
    zs = args.ch_zscore
    train_ds = VSReDataset(data_root, [sid], subj_map, args.n_ch, "train",
                           args.seed, args.max_sessions,
                           baseline_correct=bl, ch_zscore=zs)
    val_ds   = VSReDataset(data_root, [sid], subj_map, args.n_ch, "val",
                           args.seed, args.max_sessions,
                           baseline_correct=bl, ch_zscore=zs)
    test_ds  = VSReDataset(data_root, [sid], subj_map, args.n_ch, "test",
                           args.seed, args.max_sessions,
                           baseline_correct=bl, ch_zscore=zs)

    if len(train_ds) == 0:
        print(f"  [SKIP] S{sid:02d}: no training data")
        return None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0, collate_fn=collate_fn)

    # EEG 채널 수 → 모델 입력 크기
    eeg_ch = args.n_ch

    # V2 encoder: occipital prior 파싱
    raw_occ = args.eeg_occipital_ids.strip().lower()
    if raw_occ == "auto":
        occipital_indices = None          # BioSemi32 default prior
    elif raw_occ == "none":
        occipital_indices = []            # uniform init (no prior)
    else:
        occipital_indices = [int(x) for x in raw_occ.split(",") if x.strip()]

    model = EEGDINORegressor(
        eeg_channels=eeg_ch, n_subjects=n_subjects,
        dino_feat_dim=dino_feat_dim, latent_dim=LATENT_DIM,
        eeg_hidden=args.eeg_hidden, eeg_out=args.eeg_out,
        subj_emb_dim=args.subj_emb_dim,
        n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout, temperature=args.temperature,
        encoder_type=args.encoder_type,
        n_classes=N_CLASSES,
        eeg_occipital_indices=occipital_indices,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warmup = max(1, args.epochs // 10)
    def lr_lambda(ep):
        if ep < warmup:
            return ep / warmup
        p = (ep - warmup) / max(1, args.epochs - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    augmenter = EEGAugment(
        noise_std=0.05, scale_range=(0.8, 1.2),
        ch_drop_prob=0.1, max_shift=25, freq_noise_std=0.02,
        p_noise=0.5, p_scale=0.5,
        p_drop=0.0 if args.no_aug else 0.3,
        p_shift=0.0 if args.no_aug else 0.3,
        p_freq=0.0,   # CUDA nvrtc 오류 회피 (freq는 CPU 연산이나 불안정)
    )

    best_val1 = 0.0
    ckpt_path = os.path.join(save_dir, f"subj{sid:02d}_best.pt")

    print(f"  S{sid:02d}  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}", flush=True)
    print(f"  {'Ep':>4}  {'Loss':>7}  {'ValT1':>6}  {'ValT3':>6}  {'ValT5':>6}  {'temp':>6}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = ep_total = 0
        for eeg, subj, lbl in train_loader:
            eeg  = augmenter(eeg.to(device))
            subj = subj.to(device)
            lbl  = lbl.to(device)
            tgt  = proto_dino[lbl]

            if args.loss_type == "infonce":
                loss, *_ = model.compute_loss(
                    eeg, subj, tgt, proto_dino, lbl,
                    use_infonce=True,
                    w_cos=args.w_cos, w_proto=args.w_proto,
                    w_infonce=args.w_infonce, w_aux=args.w_aux,
                )
            else:  # supcon or supcon_proto
                eeg_lat = model.encode_eeg(eeg, subj)
                img_lat = model.encode_img(tgt)
                temp    = model.log_temp.exp().clamp(0.01, 1.0).item()
                loss_sc = EEGDINORegressor.supcon_loss(
                    eeg_lat, img_lat, lbl, temperature=temp
                )
                aux_logits  = model.aux_cls_head(eeg_lat)
                loss_aux_ce = F.cross_entropy(aux_logits, lbl)
                if args.loss_type == "supcon_proto":
                    proto_lat  = model.encode_img(proto_dino)
                    sim_proto  = eeg_lat @ proto_lat.T / temp
                    loss_proto = F.cross_entropy(sim_proto, lbl)
                else:
                    loss_proto = torch.tensor(0.0, device=device)
                loss = (args.w_supcon * loss_sc
                        + args.w_proto * loss_proto
                        + args.w_aux   * loss_aux_ce)

            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss  += loss.item() * eeg.size(0)
            ep_total += eeg.size(0)

        scheduler.step()
        model.eval()
        val_k = evaluate_retrieval_vsre(model, val_loader, proto_dino, device)

        if val_k[1] >= best_val1:
            best_val1 = val_k[1]
            torch.save({"model": model.state_dict(),
                        "config": vars(args), "sid": sid}, ckpt_path)

        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            temp = model.log_temp.exp().item()
            print(f"  {epoch:>4}  {ep_loss/ep_total:>7.4f}  "
                  f"{val_k[1]:>6.4f}  {val_k[3]:>6.4f}  {val_k[5]:>6.4f}  "
                  f"{temp:>6.3f}", flush=True)

    # Test + confusion matrix
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    test_k = evaluate_retrieval_vsre(model, test_loader, proto_dino, device)
    cm     = compute_confusion(model, test_loader, proto_dino, device)
    save_confusion_csv(cm, os.path.join(save_dir, f"subj{sid:02d}_confusion.csv"))

    print(f"  [Test S{sid:02d}]  "
          f"Top-1={test_k[1]:.4f}  Top-3={test_k[3]:.4f}  Top-5={test_k[5]:.4f}  "
          f"(random={1/N_CLASSES:.4f})", flush=True)
    return {"top1": test_k[1], "top3": test_k[3], "top5": test_k[5], "sid": sid}


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    type=str, default="./preproc_vs_re")
    parser.add_argument("--img_root",     type=str, default="./preproc_data_vi/images")
    parser.add_argument("--subject_ids",  type=str, default="all",
                        help="all / 1,2,3 / 1-5,18")
    parser.add_argument("--n_ch",         type=int, default=32,
                        help="EEG channels (32=EEG only, 40=EEG+EX)")
    parser.add_argument("--max_sessions", type=int, default=None,
                        help="session-capped: max sessions per subject")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--dino_model",   type=str,   default="dinov2_vits14")
    parser.add_argument("--eeg_hidden",   type=int,   default=256)
    parser.add_argument("--eeg_out",      type=int,   default=256)
    parser.add_argument("--subj_emb_dim", type=int,   default=32,
                        help="subject embedding dim (1 subject → small dim ok)")
    parser.add_argument("--n_heads",      type=int,   default=4)
    parser.add_argument("--n_layers",     type=int,   default=4)
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--temperature",  type=float, default=0.1)
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--wd",           type=float, default=1e-4)
    parser.add_argument("--w_cos",        type=float, default=1.0)
    parser.add_argument("--w_proto",      type=float, default=1.0)
    parser.add_argument("--w_infonce",    type=float, default=1.0)
    parser.add_argument("--w_aux",        type=float, default=0.5,
                        help="auxiliary classification head weight")
    parser.add_argument("--encoder_type", type=str,   default="transformer",
                        choices=["transformer", "conv", "v2"],
                        help="EEG encoder: transformer | conv | v2 (OccipitalGate+MultiScaleStem)")
    parser.add_argument("--eeg_occipital_ids", type=str, default="auto",
                        help="V2 only: 'auto'=BioSemi32 prior, 'none'=no prior, '14,15,16'=custom")
    parser.add_argument("--no_aug",       action="store_true", default=False)
    parser.add_argument("--ckpt_root",    type=str,   default="./checkpoints_vsre_dino")
    parser.add_argument("--loss_type",    type=str,   default="infonce",
                        choices=["infonce", "supcon", "supcon_proto"],
                        help="Loss: infonce=current default, supcon=SupCon only, supcon_proto=SupCon+proto")
    parser.add_argument("--w_supcon",     type=float, default=1.0,
                        help="SupCon loss weight (used when loss_type in supcon/supcon_proto)")
    parser.add_argument("--baseline_correct", action="store_true", default=False,
                        help="Subtract pre-stimulus mean (-1~0 sec) from signal (Exp24)")
    parser.add_argument("--ch_zscore",    action="store_true", default=False,
                        help="Channel-wise z-score normalization per trial (Exp24)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}  n_ch={args.n_ch}  max_sessions={args.max_sessions}  loss_type={args.loss_type}")

    # subject 파싱
    all_sids = available_subjects(args.data_root)
    if args.subject_ids == "all":
        subject_ids = all_sids
    else:
        subject_ids = []
        for tok in args.subject_ids.split(","):
            tok = tok.strip()
            if "-" in tok:
                a, b = tok.split("-")
                subject_ids.extend(range(int(a), int(b)+1))
            else:
                subject_ids.append(int(tok))
        subject_ids = [s for s in subject_ids if s in all_sids]

    sc = session_counts(args.data_root)
    print(f"[INFO] Subjects ({len(subject_ids)}): {subject_ids}")
    print(f"[INFO] Session counts: { {s: sc[s] for s in subject_ids} }")

    # DINO 로드
    print(f"[INFO] Loading DINO: {args.dino_model}")
    dino          = load_dino_encoder(args.dino_model, device)
    dino_feat_dim = DINO_DIM[args.dino_model]
    proto_dino    = compute_class_prototypes(dino, args.img_root, CLS_LIST, device)
    print(f"  proto_dino: {proto_dino.shape}")

    # 저장 경로
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sess_tag = f"cap{args.max_sessions}" if args.max_sessions else "merged"
    loss_tag = args.loss_type if args.loss_type != "infonce" else ""
    bl_tag   = "_bl" if args.baseline_correct else ""
    zs_tag   = "_zs" if args.ch_zscore else ""
    loss_sfx = f"_{loss_tag}" if loss_tag else ""
    save_dir = os.path.join(args.ckpt_root,
                            f"{ts}_ch{args.n_ch}_{sess_tag}_ep{args.epochs}{loss_sfx}{bl_tag}{zs_tag}")
    os.makedirs(save_dir, exist_ok=True)

    # 피험자별 순차 학습
    all_results = []
    for sid in subject_ids:
        print(f"\n{'='*55}")
        print(f"  Subject {sid:02d}  (sessions={sc.get(sid,0)})", flush=True)
        r = train_subject(sid, args.data_root, args.img_root, args,
                          dino, proto_dino, dino_feat_dim, device, save_dir)
        if r:
            all_results.append(r)

    # 요약
    print(f"\n{'='*55}")
    print(f"  Summary  [{sess_tag}, ch={args.n_ch}]")
    print(f"  {'Subj':>5}  {'Top-1':>7}  {'Top-3':>7}  {'Top-5':>7}")
    print(f"  {'-'*32}")

    rows = []
    for r in all_results:
        print(f"  S{r['sid']:02d}    {r['top1']:>7.4f}  {r['top3']:>7.4f}  {r['top5']:>7.4f}")
        rows.append(r)

    if rows:
        t1m = np.mean([r["top1"] for r in rows])
        t3m = np.mean([r["top3"] for r in rows])
        t5m = np.mean([r["top5"] for r in rows])
        print(f"  {'-'*32}")
        print(f"  Mean   {t1m:>7.4f}  {t3m:>7.4f}  {t5m:>7.4f}")
        print(f"  Random {1/N_CLASSES:>7.4f}")

    # CSV 저장
    out_csv = os.path.join(save_dir, "results_per_subject.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "top1", "top3", "top5", "random"])
        for r in rows:
            w.writerow([f"S{r['sid']:02d}", r["top1"], r["top3"], r["top5"],
                        round(1/N_CLASSES, 4)])
        if rows:
            w.writerow(["Mean", round(t1m, 4), round(t3m, 4), round(t5m, 4), ""])
    print(f"\n[INFO] Saved: {save_dir}")


if __name__ == "__main__":
    main()
