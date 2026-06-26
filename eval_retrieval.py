"""
eval_retrieval.py
─────────────────────────────────────────────────────────────────────────────
학습된 EEGDINORegressor 평가: top-1 / top-3 / top-5 retrieval
within-subject / cross-subject(LOSO) 분리 보고

사용 예시
─────────
  # within-subject 체크포인트 평가
  python eval_retrieval.py --mode within --group_ids 1,2,3

  # LOSO 체크포인트 평가
  python eval_retrieval.py --mode loso --group_ids 1

  # 체크포인트 직접 지정
  python eval_retrieval.py --ckpt_path ./checkpoints_dino/.../within_best.pt
"""

import os, glob, argparse, csv
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

from model_eeg_dino import EEGDINORegressor, load_dino_encoder, DINO_DIM
from train_crosssubj_dino import (
    build_dataset, collate_fn, compute_class_prototypes,
    evaluate_retrieval, GROUP_CLASSES, EEGDataset,
)

dino_transform = T.Compose([
    T.Resize(224), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


def load_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    model = EEGDINORegressor(
        eeg_channels=cfg["eeg_ch"],
        n_subjects=cfg["n_subjects"],
        dino_feat_dim=cfg["dino_feat_dim"],
        eeg_hidden=cfg["eeg_hidden"],
        eeg_out=cfg["eeg_out"],
        subj_emb_dim=cfg["subj_emb_dim"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        temperature=cfg["temperature"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def find_ckpt(ckpt_root, mode, group_tag):
    pattern = os.path.join(ckpt_root, f"*{mode}_{group_tag}*", f"{mode}_best.pt")
    cands   = sorted(glob.glob(pattern))
    return cands[-1] if cands else None


# ── within-subject 평가 ───────────────────────────────────────────────────
def eval_within(ckpt_path, args, device):
    print(f"\n[Within] {ckpt_path}")
    model, cfg = load_model_from_ckpt(ckpt_path, device)
    cls_list   = cfg["cls_list"]
    subj_map   = cfg["subj_map"]
    subject_ids= list(subj_map.keys())
    dino       = load_dino_encoder(cfg["dino_model"], device)
    img_root   = args.img_root or "./preproc_data_vi/images"
    proto_dino = compute_class_prototypes(dino, img_root, cls_list, device)
    n_classes  = len(cls_list)

    # 전체 test
    test_ds = build_dataset(args.data_root, subject_ids, cls_list, "test", subj_map)
    loader  = DataLoader(test_ds, batch_size=64, shuffle=False,
                         num_workers=0, collate_fn=collate_fn)
    overall = evaluate_retrieval(model, loader, proto_dino, device, k_list=(1,3,5))

    # per-subject
    rows = []
    for sid in sorted(subject_ids):
        si   = subj_map[sid]
        s_ds = build_dataset(args.data_root, [sid], cls_list, "test", subj_map)
        if len(s_ds) == 0: continue
        s_loader = DataLoader(s_ds, batch_size=64, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)
        sk = evaluate_retrieval(model, s_loader, proto_dino, device, k_list=(1,3,5))
        rows.append({"subject": f"S{sid:02d}", "top1": sk[1], "top3": sk[3], "top5": sk[5]})

    return overall, rows, n_classes


# ── LOSO 평가 ─────────────────────────────────────────────────────────────
def eval_loso(ckpt_root, group_tag, args, device):
    """LOSO 체크포인트 전체를 로드해 per-subject 결과 집계."""
    pattern = os.path.join(ckpt_root, f"*loso_{group_tag}*", "loso_test*_best.pt")
    cands   = sorted(glob.glob(pattern))
    if not cands:
        print(f"[SKIP] LOSO 체크포인트 없음: {group_tag}")
        return None, None, None

    rows = []
    for cp in cands:
        # 파일명에서 test subject 파싱: loso_test03_best.pt
        fname  = os.path.basename(cp)
        import re
        m = re.search(r"loso_test(\d+)_best", fname)
        if not m: continue
        test_sid = int(m.group(1))

        model, cfg = load_model_from_ckpt(cp, device)
        cls_list   = cfg["cls_list"]
        subj_map   = cfg["subj_map"]
        dino       = load_dino_encoder(cfg["dino_model"], device)
        img_root   = args.img_root or "./preproc_data_vi/images"
        proto_dino = compute_class_prototypes(dino, img_root, cls_list, device)

        # test subject의 데이터만 평가
        s_ds = build_dataset(args.data_root, [test_sid], cls_list, "test", subj_map)
        if len(s_ds) == 0: continue
        s_loader = DataLoader(s_ds, batch_size=64, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)
        sk = evaluate_retrieval(model, s_loader, proto_dino, device, k_list=(1,3,5))
        rows.append({"subject": f"S{test_sid:02d}", "top1": sk[1], "top3": sk[3], "top5": sk[5]})

    n_classes = len(cls_list) if rows else 3
    overall   = {
        1: np.mean([r["top1"] for r in rows]),
        3: np.mean([r["top3"] for r in rows]),
        5: np.mean([r["top5"] for r in rows]),
    }
    return overall, rows, n_classes


# ── 출력 & CSV ────────────────────────────────────────────────────────────
def print_and_save(overall, rows, n_classes, mode, group_tag, out_csv):
    rand = 1 / n_classes

    print(f"\n{'='*58}")
    print(f"  [{mode.upper()}]  {group_tag}  (random={rand:.4f})")
    print(f"  {'Subject':>8}  {'Top-1':>7}  {'Top-3':>7}  {'Top-5':>7}")
    print(f"  {'-'*40}")
    for r in rows:
        print(f"  {r['subject']:>8}  {r['top1']:>7.4f}  {r['top3']:>7.4f}  {r['top5']:>7.4f}")
    print(f"  {'-'*40}")
    print(f"  {'Mean':>8}  {overall[1]:>7.4f}  {overall[3]:>7.4f}  {overall[5]:>7.4f}")
    print(f"  {'Random':>8}  {rand:>7.4f}  {min(rand*3,1):>7.4f}  {min(rand*5,1):>7.4f}")

    write_header = not os.path.isfile(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["mode","group","subject","top1","top3","top5"])
        for r in rows:
            w.writerow([mode, group_tag, r["subject"],
                        round(r["top1"],4), round(r["top3"],4), round(r["top5"],4)])
        w.writerow([mode, group_tag, "Mean",
                    round(overall[1],4), round(overall[3],4), round(overall[5],4)])
        w.writerow([mode, group_tag, "Random", round(rand,4), "", ""])

    print(f"  → 저장: {out_csv}")


# ── main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=str, default="./preproc_for_gan_vs")
    parser.add_argument("--img_root",   type=str, default="")
    parser.add_argument("--ckpt_root",  type=str, default="./checkpoints_dino")
    parser.add_argument("--ckpt_path",  type=str, default=None,
                        help="체크포인트 직접 지정 (within 모드용)")
    parser.add_argument("--mode",       type=str, default="within",
                        choices=["within","loso","both"])
    parser.add_argument("--group_ids",  type=str, default="1,2,3")
    parser.add_argument("--out_csv",    type=str, default="./retrieval_eval.csv")
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    group_ids = [int(g) for g in args.group_ids.split(",")]

    # 기존 파일 초기화
    if os.path.isfile(args.out_csv):
        os.remove(args.out_csv)

    for gid in group_ids:
        cls_list  = GROUP_CLASSES[gid]
        group_tag = f"g{gid}_cls{cls_list[0]}-{cls_list[-1]}"

        if args.mode in ("within", "both"):
            ckpt = args.ckpt_path or find_ckpt(args.ckpt_root, "within", group_tag)
            if ckpt:
                overall, rows, n_cls = eval_within(ckpt, args, device)
                print_and_save(overall, rows, n_cls, "within", group_tag, args.out_csv)
            else:
                print(f"[SKIP] within 체크포인트 없음: {group_tag}")

        if args.mode in ("loso", "both"):
            overall, rows, n_cls = eval_loso(args.ckpt_root, group_tag, args, device)
            if overall:
                print_and_save(overall, rows, n_cls, "loso", group_tag, args.out_csv)

    print(f"\n[완료] {args.out_csv}")


if __name__ == "__main__":
    main()
