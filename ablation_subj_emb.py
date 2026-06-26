"""
ablation_subj_emb.py
─────────────────────────────────────────────────────────────────────────────
Subject embedding 차원 ablation (32 / 64 / 128) + confusion matrix 분석

사용:
    python ablation_subj_emb.py --mode within
    python ablation_subj_emb.py --mode loso
"""

import os, sys, csv, argparse, datetime
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# train 스크립트 함수 재사용
from train_crosssubj_dino import (
    set_seed, GROUP_CLASSES, dino_transform,
    build_dataset, collate_fn, compute_class_prototypes,
    evaluate_retrieval, EEGAugment, run_training,
)
from model_eeg_dino import EEGDINORegressor, load_dino_encoder, DINO_DIM, LATENT_DIM
from scipy.io import loadmat
from torch.utils.data import DataLoader


# ── confusion matrix 수집 ────────────────────────────────────────────────
@torch.no_grad()
def collect_predictions(model, loader, proto_dino, device):
    all_pred, all_true = [], []
    for eeg, subj, lbl0, _ in loader:
        eeg   = eeg.to(device)
        subj  = subj.to(device)
        lbl0  = lbl0.to(device)
        _, pred, _ = model.predict(eeg, subj, proto_dino)
        all_pred.extend(pred.cpu().tolist())
        all_true.extend(lbl0.cpu().tolist())
    return np.array(all_true), np.array(all_pred)


def plot_confusion(cm_norm, cls_list, title, save_path):
    """cm_norm: row-normalized confusion matrix (values 0.0~1.0)."""
    n = len(cls_list)
    fig, ax = plt.subplots(figsize=(n*0.8+1.5, n*0.8+1.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f"cls{c}" for c in cls_list], rotation=45, ha="right")
    ax.set_yticklabels([f"cls{c}" for c in cls_list])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            # use cm_norm (normalized float) for annotation, not raw counts
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i,j] > 0.5 else "black", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Proportion")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


# ── 단일 ablation 실행 ───────────────────────────────────────────────────
def run_ablation(args, subj_emb_dim, subject_ids, subj_map, n_subjects,
                 cls_list, dino, proto_dino, dino_feat_dim, eeg_ch,
                 device, save_dir):

    tag = f"subj{subj_emb_dim}"
    print(f"\n{'='*55}")
    print(f"  Subject embedding dim = {subj_emb_dim}  (mode={args.mode})")
    print(f"{'='*55}")

    orig = args.subj_emb_dim
    args.subj_emb_dim = subj_emb_dim

    if args.mode == "within":
        result = run_training(
            args,
            train_subjs=subject_ids, val_subjs=subject_ids, test_subjs=subject_ids,
            subj_map=subj_map, n_subjects=n_subjects,
            cls_list=cls_list, dino=dino, proto_dino=proto_dino,
            dino_feat_dim=dino_feat_dim, eeg_ch=eeg_ch,
            device=device, save_dir=save_dir, tag=tag,
        )
        # confusion matrix from within test set
        ckpt_path    = os.path.join(save_dir, f"{tag}_best.pt")
        test_subjs   = subject_ids
        cm_subj_map  = subj_map

    else:  # loso
        # LOSO: average top-k across all leave-one-out folds
        loso_results = {}
        all_true, all_pred = [], []

        for test_sid in subject_ids:
            train_sids  = [s for s in subject_ids if s != test_sid]
            loso_map    = {sid: i for i, sid in enumerate(train_sids)}
            loso_map[test_sid] = len(train_sids)   # extra slot
            fold_tag    = f"{tag}_loso{test_sid:02d}"

            r = run_training(
                args,
                train_subjs=train_sids, val_subjs=train_sids,
                test_subjs=[test_sid],
                subj_map=loso_map, n_subjects=n_subjects,
                cls_list=cls_list, dino=dino, proto_dino=proto_dino,
                dino_feat_dim=dino_feat_dim, eeg_ch=eeg_ch,
                device=device, save_dir=save_dir, tag=fold_tag,
            )
            loso_results[test_sid] = r

            # collect predictions for aggregated confusion matrix
            fold_ckpt = os.path.join(save_dir, f"{fold_tag}_best.pt")
            fold_ckpt_data = torch.load(fold_ckpt, map_location=device,
                                        weights_only=False)
            m = EEGDINORegressor(
                eeg_channels=eeg_ch, n_subjects=n_subjects,
                dino_feat_dim=dino_feat_dim, latent_dim=LATENT_DIM,
                eeg_hidden=args.eeg_hidden, eeg_out=args.eeg_out,
                subj_emb_dim=subj_emb_dim,
                n_heads=args.n_heads, n_layers=args.n_layers,
                dropout=args.dropout, temperature=args.temperature,
            ).to(device)
            m.load_state_dict(fold_ckpt_data["model"])
            m.eval()
            s_ds = build_dataset(args.data_root, [test_sid], cls_list,
                                 "test", loso_map)
            s_loader = DataLoader(s_ds, batch_size=64, shuffle=False,
                                  num_workers=0, collate_fn=collate_fn)
            yt, yp = collect_predictions(m, s_loader, proto_dino, device)
            all_true.extend(yt.tolist()); all_pred.extend(yp.tolist())

        valid = {k: v for k, v in loso_results.items() if v is not None}
        avg_k = {k: np.mean([v[k] for v in valid.values()]) for k in (1,3,5)}
        result = avg_k

        # aggregated confusion matrix over all LOSO folds
        cm = confusion_matrix(all_true, all_pred,
                              labels=list(range(len(cls_list))))
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        plot_confusion(
            cm_norm, cls_list,
            title=f"LOSO Confusion (subj_emb={subj_emb_dim}, top1={avg_k[1]:.4f})",
            save_path=os.path.join(save_dir, f"cm_loso_subj{subj_emb_dim}.png"),
        )
        args.subj_emb_dim = orig
        return result

    args.subj_emb_dim = orig

    # within mode: confusion matrix from test split
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EEGDINORegressor(
        eeg_channels=eeg_ch, n_subjects=n_subjects,
        dino_feat_dim=dino_feat_dim, latent_dim=LATENT_DIM,
        eeg_hidden=args.eeg_hidden, eeg_out=args.eeg_out,
        subj_emb_dim=subj_emb_dim,
        n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout, temperature=args.temperature,
    ).to(device)
    model.load_state_dict(ckpt_data["model"])
    model.eval()

    test_ds     = build_dataset(args.data_root, subject_ids, cls_list,
                                "test", subj_map)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=0, collate_fn=collate_fn)
    y_true, y_pred = collect_predictions(model, test_loader, proto_dino, device)

    cm      = confusion_matrix(y_true, y_pred, labels=list(range(len(cls_list))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    plot_confusion(
        cm_norm, cls_list,
        title=f"Within Confusion (subj_emb={subj_emb_dim}, top1={result[1]:.4f})",
        save_path=os.path.join(save_dir, f"cm_within_subj{subj_emb_dim}.png"),
    )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    type=str,   default="./preproc_for_gan_vs")
    parser.add_argument("--img_root",     type=str,   default="")
    parser.add_argument("--subject_ids",  type=str,   default="1-20")
    parser.add_argument("--group_id",     type=int,   default=0, choices=[0,1,2,3])
    parser.add_argument("--mode",         type=str,   default="within")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--dino_model",   type=str,   default="dinov2_vits14")
    parser.add_argument("--eeg_hidden",   type=int,   default=256)
    parser.add_argument("--eeg_out",      type=int,   default=256)
    parser.add_argument("--subj_emb_dim", type=int,   default=64)   # ablation으로 덮어씀
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
    parser.add_argument("--use_infonce",  action="store_true", default=True)
    parser.add_argument("--no_infonce",   action="store_true", default=False)
    parser.add_argument("--no_aug",       action="store_true", default=False)
    parser.add_argument("--lambda_align", type=float, default=1.0)
    parser.add_argument("--ckpt_root",    type=str,   default="./checkpoints_dino")
    parser.add_argument("--subj_emb_dims",type=str,   default="32,64,128",
                        help="ablation할 subject embedding 차원 목록")
    args = parser.parse_args()

    set_seed(args.seed)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    eeg_ch = int(loadmat(
        os.path.join(args.data_root, f"subj_{subject_ids[0]:02d}.mat")
    )["X"].shape[0])

    dino          = load_dino_encoder(args.dino_model, device)
    dino_feat_dim = DINO_DIM[args.dino_model]
    img_root      = args.img_root or "./preproc_data_vi/images"
    proto_dino    = compute_class_prototypes(dino, img_root, cls_list, device)

    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.ckpt_root, f"{ts}_ablation_subjemb_{group_tag}")
    os.makedirs(save_dir, exist_ok=True)

    dims    = [int(d) for d in args.subj_emb_dims.split(",")]
    results = {}

    for dim in dims:
        r = run_ablation(
            args, dim, subject_ids, subj_map, n_subjects,
            cls_list, dino, proto_dino, dino_feat_dim, eeg_ch,
            device, save_dir,
        )
        results[dim] = r

    # 요약 출력 및 CSV
    print(f"\n{'='*55}")
    print(f"  Subject Embedding Ablation 결과  [{group_tag}]")
    print(f"  {'SubjDim':>8}  {'Top-1':>7}  {'Top-3':>7}  {'Top-5':>7}")
    print(f"  {'-'*38}")
    rows = []
    for dim in dims:
        r = results[dim]
        if r:
            print(f"  {dim:>8}  {r[1]:>7.4f}  {r[3]:>7.4f}  {r[5]:>7.4f}")
            rows.append([dim, round(r[1],4), round(r[3],4), round(r[5],4)])
    print(f"  Random   {1/len(cls_list):>7.4f}")

    with open(os.path.join(save_dir, "ablation_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subj_emb_dim","top1","top3","top5"])
        w.writerows(rows)
        w.writerow(["random", round(1/len(cls_list),4), "", ""])
    print(f"\n[저장] {save_dir}")


if __name__ == "__main__":
    main()
