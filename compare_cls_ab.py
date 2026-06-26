"""
compare_cls_ab.py
─────────────────────────────────────────────────────────────────────────────
방법 A: 별도 EEG 분류 모델 (checkpoints_vs_cls/*/test_acc.npy)
방법 B: 생성 이미지 → ResNet18 feature 유사도로 클래스 예측

사용:
    python compare_cls_ab.py
"""

import os, glob, re
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLS_ROOT    = "./checkpoints_vs_cls"
SAMPLE_ROOT = "./samples_vs_repa"
IMG_ROOT    = "./preproc_data_vi/images"

GROUP_CLASSES = {1: [1,2,3], 2: [4,5,6], 3: [7,8,9]}

# ── ResNet18 feature extractor ────────────────────────────────────────────
def build_extractor():
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    extractor = torch.nn.Sequential(*list(backbone.children())[:-1])
    extractor.eval().to(DEVICE)
    for p in extractor.parameters():
        p.requires_grad = False
    return extractor

transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def extract_feat(extractor, img_path):
    img = Image.open(img_path).convert("RGB")
    x   = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = extractor(x).flatten(1)
    return F.normalize(f, dim=1)

# ── GT 클래스 이미지 feature 캐시 ─────────────────────────────────────────
def build_gt_feats(extractor):
    gt = {}
    for cls in range(1, 10):
        p = os.path.join(IMG_ROOT, f"{cls:02d}.png")
        if os.path.isfile(p):
            gt[cls] = extract_feat(extractor, p)
    return gt

# ── 방법 A: test_acc.npy 로드 ─────────────────────────────────────────────
def load_method_a():
    results = {}
    for path in sorted(glob.glob(os.path.join(CLS_ROOT, "*/test_acc.npy"))):
        name = os.path.basename(os.path.dirname(path))
        acc  = float(np.load(path)[0])
        m    = re.search(r'vs(\d+)_(g\d+)_', name)
        if m:
            subj = int(m.group(1))
            grp  = m.group(2)
            results.setdefault(subj, {})[grp] = acc
    return results

# ── 방법 B: 생성 이미지 → feature similarity 분류 ────────────────────────
def classify_method_b(extractor, gt_feats):
    results = {}

    sample_dirs = sorted(glob.glob(os.path.join(SAMPLE_ROOT, "*", "*")))
    for d in sample_dirs:
        if not os.path.isdir(d):
            continue

        # 폴더명에서 subject / group 파싱: vs01_g1_cls1-3_test
        folder = os.path.basename(d)
        m = re.match(r'vs(\d+)_(g\d+)_cls(\d+)-(\d+)', folder)
        if not m:
            continue
        subj   = int(m.group(1))
        grp    = m.group(2)
        gid    = int(grp[1])
        cls_list = GROUP_CLASSES[gid]

        gen_files = sorted(glob.glob(os.path.join(d, "*_GEN.png")))
        if not gen_files:
            continue

        correct = 0
        for gf in gen_files:
            # 파일명에서 GT 클래스 추출: vs01_trial0000_cls01_ddim_GEN.png
            fm = re.search(r'_cls(\d+)_', os.path.basename(gf))
            if not fm:
                continue
            gt_cls = int(fm.group(1))
            if gt_cls not in cls_list:
                continue

            # 생성 이미지 feature
            gen_feat = extract_feat(extractor, gf)

            # 그룹 내 클래스 이미지와 유사도 비교
            sims = {}
            for c in cls_list:
                if c in gt_feats:
                    sims[c] = float((gen_feat * gt_feats[c]).sum())

            if not sims:
                continue
            pred_cls = max(sims, key=sims.get)
            if pred_cls == gt_cls:
                correct += 1

        acc = correct / len(gen_files) if gen_files else 0.0
        results.setdefault(subj, {})[grp] = acc

    return results

# ── 출력 ──────────────────────────────────────────────────────────────────
def print_comparison(a_res, b_res):
    all_subjs = sorted(set(list(a_res.keys()) + list(b_res.keys())))
    all_grps  = ['g1', 'g2', 'g3']

    header = f"{'Subj':>5}  {'A_g1':>6} {'A_g2':>6} {'A_g3':>6} {'A_avg':>6}  |  {'B_g1':>6} {'B_g2':>6} {'B_g3':>6} {'B_avg':>6}"
    sep    = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("  분류 성능 비교  (A: EEG 분류 모델 | B: 생성이미지 유사도)")
    print(f"  random baseline = 0.333")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    a_all, b_all = [], []
    a_g = {g:[] for g in all_grps}
    b_g = {g:[] for g in all_grps}

    for subj in all_subjs:
        a_row = [a_res.get(subj, {}).get(g, float('nan')) for g in all_grps]
        b_row = [b_res.get(subj, {}).get(g, float('nan')) for g in all_grps]
        a_avg = float(np.nanmean(a_row))
        b_avg = float(np.nanmean(b_row))

        a_all.extend([v for v in a_row if not np.isnan(v)])
        b_all.extend([v for v in b_row if not np.isnan(v)])
        for i, g in enumerate(all_grps):
            if not np.isnan(a_row[i]): a_g[g].append(a_row[i])
            if not np.isnan(b_row[i]): b_g[g].append(b_row[i])

        def fmt(v): return f"{v:>6.3f}" if not np.isnan(v) else f"{'N/A':>6}"
        print(f"{subj:>5}  {fmt(a_row[0])} {fmt(a_row[1])} {fmt(a_row[2])} {a_avg:>6.3f}  |  {fmt(b_row[0])} {fmt(b_row[1])} {fmt(b_row[2])} {b_avg:>6.3f}")

    print(sep)
    a_gm = [float(np.mean(a_g[g])) if a_g[g] else float('nan') for g in all_grps]
    b_gm = [float(np.mean(b_g[g])) if b_g[g] else float('nan') for g in all_grps]
    a_total = float(np.mean(a_all)) if a_all else float('nan')
    b_total = float(np.mean(b_all)) if b_all else float('nan')

    print(f"{'mean':>5}  {a_gm[0]:>6.3f} {a_gm[1]:>6.3f} {a_gm[2]:>6.3f} {a_total:>6.3f}  |  {b_gm[0]:>6.3f} {b_gm[1]:>6.3f} {b_gm[2]:>6.3f} {b_total:>6.3f}")
    print(f"{'='*len(header)}")

    print(f"\n[요약]")
    print(f"  A (EEG 분류 모델):    {a_total:.4f}")
    print(f"  B (생성이미지 유사도): {b_total:.4f}")
    print(f"  random baseline:      0.3333")
    if a_total > b_total:
        print(f"  → A가 {(a_total - b_total)*100:.1f}%p 우세")
    else:
        print(f"  → B가 {(b_total - a_total)*100:.1f}%p 우세")

    # CSV 저장
    import csv
    out_csv = "./cls_comparison_ab.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subj","A_g1","A_g2","A_g3","A_avg","B_g1","B_g2","B_g3","B_avg"])
        for subj in all_subjs:
            a_row = [a_res.get(subj, {}).get(g, "") for g in all_grps]
            b_row = [b_res.get(subj, {}).get(g, "") for g in all_grps]
            a_avg = float(np.nanmean([v for v in a_row if v != ""])) if any(v != "" for v in a_row) else ""
            b_avg = float(np.nanmean([v for v in b_row if v != ""])) if any(v != "" for v in b_row) else ""
            writer.writerow([subj] + a_row + [a_avg] + b_row + [b_avg])
    print(f"\n[저장] {out_csv}")


# ── main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[INFO] Device: {DEVICE}")
    print("[INFO] ResNet18 feature extractor 로드 중...")
    extractor = build_extractor()

    print("[INFO] GT 클래스 이미지 feature 추출 중...")
    gt_feats = build_gt_feats(extractor)
    print(f"       {len(gt_feats)}개 클래스 이미지 로드 완료")

    print("[INFO] 방법 A: EEG 분류 모델 결과 로드...")
    a_res = load_method_a()
    print(f"       {len(a_res)}명 피험자 결과 로드")

    print("[INFO] 방법 B: 생성 이미지 분류 중...")
    b_res = classify_method_b(extractor, gt_feats)
    print(f"       {len(b_res)}명 피험자 분류 완료")

    print_comparison(a_res, b_res)
