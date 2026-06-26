"""
plot_cls_table.py  —  A/B 분류 성능 피험자별 테이블 시각화
"""
import os, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLS_ROOT    = "./checkpoints_vs_cls"
SAMPLE_ROOT = "./samples_vs_repa"
IMG_ROOT    = "./preproc_data_vi/images"
GROUP_CLASSES = {1:[1,2,3], 2:[4,5,6], 3:[7,8,9]}

# ── feature extractor ─────────────────────────────────────────────────────
def build_extractor():
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    ext = torch.nn.Sequential(*list(backbone.children())[:-1])
    ext.eval().to(DEVICE)
    for p in ext.parameters(): p.requires_grad = False
    return ext

tfm = T.Compose([T.Resize((128,128)), T.ToTensor(),
                 T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def feat(ext, path):
    x = tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = ext(x).flatten(1)
    return F.normalize(f, dim=1)

def load_a():
    res = {}
    for path in sorted(glob.glob(os.path.join(CLS_ROOT,"*/test_acc.npy"))):
        name = os.path.basename(os.path.dirname(path))
        acc  = float(np.load(path)[0])
        m    = re.search(r'vs(\d+)_(g\d+)_', name)
        if m: res.setdefault(int(m.group(1)),{})[m.group(2)] = acc
    return res

def load_b(ext, gt_feats):
    res = {}
    for d in sorted(glob.glob(os.path.join(SAMPLE_ROOT,"*","*"))):
        if not os.path.isdir(d): continue
        m = re.match(r'vs(\d+)_(g\d+)_cls(\d+)-(\d+)', os.path.basename(d))
        if not m: continue
        subj, grp, gid = int(m.group(1)), m.group(2), int(m.group(2)[1])
        cls_list = GROUP_CLASSES[gid]
        gens = sorted(glob.glob(os.path.join(d,"*_GEN.png")))
        if not gens: continue
        correct = 0
        for gf in gens:
            fm = re.search(r'_cls(\d+)_', os.path.basename(gf))
            if not fm: continue
            gt_cls = int(fm.group(1))
            if gt_cls not in cls_list: continue
            gf_feat = feat(ext, gf)
            pred = max({c: float((gf_feat*gt_feats[c]).sum()) for c in cls_list if c in gt_feats},
                       key=lambda c: float((gf_feat*gt_feats[c]).sum()))
            if pred == gt_cls: correct += 1
        res.setdefault(subj,{})[grp] = correct/len(gens) if gens else 0.0
    return res

# ── 테이블 그리기 ─────────────────────────────────────────────────────────
def draw_table(a_res, b_res):
    grps    = ['g1','g2','g3']
    subjs   = sorted(set(list(a_res)+list(b_res)))
    n_subj  = len(subjs)
    RAND    = 1/3

    # 데이터 수집
    rows = []
    for s in subjs:
        a = [a_res.get(s,{}).get(g, np.nan) for g in grps]
        b = [b_res.get(s,{}).get(g, np.nan) for g in grps]
        rows.append(a + [np.nanmean(a)] + b + [np.nanmean(b)])

    col_labels = ['A g1','A g2','A g3','A avg','B g1','B g2','B g3','B avg']
    row_labels = [f"S{s:02d}" for s in subjs]

    # mean row
    arr = np.array(rows)
    mean_row = np.nanmean(arr, axis=0).tolist()
    rows.append(mean_row)
    row_labels.append("Mean")

    # ── figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 0.42*(n_subj+3)+1.5))
    ax.axis('off')

    # 컬러맵: 0.0(빨강) ~ 0.333(흰색) ~ 1.0(파랑)
    def cell_color(v):
        if np.isnan(v): return (0.9, 0.9, 0.9)
        if v <= RAND:
            t = v / RAND          # 0~1
            r = 1.0
            g_c = t
            b_c = t
        else:
            t = (v - RAND) / (1 - RAND)
            r = 1 - t
            g_c = 1 - t*0.3
            b_c = 1.0
        return (r, g_c, b_c)

    cell_text   = [[f"{v:.3f}" if not np.isnan(v) else "-" for v in row] for row in rows]
    cell_colors = [[cell_color(v) for v in row] for row in rows]

    # Mean row는 굵은 테두리
    tbl = ax.table(
        cellText   = cell_text,
        rowLabels  = row_labels,
        colLabels  = col_labels,
        cellColours= cell_colors,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)

    # 헤더 배경 색 구분 (A=청록, B=주황)
    for j, label in enumerate(col_labels):
        cell = tbl[0, j]
        cell.set_facecolor('#2196F3' if label.startswith('A') else '#FF9800')
        cell.set_text_props(color='white', fontweight='bold')

    # Mean row 강조
    for j in range(len(col_labels)):
        tbl[n_subj+1, j].set_facecolor('#E0E0E0')
        tbl[n_subj+1, j].set_text_props(fontweight='bold')

    # 구분선: avg 컬럼 (3, 7) 굵게
    for i in range(n_subj+2):
        for j_sep in [3, 7]:
            if i <= n_subj+1:
                try:
                    tbl[i, j_sep].set_edgecolor('#555555')
                    tbl[i, j_sep].set_linewidth(1.5)
                except: pass

    # A/B 구분 세로선
    for i in range(n_subj+2):
        try:
            tbl[i, 4].set_edgecolor('#FF5722')
            tbl[i, 4].set_linewidth(2.0)
        except: pass

    a_mean = float(np.nanmean([r[3] for r in rows[:-1]]))
    b_mean = float(np.nanmean([r[7] for r in rows[:-1]]))
    winner = "B" if b_mean > a_mean else "A"
    diff   = abs(b_mean - a_mean)*100

    title = (f"EEG Classification Performance: A (EEG Classifier) vs B (Generated Image Similarity)\n"
             f"A avg={a_mean:.3f}  |  B avg={b_mean:.3f}  |  Random=0.333  "
             f"→  {winner} wins by {diff:.1f}%p")
    ax.set_title(title, fontsize=11, pad=12, fontweight='bold')

    # 컬러바 범례
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu,
                                norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                        fraction=0.03, pad=0.02, shrink=0.4)
    cbar.set_label("Accuracy  (white = random baseline 0.333)", fontsize=8)
    cbar.set_ticks([0, 0.333, 0.5, 0.75, 1.0])

    plt.tight_layout()
    out = "./cls_comparison_table.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"[저장] {out}")
    plt.close()

# ── main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[DEVICE] {DEVICE}")
    ext      = build_extractor()
    gt_feats = {c: feat(ext, os.path.join(IMG_ROOT,f"{c:02d}.png"))
                for c in range(1,10)
                if os.path.isfile(os.path.join(IMG_ROOT,f"{c:02d}.png"))}
    print(f"GT feats: {len(gt_feats)}개")
    a_res = load_a()
    b_res = load_b(ext, gt_feats)
    draw_table(a_res, b_res)
