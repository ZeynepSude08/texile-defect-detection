# -*- coding: utf-8 -*-
"""
Qualitative comparison figure with rule-based sample selection.

1) Predictions are read from the <slug>_test_predictions.json files written by
   eval_pycocotools.py (the same predictions behind the reported numbers).
2) Each model is evaluated at its own threshold (default: best-F1 threshold on the test set).
3) Per-image TP/FP/FN and F1 are computed per model (IoU=0.5, one-to-one matching).
4) Images are split into Easy / Medium / Hard by a model-independent difficulty score
   (mean F1 over all models) using pre-defined thresholds.
5) Samples are drawn at random within each group with a fixed seed.

Outputs (OUT_DIR):
  figure_qualitative_main.png          main figure (N_MAIN samples per group)
  figure_qualitative_supp_<group>.png  supplementary figures (N_SUPP samples per group)
  per_image_metrics.csv                per-model TP/FP/FN/F1 and group for every test image
  selection_log.txt                    rule, seed and selected file names

Set the root directory with the BENCH_BASE environment variable.
"""
import os
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# SETTINGS 

BASE = os.environ.get('BENCH_BASE', '.')

DATA_ROOT = os.path.join(BASE, 'data', 'COCO')
RESULTS_ROOT = os.path.join(BASE, 'results')

EVAL_DIR = os.path.join(RESULTS_ROOT, 'evaluation_predict')   # same as OUT_DIR in eval_pycocotools.py
IMG_DIR = os.path.join(DATA_ROOT, 'test')
ANN_PATH = os.path.join(IMG_DIR, '_annotations.coco.json')
OUT_DIR = os.path.join(RESULTS_ROOT, 'qualitative_figures')

SEED = 13
THRESHOLD_MODE = 'best_f1'      # 'best_f1': each model at its own best-F1 threshold | 'fixed': all at FIXED_CONF
FIXED_CONF = 0.5
IOU_THR = 0.5

DIFFICULTY_BY = 'mean'          # 'mean': mean F1 over all models | or a model name
EASY_MIN = 0.8                  # difficulty >= EASY_MIN -> Easy
HARD_MAX = 0.4                  # difficulty <  HARD_MAX -> Hard, otherwise Medium
N_MAIN = 1                      # samples per group in the main figure
N_SUPP = 4                      # samples per group in supplementary figures (includes main samples)
SHOW_SCORES = True

# (display name, file slug, color); display name must match 'model' in summary_test.csv
MODELS = [
    ('RF-DETR 2XL', 'rf_detr_2xl', '#d62728'),
    ('RF-DETR L',   'rf_detr_l',   '#1f3fff'),
    ('RF-DETR M',   'rf_detr_m',   '#c9951a'),
    ('YOLOv8n',     'yolov8n',     '#17c3a5'),
    ('YOLO11m',     'yolo11m',     '#e61fe6'),
]
STRATA = ['Easy', 'Medium', 'Hard']

plt.rcParams['font.family'] = 'serif'



# HELPERS

def iou_matrix(a, b):
    """a: (N,4), b: (M,4), xyxy -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_image(gt, pred_boxes, pred_scores, iou_thr):
    """Score-ordered one-to-one matching, close to COCOeval. Returns tp, fp, fn."""
    tp = 0
    if len(gt) and len(pred_boxes):
        ious = iou_matrix(pred_boxes, gt)
        used = np.zeros(len(gt), dtype=bool)
        for i in np.argsort(-pred_scores):
            cand = np.where((~used) & (ious[i] >= iou_thr))[0]
            if len(cand):
                used[cand[np.argmax(ious[i][cand])]] = True
                tp += 1
    return tp, len(pred_boxes) - tp, len(gt) - tp


def load_gt(coco):
    gt = {}
    for a in coco['annotations']:
        x, y, w, h = a['bbox']
        gt.setdefault(a['image_id'], []).append([x, y, x + w, y + h])
    return {k: np.array(v, dtype=float) for k, v in gt.items()}


def load_preds(path):
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    out = {}
    for d in raw:
        x, y, w, h = d['bbox']
        out.setdefault(d['image_id'], []).append([x, y, x + w, y + h, d['score']])
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def get_thresholds():
    if THRESHOLD_MODE == 'fixed':
        return {name: FIXED_CONF for name, _, _ in MODELS}
    summ = pd.read_csv(os.path.join(EVAL_DIR, 'summary_test.csv')).set_index('model')
    return {name: float(summ.loc[name, 'best_f1_conf']) for name, _, _ in MODELS}


def preds_at(preds, img_id, thr):
    arr = preds.get(img_id)
    if arr is None:
        return np.zeros((0, 4)), np.zeros((0,))
    arr = arr[arr[:, 4] >= thr]
    return arr[:, :4], arr[:, 4]



# 1) PER-IMAGE METRICS

def compute_per_image(coco, gt, all_preds, thr):
    rows = []
    for info in coco['images']:
        iid = info['id']
        g = gt.get(iid, np.zeros((0, 4)))
        row = {'image_id': iid, 'file_name': info['file_name'], 'n_gt': len(g)}
        for name, _, _ in MODELS:
            pb, ps = preds_at(all_preds[name], iid, thr[name])
            tp, fp, fn = match_image(g, pb, ps, IOU_THR)
            denom = 2 * tp + fp + fn
            row[f'{name}|tp'], row[f'{name}|fp'], row[f'{name}|fn'] = tp, fp, fn
            row[f'{name}|f1'] = (2 * tp / denom) if denom > 0 else 1.0
        rows.append(row)
    df = pd.DataFrame(rows)

    f1_cols = [f'{n}|f1' for n, _, _ in MODELS]
    if DIFFICULTY_BY == 'mean':
        df['difficulty'] = df[f1_cols].mean(axis=1)
    else:
        df['difficulty'] = df[f'{DIFFICULTY_BY}|f1']

    def stratum(v):
        return 'Easy' if v >= EASY_MIN else ('Hard' if v < HARD_MAX else 'Medium')
    df['stratum'] = df['difficulty'].apply(stratum)
    return df


def sanity_check(df, thr):
    """Compare total TP/FP/FN with summary_test.csv (small differences are possible)."""
    path = os.path.join(EVAL_DIR, 'summary_test.csv')
    if not os.path.exists(path) or THRESHOLD_MODE != 'best_f1':
        return
    summ = pd.read_csv(path).set_index('model')
    print('\nSanity check (this script vs summary_test.csv, at best-F1 threshold):')
    for name, _, _ in MODELS:
        mine = tuple(int(df[f'{name}|{k}'].sum()) for k in ('tp', 'fp', 'fn'))
        ref = tuple(int(summ.loc[name, k]) for k in ('TP', 'FP', 'FN'))
        flag = 'OK' if mine == ref else 'DIFFERENT'
        print(f'  {name:12s} script TP/FP/FN = {mine}   csv = {ref}   {flag}')


# 2) SAMPLE SELECTION 

def draw_samples(df, n_total):
    rng = random.Random(SEED)
    picked = {}
    for s in STRATA:
        pool = df[(df['stratum'] == s) & (df['n_gt'] > 0)].sort_values('file_name')['image_id'].tolist()
        rng.shuffle(pool)
        picked[s] = pool[:n_total]
    return picked



# 3) PLOTTING

def draw_boxes(ax, boxes, color, scores=None):
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=1.6))
        if scores is not None and SHOW_SCORES:
            ax.text(x1 + 2, y1 + 2, f'{scores[i]:.2f}', color='white', fontsize=6.5, va='top',
                    bbox=dict(facecolor=color, edgecolor='none', pad=1.0, alpha=0.9))


def make_figure(entries, df, coco_imgs, gt, all_preds, thr, out_path):
    """entries: [(stratum, image_id), ...]; one row per image."""
    n_rows, n_cols = len(entries), 2 + len(MODELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.15 * n_rows), squeeze=False)
    headers = ['Original image', 'Ground truth'] + [m[0] for m in MODELS]

    for r, (stratum, iid) in enumerate(entries):
        info = coco_imgs[iid]
        img = Image.open(os.path.join(IMG_DIR, info['file_name'])).convert('RGB')
        row = df[df['image_id'] == iid].iloc[0]
        g = gt.get(iid, np.zeros((0, 4)))

        for c in range(n_cols):
            ax = axes[r][c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(headers[c], fontsize=12, pad=6)

        draw_boxes(axes[r][1], g, '#00c800')
        axes[r][1].set_xlabel(f'{len(g)} object(s)', fontsize=9)
        axes[r][0].set_ylabel(f'{stratum}\n(mean F1 = {row["difficulty"]:.2f})', fontsize=11)

        for k, (name, _, color) in enumerate(MODELS):
            ax = axes[r][2 + k]
            pb, ps = preds_at(all_preds[name], iid, thr[name])
            draw_boxes(ax, pb, color, ps)
            ax.set_xlabel(f'TP {int(row[f"{name}|tp"])} · FP {int(row[f"{name}|fp"])} · '
                          f'FN {int(row[f"{name}|fn"])}', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)



# MAIN

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(ANN_PATH, encoding='utf-8') as f:
        coco = json.load(f)
    coco_imgs = {i['id']: i for i in coco['images']}
    gt = load_gt(coco)

    all_preds = {}
    for name, slug, _ in MODELS:
        p = os.path.join(EVAL_DIR, f'{slug}_test_predictions.json')
        if not os.path.exists(p):
            raise FileNotFoundError(f'Prediction file not found: {p}')
        all_preds[name] = load_preds(p)

    thr = get_thresholds()
    print('Thresholds:', {k: round(v, 3) for k, v in thr.items()})

    df = compute_per_image(coco, gt, all_preds, thr)
    sanity_check(df, thr)
    df.to_csv(os.path.join(OUT_DIR, 'per_image_metrics.csv'), index=False)

    counts = df[df['n_gt'] > 0]['stratum'].value_counts().reindex(STRATA, fill_value=0)
    n_with_gt = int((df['n_gt'] > 0).sum())
    print(f'\nDifficulty groups of the {n_with_gt} images with ground truth:')
    for s in STRATA:
        print(f'  {s:7s}: {counts[s]:4d}  ({100 * counts[s] / max(n_with_gt, 1):.1f}%)')

    picked = draw_samples(df, max(N_MAIN, N_SUPP))

    main_entries = [(s, iid) for s in STRATA for iid in picked[s][:N_MAIN]]
    make_figure(main_entries, df, coco_imgs, gt, all_preds, thr,
                os.path.join(OUT_DIR, 'figure_qualitative_main.png'))

    for s in STRATA:
        entries = [(s, iid) for iid in picked[s][:N_SUPP]]
        if entries:
            make_figure(entries, df, coco_imgs, gt, all_preds, thr,
                        os.path.join(OUT_DIR, f'figure_qualitative_supp_{s.lower()}.png'))

    with open(os.path.join(OUT_DIR, 'selection_log.txt'), 'w', encoding='utf-8') as f:
        f.write(f'SEED={SEED}\nTHRESHOLD_MODE={THRESHOLD_MODE} (FIXED_CONF={FIXED_CONF})\n'
                f'IOU_THR={IOU_THR}\nDIFFICULTY_BY={DIFFICULTY_BY}\n'
                f'Easy: score >= {EASY_MIN}; Hard: score < {HARD_MAX}; Medium: otherwise\n'
                f'Thresholds: { {k: round(v, 3) for k, v in thr.items()} }\n\n')
        f.write('Group sizes (images with ground truth): ' +
                ', '.join(f'{s}={counts[s]}' for s in STRATA) + '\n\n')
        for s in STRATA:
            f.write(f'[{s}] selected (in order): ' +
                    ', '.join(coco_imgs[i]['file_name'] for i in picked[s][:N_SUPP]) + '\n')
        f.write('\nUsed in the main figure: ' +
                ', '.join(coco_imgs[i]['file_name'] for _, i in main_entries) + '\n')

    print(f'\nSaved: {OUT_DIR}')
    print('Main figure images:', [coco_imgs[i]['file_name'] for _, i in main_entries])


if __name__ == '__main__':
    main()