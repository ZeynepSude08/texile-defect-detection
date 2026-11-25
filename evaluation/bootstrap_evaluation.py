# -*- coding: utf-8 -*-
"""
Paired bootstrap confidence intervals on the test set (pycocotools mAP).

Per-image detection matches are computed once with pycocotools; test images are then
resampled with replacement (B times), applying the same resample to all models.
Outputs per-model 95% CIs and pairwise differences (A - B) with bootstrap p-values
(Holm correction over the pre-specified primary pairs).

Requires the *_test_predictions.json files written by eval_pycocotools.py.
Set the root directory with the BENCH_BASE environment variable.
"""
import os
import io
import json
import time
import itertools
import contextlib

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# SETTINGS (fix these before looking at results)

BASE = os.environ.get('BENCH_BASE', '.')

DATA_ROOT = os.path.join(BASE, 'data', 'COCO')
RESULTS_ROOT = os.path.join(BASE, 'results')

EVAL_DIR = os.path.join(RESULTS_ROOT, 'evaluation_predict')   # same as OUT_DIR in eval_pycocotools.py
ANN_PATH = os.path.join(DATA_ROOT, 'test', '_annotations.coco.json')
OUT_DIR = os.path.join(RESULTS_ROOT, 'bootstrap_results')

B = 2000
SEED = 42
CI_LEVEL = 95
MAX_DETS = 300          

# (display name, prediction-file slug, model name in summary_test.csv or None)
MODELS = [
    ('RF-DETR 2XL', 'rf_detr_2xl', 'RF-DETR 2XL'),
    ('RF-DETR L',   'rf_detr_l',   'RF-DETR L'),
    ('RF-DETR M',   'rf_detr_m',   'RF-DETR M'),
    ('YOLOv8n',     'yolov8n',     'YOLOv8n'),
    ('YOLO11m',     'yolo11m',     'YOLO11m'),
]

# Pre-specified primary comparisons (A - B); Holm correction applies to these only.
PRIMARY_PAIRS = [
    ('RF-DETR 2XL', 'RF-DETR M'),
    ('RF-DETR M',   'RF-DETR L'),
    ('RF-DETR 2XL', 'RF-DETR L'),
    ('RF-DETR L',   'YOLO11m'),      # closest cross-family pair (weakest RF-DETR vs strongest YOLO)
    ('YOLO11m',     'YOLOv8n'),
]



# 1) Per-image data from pycocotools matching

def run_cocoeval(coco_gt, dets, used_cats):
    coco_dt = coco_gt.loadRes(dets)
    E = COCOeval(coco_gt, coco_dt, 'bbox')
    E.params.catIds = used_cats
    E.params.maxDets = [1, 10, MAX_DETS]
    with contextlib.redirect_stdout(io.StringIO()):
        E.evaluate()
        E.accumulate()
    return E


def extract_arrays(E):
    """Flatten per-image matches into globally score-sorted arrays.
    Returns dict(tp [T,ND], fp [T,ND], det_img [ND], n_gt [N], ref_ap [T])."""
    p = E.params
    n_img = len(p.imgIds)
    n_area = len(p.areaRng)
    n_thr = len(p.iouThrs)

    scores, tps, fps, det_img = [], [], [], []
    n_gt = np.zeros(n_img)
    for i in range(n_img):
        e = E.evalImgs[0 * n_area * n_img + 0 * n_img + i]     # single category, area='all'
        if e is None:
            continue
        n_gt[i] = int(np.sum(np.asarray(e['gtIgnore']) == 0))
        s = np.asarray(e['dtScores'], dtype=float)
        if len(s) == 0:
            continue
        dtm = np.asarray(e['dtMatches'])
        dig = np.asarray(e['dtIgnore'], dtype=bool)
        scores.append(s)
        tps.append(np.logical_and(dtm, ~dig).astype(float))
        fps.append(np.logical_and(~dtm.astype(bool), ~dig).astype(float))
        det_img.append(np.full(len(s), i))

    scores = np.concatenate(scores)
    order = np.argsort(-scores, kind='mergesort')               # same ordering as pycocotools
    out = {
        'tp': np.concatenate(tps, axis=1)[:, order],
        'fp': np.concatenate(fps, axis=1)[:, order],
        'det_img': np.concatenate(det_img)[order],
        'n_gt': n_gt,
        'n_thr': n_thr,
        'rec_thr': np.asarray(p.recThrs),
    }
    prec = E.eval['precision'][:, :, 0, 0, -1]                  # [T, R]: category 0, area=all, maxDets=300
    out['ref_ap'] = np.where(prec > -1, prec, 0).mean(axis=1)
    return out



# 2) Weighted AP (pycocotools accumulate logic)

def weighted_ap(d, w_img):
    """w_img: per-image resample counts (0, 1, 2, ...). Returns AP per IoU threshold [T]."""
    w_det = w_img[d['det_img']]
    npig = float(np.sum(w_img * d['n_gt']))
    n_det = d['tp'].shape[1]
    tp_cum = np.cumsum(d['tp'] * w_det, axis=1)
    fp_cum = np.cumsum(d['fp'] * w_det, axis=1)
    rc = tp_cum / npig
    pr = tp_cum / (fp_cum + tp_cum + np.spacing(1))
    pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]    # precision envelope
    ap = np.zeros(d['n_thr'])
    for t in range(d['n_thr']):
        inds = np.searchsorted(rc[t], d['rec_thr'], side='left')
        valid = inds < n_det
        q = np.zeros(len(d['rec_thr']))
        q[valid] = pr[t, inds[valid]]
        ap[t] = q.mean()
    return ap



# 3) Statistics helpers

def ci_bounds(x):
    a = (100 - CI_LEVEL) / 2
    return np.percentile(x, a), np.percentile(x, 100 - a)


def boot_p(diff):
    """Two-sided bootstrap p-value with +1 correction (lower bound ~1/(B+1))."""
    n = len(diff)
    lo = (np.sum(diff <= 0) + 1) / (n + 1)
    hi = (np.sum(diff >= 0) + 1) / (n + 1)
    return min(1.0, 2 * min(lo, hi))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.zeros(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj



# MAIN

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model_names = {m[0] for m in MODELS}
    for a, b_ in PRIMARY_PAIRS:
        assert a in model_names and b_ in model_names, f'PRIMARY_PAIRS contains a model not in MODELS: {a}, {b_}'
    coco_gt = COCO(ANN_PATH)
    used_cats = sorted({a['category_id'] for a in coco_gt.dataset['annotations']})
    assert len(used_cats) == 1, f'Single-class data expected, found: {used_cats}'
    if min(a['id'] for a in coco_gt.dataset['annotations']) == 0:
        print('WARNING: a GT annotation has id=0. pycocotools treats a matched GT id of 0 as unmatched, '
              'which can affect this one object (eval_pycocotools.py behaves the same). '
              'Consider starting annotation ids at 1.')

    n_img = len(coco_gt.getImgIds())
    print(f'Test: {n_img} images, {len(coco_gt.dataset["annotations"])} objects')

    # ---- Matching per model (once) ----
    data = {}
    for name, slug, _ in MODELS:
        path = os.path.join(EVAL_DIR, f'{slug}_test_predictions.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Prediction file not found: {path}')
        with open(path, encoding='utf-8') as f:
            dets = json.load(f)
        t0 = time.time()
        E = run_cocoeval(coco_gt, dets, used_cats)
        data[name] = extract_arrays(E)
        print(f'  {name:18s} {len(dets):7d} detections, matching {time.time() - t0:.1f} s')

    # ---- Sanity check: full sample must equal pycocotools ----
    print('\nSanity check (all weights 1):')
    ones = np.ones(n_img)
    summ = None
    sp = os.path.join(EVAL_DIR, 'summary_test.csv')
    if os.path.exists(sp):
        summ = pd.read_csv(sp).set_index('model')
    point = {}
    for name, _, sname in MODELS:
        ap_t = weighted_ap(data[name], ones)
        ref = data[name]['ref_ap']
        assert np.allclose(ap_t, ref, atol=1e-9), f'{name}: weighted AP does not match pycocotools!'
        point[name] = (ap_t.mean(), ap_t[0])
        msg = f'  {name:18s} mAP50-95={ap_t.mean():.4f}  mAP50={ap_t[0]:.4f}  [matches pycocotools]'
        if summ is not None and sname in summ.index and 'mAP50_95' in summ.columns:
            diff = abs(ap_t.mean() - float(summ.loc[sname, 'mAP50_95']))
            msg += f'  | diff vs summary_test.csv: {diff:.1e}'
        print(msg)

    # ---- Bootstrap ----
    rng = np.random.default_rng(SEED)
    names = [m[0] for m in MODELS]
    samples = {n: np.zeros((B, 2)) for n in names}              # [:,0]=mAP50-95, [:,1]=mAP50
    t0 = time.time()
    for b in range(B):
        idx = rng.integers(0, n_img, n_img)
        w_img = np.bincount(idx, minlength=n_img).astype(float)
        for n in names:                                         # same resample for all models (paired)
            ap_t = weighted_ap(data[n], w_img)
            samples[n][b] = (ap_t.mean(), ap_t[0])
        if (b + 1) % 100 == 0:
            el = time.time() - t0
            print(f'  bootstrap {b + 1}/{B}  (elapsed {el / 60:.1f} min, remaining ~{el / (b + 1) * (B - b - 1) / 60:.1f} min)')

    np.savez(os.path.join(OUT_DIR, 'bootstrap_samples.npz'),
             **{n.replace(' ', '_'): samples[n] for n in names})

    # ---- Per-model confidence intervals ----
    rows = []
    for n in names:
        lo95, hi95 = ci_bounds(samples[n][:, 0])
        lo50, hi50 = ci_bounds(samples[n][:, 1])
        rows.append({'model': n,
                     'mAP50_95': point[n][0], 'mAP50_95_ci_lo': lo95, 'mAP50_95_ci_hi': hi95,
                     'mAP50': point[n][1], 'mAP50_ci_lo': lo50, 'mAP50_ci_hi': hi50})
    df_model = pd.DataFrame(rows)
    df_model.to_csv(os.path.join(OUT_DIR, 'bootstrap_model_ci.csv'), index=False)

    # ---- Pairwise differences (A - B) ----
    primary = [tuple(p) for p in PRIMARY_PAIRS]
    primary_set = {frozenset(p) for p in primary}
    pairs = list(primary)
    for a, b_ in itertools.combinations(names, 2):
        if frozenset((a, b_)) not in primary_set:
            pairs.append((a, b_))                               # exploratory 

    prow = []
    for a, b_ in pairs:
        for k, metric in enumerate(['mAP50_95', 'mAP50']):
            diff = samples[a][:, k] - samples[b_][:, k]
            lo, hi = ci_bounds(diff)
            prow.append({'A': a, 'B': b_, 'metric': metric,
                         'diff_A_minus_B': point[a][k] - point[b_][k],
                         'ci_lo': lo, 'ci_hi': hi,
                         'ci_excludes_0': bool(lo > 0 or hi < 0),
                         'p_boot': boot_p(diff),
                         'primary': frozenset((a, b_)) in primary_set})
    df_pair = pd.DataFrame(prow)
    for metric in ['mAP50_95', 'mAP50']:
        m = (df_pair['primary']) & (df_pair['metric'] == metric)    # Holm over primary pairs only
        df_pair.loc[m, 'p_holm'] = holm(df_pair.loc[m, 'p_boot'].values)
        ma = df_pair['metric'] == metric                            # Holm over all pairs (more conservative)
        df_pair.loc[ma, 'p_holm_all'] = holm(df_pair.loc[ma, 'p_boot'].values)
    df_pair.to_csv(os.path.join(OUT_DIR, 'bootstrap_pairwise_diff.csv'), index=False)

    # ---- Console summary ----
    print(f'\n=== Per-model mAP50-95 ({CI_LEVEL}% bootstrap interval, B={B}) ===')
    for _, r in df_model.iterrows():
        print(f'  {r["model"]:18s} {r["mAP50_95"]:.3f}  [{r["mAP50_95_ci_lo"]:.3f}, {r["mAP50_95_ci_hi"]:.3f}]'
              f'   mAP50 {r["mAP50"]:.3f}  [{r["mAP50_ci_lo"]:.3f}, {r["mAP50_ci_hi"]:.3f}]')

    print(f'\n=== Primary comparisons, mAP50-95 difference (A - B), {CI_LEVEL}% interval ===')
    sub = df_pair[(df_pair['primary']) & (df_pair['metric'] == 'mAP50_95')]
    n_primary = len(primary)
    n_all = len(pairs)
    for _, r in sub.iterrows():
        print(f'  {r["A"]:18s} - {r["B"]:18s} {r["diff_A_minus_B"]:+.3f}  [{r["ci_lo"]:+.3f}, {r["ci_hi"]:+.3f}]'
              f'  excludes 0: {"YES" if r["ci_excludes_0"] else "no"}   '
              f'p_holm({n_primary} pairs)={r["p_holm"]:.3f}   p_holm_all({n_all} pairs)={r["p_holm_all"]:.3f}')
    print(f'\nSaved: {OUT_DIR}')


if __name__ == '__main__':
    main()