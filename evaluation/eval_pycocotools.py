# -*- coding: utf-8 -*-
"""
YOLO vs RF-DETR comparison: pycocotools (mAP, PR, F1-confidence) + latency benchmark.
All models are evaluated through predict().

Stage 1 (accuracy): conf=0.001, max 300 boxes -> mAP50-95 / mAP50 / mAP75, PR, F1-conf
Stage 2 (latency) : conf=0.5, warm-up, 3 trials, FP32, batch=1, end-to-end (test split only)

Set the root directory with the BENCH_BASE environment variable.

Usage:
  python eval_pycocotools.py --split test --models "YOLOv8n" "RF-DETR M" --reuse
"""

import os
import re
import gc
import json
import time
import argparse

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# CONFIGURATION

BASE = os.environ.get('BENCH_BASE', '.')

DATA_ROOT = os.path.join(BASE, 'data', 'COCO')
RESULTS_ROOT = os.path.join(BASE, 'results')

OUT_DIR = os.path.join(RESULTS_ROOT, 'evaluation_predict')
YOLO_ROOT = os.path.join(BASE, 'runs', 'train')

RUN_SPLITS = ['test']
RUN_MODELS = None              # None = all models
RUN_ACCURACY = True
RUN_LATENCY = True             # test split only
REUSE = False                  # True: reuse existing predictions JSON

MAX_DETS = 300
CONF_FOR_MAP = 0.001
CONF_FOR_LATENCY = 0.5
YOLO_NMS_IOU = 0.7
WARMUP_ITERS = 50
LATENCY_TRIALS = 3
COOLDOWN_SEC = 30
NUMERICAL_PRECISION = 'FP32'
BATCH_SIZE = 1
NUM_CLASSES = 1

RFDETR_CLASS_ID = 0            # class channel of the real object; None = no filtering

DATASET = {
    'val': {
        'ann': os.path.join(DATA_ROOT, 'valid', '_annotations.coco.json'),
        'img_dir': os.path.join(DATA_ROOT, 'valid'),
    },
    'test': {
        'ann': os.path.join(DATA_ROOT, 'test', '_annotations.coco.json'),
        'img_dir': os.path.join(DATA_ROOT, 'test'),
    },
}


def rf(folder):
    root = os.path.join(BASE, 'runs', folder)
    return {
        'weights': os.path.join(root, 'checkpoint_best_total.pth'),
        'train_path': os.path.join(root, 'metrics.csv')
    }


def yl(folder):
    root = os.path.join(YOLO_ROOT, folder)
    return {'weights': os.path.join(root, 'weights', 'best.pt'),
            'train_path': os.path.join(root, 'results.csv')}


MODELS = {
    'YOLOv8n': {'type': 'yolo', 'imgsz': 640,
                **yl('yolov8n_default_150ep'),
                'color': '#1f77b4', 'linestyle': '-'},
    'YOLO11m': {'type': 'yolo', 'imgsz': 640,
                **yl('yolo11m_default_150ep'),
                'color': '#ff7f0e', 'linestyle': '-'},
    'RF-DETR M': {'type': 'rf-detr', 'rfdetr_class': 'RFDETRMedium',
                  **rf('rfdetr_medium_default'),
                  'color': '#2ca02c', 'linestyle': '--'},
    'RF-DETR L': {'type': 'rf-detr', 'rfdetr_class': 'RFDETRLarge',
                  **rf('rfdetr_large_default'),
                  'color': '#d62728', 'linestyle': '--'},
    'RF-DETR 2XL': {'type': 'rf-detr', 'rfdetr_class': 'RFDETR2XLarge',
                    **rf('rfdetr_2xl_default'),
                    'color': '#9467bd', 'linestyle': '--'},
}

CONF_GRID = np.linspace(0.0, 1.0, 1001)
SEEN_CLASS_IDS = set()



# HELPERS

def slug(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def out_path(name, split, suffix):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, f'{slug(name)}_{split}_{suffix}')


def used_category_ids(coco_gt):
    # Ignores empty categories (e.g. Roboflow's unused id-0 super-category)
    return sorted({a['category_id'] for a in coco_gt.dataset['annotations']})


def check_image_size(path, info, counter):
    from PIL import Image
    with Image.open(path) as im:
        if im.size != (info['width'], info['height']):
            counter['size_mismatch'] += 1


def check_paths(model_names, split):
    missing = []
    ds = DATASET[split]
    for p in (ds['ann'], ds['img_dir']):
        if not os.path.exists(p):
            missing.append(p)
    for name in model_names:
        if not os.path.exists(MODELS[name]['weights']):
            missing.append(f'[{name}] {MODELS[name]["weights"]}')
        if not os.path.exists(MODELS[name]['train_path']):
            print(f'Warning: [{name}] training CSV not found: {MODELS[name]["train_path"]}')
    if missing:
        raise FileNotFoundError('Missing files:\n  ' + '\n  '.join(missing))


def cooldown(seconds=COOLDOWN_SEC):
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    if seconds > 0:
        print(f'   Waiting {seconds}s for the GPU to cool down...')
        time.sleep(seconds)


def get_gpu_temperature():
    try:
        import subprocess
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL).decode().strip().split('\n')[0]
        return int(out)
    except Exception:
        return None


def cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def count_params_m(model, cfg):
    # For YOLO this is the fused count after inference (matches Ultralytics)
    try:
        net = model.model if cfg['type'] == 'yolo' else model.model.model
        return sum(p.numel() for p in net.parameters()) / 1e6
    except Exception as e:
        print(f'  Could not get parameter count: {e}')
        return None



# MODEL LOADING AND INFERENCE

def get_rfdetr_class(class_name):
    import rfdetr
    if hasattr(rfdetr, class_name):
        return getattr(rfdetr, class_name)
    import rfdetr_plus          # 2XL is here in some versions
    return getattr(rfdetr_plus, class_name)


def load_model(cfg):
    if cfg['type'] == 'yolo':
        from ultralytics import YOLO
        return YOLO(cfg['weights'])
    cls = get_rfdetr_class(cfg['rfdetr_class'])
    return cls(pretrain_weights=cfg['weights'], num_classes=NUM_CLASSES)


def infer_one(model, cfg, path, conf):
    """End-to-end (image loading + inference + post-processing).
    Returns xyxy (N,4) in pixels and score (N,)."""
    if cfg['type'] == 'yolo':
        r = model.predict(path, conf=conf, iou=YOLO_NMS_IOU, imgsz=cfg['imgsz'],
                          max_det=MAX_DETS, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4)), np.zeros((0,))
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
    from PIL import Image
    img = Image.open(path).convert('RGB')
    det = model.predict(img, threshold=conf)
    if det is None or len(det.xyxy) == 0:
        return np.zeros((0, 4)), np.zeros((0,))
    cid = np.asarray(det.class_id)
    SEEN_CLASS_IDS.update(int(c) for c in cid)
    xyxy, score = np.asarray(det.xyxy), np.asarray(det.confidence)
    if RFDETR_CLASS_ID is not None:
        keep = cid == RFDETR_CLASS_ID
        xyxy, score = xyxy[keep], score[keep]
    return xyxy, score



# STAGE 1: ACCURACY

def predict_dataset(model, cfg, coco_gt, img_dir, cat_id):
    dets, counter = [], {'size_mismatch': 0}
    SEEN_CLASS_IDS.clear()
    for n, (img_id, info) in enumerate(sorted(coco_gt.imgs.items()), 1):
        path = os.path.join(img_dir, info['file_name'])
        check_image_size(path, info, counter)
        xyxy, score = infer_one(model, cfg, path, CONF_FOR_MAP)
        if len(score):
            order = np.argsort(-score)[:MAX_DETS]
            for (x1, y1, x2, y2), s in zip(xyxy[order], score[order]):
                dets.append({'image_id': int(img_id), 'category_id': int(cat_id),
                             'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                             'score': float(s)})
        if n % 50 == 0:
            print(f'  {n}/{len(coco_gt.imgs)} images')
    if cfg['type'] == 'rf-detr':
        print(f'  RF-DETR class_id values (before filtering): {sorted(SEEN_CLASS_IDS)} '
              f'-> only class_id={RFDETR_CLASS_ID} was used')
    return dets, counter


def evaluate(coco_gt, dets, used_cats):
    coco_dt = coco_gt.loadRes(dets)
    E = COCOeval(coco_gt, coco_dt, 'bbox')
    E.params.catIds = used_cats
    E.params.maxDets = [1, 10, MAX_DETS]
    E.evaluate()
    E.accumulate()
    E.summarize()
    assert np.isclose(E.params.iouThrs[0], 0.5)
    # pycocotools hard-codes maxDets=100 for stats[0]; with a custom maxDets it returns -1,
    # so mAP50-95 is recomputed here at maxDets=300.
    E.stats[0] = ap_all_ious(E)
    return E


def ap_all_ious(E):
    p = E.eval['precision'][:, :, :, 0, -1]       # [T, R, K, area=all, maxDets=300]
    p = p[p > -1]
    return float(np.mean(p)) if p.size else float('nan')


def pr_curve_interpolated(E):
    p = E.eval['precision'][0, :, :, 0, -1]        # IoU=0.5
    p = np.where(p < 0, np.nan, p)
    return E.params.recThrs, np.nanmean(p, axis=1)


def f1_confidence_curve(E, used_cats):
    """Precision, recall and F1 at each confidence threshold (IoU=0.5), no smoothing."""
    p = E.params
    I0, A0 = len(p.imgIds), len(p.areaRng)
    scores, is_tp, npos = [], [], 0
    for k, cid in enumerate(p.catIds):
        if cid not in used_cats:
            continue
        for i in range(I0):
            e = E.evalImgs[k * A0 * I0 + 0 * I0 + i]
            if e is None:
                continue
            keep = ~np.asarray(e['dtIgnore'][0], dtype=bool)
            scores.append(np.asarray(e['dtScores'])[keep])
            is_tp.append(np.asarray(e['dtMatches'][0])[keep] > 0)
            npos += int(np.sum(np.asarray(e['gtIgnore']) == 0))

    s = np.concatenate(scores) if scores else np.array([])
    t = np.concatenate(is_tp) if is_tp else np.array([], dtype=bool)
    order = np.argsort(-s, kind='mergesort')
    s, t = s[order], t[order]
    tp_cum = np.cumsum(t)

    n = np.searchsorted(-s, -CONF_GRID, side='right')
    TP = np.where(n > 0, tp_cum[np.maximum(n - 1, 0)] if len(tp_cum) else 0, 0)
    FP = n - TP
    FN = npos - TP
    prec = np.where(n > 0, TP / np.maximum(n, 1), np.nan)
    rec = TP / max(npos, 1)
    f1 = np.where(n > 0, 2 * np.nan_to_num(prec) * rec / (np.nan_to_num(prec) + rec + 1e-16), 0.0)
    return pd.DataFrame({'confidence': CONF_GRID, 'precision': prec, 'recall': rec, 'f1': f1,
                         'TP': TP, 'FP': FP, 'FN': FN}), npos


def load_summary(name, split):
    p = out_path(name, split, 'summary.json')
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    return {'model': name, 'split': split}


def save_summary(name, split, summary):
    with open(out_path(name, split, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def summarize_and_save(name, split, coco_gt, dets, used_cats, extra=None):
    if len(dets) == 0:
        raise RuntimeError(f'{name}: no predictions were produced. Check weights/threshold.')
    E = evaluate(coco_gt, dets, used_cats)
    rec, prec = pr_curve_interpolated(E)
    pd.DataFrame({'recall': rec, 'precision': prec}).to_csv(out_path(name, split, 'pr.csv'), index=False)

    f1df, npos = f1_confidence_curve(E, used_cats)
    f1df.to_csv(out_path(name, split, 'f1_conf.csv'), index=False)

    i = int(f1df['f1'].idxmax())
    st = E.stats
    summary = load_summary(name, split)       # keeps existing latency fields
    summary.update({
        'model': name, 'split': split,
        'n_images': len(coco_gt.imgs), 'n_gt': int(npos), 'n_dets': len(dets),
        'mAP50_95': float(st[0]), 'mAP50': float(st[1]), 'mAP75': float(st[2]),
        'AP_small': float(st[3]), 'AP_medium': float(st[4]), 'AP_large': float(st[5]),
        f'AR@{MAX_DETS}': float(st[8]),
        'best_f1': float(f1df.loc[i, 'f1']), 'best_f1_conf': float(f1df.loc[i, 'confidence']),
        'precision_at_best_f1': float(f1df.loc[i, 'precision']),
        'recall_at_best_f1': float(f1df.loc[i, 'recall']),
        'TP': int(f1df.loc[i, 'TP']), 'FP': int(f1df.loc[i, 'FP']), 'FN': int(f1df.loc[i, 'FN']),
    })
    if extra:
        summary.update(extra)
    save_summary(name, split, summary)
    return summary



# STAGE 2: LATENCY

def measure_latency(model, cfg, image_paths, name):
    """Pools per-image end-to-end latencies (ms) across all trials."""
    from tqdm import tqdm
    print(f'  Latency: warmup={WARMUP_ITERS}, trials={LATENCY_TRIALS}, conf={CONF_FOR_LATENCY}, '
          f'{NUMERICAL_PRECISION}, batch={BATCH_SIZE}, {len(image_paths)} images')

    for _ in range(WARMUP_ITERS):
        infer_one(model, cfg, image_paths[0], CONF_FOR_LATENCY)
    cuda_sync()

    all_lat, trial_means, temps = [], [], []
    for trial in range(LATENCY_TRIALS):
        if trial > 0:
            cooldown(COOLDOWN_SEC)
        t_before = get_gpu_temperature()
        lat = []
        for p in tqdm(image_paths, desc=f'{name} [latency T{trial + 1}/{LATENCY_TRIALS}]'):
            cuda_sync()
            t0 = time.perf_counter()
            infer_one(model, cfg, p, CONF_FOR_LATENCY)
            cuda_sync()
            lat.append(time.perf_counter() - t0)
        arr = np.array(lat) * 1000.0
        all_lat.append(arr)
        trial_means.append(float(arr.mean()))
        temps.append({'before': t_before, 'after': get_gpu_temperature()})
        print(f'    Trial {trial + 1}: mean={arr.mean():.2f} ms, median={np.median(arr):.2f} ms, '
              f'GPU {temps[-1]["before"]}->{temps[-1]["after"]} C')

    pooled = np.concatenate(all_lat)

    pd.DataFrame({
        'trial': np.repeat(np.arange(1, LATENCY_TRIALS + 1), len(image_paths)),
        'image': [os.path.basename(p) for p in image_paths] * LATENCY_TRIALS,
        'latency_ms': pooled,
    }).to_csv(out_path(name, 'test', 'latency_per_image.csv'), index=False)

    mean = float(pooled.mean())
    return {
        'avg_latency_ms': mean,
        'std_latency_ms': float(pooled.std(ddof=1)),
        'median_latency_ms': float(np.median(pooled)),
        'p95_latency_ms': float(np.percentile(pooled, 95)),
        'p99_latency_ms': float(np.percentile(pooled, 99)),
        'fps': 1000.0 / mean if mean > 0 else None,
        'n_latency_samples': int(pooled.size),
        'trial_means_ms': trial_means,
        'gpu_temperatures': temps,
        'latency_conf': CONF_FOR_LATENCY, 'latency_warmup': WARMUP_ITERS,
        'latency_trials': LATENCY_TRIALS, 'latency_precision': NUMERICAL_PRECISION,
        'latency_batch_size': BATCH_SIZE,
    }



# MAIN

def collect_summaries(split):
    rows = []
    for name in MODELS:
        p = out_path(name, split, 'summary.json')
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                rows.append(json.load(f))
    if rows:
        df = pd.DataFrame(rows)
        drop = [c for c in df.columns if c.startswith('trial_') or c == 'gpu_temperatures']
        df.drop(columns=drop).to_csv(os.path.join(OUT_DIR, f'summary_{split}.csv'), index=False)
        return df
    return None


def run_split(split, model_names, reuse):
    ds = DATASET[split]
    check_paths(model_names, split)
    coco_gt = COCO(ds['ann'])
    used = used_category_ids(coco_gt)
    if len(used) != 1:
        raise NotImplementedError(
            f'GT contains {len(used)} classes ({used}). This script is for single-class datasets.')
    cat_id = used[0]
    image_paths = [os.path.join(ds['img_dir'], info['file_name'])
                   for _, info in sorted(coco_gt.imgs.items())]
    print(f'\n[{split}] {len(coco_gt.imgs)} images, '
          f'{len(coco_gt.dataset["annotations"])} boxes, category id={cat_id}')

    do_latency = RUN_LATENCY and split == 'test'
    if RUN_LATENCY and not do_latency:
        print(f'  Note: latency is only measured on the test split; skipped for [{split}].')

    try:
        import torch
        if torch.cuda.is_available():
            print(f'GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}')
    except Exception:
        pass

    for i, name in enumerate(model_names):
        cfg = MODELS[name]
        if i > 0:
            cooldown(COOLDOWN_SEC)
        print(f'\n=== {name} ({cfg["weights"]}) ===')
        model = load_model(cfg)

        if RUN_ACCURACY:
            pred_file = out_path(name, split, 'predictions.json')
            extra = {}
            if reuse and os.path.exists(pred_file):
                print('  Using existing predictions')
                with open(pred_file, encoding='utf-8') as f:
                    dets = json.load(f)
            else:
                print(f'  Accuracy: generating predictions (conf={CONF_FOR_MAP})')
                dets, counter = predict_dataset(model, cfg, coco_gt, ds['img_dir'], cat_id)
                extra['pipeline'] = 'predict()'
                with open(pred_file, 'w', encoding='utf-8') as f:
                    json.dump(dets, f)
                if counter['size_mismatch']:
                    print(f'  WARNING: {counter["size_mismatch"]} images differ in size from the GT JSON '
                          f'(EXIF rotation?). Box coordinates may be shifted!')
                extra['size_mismatch'] = counter['size_mismatch']
            s = summarize_and_save(name, split, coco_gt, dets, used, extra)
            print(f'  mAP50-95={s["mAP50_95"]:.4f}  mAP50={s["mAP50"]:.4f}  '
                  f'bestF1={s["best_f1"]:.4f}@conf={s["best_f1_conf"]:.3f}')

        if do_latency:
            if RUN_ACCURACY:
                cooldown(COOLDOWN_SEC)
            lat = measure_latency(model, cfg, image_paths, name)
            summary = load_summary(name, split)
            summary.update(lat)
            save_summary(name, split, summary)
            print(f'  Latency = {lat["avg_latency_ms"]:.1f} +- {lat["std_latency_ms"]:.1f} ms '
                  f'({lat["fps"]:.1f} FPS)')

        params_m = count_params_m(model, cfg)
        if params_m is not None:
            summary = load_summary(name, split)
            summary['params_M'] = params_m
            save_summary(name, split, summary)
            print(f'  Parameter count = {params_m:.2f} M')

        del model
        gc.collect()

    df = collect_summaries(split)
    if df is not None:
        cols = [c for c in ['model', 'mAP50_95', 'mAP50', 'mAP75', 'best_f1', 'best_f1_conf',
                            'params_M', 'avg_latency_ms', 'std_latency_ms', 'fps'] if c in df.columns]
        print(f'\n=== Summary ({split}) ===')
        print(df[cols].round(4).to_string(index=False))
        print(f'\nSaved: {os.path.join(OUT_DIR, f"summary_{split}.csv")}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['val', 'test'], default=None)
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--reuse', action='store_true')
    args, _ = ap.parse_known_args()     # tolerate extra arguments from Spyder

    splits = [args.split] if args.split else RUN_SPLITS
    model_names = args.models if args.models else (RUN_MODELS or list(MODELS))
    reuse = args.reuse or REUSE

    os.makedirs(OUT_DIR, exist_ok=True)
    for split in splits:
        run_split(split, model_names, reuse)


if __name__ == '__main__':
    main()