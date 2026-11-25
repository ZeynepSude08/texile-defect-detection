# -*- coding: utf-8 -*-
"""
Figure 6: (a) YOLO loss, (b) RF-DETR loss, (c) validation mAP50-95   [training logs]
Figure 7: (a) Precision-recall (IoU=0.5), (b) F1-confidence          [pycocotools, test set]
Figure 8: Accuracy-latency trade-off (mAP50-95 vs ms)
Tables  : results_table_test.csv (accuracy), efficiency_table_test.csv (latency / parameters)

Run first:  python evaluation/eval_pycocotools.py --split test
This file must be in the same folder as eval_pycocotools.py.
Set the root directory with the BENCH_BASE environment variable.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import eval_pycocotools as ev

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

SPLIT = 'test'
FIG_DIR = os.path.join(ev.RESULTS_ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(ev.OUT_DIR, exist_ok=True)


def clean_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def first_col(df, names):
    for c in names:
        if c in df.columns and df[c].notna().any():
            return df[c]
    return None


def read_train_csv(name, cfg):
    if not os.path.exists(cfg['train_path']):
        print(f'Warning: [{name}] {cfg["train_path"]} not found, skipping.')
        return None
    df = pd.read_csv(cfg['train_path'])
    df.columns = df.columns.str.strip()

    if cfg['type'] == 'yolo':
        out = pd.DataFrame({'epoch': df['epoch']})
        out['train_loss'] = df[['train/box_loss', 'train/cls_loss', 'train/dfl_loss']].sum(axis=1)
        out['val_loss'] = df[['val/box_loss', 'val/cls_loss', 'val/dfl_loss']].sum(axis=1)
        out['mAP50_95'] = df['metrics/mAP50-95(B)']
        return out

    # RF-DETR: train and val values may sit on different rows of the same epoch,
    # so take the first non-null value per epoch.
    cols = {
        'train_loss': first_col(df, ['train/loss']),
        'val_loss': first_col(df, ['val/loss']),
        'mAP_reg': first_col(df, ['val/mAP_50_95']),
        'mAP_ema': first_col(df, ['val/ema_mAP_50_95']),
    }
    out = pd.DataFrame({'epoch': df['epoch']})
    for k, v in cols.items():
        out[k] = v if v is not None else np.nan
    if all(v is None for v in cols.values()):
        print(f'Warning: [{name}] expected columns not found. Available columns: {list(df.columns)}')
        return None
    out = out.groupby('epoch', as_index=False).first()
    # best_total checkpoint picks the better of regular/EMA weights -> per-epoch maximum
    out['mAP50_95'] = out[['mAP_reg', 'mAP_ema']].max(axis=1)
    out['epoch'] = out['epoch'] + 1          # RF-DETR starts at 0; align with YOLO
    return out



# FIGURE 6

data = {n: read_train_csv(n, c) for n, c in ev.MODELS.items()}

fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(18, 5))
for name, df in data.items():
    if df is None:
        continue
    cfg = ev.MODELS[name]
    ax = axa if cfg['type'] == 'yolo' else axb
    ax.plot(df['epoch'], df['train_loss'], color=cfg['color'], ls='-', lw=1.8, label=f'{name} train')
    ax.plot(df['epoch'], df['val_loss'], color=cfg['color'], ls='--', lw=1.8, label=f'{name} val')
    axc.plot(df['epoch'], df['mAP50_95'], color=cfg['color'], lw=1.8, label=name)

axa.set_title('(a) YOLO loss (box+cls+dfl)', fontsize=13, pad=12)
axb.set_title('(b) RF-DETR loss', fontsize=13, pad=12)
axc.set_title('(c) Validation mAP50-95', fontsize=13, pad=12)
for ax in (axa, axb, axc):
    ax.set_xlabel('Epoch')
    clean_spines(ax)
axa.set_ylabel('Loss')
axb.set_ylabel('Loss')
axc.set_ylabel('mAP50-95')
axc.set_ylim(0, 1)
# The very large first-epoch val loss would squash the axis, so clip relative to train loss
_ytop = 1.7 * max(d['train_loss'].max() for n, d in data.items()
                   if d is not None and ev.MODELS[n]['type'] == 'yolo')
axa.set_ylim(0, _ytop)
axb.xaxis.set_major_locator(MaxNLocator(integer=True))
axa.legend(frameon=False, fontsize=9)
axb.legend(frameon=False, fontsize=9)
axc.legend(frameon=False, fontsize=10, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figure6_loss_map.png'), dpi=300, bbox_inches='tight')
plt.show()


# FIGURE 7

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(13, 5))
for name, cfg in ev.MODELS.items():
    pr_p, f1_p = ev.out_path(name, SPLIT, 'pr.csv'), ev.out_path(name, SPLIT, 'f1_conf.csv')
    if not (os.path.exists(pr_p) and os.path.exists(f1_p)):
        print(f'Warning: {SPLIT} outputs for {name} not found, skipping.')
        continue
    pr, f1 = pd.read_csv(pr_p), pd.read_csv(f1_p)
    pr = pr[pr['precision'] > 0]          # drop points beyond the maximum reached recall
    ax3.plot(pr['recall'], pr['precision'], color=cfg['color'], ls=cfg['linestyle'], lw=1.8, label=name)
    ax4.plot(f1['confidence'], f1['f1'], color=cfg['color'], ls=cfg['linestyle'], lw=1.8, label=name)

ax3.set_title('(a) Precision-recall (IoU = 0.5)', fontsize=13, pad=12)
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_xlim(0, 1.02)
ax3.set_ylim(0, 1.02)
ax4.set_title('(b) F1-confidence curve', fontsize=13, pad=12)
ax4.set_xlabel('Confidence')
ax4.set_ylabel('F1')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1.02)
for ax in (ax3, ax4):
    clean_spines(ax)
ax3.legend(frameon=False, fontsize=10, loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figure7_pr_f1_conf_test.png'), dpi=300, bbox_inches='tight')
plt.show()


# TABLES + FIGURE 8

df = ev.collect_summaries(SPLIT)
if df is not None:
    cols = ['model', 'mAP50_95', 'mAP50', 'mAP75', 'precision_at_best_f1', 'recall_at_best_f1',
            'best_f1', 'best_f1_conf', 'AP_small', 'AP_medium', 'AP_large']
    table = df[[c for c in cols if c in df]].round(4)
    table.to_csv(os.path.join(ev.OUT_DIR, 'results_table_test.csv'), index=False)
    print('\n=== Accuracy table (test, pycocotools) ===')
    print(table.to_string(index=False))

    ecols = ['model', 'params_M', 'avg_latency_ms', 'median_latency_ms', 'std_latency_ms', 'fps']
    etable = df[[c for c in ecols if c in df]].round(2)
    etable.to_csv(os.path.join(ev.OUT_DIR, 'efficiency_table_test.csv'), index=False)
    print('\n=== Efficiency table (test, FP32, batch=1, end-to-end) ===')
    print(etable.to_string(index=False))

    if {'avg_latency_ms', 'mAP50_95'}.issubset(df.columns):
        COL_YOLO = '#2A7BDB'
        COL_DETR = '#F06B35'
        GRID_C = '#e6e6e6'
        SPINE_C = '#888888'
        TXT_C = '#333333'

        # Label placement: (x_offset_pt, y_offset_pt, ha). Keys must match the names in ev.MODELS.
        ABOVE = (0, 14, 'center')
        BELOW_RIGHT = (10, -22, 'left')
        LABEL_POS = {
            'YOLOv8n': ABOVE,
            'YOLO11m': BELOW_RIGHT,
            'RF-DETR M': (0, 20, 'center'),
            'RF-DETR L': BELOW_RIGHT,
            'RF-DETR 2XL': (6, 16, 'center'),
        }

        scale = 100 if df['mAP50_95'].dropna().max() <= 1.0 else 1    # convert to percent

        fig3, ax5 = plt.subplots(figsize=(8.5, 5.2))
        for _, r in df.iterrows():
            cfg = ev.MODELS.get(r['model'])
            if cfg is None or pd.isna(r.get('avg_latency_ms')):
                continue
            is_yolo = cfg['type'] == 'yolo'
            x, y = r['avg_latency_ms'], r['mAP50_95'] * scale
            ax5.scatter(x, y, s=130, color=COL_YOLO if is_yolo else COL_DETR,
                        marker='o' if is_yolo else 's', edgecolor='none', zorder=3)
            dx, dy, ha = LABEL_POS.get(r['model'], ABOVE)
            ax5.annotate(r['model'], (x, y), textcoords='offset points',
                         xytext=(dx, dy), ha=ha, fontsize=12, color=TXT_C)

        x_max = int(np.ceil(df['avg_latency_ms'].max() * 1.1 / 10.0) * 10)
        ax5.set_xlim(0, x_max)
        ax5.set_ylim(15, 65)
        ax5.set_xticks(range(0, x_max + 1, 10))
        ax5.set_yticks(range(20, 61, 10))
        ax5.tick_params(axis='both', labelsize=13, color=SPINE_C)
        ax5.set_xlabel('Mean inference latency (ms/image)', fontsize=14)
        ax5.set_ylabel('mAP50-95 (%)', fontsize=14)
        ax5.grid(True, color=GRID_C, linewidth=1.0, zorder=0)
        ax5.set_axisbelow(True)
        clean_spines(ax5)
        for s in ('left', 'bottom'):
            ax5.spines[s].set_color(SPINE_C)
            ax5.spines[s].set_linewidth(1.2)

        ax5.legend(handles=[
            Line2D([], [], marker='o', ls='', color=COL_YOLO, markersize=11, label='CNN (YOLO)'),
            Line2D([], [], marker='s', ls='', color=COL_DETR, markersize=11,
                   label='Vision Transformer (RF-DETR)')],
            frameon=False, loc='lower right', fontsize=13, handletextpad=1.2, labelspacing=0.9)

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'figure8_accuracy_vs_latency.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()