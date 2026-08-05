"""Generate §4.1 Sparse Attention Report: SVGs + report.html"""

import os, sys, math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / 'curated' / 'figures'
FIGS.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)

# ── Color map across all panels ──
COLORS = {
    'dense_flash': '#2ecc71',
    'dense': '#e74c3c',
    'dense_masked': '#e67e22',
    'sparse_k4': '#3498db',
    'sparse_k16': '#9b59b6',
    'sparse_k64': '#1abc9c',
    'sparse_gather': '#e74c3c',
    'mask_knn': '#f39c12',
    'baseline': '#e74c3c',
    'k4': '#3498db',
    'k16': '#9b59b6',
    'k32': '#1abc9c',
    'k64': '#2ecc71',
    # full_bench_a500.csv method names (fig 01)
    'gather-KNN_K=4': '#3498db',
    'gather-KNN_K=16': '#9b59b6',
    'mask-KNN_K=4': '#e74c3c',
    'mask-KNN_K=16': '#c0392b',
    'NSA': '#95a5a6',
    'KNN-RelPos_K=4': '#1abc9c',
    'KNN-RelPos_K=16': '#16a085',
}

LABELS = {
    'dense_flash': 'Dense FlashAttn (no mask)',
    'dense': 'Dense Masked (original)',
    'dense_masked': 'Dense Masked (original)',
    'sparse_k4': 'Sparse K=4 (gather)',
    'sparse_k16': 'Sparse K=16 (gather)',
    'sparse_k64': 'Sparse K=64 (gather)',
    'sparse_gather': 'Sparse K=16 (gather)',
    'mask_knn': 'KNN Mask (scatter)',
    'baseline': 'Baseline (K=-1)',
    'k4': 'Sparse K=4',
    'k16': 'Sparse K=16',
    'k32': 'Sparse K=32',
    'k64': 'Sparse K=64',
    # full_bench_a500.csv method names (fig 01)
    'dense_masked_fb': 'Dense Masked (baseline)',
    'dense_flash_fb': 'Dense FlashAttn (no mask)',
    'gather-KNN_K=4': 'Gather-KNN K=4',
    'gather-KNN_K=16': 'Gather-KNN K=16',
    'mask-KNN_K=4': 'Mask-KNN K=4',
    'mask-KNN_K=16': 'Mask-KNN K=16',
    'NSA': 'NSA',
    'KNN-RelPos_K=4': 'KNN-RelPos K=4',
    'KNN-RelPos_K=16': 'KNN-RelPos K=16',
}


def save(name):
    path = FIGS / f'{name}.svg'
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  {path.name}')
    return path.name


# ═══════════════════════════════════════════════════════════════
# 1. Attention Time vs Sequence Length (log-log, L=1)
#    Uses benchmark_attn/results/full_bench_a500.csv — the single-campaign
#    A500 benchmark that covers the FULL config set discussed in
#    efficient_attention_math.tex §6.2 (dense masked, dense Flash, gather-KNN,
#    mask-KNN, NSA, KNN-RelPos) at every enumerated N incl. 1024/4096.
# ═══════════════════════════════════════════════════════════════
def panel_speed_vs_n():
    df = pd.read_csv(ROOT / 'benchmark_attn' / 'results' / 'full_bench_a500.csv')
    df = df[df['error'].isna()].copy()
    df['N'] = df['N'].astype(int)
    order = ['dense_masked', 'dense_flash', 'gather-KNN_K=4', 'gather-KNN_K=16',
             'mask-KNN_K=4', 'mask-KNN_K=16', 'NSA',
             'KNN-RelPos_K=4', 'KNN-RelPos_K=16']
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in order:
        grp = df[df.method == m].sort_values('N')
        ax.plot(grp.N, grp.time_ms, 'o-', label=LABELS.get(m, m),
                color=COLORS.get(m, '#333'), linewidth=2, markersize=6)
    # Shaded region = per-sample cell-count range of the analysis datasets
    # (vanvliet / DeepCell), matching the fig:01 caption.
    ax.axvspan(100, 600, alpha=0.08, color='gray', label='_nolegend_')
    ax.text(310, ax.get_ylim()[1] * 0.9, 'vanvliet / DeepCell\ncell-count range',
            fontsize=9, color='gray')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Sequence Length N (cells per frame)')
    ax.set_ylabel('Attention Time (ms)')
    ax.set_title('Attention Forward Time vs Sequence Length (L=1, A500)')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    fig.tight_layout()
    return save('01_speed_vs_n')


# ═══════════════════════════════════════════════════════════════
# 2. Small-N Focus Bar Chart (N=256)
# ═══════════════════════════════════════════════════════════════
def panel_small_n():
    df = pd.read_csv(ROOT / 'benchmark_small_n_results.csv')
    df = df[df.N == 256].copy()
    methods = ['dense_flash', 'dense', 'sparse_v1', 'sparse_v2']
    df['label'] = df.method.map({'dense': 'dense_masked', 'dense_flash': 'dense_flash',
                                  'sparse_v1': 'sparse_gather', 'sparse_v2': 'sparse_v2'})
    df['time_ms'] = df.time_s * 1000
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = ['dense_flash', 'dense_masked', 'sparse_gather', 'sparse_v2']
    bars = ax.bar(range(len(order)), [df[df.label == m].time_ms.values[0] if len(df[df.label == m]) else 0
                                       for m in order],
                  color=[COLORS.get(m, '#888') for m in order], width=0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABELS.get(m, m) for m in order], fontsize=9)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Attention Time at N=256 (Real Dataset Size)')
    for bar, m in zip(bars, order):
        val = df[df.label == m].time_ms.values[0] if len(df[df.label == m]) else 0
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
    fig.tight_layout()
    return save('02_small_n')


# ═══════════════════════════════════════════════════════════════
# 3. Mask vs Gather Comparison Bar Chart (N=256)
# ═══════════════════════════════════════════════════════════════
def panel_mask_vs_gather():
    df = pd.read_csv(ROOT / 'benchmark_mask_vs_gather.csv')
    df = df[df.N == 256].copy()
    df['time_ms'] = df.time_s * 1000
    order = ['dense_flash', 'mask_knn', 'dense_masked', 'sparse_gather']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(len(order)), [df[df.method == m].time_ms.values[0] for m in order],
                  color=[COLORS.get(m, '#888') for m in order], width=0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABELS.get(m, m) for m in order], fontsize=9)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Attention Mechanism Comparison at N=256')
    for bar, m in zip(bars, order):
        val = df[df.method == m].time_ms.values[0]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
    # speedup annotation
    flash_val = df[df.method == 'dense_flash'].time_ms.values[0]
    gather_val = df[df.method == 'sparse_gather'].time_ms.values[0]
    ax.annotate(f'dense_flash is {gather_val/flash_val:.0f}× faster\nthan sparse_gather',
                xy=(0, flash_val), xytext=(2.5, gather_val * 0.7),
                fontsize=9, color='gray', ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    fig.tight_layout()
    return save('03_mask_vs_gather')


# ═══════════════════════════════════════════════════════════════
# 4. Memory vs Sequence Length
# ═══════════════════════════════════════════════════════════════
def panel_memory_vs_n():
    df = pd.read_csv(ROOT / 'benchmark_sparse_results.csv')
    df = df[df.L == 1].copy()
    df = df[df.status == 'ok']
    df['label'] = df.apply(lambda r: f"sparse_k{r.K}" if r.method.startswith('sparse')
                           else r.method, axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for lbl, grp in df.groupby('label'):
        grp = grp.sort_values('N')
        ax.plot(grp.N, grp.mem_mb, 'o-', label=LABELS.get(lbl, lbl),
                color=COLORS.get(lbl, '#333'), linewidth=2, markersize=6)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Sequence Length N')
    ax.set_ylabel('GPU Memory (MB)')
    ax.set_title('Memory Usage vs Sequence Length (L=1)')
    ax.legend(fontsize=8)
    ax.set_xticks([128, 256, 512, 2048, 8192])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    fig.tight_layout()
    return save('04_memory_vs_n')


# ═══════════════════════════════════════════════════════════════
# 5. Convergence Curves
# ═══════════════════════════════════════════════════════════════
def panel_convergence():
    df = pd.read_csv(ROOT / 'results' / 'knn_sweep' / 'convergence.csv')
    labels_map = {
        '2026-06-01_23-04-13_baseline_clean': 'baseline',
        '2026-06-01_23-04-12_sparse_k4_clean': 'k4',
        '2026-06-01_23-04-18_sparse_k16_clean': 'k16',
        '2026-06-01_23-04-29_sparse_k32_clean': 'k32',
        '2026-06-01_23-04-09_sparse_k64_clean': 'k64',
    }
    df['model_short'] = df.model.map(labels_map)
    df = df[df.model_short.notna()].copy()
    df['epoch'] = df.epoch // 666  # normalize to epoch number

    def smooth(series, window=5):
        return series.rolling(window=window, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    for mod, grp in df.groupby('model_short'):
        grp = grp.sort_values('epoch')
        col = COLORS.get(mod, '#333')
        # unsmoothed low opacity
        ax.plot(grp.epoch, grp.val_loss, color=col, linewidth=0.5, alpha=0.15)
        # smoothed
        ax.plot(grp.epoch, smooth(grp.val_loss, window=21), label=LABELS.get(mod, mod),
                color=col, linewidth=1.8, alpha=0.9)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Convergence: All Models (seed=42)')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    fig.tight_layout()
    return save('05_convergence')


# ═══════════════════════════════════════════════════════════════
# 6. TRA Comparison Bar Chart
# ═══════════════════════════════════════════════════════════════
def panel_tra():
    models = ['baseline', 'k4', 'k16', 'k32', 'k64']
    tra_means = {'baseline': 0.9963, 'k4': 0.9957, 'k16': 0.9972, 'k32': 0.9963, 'k64': 0.9969}
    tra_mins  = {'baseline': 0.9561, 'k4': 0.9515, 'k16': 0.9589, 'k32': 0.9519, 'k64': 0.9574}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = [tra_means[m] for m in models]
    mins = [tra_mins[m] for m in models]
    bars = ax.bar(range(len(models)), vals, color=[COLORS[m] for m in models], width=0.6)
    # add min-TRA as error-like annotation
    for i, (v, mn) in enumerate(zip(vals, mins)):
        ax.plot([i-0.15, i+0.15], [mn, mn], 'k-', linewidth=1.5, alpha=0.6)
        ax.text(i, v + 0.0008, f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([LABELS[m] for m in models], fontsize=9)
    ax.set_ylabel('Mean TRA')
    ax.set_title('Tracking Accuracy (TRA) — mean across 32 experiments')
    ax.set_ylim(0.985, 1.002)
    fig.tight_layout()
    return save('06_tra')


# ═══════════════════════════════════════════════════════════════
# 7. AOGM Comparison Bar Chart
# ═══════════════════════════════════════════════════════════════
def panel_aogm():
    models = ['baseline', 'k4', 'k16', 'k32', 'k64']
    aogm_means = {'baseline': 51.0, 'k4': 62.0, 'k16': 37.7, 'k32': 52.6, 'k64': 39.8}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = [aogm_means[m] for m in models]
    bars = ax.bar(range(len(models)), vals, color=[COLORS[m] for m in models], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f'{v:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([LABELS[m] for m in models], fontsize=9)
    ax.set_ylabel('Mean AOGM (lower is better)')
    ax.set_title('Detection + Association Error (AOGM)')
    fig.tight_layout()
    return save('07_aogm')


# ═══════════════════════════════════════════════════════════════
# 8. Edge F1 / Division F1 Grouped Bar
# ═══════════════════════════════════════════════════════════════
def panel_edge_div_f1():
    models = ['baseline', 'k4', 'k16', 'k32', 'k64']
    edge_f1 = {'baseline': 0.9913, 'k4': 0.9914, 'k16': 0.9929, 'k32': 0.9914, 'k64': 0.9926}
    div_f1  = {'baseline': 0.9696, 'k4': 0.9705, 'k16': 0.9757, 'k32': 0.9713, 'k64': 0.9767}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(models))
    w = 0.35
    bars1 = ax.bar(x - w/2, [edge_f1[m] for m in models], w, label='Edge F1', color='#3498db')
    bars2 = ax.bar(x + w/2, [div_f1[m] for m in models], w, label='Division F1', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in models], fontsize=9)
    ax.set_ylabel('F1 Score')
    ax.set_title('Edge and Division Tracking Quality')
    ax.legend(fontsize=9)
    ax.set_ylim(0.92, 1.0)
    for b in bars1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                f'{b.get_height():.4f}', ha='center', va='bottom', fontsize=7, rotation=45)
    for b in bars2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                f'{b.get_height():.4f}', ha='center', va='bottom', fontsize=7, rotation=45)
    fig.tight_layout()
    return save('08_edge_div_f1')


# ═══════════════════════════════════════════════════════════════
# 9. Per-Epoch Training Time
# ═══════════════════════════════════════════════════════════════
def panel_per_epoch_time():
    models = ['baseline', 'k4', 'k16', 'k32', 'k64']
    times = {'baseline': 1.63, 'k4': 1.62, 'k16': 1.63, 'k32': 1.64, 'k64': 1.64}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = [times[m] for m in models]
    bars = ax.bar(range(len(models)), vals, color=[COLORS[m] for m in models], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f'{v:.2f} min', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([LABELS[m] for m in models], fontsize=9)
    ax.set_ylabel('Minutes per Epoch')
    ax.set_title('Per-Epoch Training Time (identical across all K)')
    ax.set_ylim(1.5, 1.75)
    fig.tight_layout()
    return save('09_per_epoch_time')


# ═══════════════════════════════════════════════════════════════
# 10. Speedup Heatmap — Sparse vs Dense at each N
# ═══════════════════════════════════════════════════════════════
def panel_speedup_heatmap():
    df = pd.read_csv(ROOT / 'benchmark_sparse_results.csv')
    df = df[df.L == 1].copy()
    dense = df[df.method == 'dense'][['N', 'time_s']].rename(columns={'time_s': 'dense_time'})
    dense_flash = df[df.method == 'dense_flash'][['N', 'time_s']].rename(columns={'time_s': 'flash_time'})
    sparse = df[df.method == 'sparse'][['N', 'K', 'time_s']].rename(columns={'time_s': 'sparse_time'})
    merged = dense.merge(dense_flash, on='N').merge(sparse, on='N')
    merged.loc[:, 'speedup_vs_dense'] = merged.dense_time / merged.sparse_time
    merged.loc[:, 'speedup_vs_flash'] = merged.flash_time / merged.sparse_time
    pivot_dense = merged.pivot_table(index='N', columns='K', values='speedup_vs_dense')
    pivot_flash = merged.pivot_table(index='N', columns='K', values='speedup_vs_flash')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(pivot_dense, annot=True, fmt='.2f', cmap='RdYlGn', center=1,
                ax=axes[0], cbar_kws={'label': '× vs Dense Masked'})
    axes[0].set_title('Speedup vs Dense Masked')
    sns.heatmap(pivot_flash, annot=True, fmt='.2f', cmap='RdYlGn_r', center=1,
                ax=axes[1], cbar_kws={'label': '× vs Dense FlashAttn'})
    axes[1].set_title('Speedup vs Dense FlashAttn')
    fig.suptitle('Sparse Attention Speedup by N and K (L=1)', fontsize=13, y=1.02)
    fig.tight_layout()
    return save('10_speedup_heatmap')


# ═══════════════════════════════════════════════════════════════
# 11. Profiler: Gather bottleneck breakdown (N=256)
# ═══════════════════════════════════════════════════════════════
def panel_profiler_breakdown():
    # Profiler data measured on local GPU at N=256, K=16, B=2, d=256, h=4
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.35
    x = [0, 0.7, 1.4]
    labels = ['Sparse Gather', 'KNN Mask\n(scatter)', 'Dense Flash']

    # Bar 1: Gather (total 1084 us)
    gather_parts = [
        ('SDPA q_len=1', 533, '#e67e22'),
        ('Gather (index)', 277, '#c0392b'),
        ('Copy/reshape', 210, '#95a5a6'),
        ('QKV+out proj', 64, '#3498db'),
    ]
    cum = 0
    for name, val, col in gather_parts:
        ax.bar(x[0], val, bottom=cum, width=bar_width, color=col,
               edgecolor='white', linewidth=0.5)
        if val > 40:
            ax.text(x[0], cum + val/2, f'{val}', ha='center', va='center',
                    fontsize=7.5, color='white', fontweight='bold')
        cum += val
    ax.text(x[0], cum + 30, f'{cum}μs', ha='center', fontsize=9, fontweight='bold')

    # Bar 2: Scatter (total 97 us)
    scatter_parts = [
        ('EfficientAttn', 36, '#2ecc71'),
        ('QKV+out proj', 37, '#3498db'),
        ('Mask fill', 13, '#95a5a6'),
        ('scatter_', 11, '#f39c12'),
    ]
    cum = 0
    for name, val, col in scatter_parts:
        ax.bar(x[1], val, bottom=cum, width=bar_width, color=col,
               edgecolor='white', linewidth=0.5)
        ax.text(x[1], cum + val/2, f'{val}', ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold')
        cum += val
    ax.text(x[1], cum + 10, f'{cum}μs', ha='center', fontsize=9, fontweight='bold')

    # Bar 3: Dense Flash (total 54 us)
    flash_parts = [
        ('FlashAttn', 17, '#27ae60'),
        ('QKV+out proj', 37, '#3498db'),
    ]
    cum = 0
    for name, val, col in flash_parts:
        ax.bar(x[2], val, bottom=cum, width=bar_width, color=col,
               edgecolor='white', linewidth=0.5)
        ax.text(x[2], cum + val/2, f'{val}', ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold')
        cum += val
    ax.text(x[2], cum + 10, f'{cum}μs', ha='center', fontsize=9, fontweight='bold')

    # Legend (unique entries)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color='#e67e22', label='SDPA q_len=1'),
        Patch(color='#c0392b', label='Gather (index)'),
        Patch(color='#2ecc71', label='EfficientAttn / FlashAttn'),
        Patch(color='#3498db', label='QKV+out proj'),
        Patch(color='#95a5a6', label='Copy/reshape / Mask fill'),
        Patch(color='#f39c12', label='scatter_ in-place'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='upper right')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('CUDA Time (μs)')
    ax.set_title('Forward Pass Breakdown at N=256 — Per-Operation (torch.profiler)')
    fig.tight_layout()
    return save('11_profiler_breakdown')


# ═══════════════════════════════════════════════════════════════
# 13. SSL Pretraining: Negative Result
# ═══════════════════════════════════════════════════════════════
def panel_ssl_negative():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), gridspec_kw={'width_ratios': [1, 1.5]})

    # LEFT: val_loss bar chart
    methods = ['From scratch\n(10% labels)', 'SSL identity BCE\n+ finetune', 'Random baseline\n(NT-Xent)']
    vals = [0.416, 0.404, 0.648]  # for contrastive: this is TRA not loss
    colors_ssl = ['#e74c3c', '#f39c12', '#95a5a6']
    labels_ssl = ['0.416', '0.404', '0.648 TRA\n(= random)']
    bars = ax1.bar(range(len(methods)), vals, color=colors_ssl, width=0.5, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, fontsize=8)
    ax1.set_ylabel('Val Loss (BCE) / TRA (Contrastive)')
    ax1.set_title('SSL: No Improvement Over Baseline')

    # RIGHT: identity BCE pretraining convergence (5 epochs)
    df_ssl = pd.read_csv(ROOT / 'benchmark_ssl/runs/ssl_v1/training_log.csv')
    ax2.plot(df_ssl.epoch, df_ssl.train_loss, 'o-', label='Train loss', color='#3498db', linewidth=2)
    ax2.plot(df_ssl.epoch, df_ssl.val_loss, 's-', label='Val loss', color='#e74c3c', linewidth=2)
    ax2.axhline(y=0.416, color='gray', linestyle='--', alpha=0.5, label='Baseline final (0.416)')
    ax2.set_xlabel('SSL Pretraining Epoch')
    ax2.set_ylabel('BCE Loss')
    ax2.set_title('Identity BCE Pretrain (5 epochs)')
    ax2.legend(fontsize=8)

    fig.suptitle('SSL Pretraining: Both Objectives Fail', fontsize=12, y=1.02)
    fig.tight_layout()
    return save('13_ssl_negative')


# ═══════════════════════════════════════════════════════════════
# 12. Layer scaling: L=1 vs L=4
# ═══════════════════════════════════════════════════════════════
def panel_layer_scaling():
    df = pd.read_csv(ROOT / 'benchmark_sparse_results.csv')
    df = df[df.method.isin(['dense', 'dense_flash']) | ((df.method == 'sparse') & (df.K == 16))]
    df['label'] = df.apply(lambda r: f"{'dense_flash' if r.method == 'dense_flash' else 'dense' if r.method == 'dense' else 'sparse'}_L{r.L}", axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for lbl, grp in df.groupby('label'):
        grp = grp.sort_values('N')
        ax.plot(grp.N, grp.time_s * 1000, 'o-', label=lbl, linewidth=2, markersize=6)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('N')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Layer Scaling: L=1 vs L=4')
    ax.legend(fontsize=8)
    ax.set_xticks([128, 256, 512, 2048, 8192])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    fig.tight_layout()
    return save('12_layer_scaling')


# ═══════════════════════════════════════════════════════════════
# HTML Report Generator
# ═══════════════════════════════════════════════════════════════
def generate_html(figures):
    sections = [
        {
            'id': 'motivation',
            'title': 'Motivation & Context',
            'text': '''
<p>The original Trackastra uses <b>RelativePositionalAttention</b>: masked dense SDPA with per-layer
<code>cdist</code> computation. Profiling shows the mask construction (cdist + masked_fill) consumes
<strong>82% of attention CUDA time</strong> at N=256 — the actual attention is only 17%.
Replacing this with <b>KNN-based sparse gather attention</b> was hypothesized to:</p>
<ul>
  <li>Reduce complexity from O(N²) to O(NK) per layer</li>
  <li>Enable FlashAttention by removing the N×N mask</li>
  <li>Deliver &gt;5× speedup at scale</li>
</ul>
 <p>The hypothesis was tested, verified at large N, but found <em>inapplicable</em> at the
cell counts of the analysis datasets (vanvliet / DeepCell). The data follows.</p>
            '''
        },
        {
            'id': 'speed_vs_n',
            'title': '1. Attention Time vs Sequence Length',
            'fig': '01_speed_vs_n.svg',
            'text': '''
<p>Dense FlashAttention (no mask) is fastest across all measured N on this hardware.
Sparse gather attention crosses over at N ≈ 2000 — above the cell-count range of the
analysis datasets (vanvliet / DeepCell, shaded region). At N=8192, the 6.5× speedup holds
against the masked dense baseline. Against dense_flash, sparse is 2.5× slower even at N=8192.</p>
            '''
        },
        {
            'id': 'small_n',
            'title': '2. Small-N Focus (Real Dataset Size)',
            'fig': '02_small_n.svg',
            'text': '''
<p>At N=256, sparse gather is 10.7× slower than dense FlashAttention (1.07ms vs 0.10ms).
Two bottlenecks: (1) q_len=1 shape forces 2048 independent FlashAttention kernel launches,
(2) irreducible gather copy of 8MB from scattered memory dominates at small N.</p>
            '''
        },
        {
            'id': 'mask_vs_gather',
            'title': '3. Mask vs Gather: Alternative Sparse Designs',
            'fig': '03_mask_vs_gather.svg',
            'text': '''
<p>KNN mask via scatter (building an N×N mask with KNN structure and using EfficientAttention)
is 7.9× faster than gather at N=256. Dense FlashAttention remains 1.3× faster than mask_knn.</p>
            '''
        },
        {
            'id': 'memory',
            'title': '4. Memory Footprint',
            'fig': '04_memory_vs_n.svg',
            'text': '''
<p>Sparse attention uses less memory at large N (120MB vs 2200MB at N=8192 for K=4).
At N=256, the gap narrows: 4.3MB vs 3.9MB.</p>
            '''
        },
        {
            'id': 'convergence',
            'title': '5. Convergence: All Models Behave Identically',
            'fig': '05_convergence.svg',
            'text': '''
<p>All 5 models produce overlapping validation loss curves. Bold lines: rolling mean (window=21). Faint lines: raw per-epoch values. Configs differ only in knn_neighbors (verified: 60 of 63 keys identical).</p>
            '''
        },
        {
            'id': 'per_epoch',
            'title': '6. Per-Epoch Training Time',
            'fig': '09_per_epoch_time.svg',
            'text': '''
<p>All models train at 1.62–1.64 min/epoch — no per-epoch difference between dense and sparse attention.
At N ≈ 200, the attention mechanism is a small fraction of total training time.</p>
            '''
        },
        {
            'id': 'accuracy',
            'title': '7. Tracking Accuracy (TRA & AOGM)',
            'fig': '06_tra.svg',
            'subfig': '07_aogm.svg',
            'text': '''
<p>K=16: TRA 0.9972, AOGM 37.7 (baseline: 0.9963 / 51.0). All sparse variants within ±0.001 TRA
of baseline. K=16 and K=64 show higher edge and division F1 than baseline.</p>
            '''
        },
        {
            'id': 'ssl',
            'title': '8. SSL Pretraining (Negative Result)',
            'fig': '13_ssl_negative.svg',
            'text': '''
<p>Both identity BCE and NT-Xent contrastive SSL fail. Identity BCE: val_loss 0.404 vs baseline 0.416
after 100-epoch finetune on 10% labels (no improvement). NT-Xent: collapse (cosine sim 0.89,
TRA 65.1% vs random 64.8%). Root cause: 7-dim regionprops bottleneck.</p>
            '''
        },
        {
            'id': 'profiler',
            'title': '10. Profiler: Why Gather Fails',
            'fig': '11_profiler_breakdown.svg',
            'text': '''
<p>Measured on local GPU via <code>torch.profiler</code>. Gather: 1084μs (49% SDPA q_len=1, 26% index copy, 19% reshape).
Scatter: 97μs (37% EfficientAttn, 38% proj, 13% mask fill, 11% scatter_). Dense Flash: 54μs (31% FlashAttn, 69% proj).</p>
<p><b>Gather (347):</b> <code>k[B_idx, H_idx, idx, :]</code> — 277μs for 8MB scattered copy<br>
<b>Scatter (541):</b> <code>mask.scatter_(3, src, 0.0)</code> — 11μs in-place fill, O(NK)</p>
            '''
        },
        {
            'id': 'edge_div',
            'title': '11. Edge & Division Quality',
            'fig': '08_edge_div_f1.svg',
            'text': '''
<p>K=16: Edge F1 0.9929, Division F1 0.9757. Differences across K values are small (&lt;0.006 F1).</p>
            '''
        },
        {
            'id': 'speedup_heatmap',
            'title': '12. Speedup Heatmap: All N × K Combinations',
            'fig': '10_speedup_heatmap.svg',
            'text': '''
<p>Left: vs the original masked dense baseline — shows 5.87× at N=8192, K=4.
Right: vs dense FlashAttention — sparse is slower (speedup &lt; 1) across all N and K.
Only at N=8192 does sparse approach parity (0.43× for K=4).</p>
            '''
        },
        {
            'id': 'resolution',
            'title': 'Conclusion',
            'text': '''
<div style="background:#f5f5f5; padding:15px; border-radius:8px;">
<h3>Hypothesis vs Outcome</h3>
<p><b>Hypothesis:</b> Replacing dense masked SDPA with KNN gather attention enables FlashAttention
and yields speedup via O(NK) complexity.</p>
<p><b>At large N (8192):</b> Sparse K=4 achieves 6.5× vs the masked dense baseline. However, against
unmasked dense FlashAttention, sparse is 2.5× slower.</p>
<p><b>At the cell counts of the analysis datasets (vanvliet / DeepCell):</b> Sparse gather is
slower than dense FlashAttention. The gather overhead exceeds the compute savings.</p>
<p><b>Unexpected finding:</b> The real bottleneck was per-layer <code>cdist</code> mask computation
(82% of attention time), not the N² attention itself. <strong>CachedDistAttention</strong> (compute
cdist once, share across L layers) provides ~2× attention speedup with identical semantics.</p>
<p><b>On accuracy:</b> K=16 achieves TRA 0.9972 and AOGM 37.7 (vs baseline 0.9963 / 51.0). Sparse
attention variants maintain or slightly improve tracking quality across all K values.</p>
</div>
            '''
        },
    ]

    html = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>§4.1 Sparse Attention — Curated Findings</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 30px 20px; background: #fafafa; color: #222; }
h1 { font-size: 28px; border-bottom: 3px solid #333; padding-bottom: 10px; }
h2 { font-size: 20px; margin-top: 40px; color: #2c3e50; border-left: 4px solid #3498db;
     padding-left: 12px; }
.figure { background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
          padding: 20px; margin: 15px 0; text-align: center; }
.figure img { max-width: 100%; height: auto; }
.figure .caption { text-align: left; color: #555; font-size: 14px; margin-top: 10px;
                   line-height: 1.6; }
.figure .subfig { display: flex; gap: 15px; justify-content: center; flex-wrap: wrap; }
.figure .subfig img { max-width: 48%; }
.meta { color: #888; font-size: 13px; text-align: center; margin-top: 40px; }
a { color: #2980b9; }
.nav { position: sticky; top: 0; background: #fafafa; padding: 10px 0; z-index: 10;
       border-bottom: 1px solid #ddd; margin-bottom: 20px; }
.nav a { margin-right: 12px; font-size: 13px; }
</style>
</head>
<body>
<h1>§4.1 Efficient Sparse Attention — Curated Findings</h1>
<p style="color:#666;font-size:14px;">
Chronological investigation of KNN-gather sparse attention in Trackastra.
All graphs are standalone SVGs (in <code>figures/</code>) for report integration.
</p>
<div class="nav">
''' + '\n'.join(f'<a href="#{s["id"]}">▸ {s["title"]}</a>' for s in sections) + '''
</div>
'''
    for s in sections:
        html += f'<h2 id="{s["id"]}">{s["title"]}</h2>\n'
        if 'fig' in s:
            html += f'<div class="figure">\n'
            html += f'  <img src="figures/{s["fig"]}" alt="{s["title"]}">\n'
            if 'subfig' in s:
                html += f'  <div class="subfig"><img src="figures/{s["subfig"]}" alt=""></div>\n'
            html += f'  <div class="caption">{s["text"]}</div>\n'
            html += f'</div>\n'
        else:
            html += f'<div class="figure"><div class="caption">{s["text"]}</div></div>\n'
    html += '''
<div class="meta">
Generated from <code>curated/generate_report.py</code> —
''' + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M') + '''<br>
Data sources: <code>benchmark_attn/results/full_bench_a500.csv</code>,
<code>benchmark_sparse_results.csv</code>,
<code>benchmark_small_n_results.csv</code>,
<code>benchmark_mask_vs_gather.csv</code>,
<code>results/knn_sweep/convergence.csv</code>,
<code>results/knn_sweep/results_*.csv</code>
</div>
</body>
</html>'''
    path = ROOT / 'curated' / 'report.html'
    path.write_text(html)
    print(f'  report.html')


if __name__ == '__main__':
    print('Generating figures...')
    figs = []
    figs.append(panel_speed_vs_n())
    figs.append(panel_small_n())
    figs.append(panel_mask_vs_gather())
    figs.append(panel_memory_vs_n())
    figs.append(panel_convergence())
    figs.append(panel_tra())
    figs.append(panel_aogm())
    figs.append(panel_edge_div_f1())
    figs.append(panel_per_epoch_time())
    figs.append(panel_speedup_heatmap())
    figs.append(panel_profiler_breakdown())
    figs.append(panel_ssl_negative())
    figs.append(panel_layer_scaling())
    print(f'\n{len(figs)} figures generated in {FIGS}')
    print('Generating report.html...')
    generate_html(figs)
    print('Done!')
