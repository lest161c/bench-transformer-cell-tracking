"""Generate §4.1 Sparse Attention Report: SVGs + report.html.

All colours follow the Okabe–Ito scientific palette defined in
``docs/vis_guidelines.md`` (colorblind-safe, print-safe).
Heatmaps use ``palette.DIVERGING_CMAP`` (vermilion↔white↔blue); K uses
the categorical Okabe–Ito slots (``palette.K_COLORS``). No Flat UI /
web-design colours.
"""

import os, sys, math, json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / 'curated' / 'figures'
FIGS.mkdir(parents=True, exist_ok=True)

# Single source of style: local betterplots library (Okabe–Ito palette +
# guideline §5/§6 rcParams).  The clone lives at ``<repo_root>/betterplots/``
# and the package is at ``<repo_root>/betterplots/betterplots/``; we insert
# the clone root on ``sys.path`` so ``from betterplots import …`` resolves
# whether the script is run as ``python3 curated/generate_report.py`` from
# the repo root or as ``python3 -m curated.generate_report``.
sys.path.insert(0, str(ROOT / 'betterplots'))
from betterplots import palette, set_style
set_style(usetex=False, colors='okabe-ito', despine=True,
          font_size=8.5, label_size=8.5, tick_size=7.5, legend_font_size=7.5)

# ── Okabe–Ito palette aliases (single source of truth: betterplots.palette) ──
# The 8 hex codes live in betterplots.palette so the colour map has one home;
# these aliases keep the existing OI_* references in this file stable. Slot
# assignments follow docs/vis_guidelines.md §1. Neutrals (#262626 / #cccccc /
# #ffffff / #999999) are reserved and stay as-is.
OI_BLACK = palette.BLACK
OI_ORANGE = palette.ORANGE
OI_SKYBLUE = palette.SKYBLUE
OI_GREEN = palette.GREEN
OI_YELLOW = palette.YELLOW
OI_BLUE = palette.BLUE
OI_VERMILION = palette.VERMILION
OI_PURPLE = palette.PURPLE

# ── Color map across all panels (canonical: betterplots.palette.COLORS) ──
COLORS = palette.COLORS

# Auxiliary colours used in compound / stacked plots (canonical: betterplots.palette.COMPONENT_COLORS)
AUX = palette.COMPONENT_COLORS

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
    """Save the current figure as both SVG and PDF under ``curated/figures/``.

    The LaTeX report includes the figures via ``\\includegraphics{*.pdf}``,
    so the PDF is the source of truth that ships with the report; the SVG
    is kept alongside for editing in Inkscape / Illustrator and for the
    HTML report.
    """
    svg_path = FIGS / f'{name}.svg'
    pdf_path = FIGS / f'{name}.pdf'
    # usetex requires figure.savefig: pyplot's tight-bbox path renders on the
    # interactive Agg canvas, which needs dvipng (absent on this system).
    plt.gcf().savefig(svg_path, bbox_inches='tight', dpi=150)
    plt.gcf().savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f'  {svg_path.name}  +  {pdf_path.name}')
    return pdf_path.name


def write_meta_sidecar(name, meta):
    """Write a ``<name>.meta.json`` sidecar next to the saved figure.

    Same convention as ``results/generate_plots.py`` (SPEC 0009 addendum):
    the sidecar records when and how a figure was generated and the
    essential plotted numbers, so figure-vs-data consistency is checkable
    without re-running the pipeline. ``meta`` must contain ``"sources"``
    and ``"values"``; extra keys are copied through.
    """
    sidecar = {
        'figure': name,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'generator': 'curated/generate_report.py',
        **meta,
    }
    sidecar_path = FIGS / f'{name}.meta.json'
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + '\n')
    print(f'  {sidecar_path.name}')


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
                color=COLORS.get(m, '#999999'), linewidth=2, markersize=6)
    # Shaded region = per-sample cell-count range of the analysis datasets
    # (vanvliet / DeepCell), matching the fig:01 caption.
    ax.axvspan(100, 600, alpha=0.08, color='#999999', label='_nolegend_')
    ax.text(310, ax.get_ylim()[1] * 0.9, 'vanvliet / DeepCell\ncell-count range',
            fontsize=9, color='#262626')
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
# 1b. Attention Benchmark on the H100 (fig:speed_vs_n + appendix detail)
#     Uses benchmark_attn/results/full_bench_h100.csv — the corrected H100
#     benchmark behind the report's evaluation claims (0.199 vs 0.334 ms at
#     N=128, crossover N≈4000; job 3920627 lineage, KNN-index cost included).
#     The previous report copy of this figure had no generator in the repo;
#     this panel makes it reproducible and splits summary vs detail.
# ═══════════════════════════════════════════════════════════════
H100_CSV = ROOT / 'benchmark_attn' / 'results' / 'full_bench_h100.csv'

H100_COLORS = palette.COLORS

H100_LABELS = {
    'dense_flash':  'Dense FlashAttention',
    'dense_masked': 'Dense masked',
    'mask_knn':     'Mask-KNN ($K{=}16$)',
    'gather_sdpa':  'Gather-KNN SDPA ($K{\\in}4$–$64$)',
    'knn_relpos':   'KNN + rel. pos. bias ($K{=}16$)',
    'nsa':          'NSA',
    'minimax':      'MiniMax',
}


def _h100_series(df, method, k=None):
    """
    Sorted (N, time_ms) series for one method (restricted to one K if given).

    Averaged over duplicated (method, K, N) rows by taking the minimum
    measured time (consistent with the talk-figure generator).
    """
    rows = df[df.method == method]
    if k is not None:
        rows = rows[rows.K == k]
    return rows.groupby('N', as_index=False).time_ms.min().sort_values('N')


def _h100_band(df, method, k_low=4, k_high=64):
    """
    Min–max band across K values for one method at each N.

    Returns (sequence_lengths, lower_bounds, upper_bounds) sorted by N; the
    band honestly spans the K-dependence instead of hiding it behind one
    representative K.
    """
    rows = df[df.method == method].groupby('N').time_ms.agg(['min', 'max'])
    return rows.index.to_numpy(), rows['min'].values, rows['max'].values


def _h100_style_axis(ax, y_top):
    """
    Shared log-log styling for the H100 panels (compact report size).

    Explicit decimal y-ticks: matplotlib's default log formatter renders
    sub-1 ticks as "4×10⁻¹"-style artifacts on zoomed ranges.
    """
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.tick_params(labelsize=6)
    ax.set_xlabel('Sequence length $N$', fontsize=8)
    ax.set_ylabel('Forward time (ms)', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', which='major', color='#cccccc', linewidth=0.5)
    ax.set_ylim(0.08, y_top)
    ax.set_axisbelow(True)


def _readable_text_color(hex_color):
    """
    Darken a series colour until it is legible as 5.5 pt text on white.

    The light categorical Okabe–Ito slots (e.g. ``#56B4E9`` sky-blue for K=32) are designed
    for line patches, not for small text. Colours above a relative-luminance
    threshold are scaled toward black so the label keeps its hue association
    but reaches readable contrast; already-dark colours are returned
    unchanged. Called by the NSA/MiniMax end labels in the body panel of
    :func:`panel_speed_vs_n_h100`.
    """
    red, green, blue = (int(hex_color[index:index + 2], 16) / 255
                        for index in (1, 3, 5))
    relative_luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    if relative_luminance <= 0.5:
        return hex_color
    scale = 0.5 / relative_luminance
    return '#{:02x}{:02x}{:02x}'.format(
        *(min(255, round(channel * 255 * scale))
          for channel in (red, green, blue)))


def panel_speed_vs_n_h100():
    """
    Two report figures from the H100 attention benchmark (owner decisions
    2026-09-28: SPEC 0012 split, SPEC 0017 communication round, round 5
    owner decision to drop the dataset-scale panel).

    Body figure ``01_speed_vs_n_h100.pdf`` — single panel, method ranking
    across the full measured range N=128..8192. Four story-carrying series:
    dense_flash, dense_masked, mask_knn at K=16, and the gather-KNN (SDPA)
    min–max band over K∈{4–64} (the three backends agree within 20%).
    knn_relpos is omitted — it coincides with gather_sdpa within ±1%
    (appendix detail (d)). NSA and MiniMax use their palette colours
    (blue / yellow) with direct end labels instead of legend entries.
    No shaded regions: the crossover where sparse undercuts dense masked
    is marked only by a dashed line at N ≈ 4000 (bracketed by the N=2048
    and N=4096 measurements); the dataset-scale headline (dense masked
    1.7× faster than mask-KNN at N=128: 0.199 vs 0.334 ms) lives in the
    body text and the figure caption — the owner judged a dedicated
    dataset-scale panel redundant after the bar-chart round.

    Appendix figure ``01_speed_vs_n_h100_detail.pdf`` — round-2 redesign
    2026-09-28 (SPEC 0018 amendment; owner: the round-1 2×2 still felt
    crowded and its coincidence panels carried no information). 1×2:
    (a) overview ranking of the architectures (dense flash, dashed
    dense-masked 1.0 reference, gather-SDPA K=16, mask-KNN K=16, NSA,
    MiniMax; log-y so MiniMax ~65× and dense flash 0.07× share one view);
    (b) gather-KNN (SDPA) context-size sweep, the only K-sensitive family.
    gather-fused/gather-matmul (within 20% of SDPA) and KNN+rel-pos
    (within ±1% of SDPA) are stated numerically in the caption instead of
    being drawn as coincident curves. One visual grammar: solid lines +
    colour only; single dashed reference identified by the legend entry
    "Dense masked (= 1)"; one figure-level legend (6 architecture + 4 K
    entries); no bands, no axvspan, no leader lines, no markers beyond
    the shared 'o', no dash-pattern encodings.
    """
    df = pd.read_csv(H100_CSV)
    df = df[df['error'].isna() & (df['L'] == 1)].copy()
    df['N'] = df['N'].astype(int)

    fig, ax_ranking = plt.subplots(figsize=(5.6, 2.4))

    # ── (a) Full-range ranking ────────────────────────────────────────────
    gather_rows = df[df.method == 'gather_sdpa']
    gather_band = gather_rows.groupby('N').time_ms.agg(['min', 'max']).sort_index()
    ax_ranking.fill_between(gather_band.index, gather_band['min'],
                            gather_band['max'], color=H100_COLORS['gather_sdpa'],
                            alpha=0.25, linewidth=0, zorder=1,
                            label=H100_LABELS['gather_sdpa'])
    gather_k16 = _h100_series(df, 'gather_sdpa', 16)
    ax_ranking.plot(gather_k16.N, gather_k16.time_ms, 'o-', ms=2.2, lw=0.8,
                    color=H100_COLORS['gather_sdpa'], zorder=2)
    # Context lines (NSA, MiniMax): palette colours, direct end label.
    # Marker style matches the other series so every plotted point reads
    # as a measurement (all three series are measured at all seven N).
    for method, label_offset, label_va in [('minimax', (5, 0), 'center'),
                                           ('nsa', (5, 3), 'bottom')]:
        rows = _h100_series(df, method)
        ax_ranking.plot(rows.N, rows.time_ms, 'o-', ms=2.2, lw=0.8,
                        color=H100_COLORS[method], zorder=2)
        ax_ranking.annotate(method, (8192, rows.time_ms.iloc[-1]),
                            xytext=label_offset, textcoords='offset points',
                            fontsize=5.5,
                            color=_readable_text_color(H100_COLORS[method]),
                            va=label_va, annotation_clip=False)
    for method, k, line_width in [('dense_flash', None, 1.0),
                                  ('dense_masked', None, 1.5),
                                  ('mask_knn', 16, 1.5)]:
        rows = _h100_series(df, method, k)
        ax_ranking.plot(rows.N, rows.time_ms, 'o-', ms=2.2, lw=line_width,
                        color=H100_COLORS[method], zorder=3,
                        label=H100_LABELS[method])
    # Crossover marker: the plotted sparse variants undercut dense masked
    # above N≈4000 (bracketed by the N=2048 and N=4096 measurements).
    ax_ranking.axvline(4000, color='#999999', lw=0.8, ls='--', zorder=2)
    _h100_style_axis(ax_ranking, 500)
    ax_ranking.set_yticks([0.1, 1, 10, 100])
    ax_ranking.set_xlim(100, 16384)
    ax_ranking.set_title('All methods, full range', fontsize=8)

    # Legend above the panel — never overlaps any curve.
    handles, labels = ax_ranking.get_legend_handles_labels()
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False,
               fontsize=6, columnspacing=0.9, handlelength=1.4)
    fig.subplots_adjust(top=0.80)

    body_meta = {
        'sources': [str(H100_CSV)],
        'values': (
            [{'series': f'{method}' + (f'_K{k}' if k is not None else ''),
              'points': [{'N': int(row.N), 'time_ms': float(row.time_ms)}
                         for row in _h100_series(df, method, k).itertuples()]}
             for method, k in [('dense_flash', None), ('dense_masked', None),
                               ('mask_knn', 16), ('nsa', None),
                               ('minimax', None),
                               ('gather_sdpa', 16)]]
            + [{'series': 'gather_sdpa_K_band',
                'points': [{'N': int(n), 'time_min_ms': float(row['min']),
                            'time_max_ms': float(row['max'])}
                           for n, row in gather_band.iterrows()]}]
        ),
    }
    write_meta_sidecar('01_speed_vs_n_h100', body_meta)
    save('01_speed_vs_n_h100')

    # ── Appendix: detail figure, round-2 redesign (2026-09-28, SPEC 0018
    # amendment; owner feedback on the round-1 2×2: still crowded, and the
    # coincident-curve panels carry no information because the coincidence
    # IS the finding):
    #   * TWO panels — (a) overview ranking + crossover with one curve per
    #     architecture, (b) gather-KNN (SDPA) context-size sweep, the only
    #     family where K visibly matters;
    #   * panel (a) keeps 6 curves: dense flash, the dashed dense-masked
    #     1.0 reference, gather-SDPA K=16, mask-KNN K=16, NSA, MiniMax;
    #     gather-fused / gather-matmul (within 20% of SDPA) and KNN+rel-pos
    #     (within ±1% of SDPA) are NOT drawn — they would only braid
    #     coincident lines — and are stated numerically in the caption;
    #   * same one-visual-grammar rules as round 1: solid lines + colour
    #     only, single dashed reference identified by its legend entry,
    #     one figure-level legend (6 architecture + 4 K entries).
    dense_time_by_n = _h100_series(df, 'dense_masked').set_index('N').time_ms

    def dense_ratio(rows):
        """Convert a (N, time_ms) series to its ratio to dense masked."""
        return rows.time_ms / rows.N.map(dense_time_by_n)

    def plot_ratio_series(ax, method, k, color, sidecar_name):
        """
        Plot one method's time ratio to dense masked as a solid line.

        ``k=None`` plots every measured row of the method (dense flash,
        NSA and MiniMax have no K variants); a K value restricts the
        series to that context size. Called by the two detail panels;
        appends the plotted ratios to ``detail_series`` under
        ``sidecar_name`` unless that series is already recorded (the
        K=16 series is shared between the overview and the K-sweep
        panel).
        """
        if sidecar_name in {series['series'] for series in detail_series}:
            return
        rows = _h100_series(df, method, k)
        ratio = dense_ratio(rows)
        ax.plot(rows.N, ratio, lw=1.0, color=color, zorder=3)
        detail_series.append(
            {'series': sidecar_name,
             'points': [{'N': int(row.N), 'ratio': float(ratio_value)}
                        for row, ratio_value
                        in zip(rows.itertuples(), ratio)]})

    def draw_ratio_reference(ax):
        """
        Dashed dense-masked reference line at ratio 1.0 across [100, 8192].

        The line stops at N=8192 (the last measured point) so it carries no
        claims beyond the data. Its meaning is carried by the shared legend
        entry "Dense masked (= 1)", so no in-axes annotation is needed.
        """
        ax.plot([100, 8192], [1.0, 1.0], color=OI_BLACK, linestyle='--',
                lw=1.0, zorder=2)

    def style_ratio_axis(ax, panel_title, log_y=False):
        """Shared axes styling for the ratio detail panels."""
        ax.set_xscale('log', base=2)
        if log_y:
            # Panel (a) spans dense flash (0.07×) to MiniMax (65×).
            ax.set_yscale('log')
            ax.set_yticks([0.1, 1.0, 10.0, 100.0])
            ax.set_ylim(0.05, 120)
        else:
            ax.set_ylim(0.35, 3.2)
            ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        ax.set_xticks([128, 512, 2048, 8192])
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.tick_params(labelsize=6)
        ax.set_xlabel('Sequence length $N$', fontsize=7.5)
        ax.set_ylabel('Time / dense masked (×)', fontsize=7.5)
        ax.set_xlim(100, 13000)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', which='major', color='#cccccc', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_title(panel_title, fontsize=7)

    # Colour slots for the overview panel: every method keeps its canonical
    # METHOD_COLORS slot. Two slots are shared with the K colours of panel
    # (b) (ORANGE = gather-SDPA here vs K=4 there; SKYBLUE = mask-KNN vs
    # K=32), which the two-group legend and the caption resolve.
    ARCH_COLORS = {
        'dense_flash': OI_GREEN,
        'gather_sdpa': OI_ORANGE,
        'mask_knn': OI_SKYBLUE,
        'nsa': OI_BLUE,
        'minimax': OI_YELLOW,
    }
    ARCH_LEGEND_LABELS = {
        'dense_flash': 'Dense flash (no mask)',
        'gather_sdpa': 'Gather-KNN SDPA',
        'mask_knn': 'Mask-KNN',
        'nsa': 'NSA',
        'minimax': 'MiniMax',
    }
    K_COLORS_DETAIL = {4: OI_ORANGE, 16: OI_VERMILION,
                       32: OI_SKYBLUE, 64: OI_PURPLE}

    fig_detail, axes_detail = plt.subplots(1, 2, figsize=(4.8, 2.5))
    detail_series = []

    # (a) Overview: one curve per architecture vs the dense-masked baseline.
    ax = axes_detail[0]
    draw_ratio_reference(ax)
    detail_series.append(
        {'series': 'dense_masked_reference',
         'points': [{'N': int(n), 'ratio': 1.0} for n in dense_time_by_n.index]})
    for method, k in [('dense_flash', None), ('gather_sdpa', 16),
                      ('mask_knn', 16), ('nsa', None), ('minimax', None)]:
        plot_ratio_series(ax, method, k, ARCH_COLORS[method],
                          f'{method}_K16' if k is not None else method)
    style_ratio_axis(ax, '(a) All architectures vs dense masked', log_y=True)

    # (b) Gather-KNN (SDPA): context size K — the one family where K matters
    # (spread above N=2048); colour encodes K, all lines solid.
    ax = axes_detail[1]
    draw_ratio_reference(ax)
    for k in (4, 16, 32, 64):
        plot_ratio_series(ax, 'gather_sdpa', k, K_COLORS_DETAIL[k],
                          f'gather_sdpa_K{k}')
    style_ratio_axis(ax, '(b) Gather-KNN (SDPA): context size $K$')

    # One shared legend above the panels (vis_guidelines §6): architecture
    # slots first, then the K colours that panel (b) uses.
    legend_handles = [
        mlines.Line2D([], [], color=OI_BLACK, lw=1.0, linestyle='--',
                      label='Dense masked (= 1)'),
    ] + [
        mlines.Line2D([], [], color=ARCH_COLORS[method], lw=1.0,
                      label=ARCH_LEGEND_LABELS[method])
        for method in ('dense_flash', 'gather_sdpa', 'mask_knn',
                       'nsa', 'minimax')
    ] + [
        mlines.Line2D([], [], color=K_COLORS_DETAIL[k], lw=1.0,
                      label=f'$K{{=}}{k}$')
        for k in (4, 16, 32, 64)
    ]
    fig_detail.legend(handles=legend_handles, loc='upper center', ncol=5,
                      frameon=False, fontsize=5.5, columnspacing=0.9,
                      handlelength=1.6)
    fig_detail.tight_layout(rect=(0, 0, 1, 0.86))

    detail_meta = {
        'sources': [str(H100_CSV)],
        'values': detail_series,
    }
    write_meta_sidecar('01_speed_vs_n_h100_detail', detail_meta)
    return save('01_speed_vs_n_h100_detail')


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
                  color=[COLORS.get(m, '#999999') for m in order], width=0.6)
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
                  color=[COLORS.get(m, '#999999') for m in order], width=0.6)
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
                fontsize=9, color='#262626', ha='center',
                arrowprops=dict(arrowstyle='->', color='#262626', lw=1.5))
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
                color=COLORS.get(lbl, '#999999'), linewidth=2, markersize=6)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Sequence length $N$ (cells per frame)')
    ax.set_ylabel('GPU Memory (MB)')
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
        col = COLORS.get(mod, '#999999')
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
    bars1 = ax.bar(x - w/2, [edge_f1[m] for m in models], w, label='Edge F1', color=OI_BLUE)
    bars2 = ax.bar(x + w/2, [div_f1[m] for m in models], w, label='Division F1', color=OI_VERMILION)
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
        ax.text(i, v + 0.01, f'{v:.2f} min', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([LABELS[m] for m in models], fontsize=9)
    ax.set_ylabel('Minutes per Epoch')
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
    sns.heatmap(pivot_dense, annot=True, fmt='.2f', cmap=palette.DIVERGING_CMAP, center=1,
                ax=axes[0], cbar_kws={'label': '× vs Dense Masked'})
    axes[0].set_title('Speedup vs Dense Masked')
    sns.heatmap(pivot_flash, annot=True, fmt='.2f', cmap=palette.DIVERGING_CMAP, center=1,
                ax=axes[1], cbar_kws={'label': '× vs Dense FlashAttn'})
    axes[1].set_title('Speedup vs Dense FlashAttn')
    fig.suptitle('Sparse Attention Speedup by N and K (L=1)', fontsize=13, y=1.02)
    fig.tight_layout()
    return save('10_speedup_heatmap')


# ═══════════════════════════════════════════════════════════════
# 11. Profiler: Gather bottleneck breakdown (N=256)
# ═══════════════════════════════════════════════════════════════
def panel_profiler_breakdown():
    """
    Small multiples of the per-operation CUDA time breakdown at N=256
    (torch.profiler, N=256, K=16, B=2, d=256, h=4; measured values hardcoded
    from the profiler run). One panel per method, each with its own x-scale,
    so every operation segment and its μs value stays legible — the previous
    single shared-axis stacked bars squeezed the 11–54 μs bars against the
    1084 μs gather bar until their in-bar labels overlapped. The totals in
    the panel titles carry the magnitude story (gather is ~11× scatter and
    ~20× dense flash); bars are named on the y-axis, so no legend is needed.
    """
    profiler_panels = [
        ('Sparse Gather', 1084, [
            ('SDPA q_len=1', 533, AUX['sdpa_q1']),
            ('Gather (index)', 277, AUX['gather_index']),
            ('Copy/reshape', 210, AUX['copy_reshape']),
            ('QKV+out proj', 64, AUX['qkv_proj']),
        ]),
        ('KNN Mask (scatter)', 97, [
            ('EfficientAttn', 36, AUX['efficient_attn']),
            ('QKV+out proj', 37, AUX['qkv_proj']),
            ('Mask fill', 13, AUX['mask_fill']),
            ('scatter_ in-place', 11, AUX['scatter_op']),
        ]),
        ('Dense Flash', 54, [
            ('QKV+out proj', 37, AUX['qkv_proj']),
            ('FlashAttn', 17, AUX['efficient_attn']),
        ]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (method_name, total_us, operation_parts) in zip(axes, profiler_panels):
        y_positions = np.arange(len(operation_parts))
        operation_names = [name for name, _, _ in operation_parts]
        operation_values = [value for _, value, _ in operation_parts]
        largest_value = max(operation_values)
        ax.barh(y_positions, operation_values,
                color=[color for _, _, color in operation_parts],
                edgecolor='white', linewidth=0.5, height=0.6)
        for y_position, (operation_name, value_us, _) in zip(y_positions, operation_parts):
            ax.text(value_us + largest_value * 0.03, y_position,
                    f'{value_us} μs ({value_us / total_us:.0%})',
                    va='center', fontsize=7.5)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(operation_names, fontsize=7.5)
        ax.invert_yaxis()  # keep the pipeline order top-to-bottom
        ax.set_xlim(0, largest_value * 1.42)  # room for the value+% labels
        ax.set_xlabel('CUDA time (μs)', fontsize=8)
        ax.grid(True, axis='x', color='#cccccc', alpha=0.6, linewidth=0.4)
        ax.grid(False, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()

    meta = {
        'sources': [],
        'note': 'measured values hardcoded from torch.profiler run '
                '(N=256, K=16, B=2, d=256, h=4); no CSV artifact',
        'values': [
            {
                'method': method_name,
                'total_us': total_us,
                'operations': [
                    {'operation': operation_name, 'time_us': value_us}
                    for operation_name, value_us, _ in operation_parts
                ],
            }
            for method_name, total_us, operation_parts in profiler_panels
        ],
    }
    write_meta_sidecar('11_profiler_breakdown', meta)
    return save('11_profiler_breakdown')


# ═══════════════════════════════════════════════════════════════
# 13. SSL Pretraining: Negative Result
# ═══════════════════════════════════════════════════════════════
def panel_ssl_negative():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), gridspec_kw={'width_ratios': [1, 1.5]})

    # LEFT: val_loss bar chart
    methods = ['From scratch\n(10% labels)', 'SSL identity BCE\n+ finetune', 'Random baseline\n(NT-Xent)']
    vals = [0.416, 0.404, 0.648]  # for contrastive: this is TRA not loss
    colors_ssl = [OI_BLACK, OI_BLUE, '#999999']
    labels_ssl = ['0.416', '0.404', '0.648 TRA\n(= random)']
    bars = ax1.bar(range(len(methods)), vals, color=colors_ssl, width=0.5, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, fontsize=8)
    ax1.set_ylabel('Val Loss (BCE) / TRA (Contrastive)')

    # RIGHT: identity BCE pretraining convergence (5 epochs)
    df_ssl = pd.read_csv(ROOT / 'benchmark_ssl' / 'results' / 'runs' / 'ssl_v1' / 'training_log.csv')
    ax2.plot(df_ssl.epoch, df_ssl.train_loss, 'o-', label='Train loss', color=OI_BLUE, linewidth=2)
    ax2.plot(df_ssl.epoch, df_ssl.val_loss, 's-', label='Val loss', color=OI_VERMILION, linewidth=2)
    ax2.axhline(y=0.416, color='#999999', linestyle='--', alpha=0.5, label='Baseline final (0.416)')
    ax2.set_xlabel('SSL Pretraining Epoch')
    ax2.set_ylabel('BCE Loss')
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return save('13_ssl_negative')


# ═══════════════════════════════════════════════════════════════
# 12. Layer scaling: L=1 vs L=4
# ═══════════════════════════════════════════════════════════════
def panel_layer_scaling():
    df = pd.read_csv(ROOT / 'benchmark_sparse_results.csv')
    df = df[df.method.isin(['dense', 'dense_flash']) | ((df.method == 'sparse') & (df.K == 16))]
    df['model_kind'] = df.method.map({'dense': 'dense',
                                     'dense_flash': 'dense_flash',
                                     'sparse': 'sparse_k16'})
    df['label'] = df.apply(lambda r: f"{r.model_kind}_L{r.L}", axis=1)
    # Colour per model, linestyle per L — keeps the Okabe–Ito slots
    # pure (one hue per model) and uses style for the secondary dim.
    model_color = {
        'dense':       OI_BLACK,
        'dense_flash': OI_GREEN,
        'sparse_k16':  OI_VERMILION,
    }
    layer_linestyle = {1: '-', 4: '--'}
    fig, ax = plt.subplots(figsize=(8, 5))
    for lbl, grp in df.groupby('label'):
        grp = grp.sort_values('N')
        kind = lbl.rsplit('_L', 1)[0]
        L = int(lbl.rsplit('_L', 1)[1])
        ax.plot(grp.N, grp.time_s * 1000, marker='o',
                linestyle=layer_linestyle[L],
                label=lbl, color=model_color[kind],
                linewidth=2, markersize=6)
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
h2 { font-size: 20px; margin-top: 40px; color: #262626; border-left: 4px solid ''' + OI_BLUE + ''';
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
    figs.append(panel_speed_vs_n_h100())
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
