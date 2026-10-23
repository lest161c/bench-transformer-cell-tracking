"""Generate §4.1 Sparse Attention Report: SVGs + report.html.

All colours follow the Okabe–Ito scientific palette defined in
``docs/vis_guidelines.md`` (colorblind-safe, print-safe).
Heatmaps use ``palette.DIVERGING_CMAP`` (vermilion↔white↔blue); K uses
the categorical Okabe–Ito slots (``palette.K_COLORS``). No Flat UI /
web-design colours.

Run (from the repository root; needs matplotlib/seaborn/pandas)::

    python curated/generate_report.py

Regenerates every panel under ``curated/figures/`` (SVG + ``.meta.json``
sidecar naming the source data). Each panel function documents its data source
in its docstring; see ``REPRODUCIBILITY.md`` for the full map.
"""

import os, sys, math, json
import inspect
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
    'mask_knn': 'SparseMask (scatter)',
    'baseline': 'Dense Baseline',
    'k4': 'Sparse K=4',
    'k16': 'Sparse K=16',
    'k32': 'Sparse K=32',
    'k64': 'Sparse K=64',
    # full_bench_a500.csv method names (fig 01)
    'dense_masked_fb': 'Dense Masked (baseline)',
    'dense_flash_fb': 'Dense FlashAttn (no mask)',
    'gather-KNN_K=4': 'SparseGather K=4',
    'gather-KNN_K=16': 'SparseGather K=16',
    'mask-KNN_K=4': 'SparseMask K=4',
    'mask-KNN_K=16': 'SparseMask K=16',
    'NSA': 'NSA',
    'KNN-RelPos_K=4': 'KNN-RelPos K=4',
    'KNN-RelPos_K=16': 'KNN-RelPos K=16',
}


def save(name, meta=None):
    """Save the current figure as both SVG and PDF under ``curated/figures/``
    and auto-write its meta sidecar (J-06 phase 2).

    The LaTeX report includes the figures via ``\\includegraphics{*.pdf}``,
    so the PDF is the source of truth that ships with the report; the SVG
    is kept alongside for editing in Inkscape / Illustrator and for the
    HTML report.

    Provenance: every figure gets a ``<name>.meta.json`` sidecar (same
    convention as ``results/generate_plots.py``). If the panel passes no
    explicit ``meta``, the calling panel function's name and docstring are
    captured automatically — panels document their data sources in their
    docstrings, so the sidecar stays correct without per-panel boilerplate.
    An explicit ``meta`` dict must contain ``"sources"`` and ``"values"``.
    """
    svg_path = FIGS / f'{name}.svg'
    pdf_path = FIGS / f'{name}.pdf'
    # usetex requires figure.savefig: pyplot's tight-bbox path renders on the
    # interactive Agg canvas, which needs dvipng (absent on this system).
    plt.gcf().savefig(svg_path, bbox_inches='tight', dpi=150)
    plt.gcf().savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f'  {svg_path.name}  +  {pdf_path.name}')
    if meta is None:
        frame = inspect.currentframe().f_back
        panel_name = frame.f_code.co_name
        panel = frame.f_globals.get(panel_name)
        meta = {
            'sources': (f'{panel_name}() in curated/generate_report.py; '
                        'data sources documented in its docstring'),
            'values': f'plotted values defined in {panel_name}() source',
            'panel_docstring': (inspect.getdoc(panel) or '') if panel else '',
        }
    write_meta_sidecar(name, meta)
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
#     Uses benchmark_attn/results/full_bench_v2_h100_cache_on.csv — the
#     VERIFIED v2 H100 grid (SPEC 0022, jobs 4375584 + 4377261; backend
#     labels probe-verified, stray-transpose bug fixed). dist_cache=on
#     throughout. Error rows (error='skip_k_ge_n') are excluded before
#     plotting. flex_attn has K=0 only; the other methods sweep K∈{2..128}.
# ═══════════════════════════════════════════════════════════════
H100_CSV = ROOT / 'benchmark_attn' / 'results' / 'full_bench_v2_h100_cache_on.csv'
B64_CSV = ROOT / 'benchmark_attn' / 'results' / 'full_bench_b64_h100_cache_on.csv'
B64_CACHE_OFF_CSV = ROOT / 'benchmark_attn' / 'results' / 'full_bench_b64_h100_cache_off.csv'
# Alias retained for panel_b64_saturation (figure 19), which is out of scope
# for the cache-off-continuation change.
B64_CSV_OFF = B64_CACHE_OFF_CSV
DECOMP_CSV = ROOT / 'benchmark_attn' / 'results' / 'cudnn_vs_fa2_mask_decomposition.csv'
H100_CACHE_OFF_CSV = ROOT / 'benchmark_attn' / 'results' / 'full_bench_v2_h100_cache_off.csv'

H100_COLORS = palette.COLORS

H100_LABELS = {
    'dense_flash':  'DenseFlash',
    'dense_masked': 'CachedDistAttention',
    'mask_knn':     'SparseMask ($K{=}16$)',
    'gather_sdpa':  'SparseGather ($K{=}16$)',
    'flex_attn':    'FlexAttention',
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


def _b64_pooled_frame():
    """The B=64 grid with cache-regime-independent rows pooled.

    The cache regime is part of the measurement for every method whose
    ``dist_cache`` column is ``on``/``off``; those rows come from the
    cache-on run only. Rows with ``dist_cache == 'na'`` (dense_flash,
    flex_attn) are regime-independent, so the two b64 runs are pooled for
    them with ok-row preference: the cache-on run recorded allocator-state
    OOMs at N >= 16384 that the cache-off run measured (dense_flash
    79.0/320.5 ms, flex 264.7/1101.7 ms at 16384/32768), which is what
    carries those curves to the top of the x-range.

    Returns:
        DataFrame with columns as in the b64 CSV (L=1, batch=64 rows only).
    """
    on = pd.read_csv(B64_CSV)
    off = pd.read_csv(B64_CSV_OFF)
    pooled = pd.concat([on, off])
    pooled = pooled[(pooled['error'].isna()) & (pooled['dist_cache'] == 'na')]
    pooled = pooled.groupby(['method', 'K', 'N', 'L', 'batch'],
                            as_index=False).agg(time_ms=('time_ms', 'min'),
                                                memory_mb=('memory_mb', 'max'))
    frame = pd.concat([
        on[on['error'].isna() & (on['dist_cache'] != 'na')],
        pooled,
    ])
    frame = frame[frame['L'] == 1].copy()
    frame['N'] = frame['N'].astype(int)
    return frame


def _h100_backend_label(df, method, k=None):
    """Comma-joined dispatched backend labels for one method/K series.

    Read from the CSV ``backend`` column so the figure sidecar records the
    kernel family that actually ran (e.g. ``matmul_cublas`` for the B=64
    SparseGather arm) rather than a label inferred from the method name.
    """
    rows = df[df.method == method]
    if k is not None:
        rows = rows[rows.K == k]
    return ', '.join(sorted(rows['backend'].dropna().unique()))


def _h100_band(df, method, k_low=4, k_high=64):
    """
    Min–max band across K values for one method at each N.

    Returns (sequence_lengths, lower_bounds, upper_bounds) sorted by N; the
    band honestly spans the K-dependence instead of hiding it behind one
    representative K.
    """
    rows = df[df.method == method].groupby('N').time_ms.agg(['min', 'max'])
    return rows.index.to_numpy(), rows['min'].values, rows['max'].values


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


ARCH_COLORS = {
    'dense_flash': OI_BLACK,
    'gather_sdpa': OI_ORANGE,
    'mask_knn': OI_SKYBLUE,
    'flex_attn': OI_PURPLE,
}

ARCH_LEGEND_LABELS = {
    'dense_flash': 'DenseFlash',
    'gather_sdpa': 'SparseGather (SDPA)',
    'mask_knn': 'SparseMask (scatter)',
    'flex_attn': 'FlexAttention',
}

# K grid extended per SPEC 0022 H-03: K=2 bottom-line sanity, K=8 fills the
# 4–16 gap.  K-sweep figure predates the K=2/8 grid extension.
K_COLORS_DETAIL = {
    2: OI_VERMILION,
    4: OI_SKYBLUE,
    8: OI_BLUE,
    16: OI_GREEN,
    32: OI_YELLOW,
    64: OI_PURPLE,
}


def draw_ratio_reference(ax):
    """Dashed horizontal line at ratio 1 (the dense-masked baseline)."""
    ax.axhline(1.0, color=OI_BLACK, lw=1.0, linestyle='--', zorder=2)


def plot_ratio_series(ax, df, method, k, colour, collector, label):
    """
    One series of time-ratio-vs-dense-masked points on ``ax``.

    Ratio per N: method time (min over duplicate rows via
    :func:`_h100_series`) divided by the dense-masked time at the same N
    (from the same filtered v2-grid frame ``df``). Appends the computed
    points to ``collector`` for the figure's meta sidecar.
    """
    reference = _h100_series(df, 'dense_masked', None).set_index('N')
    series = _h100_series(df, method, k)
    ratios = []
    for row in series.itertuples():
        if row.N in reference.index:
            ratio = float(row.time_ms / reference.loc[row.N, 'time_ms'])
            ratios.append({'N': int(row.N), 'ratio': ratio})
    if not ratios:
        return
    ax.plot([point['N'] for point in ratios],
            [point['ratio'] for point in ratios],
            'o-', ms=2.0, lw=0.8, color=colour, zorder=3, label=label)
    collector.append({'series': label, 'points': ratios})


def style_ratio_axis(ax, title, ylim=(0, 1.5), yticks=(0.1, 0.5, 1.0)):
    """Shared styling for the detail-figure ratio panels.

    Linear ratio axis with decimal ticks (defaults 0.1 / 0.5 / 1.0): the
    ratios cluster in a narrow band around 1.0, where a log axis
    compresses the decision-relevant region and its default formatter
    renders ticks as "6x10^0"-style artifacts. Panel (b) overrides both
    limits and ticks because its K=64 sweep reaches ratios of ~2.25.
    """
    ax.set_xscale('log', base=2)
    ax.set_ylim(*ylim)
    ax.set_yticks(list(yticks))
    ax.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(labelsize=6)
    ax.set_xlabel(title, fontsize=7)
    ax.set_ylabel('Time ratio vs CachedDistAttention', fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', which='major', color='#cccccc', linewidth=0.5)
    ax.set_axisbelow(True)


def style_time_axis(ax, title):
    """Shared styling for the detail-figure absolute-time panels (log-log).

    Explicit decimal ticks (0.3/1/3/5 ms) avoid the default log
    formatter's "3x10^0"-style artifacts on this zoomed range; minor
    ticks stay unlabeled for the same reason.
    """
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_ylim(0.2, 6.5)
    ax.set_yticks([0.3, 1.0, 3.0, 5.0])
    ax.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(labelsize=6)
    ax.set_xlabel(title, fontsize=7)
    ax.set_ylabel('Forward time (ms)', fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', which='major', color='#cccccc', linewidth=0.5)
    ax.set_axisbelow(True)


def panel_speed_vs_n_h100():
    """
    Two report figures from the VERIFIED H100 v2 grid (SPEC 0022 jobs
    4375584 + 4377261; dist_cache=on; backend labels probe-verified,
    stray-transpose bug fixed), the B=64 saturation grid
    (``full_bench_b64_h100_cache_on.csv``, dist_cache=on), and the B=64
    cache-off run (``full_bench_b64_h100_cache_off.csv``). Error rows —
    including OOM and CUDA launch-configuration failures — are excluded
    before plotting, so each curve is a gap-free run of successful
    measurements and ends where a configuration fails.

    Body figure ``01_speed_vs_n_h100.pdf`` — single panel, five series
    (SPEC 0026 T1): dense_flash (K=0), dense_masked (K=0), mask_knn
    (K=16), gather_sdpa (K=16), flex_attn (K=0), each drawn at B=1
    (dashed, v2 grid, N up to 8192) and B=64 (solid, B=64 cache-on grid,
    x-axis up to 32768). The B=64 grid renames mask_knn→sparse_mask;
    SparseGather at B=64 uses ``sparse_gather_matmul`` K=16 because the
    SDPA arm errors at N>=1024. DenseFlash and FlexAttention have cache-on
    rows only to N=8192 (cache-on OOMs above that), so their B=64 curves
    continue from the cache-off 8192 point with a thinner dotted line and
    open markers to N=32768; no cache-off points are used for the masked
    or gathered series. Log-log axes (time_ms vs N).

    Okabe–Ito slot assignment (vis_guidelines §1; colourblind-safe,
    print-safe, one hue per series):
        dense_flash  → OI_BLACK    #000000  (baseline curve slot)
        dense_masked → OI_VERMILION #D55E00  (ablation/loss slot)
        mask_knn     → OI_SKYBLUE  #56B4E9  (variant-B slot)
        gather_sdpa  → OI_ORANGE   #E69F00  (variant-A slot, "ours")
        flex_attn    → OI_PURPLE   #CC79A7  (attention-overlay slot)

    Legend labels: H100_LABELS (CamelCase display names shared with
    panel_memory_vs_n), so both H100 panels cannot drift apart. A second,
    small line-style legend distinguishes B=1 (dashed) from B=64 (solid).

    Appendix figure ``01_speed_vs_n_h100_detail.pdf`` — (a) forward-time
    ratio vs CachedDistAttention for the other four architectures (linear
    decimal ticks); (b) the SparseGather (SDPA) K-sweep in ABSOLUTE forward
    time with the CachedDistAttention reference (log-log). Panel (b)
    deliberately shows absolute times, not ratios: the reference is
    step-shaped (launch plateau ~0.25 ms through N=2048, quadratic dominance
    from 4096), so ratio normalization manufactures a spike (K=64 at N=2048,
    ratio 2.25) on a perfectly smooth O(N.K) curve, while absolute times
    expose the O(N.K) scaling and the crossover with CachedDistAttention's
    quadratic term directly.
    """
    df = pd.read_csv(H100_CSV)
    df = df[df['error'].isna() & (df['L'] == 1)].copy()
    df['N'] = df['N'].astype(int)

    df_b64_on = pd.read_csv(B64_CSV)
    df_b64_on = df_b64_on[df_b64_on['error'].isna() &
                          (df_b64_on['L'] == 1)].copy()
    df_b64_on['N'] = df_b64_on['N'].astype(int)

    # The cache-off run is used only as a continuation for the two methods
    # whose rows are regime-independent (dist_cache == 'na'): the cache-on
    # run OOMs at N >= 16384, while cache-off measured dense_flash
    # 79.0/320.5 ms and flex_attn 264.7/1101.0 ms at 16384/32768. Masked
    # and gathered series stay on their cache-on rows.
    df_b64_off = pd.read_csv(B64_CACHE_OFF_CSV)
    df_b64_off = df_b64_off[df_b64_off['error'].isna() &
                            (df_b64_off['L'] == 1)].copy()
    df_b64_off['N'] = df_b64_off['N'].astype(int)

    series_specs = [
        ('dense_flash',  0, OI_BLACK),
        ('dense_masked', 0, OI_VERMILION),
        ('mask_knn',    16, OI_SKYBLUE),
        ('gather_sdpa', 16, OI_ORANGE),
        ('flex_attn',    0, OI_PURPLE),
    ]
    # The B=64 campaign uses the renamed method keys for the KNN variants.
    # SparseGather switches to the matmul backend: the SDPA arm errors for
    # N >= 1024 (B*N exceeds the CUDA grid-dimension limit).
    b64_methods = {
        'dense_flash':  'dense_flash',
        'dense_masked': 'dense_masked',
        'mask_knn':     'sparse_mask',
        'gather_sdpa':  'sparse_gather_matmul',
        'flex_attn':    'flex_attn',
    }
    cache_off_continuation = {'dense_flash', 'flex_attn'}

    fig, ax_ranking = plt.subplots(figsize=(5.6, 2.4))
    method_handles = []
    meta_values = []
    for method, k, colour in series_specs:
        series_name = method if k == 0 else f'{method}_K{k}'

        # B=1 reference (v2 grid, cache on).
        group_b1 = _h100_series(df, method, k)
        ax_ranking.plot(group_b1.N, group_b1.time_ms, 'o--',
                        ms=2.2, lw=0.8, color=colour, zorder=3)
        meta_values.append({
            'series': series_name,
            'batch': 1,
            'source': 'cache_on',
            'method': method,
            'backend': _h100_backend_label(df, method, k),
            'points': [{'N': int(row.N), 'time_ms': float(row.time_ms)}
                       for row in group_b1.itertuples()],
        })

        # B=64 solid curve from the cache-on run.
        b64_method = b64_methods[method]
        group_b64 = _h100_series(df_b64_on, b64_method, k)
        ax_ranking.plot(group_b64.N, group_b64.time_ms, 'o-',
                        ms=2.2, lw=0.8, color=colour, zorder=3)
        meta_values.append({
            'series': series_name,
            'batch': 64,
            'source': 'cache_on',
            'method': b64_method,
            'backend': _h100_backend_label(df_b64_on, b64_method, k),
            'points': [{'N': int(row.N), 'time_ms': float(row.time_ms)}
                       for row in group_b64.itertuples()],
        })

        # DenseFlash/FlexAttention continue past the cache-on OOM with the
        # cache-off measurement, starting at the shared N=8192 point so the
        # dotted line visibly joins the solid curve.
        if method in cache_off_continuation:
            continuation = _h100_series(df_b64_off, b64_method, k)
            continuation = continuation[continuation.N >= 8192]
            ax_ranking.plot(continuation.N, continuation.time_ms,
                            linestyle=':', marker='o', ms=2.2, lw=0.65,
                            markerfacecolor='none', markeredgewidth=0.6,
                            color=colour, zorder=3)
            meta_values.append({
                'series': series_name,
                'batch': 64,
                'source': 'cache_off',
                'method': b64_method,
                'backend': _h100_backend_label(df_b64_off, b64_method, k),
                'points': [{'N': int(row.N), 'time_ms': float(row.time_ms)}
                           for row in continuation.itertuples()],
            })

        method_handles.append(mlines.Line2D(
            [], [], color=colour, lw=0.8, marker='o', ms=2.2,
            label=H100_LABELS[method]))
    ax_ranking.set_xscale('log', base=2)
    ax_ranking.set_yscale('log')
    ax_ranking.set_yticks([0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    ax_ranking.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
    ax_ranking.get_yaxis().set_minor_formatter(mticker.NullFormatter())
    ax_ranking.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    ax_ranking.set_xlim(128, 32768)
    ax_ranking.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax_ranking.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax_ranking.set_xlabel('Sequence length $N$')
    ax_ranking.set_ylabel('Forward time (ms)')
    ax_ranking.set_title('Attention forward time vs sequence length')
    method_legend = ax_ranking.legend(handles=method_handles, fontsize=6,
                                      frameon=False, loc='upper left')
    ax_ranking.add_artist(method_legend)
    style_handles = [
        mlines.Line2D([], [], color=OI_BLACK, lw=0.8, linestyle='--',
                      label='B=1'),
        mlines.Line2D([], [], color=OI_BLACK, lw=0.8, linestyle='-',
                      label='B=64'),
    ]
    ax_ranking.legend(handles=style_handles, fontsize=6, frameon=False,
                      loc='lower right', title='Batch', title_fontsize=6)
    fig.tight_layout()

    body_meta = {
        'sources': [str(H100_CSV), str(B64_CSV), str(B64_CACHE_OFF_CSV)],
        'values': meta_values,
    }
    save('01_speed_vs_n_h100', body_meta)

    # ── Appendix: detail figure (2-panel) ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.5))
    detail_series = []

    # (a) Overview: every architecture as time ratio vs CachedDistAttention;
    # CachedDistAttention itself is the dashed line at ratio 1, so it gets no
    # separate curve.
    ax = axes[0]
    draw_ratio_reference(ax)
    for method, k, colour in series_specs:
        if method == 'dense_masked':
            continue
        plot_ratio_series(ax, df, method, k, colour, detail_series,
                          f'{method}_K{k}' if k is not None else method)
    style_ratio_axis(ax, '(a) All architectures vs CachedDistAttention')

    # (b) SparseGather (SDPA) context-size sweep in absolute forward time
    # with the CachedDistAttention reference curve.
    ax = axes[1]
    reference = _h100_series(df, 'dense_masked').sort_values('N')
    ax.plot(reference.N, reference.time_ms, '--', color=OI_BLACK, lw=1.0,
            zorder=3)
    detail_series.append({
        'series': 'dense_masked_reference',
        'points': [{'N': int(row.N), 'time_ms': float(row.time_ms)}
                   for row in reference.itertuples()]})
    for k in (2, 4, 8, 16, 32, 64):
        series = _h100_series(df, 'gather_sdpa', k).sort_values('N')
        ax.plot(series.N, series.time_ms, 'o-', ms=2.0, lw=0.8,
                color=K_COLORS_DETAIL[k], zorder=3)
        detail_series.append({
            'series': f'gather_sdpa_K{k}',
            'points': [{'N': int(row.N), 'time_ms': float(row.time_ms)}
                       for row in series.itertuples()]})
    style_time_axis(ax, '(b) SparseGather (SDPA): context size $K$')

    # One shared legend above the panels (vis_guidelines §6): the
    # CachedDistAttention reference (dashed in both panels), the architecture
    # colours of panel (a), then the K colours of panel (b).
    legend_handles = [
        mlines.Line2D([], [], color=OI_BLACK, lw=1.0, linestyle='--',
                      label='CachedDistAttention'),
    ] + [
        mlines.Line2D([], [], color=ARCH_COLORS[method], lw=1.0,
                      label=ARCH_LEGEND_LABELS[method])
        for method in ('dense_flash', 'gather_sdpa', 'mask_knn', 'flex_attn')
    ] + [
        mlines.Line2D([], [], color=K_COLORS_DETAIL[k], lw=1.0,
                      label=f'$K{{=}}{k}$')
        for k in (2, 4, 8, 16, 32, 64)
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=5,
               frameon=False, fontsize=5.5, columnspacing=0.9,
               handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

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
def _project_memory(ns, mems, model, n_to=32768, n_points=10):
    """Fit memory(N) on measured points and project it to ``n_to``.

    Args:
        ns: Measured sequence lengths (sorted).
        mems: Measured incremental peak memory (MB), same order.
        model: ``'quad'`` (masked methods: a quadratic dominant term) or
            ``'lin'`` (gather/flash/flex: linear scaling).
        n_to: Upper projection bound.
        n_points: Number of projected points.

    Returns:
        (projected_ns, projected_mbs, coeff_description) where the
        projection covers (last measured N, n_to]; empty arrays when fewer
        than two measured points exist.
    """
    import numpy as np
    if len(ns) < 2 or ns[-1] >= n_to:
        return [], [], ''
    degree = 2 if model == 'quad' else 1
    coefficients = np.polyfit(ns, mems, degree)
    projected = np.linspace(ns[-1], n_to, n_points)[1:]
    values = np.polyval(coefficients, projected)
    terms = ' + '.join(
        f'{c:.3e}·N^{degree - i}' for i, c in enumerate(coefficients))
    return list(projected), list(np.maximum(values, 0.0)), terms


def panel_memory_vs_n():
    """
    Report figure ``04_memory_vs_n.pdf`` — incremental peak GPU memory vs
    sequence length at B=64, from the b64 saturation grid (cache-regime-
    independent rows pooled per :func:`_b64_pooled_frame`).

    OOM is a soft, plotted outcome, never a curve end: every curve is
    drawn as far as it was MEASURED (solid); from there it continues as a
    DOTTED projection of its fitted scaling (quadratic for the masked
    methods, linear for the gather/flash/flex family) through the full
    N in {128..32768} range. The measured OOM shapes are marked with a
    cross at the projected memory of that shape, so the marker sits ON the
    dotted curve and shows measured failure and projection agreeing (e.g.
    CachedDistAttention OOMs at B=64/N=8192 where the projection says
    145 GB, far above the grey 80 GB H100 VRAM line). SparseGather's OOM
    marker sits BELOW the line (its projected attention working set at
    N=16384 is only ~44 GB): that failure is caused by the O(B N^2) KNN
    distance matrix built for the indices, not by the attention tensors.

    Log-log axes; Okabe-Ito colours per method, shared with the speed
    panel. Per-series rendering is guarded: a malformed or missing series
    is skipped with a printed note instead of aborting the figure.
    """
    df64 = _b64_pooled_frame()
    # OOM shapes per method from BOTH b64 runs (soft outcomes, not curve
    # ends): first N whose status is oom, per (method, K).
    oom_frames = []
    for path in (B64_CSV, B64_CSV_OFF):
        frame = pd.read_csv(path)
        oom_frames.append(frame[frame['error'].astype(str).str.contains('oom', na=False)])
    oom_rows = pd.concat(oom_frames)

    series_specs = [
        # (label-key, K, colour, b64 method key, projection model)
        ('dense_masked', 0,  OI_VERMILION, 'dense_masked',       'quad'),
        ('mask_knn',     16, OI_SKYBLUE,   'sparse_mask',        'quad'),
        ('gather_sdpa',  16, OI_ORANGE,    'sparse_gather_sdpa', 'lin'),
        ('dense_flash',  0,  OI_BLACK,     'dense_flash',        'lin'),
        ('flex_attn',    0,  OI_PURPLE,    'flex_attn',          'lin'),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    meta_values = []
    for method, k, colour, b64_key, model in series_specs:
        try:
            rows = df64[(df64.method == b64_key)]
            if k:
                rows = rows[rows.K == k]
            rows = rows.dropna(subset=['memory_mb'])
            rows = rows[rows.memory_mb > 0].groupby(
                'N', as_index=False).memory_mb.max().sort_values('N')
            if rows.empty:
                print(f'  [memory] no measured rows for {b64_key} '
                      f'(K={k}) — skipped')
                continue
            ns, mems = rows.N.to_numpy(), rows.memory_mb.to_numpy()
            ax.plot(ns, mems, '-', marker='o', ms=2.2, lw=0.8,
                    color=colour, zorder=3)
            series_name = method if k == 0 else f'{method}_K{k}'
            meta_values.append({
                'series': series_name, 'batch': 64, 'kind': 'measured',
                'points': [{'N': int(n), 'memory_mb': float(m)}
                           for n, m in zip(ns, mems)]})
            proj_ns, proj_ms, terms = _project_memory(
                list(ns), list(mems), model)
            if proj_ns:
                ax.plot(proj_ns, proj_ms, ':', marker='o', ms=2.0,
                        lw=0.8, color=colour, zorder=2, alpha=0.75)
                meta_values.append({
                    'series': series_name, 'batch': 64, 'kind': 'projected',
                    'fit': terms,
                    'points': [{'N': float(n), 'memory_mb': float(m)}
                               for n, m in zip(proj_ns, proj_ms)]})
            # OOM markers at the projected value of the OOM shape: the
            # cross sits ON the dotted curve, showing that the measured
            # failure and the theoretical scaling agree.
            oom = oom_rows[(oom_rows['method'] == b64_key)]
            if k:
                oom = oom[oom['K'] == k]
            first_oom = oom['N'].min() if not oom.empty else None
            if first_oom is not None and proj_ns:
                import numpy as np
                degree = 2 if model == 'quad' else 1
                coefficients = np.polyfit(ns, mems, degree)
                y_at_oom = float(np.polyval(coefficients, first_oom))
                ax.plot([first_oom], [y_at_oom], 'x', ms=5, mew=1.4,
                        color=colour, zorder=4)
                meta_values.append({
                    'series': series_name, 'batch': 64, 'kind': 'oom_marker',
                    'points': [{'N': int(first_oom), 'memory_mb': y_at_oom}]})
        except Exception as exc:  # noqa: BLE001 — one bad series must not
            # abort the figure; OOM and malformed rows are soft outcomes.
            print(f'  [memory] series {b64_key} (K={k}) skipped: {exc}')
    # 80 GB H100 VRAM limit
    ax.axhline(80000.0, color='#999999', lw=1.0, linestyle='--', zorder=1)
    ax.text(140, 95000, '80 GB H100 VRAM', fontsize=5.5, color='#666666')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel('Sequence length $N$')
    ax.set_ylabel('GPU Memory (MB)')
    ax.set_title('GPU memory vs sequence length (B=64)')
    method_handles = [
        mlines.Line2D([], [], color=colour, lw=0.8, marker='o', ms=2.2,
                      label=H100_LABELS[method])
        for method, _, colour, _, _ in series_specs]
    style_handles = [
        mlines.Line2D([], [], color=OI_BLACK, lw=0.8, linestyle='-',
                      label='measured'),
        mlines.Line2D([], [], color=OI_BLACK, lw=0.8, linestyle=':',
                      label='projected scaling'),
        mlines.Line2D([], [], color=OI_BLACK, lw=0.8, marker='x',
                      linestyle='none', ms=5, mew=1.4, label='measured OOM'),
    ]
    ax.legend(handles=method_handles, fontsize=5.5, frameon=False,
              loc='upper left')
    ax.add_artist(ax.legend(handles=style_handles, fontsize=5.5,
                            frameon=False, loc='lower right'))
    fig.tight_layout()

    body_meta = {
        'sources': [str(B64_CSV), str(B64_CSV_OFF)],
        'values': meta_values,
    }
    return save('04_memory_vs_n', meta=body_meta)


# ═══════════════════════════════════════════════════════════════
# 4b. Distance-cache effect (SPEC 0026 T3; standalone per Q1c)
# ═══════════════════════════════════════════════════════════════
def panel_dist_cache_effect():
    """
    Report figure ``16_dist_cache_effect.pdf`` (SPEC 0026 T3; reworked after
    owner review 2026-10-11) — MEASURED off/on wall-time ratio (forward time
    without cache divided by with cache; >1 = the cache helps) per method at
    N in {256, 512, 1024, 2048} from the VERIFIED H100 v2 grid (jobs
    4375584 + 4377261). Two series, matching the section claim:
    dense_masked (K=0) = CachedDistAttention and mask_knn (K=16) =
    SparseMaskAttention, colored with the canonical palette slots used by
    the sibling speed/memory panels (vermilion/sky-blue). The three gather
    backends are not part of this claim and remain in the appendix
    speed-detail figure (fig:app_speed_detail). Direct combined measurement
    replacing the retired additivity argument of the old section 3.5.
    """
    on = pd.read_csv(H100_CSV)
    off = pd.read_csv(H100_CACHE_OFF_CSV)
    for frame in (on, off):
        frame = frame[frame['error'].isna()]
        frame['N'] = frame['N'].astype(int)

    seq_lengths = [256, 512, 1024, 2048]
    series_specs = [
        ('dense_masked', 0, OI_VERMILION, 'CachedDistAttention (dense masked)'),
        ('mask_knn',    16, OI_SKYBLUE,   'SparseMask'),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    panel_width = 0.8 / len(series_specs)
    meta_values = []
    for index, (method, k, colour, label) in enumerate(series_specs):
        ratios = []
        for position, seq_len in enumerate(seq_lengths):
            on_row = on[(on['method'] == method) & (on['K'] == k) & (on['N'] == seq_len)]
            off_row = off[(off['method'] == method) & (off['K'] == k) & (off['N'] == seq_len)]
            ratio = float(off_row.iloc[0]['time_ms'] / on_row.iloc[0]['time_ms'])
            ax.bar(position + index * panel_width, ratio, width=panel_width,
                   color=colour, zorder=3,
                   label=label if position == 0 else None)
            ratios.append(ratio)
        meta_values.append({'series': f'{method}_K{k}', 'ratios': ratios})
    ax.axhline(1.0, color='k', lw=0.8, linestyle='--', zorder=2)
    ax.set_xticks([i + 0.4 - panel_width / 2 for i in range(len(seq_lengths))])
    ax.set_xticklabels([str(n) for n in seq_lengths])
    ax.set_xlabel('Sequence length $N$')
    ax.set_ylabel('Forward time ratio (off / on)')
    ax.set_ylim(0, 3.1)
    ax.legend(fontsize=6, frameon=False, loc='upper left')
    fig.tight_layout()
    return save('16_dist_cache_effect',
                meta={'sources': [str(H100_CSV), str(H100_CACHE_OFF_CSV)],
                      'values': meta_values})


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
# B=64 batch-saturation grid + mask-cost decomposition (2026-10-23)
# ═══════════════════════════════════════════════════════════════

def panel_b64_saturation():
    """Figure ``19_b64_saturation``: B=1 vs B=64 forward time and ratio.

    Reads the B=64 grid (``full_bench_b64_h100_cache_on.csv``, header with
    ``batch`` column, new method keys) and the B=1 v2 reference grid
    (``full_bench_v2_h100_cache_on.csv``, old method keys), dist_cache=on,
    L=1, error rows excluded.

    Panel (a): forward time vs N (log-log) for DenseFlash, DenseMasked
    (CachedDistAttention) and SparseMask K=16 at B=1 (dashed) and B=64
    (solid).  SparseMask/``mask_knn`` uses K=16 in both grids.

    Panel (b): dense_masked/dense_flash ratio vs N at B=1 and B=64 — the
    masked-dense penalty grows from ~1.75x at B=1 (launch-bound regime) to
    6.4x at N=512 and >11x at N>=2048 under saturation, and dense_masked
    OOMs at B=64/N=8192 while SparseMask K=16 still runs at N=4096.

    Returns the saved figure name for the html report list.
    """
    b64_on = pd.read_csv(B64_CSV)
    b64_off = pd.read_csv(B64_CSV_OFF)
    # Rows whose dist_cache column is 'na' (dense_flash, flex_attn) do not
    # depend on the cache regime; for those the two runs are pooled and a
    # row measured 'ok' in either run is used (the cache-on run recorded
    # allocator-state OOMs at N >= 16384 that the cache-off run measured).
    # All other methods keep their own regime's rows only.
    pooled_na = pd.concat([b64_on, b64_off])
    pooled_na = pooled_na[(pooled_na['error'].isna()) & (pooled_na['dist_cache'] == 'na')]
    pooled_na = pooled_na.groupby(['method', 'K', 'N', 'L', 'batch'],
                                  as_index=False).time_ms.min()
    b64 = pd.concat([
        b64_on[b64_on['error'].isna() & (b64_on['dist_cache'] != 'na')],
        pooled_na,
    ])
    b64 = b64[(b64['L'] == 1) & (b64['batch'] == 64)].copy()
    b64['N'] = b64['N'].astype(int)
    v2 = pd.read_csv(H100_CSV)
    v2 = v2[v2['error'].isna() & (v2['L'] == 1)].copy()
    v2['N'] = v2['N'].astype(int)

    # method name per grid: b64 uses the renamed keys, v2 the historical ones
    series = [
        # (label, colour, b64 method/key, v2 method/key, K)
        ('DenseFlash', OI_BLACK, 'dense_flash', 'dense_flash', None),
        ('DenseMasked (CachedDist)', OI_VERMILION, 'dense_masked', 'dense_masked', None),
        ('SparseMask $K{=}16$', OI_SKYBLUE, 'sparse_mask', 'mask_knn', 16),
        ('FlexAttention', OI_PURPLE, 'flex_attn', 'flex_attn', None),
    ]
    fig, (ax_time, ax_ratio) = plt.subplots(1, 2, figsize=(8.4, 2.6))

    meta_series = []
    for label, colour, key_b64, key_v2, knn in series:
        for df, batch, style in ((v2, 1, '--'), (b64, 64, '-')):
            rows = df[df.method == (key_v2 if batch == 1 else key_b64)]
            if knn is not None:
                rows = rows[rows.K == knn]
            rows = rows.groupby('N', as_index=False).time_ms.min().sort_values('N')
            if rows.empty:
                continue
            ax_time.plot(rows.N, rows.time_ms, 'o' + style, ms=2.2, lw=0.9,
                         color=colour, zorder=3)
            meta_series.append({'series': label, 'batch': batch,
                                'points': [{'N': int(row.N),
                                            'time_ms': float(row.time_ms)}
                                           for row in rows.itertuples()]})
        # single legend entry per (label, batch) via dashed/solid proxy handles
    handles = []
    for label, colour, _, _, _ in series:
        for batch, style in ((1, '--'), (64, '-')):
            handles.append(mlines.Line2D([], [], color=colour, lw=0.9,
                                         linestyle=style,
                                         label=f'{label}, B={batch}'))
    ax_time.set_xscale('log', base=2)
    ax_time.set_yscale('log')
    ax_time.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    ax_time.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax_time.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax_time.set_xlabel('Sequence length $N$')
    ax_time.set_ylabel('Forward time (ms)')
    ax_time.legend(handles=handles, fontsize=5.2, frameon=False, ncol=2)
    ax_time.set_title('(a) Forward time: B=1 vs B=64', fontsize=8)

    # ratio panel
    for batch, df, key_dense, key_sparse, knn in (
            (1, v2, 'dense_masked', 'mask_knn', 16),
            (64, b64, 'dense_masked', 'sparse_mask', 16)):
        ref = df[(df.method == key_dense)].groupby('N', as_index=False).time_ms.min().set_index('N')
        rows = df[df.method == key_sparse]
        if knn is not None:
            rows = rows[rows.K == knn]
        # dense-masked/dense-flash ratio (the reported architecture penalty)
        ref_flash = df[(df.method == 'dense_flash')].groupby('N', as_index=False).time_ms.min().set_index('N')
        ref = ref.join(ref_flash, rsuffix='_flash')
        rows = rows.groupby('N', as_index=False).time_ms.min().sort_values('N')
        pts = [(int(row.N), float(ref.loc[row.N, 'time_ms'] / ref.loc[row.N, 'time_ms_flash']))
               for row in rows.itertuples() if row.N in ref.index]
        if pts:
            ax_ratio.plot([p[0] for p in pts], [p[1] for p in pts], 'o-',
                          ms=2.2, lw=0.9, color=OI_VERMILION if batch == 1 else OI_BLACK,
                          zorder=3, label=f'DenseMasked / DenseFlash, B={batch}')
            meta_series.append({'series': 'dense_masked/dense_flash',
                                'batch': batch,
                                'points': [{'N': p[0], 'ratio': p[1]} for p in pts]})
        # sparse-mask vs dense-flash ratio (production-relevant ranking)
        pts_sm = [(int(row.N), float(row.time_ms / ref.loc[row.N, 'time_ms_flash']))
                  for row in rows.itertuples() if row.N in ref.index]
        if pts_sm:
            ax_ratio.plot([p[0] for p in pts_sm], [p[1] for p in pts_sm], 'o-',
                          ms=2.2, lw=0.9, color=OI_SKYBLUE,
                          zorder=3, label=f'SparseMask $K{{=}}16$ / DenseFlash, B={batch}')
            meta_series.append({'series': 'sparse_mask16/dense_flash',
                                'batch': batch,
                                'points': [{'N': p[0], 'ratio': p[1]} for p in pts_sm]})
    # FlexAttention vs DenseFlash ratio: the mask-capable survivor at the
    # top of the range (masked + gathered variants OOM at B=64, N>=16384).
    for batch, df in ((1, v2), (64, b64)):
        ref_flash = df[(df.method == 'dense_flash')].groupby(
            'N', as_index=False).time_ms.min().set_index('N')
        rows = df[(df.method == 'flex_attn')].groupby(
            'N', as_index=False).time_ms.min().sort_values('N')
        pts = [(int(row.N), float(row.time_ms / ref_flash.loc[row.N, 'time_ms']))
               for row in rows.itertuples() if row.N in ref_flash.index]
        if pts:
            ax_ratio.plot([p[0] for p in pts], [p[1] for p in pts], 'o-',
                          ms=2.2, lw=0.9, color=OI_PURPLE, zorder=3,
                          label=f'FlexAttention / DenseFlash, B={batch}')
            meta_series.append({'series': 'flex/dense_flash',
                                'batch': batch,
                                'points': [{'N': p[0], 'ratio': p[1]} for p in pts]})
    ax_ratio.axhline(1.0, color='#999999', lw=0.8, linestyle='--', zorder=1)
    ax_ratio.set_xscale('log', base=2)
    ax_ratio.set_ylim(0, 16)
    ax_ratio.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    ax_ratio.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax_ratio.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax_ratio.set_xlabel('Sequence length $N$')
    ax_ratio.set_ylabel('Time ratio vs DenseFlash')
    ax_ratio.legend(fontsize=5.2, frameon=False)
    ax_ratio.set_title('(b) Ratio vs unmasked dense', fontsize=8)
    fig.tight_layout()
    write_meta_sidecar('19_b64_saturation', {
        'sources': [str(B64_CSV), str(H100_CSV)],
        'values': meta_series,
    })
    fig.savefig(FIGS / '19_b64_saturation.pdf')
    fig.savefig(FIGS / '19_b64_saturation.svg')
    plt.close(fig)
    return '19_b64_saturation'


def panel_mask_decomposition():
    """Figure ``20_mask_decomposition``: where the masked-dense cost enters.

    Reads ``cudnn_vs_fa2_mask_decomposition.csv`` (stages A-E, B=1, fp16,
    d_model=320/n_head=8; kernel-level arms at d_head in {40, 64}).

    Panel (a): stage C, the additive-mask construction of
    CachedDistAttention._build_mask in isolation (v0/v1 decay, cache
    on/off) — quadratic in N and caller-owned.

    Panel (b): stage B/D/E kernel-level times at d_head=64 (both kernels
    dispatchable): unmasked FA2 vs unmasked cuDNN (kernel-intrinsic
    parity), cuDNN with the prebuilt float mask (carriage penalty),
    mem-efficient with the same mask, and cuDNN with is_causal=True (no
    NxN mask input at all).

    Panel (c): stage E incremental peak memory — cuDNN causal (linear)
    vs cuDNN with a prebuilt BOOL mask (the dispatcher materializes it to
    float at O(N^2); 1032 MB at N=8192).

    Returns the saved figure name for the html report list.
    """
    df = pd.read_csv(DECOMP_CSV)
    fig, (ax_build, ax_kern, ax_mem) = plt.subplots(1, 3, figsize=(8.6, 2.6))

    # (a) mask build isolation
    build = df[(df.stage == 'C_mask_build_isolated')]
    for (dm, dc), colour in ((('v1', 'on'), OI_VERMILION), (('v1', 'off'), OI_BLACK),
                             (('v0', 'on'), OI_SKYBLUE), (('v0', 'off'), OI_PURPLE)):
        rows = build[(build.dist_mode == dm) & (build.dist_cache == dc) &
                     (build.d_head == 40)].sort_values('N')
        if rows.empty:
            continue
        ax_build.plot(rows.N, rows.time_ms, 'o-', ms=2.0, lw=0.8, color=colour,
                      zorder=3, label=f'$v_1$ decay, cache {dc}' if dm == 'v1'
                      else f'$v_0$ decay, cache {dc}')
    ax_build.set_xscale('log', base=2)
    ax_build.set_yscale('log')
    ax_build.set_xlabel('Sequence length $N$')
    ax_build.set_ylabel('Mask build time (ms)')
    ax_build.legend(fontsize=5.2, frameon=False)
    ax_build.set_title('(a) Mask construction (stage C)', fontsize=8)

    # (b) kernel-level times at d_head=64
    kern = df[(df.d_head == 64)]
    series_b = [
        ('flash_nomask', 'FA2, no mask', OI_BLACK, 'B_kernel_intrinsic_nomask'),
        ('cudnn_nomask', 'cuDNN, no mask', OI_VERMILION, 'B_kernel_intrinsic_nomask'),
        ('cudnn_masked_prebuilt', 'cuDNN, prebuilt float mask', OI_ORANGE, 'D_mask_carriage'),
        ('efficient_masked_prebuilt', 'cutlass, prebuilt float mask', OI_SKYBLUE, 'D_mask_carriage'),
        ('cudnn_causal_procedural', 'cuDNN, causal (no mask input)', OI_PURPLE, 'E_causal_vs_mask_dtype'),
    ]
    for method, label, colour, stage in series_b:
        rows = kern[(kern.stage == stage) & (kern.method == method) &
                    kern.time_ms.notna() & (kern.time_ms > 0)].sort_values('N')
        if rows.empty:
            continue
        ax_kern.plot(rows.N, rows.time_ms, 'o-', ms=2.0, lw=0.8, color=colour,
                     zorder=3, label=label)
    ax_kern.set_xscale('log', base=2)
    ax_kern.set_yscale('log')
    ax_kern.set_xlabel('Sequence length $N$')
    ax_kern.set_ylabel('Forward time (ms)')
    ax_kern.legend(fontsize=4.6, frameon=False)
    ax_kern.set_title('(b) Kernel arms, $d_{head}=64$', fontsize=8)

    # (c) memory: causal vs carried bool mask
    mem = df[(df.stage == 'E_causal_vs_mask_dtype') & (df.d_head == 64)]
    for method, label, colour in (
            ('cudnn_causal_procedural', 'cuDNN causal (no mask input)', OI_PURPLE),
            ('cudnn_bool_prebuilt', 'cuDNN, prebuilt bool mask', OI_ORANGE)):
        rows = mem[(mem.method == method) & mem.mem_mb.notna() &
                   (mem.mem_mb > 0)].sort_values('N')
        if rows.empty:
            continue
        ax_mem.plot(rows.N, rows.mem_mb, 'o-', ms=2.0, lw=0.8, color=colour,
                    zorder=3, label=label)
    ax_mem.set_xscale('log', base=2)
    ax_mem.set_yscale('log')
    ax_mem.set_xlabel('Sequence length $N$')
    ax_mem.set_ylabel('Incremental peak memory (MB)')
    ax_mem.legend(fontsize=5.2, frameon=False)
    ax_mem.set_title('(c) Mask dtype materialization', fontsize=8)

    for ax in (ax_build, ax_kern, ax_mem):
        ax.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192])
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
        ax.tick_params(labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', which='major', color='#cccccc', linewidth=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout()

    write_meta_sidecar('20_mask_decomposition', {
        'sources': [str(DECOMP_CSV)],
        'values': [
            {'series': 'stage_C_mask_build_v1',
             'points': [{'N': int(row.N), 'time_ms': float(row.time_ms)}
                        for row in build[(build.dist_mode == 'v1') &
                                         (build.dist_cache == 'on') &
                                         (build.d_head == 40)].sort_values('N').itertuples()]},
            {'series': 'stage_B_cudnn_nomask_d64',
             'points': [{'N': int(row.N), 'time_ms': float(row.time_ms)}
                        for row in kern[(kern.stage == 'B_kernel_intrinsic_nomask') &
                                        (kern.method == 'cudnn_nomask')].sort_values('N').itertuples()]},
            {'series': 'stage_D_cudnn_masked_prebuilt_d64',
             'points': [{'N': int(row.N), 'time_ms': float(row.time_ms)}
                        for row in kern[(kern.stage == 'D_mask_carriage') &
                                        (kern.method == 'cudnn_masked_prebuilt')].sort_values('N').itertuples()]},
            {'series': 'stage_E_cudnn_causal_d64',
             'points': [{'N': int(row.N), 'time_ms': float(row.time_ms)}
                        for row in kern[(kern.stage == 'E_causal_vs_mask_dtype') &
                                        (kern.method == 'cudnn_causal_procedural')].sort_values('N').itertuples()]},
            {'series': 'stage_E_cudnn_bool_prebuilt_mem_d64',
             'points': [{'N': int(row.N), 'mem_mb': float(row.mem_mb)}
                        for row in mem[(mem.method == 'cudnn_bool_prebuilt')].sort_values('N').itertuples()]},
        ],
    })
    fig.savefig(FIGS / '20_mask_decomposition.pdf')
    fig.savefig(FIGS / '20_mask_decomposition.svg')
    plt.close(fig)
    return '20_mask_decomposition'


# ═══════════════════════════════════════════════════════════════
# 9. Per-Epoch Training Time
# ═══════════════════════════════════════════════════════════════
def panel_per_epoch_time():
    # Ascending K with the dense baseline (dense = all neighbours) last: it is
    # the right-hand limit of the K axis, matching the other K-sweep figures.
    models = ['k4', 'k16', 'k32', 'k64', 'baseline']
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
        ('SparseMask (scatter)', 97, [
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
    """SSL negative-result figure: loss and accuracy strictly on separate axes.

    D-12 fix (owner decision 2026-10-08): the previous version mixed two
    validation BCE losses with a downstream TRA value on ONE bar axis
    (ylabel 'Val Loss (BCE) / TRA (Contrastive)'), so the direction of
    "good" flipped between bars. Now: left = BCE losses only, right =
    downstream TRA on an accuracy axis. The identity-BCE convergence
    curves live in panel_ssl_convergence (appendix figure 17), per the
    owner decision to move the loss-decay graph to the appendix.
    TRA values are the frozen-embedding + Hungarian label-sweep proxy
    results reported in evaluation §3.6 (NT-Xent 0.651 vs random 0.648).
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.5, 4.2), gridspec_kw={'width_ratios': [1, 0.8]})

    # LEFT: fine-tune comparison — validation BCE loss only (lower = better).
    methods = ['From scratch\n(10% labels)', 'SSL identity BCE\n+ finetune']
    vals = [0.416, 0.404]
    bars = ax1.bar(range(len(methods)), vals, color=[OI_BLACK, OI_BLUE],
                   width=0.5, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, fontsize=8)
    ax1.set_ylabel('Validation BCE loss')

    # RIGHT: NT-Xent downstream accuracy — own axis, same unit for both bars.
    tra_labels = ['NT-Xent\n(frozen + proxy)', 'Random\nbaseline']
    tra_vals = [0.651, 0.648]
    bars = ax2.bar(range(len(tra_labels)), tra_vals,
                   color=[OI_VERMILION, '#999999'], width=0.5, edgecolor='white')
    for bar, v in zip(bars, tra_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(range(len(tra_labels)))
    ax2.set_xticklabels(tra_labels, fontsize=8)
    ax2.set_ylabel('TRA (frozen-embedding proxy)')

    fig.tight_layout()
    return save('13_ssl_negative')


def panel_ssl_convergence():
    """Appendix figure: identity-BCE pretraining convergence curves.

    Split off from panel_ssl_negative per the owner decision to move the
    loss-decay graph to the appendix (D-12a/D-13). Reads the identity-BCE
    training log (ssl_v1/training_log.csv); the dashed line marks the
    from-scratch baseline's final validation loss (0.416) that the
    fine-tuned SSL model marginally undercuts (0.404).
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    df_ssl = pd.read_csv(ROOT / 'benchmark_ssl' / 'results' / 'runs' / 'ssl_v1' / 'training_log.csv')
    ax.plot(df_ssl.epoch, df_ssl.train_loss, 'o-', label='Train loss', color=OI_BLUE, linewidth=2)
    ax.plot(df_ssl.epoch, df_ssl.val_loss, 's-', label='Val loss', color=OI_VERMILION, linewidth=2)
    ax.axhline(y=0.416, color='#999999', linestyle='--', alpha=0.5, label='Baseline final (0.416)')
    ax.set_xlabel('Identity-BCE pretraining epoch')
    ax.set_ylabel('BCE loss')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save('17_ssl_convergence')


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
is 7.9× faster than gather at N=256. DenseFlash remains 1.3× faster than SparseMask.</p>
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
    figs.append(panel_dist_cache_effect())
    figs.append(panel_b64_saturation())
    figs.append(panel_mask_decomposition())
    figs.append(panel_convergence())
    figs.append(panel_tra())
    figs.append(panel_aogm())
    figs.append(panel_edge_div_f1())
    figs.append(panel_per_epoch_time())
    figs.append(panel_speedup_heatmap())
    figs.append(panel_profiler_breakdown())
    figs.append(panel_ssl_negative())
    figs.append(panel_ssl_convergence())
    figs.append(panel_layer_scaling())
    print(f'\n{len(figs)} figures generated in {FIGS}')
    print('Generating report.html...')
    generate_html(figs)
    print('Done!')
