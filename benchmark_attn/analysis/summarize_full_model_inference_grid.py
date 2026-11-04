#!/usr/bin/env python3
"""Summarize the SPEC 0062 full-model inference grid raw CSV.

The raw benchmark CSV contains one row per ``(variant, N, B, repeat_idx)`` with
``status`` set to ``ok``, ``oom``, or ``err: <message>``. This script groups
all rows by ``(variant, N, B, linker_mode)``, counts the outcomes, and computes
mean plus sample standard deviation (``n-1``) over the ``ok`` repeats only.
It also derives the relative end-to-end speedups and attention shares from the
SPEC 0062 §4.2 contract.

Only ``status="ok"`` repeats contribute to the numeric statistics. An
``oom``/``err`` repeat is counted but never silently dropped: a cell with no
successful repeat keeps its row with ``n_ok=0`` and empty numeric cells.

Usage (from ``benchmark_attn``)::

    python analysis/summarize_full_model_inference_grid.py

    python analysis/summarize_full_model_inference_grid.py \
        --raw results/full_model_inference_grid_smoke.csv \
        --env-json results/full_model_inference_grid_smoke_env.json \
        --out /tmp/f6_smoke_summary.csv

The optional ``--plot`` writes a PNG when matplotlib is importable; pass a
path or use the flag alone for the default summary PNG.

The optional ``--plot-all`` writes an eight-panel PNG of every measured
summary metric — total and forward speedup, total/full-forward time,
attention module time, normalisation time, CPU linker time, and attention
share — against ``N`` on a log2 axis, one line per variant, with the same
matplotlib fallback behaviour as ``--plot``.

The optional ``--plot-parts`` writes three decomposition PNGs from one path
prefix: ``<prefix>_time_fractions.png`` shows, per variant (2×2 panels,
``B=1`` only), a 100% stacked area over ``N`` of the attention
(``encoder_attn_ms_mean + decoder_attn_ms_mean``), normalisation
(``normalize_ms_mean``), linker (``linker_ms_mean``), and remaining "other"
fractions of ``total_ms_mean``. ``<prefix>_forward_time.png`` and
``<prefix>_memory.png`` are separate single-panel log-log figures styled
after the attention-only H100 report panels
(``curated/generate_report.py::panel_speed_vs_n_h100`` and
``panel_memory_vs_n``): one solid measured line per canonical variant in
Okabe-Ito colours (``o`` markers, size 2.2, linewidth 0.8), a frameless
method legend in the upper left, y major ticks labelled as base-10 powers
via ``LogFormatterMathtext``, and no projections because every plotted
``N`` is measured. The forward-time figure plots ``full_forward_ms_mean``;
the memory figure plots ``peak_mem_mb_max`` plus a grey dashed 94000 MB
H100 VRAM reference line with a small label. All three figures space ``N``
logarithmically (base 2) but label the x ticks with the plain integer
sequence lengths (512, 1024, 2048, …), never power notation, and skip with
a warning when matplotlib is unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# ─── Contract constants ─────────────────────────────────────────────────────

#: Numeric columns that receive a mean and a sample standard deviation.
METRIC_COLUMNS: tuple[str, ...] = (
    "full_forward_ms",
    "total_ms",
    "embed_ms",
    "encoder_ms",
    "encoder_attn_ms",
    "decoder_ms",
    "decoder_attn_ms",
    "head_logits_ms",
    "einsum_ms",
    "normalize_ms",
    "linker_build_graph_ms",
    "linker_greedy_ms",
    "linker_ms",
)

#: Grouping keys for one summary row.
GROUP_KEYS: tuple[str, ...] = ("variant", "N", "B", "linker_mode")

#: The baseline variant used for the relative speedups.
BASELINE_VARIANT = "relative_pos"

#: The sequence length of the compact printed table.
TABLE_SEQUENCE_LENGTH = 32768

DEFAULT_RAW_CSV = Path("results/full_model_inference_grid.csv")
DEFAULT_ENV_JSON = Path("results/full_model_inference_grid_env.json")
DEFAULT_SUMMARY_CSV = Path("results/full_model_inference_grid_summary.csv")
DEFAULT_SUMMARY_PNG = Path("results/full_model_inference_grid_summary.png")
DEFAULT_ALL_METRICS_PNG = Path(
    "results/full_model_inference_grid_all_metrics.png"
)
DEFAULT_PARTS_PNG_PREFIX = Path("results/full_model_inference_grid_parts")

#: Canonical variant panel order of the ``--plot-parts`` time-fraction
#: figure (SPEC 0062 variant table).
PARTS_VARIANT_ORDER: tuple[str, ...] = (
    "relative_pos",
    "cached_dist",
    "sparse_mask_k16",
    "sparse_gather_k16",
)

#: Label and fill colour of the four stacked time-fraction components, in
#: stacking order (bottom to top).
TIME_FRACTION_STYLES: tuple[tuple[str, str], ...] = (
    ("attention (encoder+decoder attn)", "#1f77b4"),
    ("normalize", "#2ca02c"),
    ("linker", "#d62728"),
    ("other (residual of total_ms)", "#7f7f7f"),
)

#: Okabe-Ito colour per canonical ``--plot-parts`` variant, matching the
#: attention-only H100 panels in ``curated/generate_report.py``
#: (``panel_speed_vs_n_h100`` / ``panel_memory_vs_n``): the ``relative_pos``
#: baseline is black, the ``cached_dist`` ablation vermilion,
#: ``sparse_mask_k16`` sky blue, and ``sparse_gather_k16`` orange.
PARTS_VARIANT_COLORS: dict[str, str] = {
    "relative_pos": "#000000",
    "cached_dist": "#D55E00",
    "sparse_mask_k16": "#56B4E9",
    "sparse_gather_k16": "#E69F00",
}

#: Figure size of the ``--plot-parts`` forward-time and memory figures,
#: matching ``H100_FIGSIZE`` of the attention-only H100 panels.
PARTS_LINE_FIGSIZE: tuple[float, float] = (5.6, 2.6)

#: Grey dashed H100 VRAM reference line of the ``--plot-parts`` memory
#: figure, in MB (the usable VRAM of an 80 GB H100, matching the attention-
#: only memory panel).
PARTS_MEMORY_VRAM_MB: float = 94000.0


class AllMetricsPanel(NamedTuple):
    """Definition of one panel of the ``--plot-all`` figure.

    Attributes:
        title: Panel heading.
        ylabel: Axis label describing the plotted metric.
        value_column: Summary column to plot (``*_mean`` columns for timings,
            derived columns for speedups and shares).
        log_y: Whether the y-axis is log-scaled; log panels draw strictly
            positive values only.
    """

    title: str
    ylabel: str
    value_column: str
    log_y: bool


#: Panel order and scaling of the ``--plot-all`` figure (SPEC 0062 §11).
ALL_METRICS_PANELS: tuple[AllMetricsPanel, ...] = (
    AllMetricsPanel(
        title="End-to-end total speedup",
        ylabel="total speedup vs relative_pos",
        value_column="speedup_vs_relative_pos",
        log_y=False,
    ),
    AllMetricsPanel(
        title="Forward-chain speedup",
        ylabel="forward speedup vs relative_pos",
        value_column="forward_speedup_vs_relative_pos",
        log_y=False,
    ),
    AllMetricsPanel(
        title="End-to-end total time",
        ylabel="total_ms (ms)",
        value_column="total_ms_mean",
        log_y=True,
    ),
    AllMetricsPanel(
        title="Full forward time",
        ylabel="full_forward_ms (ms)",
        value_column="full_forward_ms_mean",
        log_y=True,
    ),
    AllMetricsPanel(
        title="Attention module time",
        ylabel="encoder_attn + decoder_attn (ms)",
        value_column="attention_module_ms",
        log_y=True,
    ),
    AllMetricsPanel(
        title="Output normalisation time",
        ylabel="normalize_ms (ms)",
        value_column="normalize_ms_mean",
        log_y=True,
    ),
    AllMetricsPanel(
        title="CPU linker time (B=1 greedy)",
        ylabel="linker_ms (ms)",
        value_column="linker_ms_mean",
        log_y=True,
    ),
    AllMetricsPanel(
        title="Attention share of total",
        ylabel="attention_share of total_ms",
        value_column="attention_share",
        log_y=False,
    ),
)


# ─── CLI ────────────────────────────────────────────────────────────────────


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the input/output paths for the summary run.

    Args:
        argv: Optional argument list for tests; defaults to ``sys.argv``.

    Returns:
        The parsed namespace with ``raw``, ``env_json``, ``out``, and the
        optional ``plot``, ``plot_all``, and ``plot_parts`` paths.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Group the SPEC 0062 full-model inference raw CSV into a summary "
            "table with counts, means, sample standard deviations, relative "
            "speedups, and attention shares."
        )
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=DEFAULT_RAW_CSV,
        help=f"Raw per-repeat CSV path (default: {DEFAULT_RAW_CSV}).",
    )
    parser.add_argument(
        "--env-json",
        type=Path,
        default=DEFAULT_ENV_JSON,
        help=(
            "Environment JSON path used for provenance display "
            f"(default: {DEFAULT_ENV_JSON})."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help=f"Summary CSV output path (default: {DEFAULT_SUMMARY_CSV}).",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const=DEFAULT_SUMMARY_PNG,
        default=None,
        type=Path,
        help=(
            "Optional summary PNG path. Pass the flag alone to use "
            f"{DEFAULT_SUMMARY_PNG}; omit it to skip plotting."
        ),
    )
    parser.add_argument(
        "--plot-all",
        nargs="?",
        const=DEFAULT_ALL_METRICS_PNG,
        default=None,
        type=Path,
        help=(
            "Optional eight-panel PNG of every measured metric (total and "
            "forward speedup, total/full-forward/attention/normalize/linker "
            "times, attention share) against N on a log2 axis. Pass the flag "
            f"alone to use {DEFAULT_ALL_METRICS_PNG}; omit it to skip."
        ),
    )
    parser.add_argument(
        "--plot-parts",
        nargs="?",
        const=DEFAULT_PARTS_PNG_PREFIX,
        default=None,
        type=Path,
        help=(
            "Optional three-figure decomposition PNG prefix. Writes "
            "<prefix>_time_fractions.png (2x2 100%%-stacked time-fraction "
            "areas per variant, B=1), <prefix>_forward_time.png (log-log "
            "forward time vs sequence length, B=1), and <prefix>_memory.png "
            "(log-log GPU memory vs sequence length with the 94 GB H100 "
            "VRAM line, B=1); plain integer N ticks on all three. Pass the "
            f"flag alone to use {DEFAULT_PARTS_PNG_PREFIX}; omit it to skip."
        ),
    )
    return parser.parse_args(argv)


# ─── Loading ────────────────────────────────────────────────────────────────


def _expected_raw_columns() -> list[str]:
    """Return the minimum raw columns needed to build the summary.

    Returns:
        The group keys, status, all metric columns, and the peak memory column.
    """
    return [*GROUP_KEYS, "status", *METRIC_COLUMNS, "peak_mem_mb"]


def load_raw_csv(raw_path: Path) -> pd.DataFrame:
    """Load the raw per-repeat CSV and coerce its numeric columns.

    Empty cells for timing columns are converted to ``NaN`` so failure rows and
    partially populated rows do not break the aggregation.

    Args:
        raw_path: Path to the raw CSV produced by the benchmark script.

    Returns:
        A DataFrame with text columns as strings and timing columns as floats.

    Raises:
        FileNotFoundError: If ``raw_path`` does not exist.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw_path}")

    try:
        frame = pd.read_csv(raw_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=_expected_raw_columns())

    missing_keys = [
        key
        for key in (*GROUP_KEYS, "status")
        if key not in frame.columns
    ]
    if missing_keys:
        raise ValueError(
            f"Raw CSV is missing required group columns {missing_keys}: "
            f"{raw_path}"
        )

    typed = frame.copy()
    for column in ("N", "B"):
        typed[column] = pd.to_numeric(
            typed[column], errors="coerce"
        ).astype("Int64")
    for column in (*METRIC_COLUMNS, "peak_mem_mb"):
        if column not in typed.columns:
            typed[column] = np.nan
        typed[column] = pd.to_numeric(typed[column], errors="coerce")
    for column in ("variant", "linker_mode", "status"):
        typed[column] = typed[column].fillna("").astype(str)
    return typed


def load_environment_json(env_path: Path) -> dict[str, object]:
    """Read the environment JSON for provenance display.

    The summary math does not depend on the environment record, so a missing or
    malformed file is reported and treated as empty rather than aborting the
    aggregation.

    Args:
        env_path: Path to the environment JSON written by the benchmark run.

    Returns:
        The parsed mapping, or an empty dict when unavailable.
    """
    if not env_path.exists():
        print(f"[warn] environment JSON not found: {env_path}")
        return {}
    try:
        with open(env_path) as env_file:
            loaded = json.load(env_file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"[warn] could not read environment JSON {env_path}: {error}")
        return {}
    return loaded if isinstance(loaded, dict) else {}


# ─── Aggregation ────────────────────────────────────────────────────────────


def summarize_grid(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Build one summary row per ``(variant, N, B, linker_mode)``.

    Counts use all repeats. Means, sample standard deviations, and the maximum
    peak memory use only ``status="ok"`` repeats. A single successful repeat
    has zero sample standard deviation by definition here, matching the
    ticket's single-repeat requirement.

    Args:
        raw_frame: Typed raw rows from :func:`load_raw_csv`.

    Returns:
        The summary frame in the SPEC 0062 §4.2 column order.
    """
    summary_columns = build_summary_columns()
    if raw_frame.empty:
        return pd.DataFrame(columns=summary_columns)

    working = raw_frame.copy()
    normalized_status = (
        working["status"].fillna("").astype(str).str.strip().str.lower()
    )
    working["_is_ok"] = normalized_status.eq("ok")
    working["_is_oom"] = normalized_status.eq("oom")
    working["_is_err"] = normalized_status.str.startswith("err")

    counts = (
        working.groupby(list(GROUP_KEYS), dropna=False, sort=False)
        .agg(
            n_ok=("_is_ok", "sum"),
            n_oom=("_is_oom", "sum"),
            n_err=("_is_err", "sum"),
        )
        .reset_index()
    )

    ok_rows = working[working["_is_ok"]]
    if ok_rows.empty:
        statistics = counts[list(GROUP_KEYS)].copy()
        for column in METRIC_COLUMNS:
            statistics[f"{column}_mean"] = np.nan
            statistics[f"{column}_std"] = np.nan
        statistics["peak_mem_mb_max"] = np.nan
    else:
        grouped_ok = ok_rows.groupby(
            list(GROUP_KEYS), dropna=False, sort=False
        )
        means = grouped_ok[list(METRIC_COLUMNS)].mean()
        standard_deviations = grouped_ok[list(METRIC_COLUMNS)].std(ddof=1)
        non_missing_counts = grouped_ok[list(METRIC_COLUMNS)].count()
        standard_deviations = standard_deviations.mask(
            non_missing_counts == 1, 0.0
        )

        means = means.rename(columns=lambda name: f"{name}_mean")
        standard_deviations = standard_deviations.rename(
            columns=lambda name: f"{name}_std"
        )
        statistics = means.join(standard_deviations)
        statistics["peak_mem_mb_max"] = grouped_ok["peak_mem_mb"].max()
        statistics = statistics.reset_index()

    summary = counts.merge(
        statistics, on=list(GROUP_KEYS), how="left", sort=False
    )

    exact_baseline, relaxed_baseline = _build_baseline_lookup(
        summary, "total_ms_mean"
    )
    exact_forward_baseline, relaxed_forward_baseline = (
        _build_baseline_lookup(summary, "full_forward_ms_mean")
    )

    summary["speedup_vs_relative_pos"] = [
        _relative_speedup(
            sequence_length=row.N,
            batch_size=row.B,
            linker_mode=row.linker_mode,
            value=row.total_ms_mean,
            exact_baseline=exact_baseline,
            relaxed_baseline=relaxed_baseline,
        )
        for row in summary.itertuples(index=False)
    ]
    summary["attention_share"] = [
        _safe_divide(
            _finite_sum(row.encoder_attn_ms_mean, row.decoder_attn_ms_mean),
            row.total_ms_mean,
        )
        for row in summary.itertuples(index=False)
    ]
    summary["attention_share_of_forward"] = [
        _safe_divide(
            _finite_sum(row.encoder_attn_ms_mean, row.decoder_attn_ms_mean),
            row.full_forward_ms_mean,
        )
        for row in summary.itertuples(index=False)
    ]
    summary["forward_speedup_vs_relative_pos"] = [
        _relative_speedup(
            sequence_length=row.N,
            batch_size=row.B,
            linker_mode=row.linker_mode,
            value=row.full_forward_ms_mean,
            exact_baseline=exact_forward_baseline,
            relaxed_baseline=relaxed_forward_baseline,
        )
        for row in summary.itertuples(index=False)
    ]

    summary = summary.reindex(columns=summary_columns)
    summary = summary.sort_values(
        ["N", "B", "variant", "linker_mode"], kind="stable"
    ).reset_index(drop=True)
    return summary


def build_summary_columns() -> list[str]:
    """Return the summary CSV column order.

    Returns:
        Group keys, counts, per-metric mean/std pairs, peak memory maximum,
        and the derived speedup/share columns.
    """
    columns = [*GROUP_KEYS, "n_ok", "n_oom", "n_err"]
    for column in METRIC_COLUMNS:
        columns.extend((f"{column}_mean", f"{column}_std"))
    columns.extend(
        (
            "peak_mem_mb_max",
            "speedup_vs_relative_pos",
            "attention_share",
            "attention_share_of_forward",
            "forward_speedup_vs_relative_pos",
        )
    )
    return columns


def _build_baseline_lookup(
    summary: pd.DataFrame, value_column: str
) -> tuple[dict[tuple[int, int, str], float], dict[tuple[int, int], float]]:
    """Index baseline variant means for the relative speedup columns.

    The exact lookup keys on ``(N, B, linker_mode)`` so an analysis with more
    than one linker mode compares like with like. The relaxed lookup keys only
    on ``(N, B)`` and is used when the matching linker-mode baseline is absent.

    Args:
        summary: The partially built summary frame.
        value_column: Either ``total_ms_mean`` or ``full_forward_ms_mean``.

    Returns:
        A pair of dictionaries ``(exact, relaxed)`` mapping the keys to the
        baseline mean.
    """
    exact: dict[tuple[int, int, str], float] = {}
    relaxed: dict[tuple[int, int], float] = {}
    baseline_rows = summary[summary["variant"] == BASELINE_VARIANT]
    for row in baseline_rows.itertuples(index=False):
        value = getattr(row, value_column)
        if pd.isna(value):
            continue
        try:
            key = (int(row.N), int(row.B))
        except (TypeError, ValueError):
            continue
        exact[(key[0], key[1], str(row.linker_mode))] = float(value)
        relaxed.setdefault(key, float(value))
    return exact, relaxed


def _relative_speedup(
    sequence_length: object,
    batch_size: object,
    linker_mode: str,
    value: float,
    exact_baseline: dict[tuple[int, int, str], float],
    relaxed_baseline: dict[tuple[int, int], float],
) -> float:
    """Return ``baseline_mean / variant_mean`` at the same grid point.

    Args:
        sequence_length: ``N`` of the row.
        batch_size: ``B`` of the row.
        linker_mode: Linker mode of the row.
        value: The row's mean for the compared metric.
        exact_baseline: Baseline lookup keyed by ``(N, B, linker_mode)``.
        relaxed_baseline: Baseline lookup keyed by ``(N, B)``.

    Returns:
        The speedup ratio, or ``NaN`` when the variant or baseline mean is
        missing, zero, or the baseline OOMed.
    """
    if pd.isna(value) or float(value) == 0.0:
        return np.nan
    try:
        baseline_key = (int(sequence_length), int(batch_size))
    except (TypeError, ValueError):
        return np.nan
    baseline = exact_baseline.get((*baseline_key, str(linker_mode)))
    if baseline is None:
        baseline = relaxed_baseline.get(baseline_key)
    if baseline is None or baseline == 0.0:
        return np.nan
    return baseline / float(value)


def _finite_sum(first: float, second: float) -> float:
    """Return ``first + second`` unless either operand is missing.

    Args:
        first: First timing mean.
        second: Second timing mean.

    Returns:
        The sum, or ``NaN`` when an operand is not finite.
    """
    if pd.isna(first) or pd.isna(second):
        return np.nan
    return float(first) + float(second)


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide two summary values, returning ``NaN`` on a missing denominator.

    Args:
        numerator: The derived numerator.
        denominator: The timing denominator.

    Returns:
        The ratio, or ``NaN`` when either value is missing or the denominator
        is zero.
    """
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0.0:
        return np.nan
    return float(numerator) / float(denominator)


# ─── Output ─────────────────────────────────────────────────────────────────


def write_summary_csv(summary: pd.DataFrame, out_path: Path) -> None:
    """Write the summary CSV, using empty cells for unavailable metrics.

    Args:
        summary: The summary frame from :func:`summarize_grid`.
        out_path: Destination CSV path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False, na_rep="")


def format_compact_table(summary: pd.DataFrame, sequence_length: int) -> str:
    """Render the compact total-speedup and attention-share table.

    Args:
        summary: The summary frame.
        sequence_length: The ``N`` to select, usually 32768.

    Returns:
        A printable multi-line table. The first line is a heading; when no
        rows exist at that ``N`` the body says so.
    """
    subset = summary[summary["N"] == sequence_length]
    heading = (
        f"N={sequence_length} total speedup vs {BASELINE_VARIANT} and "
        "attention share"
    )
    if subset.empty:
        return "\n".join((heading, "(no rows at this N in the raw CSV)"))

    lines = [
        heading,
        f"{'variant':<20} {'n_ok':>5} {'total_speedup':>15} {'attn_share':>12}",
        "-" * 55,
    ]
    for row in subset.itertuples(index=False):
        speedup_text = (
            f"{row.speedup_vs_relative_pos:.2f}x"
            if row.n_ok > 0 and not pd.isna(row.speedup_vs_relative_pos)
            else "n/a"
        )
        share_text = (
            f"{row.attention_share:.3f}"
            if row.n_ok > 0 and not pd.isna(row.attention_share)
            else "n/a"
        )
        lines.append(
            f"{str(row.variant):<20} {int(row.n_ok):>5} "
            f"{speedup_text:>15} {share_text:>12}"
        )
    return "\n".join(lines)


def write_summary_plot(summary: pd.DataFrame, plot_path: Path) -> bool:
    """Write an optional speedup/attention-share PNG when matplotlib exists.

    Args:
        summary: The summary frame.
        plot_path: Destination PNG path.

    Returns:
        ``True`` when a figure was written, ``False`` when matplotlib is
        unavailable, no rows are plottable, or the file cannot be written.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib is not available; skipping --plot")
        return False

    plottable = summary.dropna(
        subset=["speedup_vs_relative_pos", "attention_share"], how="all"
    )
    if plottable.empty:
        print("[warn] no numeric summary rows to plot; skipping --plot")
        return False

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for variant_name, variant_rows in plottable.groupby("variant", sort=False):
        variant_rows = variant_rows.sort_values("N")
        axes[0].plot(
            variant_rows["N"],
            variant_rows["speedup_vs_relative_pos"],
            marker="o",
            label=variant_name,
        )
        axes[1].plot(
            variant_rows["N"],
            variant_rows["attention_share"],
            marker="o",
            label=variant_name,
        )

    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("total speedup vs relative_pos")
    axes[0].set_title("End-to-end total speedup")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize="small")

    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("attention share of total_ms")
    axes[1].set_title("Attention share")
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    except OSError as error:
        print(f"[warn] could not write plot {plot_path}: {error}")
        plt.close(figure)
        return False
    plt.close(figure)
    print(f"wrote summary plot: {plot_path}")
    return True


def _prepare_all_metrics_frame(summary: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the summary with the derived attention-module column.

    The attention module time is ``encoder_attn_ms_mean +
    decoder_attn_ms_mean``; it is ``NaN`` when either operand is missing,
    matching the ``attention_share`` semantics. The copy leaves the caller's
    frame untouched.

    Args:
        summary: The summary frame from :func:`summarize_grid`.

    Returns:
        A copy with an ``attention_module_ms`` column (all ``NaN`` when the
        encoder/decoder attention mean columns are absent).
    """
    plot_frame = summary.copy()
    attention_columns = ("encoder_attn_ms_mean", "decoder_attn_ms_mean")
    if set(attention_columns).issubset(plot_frame.columns):
        plot_frame["attention_module_ms"] = [
            _finite_sum(row.encoder_attn_ms_mean, row.decoder_attn_ms_mean)
            for row in plot_frame.itertuples(index=False)
        ]
    else:
        plot_frame["attention_module_ms"] = np.nan
    return plot_frame


def _annotate_empty_panel(axis: "plt.Axes") -> None:
    """Mark a panel that has no plottable value for any variant.

    Args:
        axis: The matplotlib axes to annotate.
    """
    axis.text(
        0.5,
        0.5,
        "no data",
        ha="center",
        va="center",
        transform=axis.transAxes,
        color="gray",
    )


def _draw_all_metrics_panel(
    axis: "plt.Axes",
    panel: AllMetricsPanel,
    plot_frame: pd.DataFrame,
    variant_colors: dict[str, str],
) -> bool:
    """Draw one metric panel of the all-metrics figure.

    Values are plotted per variant in the frame's variant order so the shared
    figure legend matches every panel. NaN/missing values are skipped; log
    panels additionally skip non-positive values because they cannot be shown
    on a log axis.

    Args:
        axis: The matplotlib axes to draw on.
        panel: Panel definition (title, label, column, y scaling).
        plot_frame: Summary frame including the derived attention-module
            column from :func:`_prepare_all_metrics_frame`.
        variant_colors: Variant name to matplotlib color mapping shared by
            all panels.

    Returns:
        ``True`` when at least one line was drawn, ``False`` otherwise.
    """
    axis.set_xscale("log", base=2)
    if panel.log_y:
        axis.set_yscale("log")
    axis.grid(True, alpha=0.3)
    axis.set_xlabel("N")
    axis.set_ylabel(panel.ylabel)
    axis.set_title(panel.title, fontsize="medium")

    if panel.value_column not in plot_frame.columns:
        _annotate_empty_panel(axis)
        return False

    drew_line = False
    for variant_name, variant_rows in plot_frame.groupby(
        "variant", sort=False
    ):
        variant_rows = variant_rows.sort_values("N")
        panel_values = pd.to_numeric(
            variant_rows[panel.value_column], errors="coerce"
        )
        plottable = panel_values.notna()
        if panel.log_y:
            plottable &= panel_values > 0
        if not plottable.any():
            continue
        kept_rows = variant_rows[plottable]
        axis.plot(
            kept_rows["N"],
            kept_rows[panel.value_column],
            marker="o",
            color=variant_colors[str(variant_name)],
            label=str(variant_name),
        )
        drew_line = True
    if not drew_line:
        _annotate_empty_panel(axis)
    return drew_line


def write_all_metrics_plot(summary: pd.DataFrame, plot_path: Path) -> bool:
    """Write the eight-panel all-metrics PNG when matplotlib exists.

    The figure shows every measured summary metric against ``N`` on a log2
    x-axis with one line per variant and one shared legend (SPEC 0062 §11).
    A variant color is fixed across panels so a variant missing from one
    panel (for example the ``B=1``-only linker time) keeps its color
    everywhere.

    Args:
        summary: The summary frame from :func:`summarize_grid`.
        plot_path: Destination PNG path.

    Returns:
        ``True`` when a figure was written, ``False`` when matplotlib is
        unavailable, no rows are plottable, or the file cannot be written.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib is not available; skipping --plot-all")
        return False

    if summary.empty:
        print("[warn] no numeric summary rows to plot; skipping --plot-all")
        return False

    plot_frame = _prepare_all_metrics_frame(summary)
    variant_order = list(dict.fromkeys(plot_frame["variant"].tolist()))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    variant_colors = {
        variant_name: color_cycle[index % len(color_cycle)]
        for index, variant_name in enumerate(variant_order)
    }

    figure, axes = plt.subplots(
        len(ALL_METRICS_PANELS) // 2,
        2,
        figsize=(12.5, 14.0),
        layout="constrained",
    )
    plotted_any = False
    for panel, axis in zip(ALL_METRICS_PANELS, axes.ravel()):
        if _draw_all_metrics_panel(axis, panel, plot_frame, variant_colors):
            plotted_any = True
    if not plotted_any:
        print("[warn] no numeric summary rows to plot; skipping --plot-all")
        plt.close(figure)
        return False

    legend_handles: list = []
    legend_labels: list[str] = []
    for axis in axes.ravel():
        handles, labels = axis.get_legend_handles_labels()
        if len(labels) > len(legend_labels):
            legend_handles, legend_labels = handles, labels
    if legend_labels:
        figure.legend(
            legend_handles,
            legend_labels,
            loc="outside upper center",
            ncol=min(len(legend_labels), 4),
            fontsize="small",
        )

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    except OSError as error:
        print(f"[warn] could not write plot {plot_path}: {error}")
        plt.close(figure)
        return False
    plt.close(figure)
    print(f"wrote all-metrics plot: {plot_path}")
    return True


def _select_b1_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the summary rows with ``B == 1``, the ``--plot-parts`` scope.

    The parts figures describe the primary production-large-``N`` case
    (``B=1``, SPEC 0062 §"Synthetic data recipe"). A missing batch size
    counts as "not ``B=1``" so an unexpected ``NaN`` group is never plotted.

    Args:
        summary: The summary frame from :func:`summarize_grid`.

    Returns:
        The rows whose batch size is exactly 1; an empty frame when ``B`` is
        absent or no row qualifies.
    """
    if summary.empty or "B" not in summary.columns:
        return summary.iloc[0:0]
    return summary[summary["B"].fillna(0) == 1]


def _apply_plain_integer_n_ticks(
    axis: "plt.Axes", sequence_lengths: pd.Series
) -> None:
    """Force the ``N`` axis to log2 spacing with plain integer tick labels.

    A log2 x-axis would otherwise be labelled with powers like ``2^9``; the
    parts figures must show the actual sequence lengths (512, 1024, …) as
    plain integers. Major ticks are pinned to the sorted unique ``N`` values
    of the data (explicit ticks) and formatted with a
    :class:`~matplotlib.ticker.FuncFormatter`; minor tick labels are removed.

    Args:
        axis: The matplotlib axes whose x-axis is configured.
        sequence_lengths: The ``N`` values plotted on this axis.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    unique_lengths = sorted(
        {int(length) for length in pd.to_numeric(sequence_lengths).dropna()}
    )
    axis.set_xscale("log", base=2)
    axis.xaxis.set_major_locator(FixedLocator(unique_lengths))
    axis.xaxis.set_major_formatter(
        FuncFormatter(
            lambda tick_value, _tick_position: f"{int(tick_value):d}"
        )
    )
    axis.xaxis.set_minor_formatter(NullFormatter())


def _stacked_fraction_frame(
    variant_rows: pd.DataFrame,
) -> pd.DataFrame | None:
    """Return per-``N`` fractions of ``total_ms_mean`` for one variant.

    The four stacked components follow the F6 contract: attention
    (``encoder_attn_ms_mean + decoder_attn_ms_mean``), normalisation
    (``normalize_ms_mean``), linker (``linker_ms_mean``), and the "other"
    residual ``total_ms_mean - attention - normalize - linker`` (embedding,
    model-level ``dist_2d``, MLP/norm blocks, head/logits, einsum, …). Every
    component is divided by ``total_ms_mean`` so the fractions sum to 1. Rows
    with a missing/NaN total or component are dropped; a stacked area needs a
    range, so fewer than two usable ``N`` points yields ``None``.

    Args:
        variant_rows: ``B=1`` summary rows of one variant (any order).

    Returns:
        A frame with columns ``N`` and the fraction columns ``attention``,
        ``normalize``, ``linker``, ``other``, sorted by ``N``; or ``None``
        when the variant has fewer than two usable rows or is missing a
        required column.
    """
    required_columns = (
        "N",
        "total_ms_mean",
        "encoder_attn_ms_mean",
        "decoder_attn_ms_mean",
        "normalize_ms_mean",
        "linker_ms_mean",
    )
    if not set(required_columns).issubset(variant_rows.columns):
        return None

    ordered_rows = variant_rows.sort_values("N")
    sequence_lengths = pd.to_numeric(
        ordered_rows["N"], errors="coerce"
    ).to_numpy(dtype=float)
    totals = pd.to_numeric(
        ordered_rows["total_ms_mean"], errors="coerce"
    ).to_numpy(dtype=float)
    attention = (
        pd.to_numeric(
            ordered_rows["encoder_attn_ms_mean"], errors="coerce"
        ).to_numpy(dtype=float)
        + pd.to_numeric(
            ordered_rows["decoder_attn_ms_mean"], errors="coerce"
        ).to_numpy(dtype=float)
    )
    normalize = pd.to_numeric(
        ordered_rows["normalize_ms_mean"], errors="coerce"
    ).to_numpy(dtype=float)
    linker = pd.to_numeric(
        ordered_rows["linker_ms_mean"], errors="coerce"
    ).to_numpy(dtype=float)

    usable = (
        np.isfinite(sequence_lengths)
        & np.isfinite(totals)
        & (totals > 0.0)
        & np.isfinite(attention)
        & np.isfinite(normalize)
        & np.isfinite(linker)
    )
    if int(usable.sum()) < 2:
        return None

    usable_totals = totals[usable]
    return pd.DataFrame(
        {
            "N": sequence_lengths[usable],
            "attention": attention[usable] / usable_totals,
            "normalize": normalize[usable] / usable_totals,
            "linker": linker[usable] / usable_totals,
            "other": (
                usable_totals - attention[usable] - normalize[usable]
                - linker[usable]
            )
            / usable_totals,
        }
    )


def _build_time_fractions_figure(
    summary: pd.DataFrame,
) -> "plt.Figure | None":
    """Build the 2×2 100%-stacked time-fraction figure (caller saves/closes).

    One panel per canonical SPEC 0062 variant in
    :data:`PARTS_VARIANT_ORDER`, each a :func:`matplotlib.axes.Axes.stackplot`
    over ``N`` of the attention, normalisation, linker, and other fractions of
    ``total_ms_mean`` (``B=1`` rows only, see :func:`_stacked_fraction_frame`).
    Panels without usable data are annotated "no data"; variants outside the
    canonical four are skipped with a warning rather than silently dropped.

    Args:
        summary: The summary frame from :func:`summarize_grid`.

    Returns:
        The matplotlib figure, or ``None`` when no variant has plottable rows.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b1_rows = _select_b1_rows(summary)
    extra_variants = sorted(set(b1_rows["variant"]) - set(PARTS_VARIANT_ORDER))
    if extra_variants:
        print(
            "[warn] --plot-parts time-fraction figure covers only "
            f"{list(PARTS_VARIANT_ORDER)}; skipping variants {extra_variants}"
        )

    figure, axes = plt.subplots(
        2, 2, figsize=(12.0, 8.0), layout="constrained"
    )
    legend_handles: list = []
    legend_labels: list[str] = []
    plotted_any = False
    for panel_index, variant_name in enumerate(PARTS_VARIANT_ORDER):
        axis = axes.ravel()[panel_index]
        variant_rows = b1_rows[b1_rows["variant"] == variant_name]
        fraction_frame = _stacked_fraction_frame(variant_rows)
        axis.set_title(str(variant_name), fontsize="medium")
        axis.set_ylabel("fraction of total_ms")
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, alpha=0.3)
        if panel_index >= 2:
            axis.set_xlabel("N")
        if fraction_frame is None:
            _annotate_empty_panel(axis)
            continue
        _apply_plain_integer_n_ticks(axis, fraction_frame["N"])
        stacked_areas = axis.stackplot(
            fraction_frame["N"],
            fraction_frame["attention"],
            fraction_frame["normalize"],
            fraction_frame["linker"],
            fraction_frame["other"],
            labels=[label for label, _color in TIME_FRACTION_STYLES],
            colors=[color for _label, color in TIME_FRACTION_STYLES],
            alpha=0.85,
        )
        plotted_any = True
        if not legend_handles:
            legend_handles = list(stacked_areas)
            legend_labels = [label for label, _color in TIME_FRACTION_STYLES]
    if not plotted_any:
        plt.close(figure)
        return None
    if legend_handles:
        figure.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="outside upper center",
            ncol=len(legend_labels),
            fontsize="small",
        )
    return figure


def _draw_variant_lines(
    axis: "plt.Axes", b1_rows: pd.DataFrame, value_column: str
) -> bool:
    """Draw one solid measured Okabe-Ito line per canonical variant.

    The line style mirrors ``panel_speed_vs_n_h100``/``panel_memory_vs_n``
    in ``curated/generate_report.py``: solid lines with ``o`` markers
    (markersize 2.2, linewidth 0.8) in the :data:`PARTS_VARIANT_COLORS`
    colour of each :data:`PARTS_VARIANT_ORDER` variant. Log axes cannot
    show missing or non-positive values, so such rows are skipped per
    variant; there are no projections because every plotted point is a
    measured summary value. Legend labels follow the draw order, which is
    the canonical variant order shared with the time-fraction figure.

    Args:
        axis: The matplotlib axes to draw on.
        b1_rows: The ``B=1`` summary rows from :func:`_select_b1_rows`.
        value_column: Summary column to plot (``full_forward_ms_mean`` or
            ``peak_mem_mb_max``).

    Returns:
        ``True`` when at least one variant line was drawn.
    """
    drew_line = False
    for variant_name in PARTS_VARIANT_ORDER:
        variant_rows = b1_rows[b1_rows["variant"] == variant_name].sort_values(
            "N"
        )
        variant_values = pd.to_numeric(
            variant_rows[value_column], errors="coerce"
        )
        plottable = variant_values.notna() & (variant_values > 0)
        if not plottable.any():
            continue
        kept_rows = variant_rows.loc[plottable]
        axis.plot(
            kept_rows["N"],
            variant_values.loc[plottable],
            color=PARTS_VARIANT_COLORS[str(variant_name)],
            marker="o",
            markersize=2.2,
            linewidth=0.8,
            linestyle="-",
            zorder=3,
            label=str(variant_name),
        )
        drew_line = True
    return drew_line


def _style_parts_log_axis(
    axis: "plt.Axes",
    *,
    ylabel: str,
    title: str,
    yticks: tuple[float, ...],
) -> None:
    """Apply the attention-only H100 panel y-axis style to a parts figure.

    Mirrors ``curated/generate_report.py::_style_h100_log_axis`` with
    ``y_log_math=True`` (the setting of both H100 body panels): a
    logarithmic y-axis whose explicit major ticks are labelled as base-10
    powers by :class:`~matplotlib.ticker.LogFormatterMathtext`, with minor
    tick labels removed. The x-axis is configured separately by
    :func:`_apply_plain_integer_n_ticks`.

    Args:
        axis: The matplotlib axes to style.
        ylabel: Y-axis label (``Forward time (ms)`` or ``GPU memory (MB)``).
        title: Panel title.
        yticks: Explicit major tick locations on the logarithmic y-axis.
    """
    from matplotlib.ticker import LogFormatterMathtext, NullFormatter

    axis.set_yscale("log")
    axis.set_yticks(list(yticks))
    axis.get_yaxis().set_major_formatter(LogFormatterMathtext(base=10))
    axis.get_yaxis().set_minor_formatter(NullFormatter())
    axis.grid(True, alpha=0.3)
    axis.set_xlabel("Sequence length $N$")
    axis.set_ylabel(ylabel)
    axis.set_title(title)


def _build_parts_variant_figure(
    summary: pd.DataFrame,
    *,
    value_column: str,
    ylabel: str,
    title: str,
    yticks: tuple[float, ...],
    vram_line_mb: float | None = None,
) -> "plt.Figure | None":
    """Build one H100-styled single-panel parts figure (caller saves/closes).

    Shared implementation of :func:`_build_forward_time_figure` and
    :func:`_build_memory_figure`. The figure is a single log-log panel of
    :data:`PARTS_LINE_FIGSIZE` with one solid measured line per canonical
    variant (see :func:`_draw_variant_lines`), plain integer ``N`` tick
    labels, the shared H100 y-axis style (see
    :func:`_style_parts_log_axis`), and a frameless method legend in the
    upper left. Variants outside the canonical four are skipped with a
    warning, matching the time-fraction figure.

    Args:
        summary: The summary frame from :func:`summarize_grid`.
        value_column: Summary column plotted per variant.
        ylabel: Y-axis label.
        title: Panel title.
        yticks: Explicit major tick locations on the logarithmic y-axis.
        vram_line_mb: When given, draw the grey dashed H100 VRAM reference
            line at this memory value with a small text label (the label
            position matches ``panel_memory_vs_n`` on the standard
            512–32768 ``N`` grid) and widen the top of the view so the line
            and label stay visible when the measured memory stays below it
            (``axhline`` never participates in autoscaling).

    Returns:
        The matplotlib figure, or ``None`` when the ``B=1`` rows are empty,
        the value column is absent, or no variant has a plottable value.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b1_rows = _select_b1_rows(summary)
    if b1_rows.empty or value_column not in b1_rows.columns:
        return None
    extra_variants = sorted(set(b1_rows["variant"]) - set(PARTS_VARIANT_ORDER))
    if extra_variants:
        print(
            "[warn] --plot-parts forward-time/memory figures cover only "
            f"{list(PARTS_VARIANT_ORDER)}; skipping variants {extra_variants}"
        )

    figure, axis = plt.subplots(figsize=PARTS_LINE_FIGSIZE)
    if not _draw_variant_lines(axis, b1_rows, value_column):
        plt.close(figure)
        return None

    _apply_plain_integer_n_ticks(axis, b1_rows["N"])
    _style_parts_log_axis(axis, ylabel=ylabel, title=title, yticks=yticks)
    axis.autoscale_view()
    if vram_line_mb is not None:
        axis.axhline(
            vram_line_mb,
            color="#999999",
            linewidth=1.0,
            linestyle="--",
            zorder=1,
        )
        axis.text(
            9000,
            105000,
            "94 GB H100 VRAM",
            fontsize=5.5,
            color="#666666",
            ha="left",
            va="bottom",
        )
        # Measured memory on this grid stays below the VRAM line, so the
        # autoscaled view would hide the reference line; widen the top of
        # the view to include it, with headroom above the line for the
        # small text label.
        _view_bottom, view_top = axis.get_ylim()
        axis.set_ylim(top=max(view_top, vram_line_mb * 2.0))
    axis.legend(loc="upper left", frameon=False, fontsize="small")
    figure.tight_layout()
    return figure


def _build_forward_time_figure(
    summary: pd.DataFrame,
) -> "plt.Figure | None":
    """Build the H100-styled forward-time figure (caller saves/closes).

    Single log-log panel of ``full_forward_ms_mean`` per canonical variant
    (``B=1`` rows only), styled after
    ``curated/generate_report.py::panel_speed_vs_n_h100``; see
    :func:`_build_parts_variant_figure` for the shared styling.

    Args:
        summary: The summary frame from :func:`summarize_grid`.

    Returns:
        The matplotlib figure, or ``None`` when no variant has a plottable
        forward-time value.
    """
    return _build_parts_variant_figure(
        summary,
        value_column="full_forward_ms_mean",
        ylabel="Forward time (ms)",
        title="Forward time vs sequence length (B=1)",
        yticks=(0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    )


def _build_memory_figure(
    summary: pd.DataFrame,
) -> "plt.Figure | None":
    """Build the H100-styled GPU-memory figure (caller saves/closes).

    Single log-log panel of ``peak_mem_mb_max`` per canonical variant
    (``B=1`` rows only) with the grey dashed 94000 MB H100 VRAM reference
    line of the attention-only memory panel, styled after
    ``curated/generate_report.py::panel_memory_vs_n``; see
    :func:`_build_parts_variant_figure` for the shared styling.

    Args:
        summary: The summary frame from :func:`summarize_grid`.

    Returns:
        The matplotlib figure, or ``None`` when no variant has a plottable
        peak-memory value.
    """
    return _build_parts_variant_figure(
        summary,
        value_column="peak_mem_mb_max",
        ylabel="GPU memory (MB)",
        title="GPU memory vs sequence length (B=1)",
        yticks=(100.0, 1000.0, 10000.0, 100000.0, 1000000.0),
        vram_line_mb=PARTS_MEMORY_VRAM_MB,
    )


def _save_parts_figure(
    figure: "plt.Figure", plot_path: Path, description: str
) -> bool:
    """Save and close one ``--plot-parts`` figure.

    Args:
        figure: The built figure; ownership passes to this function.
        plot_path: Destination PNG path.
        description: Human-readable figure name for the log line.

    Returns:
        ``True`` when the file was written, ``False`` on an ``OSError``.
    """
    import matplotlib.pyplot as plt

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    except OSError as error:
        print(f"[warn] could not write plot {plot_path}: {error}")
        plt.close(figure)
        return False
    plt.close(figure)
    print(f"wrote {description}: {plot_path}")
    return True


def write_time_fractions_plot(summary: pd.DataFrame, plot_path: Path) -> bool:
    """Write the 2×2 time-fraction PNG when matplotlib exists.

    Args:
        summary: The summary frame from :func:`summarize_grid`.
        plot_path: Destination PNG path.

    Returns:
        ``True`` when a figure was written, ``False`` when matplotlib is
        unavailable or no ``B=1`` variant has plottable rows.
    """
    try:
        figure = _build_time_fractions_figure(summary)
    except ImportError:
        print("[warn] matplotlib is not available; skipping --plot-parts")
        return False
    if figure is None:
        print(
            "[warn] no B=1 time-fraction rows to plot; "
            "skipping --plot-parts time-fractions figure"
        )
        return False
    return _save_parts_figure(figure, plot_path, "time-fractions plot")


def write_forward_time_plot(summary: pd.DataFrame, plot_path: Path) -> bool:
    """Write the H100-styled forward-time PNG when matplotlib exists.

    Args:
        summary: The summary frame from :func:`summarize_grid`.
        plot_path: Destination PNG path.

    Returns:
        ``True`` when a figure was written, ``False`` when matplotlib is
        unavailable or no ``B=1`` variant has a plottable forward-time value.
    """
    try:
        figure = _build_forward_time_figure(summary)
    except ImportError:
        print("[warn] matplotlib is not available; skipping --plot-parts")
        return False
    if figure is None:
        print(
            "[warn] no B=1 forward-time rows to plot; "
            "skipping --plot-parts forward-time figure"
        )
        return False
    return _save_parts_figure(figure, plot_path, "forward-time plot")


def write_memory_plot(summary: pd.DataFrame, plot_path: Path) -> bool:
    """Write the H100-styled GPU-memory PNG when matplotlib exists.

    Args:
        summary: The summary frame from :func:`summarize_grid`.
        plot_path: Destination PNG path.

    Returns:
        ``True`` when a figure was written, ``False`` when matplotlib is
        unavailable or no ``B=1`` variant has a plottable peak-memory value.
    """
    try:
        figure = _build_memory_figure(summary)
    except ImportError:
        print("[warn] matplotlib is not available; skipping --plot-parts")
        return False
    if figure is None:
        print(
            "[warn] no B=1 memory rows to plot; "
            "skipping --plot-parts memory figure"
        )
        return False
    return _save_parts_figure(figure, plot_path, "memory plot")


def write_parts_plots(summary: pd.DataFrame, prefix: Path) -> bool:
    """Write the three ``--plot-parts`` figures derived from ``prefix``.

    Writes ``<prefix>_time_fractions.png``, ``<prefix>_forward_time.png``,
    and ``<prefix>_memory.png``. Each figure is skipped with a warning when
    matplotlib is unavailable or has no ``B=1`` rows to draw; the two
    existing ``--plot``/``--plot-all`` figures are untouched.

    Args:
        summary: The summary frame from :func:`summarize_grid`.
        prefix: Path prefix for the three PNG file names.

    Returns:
        ``True`` only when all three PNG files were written.
    """
    time_fractions_written = write_time_fractions_plot(
        summary, Path(f"{prefix}_time_fractions.png")
    )
    forward_time_written = write_forward_time_plot(
        summary, Path(f"{prefix}_forward_time.png")
    )
    memory_written = write_memory_plot(summary, Path(f"{prefix}_memory.png"))
    return time_fractions_written and forward_time_written and memory_written


def print_environment_provenance(environment: dict[str, object]) -> None:
    """Print the provenance fields that matter when reading the numbers.

    Args:
        environment: Parsed environment JSON, possibly empty.
    """
    if not environment:
        return
    fields = (
        "device",
        "dtype",
        "gpu_name",
        "torch_version",
        "cuda_version",
        "checkpoint_sha256",
        "git_commit",
        "repeats",
        "seed",
        "linker_mode",
    )
    rendered = ", ".join(
        f"{field}={environment.get(field)}" for field in fields
    )
    print(f"environment: {rendered}")


# ─── Entry point ────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Load the raw CSV, write the summary CSV/PNG(s), and print the N table.

    Args:
        argv: Optional argument list for tests; defaults to ``sys.argv``.

    Returns:
        Process exit code (always 0 on success).
    """
    arguments = parse_arguments(argv)
    raw_frame = load_raw_csv(arguments.raw)
    environment = load_environment_json(arguments.env_json)
    print_environment_provenance(environment)

    summary = summarize_grid(raw_frame)
    write_summary_csv(summary, arguments.out)
    print(f"wrote {len(summary)} summary rows to {arguments.out}")
    print(format_compact_table(summary, TABLE_SEQUENCE_LENGTH))

    if arguments.plot is not None:
        write_summary_plot(summary, arguments.plot)
    if arguments.plot_all is not None:
        write_all_metrics_plot(summary, arguments.plot_all)
    if arguments.plot_parts is not None:
        write_parts_plots(summary, arguments.plot_parts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
