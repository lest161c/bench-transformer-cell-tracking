"""Tests for the SPEC 0062 full-model inference grid summary script.

The synthetic raw CSVs contain a small, hand-checkable grid: two variants, two
repeats, and an explicit ``oom`` row. The assertions cover counts, sample
standard deviation (including the single-repeat ``std=0`` case), the derived
speedups and attention shares, the ``n/a`` handling when the baseline or a
variant did not complete, the total-peak-VRAM mean/std/max columns of
SPEC 0062 §12, the optional ``--plot-all`` PNG output, the optional
``--plot-parts`` three-figure decomposition output (including the
``--plot-batch 64`` selection of B>1 rows whose linker timing is skipped),
and the optional ``--plot-ratios`` component-ratio PNG output.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

from analysis import summarize_full_model_inference_grid as summary_module  # noqa: E402

RAW_HEADER = [
    "variant",
    "N",
    "B",
    "repeat_idx",
    "status",
    "linker_mode",
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
    "peak_mem_mb",
    "baseline_mem_mb",
    "peak_mem_total_mb",
    "peak_mem_reserved_mb",
    "forward_peak_mem_total_mb",
    "forward_peak_mem_reserved_mb",
    "normalize_peak_mem_total_mb",
    "normalize_peak_mem_reserved_mb",
]

DEFAULT_METRIC_VALUES = {
    "embed_ms": 1.0,
    "encoder_ms": 2.0,
    "decoder_ms": 3.0,
    "head_logits_ms": 4.0,
    "einsum_ms": 5.0,
    "normalize_ms": 6.0,
    "linker_build_graph_ms": 7.0,
    "linker_greedy_ms": 8.0,
    "linker_ms": 15.0,
}

#: Deterministic total-peak-VRAM values (SPEC 0062 §12) merged into every
#: ``_ok_row`` so the memory mean/std/max aggregation is exercised.
DEFAULT_MEMORY_METRIC_VALUES = {
    "baseline_mem_mb": 100.0,
    "peak_mem_total_mb": 400.0,
    "peak_mem_reserved_mb": 500.0,
    "forward_peak_mem_total_mb": 350.0,
    "forward_peak_mem_reserved_mb": 450.0,
    "normalize_peak_mem_total_mb": 400.0,
    "normalize_peak_mem_reserved_mb": 500.0,
}


def _write_raw_csv(raw_path: Path, rows: list[dict[str, object]]) -> None:
    """Write synthetic raw rows with the full summary-relevant header.

    Args:
        raw_path: Destination CSV path.
        rows: Row dictionaries; missing cells are written as empty strings.
    """
    with open(raw_path, "w", newline="") as raw_file:
        writer = csv.DictWriter(raw_file, fieldnames=RAW_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RAW_HEADER})


def _ok_row(
    variant: str,
    sequence_length: int,
    repeat_idx: int,
    full_forward_ms: float,
    total_ms: float,
    encoder_attn_ms: float,
    decoder_attn_ms: float,
    peak_mem_mb: float,
    batch_size: int = 1,
    memory_columns: dict[str, float] | None = None,
) -> dict[str, object]:
    """Build one successful raw row with deterministic filler metrics.

    Args:
        variant: Variant key.
        sequence_length: ``N`` for the row.
        repeat_idx: Repeat index.
        full_forward_ms: Full forward timing.
        total_ms: End-to-end total timing.
        encoder_attn_ms: Encoder attention timing.
        decoder_attn_ms: Decoder attention timing.
        peak_mem_mb: Incremental peak memory.
        batch_size: ``B`` for the row (1 by default, 64 for batch runs).
        memory_columns: Overrides for the total-peak-VRAM columns
            (SPEC 0062 §12); ``None`` uses :data:`DEFAULT_MEMORY_METRIC_VALUES`.

    Returns:
        A dictionary covering every raw header column used by the summary.
    """
    row: dict[str, object] = {
        "variant": variant,
        "N": sequence_length,
        "B": batch_size,
        "repeat_idx": repeat_idx,
        "status": "ok",
        "linker_mode": "greedy",
        "full_forward_ms": full_forward_ms,
        "total_ms": total_ms,
        "encoder_attn_ms": encoder_attn_ms,
        "decoder_attn_ms": decoder_attn_ms,
        "peak_mem_mb": peak_mem_mb,
    }
    row.update(DEFAULT_METRIC_VALUES)
    row.update(
        DEFAULT_MEMORY_METRIC_VALUES if memory_columns is None
        else memory_columns
    )
    return row


def _oom_row(
    variant: str, sequence_length: int, repeat_idx: int
) -> dict[str, object]:
    """Build one OOM raw row with empty numeric cells.

    Args:
        variant: Variant key.
        sequence_length: ``N`` for the row.
        repeat_idx: Repeat index.

    Returns:
        A row with ``status="oom"`` and no timing values.
    """
    return {
        "variant": variant,
        "N": sequence_length,
        "B": 1,
        "repeat_idx": repeat_idx,
        "status": "oom",
        "linker_mode": "greedy",
    }


def _row_for(summary: pd.DataFrame, variant: str, sequence_length: int):
    """Select the single summary row for a variant and ``N``.

    Args:
        summary: Summary frame.
        variant: Variant key.
        sequence_length: ``N`` to select.

    Returns:
        The matching row as a pandas Series.
    """
    selected = summary[
        (summary["variant"] == variant)
        & (summary["N"] == sequence_length)
    ]
    assert len(selected) == 1
    return selected.iloc[0]


def test_summary_counts_means_std_and_derived_metrics(tmp_path):
    """Means, sample std, speedups, and shares match hand calculations."""
    raw_path = tmp_path / "raw.csv"
    summary_path = tmp_path / "summary.csv"

    rows = [
        _ok_row("relative_pos", 128, 0, 80.0, 100.0, 20.0, 10.0, 100.0),
        _ok_row(
            "relative_pos", 128, 1, 100.0, 120.0, 24.0, 12.0, 200.0,
            memory_columns={
                **DEFAULT_MEMORY_METRIC_VALUES,
                "peak_mem_reserved_mb": 700.0,
            },
        ),
        _ok_row("sparse_gather_k16", 128, 0, 50.0, 60.0, 8.0, 4.0, 300.0),
        _oom_row("sparse_gather_k16", 128, 1),
    ]
    _write_raw_csv(raw_path, rows)

    exit_code = summary_module.main(
        [
            "--raw",
            str(raw_path),
            "--env-json",
            str(tmp_path / "missing_env.json"),
            "--out",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    summary = pd.read_csv(summary_path)

    relative_row = _row_for(summary, "relative_pos", 128)
    gather_row = _row_for(summary, "sparse_gather_k16", 128)

    assert relative_row["n_ok"] == 2
    assert relative_row["n_oom"] == 0
    assert relative_row["n_err"] == 0
    assert gather_row["n_ok"] == 1
    assert gather_row["n_oom"] == 1
    assert gather_row["n_err"] == 0

    assert relative_row["total_ms_mean"] == pytest.approx(110.0)
    assert relative_row["total_ms_std"] == pytest.approx(math.sqrt(200.0))
    assert relative_row["full_forward_ms_mean"] == pytest.approx(90.0)
    assert relative_row["full_forward_ms_std"] == pytest.approx(
        math.sqrt(200.0)
    )
    assert gather_row["total_ms_mean"] == pytest.approx(60.0)
    assert gather_row["total_ms_std"] == pytest.approx(0.0)
    assert gather_row["full_forward_ms_std"] == pytest.approx(0.0)

    assert relative_row["speedup_vs_relative_pos"] == pytest.approx(1.0)
    assert gather_row["speedup_vs_relative_pos"] == pytest.approx(
        110.0 / 60.0
    )
    assert gather_row["forward_speedup_vs_relative_pos"] == pytest.approx(
        90.0 / 50.0
    )
    assert gather_row["attention_share"] == pytest.approx(12.0 / 60.0)
    assert gather_row["attention_share_of_forward"] == pytest.approx(12.0 / 50.0)
    assert relative_row["attention_share"] == pytest.approx(33.0 / 110.0)

    assert relative_row["peak_mem_mb_max"] == pytest.approx(200.0)
    assert gather_row["peak_mem_mb_max"] == pytest.approx(300.0)

    assert relative_row["baseline_mem_mb_mean"] == pytest.approx(100.0)
    assert relative_row["peak_mem_total_mb_max"] == pytest.approx(400.0)
    assert relative_row["peak_mem_reserved_mb_mean"] == pytest.approx(600.0)
    assert relative_row["peak_mem_reserved_mb_std"] == pytest.approx(
        math.sqrt(20000.0)
    )
    assert relative_row["peak_mem_reserved_mb_max"] == pytest.approx(700.0)
    assert gather_row["peak_mem_reserved_mb_max"] == pytest.approx(500.0)
    assert gather_row["peak_mem_reserved_mb_std"] == pytest.approx(0.0)
    assert gather_row["forward_peak_mem_total_mb_max"] == pytest.approx(350.0)
    assert gather_row["normalize_peak_mem_total_mb_max"] == pytest.approx(
        400.0
    )


def test_summary_includes_total_peak_vram_mean_std_and_max_columns(tmp_path):
    """The total-VRAM columns are aggregated with mean/std and the maxima.

    SPEC 0062 §12: the seven new raw memory columns receive mean/std, and
    exactly four of them additionally receive a ``*_max`` column over the
    ``ok`` repeats, while ``peak_mem_mb`` keeps its delta-only
    ``peak_mem_mb_max`` aggregation.
    """
    expected_memory_mean_std_columns = (
        "baseline_mem_mb",
        "peak_mem_total_mb",
        "peak_mem_reserved_mb",
        "forward_peak_mem_total_mb",
        "forward_peak_mem_reserved_mb",
        "normalize_peak_mem_total_mb",
        "normalize_peak_mem_reserved_mb",
    )
    expected_peak_max_columns = (
        "peak_mem_total_mb_max",
        "peak_mem_reserved_mb_max",
        "forward_peak_mem_total_mb_max",
        "normalize_peak_mem_total_mb_max",
    )

    raw_path = tmp_path / "raw.csv"
    summary_path = tmp_path / "summary.csv"
    rows = [
        _ok_row("relative_pos", 256, 0, 80.0, 100.0, 20.0, 10.0, 100.0),
        _ok_row("relative_pos", 256, 1, 90.0, 110.0, 22.0, 11.0, 150.0),
        _oom_row("cached_dist", 256, 0),
    ]
    _write_raw_csv(raw_path, rows)

    exit_code = summary_module.main(
        [
            "--raw",
            str(raw_path),
            "--env-json",
            str(tmp_path / "missing_env.json"),
            "--out",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    summary = pd.read_csv(summary_path)
    summary_columns = list(summary.columns)
    for column in expected_memory_mean_std_columns:
        assert f"{column}_mean" in summary_columns
        assert f"{column}_std" in summary_columns
    for column in expected_peak_max_columns:
        assert column in summary_columns
    assert "peak_mem_mb_max" in summary_columns

    relative_row = _row_for(summary, "relative_pos", 256)
    cached_row = _row_for(summary, "cached_dist", 256)
    for column in expected_peak_max_columns:
        assert math.isnan(cached_row[column])
    assert relative_row["peak_mem_total_mb_max"] == pytest.approx(400.0)
    assert relative_row["peak_mem_reserved_mb_max"] == pytest.approx(500.0)
    assert relative_row["forward_peak_mem_total_mb_max"] == pytest.approx(
        350.0
    )
    assert relative_row["normalize_peak_mem_total_mb_max"] == pytest.approx(
        400.0
    )
    assert relative_row["forward_peak_mem_reserved_mb_mean"] == pytest.approx(
        450.0
    )
    assert relative_row["normalize_peak_mem_reserved_mb_std"] == pytest.approx(
        0.0
    )


def test_compact_table_marks_oom_and_missing_baseline_as_na(tmp_path, capsys):
    """The N table prints n/a when the baseline or the variant is missing."""
    raw_path = tmp_path / "raw.csv"
    summary_path = tmp_path / "summary.csv"

    rows = [
        _ok_row("relative_pos", 32768, 0, 80.0, 100.0, 20.0, 10.0, 100.0),
        _oom_row("sparse_gather_k16", 32768, 0),
        _ok_row("cached_dist", 16384, 0, 70.0, 90.0, 15.0, 9.0, 50.0),
    ]
    _write_raw_csv(raw_path, rows)

    exit_code = summary_module.main(
        [
            "--raw",
            str(raw_path),
            "--env-json",
            str(tmp_path / "missing_env.json"),
            "--out",
            str(summary_path),
        ]
    )
    printed = capsys.readouterr().out

    assert exit_code == 0
    assert "N=32768" in printed
    assert "sparse_gather_k16" in printed
    assert "n/a" in printed

    summary = pd.read_csv(summary_path)
    assert math.isnan(_row_for(summary, "sparse_gather_k16", 32768)[
        "speedup_vs_relative_pos"
    ])
    # No relative_pos at N=16384, so the relaxed baseline lookup has no value.
    assert math.isnan(
        _row_for(summary, "cached_dist", 16384)["speedup_vs_relative_pos"]
    )
    assert math.isnan(
        _row_for(summary, "cached_dist", 16384)[
            "forward_speedup_vs_relative_pos"
        ]
    )


def test_empty_raw_csv_writes_header_only_summary(tmp_path):
    """An empty raw file yields an empty, schema-correct summary CSV."""
    raw_path = tmp_path / "empty.csv"
    raw_path.write_text("")
    summary_path = tmp_path / "summary.csv"

    exit_code = summary_module.main(
        [
            "--raw",
            str(raw_path),
            "--env-json",
            str(tmp_path / "missing_env.json"),
            "--out",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    summary = pd.read_csv(summary_path)
    assert list(summary.columns) == summary_module.build_summary_columns()
    assert summary.empty


def test_plot_all_writes_multi_panel_png(tmp_path):
    """``--plot-all`` writes a non-trivial PNG when matplotlib is available.

    The synthetic grid covers both speedup panels (a ``relative_pos``
    baseline at two ``N`` values), the timing panels, and one variant whose
    only row is ``oom`` so the NaN/missing-value path is exercised too.
    """
    pytest.importorskip("matplotlib")
    raw_path = tmp_path / "raw.csv"
    plot_path = tmp_path / "all_metrics.png"

    rows = [
        _ok_row("relative_pos", 128, 0, 80.0, 100.0, 20.0, 10.0, 100.0),
        _ok_row("relative_pos", 256, 0, 160.0, 200.0, 40.0, 20.0, 200.0),
        _ok_row("sparse_gather_k16", 128, 0, 50.0, 60.0, 8.0, 4.0, 300.0),
        _ok_row("sparse_gather_k16", 256, 0, 90.0, 110.0, 12.0, 6.0, 400.0),
        _oom_row("cached_dist", 128, 0),
    ]
    _write_raw_csv(raw_path, rows)

    exit_code = summary_module.main(
        [
            "--raw",
            str(raw_path),
            "--env-json",
            str(tmp_path / "missing_env.json"),
            "--out",
            str(tmp_path / "summary.csv"),
            "--plot-all",
            str(plot_path),
        ]
    )

    assert exit_code == 0
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_plot_parts_writes_three_pngs(tmp_path):
    """``--plot-parts`` writes the time-fraction, forward-time, and memory PNGs.

    The synthetic grid gives every canonical variant two ``N`` values so the
    stacked areas have a range and both log-log panels have lines; one
    ``N=2048`` row with an empty ``normalize_ms`` cell exercises the NaN-drop
    path, and the flag writes all three files through the full ``main()``
    CLI path. The removed twin-axis ``<prefix>_forward_time_memory.png``
    must no longer be written.
    """
    pytest.importorskip("matplotlib")
    raw_path = tmp_path / "raw.csv"
    parts_prefix = tmp_path / "parts"

    rows = []
    for sequence_length, scale in ((512, 1.0), (1024, 2.0)):
        rows.append(
            _ok_row(
                "relative_pos", sequence_length, 0,
                80.0 * scale, 100.0 * scale, 20.0 * scale, 10.0 * scale,
                100.0 * scale,
            )
        )
        rows.append(
            _ok_row(
                "cached_dist", sequence_length, 0,
                70.0 * scale, 90.0 * scale, 15.0 * scale, 9.0 * scale,
                150.0 * scale,
            )
        )
        rows.append(
            _ok_row(
                "sparse_mask_k16", sequence_length, 0,
                60.0 * scale, 80.0 * scale, 12.0 * scale, 8.0 * scale,
                200.0 * scale,
            )
        )
        rows.append(
            _ok_row(
                "sparse_gather_k16", sequence_length, 0,
                50.0 * scale, 70.0 * scale, 10.0 * scale, 6.0 * scale,
                250.0 * scale,
            )
        )
    incomplete_row = _ok_row(
        "sparse_mask_k16", 2048, 0, 120.0, 160.0, 24.0, 16.0, 300.0
    )
    incomplete_row["normalize_ms"] = ""
    rows.append(incomplete_row)
    _write_raw_csv(raw_path, rows)

    exit_code = summary_module.main(
        [
            "--raw",
            str(raw_path),
            "--env-json",
            str(tmp_path / "missing_env.json"),
            "--out",
            str(tmp_path / "summary.csv"),
            "--plot-parts",
            str(parts_prefix),
        ]
    )

    time_fractions_png = tmp_path / "parts_time_fractions.png"
    forward_time_png = tmp_path / "parts_forward_time.png"
    memory_png = tmp_path / "parts_memory.png"

    assert exit_code == 0
    assert time_fractions_png.exists()
    assert time_fractions_png.stat().st_size > 0
    assert forward_time_png.exists()
    assert forward_time_png.stat().st_size > 0
    assert memory_png.exists()
    assert memory_png.stat().st_size > 0
    assert not (tmp_path / "parts_forward_time_memory.png").exists()


def test_plot_parts_b64_plots_rows_with_skipped_linker(tmp_path):
    """``--plot-batch 64`` plots B=64 rows whose linker cells are empty.

    For ``B>1`` the benchmark skips the CPU linker, so ``linker_ms`` is
    empty and ``total_ms = full_forward_ms + normalize_ms``. Before the
    ``--plot-batch`` change the stacked-fraction builder required a finite
    linker and silently dropped every B=64 row into "no data" panels; this
    test locks in the missing-linker-as-zero rule through the full ``main()``
    CLI path.
    """
    pytest.importorskip("matplotlib")
    raw_path = tmp_path / "raw_b64.csv"
    parts_prefix = tmp_path / "parts_b64"

    rows = []
    for sequence_length, scale in ((512, 1.0), (1024, 2.0)):
        for variant_index, variant in enumerate(
            (
                "relative_pos",
                "cached_dist",
                "sparse_mask_k16",
                "sparse_gather_k16",
            )
        ):
            row = _ok_row(
                variant,
                sequence_length,
                0,
                40.0 * scale + variant_index,
                100.0 * scale + variant_index,
                20.0 * scale,
                10.0 * scale,
                100.0 * scale + variant_index,
                batch_size=64,
            )
            for linker_column in (
                "linker_build_graph_ms",
                "linker_greedy_ms",
                "linker_ms",
            ):
                row[linker_column] = ""
            rows.append(row)
    _write_raw_csv(raw_path, rows)

    exit_code = summary_module.main(
        [
            "--raw",
            str(raw_path),
            "--env-json",
            str(tmp_path / "missing_env.json"),
            "--out",
            str(tmp_path / "summary_b64.csv"),
            "--plot-batch",
            "64",
            "--plot-parts",
            str(parts_prefix),
        ]
    )

    time_fractions_png = tmp_path / "parts_b64_time_fractions.png"
    forward_time_png = tmp_path / "parts_b64_forward_time.png"
    memory_png = tmp_path / "parts_b64_memory.png"

    assert exit_code == 0
    assert time_fractions_png.exists()
    assert time_fractions_png.stat().st_size > 0
    assert forward_time_png.exists()
    assert forward_time_png.stat().st_size > 0
    assert memory_png.exists()
    assert memory_png.stat().st_size > 0


def test_plot_parts_cli_default_prefix():
    """The bare flag selects the default prefix; omitting it selects None."""
    arguments = summary_module.parse_arguments(["--plot-parts"])
    assert arguments.plot_parts == summary_module.DEFAULT_PARTS_PNG_PREFIX
    assert summary_module.parse_arguments([]).plot_parts is None
    assert summary_module.parse_arguments([]).plot_batch == 1


def test_plot_ratios_writes_component_ratios_png(tmp_path):
    """``--plot-ratios`` writes the component-ratio PNG when matplotlib exists.

    All four canonical variants at two ``N`` values with linker timings give
    the five-panel B=1 layout; one ``sparse_gather_k16`` OOM row at the
    larger ``N`` exercises the missing-ratio path (a baseline or variant
    ``oom`` must not break the figure).
    """
    pytest.importorskip("matplotlib")
    raw_path = tmp_path / "raw.csv"
    ratios_prefix = tmp_path / "ratios"

    rows = []
    for sequence_length, scale in ((512, 1.0), (1024, 2.0)):
        rows.append(
            _ok_row(
                "relative_pos", sequence_length, 0,
                80.0 * scale, 100.0 * scale, 20.0 * scale, 10.0 * scale,
                100.0 * scale,
            )
        )
        rows.append(
            _ok_row(
                "cached_dist", sequence_length, 0,
                70.0 * scale, 90.0 * scale, 15.0 * scale, 9.0 * scale,
                150.0 * scale,
            )
        )
        rows.append(
            _ok_row(
                "sparse_mask_k16", sequence_length, 0,
                60.0 * scale, 80.0 * scale, 12.0 * scale, 8.0 * scale,
                200.0 * scale,
            )
        )
    rows.append(_ok_row("sparse_gather_k16", 512, 0, 50.0, 70.0, 10.0, 6.0, 250.0))
    rows.append(_oom_row("sparse_gather_k16", 1024, 0))
    _write_raw_csv(raw_path, rows)

    exit_code = summary_module.main(
        [
            "--raw",
            str(raw_path),
            "--env-json",
            str(tmp_path / "missing_env.json"),
            "--out",
            str(tmp_path / "summary.csv"),
            "--plot-ratios",
            str(ratios_prefix),
        ]
    )

    ratios_png = tmp_path / "ratios_component_ratios.png"

    assert exit_code == 0
    assert ratios_png.exists()
    assert ratios_png.stat().st_size > 0


def test_plot_ratios_cli_default_prefix():
    """The bare --plot-ratios flag selects the default prefix; omit gives None."""
    arguments = summary_module.parse_arguments(["--plot-ratios"])
    assert arguments.plot_ratios == summary_module.DEFAULT_RATIOS_PNG_PREFIX
    assert summary_module.parse_arguments([]).plot_ratios is None
