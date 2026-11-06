#!/usr/bin/env python3
"""Plot the real-sequence aggregate inference stage shares.

Reads ``results/real_sequence_pipeline_v2_summary.csv`` (the H100 aggregate
over real vanvliet sequence prefixes) and writes a one-row stacked bar of the
stage shares of the compute time. This is the natural end-to-end view: the
linker is a small fraction, unlike the synthetic fixed-10-frame stress test.

Usage (from ``benchmark_attn/``)::

    python analysis/plot_real_sequence_pipeline.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BENCHMARK_ROOT.parent
DEFAULT_SUMMARY = BENCHMARK_ROOT / "results" / "real_sequence_pipeline_v2_summary.csv"
DEFAULT_OUT = PROJECT_ROOT / "curated" / "figures" / "real_sequence_pipeline_stage_shares"

# Stage order and Okabe-Ito-style colours. The first four match the report's
# inference palette: attention/predict blue, normalize green, linker red,
# other grey. Features and windows get distinct supporting colours.
STAGES = (
    ("predict (forward + aggregation)", "predict_ms", "#1f77b4"),
    ("regionprops features", "feature_ms", "#009E73"),
    ("build_graph", "build_graph_ms", "#d62728"),
    ("normalize", "normalize_ms", "#2ca02c"),
    ("greedy", "greedy_ms", "#CC79A7"),
    ("build_windows", "windows_ms", "#7f7f7f"),
)


def _load_summary(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Expected one summary row, got {len(frame)}")
    return frame.iloc[0]


def _plot(summary: pd.Series, out_prefix: Path) -> None:
    compute_ms = float(summary["compute_ms"])
    shares = [
        (label, float(summary[column]) / compute_ms, color)
        for label, column, color in STAGES
    ]

    fig, ax = plt.subplots(figsize=(8.2, 1.05), layout="constrained")
    left = 0.0
    for label, share, color in shares:
        ax.barh(
            [0],
            [share],
            left=left,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            height=0.55,
            label=label,
        )
        if share >= 0.035:
            ax.text(
                left + share / 2,
                0,
                f"{share * 100:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )
        left += share

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=3,
        frameon=False,
        fontsize=7.0,
        handlelength=1.2,
        columnspacing=1.0,
    )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    meta = {
        "figure": out_prefix.name,
        "sources": [str(DEFAULT_SUMMARY.resolve())],
        "values": (
            "stage shares of compute_ms for the real vanvliet aggregate at "
            "N=32768 (predict, regionprops features, build_graph, normalize, "
            "greedy, build_windows)"
        ),
        "note": (
            "timing-only H100 job 4538990; real-sequence aggregate, not the "
            "synthetic fixed-10-frame stress test; no accuracy claim"
        ),
    }
    out_prefix.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = _load_summary(args.summary)
    _plot(summary, args.out)
    print(f"wrote {args.out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
