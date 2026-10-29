#!/usr/bin/env python3
"""Plot frame-CV and LOCO top-1 accuracy for the seven KNN MLP feature rows.

The figure supports SPEC 0046 (meeting 2026-10-22 remarks) by comparing the
within-experiment frame-pair cross-validation (frame CV) probe with the
leave-one-condition-out (LOCO) overall probe for each available MLP feature
row. Values are the mean over probe seeds 42, 43, and 44; the aggregate
artifact ``docs/agent-state/specs/0046-meeting-22-10-remarks/artifacts/
knn_probe_summary.csv`` holds the full precision/recall/F1/NLL table.

Run from the repository root with the benchmark environment::

    benchmark_ssl/.venv/bin/python report-shorten/scripts/plot_edge_probe_knn.py

Writes ``report-shorten/figs/21_edge_probe_knn.pdf`` and a ``.meta.json``
sidecar recording the source CSVs, plotted values, generator, and timestamp.
The two source summaries are read directly so the figure is reproducible
without the local-only ``docs/agent-state`` artifact.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
FIGURES = REPO / "report-shorten" / "figs"
OUTPUT = FIGURES / "21_edge_probe_knn.pdf"
FRAME_CV_SUMMARY = REPO / "benchmark_ssl/results/probes/knn/cluster/frame_cv/summary.csv"
LOCO_SUMMARY = REPO / "benchmark_ssl/results/probes/knn/cluster/loco/summary.csv"

PROTOCOL_K = 19
CHANCE_LEVEL = 1.0 / PROTOCOL_K
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_SEED_COUNT = len(EXPECTED_SEEDS)

# Okabe-Ito slots from docs/vis_guidelines.md. Blue and orange distinguish the
# two protocols; the neutral grey marks chance.
OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "neutral": "#999999",
    "text": "#262626",
    "grid": "#cccccc",
}

FEATURE_LABELS = {
    "regionprops_7d": "Regionprops 7D",
    "regionprops_13d": "Regionprops 13D",
    "hu_20d": "Hu 20D",
    "dino": "DINOv2",
    "cnn_frozen": "CNN frozen",
    "cnn_ntxent": "CNN NT-Xent",
    "posenc": "Position-only",
}


def _aggregate_top1(summary_path: Path, heldout_condition: str | None) -> pd.DataFrame:
    """Aggregate one summary CSV to per-feature MLP top-1 mean and std.

    Parameters
    ----------
    summary_path : Path
        Probe ``summary.csv`` produced by the KNN probe CLI.
    heldout_condition : str | None
        ``"overall"`` for the LOCO summary; ``None`` for frame CV, whose rows
        carry an empty held-out condition.

    Returns
    -------
    pandas.DataFrame
        One row per feature with columns ``feature``, ``mean``, ``std``,
        ``n_seeds``, and ``protocol_k``. The mean and standard deviation are
        taken across the per-seed fold/condition means.
    """
    summary = pd.read_csv(summary_path)
    if heldout_condition is None:
        selected = summary[
            (summary["protocol"] == "frame_cv") & (summary["probe"] == "mlp")
        ]
    else:
        selected = summary[
            (summary["protocol"] == "loco")
            & (summary["probe"] == "mlp")
            & (summary["heldout_condition"] == heldout_condition)
        ]

    seeds = tuple(sorted(int(seed) for seed in selected["seed"].unique()))
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"{summary_path} has seeds {seeds}, expected {EXPECTED_SEEDS}")
    protocol_ks = tuple(int(value) for value in selected["protocol_k"].unique())
    if protocol_ks != (PROTOCOL_K,):
        raise ValueError(
            f"{summary_path} has protocol_k {protocol_ks}, expected {PROTOCOL_K}"
        )

    grouped = selected.groupby("feature", sort=False)["top1_accuracy_mean"]
    aggregate = grouped.agg(["mean", "std", "count"]).reset_index()
    aggregate = aggregate.rename(columns={"count": "n_seeds"})
    if (aggregate["n_seeds"] != EXPECTED_SEED_COUNT).any():
        raise ValueError(f"{summary_path} does not have three seeds per feature")
    aggregate["protocol_k"] = PROTOCOL_K
    return aggregate


def _plot(
    frame_cv: pd.DataFrame, loco: pd.DataFrame
) -> tuple[list[str], dict[str, dict[str, dict[str, float]]]]:
    """Draw the grouped bar chart and return the feature order and values.

    The features are ordered by frame-CV top-1 descending so the figure reads
    as a ranking. The y-axis is top-1 accuracy on a fixed 0-1 scale, and a
    dashed neutral line marks chance at ``1 / protocol_k``.
    """
    frame_by_feature = frame_cv.set_index("feature")
    loco_by_feature = loco.set_index("feature")
    features = list(frame_by_feature.sort_values("mean", ascending=False).index)

    frame_values = frame_by_feature.loc[features, "mean"].to_numpy()
    frame_stds = frame_by_feature.loc[features, "std"].to_numpy()
    loco_values = loco_by_feature.loc[features, "mean"].to_numpy()
    loco_stds = loco_by_feature.loc[features, "std"].to_numpy()

    figure, axis = plt.subplots(figsize=(7.2, 3.9))
    positions = np.arange(len(features))
    bar_width = 0.38

    frame_bars = axis.bar(
        positions - bar_width / 2,
        frame_values,
        bar_width,
        label="Frame CV",
        color=OKABE["blue"],
        edgecolor="none",
    )
    loco_bars = axis.bar(
        positions + bar_width / 2,
        loco_values,
        bar_width,
        label="LOCO",
        color=OKABE["orange"],
        edgecolor="none",
    )

    # Value labels make the plotted numbers legible in the PDF text layer too.
    for bars, values in ((frame_bars, frame_values), (loco_bars, loco_values)):
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.4,
                color=OKABE["text"],
            )

    axis.axhline(
        CHANCE_LEVEL,
        color=OKABE["neutral"],
        linestyle="--",
        linewidth=1.0,
        label=f"Chance (1/{PROTOCOL_K} = {CHANCE_LEVEL:.3f})",
        zorder=0,
    )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [FEATURE_LABELS[feature] for feature in features],
        rotation=0,
        ha="center",
        fontsize=7.0,
    )
    axis.set_ylabel("Top-1 accuracy", fontsize=8.5)
    axis.set_xlabel("Feature row", fontsize=8.5)
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.tick_params(axis="both", labelsize=7.5)
    axis.grid(axis="y", color=OKABE["grid"], linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(OKABE["text"])
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        ncols=3,
        frameon=False,
        fontsize=7.5,
        handlelength=1.5,
        columnspacing=1.2,
    )

    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=300)
    plt.close(figure)

    values = {
        "frame_cv": {
            feature: {
                "mean": float(frame_by_feature.loc[feature, "mean"]),
                "std": float(frame_by_feature.loc[feature, "std"]),
                "n_seeds": int(frame_by_feature.loc[feature, "n_seeds"]),
            }
            for feature in features
        },
        "loco_overall": {
            feature: {
                "mean": float(loco_by_feature.loc[feature, "mean"]),
                "std": float(loco_by_feature.loc[feature, "std"]),
                "n_seeds": int(loco_by_feature.loc[feature, "n_seeds"]),
            }
            for feature in features
        },
    }

    # A rank check keeps the figure honest if the upstream summaries change.
    if not np.all(np.isfinite(frame_stds)) or not np.all(np.isfinite(loco_stds)):
        raise ValueError("non-finite standard deviation in the probe summaries")
    return features, values


def main() -> int:
    """Generate the figure and its metadata sidecar."""
    frame_cv = _aggregate_top1(FRAME_CV_SUMMARY, heldout_condition=None)
    loco = _aggregate_top1(LOCO_SUMMARY, heldout_condition="overall")
    features, values = _plot(frame_cv, loco)

    meta = {
        "figure": "21_edge_probe_knn",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "report-shorten/scripts/plot_edge_probe_knn.py",
        "sources": [
            str(FRAME_CV_SUMMARY.relative_to(REPO)),
            str(LOCO_SUMMARY.relative_to(REPO)),
            "docs/agent-state/specs/0046-meeting-22-10-remarks/artifacts/knn_probe_summary.csv",
        ],
        "metric": "top1_accuracy",
        "probe": "mlp",
        "protocol_k": PROTOCOL_K,
        "chance_level": CHANCE_LEVEL,
        "seeds": list(EXPECTED_SEEDS),
        "aggregation": (
            "mean and sample standard deviation (ddof=1) across seeds of each "
            "seed's mean over 10 frame-CV folds or 6 LOCO held-out conditions"
        ),
        "feature_order": features,
        "values": values,
        "note": (
            "Seven MLP feature rows are plotted; sam21 is unavailable because "
            "the optional sam2 package and checkpoint are missing."
        ),
    }
    OUTPUT.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {OUTPUT} and {OUTPUT.with_suffix('.meta.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
