#!/usr/bin/env python3
"""Generate the report's learned-feature comparison figure (SPEC 0033 S8 / T14).

Two-panel bar chart of AOGM for our DynaCLR ScaledCNN+BN encoder against the
reference's published rows, on Bacteria (vanvliet) and DeepCell (linear axes;
values are the mean across the test videos, the reference's Tables 3.4/3.5
convention). Data: Achard's thesis Tables 3.4/3.5 (reference) and our Et-Ultra
runs (SPEC 0033; `benchmark_ssl/results/etultra_eval/`); see labbook E19 for the
aggregation-convention correction (Bacteria 8.81). The DeepCell panel carries
the protocol-matched 91/27 head-training arm (19.35, SPEC 0044 T8) on the same
frozen encoder and test videos as the reference; the earlier 28/2 arm (26.9)
is tabulated in Tables 5/13 rather than plotted.

Run from the repository root:
    benchmark_ssl/.venv/bin/python report-shorten/scripts/generate_feature_injection_figure.py
Writes report-shorten/figs/18_feature_injection_comparison.pdf and a
.meta.json sidecar.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT = REPO_ROOT / "report-shorten"
OUT = REPORT / "figs" / "18_feature_injection_comparison.pdf"

# Okabe-Ito slots, matching the report style. Our single DeepCell bar uses the
# vermilion slot; the 28/2 arm is not plotted (see Tables 5/13).
OKABE = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
         "green": "#009E73", "blue": "#0072B2", "vermilion": "#D55E00"}

# AOGM (lower is better). "ours" is our DynaCLR ScaledCNN+BN (mean over 3 seeds).
# Each row is (label, AOGM, facecolor, hatch-or-None).
BACTERIA = [
    ("ours", 8.81, OKABE["vermilion"], None),
    ("SAM2.1", 7.00, OKABE["blue"], None),
    ("WRFEAT-DO", 11.25, OKABE["green"], None),
    ("CoTracker3", 20.81, OKABE["skyblue"], None),
    ("General-2D", 49.58, OKABE["orange"], None),
]
DEEPCELL = [
    ("ours", 19.35, OKABE["vermilion"], None),
    ("SAM2.1", 19.93, OKABE["blue"], None),
    ("WRFEAT-DO", 19.31, OKABE["green"], None),
    ("CoTracker3", 18.62, OKABE["skyblue"], None),
    ("General-2D", 23.04, OKABE["orange"], None),
]


def _panel(ax, rows, title):
    """Draw one AOGM bar panel (linear axis; value = mean across test videos)."""
    labels = [name for name, _, _, _ in rows]
    values = [value for _, value, _, _ in rows]
    colors = [color for _, _, color, _ in rows]
    hatches = [hatch for _, _, _, hatch in rows]
    x = np.arange(len(rows))
    bars = ax.bar(x, values, color=colors, width=0.62)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor("white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=6.4)
    ax.set_ylabel("AOGM")
    ax.set_title(title, fontsize=9, pad=8)
    ax.set_ylim(0, max(values) * 1.20)
    for xi, value in zip(x, values):
        ax.text(xi, value, f"{value:g}", ha="center", va="bottom", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> int:
    """Render the figure and write the sidecar."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    _panel(axes[0], BACTERIA, "Bacteria (in-domain)")
    _panel(axes[1], DEEPCELL, "DeepCell (held out)")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.27)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    plt.close(fig)

    meta = {
        "figure": "18_feature_injection_comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "report-shorten/scripts/generate_feature_injection_figure.py",
        "sources": [
            "papers/cyril_thesis.txt (Tables 3.4/3.5 reference rows)",
            "benchmark_ssl/results/etultra_eval/{bacteria,deepcell}_summary_mean_std.csv",
            "docs/agent-state/research/24-etultra-deepcell-fullsplit-eval.md (ours 91/27)",
        ],
        "values": {
            "bacteria": {name.replace("\n", " "): value for name, value, _, _ in BACTERIA},
            "deepcell": {name.replace("\n", " "): value for name, value, _, _ in DEEPCELL},
        },
        "note": "AOGM per model, mean across the test videos (his Tables 3.4/3.5 convention); ours = DynaCLR ScaledCNN+BN, 3 seeds. The DeepCell panel is our protocol-matched arm on the frozen encoder and test videos; the head-training split details are in the caption and Tables 5/13.",
    }
    OUT.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {OUT} and sidecar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
