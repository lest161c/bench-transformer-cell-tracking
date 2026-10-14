"""Evaluate SPEC 0032 training runs with Achard's tracking metrics.

Why: SPEC 0032 reproduces Achard's trained-Trackastra benchmark with the
DynaCLR encoder. Training produces one run directory per seed under a shared
outdir; this script runs his ``compute_from_config`` over every completed run
directory and aggregates the per-video metrics into per-run and mean/std tables.

Aggregation convention (matches the thesis: a per-model point estimate is the
mean across test videos, and the reported spread is over the runs):
- per-video table: every (run, video) row as produced by his ``metrics``;
- per-run table: the mean over each run's test videos;
- mean/std table: mean and std across the runs, per metric column.

Usage:
    python eval_etultra.py <train_outdir> <config_yaml> <eval_outdir>

The fork checkout is expected at FORK_ROOT; its ``scripts/eval`` directory is
put on ``sys.path`` so ``metrics_utils`` can resolve ``metrics``.
"""

import sys
from pathlib import Path

import pandas as pd

import etultra_dynaclr.fork_registry_bridge  # noqa: F401  registers dynaclr-scaledcnn

FORK_ROOT = Path("/data/cat/ws/lest161c-ssl_cell_tracking/bench-transformer-cell-tracking/cyril_trackastra")
sys.path.insert(0, str(FORK_ROOT / "scripts" / "eval"))

from metrics_utils import compute_from_config  # noqa: E402


def evaluate(train_outdir: Path, config_path: str, eval_root: Path) -> None:
    """Run his metrics for each run dir under ``train_outdir`` and aggregate."""
    eval_root.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(
        d for d in train_outdir.iterdir() if d.is_dir() and (d / "model.pt").exists()
    )
    if not run_dirs:
        print("NO_RUNS")
        sys.exit(1)

    for run_dir in run_dirs:
        print(f"EVAL_RUN {run_dir.name}", flush=True)
        compute_from_config(
            config_dir=config_path, model_dir=run_dir, outdir=eval_root / run_dir.name
        )

    frames = []
    for csv_path in eval_root.glob("**/metrics/metrics.csv"):
        metrics_df = pd.read_csv(csv_path)
        metrics_df["run"] = csv_path.relative_to(eval_root).parts[0]
        frames.append(metrics_df)
    if not frames:
        print("NO_METRICS")
        sys.exit(1)

    per_video = pd.concat(frames, ignore_index=True)
    per_video.to_csv(eval_root / "summary_per_video.csv", index=False)

    numeric = per_video.select_dtypes(include="number").copy()
    numeric["run"] = per_video["run"]
    per_run = numeric.groupby("run").mean()
    per_run.to_csv(eval_root / "summary_per_run.csv")

    summary = per_run.agg(["mean", "std"]).T
    summary.to_csv(eval_root / "summary_mean_std.csv")
    print(summary)
    print("EVAL_DONE_OK")


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) != 4:
        raise SystemExit("usage: eval_etultra.py <train_outdir> <config_yaml> <eval_outdir>")
    evaluate(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))


if __name__ == "__main__":
    main()
