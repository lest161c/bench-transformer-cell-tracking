"""Evaluate SPEC 0032 training runs with Achard's tracking metrics.

Why: SPEC 0032 reproduces Achard's trained-Trackastra benchmark with the
DynaCLR encoder. Each seed writes its own timestamped run directory under a
shared outdir; this script selects exactly one run per seed, runs his
``compute_from_config`` (greedy tracking + traccuracy metrics) over each, and
aggregates the per-video metrics.

Aggregation convention (matches the thesis: a per-model point estimate is the
mean across test videos, and the reported spread is over the runs):
- per-video table: every (run, video) row produced by his ``metrics``;
- per-run table: the mean over each run's test videos;
- mean/std table: mean and std across the selected runs.

Usage:
    python eval_etultra.py <train_outdir> <config_yaml> <eval_outdir> [seed ...]

With no seeds given, seeds 0, 1, 2 are expected. For each expected seed the
newest run directory is used, so a stale directory from a killed run is ignored.
The fork checkout is expected at FORK_ROOT.
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

import etultra_dynaclr.fork_registry_bridge  # noqa: F401  registers dynaclr-scaledcnn

FORK_ROOT = Path("/data/cat/ws/lest161c-ssl_cell_tracking/bench-transformer-cell-tracking/cyril_trackastra")
sys.path.insert(0, str(FORK_ROOT / "scripts" / "eval"))

from metrics_utils import compute_from_config  # noqa: E402

DEFAULT_SEEDS = (0, 1, 2)


def _run_seed(run_dir: Path) -> int | None:
    """Return the seed recorded in a run directory's config, or None."""
    for name in ("train_config.yaml", "config.yaml"):
        config_path = run_dir / name
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text())
            if isinstance(config, dict) and "seed" in config:
                return int(config["seed"])
    return None


def select_runs(train_outdir: Path, expected_seeds) -> list[Path]:
    """Return the newest run directory per expected seed (stale dirs ignored)."""
    newest_by_seed: dict[int | None, Path] = {}
    for run_dir in sorted(train_outdir.iterdir()):
        if not (run_dir.is_dir() and (run_dir / "model.pt").exists()):
            continue
        newest_by_seed[_run_seed(run_dir)] = run_dir  # timestamp-ordered -> newest wins
    return [newest_by_seed[seed] for seed in expected_seeds if seed in newest_by_seed]


def evaluate(train_outdir: Path, config_path: str, eval_root: Path, seeds) -> None:
    """Run his metrics for the selected runs and aggregate."""
    eval_root.mkdir(parents=True, exist_ok=True)
    run_dirs = select_runs(train_outdir, seeds)
    if not run_dirs:
        print("NO_RUNS")
        sys.exit(1)
    for run_dir in run_dirs:
        print(f"EVAL_RUN seed={_run_seed(run_dir)} dir={run_dir.name}", flush=True)
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
    if len(sys.argv) < 4:
        raise SystemExit("usage: eval_etultra.py <train_outdir> <config_yaml> <eval_outdir> [seed ...]")
    seeds = tuple(int(s) for s in sys.argv[4:]) or DEFAULT_SEEDS
    evaluate(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), seeds)


if __name__ == "__main__":
    main()
