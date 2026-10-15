"""Evaluate SPEC 0032 training runs with Achard's tracking metrics.

Why: SPEC 0032 reproduces Achard's trained-Trackastra benchmark with the
DynaCLR encoder. Each seed writes its own timestamped run directory under a
shared outdir; this script selects one run per seed, runs his
``compute_from_config`` (greedy tracking + traccuracy metrics) over each, and
aggregates through his own ``aggregate_metrics`` so the numbers are computed the
same way as in the thesis.

Aggregation (matches his pipeline and the thesis convention):
- per-frame ``metrics()`` rows are collapsed per video by ``aggregate_metrics``
  (sums AOGM/edge counts, means the ratios);
- the per-video table is the result of that collapse;
- the per-run table is the mean over each run's test videos;
- the mean/std table is mean and std across the selected runs.

Usage:
    python eval_etultra.py <train_outdir> <config_yaml> <eval_outdir> [seed ...]

With no seeds given, seeds 0, 1, 2 are expected; for each expected seed the
newest run directory is used, so a stale directory from a killed run is ignored.
His ``cy/metrics`` code is vendored next to this script (metrics.py,
metrics_utils.py).
"""

import os
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

import etultra_dynaclr.fork_registry_bridge  # noqa: F401  registers dynaclr-scaledcnn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics_utils import aggregate_metrics, compute_from_config  # noqa: E402

DEFAULT_SEEDS = (0, 1, 2)


def _install_feature_ablation() -> None:
    """Zero or shuffle pretrained features for a feature-usage control (S3)."""
    mode = os.environ.get("DYNACLR_ABLATE")
    if not mode:
        return
    from etultra_dynaclr.fork_registry_bridge import ForkScaledCNNDynaCLRFeatures

    original = ForkScaledCNNDynaCLRFeatures.compute_region_features

    def ablated(self, coords=None, masks=None, timepoints=None, labels=None):
        feats = original(self, coords=coords, masks=masks, timepoints=timepoints, labels=labels)
        if mode == "zero":
            return torch.zeros_like(feats)
        if mode == "shuffle":
            return feats[torch.randperm(feats.shape[0])]
        return feats

    ForkScaledCNNDynaCLRFeatures.compute_region_features = ablated
    print(f"FEATURE_ABLATION={mode}")


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


def _video_rows(eval_root: Path) -> list[pd.DataFrame]:
    """Collapse each run/video ``metrics.csv`` with his ``aggregate_metrics``."""
    rows = []
    for csv_path in sorted(eval_root.glob("*/tracked_*/metrics/metrics.csv")):
        parts = csv_path.relative_to(eval_root).parts
        run_name = parts[0]
        video = parts[1].replace("tracked_", "")
        agg = aggregate_metrics(pd.read_csv(csv_path))
        agg.insert(0, "video", video)
        agg.insert(0, "run", run_name)
        rows.append(agg)
    return rows


def evaluate(train_outdir: Path, config_path: str, eval_root: Path, seeds) -> None:
    """Run his metrics for the selected runs and aggregate."""
    _install_feature_ablation()
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

    rows = _video_rows(eval_root)
    if not rows:
        print("NO_METRICS")
        sys.exit(1)

    per_video = pd.concat(rows, ignore_index=True)
    # Reproducible AOGM decomposition (CTC weights used by his metrics.py):
    # AOGM-A = fp_edges + 1.5*fn_edges + ws_edges ; AOGM-D = fp_nodes + 10*fn_nodes + 5*ns_nodes.
    per_video["AOGM-A"] = per_video["fp_edges"] + 1.5 * per_video["fn_edges"] + per_video["ws_edges"]
    per_video["AOGM-D"] = (
        per_video["fp_nodes"] + 10 * per_video["fn_nodes"] + 5 * per_video["ns_nodes"]
    )
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
