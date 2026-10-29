#!/usr/bin/env python3
"""Extract per-epoch wall times from the multi-seed K-sweep tensorboard logs.

Purpose
    For every (config, seed) run of the multi-seed K-sweep, read the
    tensorboard event file of the newest run directory that contains
    evaluation outputs (the same newest-with-eval resolution as
    ``benchmark_training/scripts/analysis/aggregate_multiseed_eval.py``) and derive the per-epoch
    wall time from the ``val_loss`` scalar series: one validation event per
    epoch, so the wall-time difference between consecutive events is the
    per-epoch duration (training + validation) in seconds.

How to run
    Designed to run ON the cluster against the mirrored-free run tree:

    scp report/scripts/extract_epoch_times.py capella:/tmp/
    ssh capella "/data/cat/ws/lest161c-ssl_cell_tracking/trackastra/.venv/bin/python \
        /tmp/extract_epoch_times.py \
        --runs /data/cat/ws/lest161c-ssl_cell_tracking/trackastra/runs \
        --out /tmp/knn_sweep_epoch_times.json"
    rsync capella:/tmp/knn_sweep_epoch_times.json results/knn_sweep_multiseed/

Output
    JSON mapping run directory name to
    ``{"config", "seed", "n_epochs", "epoch_times_s": [...], "mean_epoch_s",
    "median_epoch_s"}``. Deterministic; no randomness.

Data provenance
    Training jobs of the multi-seed K-sweep on Capella (see
    ``benchmark_training/scripts/analysis/aggregate_multiseed_eval.py`` docstring); tensorboard
    event files under ``<run>/tb/version_0``.
"""

import argparse
import glob
import json
import os
import re
import sys

CONFIGS = ["baseline", "k2", "k4", "k8", "k16", "k32", "k64"]
SEEDS = [42, 43, 44]
RUN_NAME_RE = re.compile(r"(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_knn_sweep_(?P<config>[a-z0-9]+)_s(?P<seed>\d+)$")


def completed_run_dirs(runs_dir: str) -> list[str]:
    """Newest run dir per (config, seed) that contains eval files.

    Mirrors the resolution logic of ``aggregate_multiseed_eval.py``: failed
    re-queue attempts create fresh directories; the newest directory with
    eval outputs belongs to the completed run.
    """
    resolved = []
    for config in CONFIGS:
        for seed in SEEDS:
            candidates = sorted(
                glob.glob(os.path.join(runs_dir, f"*knn_sweep_{config}_s{seed}")))
            done = [c for c in candidates
                    if glob.glob(os.path.join(c, "eval_*.csv"))]
            if not done:
                print(f"  SKIP {config}_s{seed}: no evaluated run dir", file=sys.stderr)
                continue
            resolved.append(done[-1])
    return resolved


def extract_epoch_times(run_dir: str) -> dict:
    """Read the tensorboard event file and return per-epoch wall times.

    Uses the ``val_loss`` scalar series (one event per epoch, logged at
    epoch end); the wall-time difference between consecutive events is the
    per-epoch duration in seconds.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tb_dir = os.path.join(run_dir, "tb", "version_0")
    acc = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    acc.Reload()
    scalars = acc.Scalars("val_loss")
    wall_times = [event.wall_time for event in scalars]
    epoch_times = [b - a for a, b in zip(wall_times, wall_times[1:])]
    return {
        "n_epochs": len(wall_times),
        "epoch_times_s": [round(t, 3) for t in epoch_times],
        "mean_epoch_s": round(sum(epoch_times) / len(epoch_times), 3) if epoch_times else None,
        "median_epoch_s": round(sorted(epoch_times)[len(epoch_times) // 2], 3) if epoch_times else None,
    }


def main() -> None:
    """Resolve the completed run dirs and write the per-epoch timing JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="Cluster runs directory")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    out = {}
    for run_dir in completed_run_dirs(args.runs):
        name = os.path.basename(run_dir)
        match = RUN_NAME_RE.search(name)
        if not match:
            print(f"  SKIP {name}: unparseable run name", file=sys.stderr)
            continue
        stats = extract_epoch_times(run_dir)
        stats.update(config=match.group("config"), seed=int(match.group("seed")))
        out[name] = stats
        print(f"  {name}: {stats['n_epochs']} epochs, "
              f"mean {stats['mean_epoch_s']} s/epoch")

    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1)
    print(f"written: {args.out} ({len(out)} runs)")


if __name__ == "__main__":
    main()
