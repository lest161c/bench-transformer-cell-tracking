#!/usr/bin/env python3
"""Aggregate the multi-seed K-sweep per-condition eval outputs into summary tables.

Purpose
-------
SPEC 0008 (docs/agent-state/specs/0008-multiseed-sweep-eval/SPEC.md) step 2:
turn the per-condition eval JSONs written by ``benchmark_training/scripts/eval_cv.py``
into the per-run and per-config CSV tables that feed the report's mean+-std
sweep table. One per-condition JSON per (run, condition); the aggregator
pools all per-experiment rows across the six held-out conditions and computes
the run-level mean TRA and mean AOGM (mirroring the historical README claim
"TRA/AOGM on 32 vanvliet experiments"), then collapses the runs to a
per-config mean +- std across seeds.

Provenance
----------
* Per-condition file format = JSON written by eval_cv.py (lines 107-118):
  a top-level dict ``{experiment_name: {"TRA": float|None, "AOGM": float|None}
  | {"error": str}}``. The ``.csv`` extension on the filename is misleading;
  the contents are JSON (the eval launcher documents this caveat too).
* Run dirs are matched by glob ``<runs-dir>/*_knn_sweep_<variant>_s<seed>``.
  The glob is intentionally date-agnostic: the training launcher
  ``run_knn_sweep.slurm`` does not pass ``--resume``, so each (re)submission
  (e.g. the OOM-resubmitted slice 4293505) creates a fresh
  ``<timestamp>_knn_sweep_<variant>_s<seed>`` directory. Among matches we
  pick the one whose six eval files are all present; ambiguity is treated
  as an unrecoverable operator decision (logged + skipped, never guessed).
* git_commit is read from ``<run>/train_config.yaml`` top-level field via a
  minimal column-0 line scan (PyYAML avoided to honour the "no new
  dependencies beyond stdlib + numpy/pandas" rule; the file is flat enough
  that scanning lines starting with the key at column 0 is reliable).

Usage
-----
    # defaults read $TRK/runs and write into results/knn_sweep_multi/
    python benchmark_training/scripts/aggregate_multiseed.py

    # explicit paths (the smoke test uses this)
    python benchmark_training/scripts/aggregate_multiseed.py \\
        --runs-dir /tmp/opencode/agg_smoke/runs \\
        --output     /tmp/opencode/agg_smoke/per_run_summary.csv \\
        --summary-output /tmp/opencode/agg_smoke/config_summary.csv

    # force a single git_commit for every row (e.g. while the actual
    # train_config.yaml values are still being pulled to the repo)
    python benchmark_training/scripts/aggregate_multiseed.py --git-commit DEADBEEF

Outputs
-------
* ``--output`` (default ``results/knn_sweep_multi/per_run_summary.csv``):
  one row per (config, seed) pair that successfully aggregated; skipped
  runs are NOT written (their absence in the CSV IS the signal of partial
  completion). Columns: config, seed, run_dir, git_commit, n_experiments,
  tra_mean, aogm_mean.
* ``--summary-output`` (default ``results/knn_sweep_multi/config_summary.csv``):
  one row per config (baseline, k4, k16, k32, k64 in that order) with
  mean +- std across the seeds that did aggregate. Columns: config,
  n_seeds, tra_mean, tra_std, aogm_mean, aogm_std. Population std (ddof=0)
  is used to match eval_cv.py's own summary convention.

Numerics: means and standard deviations are computed in float64 via numpy
(np.mean / np.std with ddof=0) and written at full precision; the README
and report layer are responsible for display rounding.

The script never raises on a single bad run or a single malformed file:
missing or incomplete runs are logged and skipped; the remaining runs are
still aggregated.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Sweep mapping — must match run_eval_multiseed.slurm VARIANTS/SEEDS arrays
# (array index = VARIANT_IDX*3 + SEED_IDX, seed-minor).
SWEEP_VARIANTS: tuple[str, ...] = ("baseline", "k4", "k16", "k32", "k64")
SWEEP_SEEDS: tuple[int, ...] = (42, 43, 44)

# Held-out conditions — must match the eval loop in run_eval_multiseed.slurm
# and the val-conditions in run_knn_sweep.slurm's --input_train.
EVAL_CONDITIONS: tuple[str, ...] = ("rpsM", "recA", "pheA", "metA", "cib", "trpL")

# SPEC 0008 lock: the full vanvliet evaluation set has 32 experiments across
# the six conditions. Used only for an informational log line, not as a
# hard gate (a future dataset extension would break that gate but still
# aggregate correctly).
EXPECTED_EXPERIMENT_COUNT: int = 32


def evaluate_condition_file(condition_file: Path, condition: str) -> list[dict] | None:
    """Load one per-condition JSON and return its per-experiment entries.

    Reads ``condition_file`` as JSON (the file has a .csv extension but
    contains JSON written by eval_cv.py) and converts its top-level dict
    ``{experiment_name: per_experiment_dict}`` into a flat list of
    per-experiment dicts tagged with their source condition.

    Parameters
    ----------
    condition_file : Path
        Path to one ``eval_<COND>.csv`` JSON document.
    condition : str
        Name of the held-out condition (e.g. ``"rpsM"``); recorded into each
        returned entry under the ``"condition"`` key for traceability.

    Returns
    -------
    list[dict] | None
        List of per-experiment dicts (each contains at least the keys
        ``"condition"`` and ``"experiment"``; ``TRA`` / ``AOGM`` / ``error``
        keys are passed through from the source JSON). Returns ``None``
        when the file is unreadable or not a JSON object — the caller is
        expected to log the failure and skip the containing run.
    """
    try:
        with condition_file.open() as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  SKIP condition {condition}: cannot parse {condition_file} ({exc})",
              file=sys.stderr)
        return None
    if not isinstance(raw, dict):
        print(f"  SKIP condition {condition}: top-level JSON is {type(raw).__name__}, expected object",
              file=sys.stderr)
        return None
    pooled: list[dict] = []
    for experiment_name, entry in raw.items():
        if not isinstance(entry, dict):
            print(f"  WARN condition {condition}: entry {experiment_name!r} is {type(entry).__name__}, skipping",
                  file=sys.stderr)
            continue
        pooled.append({"condition": condition, "experiment": experiment_name, **entry})
    return pooled


def pool_run_experiments(run_dir: Path) -> tuple[list[dict], list[str]] | None:
    """Pool per-experiment entries across the six eval files in a run dir.

    Each eval file must exist AND parse successfully for the run to be
    considered complete; this is the "incomplete eval files -> skip the
    whole run" rule (a partial pooled mean would not be comparable across
    runs, and the sweep IS partially complete on the cluster today). A run
    that has all six files present but contains per-experiment error rows
    is still considered complete (eval_cv attempted every experiment and
    reported per-experiment outcomes); per-experiment error filtering
    happens at the mean step.

    Parameters
    ----------
    run_dir : Path
        Resolved per-run directory (e.g. ``<runs-dir>/.../knn_sweep_k16_s43``).

    Returns
    -------
    (pooled_entries, skip_reasons) | None
        ``pooled_entries`` is the flat list of per-experiment dicts from all
        six condition files (including error entries); ``skip_reasons`` is
        the list of human-readable strings explaining why any condition was
        skipped (empty on success). Returns ``None`` if any condition file
        is missing or unparseable, in which case the caller should skip the
        entire run.
    """
    pooled: list[dict] = []
    skip_reasons: list[str] = []
    for condition in EVAL_CONDITIONS:
        condition_file = run_dir / f"eval_{condition}.csv"
        if not condition_file.is_file():
            skip_reasons.append(f"missing {condition_file.name}")
            continue
        entries = evaluate_condition_file(condition_file, condition)
        if entries is None:
            skip_reasons.append(f"unparseable {condition_file.name}")
            continue
        pooled.extend(entries)
    if skip_reasons:
        print(f"  SKIP run {run_dir.name}: incomplete eval files ({'; '.join(skip_reasons)})",
              file=sys.stderr)
        return None
    return pooled, skip_reasons


def compute_run_means(pooled_entries: list[dict]) -> tuple[int, float | None, float | None]:
    """Compute ``(n_experiments, tra_mean, aogm_mean)`` from pooled per-experiment entries.

    Means are taken over the entries that carry a non-None TRA or AOGM
    value (per-experiment error rows lack these keys and are excluded
    automatically). ``n_experiments`` is the total number of pooled entries
    INCLUDING error rows; the summary log line printed by the caller makes
    the error count visible alongside.

    Parameters
    ----------
    pooled_entries : list[dict]
        Output of :func:`pool_run_experiments` (flat list of per-experiment
        dicts across the six conditions).

    Returns
    -------
    (n_experiments, tra_mean, aogm_mean)
        ``n_experiments`` is always set; ``tra_mean`` / ``aogm_mean`` are
        ``None`` when no pooled entry carries that metric (e.g. a run whose
        eval files are all error rows). The caller is expected to treat
        ``None`` means as "skip this run".
    """
    n_experiments = len(pooled_entries)
    tra_values = [entry["TRA"] for entry in pooled_entries
                  if isinstance(entry, dict) and entry.get("TRA") is not None]
    aogm_values = [entry["AOGM"] for entry in pooled_entries
                   if isinstance(entry, dict) and entry.get("AOGM") is not None]
    tra_mean = float(np.mean(tra_values)) if tra_values else None
    aogm_mean = float(np.mean(aogm_values)) if aogm_values else None
    return n_experiments, tra_mean, aogm_mean


def read_git_commit_from_train_config(run_dir: Path) -> str | None:
    """Read the top-level ``git_commit`` field from ``<run>/train_config.yaml``.

    train.py (trackastra/scripts/train.py line 782) writes this file at
    training time capturing the commit the model was trained from. The
    file is a flat ``key: value`` dump; a minimal column-0 line scan is
    sufficient to extract one key without pulling in PyYAML (which is not
    in the allowed-dependency set).

    Parameters
    ----------
    run_dir : Path
        Per-run directory; ``train_config.yaml`` is read from it directly.

    Returns
    -------
    str | None
        The commit hash (typically 40 hex chars), or ``None`` when the
        file is missing, unreadable, or the field is absent / null /
        empty. The caller is responsible for logging a warning when
        ``None`` is returned (it breaks provenance traceability per
        SPEC 0008's acceptance criterion).
    """
    train_config_path = run_dir / "train_config.yaml"
    if not train_config_path.is_file():
        return None
    try:
        with train_config_path.open() as handle:
            for line in handle:
                if not line.startswith("git_commit:"):
                    continue
                raw_value = line[len("git_commit:"):].strip()
                # Defensive quote-strip (yaml dumps plain strings unquoted,
                # but a hand-edited file could carry quotes).
                raw_value = raw_value.strip("'\"")
                if raw_value in ("", "null", "~"):
                    return None
                return raw_value
    except OSError:
        return None
    return None


def resolve_run_dir(runs_dir: Path, variant: str, seed: int) -> Path | None:
    """Resolve the unique complete-eval run dir for one (variant, seed) pair.

    Globs ``<runs-dir>/*_knn_sweep_<variant>_s<seed>``, filters to entries
    whose six per-condition eval files are all present, and returns the
    unique match. The date prefix is intentionally absent (see module
    docstring "Provenance" section for the resubmission-timestamp reason).

    Parameters
    ----------
    runs_dir : Path
        Root directory containing all multi-seed run dirs.
    variant : str
        Sweep variant name (one of :data:`SWEEP_VARIANTS`).
    seed : int
        Random seed (one of :data:`SWEEP_SEEDS`).

    Returns
    -------
    Path | None
        Resolved path to the matching run dir, or ``None`` when no match,
        no match with a complete eval set, or multiple ambiguous matches
        are found. Each failure mode is logged with a distinctive prefix.
    """
    pattern = f"*_knn_sweep_{variant}_s{seed}"
    candidates = sorted(p for p in runs_dir.glob(pattern) if p.is_dir())
    if not candidates:
        print(f"  SKIP {variant}/s{seed}: no run dir under {runs_dir} matching {pattern}",
              file=sys.stderr)
        return None
    complete = [c for c in candidates
                if all((c / f"eval_{cond}.csv").is_file() for cond in EVAL_CONDITIONS)]
    if not complete:
        matched = ", ".join(c.name for c in candidates)
        print(f"  SKIP {variant}/s{seed}: run dir(s) found ({matched}) but none has the full set "
              f"of {len(EVAL_CONDITIONS)} eval files", file=sys.stderr)
        return None
    if len(complete) > 1:
        ambiguous = ", ".join(c.name for c in complete)
        print(f"  SKIP {variant}/s{seed}: {len(complete)} run dirs have a complete eval set "
              f"({ambiguous}); resolve manually before re-running", file=sys.stderr)
        return None
    return complete[0]


def aggregate_run(runs_dir: Path, variant: str, seed: int,
                  git_commit_override: str | None) -> dict | None:
    """Aggregate one (variant, seed) pair into a per-run summary row.

    Combines run-dir resolution, eval-file pooling, metric computation, and
    git-commit resolution. Returns ``None`` for any failure mode (the
    underlying functions have already logged the specific reason).

    Returns a dict with the per_run_summary CSV columns:
    ``{"config", "seed", "run_dir", "git_commit", "n_experiments",
    "tra_mean", "aogm_mean"}``.
    """
    run_dir = resolve_run_dir(runs_dir, variant, seed)
    if run_dir is None:
        return None
    pooled_result = pool_run_experiments(run_dir)
    if pooled_result is None:
        return None
    pooled, _ = pooled_result
    n_experiments, tra_mean, aogm_mean = compute_run_means(pooled)
    if tra_mean is None or aogm_mean is None:
        print(f"  SKIP {variant}/s{seed}: no valid per-experiment TRA/AOGM entries in "
              f"{n_experiments} pooled row(s) of {run_dir}", file=sys.stderr)
        return None
    n_error = sum(1 for entry in pooled if isinstance(entry, dict) and "error" in entry)
    print(f"  OK   {variant}/s{seed}: run={run_dir.name} n_experiments={n_experiments} "
          f"(valid={n_experiments - n_error}, error={n_error}) "
          f"TRA={tra_mean:.4f} AOGM={aogm_mean:.1f}",
          file=sys.stderr)
    if n_experiments != EXPECTED_EXPERIMENT_COUNT:
        print(f"  NOTE {variant}/s{seed}: pooled {n_experiments} experiments, expected "
              f"{EXPECTED_EXPERIMENT_COUNT} for the full vanvliet set; see SPEC 0008",
              file=sys.stderr)
    if git_commit_override is not None:
        git_commit = git_commit_override
    else:
        git_commit = read_git_commit_from_train_config(run_dir)
        if git_commit is None:
            print(f"  WARN {variant}/s{seed}: no git_commit in {run_dir}/train_config.yaml; "
                  "provenance gap (SPEC 0008 acceptance criterion)", file=sys.stderr)
            git_commit = ""
    return {
        "config": variant,
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "git_commit": git_commit,
        "n_experiments": n_experiments,
        "tra_mean": tra_mean,
        "aogm_mean": aogm_mean,
    }


def write_per_run_table(rows: list[dict], output_path: Path) -> None:
    """Write the per-run summary CSV (one row per successfully aggregated run).

    Columns: config, seed, run_dir, git_commit, n_experiments, tra_mean,
    aogm_mean. Skipped runs are absent from the file; their absence IS the
    signal of partial completion (no placeholder rows are emitted).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["config", "seed", "run_dir", "git_commit",
                  "n_experiments", "tra_mean", "aogm_mean"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_config_summary_table(rows: list[dict], summary_output_path: Path) -> None:
    """Write the per-config mean +- std summary CSV.

    Groups per-run rows by ``config`` (in :data:`SWEEP_VARIANTS` order) and
    writes one row per config with mean +- std across the seeds that did
    aggregate. Population std (``ddof=0``) matches eval_cv.py's own summary
    convention. Configs with zero aggregated seeds are omitted (logged).
    """
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["config"]].append(row)
    fieldnames = ["config", "n_seeds", "tra_mean", "tra_std", "aogm_mean", "aogm_std"]
    with summary_output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        written = 0
        for variant in SWEEP_VARIANTS:
            variant_rows = grouped.get(variant, [])
            if not variant_rows:
                print(f"  NOTE config {variant}: 0 aggregated seeds, omitted from summary",
                      file=sys.stderr)
                continue
            tra_values = [r["tra_mean"] for r in variant_rows]
            aogm_values = [r["aogm_mean"] for r in variant_rows]
            writer.writerow({
                "config": variant,
                "n_seeds": len(variant_rows),
                "tra_mean": float(np.mean(tra_values)),
                "tra_std": float(np.std(tra_values, ddof=0)),
                "aogm_mean": float(np.mean(aogm_values)),
                "aogm_std": float(np.std(aogm_values, ddof=0)),
            })
            written += 1
    print(f"  Summary: wrote {written}/{len(SWEEP_VARIANTS)} config rows to {summary_output_path}",
          file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args.

    ``--runs-dir`` defaults to ``$TRK/runs`` (the cluster workspace's
    trackastra runs root); a clear error is raised when neither the flag
    nor the env var is set.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=None,
        help="Root directory containing multi-seed run dirs. Defaults to $TRK/runs.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/knn_sweep_multi/per_run_summary.csv"),
        help="Per-run summary CSV path (default: %(default)s).",
    )
    parser.add_argument(
        "--summary-output", type=Path,
        default=Path("results/knn_sweep_multi/config_summary.csv"),
        help="Per-config mean+-std summary CSV path (default: %(default)s).",
    )
    parser.add_argument(
        "--git-commit", type=str, default=None,
        help="Override git_commit for every row (otherwise read per-run from "
             "<run>/train_config.yaml). Useful while pulling results from the "
             "cluster when train_config.yaml values are still in flight.",
    )
    args = parser.parse_args(argv)
    if args.runs_dir is None:
        trk = os.environ.get("TRK")
        if trk:
            args.runs_dir = Path(trk) / "runs"
        else:
            parser.error(
                "--runs-dir not given and $TRK is not set; pass --runs-dir explicitly "
                "(see module docstring Usage)."
            )
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point: aggregate the 15 (variant, seed) pairs and write both CSVs."""
    args = parse_args(argv)
    runs_dir: Path = args.runs_dir
    print(f"Aggregating multi-seed sweep under {runs_dir}", file=sys.stderr)
    rows: list[dict] = []
    for variant in SWEEP_VARIANTS:
        for seed in SWEEP_SEEDS:
            row = aggregate_run(runs_dir, variant, seed, args.git_commit)
            if row is not None:
                rows.append(row)
    write_per_run_table(rows, args.output)
    print(f"  Per-run: wrote {len(rows)}/{len(SWEEP_VARIANTS) * len(SWEEP_SEEDS)} rows to "
          f"{args.output}", file=sys.stderr)
    write_config_summary_table(rows, args.summary_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
