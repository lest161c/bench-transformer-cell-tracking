#!/usr/bin/env python3
"""Aggregate the multi-seed K-sweep held-out/per-condition evaluations
(SPEC 0011) into a single summary CSV.

Purpose
    Turns the per-condition evaluation files written by
    ``benchmark_training/scripts/slurm/run_eval_multiseed.slurm`` (job
    4313458; one ``eval_<COND>.csv`` per vanvliet condition, JSON content
    ``{sequence: {TRA, AOGM}}``) into the per-config mean+-std table that
    replaces the single-seed K=16 claims in the report's evaluation
    section (SPEC 0011).

How to run
    1. Mirror the cluster eval files (VPN required):
       rsync -av --include='*/' --include='eval_*.csv' \\
           --include='eval_*.provenance.json' --exclude='*' \\
           capella:/data/cat/ws/lest161c-ssl_cell_tracking/trackastra/runs/ \\
           results/knn_sweep_multiseed/runs/
    2. python3 report/scripts/aggregate_multiseed_eval.py
       [--runs-dir DIR] [--out CSV]

    Writes ``results/knn_sweep_multiseed/summary.csv`` (one row per
    config and seed, both evaluation scopes) and prints the per-config
    mean+-std table. Deterministic; no randomness.

Provenance gate (SPEC 0039)
    Every aggregated eval file must have a parseable sidecar
    (``eval_<COND>.csv.provenance.json``, written by eval_cv.py) whose
    ``run_dir`` matches the directory the file sits in. Missing, unparseable,
    or mismatched sidecars abort the aggregation WITHOUT writing the summary —
    the 2026-10 sweep aggregated duplicated eval files precisely because no
    such gate existed. The sidecar's checkpoint md5 and run dir are recorded
    in the summary rows (``checkpoint_md5``, ``provenance_run_dir``).

Aggregation protocol (locked)
    * For each (config, seed): the NEWEST run dir containing eval files
      is used (failed re-queue attempts create fresh dirs; the newest is
      the completed one -- mirrors run_eval_multiseed.slurm's own
      run-dir resolution).
    * Per condition: mean TRA / mean AOGM over the condition's sequences.
    * in-sample scope: mean over the 5 training-condition means
      (26 experiments; trpL excluded from K-sweep training since the
      SPEC 0030 launcher fix, so its evaluation is genuinely held out).
    * trpL held-out scope: the trpL condition mean only (6 experiments).
      NOTE: for eval files produced BEFORE the launcher fix, trpL was part
      of training and this column was in-sample despite its label.
    * Per config: mean and sample std (ddof=1) across the 3 seeds.

Data provenance
    Training: jobs 4281394 (array 0-5), 4286428 (9/11/12/13),
    4294651 (6/7/8/10/14) on Capella; evaluation: job 4313458.
    Evaluation protocol per SPEC 0008: greedy linker, ground-truth
    detections, max link distance 50 px, --heldout <COND> per condition.
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = REPO_ROOT / "results" / "knn_sweep_multiseed" / "runs"
DEFAULT_OUT_CSV = REPO_ROOT / "results" / "knn_sweep_multiseed" / "summary.csv"

CONFIGS = ["baseline", "k2", "k4", "k8", "k16", "k32", "k64"]
SEEDS = [42, 43, 44]
# The condition excluded from K-sweep training (run_knn_sweep.slurm) so that
# its evaluation is genuinely held out. The in-sample scope pools the rest.
HELDOUT_CONDITION = "trpL"
# Owner decision D-11 (OWNER_DECISIONS_2026-10-08.md): the deliberately
# bad-flagged vanvliet experiment is excluded from every evaluation scope,
# in both protocols (old and rerun), so the trpL columns stay like-for-like.
EXCLUDED_SEQUENCES = ("150303-08_BAD",)


def completed_run_dir(config: str, seed: int, runs_dir: Path) -> str | None:
    """Newest run dir for (config, seed) that contains eval files.

    Mirrors run_eval_multiseed.slurm's resolution logic (newest candidate
    holds the completed model); the mirrored tree contains only eval
    outputs, so the newest dir with eval files is the completed run.

    Parameters
    ----------
    config : str
        Sweep variant name (e.g. ``"k16"`` or ``"baseline"``).
    seed : int
        Training seed.
    runs_dir : Path
        Directory containing the mirrored per-run eval files.

    Returns
    -------
    str | None
        Path of the newest run dir with eval files, or None.
    """
    candidates = sorted(glob.glob(str(runs_dir / f"*_{config}_s{seed}")))
    done = [c for c in candidates if glob.glob(os.path.join(c, "eval_*.csv"))]
    return done[-1] if done else None


def read_condition(path: str) -> tuple[float, float]:
    """Return (mean TRA, mean AOGM) over the sequences of one condition.

    Owner-excluded sequences (EXCLUDED_SEQUENCES) are dropped before
    averaging so every scope uses the same sequence population.

    Parameters
    ----------
    path : str
        Path to one eval_<COND>.csv JSON file.

    Returns
    -------
    tuple[float, float]
        (mean TRA, mean AOGM) over the non-excluded sequences.
    """
    data = json.load(open(path))
    for sequence in EXCLUDED_SEQUENCES:
        data.pop(sequence, None)
    tras = [v["TRA"] for v in data.values()]
    aogms = [v["AOGM"] for v in data.values()]
    return sum(tras) / len(tras), sum(aogms) / len(aogms)


def check_provenance(eval_path: str) -> tuple[str, str] | None:
    """Validate the provenance sidecar of one eval file (SPEC 0039 gate).

    The sidecar must exist next to the eval file, parse as JSON, and its
    ``run_dir`` must name the directory the eval file sits in (a sidecar from
    another run dir means the file was copied across runs — the K-sweep
    corruption signature).

    Parameters
    ----------
    eval_path : str
        Path to one eval_<COND>.csv JSON file.

    Returns
    -------
    tuple[str, str] | None
        (checkpoint_md5, provenance_run_dir) on success, None on any gate
        violation (the caller aborts and reports).
    """
    sidecar_path = eval_path + ".provenance.json"
    if not os.path.isfile(sidecar_path):
        print(f"  GATE: missing provenance sidecar: {sidecar_path}", file=sys.stderr)
        return None
    try:
        provenance = json.load(open(sidecar_path))
    except json.JSONDecodeError as exc:
        print(f"  GATE: unparseable provenance sidecar {sidecar_path}: {exc}",
              file=sys.stderr)
        return None
    recorded_run_dir = provenance.get("run_dir")
    if recorded_run_dir is None:
        print(f"  GATE: provenance sidecar {sidecar_path} has no run_dir",
              file=sys.stderr)
        return None
    sidecar_dir = os.path.basename(os.path.dirname(eval_path))
    if os.path.basename(recorded_run_dir.rstrip("/")) != sidecar_dir:
        print(f"  GATE: {sidecar_path} records run_dir {recorded_run_dir!r} but "
              f"sits in {sidecar_dir!r} — eval file copied across runs?",
              file=sys.stderr)
        return None
    checkpoint_files = provenance.get("checkpoint_files") or []
    if not checkpoint_files:
        print(f"  GATE: provenance sidecar {sidecar_path} has no checkpoint_files",
              file=sys.stderr)
        return None
    return checkpoint_files[0]["md5"], recorded_run_dir


def main() -> None:
    """Aggregate all evaluated (config, seed) runs and write the CSV.

    Runs the SPEC 0039 provenance gate first: every aggregated eval file must
    carry a sidecar whose run_dir matches its directory. Any gate violation
    aborts WITHOUT writing the summary.

    Scope semantics (SPEC 0030 audit correction, 2026-10-07): the K-sweep
    training launcher originally consumed ALL six condition directories,
    which made the trpL column in-sample despite its "held-out" label. The
    launcher now excludes the held-out condition from ``--input_train``, so
    the ``insample`` columns pool ONLY the five training conditions (26
    experiments) and the ``trpl_heldout`` columns are a genuine held-out
    measurement (6 experiments). Applying this aggregator to the OLD
    (pre-fix) eval files therefore understates the in-sample scope by the
    trpL subset — the summary must be regenerated from the retrained runs.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR,
                        help="Mirrored per-run eval tree")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_CSV,
                        help="Summary CSV to write")
    arguments = parser.parse_args()
    runs_dir = arguments.runs_dir
    out_csv = arguments.out

    rows = []
    gate_errors: list[str] = []
    for config in CONFIGS:
        for seed in SEEDS:
            run = completed_run_dir(config, seed, runs_dir)
            if run is None:
                print(f"  SKIP {config}_s{seed}: no evaluated run dir "
                      f"(variant not trained in this sweep?)", file=sys.stderr)
                continue
            train_tra, train_aogm = [], []
            trp_row = (None, None)
            checkpoint_md5 = None
            provenance_run_dir = None
            run_gate_failed = False
            for path in sorted(glob.glob(os.path.join(run, "eval_*.csv"))):
                gate = check_provenance(path)
                if gate is None:
                    gate_errors.append(path)
                    run_gate_failed = True
                    continue
                md5, recorded_run_dir = gate
                checkpoint_md5 = md5
                provenance_run_dir = recorded_run_dir
                cond = os.path.basename(path)[len("eval_"):-len(".csv")]
                tra, aogm = read_condition(path)
                if cond == HELDOUT_CONDITION:
                    trp_row = (tra, aogm)
                    continue
                train_tra.append(tra)
                train_aogm.append(aogm)
            if run_gate_failed:
                continue
            rows.append({
                "config": config,
                "seed": seed,
                "run_dir": os.path.basename(run),
                "n_conditions": len(train_tra),
                "tra_insample": sum(train_tra) / len(train_tra),
                "aogm_insample": sum(train_aogm) / len(train_aogm),
                "tra_trpl_heldout": trp_row[0],
                "aogm_trpl_heldout": trp_row[1],
                "checkpoint_md5": checkpoint_md5,
                "provenance_run_dir": provenance_run_dir,
            })

    if gate_errors:
        print(f"\nPROVENANCE GATE FAILED: {len(gate_errors)} eval file(s) without "
              f"valid sidecars; summary NOT written:", file=sys.stderr)
        for path in gate_errors:
            print(f"  {path}", file=sys.stderr)
        sys.exit(1)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"written: {out_csv} ({len(rows)} rows)\n")
    print(f"{'config':10} {'TRA-32':>16} {'AOGM-32':>16} "
          f"{'TRA-trpL':>16} {'AOGM-trpL':>16}")
    for config in CONFIGS:
        sel = [r for r in rows if r["config"] == config]
        if not sel:
            print(f"  {config:10} {'—':>16} {'—':>16} {'—':>16} {'—':>16}"
                  f"  (no evaluated runs)")
            continue
        parts = []
        for metric in ("tra_insample", "aogm_insample",
                       "tra_trpl_heldout", "aogm_trpl_heldout"):
            values = [r[metric] for r in sel]
            mean = sum(values) / len(values)
            if len(values) > 1:
                var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                std = var ** 0.5
            else:
                std = 0.0
            parts.append(f"{mean:.4f} ± {std:.4f}")
        print(f"  {config:10} {parts[0]:>16} {parts[1]:>16} "
              f"{parts[2]:>16} {parts[3]:>16}")


if __name__ == "__main__":
    main()
