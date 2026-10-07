#!/usr/bin/env python3
"""Evaluate a trained Trackastra checkpoint on the vanvliet validation conditions.

Unlike eval_cv.py (single held-out condition, 6-fold CV protocol), this script
evaluates ALL validation conditions (rpsM, recA, pheA) in one run and produces
a single combined summary. It loads a model checkpoint, runs greedy tracking
on each validation sequence, and computes TRA/AOGM metrics using traccuracy.

Usage:
    python eval_validation_conditions.py --run-dir <trackastra_run_dir> [--device cuda]
    python eval_validation_conditions.py --run-dir runs/2026-08-17_12-03-17_dense_flash_no_cutoff --dry-run

Output:
    <run_dir>/eval_results.json   <- per-sequence TRA, AOGM
    <run_dir>/eval_summary.json   <- aggregated mean/std
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

TRK = Path("/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra")
DATA_ROOT = Path("/data/cat/ws/mawe985g-data/data/celltracking/vanvliet")

# Validation conditions used in training (from train_config.yaml)
VAL_CONDITIONS = ["rpsM", "recA", "pheA"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Path to trackastra run directory (containing model.pt + config.yaml)",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: Run directory does not exist: {run_dir}")
        sys.exit(1)

    # Collect validation sequences
    val_seqs = []
    for cond in VAL_CONDITIONS:
        cond_dir = DATA_ROOT / cond
        if not cond_dir.exists():
            print(f"WARNING: Condition directory not found: {cond_dir}")
            continue
        for exp_dir in sorted(cond_dir.iterdir()):
            if exp_dir.is_dir() and (exp_dir / "TRA").exists():
                val_seqs.append((cond, exp_dir.name, exp_dir))

    print(f"Found {len(val_seqs)} validation sequences:")
    for cond, exp, _ in val_seqs:
        print(f"  {cond}/{exp}")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        for cond, exp, exp_dir in val_seqs:
            print(f"  {cond}/{exp}: GT={exp_dir / 'TRA'}")
        return

    # Lazy imports
    from trackastra.model import Trackastra
    from traccuracy import run_metrics
    from traccuracy.loaders import load_ctc_data
    from traccuracy.matchers import CTCMatcher
    from traccuracy.metrics import CTCMetrics

    print(f"\nLoading model from {run_dir} … ", end="", flush=True)
    t0 = time.time()
    model = Trackastra.from_folder(run_dir, device=args.device)
    print(f"done ({time.time() - t0:.1f}s)")

    all_results = {}

    for cond, exp, exp_dir in val_seqs:
        seq_name = f"{cond}/{exp}"
        img_dir = exp_dir / "img"
        gt_dir = exp_dir / "TRA"

        print(f"\n  [{seq_name}] tracking … ", end="", flush=True)
        t0 = time.time()

        try:
            track_graph, masks = model.track_from_disk(
                imgs_path=img_dir,
                masks_path=gt_dir,
                mode="greedy",
            )
            print(f"done ({time.time() - t0:.1f}s) "
                  f"graph: {track_graph.number_of_nodes()} nodes, "
                  f"{track_graph.number_of_edges()} edges")
        except Exception as exc:
            print(f"FAILED: {exc}")
            all_results[seq_name] = {"error": str(exc)}
            continue

        # Save CTC predictions to temp dir for traccuracy
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from trackastra.tracking.utils import graph_to_ctc
            graph_to_ctc(track_graph, masks, outdir=Path(tmpdir))

            print(f"  [{seq_name}] computing metrics … ", end="", flush=True)
            t0 = time.time()
            try:
                gt_data = load_ctc_data(str(gt_dir))
                pred_data = load_ctc_data(str(tmpdir))

                ctc_result, _ = run_metrics(
                    gt_data, pred_data, CTCMatcher(), [CTCMetrics()]
                )

                tra = ctc_result[0]["results"].get("TRA", None)
                aogm = ctc_result[0]["results"].get("AOGM", None)

                print(f"done ({time.time() - t0:.1f}s) "
                      f"TRA={tra:.4f}  AOGM={aogm:.1f}")

                all_results[seq_name] = {
                    "TRA": float(tra) if tra is not None else None,
                    "AOGM": float(aogm) if aogm is not None else None,
                }
            except Exception as exc:
                print(f"FAILED: {exc}")
                all_results[seq_name] = {"error": str(exc)}

    # Save results
    results_path = run_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nPer-sequence results saved to {results_path}")

    # Summary
    valid = [r for r in all_results.values() if "error" not in r]
    if valid:
        tra_vals = [r["TRA"] for r in valid if r.get("TRA") is not None]
        aogm_vals = [r["AOGM"] for r in valid if r.get("AOGM") is not None]

        summary = {
            "n_sequences": len(valid),
            "n_failed": len(all_results) - len(valid),
            "mean_TRA": float(np.mean(tra_vals)) if tra_vals else None,
            "std_TRA": float(np.std(tra_vals)) if tra_vals else None,
            "mean_AOGM": float(np.mean(aogm_vals)) if aogm_vals else None,
            "std_AOGM": float(np.std(aogm_vals)) if aogm_vals else None,
        }

        summary_path = run_dir / "eval_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"  SUMMARY ({len(valid)}/{len(all_results)} sequences)")
        print(f"{'=' * 60}")
        print(f"  mean TRA:  {summary['mean_TRA']:.4f} ± {summary['std_TRA']:.4f}")
        print(f"  mean AOGM: {summary['mean_AOGM']:.1f} ± {summary['std_AOGM']:.1f}")
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
