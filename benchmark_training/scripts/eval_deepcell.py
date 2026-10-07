#!/usr/bin/env python3
"""Evaluate a trained Trackastra checkpoint on the held-out deepcell test set.

The deepcell test set (12 sequences, 00-11) is never used in training.
This gives a true generalization measure, unlike evaluating on vanvliet
validation data (which is a subset of the training data).

Usage:
    python eval_deepcell.py --run-dir <trackastra_run_dir> [--device cuda]

Output:
    <run_dir>/deepcell_eval_results.json   ← per-sequence TRA, AOGM
    <run_dir>/deepcell_eval_summary.json   ← aggregated mean/std
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DATA_ROOT = Path("/data/cat/ws/mawe985g-data/data/celltracking/deepcell")


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

    test_seqs = sorted(
        d.name for d in (DATA_ROOT / "test").iterdir() if d.is_dir()
    )

    print(f"Found {len(test_seqs)} deepcell test sequences:")
    for seq in test_seqs:
        print(f"  {seq}")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        for seq in test_seqs:
            seq_dir = DATA_ROOT / "test" / seq
            img_dir = seq_dir / "img"
            gt_dir = seq_dir / "TRA"
            print(f"  {seq}: img={img_dir}  GT={gt_dir}")
        return

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

    for seq in test_seqs:
        seq_dir = DATA_ROOT / "test" / seq
        img_dir = seq_dir / "img"
        gt_dir = seq_dir / "TRA"

        print(f"\n  [{seq}] tracking … ", end="", flush=True)
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
            all_results[seq] = {"error": str(exc)}
            continue

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from trackastra.tracking.utils import graph_to_ctc
            graph_to_ctc(track_graph, masks, outdir=Path(tmpdir))

            print(f"  [{seq}] computing metrics … ", end="", flush=True)
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

                all_results[seq] = {
                    "TRA": float(tra) if tra is not None else None,
                    "AOGM": float(aogm) if aogm is not None else None,
                }
            except Exception as exc:
                print(f"FAILED: {exc}")
                all_results[seq] = {"error": str(exc)}

    results_path = run_dir / "deepcell_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nPer-sequence results saved to {results_path}")

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

        summary_path = run_dir / "deepcell_eval_summary.json"
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
