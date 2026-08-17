#!/usr/bin/env python3
"""Evaluate a CV-trained Trackastra checkpoint on its held-out condition.

Used by run_cv.slurm. Finds all experiment subdirectories under the held-out
condition and runs TRA/AOGM metrics on each.

Usage:
    python eval_cv.py --run-dir <run_dir> --heldout <cond> --data-root <root> \
        --output <results.json>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--heldout", type=str, required=True,
                        help="Name of the held-out condition (e.g. rpsM)")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory containing all conditions")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-distance", type=int, default=50,
                        help="Max distance for candidate graph edges")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_root = Path(args.data_root)

    heldout_dir = data_root / args.heldout
    if not heldout_dir.exists():
        print(f"ERROR: Held-out directory not found: {heldout_dir}")
        sys.exit(1)

    # Find all experiment subdirectories
    exp_dirs = sorted(d for d in heldout_dir.iterdir()
                      if d.is_dir() and (d / "TRA").exists())

    print(f"Held-out: {args.heldout}")
    print(f"Experiments: {len(exp_dirs)}")
    for d in exp_dirs:
        print(f"  {d.name}")
    print(f"Model: {run_dir}")

    # Lazy imports
    from trackastra.model import Trackastra
    from traccuracy import run_metrics
    from traccuracy.loaders import load_ctc_data
    from traccuracy.matchers import CTCMatcher
    from traccuracy.metrics import CTCMetrics

    print(f"\nLoading model … ", end="", flush=True)
    t0 = time.time()
    model = Trackastra.from_folder(run_dir, device=args.device)
    print(f"done ({time.time() - t0:.1f}s)")

    results = {}
    for exp_dir in exp_dirs:
        img_dir = exp_dir / "img"
        gt_dir = exp_dir / "TRA"

        print(f"\n  [{exp_dir.name}] tracking … ", end="", flush=True)
        t0 = time.time()
        try:
            track_graph, masks = model.track_from_disk(
                imgs_path=img_dir,
                masks_path=gt_dir,
                mode="greedy",
                max_distance=args.max_distance,
            )
            print(f"done ({time.time() - t0:.1f}s) "
                  f"graph: {track_graph.number_of_nodes()} nodes, "
                  f"{track_graph.number_of_edges()} edges")
        except Exception as exc:
            print(f"FAILED: {exc}")
            results[exp_dir.name] = {"error": str(exc)}
            continue

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from trackastra.tracking.utils import graph_to_ctc
            graph_to_ctc(track_graph, masks, outdir=Path(tmpdir))

            print(f"  [{exp_dir.name}] metrics … ", end="", flush=True)
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

                results[exp_dir.name] = {
                    "TRA": float(tra) if tra is not None else None,
                    "AOGM": float(aogm) if aogm is not None else None,
                }
            except Exception as exc:
                print(f"FAILED: {exc}")
                results[exp_dir.name] = {"error": str(exc)}

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    valid = [r for r in results.values() if "error" not in r]
    if valid:
        tra_vals = [r["TRA"] for r in valid if r.get("TRA") is not None]
        aogm_vals = [r["AOGM"] for r in valid if r.get("AOGM") is not None]

        summary = {
            "n_sequences": len(valid),
            "n_failed": len(results) - len(valid),
            "mean_TRA": float(np.mean(tra_vals)) if tra_vals else None,
            "std_TRA": float(np.std(tra_vals)) if tra_vals else None,
            "mean_AOGM": float(np.mean(aogm_vals)) if aogm_vals else None,
            "std_AOGM": float(np.std(aogm_vals)) if aogm_vals else None,
        }

        print(f"\n{'=' * 60}")
        print(f"  SUMMARY ({len(valid)}/{len(results)} experiments)")
        print(f"{'=' * 60}")
        print(f"  mean TRA:  {summary['mean_TRA']:.4f} ± {summary['std_TRA']:.4f}")
        print(f"  mean AOGM: {summary['mean_AOGM']:.1f} ± {summary['std_AOGM']:.1f}")

        # Save summary alongside results
        summary_path = Path(args.output).with_name("summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {args.output}  +  {summary_path}")


if __name__ == "__main__":
    main()
