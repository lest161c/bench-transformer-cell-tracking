#!/usr/bin/env python3
"""
Cross-dataset evaluation of Trackastra checkpoints on the deepcell dataset.

Evaluates all 6 trained checkpoints (trained on vanvliet) on the held-out
deepcell dataset. Computes TRA, cHOTA, and AOGM metrics per sequence and
aggregated across all 12 test sequences.

Usage:
    python cross_dataset_eval.py [--device cuda|cpu] [--dry-run]

Output:
    results/cross_dataset/deepcell/{variant}/{seq}/  ← CTC-format predictions
    results/cross_dataset/deepcell/{variant}/metrics.json   ← per-sequence metrics
    results/cross_dataset/deepcell/summary.json             ← aggregated results
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
TRK = Path("/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra")
BENCH = Path("/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking")
DATA_ROOT = Path("/data/cat/ws/mawe985g-data/data/celltracking/deepcell")
OUTPUT_ROOT = TRK / "results" / "cross_dataset" / "deepcell"

# The 6 checkpoints to evaluate (run_dir_name, short_label)
CHECKPOINTS = [
    ("2026-06-17_01-32-30_vanvliet_baseline",   "baseline"),
    ("2026-07-21_17-43-05_vanvliet_cnn_concat",  "cnn_concat"),
    ("2026-07-19_21-44-17_vanvliet_cnn_dropout", "cnn_dropout"),
    ("2026-07-20_09-04-06_vanvliet_cnn_both",    "cnn_both"),
    ("2026-07-11_20-28-00_vanvliet_baseline_cnn","baseline_cnn"),
    ("2026-07-20_03-32-50_vanvliet_cnn_trainable","cnn_trainable"),
]

# Sequences 00–11 are the deepcell test split (held-out, never used in training)
TEST_SEQS = sorted(
    d.name for d in (DATA_ROOT / "test").iterdir() if d.is_dir()
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit")
    args = parser.parse_args()

    # ── Lazy imports (heavy) ──────────────────────────────────────────────
    # Packages are installed in the venv
    from trackastra.model import Trackastra
    from trackastra.tracking.utils import graph_to_ctc
    from traccuracy import run_metrics
    from traccuracy.loaders import load_ctc_data
    from traccuracy.matchers import CTCMatcher
    from traccuracy.metrics import CTCMetrics, CHOTAMetric
    # CTCMetrics already includes AOGM as part of its output

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for ckpt_dir, variant in CHECKPOINTS:
        model_dir = TRK / "runs" / ckpt_dir
        print(f"\n{'=' * 80}")
        print(f"  [{variant}]  {ckpt_dir}")
        print(f"{'=' * 80}")

        if args.dry_run:
            print(f"  Would load model from: {model_dir}")
            for seq in TEST_SEQS:
                img_dir = DATA_ROOT / "test" / seq / "img"
                mask_dir = DATA_ROOT / "test" / seq / "CALIBAN"
                out_dir = OUTPUT_ROOT / variant / seq
                gt_dir = DATA_ROOT / "test" / seq / "TRA"
                print(f"    {seq}: img={img_dir}  masks={mask_dir}  gt={gt_dir}  → {out_dir}")
            print()
            continue

        # ── Load model ────────────────────────────────────────────────────
        print(f"  Loading model … ", end="", flush=True)
        t0 = time.time()
        model = Trackastra.from_folder(model_dir, device=args.device)
        print(f"done  ({time.time() - t0:.1f}s)")

        variant_results = {}

        for seq in TEST_SEQS:
            seq_dir = DATA_ROOT / "test" / seq
            img_dir = seq_dir / "img"
            mask_dir = seq_dir / "CALIBAN"
            gt_dir = seq_dir / "TRA"
            out_dir = OUTPUT_ROOT / variant / seq

            # ── Run tracking ──────────────────────────────────────────────
            print(f"  [{seq}] tracking … ", end="", flush=True)
            t0 = time.time()
            try:
                track_graph, masks = model.track_from_disk(
                    imgs_path=img_dir,
                    masks_path=mask_dir,
                    mode="greedy",
                )
            except Exception as e:
                print(f"FAILED ({e})")
                variant_results[seq] = {"error": str(e)}
                continue
            print(f"done  ({time.time() - t0:.1f}s)  graph has {track_graph.number_of_nodes()} nodes, {track_graph.number_of_edges()} edges", end="")

            # ── Save CTC output ───────────────────────────────────────────
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            graph_to_ctc(track_graph, masks, outdir=out_dir)
            print(f"  saved  ({time.time() - t0:.1f}s)")

            # ── Compute metrics ───────────────────────────────────────────
            print(f"  [{seq}] metrics … ", end="", flush=True)
            t0 = time.time()
            try:
                gt_data = load_ctc_data(str(gt_dir))
                pred_data = load_ctc_data(str(out_dir))

                # TRA + AOGM (CTCMetrics includes both)
                ctc_result, _ = run_metrics(
                    gt_data, pred_data, CTCMatcher(), [CTCMetrics()]
                )
                tra = ctc_result[0]["results"].get("TRA", None)
                aogm = ctc_result[0]["results"].get("AOGM", None)

                # cHOTA
                chota_result, _ = run_metrics(
                    gt_data, pred_data, CTCMatcher(), [CHOTAMetric()]
                )
                chota = chota_result[0]["results"].get("CHOTA", None)

                print(f"done  ({time.time() - t0:.1f}s)  "
                      f"TRA={tra:.4f}  cHOTA={chota:.4f}  AOGM={aogm:.4f}")

                variant_results[seq] = {
                    "TRA": tra,
                    "cHOTA": chota,
                    "AOGM": aogm,
                }

            except Exception as e:
                print(f"FAILED ({e})")
                variant_results[seq] = {"error": str(e)}

        # ── Save per-variant metrics ──────────────────────────────────────
        (OUTPUT_ROOT / variant).mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_ROOT / variant / "metrics.json", "w") as f:
            json.dump(variant_results, f, indent=2)

        all_results[variant] = variant_results

    # ── Summary ─────────────────────────────────────────────────────────────
    if not args.dry_run and all_results:
        print(f"\n{'=' * 80}")
        print("  SUMMARY")
        print(f"{'=' * 80}")

        summary = {}
        for variant, var_results in all_results.items():
            valid = [r for r in var_results.values() if "error" not in r]
            if not valid:
                summary[variant] = {"error": "all sequences failed"}
                print(f"  {variant:20s}  ALL FAILED")
                continue

            tra_vals = [r["TRA"] for r in valid if r.get("TRA") is not None]
            chota_vals = [r["cHOTA"] for r in valid if r.get("cHOTA") is not None]
            aogm_vals = [r["AOGM"] for r in valid if r.get("AOGM") is not None]

            summary[variant] = {
                "mean_TRA": float(np.mean(tra_vals)) if tra_vals else None,
                "std_TRA": float(np.std(tra_vals)) if tra_vals else None,
                "mean_cHOTA": float(np.mean(chota_vals)) if chota_vals else None,
                "std_cHOTA": float(np.std(chota_vals)) if chota_vals else None,
                "mean_AOGM": float(np.mean(aogm_vals)) if aogm_vals else None,
                "std_AOGM": float(np.std(aogm_vals)) if aogm_vals else None,
                "per_sequence": var_results,
            }

            print(f"  {variant:20s}  TRA={summary[variant]['mean_TRA']:.4f}  "
                  f"cHOTA={summary[variant]['mean_cHOTA']:.4f}  "
                  f"AOGM={summary[variant]['mean_AOGM']:.4f}")

        with open(OUTPUT_ROOT / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n  Results saved to: {OUTPUT_ROOT}/")

    elif args.dry_run:
        print("\n  DRY RUN — use --dry-run to see plan")
        print(f"  Sequences to process: {len(TEST_SEQS)}")
        print(f"  Checkpoints to evaluate: {len(CHECKPOINTS)}")
        print(f"  Output root: {OUTPUT_ROOT}")

    print("\nDone.")


if __name__ == "__main__":
    main()
