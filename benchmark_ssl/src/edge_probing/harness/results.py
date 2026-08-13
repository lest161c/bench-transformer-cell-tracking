"""Results display and JSON serialization.

Pretty-prints the per-feature/per-probe results table to stdout and
serializes the results dict to JSON (excluding per-epoch training
histories) for downstream consumption.
"""

import json
import logging
from pathlib import Path

from src.edge_probing.harness.evaluation import FEATURE_CONFIGS

logger = logging.getLogger("edge_probing.results")

EM_DASH = '\u2014'


def print_results_table(all_results):
    """Print formatted results table. Handles both CV and non-CV results.

    Args:
        all_results: dict mapping feature key -> {probe key -> metrics dict},
            where a feature that was skipped maps to None.

    Returns:
        List of per-probe row dicts (feature display name, probe, metrics);
        used by the orchestrator for the JSON "rows" section.
    """
    # Auto-detect CV mode
    is_cv_mode = any(
        result is not None
        and any(isinstance(metrics, dict) and metrics.get("is_cv", False) for metrics in result.values())
        for result in all_results.values()
    )

    print()
    print("=" * 120)
    print("Unified Edge Probing Benchmark \u2014 Results")
    print("=" * 120)

    if is_cv_mode:
        header = (f"{'Feature':<28} {'Probe':<8} {'BalAcc (mean±std)':<22} "
                  f"{'F1 (mean±std)':<22} {'Status':<10}")
    else:
        header = (f"{'Feature':<25} {'Probe':<8} {'BalAcc':<10} {'F1':<10} "
                  f"{'Precision':<10} {'Recall':<10} {'Status':<10}")
    print(header)
    print("-" * 120)

    rows = []
    has_shuffled = False

    for fkey, cfg in FEATURE_CONFIGS.items():
        if fkey not in all_results or all_results[fkey] is None:
            if is_cv_mode:
                print(f"{cfg['display']:<28} {EM_DASH:<8} {EM_DASH:<22} {EM_DASH:<22} {'SKIP':<10}")
            else:
                print(f"{cfg['display']:<25} {EM_DASH:<8} {EM_DASH:<10} {EM_DASH:<10} "
                      f"{EM_DASH:<10} {EM_DASH:<10} {'SKIP':<10}")
            continue

        result = all_results[fkey]
        for ptype in ["linear", "mlp"]:
            if ptype not in result or result[ptype] is None:
                continue
            metrics = result[ptype]

            if metrics.get("is_cv", False):
                print(f"{cfg['display']:<28} {ptype:<8} "
                      f"{metrics['bal_acc_mean']:.4f}\u00b1{metrics['bal_acc_std']:<.4f}        "
                      f"{metrics['f1_mean']:.4f}\u00b1{metrics['f1_std']:<.4f}        "
                      f"{'OK':<10}")
                rows.append({
                    "feature": cfg['display'],
                    "probe": ptype,
                    "bal_acc": metrics['bal_acc_mean'],
                    "bal_acc_std": metrics['bal_acc_std'],
                    "f1": metrics['f1_mean'],
                    "f1_std": metrics['f1_std'],
                    "is_cv": True,
                })
            else:
                print(f"{cfg['display']:<25} {ptype:<8} "
                      f"{metrics['final_bal_acc']:<10.4f} {metrics['final_f1']:<10.4f} "
                      f"{metrics['final_precision']:<10.4f} {metrics['final_recall']:<10.4f} "
                      f"{'OK':<10}")
                rows.append({
                    "feature": cfg['display'],
                    "probe": ptype,
                    "bal_acc": metrics['final_bal_acc'],
                    "f1": metrics['final_f1'],
                    "precision": metrics['final_precision'],
                    "recall": metrics['final_recall'],
                })

        # Check for shuffled results
        for ptype in ["linear_shuffled", "mlp_shuffled"]:
            if ptype in result and result[ptype] is not None:
                has_shuffled = True

    # Print shuffled baseline section
    if has_shuffled:
        print("--- shuffled baseline ---")
        for fkey, cfg in FEATURE_CONFIGS.items():
            if fkey not in all_results or all_results[fkey] is None:
                continue
            result = all_results[fkey]
            for ptype, label in [("linear_shuffled", "linear"), ("mlp_shuffled", "mlp")]:
                if ptype not in result or result[ptype] is None:
                    continue
                metrics = result[ptype]
                display_name = f"{cfg['display']} (shuffled)"
                print(f"{display_name:<28} {label:<8} "
                      f"{metrics['bal_acc_mean']:.4f}\u00b1{metrics['bal_acc_std']:<.4f}        "
                      f"{metrics['f1_mean']:.4f}\u00b1{metrics['f1_std']:<.4f}        "
                      f"{'SHUF':<10}")

    print("-" * 120)
    print()

    # Best performer
    if rows:
        if is_cv_mode:
            best = max(rows, key=lambda row: row["bal_acc"])
            print(f"Best performer: {best['feature']} + {best['probe']} "
                  f"(bal_acc={best['bal_acc']:.4f}\u00b1{best['bal_acc_std']:.4f}, "
                  f"f1={best['f1']:.4f}\u00b1{best['f1_std']:.4f})")
        else:
            best = max(rows, key=lambda row: row["bal_acc"])
            print(f"Best performer: {best['feature']} + {best['probe']} "
                  f"(bal_acc={best['bal_acc']:.4f}, f1={best['f1']:.4f})")
        print()

    return rows


def save_results_json(all_results, args, output_path):
    """Write benchmark results to a JSON file with consistent formatting.

    Args:
        all_results: Dict mapping feature_key -> {probe_key -> metrics}
            or None for skipped features.
        args: The argparse Namespace from ``parse_args()``.
        output_path: Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for fkey, result in all_results.items():
        if result is None:
            serializable[fkey] = None
        else:
            serializable[fkey] = {}
            for ptype, metrics in result.items():
                if ptype in ("linear", "mlp"):
                    serializable[fkey][ptype] = {
                        key: value for key, value in metrics.items() if key != "history"
                    }

    rows = []  # Re-compute rows from serializable? Or return them separately?
    # For now, write what we have. The orchestrator can pass rows separately.
    with open(output_path, "w") as file_handle:
        json.dump({
            "args": vars(args),
            "results": serializable,
        }, file_handle, indent=2)
    logger.info(f"Results saved to {output_path}")
