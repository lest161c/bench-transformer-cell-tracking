#!/usr/bin/env python3
"""Unified edge-probing benchmark for cell tracking.

Tests how well different per-cell feature representations predict
cell-cell associations (edges) between consecutive frames using
linear and MLP probes.

Feature types (registered in ``harness.registry``):
  rp        -- Regionprops 7D (wrfeat-style)              [7 dim]
  rp_fourier -- Regionprops 7D + Fourier PE of positions  [55 dim]
  cnn_frozen -- ScaledCNN with NT-Xent checkpoint         [128 dim]
  cnn_e2e   -- ScaledCNN trained jointly with probe       [128 dim]
  dino      -- DINOv2 (dinov2_vits14, frozen)             [384 dim]
  hoct19    -- HOCT-style 13D (2D-adapted from 19D)       [13 dim]
  hoct19_fourier -- HOCT 13D + Fourier PE of positions    [43 dim]

Probe architectures (registered in ``harness.registry``):
  Linear: nn.Linear(2*feat_dim, 1) over concat(feat_anchor, feat_query)
  MLP:    nn.Sequential(Linear(2*feat_dim, 128), ReLU, Linear(128, 1))

Usage:
  # test all features with both probes, 5-fold CV
  python -m src.edge_probing.unified_edge_probe \\
      --features all --probe both --cv-folds 5 --epochs 200

  # quick test: only regionprops, linear probe
  python -m src.edge_probing.unified_edge_probe \\
      --features rp --probe linear --epochs 50

  # save results to JSON
  python -m src.edge_probing.unified_edge_probe \\
      --features all --probe both --output results/probe_results.json
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.edge_probing.harness.evaluation import (
    FEATURE_CONFIGS,
    evaluate_feature,
)
from src.edge_probing.harness.feature_extractors import PROJECT_ROOT, device
from src.edge_probing.harness.frame_data import scan_consecutive_pairs
from src.edge_probing.harness.results import print_results_table, save_results_json

logger = logging.getLogger("unified_edge_probe")

# Default checkpoint for cnn_frozen
DEFAULT_CHECKPOINT = str(PROJECT_ROOT / "results" / "checkpoints" / "cnn" / "cnn_ntxent_large.pt")


def parse_features(features_arg, available_features):
    """Parse --features argument into list of feature keys.

    Args:
        features_arg: Comma-separated feature keys or an alias ("all",
            "shallow", "deep").
        available_features: List of all valid feature keys.

    Returns:
        List of feature keys to evaluate.
    """
    from src.edge_probing.harness.registry import FEATURE_ALIASES
    if features_arg in FEATURE_ALIASES:
        return list(FEATURE_ALIASES[features_arg])
    keys = [k.strip() for k in features_arg.split(",")]
    for key in keys:
        if key not in available_features:
            logger.warning(f"Unknown feature: {key}. Available: {available_features}")
    return [key for key in keys if key in available_features]


def parse_args(argv=None):
    """Parse command-line arguments for the unified edge probing benchmark."""
    from src.edge_probing.harness.registry import list_features
    available_features = list_features()

    parser = argparse.ArgumentParser(
        description="Unified edge probing benchmark for cell tracking"
    )
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT.parent / "data" / "vanvliet"),
        help="Path to vanvliet data root",
    )
    parser.add_argument("--conditions", default="rpsM,recA,pheA,metA,cib,trpL",
                   help="Comma-separated conditions to scan")
    parser.add_argument("--train-conditions", default="rpsM,recA,pheA,metA",
                   help="Conditions for training")
    parser.add_argument("--val-conditions", default="cib,trpL",
                   help="Conditions for validation")
    parser.add_argument("--features", default="rp",
                   help=f"Features to test. Choices: {available_features}, "
                        f"or aliases: all, shallow, deep")
    parser.add_argument("--probe", default="both", choices=["linear", "mlp", "both"],
                   help="Probe architecture to use")
    parser.add_argument("--max-pairs", type=int, default=30,
                   help="Max consecutive frame pairs per condition")
    parser.add_argument("--epochs", type=int, default=200,
                   help="Training epochs per probe")
    parser.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate for linear probe (MLP uses lr/10)")
    parser.add_argument("--batch-size-linear", type=int, default=256,
                   help="Batch size for linear probe")
    parser.add_argument("--batch-size-mlp", type=int, default=128,
                   help="Batch size for MLP probe")
    parser.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience (epochs)")
    parser.add_argument("--eval-every", type=int, default=1,
                   help="Evaluate every N epochs")
    parser.add_argument(
        "--checkpoint", type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to NT-Xent checkpoint for cnn_frozen",
    )
    parser.add_argument("--output", type=str, default="",
                   help="Path to save JSON results (default: no save)")
    parser.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    parser.add_argument("--no-cache", action="store_true",
                   help="Skip loading/saving feature cache")
    parser.add_argument("--cv-folds", type=int, default=0,
                   help="Number of CV folds (0 = use original train/val split)")
    parser.add_argument("--shuffle-baseline", action="store_true",
                   help="Also run on shuffled features as overfitting baseline")
    parser.add_argument("--val-frac", type=float, default=0.2,
                   help="Validation fraction for non-CV mode (used in random split fallback)")
    parser.add_argument("--standardize", dest="standardize", action="store_true",
                   default=True,
                   help="Z-score standardize frozen features per feature dim, "
                        "fit on the training split only (HOCT-paper default: on)")
    parser.add_argument("--no-standardize", dest="standardize", action="store_false",
                   help="Disable feature standardization (legacy behavior)")
    return parser.parse_args(argv)


def main():
    """Run the unified edge probing benchmark for the requested feature types."""
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Resolve checkpoint
    checkpoint_path = args.checkpoint
    if not Path(checkpoint_path).exists():
        logger.warning(f"Checkpoint not found at {checkpoint_path}")
        checkpoint_path = None

    # Log configuration
    conditions = [c.strip() for c in args.conditions.split(",")]
    features_to_run = parse_features(args.features, list(FEATURE_CONFIGS.keys()))

    logger.info("=" * 60)
    logger.info("Unified Edge Probing Benchmark")
    logger.info("=" * 60)
    logger.info(f"  Data root:      {args.data_root}")
    logger.info(f"  Conditions:     {conditions}")
    logger.info(f"  Train:          {args.train_conditions}")
    logger.info(f"  Val:            {args.val_conditions}")
    logger.info(f"  Features:       {features_to_run}")
    logger.info(f"  Probe:          {args.probe}")
    logger.info(f"  Epochs:         {args.epochs}")
    logger.info(f"  LR:             {args.lr}")
    logger.info(f"  Batch linear:   {args.batch_size_linear}")
    logger.info(f"  Batch MLP:      {args.batch_size_mlp}")
    logger.info(f"  Device:         {device}")
    logger.info(f"  Standardize:    {args.standardize}")
    logger.info("=" * 60)

    # Scan pairs
    logger.info("Scanning consecutive frame pairs...")
    all_pairs = scan_consecutive_pairs(args.data_root, conditions, args.max_pairs)
    logger.info(f"Found {len(all_pairs)} consecutive frame pairs")

    # Log condition breakdown
    cond_counts = Counter(pair[5] for pair in all_pairs)
    for cond, count in sorted(cond_counts.items()):
        logger.info(f"  {cond}: {count} pairs")

    if len(all_pairs) < 3:
        logger.error("Need at least 3 consecutive frame pairs. Check --data-root.")
        sys.exit(1)

    # Run each feature type
    all_results = {}
    for fkey in features_to_run:
        try:
            result = evaluate_feature(fkey, all_pairs, args, checkpoint_path)
            all_results[fkey] = result
        except Exception as exc:
            logger.error(f"Error running {fkey}: {exc}", exc_info=True)
            all_results[fkey] = None

    # Print results
    rows = print_results_table(all_results)

    # Save results
    if args.output:
        save_results_json(all_results, args, args.output)

    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
