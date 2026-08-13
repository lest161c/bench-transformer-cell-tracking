#!/bin/bash
#SBATCH --job-name=cross_dataset_eval
#SBATCH --account=p_scads_celltracking
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=logs/cross_dataset_eval-%j.out
#SBATCH --error=logs/cross_dataset_eval-%j.err

# Config rationale: 6 of 2^5=32 factorial — baseline (control) +
# {concat, additive, both}×dropout + {vanilla, trainable} to isolate
# integration, pretraining, and freezing effects. Full factorial adds
# no independent signal.

# Cross-dataset evaluation of Trackastra checkpoints on deepcell.
# Evaluates all 6 trained checkpoints (trained on vanvliet) on the held-out
# deepcell dataset. Computes TRA, cHOTA, AOGM per sequence.

# ─── REPO_ROOT ────────────────────────────────────────────────────────
# TODO: REPLACE with the absolute path to your bench-transformer-cell-tracking
# checkout on the cluster.  This is the only cluster-specific value in the
# file — everything else below is resolved relative to it.
REPO_ROOT=/path/to/bench-transformer-cell-tracking

source "$REPO_ROOT/slurm/load_cluster_env.sh"

cd "$BENCH/benchmark_ssl" || exit 1
uv sync
mkdir -p logs

uv run python -m src.cnn_encoder.cross_dataset_eval --device cuda
EVAL_EXIT=$?
echo "=== Evaluation finished with exit code $EVAL_EXIT ==="
echo "  Results: $TRK/results/cross_dataset/deepcell/summary.json"
