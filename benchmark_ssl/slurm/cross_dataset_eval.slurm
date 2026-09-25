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

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

cd "$BENCH/benchmark_ssl" || exit 1
mkdir -p logs

.venv/bin/python -m src.cnn_encoder.cross_dataset_eval --device cuda
EVAL_EXIT=$?
echo "=== Evaluation finished with exit code $EVAL_EXIT ==="
echo "  Results: $TRK/results/cross_dataset/deepcell/summary.json"

# ── Optional: multi-seed K-sweep checkpoints on DeepCell (commented) ─────────
# After the SPEC 0008 D2 sweep completes, generate a checkpoints JSON
# (list of [run_dir_name, label] pairs, e.g. ["2026-09-22_10-00-00_k16_s42",
# "k16_s42"]) and pass --checkpoints-file to cross_dataset_eval.py:
#   .venv/bin/python -m src.cnn_encoder.cross_dataset_eval \
#       --device cuda --checkpoints-file "$BENCH/knn_sweep_checkpoints.json"
# Appended entries are evaluated with the same protocol and land in the same
# output tree ($TRK/results/cross_dataset/deepcell/<label>/).
