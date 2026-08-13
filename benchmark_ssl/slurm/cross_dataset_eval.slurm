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

# Cross-dataset evaluation of Trackastra checkpoints on deepcell.
# Evaluates all 6 trained checkpoints (trained on vanvliet) on the held-out
# deepcell dataset. Computes TRA, cHOTA, AOGM per sequence.

source "$(dirname "$0")/cluster_env.sh"

python -m src.cnn_encoder.cross_dataset_eval --device cuda
EVAL_EXIT=$?
echo "=== Evaluation finished with exit code $EVAL_EXIT ==="
echo "  Results: $TRK/results/cross_dataset/deepcell/summary.json"
