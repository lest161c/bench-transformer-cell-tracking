#!/bin/bash
#SBATCH --job-name=eval_dense_flash
#SBATCH --account=p_scads_celltracking
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=logs/eval_dense_flash-%j.out
#SBATCH --error=logs/eval_dense_flash-%j.err
#
# Evaluate the dense_flash (no spatial cutoff) checkpoint on vanvliet
# validation data. Computes TRA and AOGM metrics.
#
# The dense_flash model uses knn_neighbors=-2 (DenseFlashAttention),
# which removes the spatial cutoff mask and enables FlashAttention-2.
# This eval answers: does removing the spatial cutoff degrade TRA?
#
# ─── Usage ──────────────────────────────────────────────────────────
#
#   sbatch scripts/slurm/run_eval_dense_flash.slurm
#
# -----------------------------------------------------------------------

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

cd "$TRK"

# The dense_flash run directory
RUN_DIR="$TRK/runs/2026-08-17_12-03-17_dense_flash_no_cutoff"

if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: Run directory not found: $RUN_DIR"
    exit 1
fi

echo "=== Evaluating dense_flash checkpoint ==="
echo "  Run dir: $RUN_DIR"
echo "  Device:  cuda"
echo "  Started: $(date)"
echo ""

PY="$TRK/.venv/bin/python"

$PY "$BENCH/benchmark_training/scripts/eval_validation_conditions.py" \
    --run-dir "$RUN_DIR" \
    --device cuda

EXIT_CODE=$?
echo ""
echo "  Eval exited with code ${EXIT_CODE} at $(date)"
exit ${EXIT_CODE}
