#!/bin/bash
#SBATCH --job-name=eval_deepcell
#SBATCH --account=p_scads_celltracking
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=logs/eval_deepcell-%j.out
#SBATCH --error=logs/eval_deepcell-%j.err
#
# Evaluate dense_flash checkpoint on held-out deepcell test set.
# Deepcell is never used in training → true generalization measure.
#
# ─── Usage ──────────────────────────────────────────────────────────
#
#   sbatch scripts/slurm/run_eval_deepcell.slurm
#
# -----------------------------------------------------------------------

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

cd "$TRK"

RUN_DIR="$TRK/runs/2026-08-17_12-03-17_dense_flash_no_cutoff"

if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: Run directory not found: $RUN_DIR"
    exit 1
fi

echo "=== Evaluating dense_flash on deepcell ==="
echo "  Run dir: $RUN_DIR"
echo "  Started: $(date)"
echo ""

PY="$TRK/.venv/bin/python"

$PY "$BENCH/benchmark_training/scripts/eval_deepcell.py" \
    --run-dir "$RUN_DIR" \
    --device cuda

EXIT_CODE=$?
echo ""
echo "  Eval exited with code ${EXIT_CODE} at $(date)"
exit ${EXIT_CODE}
