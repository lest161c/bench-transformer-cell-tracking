#!/bin/bash
#SBATCH --account=p_scads_celltracking
#SBATCH --job-name=cross_dataset_eval
#SBATCH --output=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking/logs/cross_dataset_eval-%j.out
#SBATCH --error=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking/logs/cross_dataset_eval-%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G

# =============================================================================
# Cross-dataset evaluation of Trackastra checkpoints on deepcell
#
# Evaluates all 6 trained checkpoints (trained on vanvliet) on the held-out
# deepcell dataset. Computes TRA, cHOTA, AOGM per sequence.
# =============================================================================

set -euo pipefail

export PIP_REQUIRE_VIRTUALENV=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME

TRK=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra
BENCH=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking
ENV_DIR="$TRK/.venv"
LOG_DIR="$BENCH/logs"

# Verify required paths exist
for p in "$TRK" "$BENCH" "$ENV_DIR/bin/activate"; do
    if [ ! -e "$p" ]; then
        echo "FATAL: Required path not found: $p"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Quick test mode: sbatch --export=TEST=1 run_cross_dataset_eval.slurm
# ---------------------------------------------------------------------------
if [ "${TEST:-0}" = "1" ]; then
    echo "=== TEST MODE: env setup only ==="
    . "$ENV_DIR/bin/activate"
    python -c "import trackastra; from trackastra.model import TrackingTransformer; print('trackastra OK:', TrackingTransformer); import torch; print('CUDA:', torch.cuda.is_available())"
    python -c "import traccuracy; print('traccuracy:', traccuracy.__version__)"
    echo "=== TEST PASSED ==="
    exit 0
fi

# ---------------------------------------------------------------------------
# Normal mode: activate env, install dependencies, run evaluation
# ---------------------------------------------------------------------------
echo "=== Setting up environment ==="
. "$ENV_DIR/bin/activate"

# Upgrade setuptools once (needed for editable installs)
python -m pip install --upgrade setuptools

# fast-regionprops is already installed in the environment
# Install trackastra from source (editable) if not already installed
if ! python -c "import trackastra" 2>/dev/null; then
    echo "Installing trackastra from $TRK ..."
    python -m pip install -e "$TRK"
else
    echo "trackastra already installed"
fi

# Install traccuracy from local development version (editable)
TRAcc_DIR="$BENCH/eval_traccuracy/traccuracy"
if [ -d "$TRAcc_DIR" ]; then
    echo "Installing traccuracy from $TRAcc_DIR ..."
    python -m pip install -e "$TRAcc_DIR"
else
    echo "FATAL: traccuracy directory not found at $TRAcc_DIR"
    exit 1
fi

mkdir -p "$LOG_DIR"

echo "=== Starting cross-dataset evaluation ==="
echo "  Checkpoints to evaluate: 6"
echo "  Dataset: deepcell (test split, 12 sequences)"
echo "  Metrics: TRA, cHOTA, AOGM"
echo "  Output: $TRK/results/cross_dataset/deepcell/"
echo ""

# Run the evaluation script
python "$BENCH/benchmark_ssl/cnn_encoder/cross_dataset_eval.py" --device cuda

EVAL_EXIT=$?
echo ""
echo "=== Evaluation finished with exit code $EVAL_EXIT ==="
echo "  Results: $TRK/results/cross_dataset/deepcell/summary.json"
