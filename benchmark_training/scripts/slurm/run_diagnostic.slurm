#!/bin/bash
#SBATCH --job-name=tr_diag
#SBATCH --account=p_scads_celltracking
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=logs/slurm-diag-%j.out
#SBATCH --error=logs/slurm-diag-%j.err
#
# Diagnostic and quick-bench script.
#
# Config rationale: smoke validates shared env (venv, imports, CUDA);
# quick_bench estimates wall time. No per-variant diagnostics needed —
# all variants share the same venv and branch setup.
#
# ─── Usage ──────────────────────────────────────────────────────────
#
#   sbatch --export=MODE=smoke scripts/slurm/run_diagnostic.slurm
#   sbatch --export=MODE=quick_bench scripts/slurm/run_diagnostic.slurm
#
# Hardware: NVIDIA H100 80 GB (gpu-h100 partition on Capella).
# -----------------------------------------------------------------------

# ─── REPO_ROOT ────────────────────────────────────────────────────────
# TODO: REPLACE with the absolute path to your bench-transformer-cell-tracking
# checkout on the cluster.  This is the only cluster-specific value in the
# file — everything else below is resolved relative to it.
REPO_ROOT=/path/to/bench-transformer-cell-tracking

source "$REPO_ROOT/slurm/load_cluster_env.sh"

CFG_DIR="$REPO_ROOT/benchmark_training/configs"

MODE="${MODE:-smoke}"

# -----------------------------------------------------------------------
# Sync project dependencies via uv
# -----------------------------------------------------------------------
cd "$REPO_ROOT/benchmark_training"
uv sync

# -----------------------------------------------------------------------
# Pull trackastra branch and install in editable mode
# -----------------------------------------------------------------------
cd "$TRK"
git pull origin feature/sparse-attention-gather
uv pip install --project "$REPO_ROOT/benchmark_training" -e "$TRK"
mkdir -p logs

case "$MODE" in
    smoke)
        echo "=== SMOKE TEST: import check ==="
        uv run --no-sync --project "$REPO_ROOT/benchmark_training" python -c "import trackastra; from trackastra.model import TrackingTransformer; print('trackastra OK:', TrackingTransformer); import torch; print('CUDA:', torch.cuda.is_available())"
        echo "=== SMOKE TEST PASSED ==="
        ;;
    quick_bench)
        echo "=== QUICK BENCH: 10-epoch baseline ==="
        date
        uv run --no-sync --project "$REPO_ROOT/benchmark_training" python scripts/train.py \
            --config "$CFG_DIR/vanvliet_baseline.yaml" \
            --epochs 10 \
            --name quick_bench_baseline
        echo "=== DONE ==="
        date
        sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,State,MaxRSS --noheader
        ;;
    *)
        echo "ERROR: Unknown MODE '$MODE'"
        echo "Valid modes: smoke quick_bench"
        exit 1
        ;;
esac
