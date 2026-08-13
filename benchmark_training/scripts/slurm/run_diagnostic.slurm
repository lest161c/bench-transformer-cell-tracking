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

source "$(dirname "${BASH_SOURCE[0]}")/../../../slurm/load_cluster_env.sh"

CFG_DIR="$SLURM_SUBMIT_DIR/configs"

MODE="${MODE:-smoke}"

# -----------------------------------------------------------------------
# Sync project dependencies via uv
# -----------------------------------------------------------------------
# $SLURM_SUBMIT_DIR is the benchmark_training/ checkout. `uv sync` creates
# .venv/ and installs all deps from pyproject.toml (torch+torchvision from
# the cu130 index, lightning, configargparse, wandb, etc.).
cd "$SLURM_SUBMIT_DIR"
uv sync

# -----------------------------------------------------------------------
# Pull trackastra branch and install in editable mode
# -----------------------------------------------------------------------
cd "$TRK"
git pull origin feature/sparse-attention-gather
uv pip install --project "$SLURM_SUBMIT_DIR" -e "$TRK"
mkdir -p logs

case "$MODE" in
    smoke)
        echo "=== SMOKE TEST: import check ==="
        uv run --no-sync --project "$SLURM_SUBMIT_DIR" python -c "import trackastra; from trackastra.model import TrackingTransformer; print('trackastra OK:', TrackingTransformer); import torch; print('CUDA:', torch.cuda.is_available())"
        echo "=== SMOKE TEST PASSED ==="
        ;;
    quick_bench)
        echo "=== QUICK BENCH: 10-epoch baseline ==="
        date
        uv run --no-sync --project "$SLURM_SUBMIT_DIR" python scripts/train.py \
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
