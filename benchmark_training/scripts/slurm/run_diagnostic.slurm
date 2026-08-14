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

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

PY="$REPO_ROOT/benchmark_training/.venv/bin/python"

case "$MODE" in
    smoke)
        echo "=== SMOKE TEST: import check ==="
        $PY -c "import trackastra; from trackastra.model import TrackingTransformer; print('trackastra OK:', TrackingTransformer); import torch; print('CUDA:', torch.cuda.is_available())"
        echo "=== SMOKE TEST PASSED ==="
        ;;
    quick_bench)
        echo "=== QUICK BENCH: 10-epoch baseline ==="
        date
        $PY scripts/train.py \
            --config "$REPO_ROOT/benchmark_training/configs/vanvliet_baseline.yaml" \
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
