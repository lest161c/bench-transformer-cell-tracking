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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME

TRK=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra
ENV_DIR="$TRK/.venv"
CFG_DIR="$SLURM_SUBMIT_DIR/configs"

MODE="${MODE:-smoke}"

. "$ENV_DIR/bin/activate"

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib

cd "$TRK"
git pull origin feature/sparse-attention-gather
python -m pip install -e .
mkdir -p logs

case "$MODE" in
    smoke)
        echo "=== SMOKE TEST: import check ==="
        python -c "import trackastra; from trackastra.model import TrackingTransformer; print('trackastra OK:', TrackingTransformer); import torch; print('CUDA:', torch.cuda.is_available())"
        echo "=== SMOKE TEST PASSED ==="
        ;;
    quick_bench)
        echo "=== QUICK BENCH: 10-epoch baseline ==="
        date
        python scripts/train.py \
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
