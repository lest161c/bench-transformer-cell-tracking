#!/bin/bash
#SBATCH --account=p_scads_celltracking
#SBATCH --job-name=tr_bench_cd
#SBATCH --output=logs/slurm-quickbench-cd-%j.out
#SBATCH --error=logs/slurm-quickbench-cd-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G

export PIP_REQUIRE_VIRTUALENV=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export WANDB_DISABLED=true
export WANDB_MODE=disabled
unset PYTHONHOME

TRK=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra
ENV_DIR="$TRK/.venv"
CFG_DIR="$SLURM_SUBMIT_DIR/benchmark_combined"

if [ "$TEST" = "1" ]; then
    echo "=== TEST MODE ==="
    . "$ENV_DIR/bin/activate"
    python -m pip install setuptools
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
    cd "$TRK"
    git pull origin cached-dist-attn
    python -m pip install -e .
    python -c "import trackastra; print('OK'); import torch; print('CUDA:', torch.cuda.is_available())"
    echo "=== TEST PASSED ==="
    exit 0
fi

. "$ENV_DIR/bin/activate"
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
export WANDB_DISABLED=true
export WANDB_MODE=disabled
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
cd "$TRK"
git pull origin cached-dist-attn
python -m pip install -e .
mkdir -p logs

echo "=== CACHED-DIST BENCH (cached-dist-attn) ==="
date

python scripts/train.py \
    --config "$CFG_DIR/vanvliet_baseline.yaml" \
    --epochs 10 \
    --name quick_bench_cached

echo "=== DONE ==="
date
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,State,MaxRSS --noheader
