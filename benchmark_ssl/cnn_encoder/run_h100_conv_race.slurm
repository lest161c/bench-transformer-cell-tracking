#!/bin/bash
#SBATCH --account=p_scads_celltracking
#SBATCH --job-name=cnn_conv_race
#SBATCH --output=logs/cnn_conv_race-%j.out
#SBATCH --error=logs/cnn_conv_race-%j.err
#SBATCH --time=24:00:00
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
#export WANDB_DISABLED=true
#export WANDB_MODE=disabled
unset PYTHONHOME

TRK=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra
BENCH=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking
ENV_DIR="$TRK/.venv"

# --- Quick test mode: sbatch --export=TEST=1 run_h100_conv_race.slurm ---
if [ "$TEST" = "1" ]; then
    echo "=== TEST MODE: env setup only ==="
    . "$ENV_DIR/bin/activate"
    python -m pip install setuptools
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
    cd "$BENCH"
    git fetch origin cached-dist-attn && git checkout cached-dist-attn && git pull origin cached-dist-attn
    mkdir -p benchmark_ssl/cnn_encoder/logs
    python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
    python -c "import sys; sys.path.insert(0, 'benchmark_ssl/cnn_encoder'); import h100_convergence; print('h100_convergence OK')"
    echo "=== TEST PASSED ==="
    exit 0
fi

# --- Normal mode ---
. "$ENV_DIR/bin/activate"
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#export WANDB_DISABLED=true
#export WANDB_MODE=disabled
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
cd "$BENCH"
git fetch origin cached-dist-attn && git checkout cached-dist-attn && git pull origin cached-dist-attn
mkdir -p logs benchmark_ssl/cnn_encoder/logs

python benchmark_ssl/cnn_encoder/h100_convergence.py \
    --data-root ../data/vanvliet \
    --checkpoint benchmark_ssl/cnn_encoder/probe/cnn_ntxent_large.pt \
    --scale large \
    --epochs 500 \
    #--no-wandb
