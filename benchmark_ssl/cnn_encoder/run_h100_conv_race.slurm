#!/bin/bash
#SBATCH --account=p_scads_celltracking
#SBATCH --partition=gpu-h100
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --job-name=cnn_conv_race
#SBATCH --output=benchmark_ssl/cnn_encoder/logs/cnn_conv_race-%j.out
#SBATCH --error=benchmark_ssl/cnn_encoder/logs/cnn_conv_race-%j.err

export PIP_REQUIRE_VIRTUALENV=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
# WANDB disabled by default — set WANDB_API_KEY env or pass --no-wandb to enable
export WANDB_DISABLED=true

WS=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601
TRK=$WS/trackastra
BENCH=$WS/bench-transformer-cell-tracking
ENV_DIR="$TRK/.venv"

# --- Quick test mode: sbatch --export=TEST=1 run_h100_conv_race.slurm ---
if [ "$TEST" = "1" ]; then
    echo "=== TEST MODE: env setup only ==="
    . "$ENV_DIR/bin/activate"
    python -m pip install setuptools
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    python -m pip install pandas scikit-image tifffile wandb scikit-learn
    cd "$BENCH" && git fetch origin cached-dist-attn && git checkout cached-dist-attn && git pull origin cached-dist-attn
    python -c "
import torch; print('CUDA:', torch.cuda.is_available());
import wandb; print('wandb OK');
from skimage.measure import regionprops; print('skimage OK');
"
    echo "=== TEST PASSED ==="
    exit 0
fi

# --- Normal mode ---
. "$ENV_DIR/bin/activate"
python -m pip install --upgrade setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install pandas scikit-image tifffile wandb scikit-learn

# Update benchmark repo to cached-dist-attn branch (which has h100 files)
cd "$BENCH"
git fetch origin cached-dist-attn
git checkout cached-dist-attn
git pull origin cached-dist-attn
mkdir -p benchmark_ssl/cnn_encoder/logs checkpoints

# --- Run H100 convergence race ---
python benchmark_ssl/cnn_encoder/h100_convergence.py \
    --data-root /data/cat/ws/mawe985g-data/data/celltracking/vanvliet \
    --checkpoint benchmark_ssl/cnn_encoder/probe/cnn_ntxent.pt \
    --epochs 500 \
    --lr 1e-4 \
    --patience 50 \
    --pos-weight 100.0 \
    --seed 42 \
    --no-wandb

echo "=== H100 convergence race complete ==="
