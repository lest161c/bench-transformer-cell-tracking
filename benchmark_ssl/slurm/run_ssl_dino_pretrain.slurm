#!/bin/bash
#SBATCH --account=p_scads_celltracking
#SBATCH --job-name=ssl_dino_pt
#SBATCH --output=logs/slurm-ssl_dino-%j.out
#SBATCH --error=logs/slurm-ssl_dino-%j.err
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
unset PYTHONHOME

TRK=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra
ENV_DIR="$TRK/.venv"
CFG_DIR="$SLURM_SUBMIT_DIR/configs"

# --- Quick test mode ---
if [ "$TEST" = "1" ]; then
    echo "=== TEST MODE: env setup only ==="
    . "$ENV_DIR/bin/activate"
    python -m pip install setuptools
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib scikit-learn
    cd "$TRK"
    git pull origin cached-dist-attn
    python -m pip install -e .
    python -c "
import trackastra; from trackastra.model import TrackingTransformer; print('trackastra OK');
import torch; print('CUDA:', torch.cuda.is_available());
from trackastra.model.dino_encoder import DINOBackbone; print('DINO:', DINOBackbone.load());
import torch.hub; m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14'); print('DINOv2 loaded');
"
    echo "=== TEST PASSED ==="
    exit 0
fi

# --- Normal mode ---
. "$ENV_DIR/bin/activate"
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib scikit-learn
cd "$TRK"
git pull origin cached-dist-attn
python -m pip install -e .
mkdir -p logs

# --- SSL DINO pretraining ---
python -m trackastra.model.ssl_dino_trainer \
    --config "$CFG_DIR/ssl_dino_pretrain.yaml" \
    --ssl_epochs 20 \
    --ssl_batch_size 4 \
    --ssl_lr 3e-4 \
    --ssl_temperature 0.05 \
    --outdir results/runs/ssl_dino_pretrain
