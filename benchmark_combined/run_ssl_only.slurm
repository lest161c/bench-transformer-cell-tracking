#!/bin/bash
#SBATCH --account=p_scads_celltracking
#SBATCH --job-name=ssl_only
#SBATCH --output=logs/slurm-ssl_only-%j.out
#SBATCH --error=logs/slurm-ssl_only-%j.err
#SBATCH --time=2:00:00
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

eval "$(conda shell.bash hook)"
TRK=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra
ENV_DIR="$TRK/trackastra_env"

# --- Quick test mode ---
if [ "$TEST" = "1" ]; then
    echo "=== TEST MODE: env setup only ==="
    conda activate "$ENV_DIR" 2>/dev/null || {
        echo "Creating env..."
        conda create --prefix "$ENV_DIR" python=3.10 -y
        conda activate "$ENV_DIR"
    }
    python -m pip install setuptools
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
    cd "$TRK"
    git pull origin feature/sparse-attention-gather
    python -m pip install -e .
    python -c "import trackastra; from trackastra.model import TrackingTransformer; print('trackastra OK:', TrackingTransformer); import torch; print('CUDA:', torch.cuda.is_available())"
    echo "=== TEST PASSED ==="
    exit 0
fi

# --- Normal mode ---
conda activate "$ENV_DIR" 2>/dev/null || {
    echo "Creating env..."
    conda create --prefix "$ENV_DIR" python=3.10 -y
    conda activate "$ENV_DIR"
}
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
export WANDB_DISABLED=true
export WANDB_MODE=disabled
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
cd "$TRK"
git pull origin feature/sparse-attention-gather
python -m pip install -e .
mkdir -p logs

python scripts/train.py \
    --config benchmark_combined/vanvliet_sparse_k16_ssl.yaml \
    --ssl_pretrain True --ssl_only True --epochs 0
