#!/bin/bash
#SBATCH --account=p_scads_celltracking
#SBATCH --job-name=cnn_viz
#SBATCH --output=logs/cnn_viz-%j.out
#SBATCH --error=logs/cnn_viz-%j.err
#SBATCH --time=02:00:00
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
BENCH=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking
ENV_DIR="$TRK/.venv"

. "$ENV_DIR/bin/activate"
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
cd "$BENCH"
git fetch origin cached-dist-attn && git checkout cached-dist-attn && git pull origin cached-dist-attn
mkdir -p logs src/cnn_encoder/logs

python -m src.cnn_encoder.visualize_embeddings \
    --data-root ../data/vanvliet \
    --checkpoint results/checkpoints/cnn/cnn_ntxent_large.pt \
    --scale large \
    --ckpt-dir checkpoints/h100_conv_race \
    --max-pairs 50
