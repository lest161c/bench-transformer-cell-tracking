#!/bin/bash
#SBATCH --account=p_scads_celltracking
#SBATCH --job-name=full_cnn_race
#SBATCH --output=logs/full_cnn_race-%j.out
#SBATCH --error=logs/full_cnn_race-%j.err
#SBATCH --time=72:00:00
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
mkdir -p logs benchmark_ssl/cnn_encoder/logs

# ═══════════════════════════════════════════════════════════════════════════════
# RUN 1: Baseline (7D only)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  RUN 1: Baseline (7D regionprops only) — $(date)"
echo "═══════════════════════════════════════════════════════════════════════════"
python scripts/train.py --config configs/vanvliet_baseline.yaml --epochs 500
BASELINE_EXIT=$?
echo "  Baseline exited with code $BASELINE_EXIT at $(date)"

# ═══════════════════════════════════════════════════════════════════════════════
# RUN 2: CNN
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  RUN 2: CNN feature injection — $(date)"
echo "═══════════════════════════════════════════════════════════════════════════"
python scripts/train.py --config configs/vanvliet_baseline_cnn.yaml --epochs 500
CNN_EXIT=$?
echo "  CNN exited with code $CNN_EXIT at $(date)"

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  RACE COMPLETE"
echo "  Baseline exit code: $BASELINE_EXIT"
echo "  CNN       exit code: $CNN_EXIT"
echo "═══════════════════════════════════════════════════════════════════════════"
