#!/bin/bash
#SBATCH --account=p_scads_celltracking
#SBATCH --job-name=lambda_decay
#SBATCH --output=logs/lambda_decay-%j.out
#SBATCH --error=logs/lambda_decay-%j.err
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

# W&B: scalar metrics only (no images, histograms, or console logs)
export WANDB_LIGHT=true
export WANDB_CONSOLE=off
export WANDB_WATCH=false
unset PYTHONPATH
unset PYTHONHOME

TRK=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra
BENCH=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking
ENV_DIR="$TRK/.venv"

. "$ENV_DIR/bin/activate"
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
cd "$TRK"
git fetch origin cached-dist-attn && git checkout cached-dist-attn && git pull origin cached-dist-attn
python -m pip install -e .
mkdir -p logs

CFG_DIR="$BENCH/configs"

# ═══════════════════════════════════════════════════════════════════════════════
# RUN 1: λ(t) decay only — frozen CNN + cosine decay 1→0
# ═══════════════════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  λ(t) decay only — $(date)"
echo "═══════════════════════════════════════════════════════════════════════════"
python scripts/train.py --config "$CFG_DIR/vanvliet_lambda_decay.yaml" --epochs 500
LAMBDA_DECAY_EXIT=$?
echo "  λ(t) decay only exited with code $LAMBDA_DECAY_EXIT at $(date)"

# ═══════════════════════════════════════════════════════════════════════════════
# RUN 2: λ(t) decay + dropout
# ═══════════════════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  λ(t) decay + dropout — $(date)"
echo "═══════════════════════════════════════════════════════════════════════════"
python scripts/train.py --config "$CFG_DIR/vanvliet_lambda_decay_dropout.yaml" --epochs 500
LAMBDA_DROPOUT_EXIT=$?
echo "  λ(t) decay + dropout exited with code $LAMBDA_DROPOUT_EXIT at $(date)"

# ═══════════════════════════════════════════════════════════════════════════════
# RUN 3: Ablation — constant 50% dropout (no λ decay)
# ═══════════════════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  Constant 50% dropout (ablation) — $(date)"
echo "═══════════════════════════════════════════════════════════════════════════"
python scripts/train.py --config "$CFG_DIR/vanvliet_lambda_half.yaml" --epochs 500
LAMBDA_HALF_EXIT=$?
echo "  Constant 50% dropout (ablation) exited with code $LAMBDA_HALF_EXIT at $(date)"

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  λ(t) decay experiment summary:"
echo "    λ(t) decay only          exit code: $LAMBDA_DECAY_EXIT"
echo "    λ(t) decay + dropout     exit code: $LAMBDA_DROPOUT_EXIT"
echo "    Constant 50% dropout     exit code: $LAMBDA_HALF_EXIT"
echo "  Finished at $(date)"
echo "═══════════════════════════════════════════════════════════════════════════"
