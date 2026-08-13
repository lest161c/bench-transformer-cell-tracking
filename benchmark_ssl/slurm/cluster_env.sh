#!/bin/bash
# cluster_env.sh — Common environment setup for all slurm scripts.
#
# Source this file at the top of every slurm script:
#   source "$(dirname "$0")/cluster_env.sh"
#
# All HPC paths are defined here so they can be changed in one place.
# Override TRK or BENCH by setting them before sourcing:
#   TRK=/custom/path BENCH=/other/path source cluster_env.sh

# ─── Cluster paths ──────────────────────────────────────────────────────
TRK="${TRK:-/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra}"
BENCH="${BENCH:-/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking}"
ENV_DIR="${ENV_DIR:-$TRK/.venv}"
DATA_DIR="${DATA_DIR:-/data/cat/ws/mawe985g-data/data/celltracking}"

# ─── Verify required paths ──────────────────────────────────────────────
for required_path in "$TRK" "$BENCH" "$ENV_DIR/bin/activate"; do
    if [ ! -e "$required_path" ]; then
        echo "FATAL: Required path not found: $required_path" >&2
        exit 1
    fi
done

# ─── Environment exports ────────────────────────────────────────────────
export PIP_REQUIRE_VIRTUALENV=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NCCL_DEBUG=WARN
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME

# ─── Activate venv and install deps ─────────────────────────────────────
. "$ENV_DIR/bin/activate"
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib

cd "$BENCH"
git fetch origin cached-dist-attn && git checkout cached-dist-attn && git pull origin cached-dist-attn
mkdir -p logs
