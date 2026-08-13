#!/bin/bash
# load_cluster_env.sh — Shared loader for cluster paths and runtime env vars.
#
# Source this file near the top of every slurm script:
#
#   # benchmark_ssl/slurm/foo.slurm, benchmark_attn/slurm/foo.slurm
#   source "$(dirname "${BASH_SOURCE[0]}")/../../slurm/load_cluster_env.sh"
#
#   # benchmark_training/scripts/slurm/foo.slurm
#   source "$(dirname "${BASH_SOURCE[0]}")/../../../slurm/load_cluster_env.sh"
#
# Responsibilities (deliberately narrow):
#   1. Discover .env at <repo_root>/.env.
#   2. Source .env (KEY=value format, no `export` — dotenv convention).
#   3. Validate that $TRK (and $BENCH, if set) exist on disk.
#   4. Export the runtime env vars expected by all training scripts
#      (OMP/MKL thread counts, NCCL/CUDA flags, sanitised PYTHONPATH).
#
# This loader deliberately does NOT:
#   - Run pip install or uv sync (each slurm script calls `uv sync`
#     inside its own project directory).
#   - Run git commands (each slurm script checks out / pulls the
#     branch it needs).
#   - Create log directories (each slurm script owns its own logs/).
#   - Activate a Python venv (uv-managed project .venvs are picked
#     up automatically by `uv run` / `uv sync`).

# ─── Discover repo root ────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ─── Source .env ───────────────────────────────────────────────────────
ENV_FILE="$REPO_ROOT/.env"
if [ ! -f "$ENV_FILE" ]; then
    cat >&2 <<EOF
FATAL: $ENV_FILE not found.

Create it from the tracked template:

    cp $REPO_ROOT/.env.example $ENV_FILE

Then edit $ENV_FILE to set your cluster paths (TRK, BENCH, DATA_DIR).
EOF
    exit 1
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# ─── Validate required paths ───────────────────────────────────────────
if [ -n "${TRK:-}" ] && [ ! -e "$TRK" ]; then
    echo "FATAL: TRK=$TRK does not exist on this host." >&2
    exit 1
fi

if [ -n "${BENCH:-}" ] && [ ! -e "$BENCH" ]; then
    echo "FATAL: BENCH=$BENCH does not exist on this host." >&2
    exit 1
fi

# ─── Export runtime env vars ───────────────────────────────────────────
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NCCL_DEBUG=WARN
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
