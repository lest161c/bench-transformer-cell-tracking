#!/bin/bash
# load_cluster_env.sh — Shared loader for cluster paths and runtime env vars.
#
# This file is sourced by every slurm script. Because slurm copies each
# script to /var/spool/slurmd/jobXXX/ before running, ${BASH_SOURCE[0]}
# does NOT resolve to the file's real repo location, and $SLURM_SUBMIT_DIR
# depends on where the reproducer called `sbatch` from. Both are unreliable.
#
# The calling slurm script therefore sets REPO_ROOT at the top of the
# file (the single cluster-specific path the reproducer must replace).
# This loader trusts that variable and uses it to find .env.
#
# Responsibilities (deliberately narrow):
#   1. Source .env at $REPO_ROOT/.env.
#   2. Validate that $TRK and $BENCH (from .env) exist on disk.
#   3. Export the runtime env vars expected by all training scripts
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
#
# Fallback for direct (non-slurm) invocation: if REPO_ROOT is unset,
# derive it from this loader's own filesystem location.

# ─── Locate repo root ─────────────────────────────────────────────────
if [ -z "${REPO_ROOT:-}" ]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [ ! -d "$REPO_ROOT" ]; then
    echo "FATAL: REPO_ROOT=$REPO_ROOT does not exist or is not a directory." >&2
    echo "Each slurm script must define REPO_ROOT at the top (see comments)." >&2
    exit 1
fi

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
