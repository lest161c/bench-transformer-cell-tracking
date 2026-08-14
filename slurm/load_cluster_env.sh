#!/bin/bash
# load_cluster_env.sh — Shared loader for cluster paths and runtime env vars.
#
# Source this file near the top of every slurm script:
#
#   source "$(dirname "${BASH_SOURCE[0]}")/../../slurm/load_cluster_env.sh"
#
# Why this is needed:
#   - Slurm copies each script to /var/spool/slurmd/jobXXX/ before running,
#     so ${BASH_SOURCE[0]} does NOT resolve to the file's real repo location.
#   - $SLURM_SUBMIT_DIR depends on where the reproducer called `sbatch` from
#     (unstable across `cd slurm && sbatch` vs `sbatch slurm/...`).
#
# Solution: the reproducer creates ~/.bench.env ONCE with all cluster paths
# (REPO_ROOT, TRK, BENCH, DATA_DIR).  This loader sources that file.  The
# slurm scripts never need to be modified after `git pull`.
#
# The .env file at the repo root is also sourced as a fallback (convenience
# for local dev), but ~/.bench.env takes priority when both exist.
#
# Responsibilities (deliberately narrow):
#   1. Source ~/.bench.env (and optionally .env at the repo root as fallback).
#   2. Validate that $TRK and $BENCH exist on disk.
#   3. Export runtime env vars (OMP/MKL/NCCL/CUDA flags, sanitised PYTHONPATH).
#
# This loader deliberately does NOT:
#   - Run pip install or uv sync (each slurm script uses the pre-built
#     .venv/bin/python that the reproducer created with `uv sync` once on
#     the login node).
#   - Run git commands (each slurm script checks out / pulls its branch).
#   - Create log directories (each slurm script owns its own logs/).

# ─── Source cluster env ──────────────────────────────────────────────
# Priority: ~/.bench.env (user-wide, gitignored) > $REPO_ROOT/.env (local).
# REPO_ROOT must already be set by the calling slurm script for the local
# fallback to work.

BENCH_ENV="$HOME/.bench.env"

if [ -f "$BENCH_ENV" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$BENCH_ENV"
    set +a
elif [ -n "${REPO_ROOT:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
    # Legacy fallback: slurm scripts that still set REPO_ROOT at the top.
    set -a
    # shellcheck disable=SC1090
    source "$REPO_ROOT/.env"
    set +a
else
    cat >&2 <<EOF
FATAL: no cluster env file found.

Create ~/.bench.env from the tracked template:

    mkdir -p ~/.bench-tracker
    cp $REPO_ROOT/.env.example ~/.bench.env
    \$EDITOR ~/.bench.env   # fill in REPO_ROOT, TRK, BENCH, DATA_DIR

Or, as a one-time fallback, set REPO_ROOT at the top of the slurm script
so the loader can find $REPO_ROOT/.env.
EOF
    exit 1
fi

# ─── Validate required paths ───────────────────────────────────────────
for var in REPO_ROOT TRK BENCH; do
    val="${!var:-}"
    if [ -z "$val" ]; then
        echo "FATAL: $var is not set (check ~/.bench.env)." >&2
        exit 1
    fi
    if [ ! -e "$val" ]; then
        echo "FATAL: $var=$val does not exist on this host." >&2
        exit 1
    fi
done

# ─── Export runtime env vars ───────────────────────────────────────────
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NCCL_DEBUG=WARN
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
