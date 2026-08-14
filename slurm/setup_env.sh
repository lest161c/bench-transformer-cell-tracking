#!/bin/bash
# setup_env.sh — One-time environment setup on the HPC login node.
#
# Run this AFTER `git clone` and BEFORE submitting any slurm jobs:
#
#   git clone <repo-url> /data/cat/ws/.../bench-transformer-cell-tracking
#   cd /data/cat/ws/.../bench-transformer-cell-tracking
#   bash slurm/setup_env.sh
#
# This script:
#   1. Creates ~/.bench.env from .env.example with your cluster paths.
#   2. Runs `uv sync` in each subproject to create .venv/ (done ONCE,
#      not repeated per slurm job).
#   3. Installs trackastra in editable mode into each .venv.
#
# After this, `git pull` will never require re-editing slurm files
# (they source ~/.bench.env, which is gitignored).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ─── Step 1: Create ~/.bench.env ─────────────────────────────────────
BENCH_ENV="$HOME/.bench.env"

if [ -f "$BENCH_ENV" ]; then
    echo ">>> ~/.bench.env already exists, skipping creation."
    echo ">>> If paths are stale, edit $BENCH_ENV directly."
else
    echo ">>> Creating $BENCH_ENV from template..."
    cp "$REPO_ROOT/.env.example" "$BENCH_ENV"
    echo ""
    echo ">>> IMPORTANT: Edit $BENCH_ENV now and replace the"
    echo ">>> /path/to/... placeholders with your actual cluster paths."
    echo ">>> Then re-run this script."
    echo ""
    exit 0
fi

# Source the env to get paths
set -a
source "$BENCH_ENV"
set +a

# Validate critical paths
for var in REPO_ROOT TRK BENCH; do
    if [ -z "${!var:-}" ]; then
        echo "FATAL: $var is not set in $BENCH_ENV" >&2
        exit 1
    fi
done

# ─── Step 2: Create .venv in each subproject ─────────────────────────
for project in benchmark_attn benchmark_ssl benchmark_training; do
    DIR="$REPO_ROOT/$project"
    if [ ! -d "$DIR" ]; then
        echo ">>> Skipping $project (directory not found)"
        continue
    fi
    echo ">>> Running uv sync in $project/..."
    cd "$DIR"
    uv sync
done

# ─── Step 3: Install trackastra in editable mode ─────────────────────
echo ">>> Installing trackastra in editable mode..."
cd "$TRK"

# Install into each subproject's venv
for project in benchmark_attn benchmark_ssl benchmark_training; do
    VENV_PIP="$REPO_ROOT/$project/.venv/bin/pip"
    if [ -f "$VENV_PIP" ]; then
        echo ">>> Installing trackastra into $project/.venv..."
        "$VENV_PIP" install -e . >/dev/null 2>&1 || echo ">>> Warning: pip install for $project failed (may already be installed)"
    fi
done

echo ""
echo "=== Setup complete ==="
echo ">>> .venv/ directories created in each subproject."
echo ">>> Slurm jobs now use .venv/bin/python directly (no uv sync overhead)."
echo ">>> To re-sync after dependency changes: cd <subproject> && uv sync"
