#!/bin/bash
#SBATCH --job-name=flex_reorder
#SBATCH --account=p_scads_celltracking
#SBATCH --partition=capella
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --output=logs/flex_reorder.out
#SBATCH --error=logs/flex_reorder.err
#
# FlexAttention sorted-vs-unsorted token-order experiment at B=64
# (owner question 2026-10-28: the B=64 campaign measured Flex time growing
# ~4x per N-doubling = quadratic, despite linear memory; hypothesis: unsorted
# token order makes the 128-token block mask dense (measured block density
# 1.000 at every N) and spatial sorting restores banded sparsity).
# Pre-registered verdict rules in the script docstring.

source "$HOME/.bench.env"
cd "$BENCH"

echo "=== flex reorder experiment: $(date) ==="
benchmark_attn/.venv/bin/python \
    benchmark_attn/scripts/benchmarks/benchmark_flex_reorder.py \
    --out benchmark_attn/results/flex_reorder_b64.csv
echo "=== exited $(date) ==="
