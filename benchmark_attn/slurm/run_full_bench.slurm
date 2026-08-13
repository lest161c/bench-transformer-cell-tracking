#!/bin/bash
#SBATCH --job-name=attn_full_bench
#SBATCH --account=p_scads_celltracking
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=logs/full_bench_%j.out
#SBATCH --error=logs/full_bench_%j.err

set -e

# ─── REPO_ROOT ────────────────────────────────────────────────────────
# TODO: REPLACE with the absolute path to your bench-transformer-cell-tracking
# checkout on the cluster.  This is the only cluster-specific value in the
# file — everything else below is resolved relative to it.
REPO_ROOT=/path/to/bench-transformer-cell-tracking

source "$REPO_ROOT/slurm/load_cluster_env.sh"

cd "$BENCH/benchmark_attn" || exit 1
uv sync
mkdir -p logs

echo "=== H100 Full Attention Benchmark ==="
echo "d_model=320  nhead=8  mode=none  dist_mode=v1"
echo "N = 128 256 512 1024 2048 4096 8192"
echo "K = 4 16"
echo ""

uv run python scripts/benchmarks/benchmark_sweep.py \
    --d 320 --nhead 8 \
    --methods gather_sdpa,gather_fused,gather_matmul,mask_knn,knn_relpos,minimax,nsa,dense_flash,dense_masked \
    --Ns 128,256,512,1024,2048,4096,8192 \
    --Ks 4,16,32,64 \
    --warmup 10 --rep 50 \
    --out full_bench_h100.csv

echo ""
echo "=== Done ==="
cat full_bench_h100.csv
