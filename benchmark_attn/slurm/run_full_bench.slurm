#!/bin/bash
#SBATCH --job-name=attn_full_bench
#SBATCH --account=p_scads_celltracking
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=logs/full_bench_%j.out
#SBATCH --error=logs/full_bench_%j.err

set -e

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

cd "$BENCH/benchmark_attn" || exit 1
mkdir -p logs

echo "=== H100 Full Attention Benchmark ==="
echo "d_model=320  nhead=8  mode=none  dist_mode=v1"
echo "N = 128 256 512 1024 2048 4096 8192"
echo "K = 4 16 32 64"
echo ""

.venv/bin/python scripts/benchmarks/benchmark_sweep.py \
    --d 320 --nhead 8 \
    --methods gather_sdpa,gather_fused,gather_matmul,mask_knn,knn_relpos,minimax,nsa,dense_flash,dense_masked \
    --Ns 128,256,512,1024,2048,4096,8192 \
    --Ks 4,16,32,64 \
    --warmup 10 --rep 50 \
    --with-knn \
    --out full_bench_h100_with_knn.csv

echo ""
echo "=== Done ==="
cat full_bench_h100.csv
