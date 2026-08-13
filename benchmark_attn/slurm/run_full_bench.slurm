#!/bin/bash
#SBATCH --job-name=attn_full_bench
#SBATCH --account=p_scads_celltracking
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=benchmark_attn/logs/full_bench_%j.out
#SBATCH --error=benchmark_attn/logs/full_bench_%j.err

set -e

export PIP_REQUIRE_VIRTUALENV=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME

WS=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601
REPO="$WS/bench-transformer-cell-tracking"
VENV="$WS/trackastra/.venv"
. "$VENV/bin/activate"

# install torch if needed
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q 2>&1 | tail -1

cd "$REPO" || exit 1

echo "=== H100 Full Attention Benchmark ==="
echo "d_model=320  nhead=8  mode=none  dist_mode=v1"
echo "N = 128 256 512 1024 2048 4096 8192"
echo "K = 4 16"
echo ""

python benchmark_attn/scripts/benchmarks/benchmark_sweep.py \
    --d 320 --nhead 8 \
    --methods gather_sdpa,gather_fused,gather_matmul,mask_knn,knn_relpos,minimax,nsa,dense_flash,dense_masked \
    --Ns 128,256,512,1024,2048,4096,8192 \
    --Ks 4,16, 32, 64 \
    --warmup 10 --rep 50 \
    --out benchmark_attn/full_bench_h100.csv

echo ""
echo "=== Done ==="
cat benchmark_attn/full_bench_h100.csv
