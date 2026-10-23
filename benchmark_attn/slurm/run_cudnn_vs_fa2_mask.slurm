#!/bin/bash
#SBATCH --job-name=cudnn_fa2_mask
#SBATCH --account=p_scads_celltracking
#SBATCH --partition=capella
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/cudnn_fa2_mask_%j.out
#SBATCH --error=logs/cudnn_fa2_mask_%j.err

set -e
source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

cd "$BENCH/benchmark_attn" || exit 1
mkdir -p logs results

echo "=== H100 cuDNN flash vs FA2 mask-overhead decomposition ==="
echo "stages: A backend census, B kernel-intrinsic nomask pair,"
echo "        C isolated mask build, D mask carriage + end-to-end anchor,"
echo "        E causal vs prebuilt float/bool masks"
echo "d_model=320  nhead=8  (head_dim 40 kernel stages, 64 both-kernel regime)"
echo "dist-mode=v1  dist-cache=on  fp16  B=1"
echo "N = 128 256 512 1024 2048 4096 8192"
echo ""

.venv/bin/python scripts/benchmarks/benchmark_cudnn_vs_fa2_mask.py \
    --seq-lens 128,256,512,1024,2048,4096,8192 \
    --d-heads 40,64 \
    --d 320 --nhead 8 \
    --dist-mode v1 --dist-cache on \
    --mode none --coord-dim 2 \
    --warmup 5 --rep 30 --repeats 3 \
    --seed 42 \
    --out results/cudnn_vs_fa2_mask_decomposition.csv

echo ""
echo "=== Done ==="
