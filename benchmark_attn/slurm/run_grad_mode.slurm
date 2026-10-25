#!/bin/bash
#SBATCH --job-name=attn_grad_mode
#SBATCH --account=p_scads_celltracking
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=logs/grad_mode_%j.out
#SBATCH --error=logs/grad_mode_%j.err

# run_grad_mode.slurm — grad-enabled vs no-grad forward ablation (SPEC 0051 H5).
#
# Runs benchmark_grad_mode.py on the contract-B harness from
# benchmark_sweep.py: one shared spatial graph per config, chunked exact KNN,
# no unused dist_2d allocation.  One representative shape per method:
#
#   dense_flash          K=0   N=32768
#   dense_masked         K=0   N=4096
#   sparse_mask          K=16  N=4096
#   sparse_gather_matmul K=16  N=32768
#   flex_attn            K=0   N=32768
#
# Each config is measured twice with measure(warmup=5, min_run_time=0.3 s,
# repeats=3): first in torch.no_grad(), then with autograd enabled.  OOM rows
# are recorded as status=oom and the run continues.  Output:
# results/grad_mode_ablation.csv
#
# DRY RUN (local validation, no GPU needed):
#   DRY_RUN=1 bash slurm/run_grad_mode.slurm

set -e

if [ -n "${DRY_RUN:-}" ]; then
    echo "[DRY_RUN=1] echoing the exact command this runner would execute on the cluster:"
    echo ""
    echo "# 0. Environment provenance probe (torch/cuda/python/GPU)."
    echo "cd \"\$BENCH/benchmark_attn\" && .venv/bin/python -c \"import torch, sys; print('torch :', torch.__version__); print('cuda  :', torch.version.cuda); print('python:', sys.version.split()[0]); print('GPU   :', torch.cuda.get_device_name(0))\""
    echo ""
    echo "# 1. no-grad vs grad-enabled forward ablation -> results/grad_mode_ablation.csv"
    echo "cd \"\$BENCH/benchmark_attn\" && .venv/bin/python scripts/benchmarks/benchmark_grad_mode.py \\"
    echo "    --batch-size 64 --seed 42 \\"
    echo "    --warmup 5 --rep 300 --repeats 3 \\"
    echo "    --device cuda \\"
    echo "    --out results/grad_mode_ablation.csv"
    echo ""
    echo "[DRY_RUN=1] done; exiting 0 without executing."
    exit 0
fi

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

cd "$BENCH/benchmark_attn" || exit 1
mkdir -p logs results

echo "=== Attention grad-mode ablation (SPEC 0051 H5) ==="
echo "Contract B harness: shared graph, chunked KNN, no unused dist_2d."
echo "One representative shape per method; no_grad measured first, then grad."
echo "Timing: --batch-size 64 --warmup 5 --rep 300 --repeats 3 --seed 42"
echo "Output: results/grad_mode_ablation.csv"
echo ""

.venv/bin/python -c "import torch, sys; print('torch   :', torch.__version__); print('cuda    :', torch.version.cuda); print('python  :', sys.version.split()[0]); print('GPU     :', torch.cuda.get_device_name(0))"

echo ""

.venv/bin/python scripts/benchmarks/benchmark_grad_mode.py \
    --batch-size 64 --seed 42 \
    --warmup 5 --rep 300 --repeats 3 \
    --device cuda \
    --out results/grad_mode_ablation.csv

echo ""
echo "=== Done ==="
