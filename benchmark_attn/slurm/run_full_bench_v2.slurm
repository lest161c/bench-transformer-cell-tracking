#!/bin/bash
#SBATCH --job-name=attn_full_bench_v2
#SBATCH --account=p_scads_celltracking
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=logs/full_bench_v2_%j.out
#SBATCH --error=logs/full_bench_v2_%j.err

# run_full_bench_v2.slurm — Full attention benchmark grid v2 (SPEC 0022, S22-T5).
# Mirrors run_full_bench.slurm but sweeps the full N × K grid with both
# distance-cache regimes (on/off) as two separate invocations with separate
# output CSVs, and adds flex_attn (registered in S22-T4a).
#
# DRY RUN (local validation, no GPU needed):
#   DRY_RUN=1 bash slurm/run_full_bench_v2.slurm
#   → echoes the exact commands it would run on the cluster, then exits 0.

set -e

# ─── DRY_RUN support: when DRY_RUN=1 is set, echo the exact commands this
#     script would run on the cluster, then exit 0. Useful for local syntax /
#     argument validation without GPU access. Set on sbatch submission by
#     exporting DRY_RUN=1 before invoking sbatch, or by passing
#     `--export=DRY_RUN=1` on the sbatch command line.
if [ -n "${DRY_RUN:-}" ]; then
    echo "[DRY_RUN=1] echoing the exact commands this runner would execute on the cluster:"
    echo ""
    echo "# 0. Environment provenance probe (torch/cuda/python/GPU)."
    echo "cd \"\$BENCH/benchmark_attn\" && .venv/bin/python -c \"import torch, sys; print('torch :', torch.__version__); print('cuda  :', torch.version.cuda); print('python:', sys.version.split()[0]); print('GPU   :', torch.cuda.get_device_name(0))\""
    echo ""
    echo "# 1. distance-cache ON  -> results/full_bench_v2_h100_cache_on.csv"
    echo "cd \"\$BENCH/benchmark_attn\" && .venv/bin/python scripts/benchmarks/benchmark_sweep.py \\"
    echo "    --d 320 --nhead 8 \\"
    echo "    --methods sparse_gather_sdpa,sparse_gather_fused,sparse_gather_matmul,sparse_mask,sparse_gather_relpos,minimax,nsa,dense_flash,dense_masked,flex_attn \\"
    echo "    --Ns 128,256,512,1024,2048,4096,8192 \\"
    echo "    --Ks 2,4,8,16,32,64,128 \\"
    echo "    --layers 1 --repeats 3 \\"
    echo "    --warmup 10 --rep 50 \\"
    echo "    --seed 42 \\"
    echo "    --dist-cache on \\"
    echo "    --out results/full_bench_v2_h100_cache_on.csv"
    echo ""
    echo "# 2. distance-cache OFF -> results/full_bench_v2_h100_cache_off.csv"
    echo "cd \"\$BENCH/benchmark_attn\" && .venv/bin/python scripts/benchmarks/benchmark_sweep.py \\"
    echo "    --d 320 --nhead 8 \\"
    echo "    --methods sparse_gather_sdpa,sparse_gather_fused,sparse_gather_matmul,sparse_mask,sparse_gather_relpos,minimax,nsa,dense_flash,dense_masked,flex_attn \\"
    echo "    --Ns 128,256,512,1024,2048,4096,8192 \\"
    echo "    --Ks 2,4,8,16,32,64,128 \\"
    echo "    --layers 1 --repeats 3 \\"
    echo "    --warmup 10 --rep 50 \\"
    echo "    --seed 42 \\"
    echo "    --dist-cache off \\"
    echo "    --out results/full_bench_v2_h100_cache_off.csv"
    echo ""
    echo "[DRY_RUN=1] done; exiting 0 without executing."
    exit 0
fi

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

cd "$BENCH/benchmark_attn" || exit 1
mkdir -p logs results

echo "=== Full attention benchmark grid v2 (SPEC 0022) ==="
echo "Grid: --Ns 128,256,512,1024,2048,4096,8192 --Ks 2,4,8,16,32,64,128 --layers 1"
echo "Timing: --repeats 3 --warmup 10 --rep 50 --seed 42 (forward only)"
echo "Methods: all registered (10, incl. flex_attn); SparseSoftmaxAttention not yet registered"
echo "dist-cache on  -> results/full_bench_v2_h100_cache_on.csv"
echo "dist-cache off -> results/full_bench_v2_h100_cache_off.csv"
echo ""

# ─── Environment provenance: torch / CUDA / Python / GPU ────────────
.venv/bin/python -c "import torch, sys; print('torch   :', torch.__version__); print('cuda    :', torch.version.cuda); print('python  :', sys.version.split()[0]); print('GPU     :', torch.cuda.get_device_name(0))"

echo ""

# ─── Section 1: distance-cache ON ───────────────────────────────────
# Amortized 2D distance matrix computed once outside the timed closure.
if [ ! -f "results/full_bench_v2_h100_cache_on.csv" ]; then
    .venv/bin/python scripts/benchmarks/benchmark_sweep.py \
        --d 320 --nhead 8 \
        --methods sparse_gather_sdpa,sparse_gather_fused,sparse_gather_matmul,sparse_mask,sparse_gather_relpos,minimax,nsa,dense_flash,dense_masked,flex_attn \
        --Ns 128,256,512,1024,2048,4096,8192 \
        --Ks 2,4,8,16,32,64,128 \
        --layers 1 \
        --repeats 3 \
        --d 320 --nhead 8 \
        --warmup 10 --rep 50 \
        --seed 42 \
        --dist-cache on \
        --out results/full_bench_v2_h100_cache_on.csv
else
    echo "[skip] results/full_bench_v2_h100_cache_on.csv exists — distance-cache ON section skipped"
fi

echo ""

# ─── Section 2: distance-cache OFF ──────────────────────────────────
# 2D cdist recomputed inside the timed closure (training-realistic cost).
if [ ! -f "results/full_bench_v2_h100_cache_off.csv" ]; then
    .venv/bin/python scripts/benchmarks/benchmark_sweep.py \
        --d 320 --nhead 8 \
        --methods sparse_gather_sdpa,sparse_gather_fused,sparse_gather_matmul,sparse_mask,sparse_gather_relpos,minimax,nsa,dense_flash,dense_masked,flex_attn \
        --Ns 128,256,512,1024,2048,4096,8192 \
        --Ks 2,4,8,16,32,64,128 \
        --layers 1 \
        --repeats 3 \
        --d 320 --nhead 8 \
        --warmup 10 --rep 50 \
        --seed 42 \
        --dist-cache off \
        --out results/full_bench_v2_h100_cache_off.csv
else
    echo "[skip] results/full_bench_v2_h100_cache_off.csv exists — distance-cache OFF section skipped"
fi

echo ""
echo "=== Done ==="
