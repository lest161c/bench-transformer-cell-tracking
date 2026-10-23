#!/bin/bash
#SBATCH --job-name=attn_b64_isolation
#SBATCH --account=p_scads_celltracking
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=logs/b64_isolation_%j.out
#SBATCH --error=logs/b64_isolation_%j.err

# run_b64_sweep_isolation.slurm — B=64 attention grid with contract B.
# Mirrors run_b64_sweep.slurm but runs the SPEC 0051 contract-B harness:
# ONE shared spatial graph per (N, config), so KNN indices and the Flex block
# mask are built batch-independently and reused across the batch.  The old
# runner's pre-fix CSVs are left untouched; this runner writes new
# *_isolation.csv files instead.
#
# Grid: 10 methods × N∈{128..32768} × K∈{2,4,8,16,32,64,128}, fp16, seed 42,
# forward-only, both distance-cache regimes (on/off) as separate CSVs.
# Dense methods OOM-record at the top of the N range (expected, harness
# records status=oom); K≥N rows are recorded as skip_k_ge_n.
#
# Both sections are resumable: a section is skipped when its isolation CSV
# already exists.
#
# DRY RUN (local validation, no GPU needed):
#   DRY_RUN=1 bash slurm/run_b64_sweep_isolation.slurm

set -e

if [ -n "${DRY_RUN:-}" ]; then
    echo "[DRY_RUN=1] echoing the exact commands this runner would execute on the cluster:"
    echo ""
    echo "# 0. Environment provenance probe (torch/cuda/python/GPU)."
    echo "cd \"\$BENCH/benchmark_attn\" && .venv/bin/python -c \"import torch, sys; print('torch :', torch.__version__); print('cuda  :', torch.version.cuda); print('python:', sys.version.split()[0]); print('GPU   :', torch.cuda.get_device_name(0))\""
    echo ""
    echo "# 1. batch 64, distance-cache ON  -> results/full_bench_b64_h100_cache_on_isolation.csv"
    echo "#    (skipped if the isolation CSV already exists)"
    echo "cd \"\$BENCH/benchmark_attn\" && .venv/bin/python scripts/benchmarks/benchmark_sweep.py \\"
    echo "    --d 320 --nhead 8 --batch 64 \\"
    echo "    --methods sparse_gather_sdpa,sparse_gather_fused,sparse_gather_matmul,sparse_mask,sparse_gather_relpos,minimax,nsa,dense_flash,dense_masked,flex_attn \\"
    echo "    --Ns 128,256,512,1024,2048,4096,8192,16384,32768 \\"
    echo "    --Ks 2,4,8,16,32,64,128 \\"
    echo "    --layers 1 --repeats 3 \\"
    echo "    --warmup 10 --rep 50 \\"
    echo "    --seed 42 \\"
    echo "    --dist-cache on \\"
    echo "    --out results/full_bench_b64_h100_cache_on_isolation.csv"
    echo ""
    echo "# 2. batch 64, distance-cache OFF -> results/full_bench_b64_h100_cache_off_isolation.csv"
    echo "#    (skipped if the isolation CSV already exists)"
    echo "cd \"\$BENCH/benchmark_attn\" && .venv/bin/python scripts/benchmarks/benchmark_sweep.py \\"
    echo "    --d 320 --nhead 8 --batch 64 \\"
    echo "    --methods sparse_gather_sdpa,sparse_gather_fused,sparse_gather_matmul,sparse_mask,sparse_gather_relpos,minimax,nsa,dense_flash,dense_masked,flex_attn \\"
    echo "    --Ns 128,256,512,1024,2048,4096,8192,16384,32768 \\"
    echo "    --Ks 2,4,8,16,32,64,128 \\"
    echo "    --layers 1 --repeats 3 \\"
    echo "    --warmup 10 --rep 50 \\"
    echo "    --seed 42 \\"
    echo "    --dist-cache off \\"
    echo "    --out results/full_bench_b64_h100_cache_off_isolation.csv"
    echo ""
    echo "[DRY_RUN=1] done; exiting 0 without executing."
    exit 0
fi

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

cd "$BENCH/benchmark_attn" || exit 1
mkdir -p logs results

echo "=== Attention benchmark grid B=64 — contract B (SPEC 0051) ==="
echo "Contract B: one shared spatial graph per (N, config); setup is batch-independent."
echo "  KNN indices and the Flex block mask are built once per (N, config)"
echo "  and reused across the batch. No unused dist_2d allocations."
echo "Grid: --batch 64 --Ns 128..32768 --Ks 2,4,8,16,32,64,128 --layers 1"
echo "Timing: --repeats 3 --warmup 10 --rep 50 --seed 42 (forward only, fp16)"
echo "Methods: all registered (10, incl. flex_attn)"
echo "dist-cache on  -> results/full_bench_b64_h100_cache_on_isolation.csv"
echo "dist-cache off -> results/full_bench_b64_h100_cache_off_isolation.csv"
echo ""

.venv/bin/python -c "import torch, sys; print('torch   :', torch.__version__); print('cuda    :', torch.version.cuda); print('python  :', sys.version.split()[0]); print('GPU     :', torch.cuda.get_device_name(0))"

echo ""

# ─── Section 1: batch 64, distance-cache ON ─────────────────────────
if [ ! -f "results/full_bench_b64_h100_cache_on_isolation.csv" ]; then
    .venv/bin/python scripts/benchmarks/benchmark_sweep.py \
        --d 320 --nhead 8 --batch 64 \
        --methods sparse_gather_sdpa,sparse_gather_fused,sparse_gather_matmul,sparse_mask,sparse_gather_relpos,minimax,nsa,dense_flash,dense_masked,flex_attn \
        --Ns 128,256,512,1024,2048,4096,8192,16384,32768 \
        --Ks 2,4,8,16,32,64,128 \
        --layers 1 \
        --repeats 3 \
        --warmup 10 --rep 50 \
        --seed 42 \
        --dist-cache on \
        --out results/full_bench_b64_h100_cache_on_isolation.csv
else
    echo "[skip] results/full_bench_b64_h100_cache_on_isolation.csv exists — section skipped (resumable)"
fi

echo ""

# ─── Section 2: batch 64, distance-cache OFF ────────────────────────
if [ ! -f "results/full_bench_b64_h100_cache_off_isolation.csv" ]; then
    .venv/bin/python scripts/benchmarks/benchmark_sweep.py \
        --d 320 --nhead 8 --batch 64 \
        --methods sparse_gather_sdpa,sparse_gather_fused,sparse_gather_matmul,sparse_mask,sparse_gather_relpos,minimax,nsa,dense_flash,dense_masked,flex_attn \
        --Ns 128,256,512,1024,2048,4096,8192,16384,32768 \
        --Ks 2,4,8,16,32,64,128 \
        --layers 1 \
        --repeats 3 \
        --warmup 10 --rep 50 \
        --seed 42 \
        --dist-cache off \
        --out results/full_bench_b64_h100_cache_off_isolation.csv
else
    echo "[skip] results/full_bench_b64_h100_cache_off_isolation.csv exists — section skipped (resumable)"
fi

echo ""
echo "=== Done ==="
