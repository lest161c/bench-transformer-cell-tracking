#!/bin/bash
#SBATCH --job-name=attn_flex_setup
#SBATCH --account=p_scads_celltracking
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=logs/flex_setup_%j.out
#SBATCH --error=logs/flex_setup_%j.err

# run_flex_setup.slurm — FlexAttention setup cost per (B, N) (SPEC 0051 H2).
#
# Measures block-mask construction, torch.compile, and steady-state forward
# time separately for B=1 and B=64 over the full N grid.  The two batch sizes
# run in SEPARATE Python processes, each with its own fresh Inductor cache
# directory, so neither process can reuse a compile artifact from the other.
#
# Why two processes / two caches:
#   torch._dynamo.reset() clears only the in-process Dynamo cache.  The
#   on-disk Inductor cache persists in TORCHINDUCTOR_CACHE_DIR, so a warm
#   cache would hide compile cost.  A fresh cache per batch size makes every
#   (B, N) compile cold.
#
# Outputs:
#   results/flex_setup_h100_b1.csv
#   results/flex_setup_h100_b64.csv
#
# DRY RUN (local validation, no GPU needed):
#   DRY_RUN=1 bash slurm/run_flex_setup.slurm

set -e

if [ -n "${DRY_RUN:-}" ]; then
    echo "[DRY_RUN=1] echoing the exact commands this runner would execute on the cluster:"
    echo ""
    echo "# 0. Environment provenance probe (torch/cuda/python/GPU)."
    echo "cd \"\$BENCH/benchmark_attn\" && .venv/bin/python -c \"import torch, sys; print('torch :', torch.__version__); print('cuda  :', torch.version.cuda); print('python:', sys.version.split()[0]); print('GPU   :', torch.cuda.get_device_name(0))\""
    echo ""
    echo "# 1. batch 1, fresh Inductor cache -> results/flex_setup_h100_b1.csv"
    echo "export TORCHINDUCTOR_CACHE_DIR=\$(mktemp -d /tmp/flex_inductor_b1_XXXXXX)"
    echo "cd \"\$BENCH/benchmark_attn\" && .venv/bin/python scripts/benchmarks/benchmark_flex_setup.py \\"
    echo "    --batch-sizes 1 \\"
    echo "    --Ns 128,256,512,1024,2048,4096,8192,16384,32768 \\"
    echo "    --warmup 5 --rep 300 --repeats 3 \\"
    echo "    --device cuda --no-grad \\"
    echo "    --out results/flex_setup_h100_b1.csv"
    echo "rm -rf \"\$TORCHINDUCTOR_CACHE_DIR\""
    echo ""
    echo "# 2. batch 64, fresh Inductor cache -> results/flex_setup_h100_b64.csv"
    echo "export TORCHINDUCTOR_CACHE_DIR=\$(mktemp -d /tmp/flex_inductor_b64_XXXXXX)"
    echo "cd \"\$BENCH/benchmark_attn\" && .venv/bin/python scripts/benchmarks/benchmark_flex_setup.py \\"
    echo "    --batch-sizes 64 \\"
    echo "    --Ns 128,256,512,1024,2048,4096,8192,16384,32768 \\"
    echo "    --warmup 5 --rep 300 --repeats 3 \\"
    echo "    --device cuda --no-grad \\"
    echo "    --out results/flex_setup_h100_b64.csv"
    echo "rm -rf \"\$TORCHINDUCTOR_CACHE_DIR\""
    echo ""
    echo "[DRY_RUN=1] done; exiting 0 without executing."
    exit 0
fi

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

cd "$BENCH/benchmark_attn" || exit 1
mkdir -p logs results

echo "=== FlexAttention setup cost — B=1 and B=64 (SPEC 0051 H2) ==="
echo "Measures prepare (block mask + compiled wrapper) and first call"
echo "(torch.compile + one forward execution) separately from steady state."
echo "Each batch size runs in a fresh process with a fresh Inductor cache."
echo "Grid: N=128,256,512,1024,2048,4096,8192,16384,32768"
echo "Timing: --warmup 5 --rep 300 --repeats 3 --device cuda --no-grad"
echo "Outputs: results/flex_setup_h100_b1.csv"
echo "         results/flex_setup_h100_b64.csv"
echo ""

.venv/bin/python -c "import torch, sys; print('torch   :', torch.__version__); print('cuda    :', torch.version.cuda); print('python  :', sys.version.split()[0]); print('GPU     :', torch.cuda.get_device_name(0))"

echo ""

# ─── Section 1: batch 1, fresh Inductor cache ───────────────────────
B1_CACHE_DIR=$(mktemp -d /tmp/flex_inductor_b1_XXXXXX)
export TORCHINDUCTOR_CACHE_DIR="$B1_CACHE_DIR"
echo "B=1 Inductor cache: $TORCHINDUCTOR_CACHE_DIR"
echo "B=1 target        : results/flex_setup_h100_b1.csv"
if ! .venv/bin/python scripts/benchmarks/benchmark_flex_setup.py \
    --batch-sizes 1 \
    --Ns 128,256,512,1024,2048,4096,8192,16384,32768 \
    --warmup 5 --rep 300 --repeats 3 \
    --device cuda --no-grad \
    --out results/flex_setup_h100_b1.csv
then
    echo "B=1 FlexAttention setup run failed; removing cache directory." >&2
    rm -rf "$B1_CACHE_DIR"
    exit 1
fi
rm -rf "$B1_CACHE_DIR"

echo ""

# ─── Section 2: batch 64, fresh Inductor cache ──────────────────────
B64_CACHE_DIR=$(mktemp -d /tmp/flex_inductor_b64_XXXXXX)
export TORCHINDUCTOR_CACHE_DIR="$B64_CACHE_DIR"
echo "B=64 Inductor cache: $TORCHINDUCTOR_CACHE_DIR"
echo "B=64 target        : results/flex_setup_h100_b64.csv"
if ! .venv/bin/python scripts/benchmarks/benchmark_flex_setup.py \
    --batch-sizes 64 \
    --Ns 128,256,512,1024,2048,4096,8192,16384,32768 \
    --warmup 5 --rep 300 --repeats 3 \
    --device cuda --no-grad \
    --out results/flex_setup_h100_b64.csv
then
    echo "B=64 FlexAttention setup run failed; removing cache directory." >&2
    rm -rf "$B64_CACHE_DIR"
    exit 1
fi
rm -rf "$B64_CACHE_DIR"

echo ""
echo "=== Done ==="
