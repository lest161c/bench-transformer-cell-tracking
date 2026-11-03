#!/bin/bash
#SBATCH --job-name=full_model_grid
#SBATCH --account=p_scads_celltracking
#SBATCH --partition=capella
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=benchmark_attn/logs/full_model_inference_grid_%j.out
#SBATCH --error=benchmark_attn/logs/full_model_inference_grid_%j.err

# run_full_model_inference_grid.slurm — H100 full-model inference grid
# (SPEC 0062, ticket F5).
#
# Runs the complete Trackastra inference chain with the four attention
# variants swapped into every encoder/decoder layer. The primary grid is
# B=1, N=512..32768, repeats=3, warmup=5, fp16 on CUDA, greedy linker.
# The optional B=8/16 grid up to N=8192 is prepared but commented out; it is
# not part of the approved primary run.
#
# This is a TIMING-ONLY experiment. It makes no accuracy, tracking-quality,
# or out-of-distribution claim. The k16 checkpoint was trained at roughly
# N=50..200 cells per frame; the large-N grid is a synthetic stress test.
#
# Submit from the repository root so the #SBATCH log paths resolve to
# benchmark_attn/logs/:
#   sbatch benchmark_attn/slurm/run_full_model_inference_grid.slurm
#
# DRY RUN (local validation, no GPU needed):
#   DRY_RUN=1 bash benchmark_attn/slurm/run_full_model_inference_grid.slurm

set -e

# ─── DRY_RUN support: echo the exact commands, then exit 0. This block runs
#     before any cluster-env sourcing so it works on a laptop.
if [ -n "${DRY_RUN:-}" ]; then
    echo "[DRY_RUN=1] echoing the exact commands this runner would execute on the cluster:"
    echo ""
    echo "# 0. Environment provenance probe (torch/cuda/python/GPU)."
    echo "cd \"\$BENCH/benchmark_attn\" && \"\$TRK/.venv/bin/python\" -c \"import torch, sys; print('torch :', torch.__version__); print('cuda  :', torch.version.cuda); print('python:', sys.version.split()[0]); print('GPU   :', torch.cuda.get_device_name(0))\""
    echo ""
    echo "# 1. Primary B=1 grid -> results/full_model_inference_grid.csv"
    echo "#    (skipped when the primary CSV already exists)"
    echo "cd \"\$BENCH/benchmark_attn\" && \"\$TRK/.venv/bin/python\" scripts/benchmarks/benchmark_full_model_inference_grid.py \\"
    echo "    --checkpoint ../checkpoints/k16 \\"
    echo "    --variants relative_pos,cached_dist,sparse_mask_k16,sparse_gather_k16 \\"
    echo "    --Ns 512,1024,2048,4096,8192,16384,32768 \\"
    echo "    --batch-sizes 1 \\"
    echo "    --repeats 3 \\"
    echo "    --warmup 5 \\"
    echo "    --seed 42 \\"
    echo "    --device cuda \\"
    echo "    --dtype fp16 \\"
    echo "    --coord-layout uniform \\"
    echo "    --linker-mode greedy \\"
    echo "    --out results/full_model_inference_grid.csv \\"
    echo "    --env-json results/full_model_inference_grid_env.json \\"
    echo "    --summary results/full_model_inference_grid_summary.csv"
    echo ""
    echo "# 2. Optional B=8/16 grid up to N=8192 (commented out; not run here):"
    echo "#    \"\$TRK/.venv/bin/python\" scripts/benchmarks/benchmark_full_model_inference_grid.py \\"
    echo "#        --checkpoint ../checkpoints/k16 \\"
    echo "#        --variants relative_pos,cached_dist,sparse_mask_k16,sparse_gather_k16 \\"
    echo "#        --Ns 512,1024,2048,4096,8192 \\"
    echo "#        --batch-sizes 8,16 \\"
    echo "#        --repeats 3 --warmup 5 --seed 42 --device cuda --dtype fp16 \\"
    echo "#        --coord-layout uniform --linker-mode greedy \\"
    echo "#        --out results/full_model_inference_grid_batch.csv \\"
    echo "#        --env-json results/full_model_inference_grid_batch_env.json \\"
    echo "#        --summary results/full_model_inference_grid_batch_summary.csv"
    echo ""
    echo "[DRY_RUN=1] done; exiting 0 without executing."
    exit 0
fi

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

cd "$BENCH/benchmark_attn" || exit 1
mkdir -p logs results

echo "=== Full-model inference grid (SPEC 0062, F5 runner) ==="
echo "checkpoint: ../checkpoints/k16 (SparseMask K=16)"
echo "variants  : relative_pos,cached_dist,sparse_mask_k16,sparse_gather_k16"
echo "grid      : B=1; N=512,1024,2048,4096,8192,16384,32768"
echo "timing    : repeats=3 warmup=5 seed=42; device=cuda dtype=fp16"
echo "linker    : greedy (build_graph + track_greedy)"
echo "scope     : timing only; no accuracy/OOD claim; trained at N~50-200"
echo ""

# ─── Environment provenance: torch / CUDA / Python / GPU ────────────
# Interpreter: all python invocations use the Trackastra checkout's venv
# because benchmark_attn/.venv lacks dask/trackastra; $TRK/.venv has them
# (trackastra installed editable). $TRK is validated by load_cluster_env.sh.
"$TRK/.venv/bin/python" -c "import torch, sys; print('torch :', torch.__version__); print('cuda  :', torch.version.cuda); print('python:', sys.version.split()[0]); print('GPU   :', torch.cuda.get_device_name(0))"

echo ""

PRIMARY_CSV="results/full_model_inference_grid.csv"
PRIMARY_ENV_JSON="results/full_model_inference_grid_env.json"
PRIMARY_SUMMARY="results/full_model_inference_grid_summary.csv"

# ─── Primary B=1 grid ───────────────────────────────────────────────
# Resumable: a completed raw CSV is never silently overwritten. The env
# JSON and the F6 summary are regenerated alongside a new raw CSV.
if [ -f "$PRIMARY_CSV" ]; then
    echo "[skip] $PRIMARY_CSV exists — primary B=1 grid not re-run."
    echo "[record] existing primary outputs:"
    ls -l "$PRIMARY_CSV" "$PRIMARY_ENV_JSON" "$PRIMARY_SUMMARY" 2>/dev/null || true
else
    if [ -f "$PRIMARY_ENV_JSON" ] || [ -f "$PRIMARY_SUMMARY" ]; then
        echo "[notice] stale env/summary output found without a raw CSV;"
        echo "         the benchmark script will regenerate them."
    fi
    "$TRK/.venv/bin/python" scripts/benchmarks/benchmark_full_model_inference_grid.py \
        --checkpoint ../checkpoints/k16 \
        --variants relative_pos,cached_dist,sparse_mask_k16,sparse_gather_k16 \
        --Ns 512,1024,2048,4096,8192,16384,32768 \
        --batch-sizes 1 \
        --repeats 3 \
        --warmup 5 \
        --seed 42 \
        --device cuda \
        --dtype fp16 \
        --coord-layout uniform \
        --linker-mode greedy \
        --out results/full_model_inference_grid.csv \
        --env-json results/full_model_inference_grid_env.json \
        --summary results/full_model_inference_grid_summary.csv
fi

# ─── Optional B=8/16 grid up to N=8192 (NOT part of the primary run) ─
# The owner approved preparing this grid but not running it. Uncomment the
# block below only after an explicit follow-up decision. It writes separate
# files so the primary CSV is never touched.
#
# if [ ! -f "results/full_model_inference_grid_batch.csv" ]; then
#     "$TRK/.venv/bin/python" scripts/benchmarks/benchmark_full_model_inference_grid.py \
#         --checkpoint ../checkpoints/k16 \
#         --variants relative_pos,cached_dist,sparse_mask_k16,sparse_gather_k16 \
#         --Ns 512,1024,2048,4096,8192 \
#         --batch-sizes 8,16 \
#         --repeats 3 \
#         --warmup 5 \
#         --seed 42 \
#         --device cuda \
#         --dtype fp16 \
#         --coord-layout uniform \
#         --linker-mode greedy \
#         --out results/full_model_inference_grid_batch.csv \
#         --env-json results/full_model_inference_grid_batch_env.json \
#         --summary results/full_model_inference_grid_batch_summary.csv
# else
#     echo "[skip] results/full_model_inference_grid_batch.csv exists."
# fi

echo ""
echo "=== Done ==="
