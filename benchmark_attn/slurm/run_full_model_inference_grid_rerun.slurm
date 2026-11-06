#!/bin/bash
#SBATCH --job-name=full_model_grid_v2
#SBATCH --account=p_scads_celltracking
#SBATCH --partition=capella
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=benchmark_attn/logs/full_model_inference_grid_v2_%j.out
#SBATCH --error=benchmark_attn/logs/full_model_inference_grid_v2_%j.err

# run_full_model_inference_grid_rerun.slurm — H100 rerun after the
# build_graph hot-loop fix, the paper-mandated 0.05 edge threshold, and the
# FlexAttention full-model adapter.
#
# This keeps the existing SPEC 0062 synthetic N grid (10 frames, uniform
# layout) and writes new v2 files so the original results are untouched.
#
# Submit from the repository root:
#   sbatch benchmark_attn/slurm/run_full_model_inference_grid_rerun.slurm
#
# DRY RUN (local validation, no GPU needed):
#   DRY_RUN=1 bash benchmark_attn/slurm/run_full_model_inference_grid_rerun.slurm

set -e

if [ -n "${DRY_RUN:-}" ]; then
    echo "[DRY_RUN=1] echoing the exact commands this runner would execute on the cluster:"
    echo ""
    echo "cd \"\$BENCH/benchmark_attn\" && \"\$TRK/.venv/bin/python\" scripts/benchmarks/benchmark_full_model_inference_grid.py \\"
    echo "    --checkpoint ../checkpoints/k16 \\"
    echo "    --variants relative_pos,cached_dist,sparse_mask_k16,sparse_gather_k16,flex_attn \\"
    echo "    --Ns 512,1024,2048,4096,8192,16384,32768 --batch-sizes 1 \\"
    echo "    --repeats 3 --warmup 5 --seed 42 --device cuda --dtype fp16 \\"
    echo "    --coord-layout uniform --linker-mode greedy \\"
    echo "    --out results/full_model_inference_grid_v2.csv \\"
    echo "    --env-json results/full_model_inference_grid_v2_env.json \\"
    echo "    --summary results/full_model_inference_grid_v2_summary.csv"
    echo ""
    echo "cd \"\$BENCH/benchmark_attn\" && \"\$TRK/.venv/bin/python\" scripts/benchmarks/benchmark_full_model_inference_grid.py \\"
    echo "    --checkpoint ../checkpoints/k16 \\"
    echo "    --variants relative_pos,cached_dist,sparse_mask_k16,sparse_gather_k16,flex_attn \\"
    echo "    --Ns 512,1024,2048,4096,8192 --batch-sizes 64 \\"
    echo "    --repeats 3 --warmup 5 --seed 42 --device cuda --dtype fp16 \\"
    echo "    --coord-layout uniform --linker-mode greedy \\"
    echo "    --out results/full_model_inference_grid_b64_v2.csv \\"
    echo "    --env-json results/full_model_inference_grid_b64_v2_env.json \\"
    echo "    --summary results/full_model_inference_grid_b64_v2_summary.csv"
    echo ""
    echo "cd \"\$BENCH/benchmark_attn\" && \"\$TRK/.venv/bin/python\" scripts/benchmarks/benchmark_real_sequence_pipeline.py \\"
    echo "    --checkpoint ../checkpoints/k16 --data-root ../data/vanvliet \\"
    echo "    --target-detections 32768 --device cuda --dtype fp16 --batch-size 4 \\"
    echo "    --seed 42 \\"
    echo "    --out results/real_sequence_pipeline_v2.csv \\"
    echo "    --env-json results/real_sequence_pipeline_v2_env.json \\"
    echo "    --summary results/real_sequence_pipeline_v2_summary.csv"
    echo ""
    echo "[DRY_RUN=1] done; exiting 0 without executing."
    exit 0
fi

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

cd "$BENCH/benchmark_attn" || exit 1
mkdir -p logs results

VARIANTS="relative_pos,cached_dist,sparse_mask_k16,sparse_gather_k16,flex_attn"

echo "=== Full-model inference grid rerun (SPEC 0062 + FlexAttention) ==="
echo "checkpoint: ../checkpoints/k16 (SparseMask K=16)"
echo "variants  : $VARIANTS"
echo "B=1 grid  : N=512,1024,2048,4096,8192,16384,32768"
echo "B=64 grid : N=512,1024,2048,4096,8192"
echo "timing    : repeats=3 warmup=5 seed=42; device=cuda dtype=fp16"
echo "linker    : greedy (build_graph + track_greedy), paper edge threshold 0.05"
echo "scope     : timing only; no accuracy/OOD claim"
echo ""

"$TRK/.venv/bin/python" -c "import torch, sys; print('torch :', torch.__version__); print('cuda  :', torch.version.cuda); print('python:', sys.version.split()[0]); print('GPU   :', torch.cuda.get_device_name(0))"

echo ""

run_grid() {
    local out_csv="$1"
    local env_json="$2"
    local summary_csv="$3"
    local ns="$4"
    local batch_sizes="$5"
    if [ -f "$out_csv" ]; then
        echo "[skip] $out_csv exists — not re-running."
        return
    fi
    "$TRK/.venv/bin/python" scripts/benchmarks/benchmark_full_model_inference_grid.py \
        --checkpoint ../checkpoints/k16 \
        --variants "$VARIANTS" \
        --Ns "$ns" \
        --batch-sizes "$batch_sizes" \
        --repeats 3 \
        --warmup 5 \
        --seed 42 \
        --device cuda \
        --dtype fp16 \
        --coord-layout uniform \
        --linker-mode greedy \
        --out "$out_csv" \
        --env-json "$env_json" \
        --summary "$summary_csv"
}

run_grid \
    results/full_model_inference_grid_v2.csv \
    results/full_model_inference_grid_v2_env.json \
    results/full_model_inference_grid_v2_summary.csv \
    512,1024,2048,4096,8192,16384,32768 \
    1

run_grid \
    results/full_model_inference_grid_b64_v2.csv \
    results/full_model_inference_grid_b64_v2_env.json \
    results/full_model_inference_grid_b64_v2_summary.csv \
    512,1024,2048,4096,8192 \
    64

# Real-sequence aggregate at N=32768. This is the natural end-to-end
# measurement: each sequence is a real video prefix, and the target N is the
# sum over sequences. It is deliberately separate from the synthetic N grid
# so the two protocols cannot be confused.
REAL_CSV="results/real_sequence_pipeline_v2.csv"
if [ -f "$REAL_CSV" ]; then
    echo "[skip] $REAL_CSV exists — real-sequence pipeline not re-run."
else
    "$TRK/.venv/bin/python" scripts/benchmarks/benchmark_real_sequence_pipeline.py \
        --checkpoint ../checkpoints/k16 \
        --data-root ../data/vanvliet \
        --target-detections 32768 \
        --device cuda \
        --dtype fp16 \
        --batch-size 4 \
        --seed 42 \
        --out "$REAL_CSV" \
        --env-json results/real_sequence_pipeline_v2_env.json \
        --summary results/real_sequence_pipeline_v2_summary.csv
fi

echo ""
echo "=== Done ==="
