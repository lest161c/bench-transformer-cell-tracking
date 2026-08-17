#!/bin/bash
#SBATCH --job-name=optimal_train
#SBATCH --account=p_scads_celltracking
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu-h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=logs/optimal_train-%j.out
#SBATCH --error=logs/optimal_train-%j.err
#
# Optimal speed training config.
#
# All optimizations are already the default in the current trackastra code:
#   1. CachedDistAttention  — cdist computed once, shared across 12 layers
#   2. fast-regionprops     — 10-80× faster CPU feature extraction (Weigert 2025)
#   3. vectorized blockwise_norm — scatter_reduce instead of serial Python loop
#
# The dense_flash ablation (job 3920745) tests whether removing the spatial
# cutoff degrades TRA. If TRA is within ±0.001, use --knn_neighbors -2
# (DenseFlashAttention, no mask → FlashAttention-2). Otherwise use -1
# (CachedDistAttention, masked → EfficientAttention).
#
# ─── Usage ──────────────────────────────────────────────────────────
#
#   sbatch scripts/slurm/run_optimal_train.slurm
#   sbatch --export=TEST=1 scripts/slurm/run_optimal_train.slurm
#
# -----------------------------------------------------------------------

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

DATA_DIR="${DATA_DIR}/vanvliet"

cd "$TRK"

if [ "$TEST" = "1" ]; then
    echo "=== TEST MODE: import check ==="
    "$TRK/.venv/bin/python" -c "
import trackastra
from trackastra.model import TrackingTransformer
from trackastra.data.features import extract_features_regionprops
import fast_regionprops
print('trackastra OK')
print('fast_regionprops OK')
print('CUDA:', __import__('torch').cuda.is_available())
"
    echo "=== TEST PASSED ==="
    exit 0
fi

# -----------------------------------------------------------------------
# Optimal training config
# -----------------------------------------------------------------------
PY="$TRK/.venv/bin/python"

$PY scripts/train.py \
    --name optimal_speed \
    --ndim 2 --window 4 --epochs 500 --warmup_epochs 5 \
    --detection_folders TRA \
    --input_train "$DATA_DIR/rpsM" "$DATA_DIR/recA" "$DATA_DIR/pheA" "$DATA_DIR/metA" "$DATA_DIR/cib" "$DATA_DIR/trpL" \
    --input_val "$DATA_DIR/rpsM" "$DATA_DIR/recA" "$DATA_DIR/pheA" \
    --outdir runs --distributed False \
    --batch_size 48 --max_tokens 2048 \
    --d_model 320 --num_encoder_layers 6 --num_decoder_layers 6 \
    --pos_embed_per_dim 32 --feat_embed_per_dim 8 \
    --dropout 0.05 --lr 1e-4 \
    --features wrfeat --causal_norm quiet_softmax \
    --attn_positional_bias rope --spatial_pos_cutoff 256 \
    --delta_cutoff 2 \
    --knn_neighbors -1 \
    --mixedp True --num_workers 16 --seed 42 --augment 3
