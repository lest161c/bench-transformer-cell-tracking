#!/bin/bash
#SBATCH --job-name=dense_flash
#SBATCH --account=p_scads_celltracking
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=logs/dense_flash-%j.out
#SBATCH --error=logs/dense_flash-%j.err
#
# Ablation: dense FlashAttention (no spatial cutoff) vs dense masked
# (with spatial cutoff).  This variant uses DenseFlashAttention which
# calls F.scaled_dot_product_attention with NO mask, getting the
# FlashAttention-2 kernel.
#
# The question: does removing the spatial cutoff degrade TRA?
# If TRA is within ±0.001 of the masked baseline, dense_flash is
# optimal and sparse attention is moot.
#
# This requires the trackastra branch to have DenseFlashAttention
# wired into the model.  Verify before submitting:
#
#   grep -r "DenseFlashAttention\|dense_flash" "$TRK/trackastra/"
#
# ─── Usage ──────────────────────────────────────────────────────────
#
#   sbatch scripts/slurm/run_dense_flash.slurm
#   sbatch --export=TEST=1 scripts/slurm/run_dense_flash.slurm
#
# -----------------------------------------------------------------------

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

DATA_DIR="${DATA_DIR}/vanvliet"

cd "$TRK"

if [ "$TEST" = "1" ]; then
    echo "=== TEST MODE: import check ==="
    "$TRK/.venv/bin/python" -c "import trackastra; from trackastra.model import TrackingTransformer; print('trackastra OK:', TrackingTransformer); import torch; print('CUDA:', torch.cuda.is_available())"
    echo "=== TEST PASSED ==="
    exit 0
fi

# -----------------------------------------------------------------------
# Normal mode: launch training with dense flash attention
# -----------------------------------------------------------------------
PY="$TRK/.venv/bin/python"

$PY scripts/train.py \
    --name dense_flash_no_cutoff \
    --ndim 2 --window 10 --epochs 100 --warmup_epochs 10 \
    --detection_folders TRA \
    --input_train "$DATA_DIR/rpsM" "$DATA_DIR/recA" "$DATA_DIR/pheA" "$DATA_DIR/metA" "$DATA_DIR/cib" "$DATA_DIR/trpL" \
    --input_val "$DATA_DIR/rpsM" "$DATA_DIR/recA" "$DATA_DIR/pheA" \
    --outdir runs --distributed False \
    --batch_size 48 --max_tokens 2048 \
    --d_model 320 --num_encoder_layers 6 --num_decoder_layers 6 \
    --pos_embed_per_dim 32 --feat_embed_per_dim 8 \
    --dropout 0.01 --lr 1e-4 \
    --features wrfeat --causal_norm quiet_softmax \
    --attn_positional_bias rope --spatial_pos_cutoff 256 \
    --delta_cutoff 2 \
    --knn_neighbors -2 \
    --mixedp True --num_workers 8 --seed 42 --augment 3
