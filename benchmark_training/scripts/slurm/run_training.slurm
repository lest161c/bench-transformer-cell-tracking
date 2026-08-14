#!/bin/bash
#SBATCH --job-name=tr_train
#SBATCH --account=p_scads_celltracking
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=logs/slurm-train-%j.out
#SBATCH --error=logs/slurm-train-%j.err
#
# Unified Trackastra training script.
#
# Config rationale: K=4,16,32 sweep sparsity (ablation shows K=4 ≈ K=16
# final loss). cached_dist isolates branch effects from sparsity.
# sparse_k16_ssl tests SSL×sparse at scale. K=8,64,128 omitted: no
# ablation evidence (K=8) or approach dense (K≥64).
#
# ─── Usage ──────────────────────────────────────────────────────────
#
#   sbatch --export=VARIANT=baseline scripts/slurm/run_training.slurm
#   sbatch --export=VARIANT=sparse_k16 scripts/slurm/run_training.slurm
#   sbatch --export=VARIANT=sparse_k16_ssl,TEST=1 scripts/slurm/run_training.slurm
#
# TEST=1 runs the environment setup and a Python import check without
# training.  See scripts/slurm/run_diagnostic.slurm for diagnostic
# and quick-bench runs.
#
# -----------------------------------------------------------------------

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

CFG_DIR="$REPO_ROOT/benchmark_training/configs"
DATA_DIR="${DATA_DIR}/vanvliet"

VARIANT="${VARIANT:-baseline}"

# -----------------------------------------------------------------------
# Variant table: branch, config file, extra CLI args
# -----------------------------------------------------------------------

case "$VARIANT" in
    baseline)
        BRANCH="feature/sparse-attention-gather"
        CONFIG="$CFG_DIR/vanvliet_baseline.yaml"
        EXTRA_ARGS=""
        ;;
    cached_dist)
        BRANCH="cached-dist-attn"
        CONFIG="$CFG_DIR/vanvliet_baseline.yaml"
        EXTRA_ARGS=""
        ;;
    sparse_k4)
        BRANCH="feature/sparse-attention-gather"
        CONFIG=""
        EXTRA_ARGS="--name sparse_k4 --ndim 2 --window 10 --epochs 100 --warmup_epochs 10 \
            --detection_folders TRA \
            --input_train $DATA_DIR/rpsM $DATA_DIR/recA $DATA_DIR/pheA $DATA_DIR/metA $DATA_DIR/cib $DATA_DIR/trpL \
            --input_val $DATA_DIR/rpsM $DATA_DIR/recA $DATA_DIR/pheA \
            --outdir runs --distributed False \
            --batch_size 48 --max_tokens 4096 \
            --d_model 320 --num_encoder_layers 6 --num_decoder_layers 6 \
            --pos_embed_per_dim 32 --feat_embed_per_dim 8 \
            --dropout 0.01 --lr 1e-4 \
            --features wrfeat --causal_norm quiet_softmax \
            --attn_positional_bias rope --spatial_pos_cutoff 256 \
            --delta_cutoff 2 --knn_neighbors 4 \
            --mixedp True --num_workers 8 --seed 42 --augment 3"
        ;;
    sparse_k16)
        BRANCH="feature/sparse-attention-gather"
        CONFIG=""
        EXTRA_ARGS="--name sparse_k16 --ndim 2 --window 10 --epochs 100 --warmup_epochs 10 \
            --detection_folders TRA \
            --input_train $DATA_DIR/rpsM $DATA_DIR/recA $DATA_DIR/pheA $DATA_DIR/metA $DATA_DIR/cib $DATA_DIR/trpL \
            --input_val $DATA_DIR/rpsM $DATA_DIR/recA $DATA_DIR/pheA \
            --outdir runs --distributed False \
            --batch_size 48 --max_tokens 4096 \
            --d_model 320 --num_encoder_layers 6 --num_decoder_layers 6 \
            --pos_embed_per_dim 32 --feat_embed_per_dim 8 \
            --dropout 0.01 --lr 1e-4 \
            --features wrfeat --causal_norm quiet_softmax \
            --attn_positional_bias rope --spatial_pos_cutoff 256 \
            --delta_cutoff 2 --knn_neighbors 16 \
            --init_encoder runs/ssl_dino_pretrain \
            --logger wandb --wandb_project trackastra \
            --mixedp True --num_workers 8 --seed 42 --augment 3"
        ;;
    sparse_k32)
        BRANCH="feature/sparse-attention-gather"
        CONFIG=""
        EXTRA_ARGS="--name sparse_k32 --ndim 2 --window 10 --epochs 100 --warmup_epochs 10 \
            --detection_folders TRA \
            --input_train $DATA_DIR/rpsM $DATA_DIR/recA $DATA_DIR/pheA $DATA_DIR/metA $DATA_DIR/cib $DATA_DIR/trpL \
            --input_val $DATA_DIR/rpsM $DATA_DIR/recA $DATA_DIR/pheA \
            --outdir runs --distributed False \
            --batch_size 48 --max_tokens 4096 \
            --d_model 320 --num_encoder_layers 6 --num_decoder_layers 6 \
            --pos_embed_per_dim 32 --feat_embed_per_dim 8 \
            --dropout 0.01 --lr 1e-4 \
            --features wrfeat --causal_norm quiet_softmax \
            --attn_positional_bias rope --spatial_pos_cutoff 256 \
            --delta_cutoff 2 --knn_neighbors 32 \
            --mixedp True --num_workers 8 --seed 42 --augment 3"
        ;;
    sparse_k16_ssl)
        BRANCH="feature/sparse-attention-gather"
        CONFIG="$CFG_DIR/vanvliet_sparse_k16_ssl.yaml"
        EXTRA_ARGS="--ssl_pretrain True"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        ;;
    *)
        echo "ERROR: Unknown VARIANT '$VARIANT'"
        echo "Valid variants: baseline cached_dist sparse_k4 sparse_k16 sparse_k32 sparse_k16_ssl"
        exit 1
        ;;
esac

# -----------------------------------------------------------------------
# Pull trackastra branch and verify env
# -----------------------------------------------------------------------
cd "$TRK"
git pull origin "$BRANCH"

if [ "$TEST" = "1" ]; then
    echo "=== TEST MODE: import check ==="
    "$REPO_ROOT/benchmark_training/.venv/bin/python" -c "import trackastra; from trackastra.model import TrackingTransformer; print('trackastra OK:', TrackingTransformer); import torch; print('CUDA:', torch.cuda.is_available())"
    echo "=== TEST PASSED ==="
    exit 0
fi

# -----------------------------------------------------------------------
# Normal mode: launch training
# -----------------------------------------------------------------------
PY="$REPO_ROOT/benchmark_training/.venv/bin/python"

if [ -n "$CONFIG" ]; then
    $PY scripts/train.py --config "$CONFIG" $EXTRA_ARGS
else
    $PY scripts/train.py $EXTRA_ARGS
fi
