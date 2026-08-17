#!/bin/bash
#SBATCH --job-name=vanilla_train
#SBATCH --account=p_scads_celltracking
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --output=logs/vanilla_train-%j.out
#SBATCH --error=logs/vanilla_train-%j.err
#
# Vanilla Trackastra training run.
#
# This runs the ORIGINAL trackastra code (main branch) with NO optimizations:
#   - RelativePositionalAttention (recomputes cdist 12× per forward pass)
#   - skimage.measure.regionprops_table (NOT fast-regionprops)
#   - Serial blockwise_causal_norm (NOT scatter_reduce vectorization)
#
# The optimized run (job 3922452) uses:
#   - CachedDistAttention (cdist once, shared across 12 layers)
#   - fast-regionprops (10-80× faster CPU feature extraction)
#   - Vectorized blockwise_causal_norm (scatter_reduce)
#
# Both runs use identical hyperparameters, same data, same seed (42).
# The comparison gives us the true speedup of our optimizations.
#
# ─── Usage ──────────────────────────────────────────────────────────
#
#   sbatch scripts/slurm/run_vanilla_train.slurm
#   sbatch --export=TEST=1 scripts/slurm/run_vanilla_train.slurm
#
# -----------------------------------------------------------------------

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

DATA_DIR="${DATA_DIR}/vanvliet"

# Switch to the vanilla main branch
cd "$TRK"

echo "=== Vanilla Trackastra Training ==="
echo "  Branch: main"
echo "  Node:   ${SLURM_NODELIST}"
echo "  Started: $(date)"
echo ""

# Check that we're on main branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "ERROR: Expected main branch, got $CURRENT_BRANCH"
    exit 1
fi

if [ "$TEST" = "1" ]; then
    echo "=== TEST MODE: import check ==="
    "$TRK/.venv/bin/python" -c "
import trackastra
from trackastra.model import TrackingTransformer
from trackastra.model.model_parts import RelativePositionalAttention
print('trackastra OK')
print('RelativePositionalAttention OK')
import torch
print('CUDA:', torch.cuda.is_available())
"
    echo "=== TEST PASSED ==="
    exit 0
fi

# -----------------------------------------------------------------------
# Vanilla training: same hyperparameters as optimized run
# -----------------------------------------------------------------------
PY="$TRK/.venv/bin/python"

$PY scripts/train.py \
    --name vanilla_baseline \
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
    --mixedp True --num_workers 14 --seed 42 --augment 3

EXIT_CODE=$?
echo ""
echo "  Training exited with code ${EXIT_CODE} at $(date)"
exit ${EXIT_CODE}
