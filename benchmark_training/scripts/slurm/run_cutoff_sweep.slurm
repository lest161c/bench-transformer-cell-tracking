#!/bin/bash
#SBATCH --job-name=cutoff_sweep
#SBATCH --account=p_scads_celltracking
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --output=logs/cutoff_sweep-%a.out
#SBATCH --error=logs/cutoff_sweep-%a.err
#SBATCH --array=0-4
#
# Spatial cutoff sweep: ablate d_max to map the TRA-vs-cutoff curve.
#
# The Trackastra paper fixes d_max=256px but never ablates it.
# This sweep tests 5 cutoff values.  All runs use CachedDistAttention
# (masked, EfficientAttention fallback) — only the cutoff radius varies.
#
# Cutoff values and field coverage (1009×1305 frame):
#   32px   — 0.2% of field (overly aggressive)
#   64px   — 0.8% of field (aggressive)
#   128px  — 3.1% of field (moderate)
#   512px  — 62% of field (nearly dense)
#   1024px — 98% of field (almost fully dense)
#
# Already running for comparison:
#   256px  — Trackastra default (optimal_train, 500 epochs)
#   ∞      — no mask, FlashAttention-2 (dense_flash, 100 epochs)
#
# All runs use 100 epochs to match dense_flash for fair comparison.
# All other hyperparameters match optimal_train.
#
# ─── Usage ──────────────────────────────────────────────────────────
#
#   sbatch scripts/slurm/run_cutoff_sweep.slurm
#   sbatch --export=TEST=1 scripts/slurm/run_cutoff_sweep.slurm
#
# -----------------------------------------------------------------------

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

DATA_DIR="${DATA_DIR}/vanvliet"

cd "$TRK"

# Map array index to cutoff value
CUTOFFS=(32 64 128 512 1024)
SPATIAL_CUTOFF=${CUTOFFS[$SLURM_ARRAY_TASK_ID]}

echo "=== Cutoff Sweep ==="
echo "  Cutoff:        ${SPATIAL_CUTOFF}px"
echo "  Array task:    ${SLURM_ARRAY_TASK_ID}"
echo "  Node:          ${SLURM_NODELIST}"
echo "  Started:       $(date)"
echo ""

if [ "$TEST" = "1" ]; then
    echo "=== TEST MODE: import + cutoff sweep check ==="
    "$TRK/.venv/bin/python" -c "
import trackastra
from trackastra.model import TrackingTransformer
from trackastra.model.model_parts import CachedDistAttention
print('trackastra OK')
print('CachedDistAttention OK')
cutoffs = [32, 64, 128, 512, 1024]
print(f'Cutoff sweep values: {cutoffs}')
import torch
print('CUDA:', torch.cuda.is_available())
"
    echo "=== TEST PASSED ==="
    exit 0
fi

# -----------------------------------------------------------------------
# Training: same hyperparameters as optimal_train, only cutoff varies
# -----------------------------------------------------------------------
PY="$TRK/.venv/bin/python"

$PY scripts/train.py \
    --name cutoff_sweep_${SPATIAL_CUTOFF} \
    --ndim 2 --window 4 --epochs 100 --warmup_epochs 5 \
    --detection_folders TRA \
    --input_train "$DATA_DIR/rpsM" "$DATA_DIR/recA" "$DATA_DIR/pheA" "$DATA_DIR/metA" "$DATA_DIR/cib" "$DATA_DIR/trpL" \
    --input_val "$DATA_DIR/rpsM" "$DATA_DIR/recA" "$DATA_DIR/pheA" \
    --outdir runs --distributed False \
    --batch_size 48 --max_tokens 2048 \
    --d_model 320 --num_encoder_layers 6 --num_decoder_layers 6 \
    --pos_embed_per_dim 32 --feat_embed_per_dim 8 \
    --dropout 0.05 --lr 1e-4 \
    --features wrfeat --causal_norm quiet_softmax \
    --attn_positional_bias rope --spatial_pos_cutoff ${SPATIAL_CUTOFF} \
    --delta_cutoff 2 \
    --knn_neighbors -1 \
    --mixedp True --num_workers 14 --seed 42 --augment 3

EXIT_CODE=$?
echo ""
echo "  Training exited with code ${EXIT_CODE} at $(date)"
exit ${EXIT_CODE}
