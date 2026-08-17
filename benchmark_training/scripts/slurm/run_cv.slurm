#!/bin/bash
#SBATCH --job-name=cv_masked_dflash
#SBATCH --account=p_scads_celltracking
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --output=logs/cv-%a.out
#SBATCH --error=logs/cv-%a.err
#SBATCH --array=0-11
#
# Proper CV experiment: 6-fold leave-one-condition-out × 2 configs
#
# The confound in the dense_flash vs baseline comparison was:
#   - baseline: window=4, knn_neighbors=-1, dropout=0.05
#   - dense_flash: window=10, knn_neighbors=-2, dropout=0.01
# This isolates the spatial cutoff effect by holding window=4, dropout=0.05
# fixed and varying only knn_neighbors.
#
# Conditions (6 vanvliet datasets):
#   0=rpsM  1=recA  2=pheA  3=metA  4=cib  5=trpL
#
# Array layout:
#   Tasks 0-5: masked (knn=-1), fold 0-5
#   Tasks 6-11: dense_flash (knn=-2), fold 0-5
#
# ─── Usage ──────────────────────────────────────────────────────────
#
#   sbatch scripts/slurm/run_cv.slurm
#
# -----------------------------------------------------------------------

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

DATA_DIR="${DATA_DIR}/vanvliet"

# Vanvliet conditions (6 datasets)
CONDITIONS=(rpsM recA pheA metA cib trpL)

# Determine config and fold from array task ID
if [ "$SLURM_ARRAY_TASK_ID" -lt 6 ]; then
    CONFIG="masked"
    KNN_NEIGHBORS=-1
    FOLD=$SLURM_ARRAY_TASK_ID
else
    CONFIG="dense_flash"
    KNN_NEIGHBORS=-2
    FOLD=$((SLURM_ARRAY_TASK_ID - 6))
fi

HELDOUT_COND=${CONDITIONS[$FOLD]}
# Train conditions: all except the held-out one
TRAIN_CONDS=()
for i in "${!CONDITIONS[@]}"; do
    if [ "$i" -ne "$FOLD" ]; then
        TRAIN_CONDS+=("${CONDITIONS[$i]}")
    fi
done

# Build the train conditions string for the CLI
TRAIN_COND_STR=""
for cond in "${TRAIN_CONDS[@]}"; do
    TRAIN_COND_STR="$TRAIN_COND_STR \"$DATA_DIR/$cond\""
done

# Also build the test conditions for later evaluation
TEST_COND_STR="$DATA_DIR/$HELDOUT_COND"

# Switch to trackastra
cd "$TRK"

# Set environment for the CV run (passed to training via env)
export CV_CONFIG="$CONFIG"
export CV_FOLD="$FOLD"
export CV_HELDOUT="$HELDOUT_COND"
export CV_TRAIN_CONDS="${TRAIN_CONDS[*]}"
export CV_TEST_COND="$TEST_COND_STR"

echo "=== Proper CV: $CONFIG ==="
echo "  Fold:          $FOLD"
echo "  Held-out:      $HELDOUT_COND"
echo "  Train conds:   ${TRAIN_CONDS[*]}"
echo "  knn_neighbors: $KNN_NEIGHBORS"
echo "  Array task:    $SLURM_ARRAY_TASK_ID"
echo "  Node:          ${SLURM_NODELIST}"
echo "  Started:       $(date)"
echo ""

if [ "$TEST" = "1" ]; then
    echo "=== TEST MODE: import + CV setup check ==="
    "$TRK/.venv/bin/python" -c "
import trackastra
from trackastra.model import TrackingTransformer
print('trackastra OK')

# Test expand_experiments
import os
from pathlib import Path
def expand(paths):
    expanded = []
    for p in paths:
        root = Path(p)
        if not root.exists():
            expanded.append(p)
            continue
        subdirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / 'TRA').exists())
        expanded.extend(str(d) for d in subdirs) if subdirs else expanded.append(p)
    return expanded

test = expand(['$DATA_DIR/rpsM', '$DATA_DIR/recA'])
print(f'Train: {len(test)} experiments')
for t in test[:3]:
    print(f'  {t}')
print('  ...')
"
    echo "=== TEST PASSED ==="
    exit 0
fi

PY="$TRK/.venv/bin/python"

# ─── Training ─────────────────────────────────────────────────────
# Train on 5 conditions, with held-out condition excluded
# Both configs use identical hyperparams except knn_neighbors

RUN_NAME="cv_${CONFIG}_fold${FOLD}_${HELDOUT_COND}"

$PY scripts/train.py \
    --name "$RUN_NAME" \
    --ndim 2 --window 4 --epochs 100 --warmup_epochs 5 \
    --detection_folders TRA \
    --input_train $TRAIN_COND_STR \
    --input_val $TRAIN_COND_STR \
    --outdir runs --distributed False \
    --batch_size 48 --max_tokens 2048 \
    --d_model 320 --num_encoder_layers 6 --num_decoder_layers 6 \
    --pos_embed_per_dim 32 --feat_embed_per_dim 8 \
    --dropout 0.05 --lr 1e-4 \
    --features wrfeat --causal_norm quiet_softmax \
    --attn_positional_bias rope --spatial_pos_cutoff 256 \
    --delta_cutoff 2 \
    --knn_neighbors $KNN_NEIGHBORS \
    --mixedp True --num_workers 14 --seed 42 --augment 3

EXIT_CODE=$?
echo ""
echo "  Training exited with code ${EXIT_CODE} at $(date)"

if [ $EXIT_CODE -ne 0 ]; then
    echo "Training failed, skipping evaluation"
    exit $EXIT_CODE
fi

# ─── Evaluation on held-out condition ────────────────────────────
# Find the trained model directory (most recent run with matching name)
echo ""
echo "=== Evaluating $CONFIG model on held-out condition: $HELDOUT_COND ==="

RUN_DIR="$TRK/runs"
MODEL_DIR=""
LATEST_MTIME=0

for dir in $(ls -dt "$RUN_DIR"/2026-*_${RUN_NAME} 2>/dev/null); do
    if [ -f "$dir/model.pt" ]; then
        MODEL_DIR="$dir"
        break
    fi
done

if [ -z "$MODEL_DIR" ]; then
    # Fallback: search by name pattern
    MODEL_DIR=$(find "$RUN_DIR" -name "model.pt" -path "*${RUN_NAME}*" -exec dirname {} \; 2>/dev/null | sort | tail -1)
fi

if [ -z "$MODEL_DIR" ] || [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Could not find model directory for $RUN_NAME"
    echo "  Searched: $RUN_DIR/*${RUN_NAME}*"
    exit 1
fi

echo "  Model: $MODEL_DIR"

# Run evaluation on the held-out condition
EVAL_OUTDIR="$MODEL_DIR/cv_eval"
mkdir -p "$EVAL_OUTDIR"

$PY "$BENCH/benchmark_training/scripts/eval_cv.py" \
    --run-dir "$MODEL_DIR" \
    --heldout "$HELDOUT_COND" \
    --data-root "$DATA_DIR" \
    --output "$EVAL_OUTDIR/eval_results.json" \
    --device cuda 2>&1 | tee "$EVAL_OUTDIR/eval_log.txt"

EVAL_EXIT_CODE=$?
echo ""
echo "  Evaluation exited with code ${EVAL_EXIT_CODE} at $(date)"
exit ${EVAL_EXIT_CODE}
