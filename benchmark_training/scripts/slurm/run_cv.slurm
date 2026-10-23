#!/bin/bash
#SBATCH --job-name=cv500_masked_dflash
#SBATCH --account=p_scads_celltracking
#SBATCH --partition=capella
# 500-epoch tasks estimated at 21-31 h (100-epoch tasks measured 4.3-6.3 h,
# jobs 4428930/4429851); 40 h gives headroom. normal QOS MaxWall is 7 days.
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
# Cluster QOS rejects --gpus-per-node submissions with QOSMinGRES (verified live 2026-09-24).
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --output=logs/cv-%a.out
#SBATCH --error=logs/cv-%a.err
#SBATCH --array=0-11
#
# Proper CV experiment: 6-fold leave-one-condition-out × 2 configs
#
# 500-epoch config-matched re-run (2026-10): the previously published CV
# ablation (tab:cv_results) trained at 100 epochs, so its absolute AOGM/TRA
# are not comparable to the 500-epoch baseline regime (optimal_train and the
# K-sweep, run_vanilla_train.slurm / run_knn_sweep.slurm). This campaign
# raises the CV arms to the baseline's 500 epochs; the LOCO train-set size
# (5 of 6 conditions) and in-sample validation (val == train) are kept,
# because they are inherent to the disclosed CV protocol and shared by both
# arms, so the paired DenseFlash-vs-CachedDistAttention comparison stays fair.
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
# ─── Locked evaluation protocol (read before interpreting tab:cv_results) ─
#
#   * The held-out condition is excluded from BOTH --input_train and --input_val
#     (this is the leave-one-condition-out fold; see TRAIN_PATHS below).
#   * --input_val == --input_train is DELIBERATE (in-sample validation): the
#     test condition stays unseen and both arms share the protocol, so the
#     ablation is fair; but validation/best-checkpoint choice is not independent
#     of training data. This differs from the K-sweep (which validates on 3
#     conditions) and must be disclosed with the table.
#   * Seed is set per array campaign via --export=CV_SEED (42 and 43 to match
#     the two-seed structure of tab:cv_results). Each seed's 6-fold CV alone
#     CANNOT quantify training stochasticity; the reported ± is
#     across-condition spread, not seed noise.
#   * Cost ≈ 312 GPU-h per seed at 500 epochs (median ~26 h/task; ~31 h wall at
#     12 concurrent H100s), within the 40 h limit.
#
# ─── Usage ──────────────────────────────────────────────────────────
#
#   sbatch --export=CV_SEED=42 scripts/slurm/run_cv.slurm
#   sbatch --export=CV_SEED=43 \
#          --output=logs/cv-s43-%a.out --error=logs/cv-s43-%a.err \
#          scripts/slurm/run_cv.slurm
#
#   * sbatch does NOT accumulate repeated --export flags (the last one
#     replaces all previous ones — verified live 2026-10-23, when a
#     "--export=TEST=1 --export=CV_SEED=42" submission silently dropped
#     TEST=1). Combine values comma-separated: --export=TEST=1,CV_SEED=42.
#   * The header's --output/--error (logs/cv-%a.out) is shared by every
#     array, so concurrent seed arrays MUST override both on the command
#     line (seed-suffixed names, as above) or they clobber each other's
#     logs. The first submitted array may keep the default names.
#
# Both seed arrays may run concurrently (24 tasks / 24 H100s, well under the
# normal-QOS 284-GPU user cap): cache dirs and run names are seed-suffixed,
# so no state is shared between the arrays.
#
# -----------------------------------------------------------------------

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

DATA_DIR="${DATA_DIR}/vanvliet"

# Vanvliet conditions (6 datasets)
CONDITIONS=(rpsM recA pheA metA cib trpL)

# Seed of this array campaign (default preserves the historical seed 42).
SEED="${CV_SEED:-42}"

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

# ─── DenseFlashAttention guard (dense_flash arm only) ────────────────
# knn_neighbors=-2 must map to DenseFlashAttention. Upstream Trackastra has
# no such class; it is added to the $TRK checkout by install_dense_flash.py,
# which rewrites the model.py factory to branch on -2 and to raise on any
# unsupported value. Without that install, -2 silently falls back to
# CachedDistAttention and BOTH ablation arms become identical (the audited
# B1 failure mode). Fail loudly here instead.
if [ "$CONFIG" = "dense_flash" ]; then
    if ! grep -q "class DenseFlashAttention" "$TRK/trackastra/model/model_parts.py" \
       || ! grep -q "if knn_neighbors == -2:" "$TRK/trackastra/model/model.py"; then
        echo "ERROR: DenseFlashAttention is not installed in $TRK."
        echo "       knn_neighbors=-2 would silently fall back to CachedDistAttention,"
        echo "       making the dense_flash and masked arms identical."
        echo "       Install it (idempotent, self-verifying) with:"
        echo "         python $BENCH/benchmark_training/scripts/slurm/install_dense_flash.py \\"
        echo "             --trackastra-root $TRK/trackastra"
        exit 1
    fi
fi

# Per-task cache directory: concurrent array tasks must NOT share the
# default $TRK/.cache (concurrent-write corruption). Mirrors
# run_knn_sweep.slurm's per-task --cachedir; removed after the run.
# Seed-suffixed so the seed-42 and seed-43 arrays can run concurrently.
CACHE_DIR=".cache_cv_${CONFIG}_fold${FOLD}_s${SEED}"
# Always drop this task's cache on exit (avoids ~76 GB of duplicate caches).
# Runs in $TRK (cd below); CACHE_DIR is relative to it.
trap 'rm -rf "$CACHE_DIR" 2>/dev/null' EXIT

HELDOUT_COND=${CONDITIONS[$FOLD]}
# Train conditions: all except the held-out one
TRAIN_CONDS=()
for i in "${!CONDITIONS[@]}"; do
    if [ "$i" -ne "$FOLD" ]; then
        TRAIN_CONDS+=("${CONDITIONS[$i]}")
    fi
done

# Build the train conditions array (use array expansion for proper quoting)
TRAIN_PATHS=()
for cond in "${TRAIN_CONDS[@]}"; do
    TRAIN_PATHS+=("$DATA_DIR/$cond")
done

# Also build the test condition path for later evaluation
TEST_PATH="$DATA_DIR/$HELDOUT_COND"

# Switch to trackastra
cd "$TRK"

# Set environment for the CV run (passed to training via env)
export CV_CONFIG="$CONFIG"
export CV_FOLD="$FOLD"
export CV_HELDOUT="$HELDOUT_COND"
export CV_TRAIN_CONDS="${TRAIN_CONDS[*]}"
export CV_TEST_PATH="$TEST_PATH"

echo "=== Proper CV (500 epochs): $CONFIG ==="
echo "  Fold:          $FOLD"
echo "  Held-out:      $HELDOUT_COND"
echo "  Train conds:   ${TRAIN_CONDS[*]}"
echo "  knn_neighbors: $KNN_NEIGHBORS"
echo "  Seed:          $SEED"
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

# Training interpreter: trackastra's own venv (has torch/trackastra, no traccuracy).
PY="$TRK/.venv/bin/python"

# Evaluation interpreter: eval_cv.py imports traccuracy, which is NOT in
# $TRK/.venv. It lives in benchmark_training/.venv (same venv documented in
# run_eval_multiseed.slurm's VENV REQUIREMENT and run_knn_sweep.slurm:196).
EVAL_PY="$BENCH/benchmark_training/.venv/bin/python"
if [ ! -x "$EVAL_PY" ]; then
    echo "ERROR: eval venv missing: $EVAL_PY"
    echo "       (needs trackastra editable + traccuracy; see run_eval_multiseed.slurm)"
    exit 1
fi

# ─── Training ─────────────────────────────────────────────────────
# Train on 5 conditions, with held-out condition excluded
# Both configs use identical hyperparams except knn_neighbors

RUN_NAME="cv500_${CONFIG}_fold${FOLD}_${HELDOUT_COND}_s${SEED}"

$PY scripts/train.py \
    --name "$RUN_NAME" \
    --ndim 2 --window 4 --epochs 500 --warmup_epochs 5 \
    --detection_folders TRA \
    --input_train "${TRAIN_PATHS[@]}" \
    --input_val "${TRAIN_PATHS[@]}" \
    --outdir runs --distributed False \
    --cachedir "$CACHE_DIR" \
    --batch_size 48 --max_tokens 2048 \
    --d_model 320 --num_encoder_layers 6 --num_decoder_layers 6 \
    --pos_embed_per_dim 32 --feat_embed_per_dim 8 \
    --dropout 0.05 --lr 1e-4 \
    --features wrfeat --causal_norm quiet_softmax \
    --attn_positional_bias rope --spatial_pos_cutoff 256 \
    --delta_cutoff 2 \
    --knn_neighbors $KNN_NEIGHBORS \
    --mixedp True --num_workers 14 --seed $SEED --augment 3

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

"$EVAL_PY" "$BENCH/benchmark_training/scripts/eval_cv.py" \
    --run-dir "$MODEL_DIR" \
    --heldout "$HELDOUT_COND" \
    --data-root "$DATA_DIR" \
    --output "$EVAL_OUTDIR/eval_results.json" \
    --device cuda 2>&1 | tee "$EVAL_OUTDIR/eval_log.txt"

EVAL_EXIT_CODE=$?
echo ""
echo "  Evaluation exited with code ${EVAL_EXIT_CODE} at $(date)"
exit ${EVAL_EXIT_CODE}
