#!/bin/bash
#SBATCH --job-name=knn_sweep
#SBATCH --account=p_scads_celltracking
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --array=0-14
#SBATCH --output=logs/knn_sweep-%A_%a.out
#SBATCH --error=logs/knn_sweep-%A_%a.err
#
# Multi-seed K-sweep: 5 K configs × 3 seeds = 15 jobs.
#
# K configs: baseline (dense), K=4, K=16, K=32, K=64
# Seeds: 42, 43, 44
#
# All variants share identical hyperparameters, differing only in
# --knn_neighbors and --seed.  No --init_encoder (all train from
# scratch) to keep the sweep controlled.
#
# Array index mapping:
#   0=baseline_s42   1=baseline_s43   2=baseline_s44
#   3=k4_s42         4=k4_s43         5=k4_s44
#   6=k16_s42        7=k16_s43        8=k16_s44
#   9=k32_s42       10=k32_s43       11=k32_s44
#  12=k64_s42       13=k64_s43       14=k64_s44
#
# ─── Usage ──────────────────────────────────────────────────────────
#
#   sbatch scripts/slurm/run_knn_sweep.slurm
#   sbatch --array=0-2 scripts/slurm/run_knn_sweep.slurm   # baseline only
#
# -----------------------------------------------------------------------

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

DATA_DIR="${DATA_DIR}/vanvliet"

# -----------------------------------------------------------------------
# Array index → (VARIANT, SEED) mapping
# -----------------------------------------------------------------------

VARIANTS=(baseline k4 k16 k32 k64)
SEEDS=(42 43 44)

NUM_VARIANTS=${#VARIANTS[@]}
NUM_SEEDS=${#SEEDS[@]}

VARIANT_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))

VARIANT="${VARIANTS[$VARIANT_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

RUN_NAME="knn_sweep_${VARIANT}_s${SEED}"

echo "=== K-Sweep Array Job ==="
echo "  Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "  Variant:       ${VARIANT}"
echo "  Seed:          ${SEED}"
echo "  Run name:      ${RUN_NAME}"
echo "  Node:          ${SLURM_NODELIST}"
echo "  Started at:    $(date)"
echo ""

# -----------------------------------------------------------------------
# Build EXTRA_ARGS for this variant
# -----------------------------------------------------------------------

case "$VARIANT" in
    baseline)
        KNN_NEIGHBORS=-1
        ;;
    k4)
        KNN_NEIGHBORS=4
        ;;
    k16)
        KNN_NEIGHBORS=16
        ;;
    k32)
        KNN_NEIGHBORS=32
        ;;
    k64)
        KNN_NEIGHBORS=64
        ;;
    *)
        echo "ERROR: Unknown VARIANT '$VARIANT'"
        exit 1
        ;;
esac

# Per-task cache directory to prevent concurrent write corruption
CACHE_DIR=".cache_knn_sweep_${SLURM_ARRAY_TASK_ID}"

# Common args for all variants — identical except knn_neighbors and seed
EXTRA_ARGS="--name ${RUN_NAME} \
    --ndim 2 --window 4 --epochs 500 --warmup_epochs 5 \
    --detection_folders TRA \
    --input_train ${DATA_DIR}/rpsM ${DATA_DIR}/recA ${DATA_DIR}/pheA ${DATA_DIR}/metA ${DATA_DIR}/cib ${DATA_DIR}/trpL \
    --input_val ${DATA_DIR}/rpsM ${DATA_DIR}/recA ${DATA_DIR}/pheA \
    --outdir runs --distributed False \
    --cachedir ${CACHE_DIR} \
    --batch_size 48 --max_tokens 2048 \
    --d_model 320 --num_encoder_layers 6 --num_decoder_layers 6 \
    --pos_embed_per_dim 32 --feat_embed_per_dim 8 \
    --dropout 0.05 --lr 1e-4 \
    --features wrfeat --causal_norm quiet_softmax \
    --attn_positional_bias rope --spatial_pos_cutoff 256 \
    --delta_cutoff 1 \
    --knn_neighbors ${KNN_NEIGHBORS} \
    --mixedp True --num_workers 8 --seed ${SEED} --augment 3"

# -----------------------------------------------------------------------
# Pull trackastra branch and verify env
# -----------------------------------------------------------------------
cd "$TRK"
git pull origin feature/sparse-attention-gather

if [ "$TEST" = "1" ]; then
    echo "=== TEST MODE: import check ==="
    "$TRK/.venv/bin/python" -c "import trackastra; from trackastra.model import TrackingTransformer; print('trackastra OK:', TrackingTransformer); import torch; print('CUDA:', torch.cuda.is_available())"
    echo "=== TEST PASSED ==="
    exit 0
fi

# -----------------------------------------------------------------------
# Normal mode: launch training
# -----------------------------------------------------------------------
PY="$TRK/.venv/bin/python"

echo "  Launching: $PY scripts/train.py $EXTRA_ARGS"
echo ""

$PY scripts/train.py $EXTRA_ARGS

EXIT_CODE=$?
echo ""
echo "  Training exited with code ${EXIT_CODE} at $(date)"

# Clean up per-task cache to avoid 90GB of duplicate caches
rm -rf "${CACHE_DIR}" 2>/dev/null

exit ${EXIT_CODE}
