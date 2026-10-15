#!/bin/bash
#SBATCH --job-name=epoch_profile
#SBATCH --account=p_scads_celltracking
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --array=0-6
#SBATCH --output=logs/epoch_profile-%A_%a.out
#SBATCH --error=logs/epoch_profile-%A_%a.err
#
# Epoch phase-decomposition measurement (SPEC 0041).
#
# Measures where the wall time of a single training epoch goes: data wait
# (dataloader starvation), GPU compute (forward + backward + optimizer
# step), and validation. Consumed by the report's Amdahl accounting
# (tab:e2e_accounting): the epoch-level attention share s and the loader
# share become measured numbers with cross-node spread instead of a
# derivation.
#
# Design (owner decisions, session 2026-10-15):
#   - --exclusive: whole-node allocations; no GPU sharing (the k16_s44
#     K-sweep run showed shared-node contamination). Node name is recorded
#     in the JSON manifest; cross-node spread is the reported uncertainty.
#   - 2 warm-up epochs are excluded from the summary statistics by the
#     callback (CUDA context, cuDNN autotune, Lustre page cache); the cold
#     first epoch is recorded in the rows but not in the medians.
#   - nvidia-smi dmon runs at 1 Hz next to the job (GPU clock, power,
#     temperature); epochs with clock dips can be rejected post hoc.
#
# Array layout (7 tasks):
#   0,1,2,5,6  num_workers=8   : the production pipeline, five nodes
#   3          num_workers=0   : inline-loading ceiling (serial load +
#                                compute per batch; isolates the load cost
#                                against the compute time of the w8 legs)
#   4          num_workers=8 + --profile : PyTorch kernel-level profile of
#                                the same loop, to verify the historical
#                                step composition (backward 57%, attention
#                                22%, norm 13%, FFN 5%, optimizer 2%)
#                                against a real profile
#
# Output (all under $TRK/runs/, git-ignored):
#   runs/epoch_profile_<leg>/epoch_profile.json  (phase decomposition)
#   runs/epoch_profile_t4/profile-*.txt          (task 4 kernel tables)
#   $TRK/logs/dmon_<jobid>_<task>.csv            (GPU state sidecar)
#
# Usage
#   sbatch scripts/slurm/profile_epoch.slurm
#   sbatch --array=0-2 scripts/slurm/profile_epoch.slurm   # w8 legs only
#
# After the run: pull the JSONs into results/epoch_profile/ and regenerate
# the report figure; see SPEC 0041 for the acceptance criteria.

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

DATA_DIR="${DATA_DIR}/vanvliet"

# -----------------------------------------------------------------------
# Array task -> leg configuration
# -----------------------------------------------------------------------

case "$SLURM_ARRAY_TASK_ID" in
    3)
        LEG="w0"
        NUM_WORKERS=0
        EPOCHS=5
        EXTRA_ARGS=""
        ;;
    4)
        LEG="prof"
        NUM_WORKERS=8
        EPOCHS=20
        EXTRA_ARGS="--profile"
        ;;
    *)
        LEG="w8_t${SLURM_ARRAY_TASK_ID}"
        NUM_WORKERS=8
        EPOCHS=20
        EXTRA_ARGS=""
        ;;
esac

RUN_NAME="epoch_profile_${LEG}_s42"
OUT_JSON="runs/${RUN_NAME}/epoch_profile.json"
# Per-task cache directory, as in run_knn_sweep.slurm: concurrent training
# processes must not share a feature cache.
CACHE_DIR=".cache_epoch_profile_${SLURM_ARRAY_TASK_ID}"

echo "=== Epoch Phase-Decomposition (SPEC 0041) ==="
echo "  Task:      ${SLURM_ARRAY_TASK_ID}"
echo "  Leg:       ${LEG} (num_workers=${NUM_WORKERS}, epochs=${EPOCHS})"
echo "  Node:      ${SLURM_NODELIST}"
echo "  Run name:  ${RUN_NAME}"
echo "  Started:   $(date)"
echo ""

# -----------------------------------------------------------------------
# GPU state sidecar: 1 Hz sampling of clock / power / utilization / memory.
# Epochs with clock dips below the steady level can be rejected post hoc;
# the CSV is the audit trail for machine-state claims.
# -----------------------------------------------------------------------

DMON_CSV="$TRK/logs/dmon_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"
nvidia-smi dmon -s pucm -d 1 -o T > "$DMON_CSV" 2>/dev/null &
DMON_PID=$!
trap 'kill "$DMON_PID" 2>/dev/null' EXIT

# -----------------------------------------------------------------------
# Training: the standard multi-seed K-sweep configuration (run_knn_sweep
# protocol), K=16 seed 42, shortened to the profiling horizon. The LR
# warm-up is disabled (--warmup_epochs 0): the learning rate does not
# change the phase composition, and a 20-epoch run does not need it.
# -----------------------------------------------------------------------

cd "$TRK"

"$TRK/.venv/bin/python" scripts/train.py \
    --name "${RUN_NAME}" \
    --ndim 2 --window 4 --epochs "${EPOCHS}" --warmup_epochs 0 \
    --detection_folders TRA \
    --input_train "${DATA_DIR}/rpsM" "${DATA_DIR}/recA" "${DATA_DIR}/pheA" "${DATA_DIR}/metA" "${DATA_DIR}/cib" \
    --input_val "${DATA_DIR}/rpsM" "${DATA_DIR}/recA" "${DATA_DIR}/pheA" \
    --outdir runs --distributed False \
    --cachedir "${CACHE_DIR}" \
    --batch_size 48 --max_tokens 2048 \
    --d_model 320 --num_encoder_layers 6 --num_decoder_layers 6 \
    --pos_embed_per_dim 32 --feat_embed_per_dim 8 \
    --dropout 0.05 --lr 1e-4 \
    --features wrfeat --causal_norm quiet_softmax \
    --attn_positional_bias rope --spatial_pos_cutoff 256 \
    --delta_cutoff 1 \
    --knn_neighbors 16 \
    --mixedp True --num_workers "${NUM_WORKERS}" --seed 42 --augment 3 \
    --profile_epoch "${OUT_JSON}" \
    ${EXTRA_ARGS}

EXIT_CODE=$?
echo ""
echo "  Training exited with code ${EXIT_CODE} at $(date)"
echo "  JSON: ${TRK}/${OUT_JSON}"
echo "  dmon: ${DMON_CSV}"

# Clean up the per-task cache (90 GB class, as in run_knn_sweep.slurm).
rm -rf "${CACHE_DIR}" 2>/dev/null

exit ${EXIT_CODE}
