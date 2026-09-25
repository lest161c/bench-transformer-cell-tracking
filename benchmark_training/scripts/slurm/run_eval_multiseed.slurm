#!/bin/bash
#SBATCH --job-name=eval_multiseed
#SBATCH --account=p_scads_celltracking
#SBATCH --partition=capella
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --array=0-14
#SBATCH --output=logs/eval_multiseed-%A_%a.out
#SBATCH --error=logs/eval_multiseed-%A_%a.err
#
# Multi-seed K-sweep evaluation: 15 (config, seed) checkpoints × 6 conditions
# -> 90 condition evaluations, each producing one output file under the run
# directory. Feeds the report's per-config mean+-std table (SPEC 0008).
#
# PURPOSE
#   Convert the 15-model multi-seed training sweep (run_knn_sweep.slurm) into
#   per-condition TRA/AOGM metrics, one output file per condition in the run
#   directory. Aggregated into the per-config mean+-std table that replaces
#   the single-seed claims in the report.
#
# EVAL PROTOCOL (locked by SPEC 0008; mirrors the historical CV sweep)
#   * 6 conditions = 32 experiments per checkpoint:
#       rpsM, recA, pheA, metA, cib, trpL
#   * greedy linker (eval_cv.py default --mode greedy)
#   * GT detections from each condition's TRA/ directory
#   * --max-distance 50 (eval_cv.py default)
#   * --device cuda (one GPU per task)
#
# ARRAY INDEX -> (VARIANT, SEED) mapping — MUST match run_knn_sweep.slurm
#   0=baseline_s42   1=baseline_s43   2=baseline_s44
#   3=k4_s42         4=k4_s43         5=k4_s44
#   6=k16_s42        7=k16_s43        8=k16_s44
#   9=k32_s42       10=k32_s43       11=k32_s44
#  12=k64_s42       13=k64_s43       14=k64_s44
#
# VENV REQUIREMENT
#   Python is sourced from the venv at benchmark_training/.venv. It must
#   contain trackastra (editable, pointing at $TRK) and traccuracy. uv venvs
#   have no pip binary; install from benchmark_training/ with:
#
#       uv pip install --python .venv/bin/python -e $TRK
#       uv pip install --python .venv/bin/python traccuracy
#
# CROSS-BUILD WALL-TIME NOTE
#   Per-epoch training wall-times must NOT be compared across builds
#   (~3.5 vs 1.63 min/epoch environmental difference). Per-build times
#   are recorded in results/knn_sweep_multi/README.md only.
#
# OUTPUT FORMAT NOTE
#   eval_cv.py writes JSON regardless of the --output extension; the
#   per-condition "CSV" files (eval_<COND>.csv) contain JSON. Downstream
#   aggregation must parse JSON, not CSV.
#
# RESUMABLE
#   If $RUN/eval_<COND>.csv already exists and is non-empty, that condition
#   is skipped on resubmission. Re-running the same array task only fills
#   in missing conditions.
#
# Usage:
#   sbatch scripts/slurm/run_eval_multiseed.slurm
#   sbatch --array=0-2 scripts/slurm/run_eval_multiseed.slurm   # baseline only
# -----------------------------------------------------------------------

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

# ─── Array index -> (VARIANT, SEED) ─────────────────────────────────
# Mirrors run_knn_sweep.slurm exactly: VARIANTS=(baseline k4 k16 k32 k64),
# SEEDS=(42 43 44), VARIANT_IDX = ID / NUM_SEEDS, SEED_IDX = ID % NUM_SEEDS.

VARIANTS=(baseline k4 k16 k32 k64)
SEEDS=(42 43 44)
NUM_SEEDS=${#SEEDS[@]}

VARIANT_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))

VARIANT="${VARIANTS[$VARIANT_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
RUN_NAME="knn_sweep_${VARIANT}_s${SEED}"

echo "=== Multi-seed Eval ==="
echo "  Task:    ${SLURM_ARRAY_TASK_ID}"
echo "  Variant: ${VARIANT}"
echo "  Seed:    ${SEED}"
echo "  Run:     ${RUN_NAME}"
echo ""

# ─── Resolve run dir by glob ───────────────────────────────────────
# Run dirs follow <timestamp>_<run_name> under $TRK/runs/. OOM-killed
# training attempts are auto-requeued by SLURM into FRESH timestamped dirs
# (the training launcher never resumes into an existing dir), so one
# (variant, seed) pair can have several candidate dirs of which only the
# newest holds a completed model. Pick the NEWEST candidate containing
# model.pt; error clearly if none exists.

RUN=""
for candidate in $(ls -dt "$TRK"/runs/*_knn_sweep_${VARIANT}_s${SEED} 2>/dev/null); do
    if [ -f "$candidate/model.pt" ]; then
        RUN="$candidate"
        break
    fi
done
if [ -z "$RUN" ]; then
    echo "ERROR: no completed run dir (with model.pt) for ${VARIANT}_s${SEED}"
    exit 1
fi
echo "  Run dir: ${RUN}"
echo ""

# ─── Per-condition eval loop ───────────────────────────────────────
# Skip if $RUN/eval_<COND>.csv exists and is non-empty (resumable).
PY="$BENCH/benchmark_training/.venv/bin/python"
CONDITIONS=(rpsM recA pheA metA cib trpL)

for COND in "${CONDITIONS[@]}"; do
    OUT="$RUN/eval_${COND}.csv"
    if [ -s "$OUT" ]; then
        echo "  Skip $COND (already evaluated: ${OUT})"
        continue
    fi
    echo "=== Eval ${RUN_NAME} on ${COND} ==="
    "$PY" "$BENCH/benchmark_training/scripts/eval_cv.py" \
        --run-dir "$RUN" \
        --heldout "$COND" \
        --data-root "$DATA_DIR/vanvliet" \
        --output "$OUT" \
        --device cuda
done

echo ""
echo "=== Done at $(date) ==="
exit 0
