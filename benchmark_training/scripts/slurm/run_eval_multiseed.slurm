#!/bin/bash
#SBATCH --job-name=eval_multiseed
#SBATCH --account=p_scads_celltracking
#SBATCH --partition=capella
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --array=0-20
#SBATCH --output=logs/eval_multiseed-%A_%a.out
#SBATCH --error=logs/eval_multiseed-%A_%a.err
#
# Multi-seed K-sweep evaluation: 21 (config, seed) checkpoints × 6 conditions
# -> 126 condition evaluations, each producing one output file under the run
# directory. Feeds the report's per-config mean+-std table (SPEC 0008).
#
# PURPOSE
#   Convert the 21-model multi-seed training sweep (run_knn_sweep.slurm; the
#   K=2/8 variants were trained via VARIANTS_OVERRIDE, SPEC 0027) into
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
# RESUMABLE (SPEC 0039)
#   A condition is skipped only when BOTH eval_<COND>.csv and its provenance
#   sidecar (eval_<COND>.csv.provenance.json, written by eval_cv.py) exist AND
#   the sidecar's checkpoint md5 matches the md5 of the run's current
#   model.pt. Otherwise the stale csv is deleted and the condition is
#   re-evaluated. This makes it impossible for a resubmission to silently keep
#   results of unknown provenance (the defect that corrupted the 2026-10 sweep).
#
# ERA GUARD (SPEC 0039)
#   The resolved run dir's train_config.yaml must NOT contain the held-out
#   condition trpL in its training paths. The pre-2026-10-07 (leaked) campaign
#   dirs share the (variant, seed) naming with the fixed ones; binding an eval
#   task to a leaked dir would poison the held-out column. eval_cv.py enforces
#   the same check independently.
#
# Usage:
#   sbatch scripts/slurm/run_eval_multiseed.slurm
#   sbatch --array=0-2 scripts/slurm/run_eval_multiseed.slurm   # baseline only
# -----------------------------------------------------------------------

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

# ─── Array index -> (VARIANT, SEED) ─────────────────────────────────
# Variant order matches the aggregator CONFIGS and the report table
# (baseline, k2, k4, k8, k16, k32, k64); seeds minor
# (k2/k8 trained via run_knn_sweep.slurm VARIANTS_OVERRIDE, SPEC 0027).
# VARIANT_IDX = ID / NUM_SEEDS, SEED_IDX = ID % NUM_SEEDS.

VARIANTS=(baseline k2 k4 k8 k16 k32 k64)
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

echo "  Candidates (newest first):"
RUN=""
for candidate in $(ls -dt "$TRK"/runs/*_knn_sweep_${VARIANT}_s${SEED} 2>/dev/null); do
    if [ ! -f "$candidate/model.pt" ]; then
        echo "    [incomplete] $candidate"
        continue
    fi
    if grep -qE "^[[:space:]]*-[[:space:]]*/.*trpL" "$candidate/train_config.yaml" 2>/dev/null; then
        echo "    [LEAKED ERA] $candidate (train_config.yaml contains trpL paths — skipped)"
        continue
    fi
    echo "    [model]      $candidate"
    if [ -z "$RUN" ]; then
        RUN="$candidate"
    fi
done
if [ -z "$RUN" ]; then
    echo "ERROR: no completed fixed-era run dir (with model.pt, no trpL in config)"
    echo "       for ${VARIANT}_s${SEED}. Bind manually after inspecting candidates."
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
    PROV="$OUT.provenance.json"
    if [ -s "$OUT" ]; then
        # Skip only when the recorded provenance matches the current model.pt
        # (SPEC 0039); otherwise the file is stale/unknown and gets re-evaluated.
        CURRENT_MD5=$(md5sum "$RUN/model.pt" | cut -d' ' -f1)
        RECORDED_MD5=$("$PY" -c "
import json
print(json.load(open('$PROV'))['checkpoint_files'][0]['md5'])
" 2>/dev/null || echo "")
        if [ -n "$RECORDED_MD5" ] && [ "$RECORDED_MD5" = "$CURRENT_MD5" ]; then
            echo "  Skip $COND (provenance matches current model: ${PROV})"
            continue
        fi
        echo "  Stale $COND (missing or mismatched provenance) — re-evaluating"
        rm "$OUT"
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
