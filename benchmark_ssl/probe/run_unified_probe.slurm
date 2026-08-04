#!/bin/bash
#SBATCH --account=p_scads_celltracking
#SBATCH --job-name=edge_probe
#SBATCH --output=logs/edge_probe-%j.out
#SBATCH --error=logs/edge_probe-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

export PIP_REQUIRE_VIRTUALENV=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME

TRK=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra
BENCH=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking
ENV_DIR="$TRK/.venv"

. "$ENV_DIR/bin/activate"
cd "$BENCH"
mkdir -p logs results

echo "═══════════════════════════════════════════════════════════════════════════"
echo "  Unified Edge Probing — $(date)"
echo "  Testing: rp, cnn_frozen, dino × linear, mlp"
echo "═══════════════════════════════════════════════════════════════════════════"

python benchmark_ssl/probe/unified_edge_probe.py \
    --features all \
    --probe both \
    --epochs 200 \
    --cv-folds 5 \
    --shuffle-baseline \
    --data-root /data/cat/ws/mawe985g-data/data/celltracking/vanvliet \
    --output results/unified_probe_results_cv.json

PROBE_EXIT=$?
echo "  Probe exited with code $PROBE_EXIT at $(date)"

# Print results summary
if [ -f results/unified_probe_results_cv.json ]; then
    echo "=== RESULTS ==="
    python -c "
import json
with open('results/unified_probe_results_cv.json') as f:
    data = json.load(f)
for r in data['results']:
    print(f\"{r['feature']:12s} {r['probe']:6s}  bal_acc={r.get('bal_acc',0):.4f}  F1={r.get('F1',0):.4f}\")
"
fi

