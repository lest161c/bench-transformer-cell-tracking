#!/bin/bash
#SBATCH --job-name=edge_probe
#SBATCH --account=p_scads_celltracking
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/edge_probe-%j.out
#SBATCH --error=logs/edge_probe-%j.err

# Config rationale: 7 features span handcrafted-vs-learned,
# frozen-vs-trainable, and PE-augmentation axes. Each tested with both
# linear and MLP probes (7×2=14, full factorial). PE dim and CNN scale
# fixed to match checkpoint.

# Unified edge probing: 5-fold CV over regionprops, HOCT, CNN-NT-Xent,
# CNN-e2e, and DINOv2 features. Tests linear and MLP probes.

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

cd "$BENCH/benchmark_ssl" || exit 1
mkdir -p logs

.venv/bin/python -m src.edge_probing.unified_edge_probe \
    --features all \
    --probe both \
    --epochs 200 \
    --cv-folds 5 \
    --shuffle-baseline \
    --data-root "$DATA_DIR/vanvliet" \
    --output results/unified_probe_results_cv.json

PROBE_EXIT=$?
echo "  Probe exited with code $PROBE_EXIT at $(date)"

if [ -f results/unified_probe_results_cv.json ]; then
    echo "=== RESULTS ==="
    .venv/bin/python -c "
import json
with open('results/unified_probe_results_cv.json') as f:
    data = json.load(f)
for r in data['results']:
    print(f\"{r['feature']:12s} {r['probe']:6s}  bal_acc={r.get('bal_acc',0):.4f}  F1={r.get('F1',0):.4f}\")
"
fi
