#!/bin/bash
#SBATCH --job-name=probe_deepcell_heldout
#SBATCH --account=p_scads_celltracking
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=logs/probe_deepcell_heldout-%j.out
#SBATCH --error=logs/probe_deepcell_heldout-%j.err

# Held-out edge probe on DeepCell TEST frames (never seen by any encoder's
# pre-extraction: it used deepcell/train + deepcell/val only). The probe
# scanner expects the two-level vanvliet layout, so a symlink view presents
# each test sequence as <root>/<seq>/<seq> -> deepcell/test/<seq>.
# SPEC 0026 R7 / labbook E15.

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

cd "$BENCH/benchmark_ssl" || exit 1
mkdir -p logs

PROBE_ROOT="$BENCH/benchmark_ssl/results/deepcell_test_probe_view"
mkdir -p "$PROBE_ROOT"
for seq in 00 01 02 03 04 05 06 07 08 09 10 11; do
  mkdir -p "$PROBE_ROOT/$seq"
  ln -sfn "$DATA_DIR/deepcell/test/$seq" "$PROBE_ROOT/$seq/$seq"
done

CONDS=$(echo 00 01 02 03 04 05 06 07 08 09 10 11 | tr ' ' ',')

.venv/bin/python -m src.edge_probing.unified_edge_probe \
    --features cnn_frozen --probe mlp --cv-folds 5 --epochs 200 --standardize \
    --no-cache --shuffle-baseline \
    --data-root "$PROBE_ROOT" \
    --conditions "$CONDS" \
    --max-pairs 120 \
    --checkpoint results/checkpoints/cnn_bn100k_step100000.pt \
    --output results/probe_results_bn_dynaclr_deepcell_test.json

.venv/bin/python -m src.edge_probing.unified_edge_probe \
    --features cnn_frozen --probe mlp --cv-folds 5 --epochs 200 --standardize \
    --no-cache --shuffle-baseline \
    --data-root "$PROBE_ROOT" \
    --conditions "$CONDS" \
    --max-pairs 120 \
    --checkpoint results/checkpoints/cnn/cnn_ntxent_large.pt \
    --output results/probe_results_canonical_deepcell_test.json

.venv/bin/python -m src.edge_probing.unified_edge_probe \
    --features dino --probe mlp --cv-folds 5 --epochs 200 --standardize \
    --no-cache --shuffle-baseline \
    --data-root "$PROBE_ROOT" \
    --conditions "$CONDS" \
    --max-pairs 120 \
    --output results/probe_results_dino_deepcell_test.json
