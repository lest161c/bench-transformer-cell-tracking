#!/bin/bash
#SBATCH --job-name=unified_probe_bn
#SBATCH --account=p_scads_celltracking
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=logs/unified_probe_bn-%j.out
#SBATCH --error=logs/unified_probe_bn-%j.err

# Edge probe (MLP, 5-fold CV) comparing the DynaCLR-trained ScaledCNN+BN
# encoder against the canonical NT-Xent encoder under the identical protocol.
# SPEC 0026 R7 / labbook E14.

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

cd "$BENCH/benchmark_ssl" || exit 1
mkdir -p logs

.venv/bin/python -m src.edge_probing.unified_edge_probe \
    --features cnn_frozen --probe mlp --cv-folds 5 --epochs 200 --standardize \
    --no-cache --shuffle-baseline \
    --data-root "$DATA_DIR/vanvliet" \
    --checkpoint results/checkpoints/cnn_bn100k_step100000.pt \
    --output results/probe_results_bn_dynaclr.json

.venv/bin/python -m src.edge_probing.unified_edge_probe \
    --features cnn_frozen --probe mlp --cv-folds 5 --epochs 200 --standardize \
    --no-cache --shuffle-baseline \
    --data-root "$DATA_DIR/vanvliet" \
    --checkpoint results/checkpoints/cnn/cnn_ntxent_large.pt \
    --output results/probe_results_canonical.json
