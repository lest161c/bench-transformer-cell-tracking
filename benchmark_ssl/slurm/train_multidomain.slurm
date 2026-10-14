#!/bin/bash
#SBATCH --job-name=cnn_ssl_multidomain
#SBATCH --account=p_scads_celltracking
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=logs/cnn_ssl_multidomain-%j.out
#SBATCH --error=logs/cnn_ssl_multidomain-%j.err

# Multi-domain 2D NT-Xent pretraining (SPEC 0026 R4).
# Domains: vanvliet + DeepCell train+val. Fluo-N2DL-HeLa is NOT used (held out).
# Corrected objective: projection head, cross-frame negatives, tau=0.5.
# H100, scale=large, batch 1024, 500 steps. Submission is owner-gated.

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

cd "$BENCH/benchmark_ssl" || exit 1
mkdir -p logs results/runs/multidomain_large

.venv/bin/python -m src.cnn_encoder.cnn_ssl \
    --loss ntxent \
    --scale large \
    --steps 500 \
    --max-frames 100000 \
    --batch-size 1024 \
    --temperature 0.5 \
    --data-dirs "$DATA_DIR/vanvliet,$DATA_DIR/deepcell/train,$DATA_DIR/deepcell/val" \
    --outdir results/runs/multidomain_large \
    --seed 42
