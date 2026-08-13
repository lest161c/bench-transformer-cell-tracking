#!/bin/bash
#SBATCH --job-name=h100_conv_race
#SBATCH --account=p_scads_celltracking
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=logs/h100_conv_race-%j.out
#SBATCH --error=logs/h100_conv_race-%j.err

# Config rationale: controlled A/B — same arch, data, schedule; only
# CNN residual injection varies. R=7D control, C=7D+CNN treatment.
# Trainable CNN tested in cross_dataset_eval; K/depth swept in
# run_training.

# H100 convergence race: Mode R (7D baseline) vs Mode C (7D + frozen CNN NT-Xent).
# Mini Trackastra at full scale: d_model=320, nhead=8, 6 enc + 6 dec layers.
# Training: BCE(pos_weight=100), AdamW, cosine schedule, early stopping.

# ─── REPO_ROOT ────────────────────────────────────────────────────────
# TODO: REPLACE with the absolute path to your bench-transformer-cell-tracking
# checkout on the cluster.  This is the only cluster-specific value in the
# file — everything else below is resolved relative to it.
REPO_ROOT=/path/to/bench-transformer-cell-tracking

source "$REPO_ROOT/slurm/load_cluster_env.sh"

cd "$BENCH/benchmark_ssl" || exit 1
uv sync
mkdir -p logs

uv run python -m src.cnn_encoder.h100_convergence \
    --data-root "$DATA_DIR/vanvliet" \
    --checkpoint results/checkpoints/cnn/cnn_ntxent_large.pt \
    --scale large \
    --epochs 500
