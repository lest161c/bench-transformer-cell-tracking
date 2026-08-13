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

# H100 convergence race: Mode R (7D baseline) vs Mode C (7D + frozen CNN NT-Xent).
# Mini Trackastra at full scale: d_model=320, nhead=8, 6 enc + 6 dec layers.
# Training: BCE(pos_weight=100), AdamW, cosine schedule, early stopping.

source "$(dirname "$0")/cluster_env.sh"

python -m src.cnn_encoder.h100_convergence \
    --data-root "$DATA_DIR/vanvliet" \
    --checkpoint results/checkpoints/cnn/cnn_ntxent_large.pt \
    --scale large \
    --epochs 500
