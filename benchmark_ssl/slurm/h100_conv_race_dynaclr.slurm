#!/bin/bash
#SBATCH --job-name=h100_conv_race_dynaclr
#SBATCH --account=p_scads_celltracking
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=logs/h100_conv_race_dynaclr-%j.out
#SBATCH --error=logs/h100_conv_race_dynaclr-%j.err

# H100 convergence race with the DynaCLR-trained ScaledCNN+BatchNorm encoder
# (SPEC 0026 R7 / E12-E13): Mode R (7D baseline) vs Mode C (7D + frozen DynaCLR CNN).
# Same protocol as h100_conv_race.slurm; only the CNN checkpoint differs.

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

cd "$BENCH/benchmark_ssl" || exit 1
mkdir -p logs

.venv/bin/python -m src.cnn_encoder.h100_convergence \
    --data-root "$DATA_DIR/vanvliet" \
    --checkpoint results/checkpoints/cnn_bn100k_step100000.pt \
    --scale large \
    --epochs 500
