#!/usr/bin/env bash
#SBATCH --job-name=etultra_train
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/lest161c/etultra_train_%j.log
set -euo pipefail
BENCH=/data/cat/ws/lest161c-ssl_cell_tracking/bench-transformer-cell-tracking
CONFIG="${1:?usage: train_etultra_dynaclr.slurm <config.yaml> <seed> [cachedir]}"
SEED="${2:?usage: train_etultra_dynaclr.slurm <config.yaml> <seed> [cachedir]}"
# One feature cache per seed: concurrent seeds must not share a zarr store.
CACHEDIR="${3:-$BENCH/.cache/etultra_seed${SEED}}"
mkdir -p "$CACHEDIR"
cd "$BENCH/cyril_trackastra"
export DYNACLR_CHECKPOINT="$BENCH/benchmark_ssl/results/checkpoints/cnn_bn100k_step100000.pt"
export PYTHONPATH="$BENCH/benchmark_ssl/src:${PYTHONPATH:-}"
"$BENCH/cyril_trackastra/.venv-etultra/bin/python" \
  "$BENCH/benchmark_ssl/slurm/run_train.py" -c "$CONFIG" --seed "$SEED" \
  --logger tensorboard --cachedir "$CACHEDIR"
echo TRAIN_DONE_OK
