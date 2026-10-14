#!/usr/bin/env bash
#SBATCH --job-name=etultra_eval
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/lest161c/etultra_eval_%j.log
set -euo pipefail
BENCH=/data/cat/ws/lest161c-ssl_cell_tracking/bench-transformer-cell-tracking
TRAIN_OUTDIR="${1:?usage: eval_etultra.slurm <train_outdir> <config.yaml> <eval_outdir>}"
CONFIG="${2:?usage: eval_etultra.slurm <train_outdir> <config.yaml> <eval_outdir>}"
EVAL_OUT="${3:?usage: eval_etultra.slurm <train_outdir> <config.yaml> <eval_outdir>}"
cd "$BENCH/cyril_trackastra"
export DYNACLR_CHECKPOINT="$BENCH/benchmark_ssl/results/checkpoints/cnn_bn100k_step100000.pt"
export PYTHONPATH="$BENCH/benchmark_ssl/src:${PYTHONPATH:-}"
"$BENCH/cyril_trackastra/.venv-etultra/bin/python" \
  "$BENCH/benchmark_ssl/slurm/eval/eval_etultra.py" "$TRAIN_OUTDIR" "$CONFIG" "$EVAL_OUT"
echo EVAL_DONE_OK
