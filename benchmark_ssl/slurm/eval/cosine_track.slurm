#!/usr/bin/env bash
#SBATCH --job-name=etultra_cosine
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/lest161c/etultra_cosine_%j.log
set -euo pipefail
BENCH=/data/cat/ws/lest161c-ssl_cell_tracking/bench-transformer-cell-tracking
CONFIG="${1:?usage: cosine_track.slurm <config.yaml> <eval_outdir> [cutoff]}"
EVAL_OUT="${2:?usage: cosine_track.slurm <config.yaml> <eval_outdir> [cutoff]}"
CUTOFF="${3:-50}"
cd "$BENCH/cyril_trackastra"
export DYNACLR_CHECKPOINT="$BENCH/benchmark_ssl/results/checkpoints/cnn_bn100k_step100000.pt"
export PYTHONPATH="$BENCH/benchmark_ssl/src:${PYTHONPATH:-}"
"$BENCH/cyril_trackastra/.venv-etultra/bin/python" \
  "$BENCH/benchmark_ssl/slurm/eval/cosine_track.py" "$CONFIG" "$EVAL_OUT" "$CUTOFF"
echo COSINE_DONE_OK
