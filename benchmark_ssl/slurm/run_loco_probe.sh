#!/bin/bash
#SBATCH --job-name=loco_edge_probe
#SBATCH --account=p_scads_celltracking
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=logs/loco_edge_probe-%j.out
#SBATCH --error=logs/loco_edge_probe-%j.err

# run_loco_probe.sh — SPEC 0041: leave-one-condition-out edge probing.
#
# For each of the six vanvliet conditions: train the probes on frame pairs of
# the other five conditions (SPEC 0040 protocol) and evaluate pooled on the
# in-sample pairs and on the held-out condition. 6 folds × 3 probe seeds,
# using the existing unified_edge_probe.py --condition-split with per-fold
# split arguments — no harness changes.
#
# Feature rows: run-A features (rp, rp_fourier, hoct2d, hoct2d_fourier, dino,
# cnn_frozen @ NT-Xent checkpoint) + cnn_frozen @ DynaCLR checkpoint
# (--no-cache: the per-frame cache is keyed by feature name only) + cnn_e2e
# (batch 64; disable with SKIP_E2E=1).
#
# Runs locally (python3, repo data) or via sbatch (cluster venv, $DATA_DIR).
#
# Usage:
#   bash benchmark_ssl/slurm/run_loco_probe.sh
#   SKIP_E2E=1 bash benchmark_ssl/slurm/run_loco_probe.sh
#   sbatch benchmark_ssl/slurm/run_loco_probe.sh
#
# Outputs: results/probes/condition_split/loco_<cond>_s<seed>.json
# (+ _dynaclr / _e2e variants)

set -uo pipefail

BSSL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BSSL"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    # Cluster: venv python, corporate-data vanvliet copy.
    source "$HOME/.bench.env"
    PY="$BSSL/.venv/bin/python"
    DATA_ROOT="$DATA_DIR/vanvliet"
else
    PY=python3
    DATA_ROOT="$(cd "$BSSL/.." && pwd)/data/vanvliet"
fi

OUTDIR="results/probes/condition_split"
mkdir -p "$OUTDIR"

NTXENT_CKPT="results/checkpoints/cnn/cnn_ntxent_large.pt"
DYNACLR_CKPT="results/checkpoints/cnn_bn100k_step100000.pt"
CONDITIONS=(rpsM recA pheA metA cib trpL)
SEEDS=(42 43 44)

run_one() {
    local heldout="$1" seed="$2"
    local tag="loco_${heldout}_s${seed}"
    local out="$OUTDIR/${tag}.json"
    if [ -s "$out" ]; then
        echo "=== SKIP $tag (exists)"
        return 0
    fi
    local in_conds=""
    for c in "${CONDITIONS[@]}"; do
        [ "$c" = "$heldout" ] || in_conds="${in_conds:+$in_conds,}$c"
    done
    echo "=== RUN $tag (heldout=$heldout)"
    $PY -m src.edge_probing.unified_edge_probe \
        --data-root "$DATA_ROOT" \
        --conditions rpsM,recA,pheA,metA,cib,trpL \
        --features rp,rp_fourier,hoct2d,hoct2d_fourier,dino,cnn_frozen \
        --probe both \
        --condition-split \
        --in-conditions "$in_conds" \
        --heldout-conditions "$heldout" \
        --max-pairs-per-condition 15 \
        --max-pairs 500 \
        --epochs 200 --patience 10 --standardize \
        --checkpoint "$NTXENT_CKPT" \
        --seed "$seed" \
        --output "$out" \
        > "$OUTDIR/${tag}.log" 2>&1
    local status=$?
    if [ $status -ne 0 ]; then
        echo "    FAILED run A (exit $status) — see $OUTDIR/${tag}.log"
        return $status
    fi
    echo "    run A done"
    if [ "${SKIP_E2E:-0}" != "1" ]; then
        $PY -m src.edge_probing.unified_edge_probe \
            --data-root "$DATA_ROOT" \
            --conditions rpsM,recA,pheA,metA,cib,trpL \
            --features cnn_frozen \
            --probe both \
            --condition-split \
            --in-conditions "$in_conds" \
            --heldout-conditions "$heldout" \
            --max-pairs-per-condition 15 \
            --max-pairs 500 \
            --epochs 200 --patience 10 --standardize \
            --checkpoint "$DYNACLR_CKPT" \
            --seed "$seed" \
            --output "$OUTDIR/${tag}_dynaclr.json" \
            --no-cache \
            > "$OUTDIR/${tag}_dynaclr.log" 2>&1 || \
            { echo "    FAILED dynaclr"; return 1; }
        $PY -m src.edge_probing.unified_edge_probe \
            --data-root "$DATA_ROOT" \
            --conditions rpsM,recA,pheA,metA,cib,trpL \
            --features cnn_e2e \
            --probe both \
            --condition-split \
            --in-conditions "$in_conds" \
            --heldout-conditions "$heldout" \
            --max-pairs-per-condition 15 \
            --max-pairs 500 \
            --epochs 200 --patience 10 \
            --batch-size-linear 64 \
            --checkpoint "$NTXENT_CKPT" \
            --seed "$seed" \
            --output "$OUTDIR/${tag}_e2e.json" \
            > "$OUTDIR/${tag}_e2e.log" 2>&1 || \
            { echo "    FAILED e2e"; return 1; }
    fi
    echo "    done -> $out (+ _dynaclr, _e2e)"
    return 0
}

overall=0
for heldout in "${CONDITIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_one "$heldout" "$seed" || overall=1
    done
done
echo "=== LOCO sweep complete (overall exit $overall)"
exit $overall
