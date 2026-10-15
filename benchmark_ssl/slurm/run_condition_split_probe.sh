#!/bin/bash
# run_condition_split_probe.sh — SPEC 0040: condition-split edge probing.
#
# Trains probes on the five in-sample vanvliet conditions and evaluates pooled
# on in-sample vs held-out (trpL), for the published feature rows plus the
# DynaCLR (ScaledCNN+BN, VisCy/DynaCell recipe, trained on vanvliet) encoder.
#
# Runs 6 invocations: run A (NT-Xent checkpoint, 6 feature rows) and run B
# (DynaCLR checkpoint, cnn_frozen only, cache disabled — the per-frame feature
# cache is keyed by feature name only and would collide across checkpoints),
# each for the three probe seeds 42/43/44.
#
# Usage:
#   bash benchmark_ssl/slurm/run_condition_split_probe.sh          # all runs
#   RUN_ONLY=A_42 bash benchmark_ssl/slurm/run_condition_split_probe.sh
#
# Outputs: results/probes/condition_split/{ntxent,dynaclr}_s<seed>.json
# (paths relative to benchmark_ssl/), log next to the JSONs.

set -uo pipefail

BSSL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BSSL"

DATA_ROOT="$(cd "$BSSL/.." && pwd)/data/vanvliet"
OUTDIR="results/probes/condition_split"
mkdir -p "$OUTDIR"

NTXENT_CKPT="results/checkpoints/cnn/cnn_ntxent_large.pt"
DYNACLR_CKPT="results/checkpoints/cnn_bn100k_step100000.pt"
FEATURES_A="rp,rp_fourier,hoct2d,hoct2d_fourier,dino,cnn_frozen"
SEEDS=(42 43 44)

run_one() {
    local run_name="$1" checkpoint="$2" features="$3" seed="$4" extra_args="$5"
    local out="$OUTDIR/${run_name}_s${seed}.json"
    if [ -s "$out" ]; then
        echo "=== SKIP ${run_name}_s${seed} (exists: $out)"
        return 0
    fi
    echo "=== RUN ${run_name}_s${seed} (features: $features)"
    python3 -m src.edge_probing.unified_edge_probe \
        --data-root "$DATA_ROOT" \
        --conditions rpsM,recA,pheA,metA,cib,trpL \
        --features "$features" \
        --probe both \
        --condition-split \
        --in-conditions rpsM,recA,pheA,metA,cib \
        --heldout-conditions trpL \
        --max-pairs-per-condition 15 \
        --max-pairs 500 \
        --epochs 200 --patience 10 --standardize \
        --checkpoint "$checkpoint" \
        --seed "$seed" \
        --output "$out" \
        $extra_args \
        > "$OUTDIR/${run_name}_s${seed}.log" 2>&1
    local status=$?
    if [ $status -ne 0 ]; then
        echo "    FAILED (exit $status) — see $OUTDIR/${run_name}_s${seed}.log"
    else
        echo "    done -> $out"
    fi
    return $status
}

overall=0
for seed in "${SEEDS[@]}"; do
    run_one "ntxent"  "$NTXENT_CKPT"  "$FEATURES_A"   "$seed" ""                    || overall=1
    run_one "dynaclr" "$DYNACLR_CKPT" "cnn_frozen"    "$seed" "--no-cache"          || overall=1
done

echo "=== sweep complete (overall exit $overall)"
exit $overall
