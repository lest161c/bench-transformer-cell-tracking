#!/bin/bash
#SBATCH --job-name=nope
#SBATCH --account=p_scads_celltracking
#SBATCH --partition=capella
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=logs/nope-%A.out
#SBATCH --error=logs/nope-%A.err
#
# Coordinate-free baseline (SPEC 0002 / T2): 7D regionprops features with
# ALL coordinate paths into the model disabled (NoPE).  Exact mirror of the
# vanvliet dense baseline (configs/vanvliet_baseline.yaml: seed 42, crop
# [320,320], dense attention) except the three NoPE switches documented in
# configs/vanvliet_nope.yaml:
#     pos_embed_per_dim=0, attn_positional_bias=none, attn_dist_mode=none
# This isolates whether spatial coordinates (via the Fourier positional
# encoding, RoPE, and the spatial-distance attention biases) carry the
# tracking signal (labbook 2026-06-02 prescription).
#
# Estimated runtime: ~6.4-11 h on one H100 (24 h limit leaves ample headroom;
# run_knn_sweep.slurm requests 48 h because its 500-epoch jobs are longer).
#
# PARTITION — resolved 2026-09-22: live `sinfo` on the cluster shows the
# H100 nodes in partition "capella*" (the default), consistent with
# HANDOVER.md.  vault/context/C11-Capella-HPC.md's "gpu-h100" name is
# stale; the --partition line above now uses "capella".
#
# ─── Usage ──────────────────────────────────────────────────────────
#
#   sbatch scripts/slurm/run_nope.slurm
#
# -----------------------------------------------------------------------

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

SEED=42
RUN_NAME="vanvliet_nope_s${SEED}"
CACHE_DIR=".cache_nope"
CONFIG="$REPO_ROOT/configs/vanvliet_nope.yaml"

echo "=== Coordinate-free (NoPE) baseline ==="
echo "  Run name:      ${RUN_NAME}"
echo "  Seed:          ${SEED}"
echo "  Config:        ${CONFIG}"
echo "  Node:          ${SLURM_NODELIST}"
echo "  Started at:    $(date)"
echo ""

# -----------------------------------------------------------------------
# Run training from the current trackastra checkout.
#
# All hyperparameters — including the NoPE switches and the exact
# train/val sequence lists mirroring configs/vanvliet_baseline.yaml — come
# from the config file (single source of truth); only launch-time
# arguments (run name, output/cache dirs, seed) are set here.
# -----------------------------------------------------------------------

cd "$TRK"

if [ "$TEST" = "1" ]; then
    echo "=== TEST MODE: import check ==="
    "$TRK/.venv/bin/python" -c "import trackastra; from trackastra.model import TrackingTransformer; from trackastra.model.model_parts import NoPositionalEncoding; print('trackastra OK:', TrackingTransformer); print('NoPositionalEncoding OK:', NoPositionalEncoding); import torch; print('CUDA:', torch.cuda.is_available())"
    echo "=== TEST PASSED ==="
    exit 0
fi

# -----------------------------------------------------------------------
# Normal mode: launch training
# -----------------------------------------------------------------------
PY="$TRK/.venv/bin/python"

EXTRA_ARGS="-c ${CONFIG} \
    --name ${RUN_NAME} \
    --outdir runs \
    --distributed False \
    --cachedir ${CACHE_DIR} \
    --seed ${SEED}"

echo "  Launching: $PY scripts/train.py $EXTRA_ARGS"
echo ""

$PY scripts/train.py $EXTRA_ARGS

EXIT_CODE=$?
echo ""
echo "  Training exited with code ${EXIT_CODE} at $(date)"

# Clean up run cache to avoid duplicate caches
rm -rf "${CACHE_DIR}" 2>/dev/null

exit ${EXIT_CODE}
