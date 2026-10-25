#!/bin/bash
#SBATCH --job-name=knn_probe
#SBATCH --account=p_scads_celltracking
#SBATCH --partition=gpu-h100
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=benchmark_ssl/logs/knn_probe_%j.out
#SBATCH --error=benchmark_ssl/logs/knn_probe_%j.err

# run_knn_probe.slurm — SPEC 0046 ticket F08: full cluster run of the
# Phase-1 K-NN feature probe (P1 frame_cv and P2 loco).
#
# Runs both protocols with the full default feature matrix and probe seeds
# 42/43/44.  The optional sam21 row is appended to the default matrix only
# when `import sam2` succeeds; otherwise the script prints the install hint
# from SAM21_SETUP.md and continues with the non-sam21 matrix, which is still
# a valid, complete Phase-1 result package.
#
# Usage (the owner submits the job; agents only prepare and dry-run it):
#   DRY_RUN=1 bash benchmark_ssl/slurm/run_knn_probe.slurm
#   sbatch benchmark_ssl/slurm/run_knn_probe.slurm
#   FEATURES=regionprops_7d,dino SEEDS=42 sbatch benchmark_ssl/slurm/run_knn_probe.slurm
#   SAM21_ONLY=1 sbatch benchmark_ssl/slurm/run_knn_probe.slurm   # after sam2 install
#
# DRY_RUN=1 echoes the exact commands and exits 0 without sourcing the cluster
# environment or executing any Python.  Each protocol is resumable: a run is
# skipped when its summary.csv already exists.  Existing results are never
# deleted.
#
# Outputs (relative to benchmark_ssl/):
#   results/probes/knn/cluster/frame_cv/{summary.csv,coverage_vs_k.json,runs/}
#   results/probes/knn/cluster/loco/{summary.csv,coverage_vs_k.json,runs/}
#   results/probes/knn/cluster/sam21/{frame_cv,loco}/...   # SAM21_ONLY=1 only

set -euo pipefail

DEFAULT_FEATURES="regionprops_7d,regionprops_13d,hu_20d,posenc,dino,cnn_frozen,cnn_ntxent"
DEFAULT_SEEDS="42,43,44"

# ─── DRY RUN: echo the commands, execute nothing ──────────────────────
# This block deliberately runs before `~/.bench.env` is sourced so the script
# can be validated on a machine that has no cluster environment file.
if [ -n "${DRY_RUN:-}" ]; then
    echo "[DRY_RUN=1] exact commands (no workload executed):"
    echo ""
    echo "# 0. Environment and optional-dependency preflight."
    echo "source \"\$HOME/.bench.env\""
    echo "source \"\$REPO_ROOT/slurm/load_cluster_env.sh\""
    echo "cd \"\$REPO_ROOT/benchmark_ssl\""
    echo ".venv/bin/python -c \"import torch, sys; print('torch  :', torch.__version__); print('cuda   :', torch.version.cuda); print('python :', sys.version.split()[0]); print('GPU    :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\""
    echo ".venv/bin/python -c \"import sam2\"   # optional; appends ',sam21' when it succeeds"
    echo ""
    echo "# FEATURES defaults to:"
    echo "#   ${DEFAULT_FEATURES}"
    echo "# and gets ',sam21' appended when 'import sam2' succeeds."
    echo "# FEATURES and SEEDS environment variables override the defaults."
    echo ""
    echo "# 1. P1 frame_cv"
    echo "cd \"\$REPO_ROOT/benchmark_ssl\" && .venv/bin/python -m src.analysis.feature_probe_knn \\"
    echo "    --data-root \"\$DATA_DIR/vanvliet\" \\"
    echo "    --conditions rpsM,recA,pheA,metA,cib,trpL \\"
    echo "    --max-pairs-per-condition 15 \\"
    echo "    --cv-folds 10 \\"
    echo "    --protocol frame_cv \\"
    echo "    --features \"\$FEATURES\" \\"
    echo "    --probes linear,mlp \\"
    echo "    --seeds \"\$SEEDS\" \\"
    echo "    --epochs 200 \\"
    echo "    --k auto \\"
    echo "    --out results/probes/knn/cluster/frame_cv"
    echo ""
    echo "# 2. P2 loco"
    echo "cd \"\$REPO_ROOT/benchmark_ssl\" && .venv/bin/python -m src.analysis.feature_probe_knn \\"
    echo "    --data-root \"\$DATA_DIR/vanvliet\" \\"
    echo "    --conditions rpsM,recA,pheA,metA,cib,trpL \\"
    echo "    --max-pairs-per-condition 15 \\"
    echo "    --protocol loco \\"
    echo "    --features \"\$FEATURES\" \\"
    echo "    --probes linear,mlp \\"
    echo "    --seeds \"\$SEEDS\" \\"
    echo "    --epochs 200 \\"
    echo "    --k auto \\"
    echo "    --out results/probes/knn/cluster/loco"
    echo ""
    echo "# 3. Optional SAM21_ONLY=1 path (only after sam2 is installed)"
    echo "cd \"\$REPO_ROOT/benchmark_ssl\" && .venv/bin/python -m src.analysis.feature_probe_knn \\"
    echo "    --data-root \"\$DATA_DIR/vanvliet\" \\"
    echo "    --conditions rpsM,recA,pheA,metA,cib,trpL \\"
    echo "    --max-pairs-per-condition 15 \\"
    echo "    --cv-folds 10 \\"
    echo "    --protocol frame_cv \\"
    echo "    --features sam21 \\"
    echo "    --probes linear,mlp \\"
    echo "    --seeds \"\$SEEDS\" \\"
    echo "    --epochs 200 \\"
    echo "    --k auto \\"
    echo "    --out results/probes/knn/cluster/sam21/frame_cv"
    echo "cd \"\$REPO_ROOT/benchmark_ssl\" && .venv/bin/python -m src.analysis.feature_probe_knn \\"
    echo "    --data-root \"\$DATA_DIR/vanvliet\" \\"
    echo "    --conditions rpsM,recA,pheA,metA,cib,trpL \\"
    echo "    --max-pairs-per-condition 15 \\"
    echo "    --protocol loco \\"
    echo "    --features sam21 \\"
    echo "    --probes linear,mlp \\"
    echo "    --seeds \"\$SEEDS\" \\"
    echo "    --epochs 200 \\"
    echo "    --k auto \\"
    echo "    --out results/probes/knn/cluster/sam21/loco"
    echo ""
    echo "[DRY_RUN=1] done; exiting 0 without executing."
    exit 0
fi

# ─── Cluster environment ──────────────────────────────────────────────
# Ordering follows the existing runners: user env first, then the shared
# loader (which sources ~/.bench.env again and validates REPO_ROOT/TRK/BENCH).
source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

cd "$REPO_ROOT/benchmark_ssl" || exit 1
mkdir -p logs results/probes/knn/cluster

PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "FATAL: $PYTHON is missing or not executable." >&2
    echo "Create the cluster environment once on the login node with 'uv sync'." >&2
    exit 1
fi

# ─── Preflight: runtime provenance and optional sam2 dependency ───────
echo "=== Preflight ==="
"$PYTHON" - <<'PY'
"""Print the Python, PyTorch, CUDA, and GPU versions used by this job."""
import sys

import torch

print("torch  :", torch.__version__)
print("cuda   :", torch.version.cuda)
print("python :", sys.version.split()[0])
if torch.cuda.is_available():
    print("GPU    :", torch.cuda.get_device_name(0))
else:
    print("GPU    : none (CUDA unavailable)")
PY

SAM21_AVAILABLE=0
if "$PYTHON" -c "import sam2" >/dev/null 2>&1; then
    SAM21_AVAILABLE=1
    echo "sam2   : importable (optional sam21 row enabled)"
else
    echo "sam2   : NOT importable (optional sam21 row disabled)"
    echo "         Install with: .venv/bin/pip install git+https://github.com/facebookresearch/sam2.git"
    echo "         Continuing with the non-sam21 matrix; it remains a valid Phase-1 package."
fi

# ─── Feature matrix, seeds, and output root ───────────────────────────
OUTPUT_ROOT="results/probes/knn/cluster"
SEEDS="${SEEDS:-$DEFAULT_SEEDS}"

if [ "${SAM21_ONLY:-0}" = "1" ]; then
    if [ "$SAM21_AVAILABLE" -ne 1 ]; then
        echo "FATAL: SAM21_ONLY=1 requires the optional sam2 dependency." >&2
        echo "Install with: .venv/bin/pip install git+https://github.com/facebookresearch/sam2.git" >&2
        exit 1
    fi
    FEATURES="sam21"
    OUTPUT_ROOT="results/probes/knn/cluster/sam21"
elif [ -z "${FEATURES:-}" ]; then
    FEATURES="$DEFAULT_FEATURES"
    if [ "$SAM21_AVAILABLE" -eq 1 ]; then
        FEATURES="$FEATURES,sam21"
    fi
else
    # An explicit FEATURES override is authoritative; the caller can include
    # sam21 deliberately once the optional dependency is provisioned.
    echo "features   : using FEATURES override (sam21 is not appended automatically)"
fi

echo "=== SPEC 0046 / F08 K-NN probe cluster run ==="
echo "features   : $FEATURES"
echo "seeds      : $SEEDS"
echo "output root: $OUTPUT_ROOT"

# ─── Protocol runner ──────────────────────────────────────────────────
run_protocol() {
    # Run one protocol, skipping it when its summary.csv already exists.
    #
    # Arguments:
    #   $1: Either frame_cv (P1) or loco (P2).
    #   $2: Directory for summary.csv, coverage_vs_k.json, and per-run JSONs.
    # Returns 0 when the protocol ran or was skipped, and the Python exit
    # status when the probe run failed.
    local protocol="$1"
    local output_directory="$2"
    local summary_path="$output_directory/summary.csv"

    if [ -f "$summary_path" ]; then
        echo "=== SKIP $protocol (summary exists: $summary_path)"
        return 0
    fi

    mkdir -p "$output_directory"
    echo "=== RUN $protocol -> $output_directory"

    local -a probe_command=(
        "$PYTHON" -m src.analysis.feature_probe_knn
        --data-root "$DATA_DIR/vanvliet"
        --conditions rpsM,recA,pheA,metA,cib,trpL
        --max-pairs-per-condition 15
    )
    if [ "$protocol" = "frame_cv" ]; then
        probe_command+=(--cv-folds 10)
    fi
    probe_command+=(
        --protocol "$protocol"
        --features "$FEATURES"
        --probes linear,mlp
        --seeds "$SEEDS"
        --epochs 200
        --k auto
        --out "$output_directory"
    )

    local status=0
    "${probe_command[@]}" || status=$?
    if [ "$status" -ne 0 ]; then
        echo "FATAL: $protocol probe run failed (exit $status)" >&2
        return "$status"
    fi
    echo "    done -> $summary_path"
    return 0
}

overall_status=0
run_protocol frame_cv "$OUTPUT_ROOT/frame_cv" || overall_status=1
run_protocol loco "$OUTPUT_ROOT/loco" || overall_status=1

# ─── Final output paths ───────────────────────────────────────────────
echo "=== Output paths ==="
echo "frame_cv: $OUTPUT_ROOT/frame_cv/summary.csv"
echo "loco    : $OUTPUT_ROOT/loco/summary.csv"
echo "artifacts: $OUTPUT_ROOT/{frame_cv,loco}/{summary.csv,coverage_vs_k.json,runs/}"
echo "=== finished with exit status $overall_status"
exit "$overall_status"
