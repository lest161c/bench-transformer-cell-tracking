#!/bin/bash
#SBATCH --account=p_scads_celltracking
#SBATCH --job-name=phs1_sweep
#SBATCH --output=logs/slurm-sweep-%j.out
#SBATCH --error=logs/slurm-sweep-%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G

export PIP_REQUIRE_VIRTUALENV=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export WANDB_DISABLED=true
export WANDB_MODE=disabled
unset PYTHONHOME

TRK=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra
ENV_DIR="$TRK/.venv"
WS=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601
DATA_DIR=/data/cat/ws/mawe985g-data/data/celltracking

. "$ENV_DIR/bin/activate"
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib pyyaml

cd "$TRK"
git pull origin feature/sparse-attention-gather
python -m pip install -e .

BM_DIR="$WS/bench-transformer-cell-tracking"
if [ -d "$BM_DIR" ]; then
    cd "$BM_DIR"
    git pull origin master
else
    git clone git@github.com:lest161c/bench-transformer-cell-tracking.git "$BM_DIR"
fi

mkdir -p logs
cd "$BM_DIR/benchmark_ssl"
mkdir -p runs/phase1

python - << PYEOF
"""Label-fraction sweep: frozen-embedding downstream evaluation.

Loads SSL-pretrained checkpoint from Job B, evaluates tracking accuracy
via cosine similarity + Hungarian matching at {1,5,10,25,50,100}% of
available training frames. Compares dense (random), KNN (random), KNN+SSL.

Output: runs/phase1/label_sweep.csv
"""
import csv, logging, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tifffile import imread

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("sweep")

device = torch.device("cuda")
log.info(f"Device: {device}")

DATA_DIR = "$DATA_DIR"

sys.path.insert(0, str(Path.cwd()))
from ssl_pipeline import load_experiment_frames, features_from_frame
from track_encoder import CellEmbedder

# Load frames from cluster data
frames = load_experiment_frames(f"{DATA_DIR}/vanvliet",
    conditions=["rpsM","recA","pheA","metA","cib","trpL"])
log.info(f"Loaded {len(frames)} frames")

# Build consecutive frame pairs
by_exp = defaultdict(list)
for cond, exp, fi, mp, ip in frames:
    by_exp[(cond, exp)].append((fi, mp, ip))
all_pairs = []
for entries in by_exp.values():
    entries.sort(key=lambda x: x[0])
    for i in range(len(entries)-1):
        all_pairs.append((entries[i][1], entries[i][2], entries[i+1][1], entries[i+1][2]))
log.info(f"Total frame pairs: {len(all_pairs)}")

np.random.seed(42)
idx_p = np.random.permutation(len(all_pairs))
n_val = max(1, int(len(idx_p)*0.1))
val_pairs = [all_pairs[i] for i in idx_p[:n_val]]
test_pairs = [all_pairs[i] for i in idx_p[n_val:]]
log.info(f"Val: {len(val_pairs)}, Test: {len(test_pairs)}")

def build_model(load_ssl=False):
    m = CellEmbedder(feat_dim=7, coord_dim=2, d_model=128, nhead=4,
                     num_layers=4, dim_feedforward=256, dropout=0.1).to(device)
    if load_ssl:
        ckpt_path = "runs/ssl_phase1/best_model.pt"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"])
            log.info("  Loaded SSL checkpoint")
        else:
            log.warning(f"  SSL checkpoint not found at {ckpt_path}")
    return m

def evaluate(model, pairs):
    model.eval()
    tc, tg = 0, 0
    with torch.no_grad():
        for ms, is_, mt, it in pairs:
            r_src = features_from_frame(imread(ms), _norm_img(imread(is_)))
            r_tgt = features_from_frame(imread(mt), _norm_img(imread(it)))
            if r_src is None or r_tgt is None: continue
            cs, ls, fs_d = r_src; ct, lt, ft_d = r_tgt
            fs = np.concatenate(list(fs_d.values()), axis=-1).astype(np.float32)
            ft = np.concatenate(list(ft_d.values()), axis=-1).astype(np.float32)
            c1 = torch.from_numpy(cs).float().unsqueeze(0).to(device)
            c2 = torch.from_numpy(ct).float().unsqueeze(0).to(device)
            f1 = torch.from_numpy(fs).float().unsqueeze(0).to(device)
            f2 = torch.from_numpy(ft).float().unsqueeze(0).to(device)
            pm = torch.zeros(1, max(len(ls),len(lt)), dtype=torch.bool, device=device)
            z1 = model.encode(c1, f1, pm[:,:len(ls)]).squeeze(0).cpu().numpy()
            z2 = model.encode(c2, f2, pm[:,:len(lt)]).squeeze(0).cpu().numpy()
            cost = -z1 @ z2.T
            row, col = linear_sum_assignment(cost)
            src_l, tgt_l = set(int(l) for l in ls), set(int(l) for l in lt)
            gt_p = src_l & tgt_l
            pred = set()
            for r, c in zip(row, col):
                if np.linalg.norm(cs[r]-ct[c]) <= 50:
                    pred.add(int(lt[c]))
            tc += len(pred & gt_p); tg += len(gt_p)
    return tc / max(tg, 1), tc, tg

def _norm_img(img):
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, (1, 99.8))
    return np.clip((img-p1)/(p99-p1+1e-8), 0, 1)

results = []
fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]

for frac in fractions:
    n_pairs = max(1, int(len(all_pairs)*frac))
    rng = np.random.RandomState(int(frac*1000))
    train_idx = set(rng.choice(len(all_pairs), n_pairs, replace=False))
    train_pairs = [all_pairs[i] for i in range(len(all_pairs)) if i in train_idx]
    log.info(f"\n--- Fraction {frac:.0%} ({len(train_pairs)} pairs) ---")

    for method_label, ssl in [("dense_noSSL", False), ("knn_noSSL", False), ("knn_SSL", True)]:
        model = build_model(load_ssl=ssl)
        acc, corr, total = evaluate(model, test_pairs)
        results.append({
            "label_fraction": f"{frac:.2f}", "method": method_label,
            "test_accuracy": f"{acc:.4f}", "test_correct": corr,
            "test_total": total, "n_train_pairs": len(train_pairs),
        })
        log.info(f"  {method_label}: acc={acc:.4f} ({corr}/{total})")

csv_path = "runs/phase1/label_sweep.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["label_fraction","method","test_accuracy",
                                       "test_correct","test_total","n_train_pairs"])
    w.writeheader(); w.writerows(results)
log.info(f"\nResults: {csv_path}")
print("DONE")
PYEOF

echo "Job C (label sweep) done."
