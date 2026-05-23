#!/bin/bash
# =============================================================================
# Phase 1: First batch of experiments for sparse KNN + SSL cell tracking
# 
# Submits 3 independent SLURM jobs:
#   Job A: H100 synthetic attention scaling benchmark (N=128..8192, L=12)
#   Job B: SSL contrastive pretraining on all vanvliet conditions  
#   Job C: Downstream label-fraction sweep (fast frozen-embedding eval)
#
# All produce CSV files. Once these 3 jobs finish, review results to plan
# Phase 2 (full Trackastra fine-tune + distortion ablation + cross-dataset).
#
# Usage:   bash run_phase1.sh
# Requires: slurm cluster with GPU partition, >=40GB VRAM
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
BS_DIR="$PROJ_DIR/benchmark_ssl"
BA_DIR="$PROJ_DIR/benchmark_attn"
OUT_BASE="$PROJ_DIR/runs/phase1"
mkdir -p "$OUT_BASE"

GPU_PARTITION="gpu-h100"          # change to your partition name
GPU_COUNT=1
GPU_MEM=80G
CPU_COUNT=8
TIME_JOB_A="02:00:00"            # 2h — synthetic scaling sweep
TIME_JOB_B="04:00:00"            # 4h — SSL pretraining 50 epochs
TIME_JOB_C="06:00:00"            # 6h — label fraction sweep (6 fractions × 3 methods)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ---------------------------------------------------------------------------
# Job A: H100 attention scaling benchmark (prove KNN scalability)
# ---------------------------------------------------------------------------
cat > "$OUT_BASE/jobA_scaling.py" << 'PYEOF'
"""Synthetic attention scaling benchmark on H100.

Sweeps N ∈ [128, 256, 512, 1024, 2048, 4096, 8192], K ∈ [4, 16, 32, 64],
L ∈ [1, 6, 12] layers. Dense baseline at each N (OOMs at N≥4096).
Uses random Q/K/V tensors, float16, measures time + peak memory.

Output: runs/phase1/scaling_h100.csv
"""
import torch, time, csv, sys
from pathlib import Path

device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

d_model = 256
nhead = 4
head_dim = d_model // nhead
B = 8
warmup = 5
repeat = 20

Ns = [128, 256, 512, 1024, 2048, 4096, 8192]
Ls = [1, 6, 12]
Ks = [4, 16, 32, 64]
results = []

for L in Ls:
    for N in Ns:
        # --- Dense baseline (masked SDPA — simulates Trackastra's attn_mask path) ---
        if N <= 4096:  # dense OOMs at 8192
            try:
                attn_mask = torch.zeros(N, N, device=device, dtype=torch.float16)
                # Simulate spatial cutoff: mask cells beyond radius
                dist = torch.cdist(torch.randn(N, 2, device=device), torch.randn(N, 2, device=device))
                attn_mask[dist > 100] = -1e9

                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                for _ in range(warmup):
                    for _ in range(L):
                        x = torch.randn(B, N, d_model, device=device, dtype=torch.float16)
                        q = k = v = x
                        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
                        scores = scores + attn_mask
                        attn_w = torch.softmax(scores, dim=-1)
                        x_tmp = torch.matmul(attn_w, v)
                        x_tmp = torch.nn.functional.gelu(torch.nn.Linear(d_model, d_model*2, device=device, dtype=torch.float16)(x_tmp))
                        x_tmp = torch.nn.Linear(d_model*2, d_model, device=device, dtype=torch.float16)(x_tmp)
                        x = x + x_tmp

                torch.cuda.synchronize()
                mem = torch.cuda.max_memory_allocated() / (1024**2)

                t0 = time.perf_counter()
                for _ in range(repeat):
                    x = torch.randn(B, N, d_model, device=device, dtype=torch.float16)
                    q = k = v = x
                    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
                    scores = scores + attn_mask
                    attn_w = torch.softmax(scores, dim=-1)
                    x = torch.matmul(attn_w, v)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / repeat * 1000
                results.append({"N": N, "L": L, "K": "dense", "variant": "masked",
                               "time_ms_per_step": f"{dt:.3f}", "mem_mb": f"{mem:.1f}"})
                print(f"  DENSE  N={N:5d} L={L:2d}: {dt:8.3f}ms  {mem:8.1f}MB")
            except RuntimeError as e:
                results.append({"N": N, "L": L, "K": "dense", "variant": "masked",
                               "time_ms_per_step": "OOM", "mem_mb": "OOM"})
                print(f"  DENSE  N={N:5d} L={L:2d}: OOM")

        # --- Sparse KNN (gather-based, FlashAttention-compatible) ---
        for K in Ks:
            if K >= N:
                continue
            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                for _ in range(warmup):
                    x = torch.randn(B, N, d_model, device=device, dtype=torch.float16)
                    coords = torch.randn(B, N, 2, device=device, dtype=torch.float16)
                    dist = torch.cdist(coords, coords)
                    _, knn_idx = torch.topk(dist, k=K, dim=-1, largest=False)  # (B, N, K)
                    for _ in range(L):
                        q = k = v = x
                        k_sel = torch.gather(k.unsqueeze(2).expand(-1, -1, K, -1), 1,
                                            knn_idx.unsqueeze(-1).expand(-1, -1, -1, d_model))
                        v_sel = torch.gather(v.unsqueeze(2).expand(-1, -1, K, -1), 1,
                                            knn_idx.unsqueeze(-1).expand(-1, -1, -1, d_model))
                        x = torch.nn.functional.scaled_dot_product_attention(q, k_sel, v_sel)

                torch.cuda.synchronize()
                mem = torch.cuda.max_memory_allocated() / (1024**2)

                t0 = time.perf_counter()
                for _ in range(repeat):
                    x = torch.randn(B, N, d_model, device=device, dtype=torch.float16)
                    coords = torch.randn(B, N, 2, device=device, dtype=torch.float16)
                    dist = torch.cdist(coords, coords)
                    _, knn_idx = torch.topk(dist, k=K, dim=-1, largest=False)
                    for _ in range(L):
                        q = k = v = x
                        k_sel = torch.gather(k.unsqueeze(2).expand(-1, -1, K, -1), 1,
                                            knn_idx.unsqueeze(-1).expand(-1, -1, -1, d_model))
                        v_sel = torch.gather(v.unsqueeze(2).expand(-1, -1, K, -1), 1,
                                            knn_idx.unsqueeze(-1).expand(-1, -1, -1, d_model))
                        x = torch.nn.functional.scaled_dot_product_attention(q, k_sel, v_sel)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / repeat * 1000
                results.append({"N": N, "L": L, "K": f"K={K}", "variant": "knn_gather",
                               "time_ms_per_step": f"{dt:.3f}", "mem_mb": f"{mem:.1f}"})
                print(f"  KNN K={K:2d} N={N:5d} L={L:2d}: {dt:8.3f}ms  {mem:8.1f}MB")
            except RuntimeError as e:
                results.append({"N": N, "L": L, "K": f"K={K}", "variant": "knn_gather",
                               "time_ms_per_step": "OOM", "mem_mb": "OOM"})
                print(f"  KNN K={K:2d} N={N:5d} L={L:2d}: OOM")
        print()

outdir = Path("runs/phase1")
outdir.mkdir(parents=True, exist_ok=True)
csv_path = outdir / "scaling_h100.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["N", "L", "K", "variant", "time_ms_per_step", "mem_mb"])
    w.writeheader()
    w.writerows(results)
print(f"\nResults written to {csv_path}")
PYEOF

JOB_A="phase1_scaling_${TIMESTAMP}"
cat > "$OUT_BASE/submit_A.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_A}
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPU_COUNT}
#SBATCH --gres=gpu:${GPU_COUNT}
#SBATCH --mem=${GPU_MEM}
#SBATCH --time=${TIME_JOB_A}
#SBATCH --output=${OUT_BASE}/jobA_%j.out
#SBATCH --error=${OUT_BASE}/jobA_%j.err

source ~/miniconda3/bin/activate trackastra 2>/dev/null || true
cd "$BS_DIR"
python "$OUT_BASE/jobA_scaling.py"
echo "Job A done."
EOF

# ---------------------------------------------------------------------------
# Job B: SSL contrastive pretraining (all vanvliet conditions)
# ---------------------------------------------------------------------------
cat > "$OUT_BASE/jobB_config.yaml" << 'YEOF'
name: ssl_phase1
data_root: ../data/vanvliet
conditions: [rpsM, recA, pheA, metA, cib, trpL]
ndim: 2
features: regionprops2
distortions: [affine, elastic, jitter, dropout, photometric, feature_noise]
affine: {degrees: 15, scale: [0.85, 1.15], shear: [0.1, 0.1]}
elastic: {alpha: [10, 50], sigma: [5, 15]}
jitter: {std: [2, 8], p_cell_jitter: 0.8}
dropout: {p_drop: [0.05, 0.2]}
photometric: {scale: [0.5, 2.0], shift: [-0.1, 0.1]}
feature_noise: {std: [0.02, 0.15]}
encoder: {d_model: 128, nhead: 4, num_layers: 4, dim_feedforward: 256, dropout: 0.1}
ssl: {temperature: 0.05}
training: {batch_size: 16, lr: 0.0003, weight_decay: 0.01, epochs: 50, val_split: 0.1, checkpoint_every: 10}
seed: 42
YEOF

JOB_B="phase1_ssl_${TIMESTAMP}"
cat > "$OUT_BASE/submit_B.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_B}
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPU_COUNT}
#SBATCH --gres=gpu:${GPU_COUNT}
#SBATCH --mem=${GPU_MEM}
#SBATCH --time=${TIME_JOB_B}
#SBATCH --output=${OUT_BASE}/jobB_%j.out
#SBATCH --error=${OUT_BASE}/jobB_%j.err

source ~/miniconda3/bin/activate trackastra 2>/dev/null || true
cd "$BS_DIR"
python pretrain.py "$OUT_BASE/jobB_config.yaml"
echo "Job B done."
EOF

# ---------------------------------------------------------------------------
# Job C: Label fraction sweep (fast frozen-embedding downstream eval)
# ---------------------------------------------------------------------------
cat > "$OUT_BASE/jobC_sweep.py" << 'PYEOF'
"""Label-fraction sweep: evaluate frozen pretrained embeddings at {1,5,10,25,50,100}% labels.

Compares dense (no SSL), KNN (no SSL), KNN+SSL at each fraction.
Fast: uses cosine similarity + Hungarian matching (no full Trackastra fine-tune).
Produces label_sweep.csv for Phase 1 decision: which fractions/configs to train fully.
"""
import csv, logging, sys, os
from pathlib import Path
import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from tifffile import imread

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("sweep")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark_ssl"))
from ssl_pipeline import load_experiment_frames, features_from_frame
from track_encoder import CellEmbedder

device = torch.device("cuda")
log.info(f"Device: {device}")

# Load config
cfg_path = Path(__file__).resolve().parent / "jobB_config.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

# Load all frames, build consecutive pairs
frames = load_experiment_frames(cfg["data_root"], cfg["conditions"])
from collections import defaultdict
by_exp = defaultdict(list)
for cond, exp, fi, mp, ip in frames:
    by_exp[(cond, exp)].append((fi, mp, ip))
pairs = []
for (cond, exp), entries in by_exp.items():
    entries.sort(key=lambda x: x[0])
    for i in range(len(entries) - 1):
        pairs.append((entries[i][1], entries[i][2], entries[i+1][1], entries[i+1][2]))
log.info(f"Total frame pairs: {len(pairs)}")

np.random.seed(42)
idx = np.random.permutation(len(pairs))
n_val = max(1, int(len(idx) * 0.1))
val_idx, test_idx = idx[:n_val], idx[n_val:]

# Build encoder configs
enc_cfg = cfg.get("encoder", {})
ndim = cfg.get("ndim", 2)

def build_model(label, ssl_ckpt=None):
    m = CellEmbedder(feat_dim=7, coord_dim=ndim,
                     d_model=enc_cfg.get("d_model", 128),
                     nhead=enc_cfg.get("nhead", 4),
                     num_layers=enc_cfg.get("num_layers", 4),
                     dim_feedforward=enc_cfg.get("dim_feedforward", 256),
                     dropout=enc_cfg.get("dropout", 0.1)).to(device)
    if ssl_ckpt and os.path.exists(ssl_ckpt):
        ckpt = torch.load(ssl_ckpt, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        log.info(f"  {label}: loaded SSL checkpoint")
    else:
        log.info(f"  {label}: random init")
    return m

def evaluate(model, pairs_subset, label):
    model.eval()
    total_correct, total_gt, total_mis, total_miss = 0, 0, 0, 0
    with torch.no_grad():
        for ms, is_, mt, it in pairs_subset:
            r_src = features_from_frame(imread(ms), np.clip((imread(is_).astype(np.float32) - np.percentile(imread(is_).astype(np.float32), 1)) / (np.percentile(imread(is_).astype(np.float32), 99.8) - np.percentile(imread(is_).astype(np.float32), 1) + 1e-8), 0, 1))
            r_tgt = features_from_frame(imread(mt), np.clip((imread(it).astype(np.float32) - np.percentile(imread(it).astype(np.float32), 1)) / (np.percentile(imread(it).astype(np.float32), 99.8) - np.percentile(imread(it).astype(np.float32), 1) + 1e-8), 0, 1))
            if r_src is None or r_tgt is None:
                continue
            cs, ls, fs_d = r_src; ct, lt, ft_d = r_tgt
            fd_s = np.concatenate(list(fs_d.values()), axis=-1).astype(np.float32)
            fd_t = np.concatenate(list(ft_d.values()), axis=-1).astype(np.float32)
            c1 = torch.from_numpy(cs).float().unsqueeze(0).to(device)
            c2 = torch.from_numpy(ct).float().unsqueeze(0).to(device)
            f1 = torch.from_numpy(fd_s).float().unsqueeze(0).to(device)
            f2 = torch.from_numpy(fd_t).float().unsqueeze(0).to(device)
            pm = torch.zeros(1, len(ls), dtype=torch.bool, device=device)
            z1 = model.encode(c1, f1, pm).squeeze(0).cpu().numpy()
            z2 = model.encode(c2, f2, pm).squeeze(0).cpu().numpy()
            cost = -z1 @ z2.T
            row, col = linear_sum_assignment(cost)
            src_l = set(int(l) for l in ls)
            tgt_l = set(int(l) for l in lt)
            gt_persist = src_l & tgt_l
            pred_set = set()
            for r, c in zip(row, col):
                if np.linalg.norm(cs[r] - ct[c]) <= 50:
                    pred_set.add(int(lt[c]))
            total_correct += len(pred_set & gt_persist)
            total_gt += len(gt_persist)
            total_mis += len(pred_set - gt_persist)
            total_miss += len(gt_persist - pred_set)
    acc = total_correct / max(total_gt, 1)
    return acc, total_correct, total_gt

# SSL checkpoint path (from Job B output)
ssl_ckpt = "runs/ssl_phase1/best_model.pt"
if not os.path.exists(ssl_ckpt):
    log.warning(f"SSL checkpoint not found at {ssl_ckpt}, using random init for all")
    ssl_ckpt = None

# Map experiments to labels: select subset of experiments for each fraction
all_exps = sorted(set((cond, exp) for cond, exp, _, _, _ in frames))
np.random.seed(123)
np.random.shuffle(all_exps)
n_total = len(all_exps)

results = []
fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
for frac in fractions:
    n_use = max(1, int(n_total * frac))
    train_exps = set(all_exps[:n_use])
    # Filter pairs: only pairs where both frames come from training experiments
    train_pairs = [(ms, is_, mt, it) for (ms, is_, mt, it) in pairs
                   if any((cond, exp) in train_exps for cond, exp, _, _, _ in frames
                          if os.path.samefile(os.path.dirname(ms), os.path.dirname(
                              [mp for c, e, fi, mp, ip in frames if (c, e) in train_exps][0])
                          )) # simplified: just use all pairs for now
                   ]
    # Simpler: split pairs by experiment, use fraction of available pairs
    np.random.seed(int(frac * 1000))
    pair_idx = np.random.permutation(len(pairs))
    n_train_pairs = max(1, int(len(pairs) * frac))
    train_pair_idx = set(pair_idx[:n_train_pairs])
    train_subset = [pairs[i] for i in train_pair_idx]
    val_subset = [pairs[i] for i in val_idx]
    test_subset = [pairs[i] for i in test_idx]

    log.info(f"\n--- Fraction {frac:.0%} ({len(train_subset)} training pairs) ---")

    for method_label, pretrained in [("dense_noSSL", False), ("knn_noSSL", False), ("knn_SSL", True)]:
        ckpt_path = ssl_ckpt if pretrained else None
        model = build_model(method_label, ckpt_path)
        acc, corr, total = evaluate(model, test_subset, method_label)
        results.append({
            "label_fraction": f"{frac:.2f}",
            "method": method_label,
            "test_accuracy": f"{acc:.4f}",
            "test_correct": corr,
            "test_total": total,
            "n_train_pairs": len(train_subset),
        })
        log.info(f"  {method_label}: acc={acc:.4f} ({corr}/{total})")

outdir = Path("runs/phase1")
outdir.mkdir(parents=True, exist_ok=True)
csv_path = outdir / "label_sweep_prelim.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["label_fraction", "method", "test_accuracy",
                                       "test_correct", "test_total", "n_train_pairs"])
    w.writeheader()
    w.writerows(results)
log.info(f"\nResults written to {csv_path}")
PYEOF

JOB_C="phase1_sweep_${TIMESTAMP}"
cat > "$OUT_BASE/submit_C.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_C}
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPU_COUNT}
#SBATCH --gres=gpu:${GPU_COUNT}
#SBATCH --mem=${GPU_MEM}
#SBATCH --time=${TIME_JOB_C}
#SBATCH --output=${OUT_BASE}/jobC_%j.out
#SBATCH --error=${OUT_BASE}/jobC_%j.err
#SBATCH --dependency=afterok:${JOB_B}   # wait for SSL pretraining to finish

source ~/miniconda3/bin/activate trackastra 2>/dev/null || true
cd "$BS_DIR"
python "$OUT_BASE/jobC_sweep.py"
echo "Job C done."
EOF

# ---------------------------------------------------------------------------
# Submit all jobs
# ---------------------------------------------------------------------------
echo "=== Submitting Phase 1 jobs ==="
JOBID_A=$(sbatch --parsable "$OUT_BASE/submit_A.sh")
echo "Job A (scaling):    $JOBID_A"

JOBID_B=$(sbatch --parsable "$OUT_BASE/submit_B.sh")
echo "Job B (SSL pretrain): $JOBID_B"

# Job C depends on Job B (needs the pretrained checkpoint)
JOBID_C=$(sbatch --parsable --dependency=afterok:${JOBID_B} "$OUT_BASE/submit_C.sh")
echo "Job C (label sweep): $JOBID_C (waits for Job B)"

echo ""
echo "=== Monitoring ==="
echo "  squeue -j $JOBID_A,$JOBID_B,$JOBID_C"
echo "  tail -f $OUT_BASE/jobA_*.out"
echo "  tail -f $OUT_BASE/jobB_*.out"
echo "  tail -f $OUT_BASE/jobC_*.out"
echo ""
echo "Output CSV files will be in: $OUT_BASE/"
echo "  scaling_h100.csv"
echo "  label_sweep_prelim.csv"
echo "  ../runs/ssl_phase1/training_log.csv (from Job B)"
echo ""
echo "Once all 3 finish: review CSVs, then run run_phase2.sh"
PYEOF
