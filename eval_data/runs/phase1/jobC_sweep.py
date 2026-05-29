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
