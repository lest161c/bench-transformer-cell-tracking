#!/usr/bin/env python3
"""Probe evaluation: frozen NT-Xent CNN vs frozen random CNN on edge prediction.

Usage:
    python cnn_probe.py --scale small --max-pairs 5 --steps 20    # quick test
    python cnn_probe.py --scale medium                            # full run
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from skimage.measure import regionprops_table, regionprops as sk_regionprops
from tifffile import imread

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cnn_probe")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

PATCH_SIZE = 64
OUT_DIM = 128


# ═══════════════════════════════════════════════════════════════════════════════
#  ScaledCNN (same arch as cnn_ssl.py)
# ═══════════════════════════════════════════════════════════════════════════════

class ScaledCNN(nn.Module):
    """ConvNet for 64×64 grayscale cell patches. Output: 128-dim embedding."""
    def __init__(self, scale='medium', out_dim=OUT_DIM):
        super().__init__()
        if scale == 'small':
            ch = [8, 16, 32]
        elif scale == 'medium':
            ch = [16, 32, 64]
        else:
            ch = [32, 64, 128]

        layers = []
        in_ch = 1
        for c in ch:
            layers += [nn.Conv2d(in_ch, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = c
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch[-1] * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, patches):
        return self.fc(self.conv(patches))


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading (matches probe/edge_probe.py conventions)
# ═══════════════════════════════════════════════════════════════════════════════

def load_frame(mask_path, img_path):
    """Returns (coords, labels, img, mask) or None."""
    mask = imread(mask_path)
    img = imread(img_path).astype(np.float32)
    p1, p998 = np.percentile(img, (1, 99.8))
    img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)
    props = regionprops_table(mask, properties=("label", "centroid"))
    if not props or len(props["label"]) < 2:
        return None
    coords = np.stack([props["centroid-0"], props["centroid-1"]], axis=-1).astype(np.float32)
    return coords, props["label"].astype(np.int32), img, mask


def scan_consecutive_pairs(data_root, conditions, max_pairs):
    """Find consecutive frame pairs with man_track.txt for edge labels."""
    dr = Path(data_root)
    pairs = []
    for cond in conditions:
        for exp in sorted(dr.glob(f"{cond}/*")):
            if not exp.is_dir():
                continue
            tra_dir = exp / "TRA"
            img_dir = exp / "img"
            if not tra_dir.exists() or not img_dir.exists():
                continue
            man_track_txt = tra_dir / "man_track.txt"
            if not man_track_txt.exists():
                continue
            masks = sorted(tra_dir.glob("man_track*.tif"))
            for i in range(len(masks) - 1):
                stem1 = masks[i].stem.replace("man_track", "")
                stem2 = masks[i + 1].stem.replace("man_track", "")
                try:
                    f1, f2 = int(stem1), int(stem2)
                except ValueError:
                    continue
                if f2 == f1 + 1:
                    ip1 = img_dir / f"t{f1:06d}.tif"
                    ip2 = img_dir / f"t{f2:06d}.tif"
                    if ip1.exists() and ip2.exists():
                        pairs.append((
                            str(masks[i]), str(masks[i + 1]),
                            str(ip1), str(ip2),
                            str(man_track_txt),
                        ))
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs


def extract_patches(img, centroids, patch_size=PATCH_SIZE):
    """Extract square patches centered on centroids. Handles edge padding."""
    h, w = img.shape[-2:]
    half = patch_size // 2
    patches = []
    for cy, cx in centroids:
        cy_i = int(round(float(cy)))
        cx_i = int(round(float(cx)))
        cy_i = np.clip(cy_i, 0, h - 1)
        cx_i = np.clip(cx_i, 0, w - 1)
        y1 = cy_i - half
        x1 = cx_i - half
        y2 = cy_i + half
        x2 = cx_i + half
        pt = max(0, -y1)
        pb = max(0, y2 - h)
        pl = max(0, -x1)
        pr = max(0, x2 - w)
        y1c = max(0, y1)
        x1c = max(0, x1)
        y2c = min(h, y2)
        x2c = min(w, x2)
        crop = img[y1c:y2c, x1c:x2c] if (y2c > y1c and x2c > x1c) else np.zeros((1, 1), dtype=np.float32)
        if pt or pb or pl or pr:
            crop = np.pad(crop, ((pt, pb), (pl, pr)), mode="reflect")
        if crop.shape != (patch_size, patch_size):
            crop = np.pad(
                crop,
                tuple((0, max(0, t)) for t in [patch_size - s for s in crop.shape]),
                mode="reflect",
            )[:patch_size, :patch_size]
        patches.append(crop)
    if patches:
        return np.stack(patches).astype(np.float32)
    return np.zeros((0, patch_size, patch_size), dtype=np.float32)


def load_tracklets(man_track_path):
    """Parse man_track.txt into dict: label -> {t1, t2, parent}."""
    df = pd.read_csv(man_track_path, delimiter=' ', header=None,
                     names=['label', 't1', 't2', 'parent'])
    tracklets = {}
    for _, row in df.iterrows():
        tracklets[int(row['label'])] = {
            't1': int(row['t1']), 't2': int(row['t2']), 'parent': int(row['parent']),
        }
    return tracklets


def build_targets(labels_t, labels_n, tracklets):
    """Build edge target: 1 for same-cell OR parent→child, 0 otherwise."""
    N1, N2 = len(labels_t), len(labels_n)
    target = torch.zeros(N1, N2, dtype=torch.float32)
    for i, lt in enumerate(labels_t):
        for j, ln in enumerate(labels_n):
            if lt == ln:
                target[i, j] = 1.0
            elif ln in tracklets and tracklets[ln]['parent'] == lt:
                target[i, j] = 1.0
    return target


# ═══════════════════════════════════════════════════════════════════════════════
#  Probe models
# ═══════════════════════════════════════════════════════════════════════════════

class LinearProbe(nn.Module):
    """Linear probe: concat(feat_t[i], feat_n[j]) → score."""
    def __init__(self, feat_dim):
        super().__init__()
        self.fc = nn.Linear(2 * feat_dim, 1)

    def forward(self, feat_t, feat_n):
        N1, D = feat_t.shape
        N2 = feat_n.shape[0]
        feat_t_exp = feat_t.unsqueeze(1).expand(-1, N2, -1)
        feat_n_exp = feat_n.unsqueeze(0).expand(N1, -1, -1)
        pairs = torch.cat([feat_t_exp, feat_n_exp], dim=-1)
        return self.fc(pairs.view(-1, 2 * D)).view(N1, N2)


class MLPProbe(nn.Module):
    """2-layer MLP probe: concat(feat_t[i], feat_n[j]) → hidden → score."""
    def __init__(self, feat_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat_t, feat_n):
        N1, D = feat_t.shape
        N2 = feat_n.shape[0]
        feat_t_exp = feat_t.unsqueeze(1).expand(-1, N2, -1)
        feat_n_exp = feat_n.unsqueeze(0).expand(N1, -1, -1)
        pairs = torch.cat([feat_t_exp, feat_n_exp], dim=-1)
        return self.net(pairs.view(-1, 2 * D)).view(N1, N2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Training helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(scores, target):
    """Compute accuracy, balanced accuracy, F1."""
    pred = (scores > 0.0).float()
    target_np = target.cpu().numpy().ravel()
    pred_np = pred.cpu().numpy().ravel()
    acc = accuracy_score(target_np, pred_np)
    bal_acc = balanced_accuracy_score(target_np, pred_np)
    f1 = f1_score(target_np, pred_np, zero_division=0)
    return acc, bal_acc, f1


def train_probe(probe, train_data, val_data, steps=200, lr=1e-3, patience=20, eval_every=20):
    """Train a probe model. Returns dict of metrics."""
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_bal_acc = 0.0
    best_state = None
    patience_counter = 0

    for step in range(steps):
        probe.train()
        train_losses = []
        for item in train_data:
            feat_t = item["feat_t"].to(device)
            feat_n = item["feat_n"].to(device)
            target = item["target"].to(device)
            scores = probe(feat_t, feat_n)
            loss = criterion(scores, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        if step % eval_every == 0 or step == steps - 1:
            probe.eval()
            val_losses = []
            val_metrics = {"acc": [], "bal_acc": [], "f1": []}
            with torch.no_grad():
                for item in val_data:
                    feat_t = item["feat_t"].to(device)
                    feat_n = item["feat_n"].to(device)
                    target = item["target"].to(device)
                    scores = probe(feat_t, feat_n)
                    loss = criterion(scores, target)
                    val_losses.append(loss.item())
                    acc, bal_acc, f1 = compute_metrics(scores, target)
                    val_metrics["acc"].append(acc)
                    val_metrics["bal_acc"].append(bal_acc)
                    val_metrics["f1"].append(f1)

            avg_val_bal_acc = float(np.mean(val_metrics["bal_acc"]))
            if avg_val_bal_acc > best_val_bal_acc:
                best_val_bal_acc = avg_val_bal_acc
                patience_counter = 0
                best_state = probe.state_dict()
            else:
                patience_counter += eval_every
                if patience_counter >= patience:
                    logger.info(f"    Early stopping at step {step}")
                    break

    if best_state is not None:
        probe.load_state_dict(best_state)

    probe.eval()
    final_metrics = {"acc": [], "bal_acc": [], "f1": []}
    with torch.no_grad():
        for item in val_data:
            feat_t = item["feat_t"].to(device)
            feat_n = item["feat_n"].to(device)
            target = item["target"].to(device)
            scores = probe(feat_t, feat_n)
            acc, bal_acc, f1 = compute_metrics(scores, target)
            final_metrics["acc"].append(acc)
            final_metrics["bal_acc"].append(bal_acc)
            final_metrics["f1"].append(f1)

    return {
        "probe": probe.cpu(),
        "best_val_bal_acc": best_val_bal_acc,
        "final_acc": float(np.mean(final_metrics["acc"])),
        "final_bal_acc": float(np.mean(final_metrics["bal_acc"])),
        "final_f1": float(np.mean(final_metrics["f1"])),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Build edge data with CNN features
# ═══════════════════════════════════════════════════════════════════════════════

def build_cnn_edge_data(pairs, cnn_model, max_pairs=30):
    """Build edge prediction data using frozen CNN features."""
    edge_data = []
    total = 0

    for mt, mn, it_, in_, man_txt in pairs[:max_pairs]:
        rt = load_frame(mt, it_)
        rn = load_frame(mn, in_)
        if rt is None or rn is None:
            continue
        ct, lt, imgt, mask_t = rt
        cn, ln, imgn, mask_n = rn

        if len(lt) < 3 or len(ln) < 3:
            continue

        tracklets = load_tracklets(man_txt)

        # Extract patches and compute CNN features
        pt = extract_patches(imgt, ct)
        pn = extract_patches(imgn, cn)

        pt_t = torch.from_numpy(pt).float().unsqueeze(1).to(device)
        pn_t = torch.from_numpy(pn).float().unsqueeze(1).to(device)

        with torch.no_grad():
            feat_t = cnn_model(pt_t).cpu()
            feat_n = cnn_model(pn_t).cpu()

        # Build target
        target = build_targets(lt, ln, tracklets)
        if target.sum() < 1:
            continue

        edge_data.append({
            "feat_t": feat_t,
            "feat_n": feat_n,
            "target": target,
            "n_pos": target.sum().item(),
            "n_neg": target.numel() - target.sum().item(),
        })
        total += 1

    logger.info(f"  Built {total} edge pairs")
    return edge_data


# ═══════════════════════════════════════════════════════════════════════════════
#  Main evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_cnn(label, cnn_model, pairs, args):
    """Evaluate frozen CNN on edge prediction."""
    logger.info(f"\n  Evaluating {label} CNN...")

    # Build edge data
    edge_data = build_cnn_edge_data(pairs, cnn_model, args.max_pairs)
    if len(edge_data) < 3:
        logger.warning(f"  Not enough data ({len(edge_data)}), skipping")
        return None

    # Train/val split
    np.random.seed(SEED)
    idx = np.random.permutation(len(edge_data))
    n_val = max(1, int(len(edge_data) * 0.2))
    train_data = [edge_data[i] for i in idx[:-n_val]]
    val_data = [edge_data[i] for i in idx[-n_val:]]
    logger.info(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # Linear probe
    logger.info("  Training linear probe...")
    probe = LinearProbe(OUT_DIM)
    lin_result = train_probe(probe, train_data, val_data,
                             steps=args.steps, lr=1e-3, patience=20, eval_every=20)
    logger.info(f"    Linear: bal_acc={lin_result['final_bal_acc']:.4f}, "
                f"f1={lin_result['final_f1']:.4f}")

    # MLP probe
    logger.info("  Training MLP probe...")
    probe = MLPProbe(OUT_DIM)
    mlp_result = train_probe(probe, train_data, val_data,
                             steps=args.steps, lr=1e-4, patience=20, eval_every=20)
    logger.info(f"    MLP: bal_acc={mlp_result['final_bal_acc']:.4f}, "
                f"f1={mlp_result['final_f1']:.4f}")

    return {"linear": lin_result, "mlp": mlp_result}


def main():
    parser = argparse.ArgumentParser(
        description="Probe evaluation: frozen NT-Xent CNN vs random CNN"
    )
    parser.add_argument("--data-root", default="../../data/vanvliet")
    parser.add_argument("--conditions", default="rpsM,recA,pheA,metA")
    parser.add_argument("--max-pairs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--checkpoint", default="probe/cnn_ntxent.pt",
                        help="Path to NT-Xent pretrained checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    global SEED
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    conditions = [c.strip() for c in args.conditions.split(",")]
    logger.info(f"Scale: {args.scale}, Checkpoint: {args.checkpoint}")
    logger.info(f"Conditions: {conditions}, Max pairs: {args.max_pairs}")

    # Scan pairs
    logger.info("Scanning consecutive frame pairs...")
    pairs = scan_consecutive_pairs(args.data_root, conditions, args.max_pairs)
    logger.info(f"Found {len(pairs)} pairs")
    if len(pairs) < 3:
        logger.error("Need at least 3 pairs. Check data.")
        sys.exit(1)

    # Build CNNs
    cnn_random = ScaledCNN(scale=args.scale, out_dim=OUT_DIM).to(device)
    cnn_random.eval()
    logger.info(f"Random CNN: {sum(p.numel() for p in cnn_random.parameters()):,} params")

    cnn_ntxent = ScaledCNN(scale=args.scale, out_dim=OUT_DIM).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cnn_ntxent.load_state_dict(ckpt["model_state_dict"])
    cnn_ntxent.eval()
    logger.info(f"NT-Xent CNN loaded from {args.checkpoint}")

    # Evaluate
    results_ntxent = evaluate_cnn("NT-Xent", cnn_ntxent, pairs, args)
    results_random = evaluate_cnn("Random", cnn_random, pairs, args)

    # Print comparison table
    print()
    print("=" * 70)
    print("CNN PROBE BENCHMARK — RESULTS")
    print("=" * 70)
    print(f"{'CNN Variant':<20} {'Probe':<8} {'BalAcc':<12} {'F1':<12} {'Acc':<12}")
    print("-" * 70)

    rows = []
    for label, results in [("NT-Xent", results_ntxent), ("Random", results_random)]:
        if results is None:
            print(f"{label:<20} {'—':<8} {'—':<12} {'—':<12} {'—':<12}")
            continue
        for ptype in ["linear", "mlp"]:
            r = results[ptype]
            print(f"{label:<20} {ptype:<8} {r['final_bal_acc']:<12.4f} "
                  f"{r['final_f1']:<12.4f} {r['final_acc']:<12.4f}")
            rows.append({
                "CNN": label, "Probe": ptype,
                "BalAcc": r["final_bal_acc"],
                "F1": r["final_f1"],
                "Acc": r["final_acc"],
            })

    print("-" * 70)

    # Verdict
    print()
    print("VERDICT:")
    if results_ntxent and results_random:
        ntx_best_bal = max(results_ntxent["linear"]["final_bal_acc"],
                           results_ntxent["mlp"]["final_bal_acc"])
        rand_best_bal = max(results_random["linear"]["final_bal_acc"],
                            results_random["mlp"]["final_bal_acc"])
        delta = ntx_best_bal - rand_best_bal
        if delta > 0.02:
            print(f"  ✓ NT-Xent CNN outperforms random CNN by Δ={delta:.4f} bal_acc")
        elif delta < -0.02:
            print(f"  ✗ NT-Xent CNN underperforms random CNN by Δ={delta:.4f} bal_acc")
        else:
            print(f"  ∼ NT-Xent and random CNN perform similarly (Δ={delta:.4f} bal_acc)")

        ntx_best_f1 = max(results_ntxent["linear"]["final_f1"],
                          results_ntxent["mlp"]["final_f1"])
        rand_best_f1 = max(results_random["linear"]["final_f1"],
                           results_random["mlp"]["final_f1"])
        print(f"  Best F1: NT-Xent={ntx_best_f1:.4f}, Random={rand_best_f1:.4f}")
    print("=" * 70)

    # Save results
    out_path = Path("probe") / "cnn_probe_results.txt"
    with open(out_path, "w") as f:
        f.write("CNN Probe Benchmark Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'CNN Variant':<20} {'Probe':<8} {'BalAcc':<12} {'F1':<12} {'Acc':<12}\n")
        f.write("-" * 70 + "\n")
        for r in rows:
            f.write(f"{r['CNN']:<20} {r['Probe']:<8} {r['BalAcc']:<12.4f} "
                    f"{r['F1']:<12.4f} {r['Acc']:<12.4f}\n")
        f.write("\n" + "=" * 70 + "\n")
    logger.info(f"Results saved: {out_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
