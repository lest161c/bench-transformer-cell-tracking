#!/usr/bin/env python3
"""Train TinyCNN on edge prediction, end-to-end with linear probe.

Trains TinyCNN + linear probe jointly using BCEWithLogitsLoss on
same-cell (positive) vs different-cell (negative) edge prediction.

Data: 30 consecutive frame pairs from vanvliet (rpsM, recA, pheA, metA).
Patches: 64x64 at cell centroids.
Labels: same-cell = positive (from man_track.txt).

Saves trained CNN state_dict to probe/cnn_probe.pt
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from skimage.measure import regionprops_table
from tifffile import imread

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("train_cnn_probe")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

PATCH_SIZE = 64


# ═══════════════════════════════════════════════════════════════════════════════
#  TinyCNN + Probe
# ═══════════════════════════════════════════════════════════════════════════════

class TinyCNN(nn.Module):
    """Tiny CNN feature extractor (~3K params)."""
    def __init__(self, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x32
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),  # 32-dim
        )
        self.fc = nn.Linear(32, out_dim)

    def forward(self, patches):
        """patches: (N, 1, 64, 64) -> (N, out_dim)"""
        x = self.conv(patches)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class EdgeProbe(nn.Module):
    """Linear probe: concat(cnn_t[i], cnn_n[j]) -> score."""
    def __init__(self, feat_dim=64):
        super().__init__()
        self.fc = nn.Linear(2 * feat_dim, 1)

    def forward(self, cnn_t, cnn_n):
        N1, D = cnn_t.shape
        N2 = cnn_n.shape[0]
        t = cnn_t.unsqueeze(1).expand(-1, N2, -1)
        n = cnn_n.unsqueeze(0).expand(N1, -1, -1)
        pairs = torch.cat([t, n], dim=-1)
        return self.fc(pairs.view(-1, 2 * D)).view(N1, N2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_frame(mask_path, img_path):
    """Load mask + image, return (coords, labels, img) or None."""
    mask = imread(mask_path)
    img = imread(img_path).astype(np.float32)
    p1, p998 = np.percentile(img, (1, 99.8))
    img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)
    props = regionprops_table(mask, properties=("label", "centroid"))
    if not props or len(props["label"]) < 2:
        return None
    coords = np.stack([props["centroid-0"], props["centroid-1"]], axis=-1).astype(np.float32)
    return coords, props["label"].astype(np.int32), img


def extract_patches(img, centroids):
    """Extract 64x64 patches at centroids with edge padding."""
    h, w = img.shape[-2:]
    half = PATCH_SIZE // 2
    patches = []
    for cy, cx in centroids:
        cy_i, cx_i = int(round(float(cy))), int(round(float(cx)))
        cy_i, cx_i = np.clip(cy_i, 0, h - 1), np.clip(cx_i, 0, w - 1)
        y1, x1 = cy_i - half, cx_i - half
        y2, x2 = cy_i + half, cx_i + half
        pt, pb = max(0, -y1), max(0, y2 - h)
        pl, pr = max(0, -x1), max(0, x2 - w)
        y1c, x1c = max(0, y1), max(0, x1)
        y2c, x2c = min(h, y2), min(w, x2)
        crop = (
            img[y1c:y2c, x1c:x2c]
            if y2c > y1c and x2c > x1c
            else np.zeros((1, 1), dtype=np.float32)
        )
        if pt or pb or pl or pr:
            crop = np.pad(crop, ((pt, pb), (pl, pr)), mode="reflect")
        if crop.shape != (PATCH_SIZE, PATCH_SIZE):
            crop = np.pad(
                crop,
                tuple((0, max(0, t)) for t in [PATCH_SIZE - s for s in crop.shape]),
                mode="reflect",
            )[:PATCH_SIZE, :PATCH_SIZE]
        patches.append(crop)
    if patches:
        return np.stack(patches).astype(np.float32)
    return np.zeros((0, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)


def load_tracklets(man_track_path):
    """Parse man_track.txt: label -> {t1, t2, parent}."""
    df = pd.read_csv(
        man_track_path, delimiter=" ", header=None, names=["label", "t1", "t2", "parent"]
    )
    tracklets = {}
    for _, row in df.iterrows():
        tracklets[int(row["label"])] = {
            "t1": int(row["t1"]),
            "t2": int(row["t2"]),
            "parent": int(row["parent"]),
        }
    return tracklets


def scan_consecutive_pairs(data_root, conditions, max_pairs):
    """Find consecutive frame pairs from experiments."""
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
                        pairs.append(
                            (str(masks[i]), str(masks[i + 1]), str(ip1), str(ip2), str(man_track_txt))
                        )
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs


def build_edge_data(pairs, max_pairs=30):
    """Build edge data with 64x64 patches and same-cell targets.

    Returns list of dicts with patches_t, patches_n, target matrices.
    target[i,j] = 1 for same-cell or parent->child, 0 otherwise.
    """
    edge_data = []
    for mt, mn, it_, in_, man_txt in pairs[:max_pairs]:
        rt = load_frame(mt, it_)
        rn = load_frame(mn, in_)
        if rt is None or rn is None:
            continue
        coords_t, labels_t, imgt = rt
        coords_n, labels_n, imgn = rn
        if len(labels_t) < 3 or len(labels_n) < 3:
            continue
        tracklets = load_tracklets(man_txt)

        # Extract 64x64 patches at centroids
        p_t = extract_patches(imgt, coords_t)
        p_n = extract_patches(imgn, coords_n)

        # Build target: same-cell OR parent->child = positive
        N1, N2 = len(labels_t), len(labels_n)
        target = torch.zeros(N1, N2, dtype=torch.float32)
        for i, lt in enumerate(labels_t):
            for j, ln in enumerate(labels_n):
                if lt == ln:
                    target[i, j] = 1.0
                elif ln in tracklets and tracklets[ln]["parent"] == lt:
                    target[i, j] = 1.0

        if target.sum() < 1:
            continue

        edge_data.append({
            "patches_t": torch.from_numpy(p_t).float(),
            "patches_n": torch.from_numpy(p_n).float(),
            "target": target,
        })
    return edge_data


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(scores, target):
    """Balanced accuracy and F1 from logits."""
    pred = (scores > 0.0).float()
    target_np = target.cpu().numpy().ravel()
    pred_np = pred.cpu().numpy().ravel()
    bal_acc = balanced_accuracy_score(target_np, pred_np)
    f1 = f1_score(target_np, pred_np, zero_division=0)
    return bal_acc, f1


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    script_dir = Path(__file__).parent.resolve()
    # benchmark_ssl/probe/ -> benchmark_ssl/ -> research-proj/data/vanvliet
    data_root = str(script_dir.parent.parent / "data" / "vanvliet")
    conditions = ["rpsM", "recA", "pheA", "metA"]
    max_pairs = 30
    steps = 200
    lr = 1e-2
    cnn_out_dim = 64

    logger.info(f"{'='*60}")
    logger.info(f"TinyCNN + EdgeProbe: End-to-End Training")
    logger.info(f"{'='*60}")
    logger.info(f"  Data root: {data_root}")
    logger.info(f"  Conditions: {conditions}")
    logger.info(f"  Max pairs: {max_pairs}")
    logger.info(f"  Steps: {steps}, LR: {lr}")
    logger.info(f"  CNN out_dim: {cnn_out_dim}")
    logger.info(f"  Device: {device}")

    # ─── Scan pairs ───────────────────────────────────────────────────────
    logger.info("Scanning consecutive frame pairs...")
    pairs = scan_consecutive_pairs(data_root, conditions, max_pairs)
    logger.info(f"Found {len(pairs)} consecutive frame pairs")
    if len(pairs) < 3:
        logger.error("Need at least 3 pairs. Check --data-root.")
        sys.exit(1)

    # ─── Build edge data ──────────────────────────────────────────────────
    logger.info("Building edge data...")
    edge_data = build_edge_data(pairs, max_pairs=max_pairs)
    logger.info(f"Edge data: {len(edge_data)} frame pairs")
    if len(edge_data) < 2:
        logger.error("Need at least 2 usable pairs.")
        sys.exit(1)

    # ─── Train/val split ──────────────────────────────────────────────────
    np.random.seed(SEED)
    idx = np.random.permutation(len(edge_data))
    n_val = max(1, int(len(edge_data) * 0.2))
    train_data = [edge_data[i] for i in idx[:-n_val]]
    val_data = [edge_data[i] for i in idx[-n_val:]]
    logger.info(f"  Train: {len(train_data)} pairs, Val: {len(val_data)} pairs")

    # ─── Build models ──────────────────────────────────────────────────────
    cnn = TinyCNN(out_dim=cnn_out_dim).to(device)
    probe = EdgeProbe(feat_dim=cnn_out_dim).to(device)
    params = list(cnn.parameters()) + list(probe.parameters())
    n_cnn = sum(p.numel() for p in cnn.parameters())
    n_total = sum(p.numel() for p in params)
    logger.info(f"  CNN params: {n_cnn}  Total params: {n_total}")

    # Compute positive weight from training data to handle extreme class imbalance
    total_pos = 0
    total_neg = 0
    for item in train_data:
        tgt = item["target"]
        total_pos += tgt.sum().item()
        total_neg += (1 - tgt).sum().item()
    pos_weight_val = max(1.0, total_neg / max(1.0, total_pos))
    logger.info(f"  Class balance: {total_pos:.0f} pos / {total_neg:.0f} neg, pos_weight={pos_weight_val:.2f}")

    opt = torch.optim.Adam(params, lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val))

    # ─── Training loop ────────────────────────────────────────────────────
    for step in range(steps):
        cnn.train()
        probe.train()
        train_losses = []
        for item in train_data:
            pt = item["patches_t"].to(device).unsqueeze(1)  # (N, 1, 64, 64)
            pn = item["patches_n"].to(device).unsqueeze(1)
            target = item["target"].to(device)

            feat_t = cnn(pt)
            feat_n = cnn(pn)
            scores = probe(feat_t, feat_n)
            loss = criterion(scores, target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        # Evaluate every 20 steps
        if step % 20 == 0 or step == steps - 1:
            cnn.eval()
            probe.eval()
            val_losses, bal_accs, f1s = [], [], []
            with torch.no_grad():
                for item in val_data:
                    pt = item["patches_t"].to(device).unsqueeze(1)
                    pn = item["patches_n"].to(device).unsqueeze(1)
                    target = item["target"].to(device)
                    feat_t = cnn(pt)
                    feat_n = cnn(pn)
                    scores = probe(feat_t, feat_n)
                    val_losses.append(criterion(scores, target).item())
                    ba, f1 = compute_metrics(scores, target)
                    bal_accs.append(ba)
                    f1s.append(f1)

            avg_loss = float(np.mean(train_losses))
            avg_vl = float(np.mean(val_losses))
            avg_ba = float(np.mean(bal_accs))
            avg_f1 = float(np.mean(f1s))
            logger.info(
                f"  Step {step:3d}: train_loss={avg_loss:.4f}  "
                f"val_loss={avg_vl:.4f}  bal_acc={avg_ba:.4f}  f1={avg_f1:.4f}"
            )

    # ─── Final evaluation ─────────────────────────────────────────────────
    cnn.eval()
    probe.eval()
    final_losses, final_bal, final_f1 = [], [], []
    with torch.no_grad():
        for item in val_data:
            pt = item["patches_t"].to(device).unsqueeze(1)
            pn = item["patches_n"].to(device).unsqueeze(1)
            target = item["target"].to(device)
            feat_t = cnn(pt)
            feat_n = cnn(pn)
            scores = probe(feat_t, feat_n)
            final_losses.append(criterion(scores, target).item())
            ba, f1 = compute_metrics(scores, target)
            final_bal.append(ba)
            final_f1.append(f1)

    final_bce = float(np.mean(final_losses))
    final_bal_acc = float(np.mean(final_bal))
    final_f1_score = float(np.mean(final_f1))

    print(f"\n{'='*60}")
    print(f"Training Results:")
    print(f"  Final BCE loss:       {final_bce:.4f}")
    print(f"  Final balanced acc:   {final_bal_acc:.4f}")
    print(f"  Final F1:             {final_f1_score:.4f}")
    print(f"{'='*60}")

    # ─── Save CNN weights ──────────────────────────────────────────────────
    save_dir = Path(__file__).parent
    save_path = save_dir / "cnn_probe.pt"
    torch.save(cnn.state_dict(), save_path)
    logger.info(f"CNN weights saved to {save_path}")

    return final_bce, final_bal_acc, final_f1_score


if __name__ == "__main__":
    main()
