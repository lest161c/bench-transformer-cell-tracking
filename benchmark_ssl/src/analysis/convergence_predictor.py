"""DINO contrastive SSL convergence predictor.

Surgical micro-tests to predict if DINO+NT-Xent training will converge,
BEFORE launching 11h H100 training.

Tests:
  1. Gap vs distortion strength curves — does signal degrade gracefully?
  2. Micro-SSL training — train 1 projection layer, track gap improvement
  3. Feature space structure (PCA) — is the space learnable?
  4. Distortion difficulty ranking — which augs to emphasize?

``--test`` / ``--outdir`` mapping (each invocation writes one sub-run):

  --test all            → runs/convergence_prediction/
  --test gap_strength   → runs/convergence_prediction/
  --test micro_ssl      → runs/convergence_prediction/
  --test generalization → runs/generalization_test/
  --test hard           → runs/hard_test/
  --test ranking        → runs/convergence_prediction/
  --test pca            → runs/convergence_prediction/

The ``generalization`` and ``hard`` sub-runs were written by separate
invocations of this script (not by ``--test all``).

Usage:
    uv run python -m src.analysis.convergence_predictor --outdir runs/convergence_prediction
    uv run python -m src.analysis.convergence_predictor --test generalization \\
        --outdir runs/generalization_test
    uv run python -m src.analysis.convergence_predictor --test hard \\
        --outdir runs/hard_test
"""

import argparse
import base64
import io
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import regionprops_table
from tifffile import imread

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ssl_conv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# ─── DINO model ────────────────────────────────────────────────────────────────

logger.info("Loading DINOv2...")
# Pin the "main" branch so torch.hub does not probe the network for the
# default branch (fails offline / on dropped connections).
dino = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vits14").to(device).eval()
DINO_DIM = 384
PATCH_SIZE = 64


def extract_patches(img, centroids):
    """Always returns (N, PATCH_SIZE, PATCH_SIZE). Robust to out-of-bounds coords."""
    h, w = img.shape[-2:]
    half = PATCH_SIZE // 2
    patches = []
    for cy, cx in centroids:
        cy_i, cx_i = int(round(float(cy))), int(round(float(cx)))
        # Clamp center to image bounds
        cy_i = np.clip(cy_i, 0, h - 1)
        cx_i = np.clip(cx_i, 0, w - 1)
        y1, y2 = cy_i - half, cy_i + half
        x1, x2 = cx_i - half, cx_i + half
        # Compute padding needed
        pt = max(0, -y1)
        pb = max(0, y2 - h)
        pl = max(0, -x1)
        pr = max(0, x2 - w)
        y1c, y2c = max(0, y1), min(h, y2)
        x1c, x2c = max(0, x1), min(w, x2)
        # Ensure y2c > y1c and x2c > x1c
        if y2c <= y1c or x2c <= x1c:
            patches.append(np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32))
            continue
        crop = img[y1c:y2c, x1c:x2c]
        if pt or pb or pl or pr:
            crop = np.pad(crop, ((pt, pb), (pl, pr)), mode="reflect")
        # Final safety: force exact size
        if crop.shape != (PATCH_SIZE, PATCH_SIZE):
            crop = np.pad(crop, ((0, max(0, PATCH_SIZE - crop.shape[0])),
                                 (0, max(0, PATCH_SIZE - crop.shape[1]))),
                          mode="reflect")[:PATCH_SIZE, :PATCH_SIZE]
        patches.append(crop)
    return np.stack(patches) if patches else np.zeros((0, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)


@torch.no_grad()
def compute_dino_embs(patches_np):
    """Compute 384D DINOv2 embeddings for a batch of normalized patches."""
    if len(patches_np) == 0:
        return np.zeros((0, DINO_DIM), dtype=np.float32)
    pmin = patches_np.min(axis=(1, 2), keepdims=True)
    pmax = patches_np.max(axis=(1, 2), keepdims=True)
    pn = (patches_np - pmin) / (pmax - pmin + 1e-8)
    t = torch.from_numpy(pn).float().unsqueeze(1).to(device)
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    t = t.expand(-1, 3, -1, -1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    t = (t - mean) / std
    return dino(t).cpu().numpy()


# ─── data ──────────────────────────────────────────────────────────────────────

def load_frame(mask_path, img_path):
    """Load a mask+image frame, returning coords, labels, and normalized image."""
    mask = imread(mask_path)
    img = imread(img_path).astype(np.float32)
    p1, p998 = np.percentile(img, (1, 99.8))
    img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)
    props = regionprops_table(mask, properties=("label", "centroid"))
    if not props or len(props["label"]) == 0:
        return None
    coords = np.stack([props["centroid-0"], props["centroid-1"]], axis=-1).astype(np.float32)
    labels = props["label"].astype(np.int32)
    return coords, labels, img


def scan_frames(data_root, conditions, max_frames=200):
    """Collect (mask_path, image_path) pairs across conditions, up to max_frames."""
    frames = []
    dr = Path(data_root)
    for cond in conditions:
        for exp in sorted((dr / cond).iterdir()):
            if not exp.is_dir(): continue
            tra, img_dir = exp / "TRA", exp / "img"
            if not tra.exists() or not img_dir.exists(): continue
            for m in sorted(tra.glob("man_track*.tif")):
                stem = m.stem.replace("man_track", "")
                try: fi = int(stem)
                except ValueError: continue
                ip = img_dir / f"t{fi:06d}.tif"
                if ip.exists():
                    frames.append((str(m), str(ip)))
                    if len(frames) >= max_frames: return frames
    return frames


# ─── metrics ───────────────────────────────────────────────────────────────────

def compute_gap(embs_a, embs_b, labels_a, labels_b):
    """Compute intra/inter cosine similarity and the separation gap."""
    fa = embs_a / (np.linalg.norm(embs_a, axis=1, keepdims=True) + 1e-12)
    fb = embs_b / (np.linalg.norm(embs_b, axis=1, keepdims=True) + 1e-12)
    sim = fa @ fb.T
    intra, inter = [], []
    for i, lbl in enumerate(labels_a):
        matches = np.where(labels_b == lbl)[0]
        for j in range(len(labels_b)):
            if j in matches:
                intra.append(sim[i, j])
            else:
                inter.append(sim[i, j])
    return {
        "intra": float(np.mean(intra)) if intra else np.nan,
        "inter": float(np.mean(inter)) if inter else np.nan,
        "gap": float(np.mean(intra) - np.mean(inter)) if intra and inter else np.nan,
    }


def compute_recall(embs_a, embs_b, labels_a, labels_b, topk=1):
    """Fraction of cells whose correct match appears in the top-k by cosine similarity."""
    fa = embs_a / (np.linalg.norm(embs_a, axis=1, keepdims=True) + 1e-12)
    fb = embs_b / (np.linalg.norm(embs_b, axis=1, keepdims=True) + 1e-12)
    sim = fa @ fb.T
    hits, total = 0, 0
    for i, lbl in enumerate(labels_a):
        if lbl not in set(labels_b): continue
        true_j = np.where(labels_b == lbl)[0][0]
        if true_j in np.argsort(-sim[i])[:topk]:
            hits += 1
        total += 1
    return hits / total if total else np.nan


# ─── distortion helpers ────────────────────────────────────────────────────────

def apply_jitter(coords, std):
    """Add Gaussian noise of the given std to coordinates."""
    return coords + np.random.randn(*coords.shape).astype(np.float32) * std

def apply_affine(coords, degrees=15, scale_range=(0.85, 1.15)):
    """Apply a random rotation (degrees) and scaling (scale_range) to coordinates."""
    theta = np.random.uniform(-degrees, degrees) / 180 * np.pi
    sx = np.random.uniform(*scale_range)
    sy = np.random.uniform(*scale_range)
    cos, sin = np.cos(theta), np.sin(theta)
    M = np.array([[sx * cos, -sx * sin], [sy * sin, sy * cos]])
    return coords @ M.T

def apply_dropout(coords, labels, p_drop=0.1):
    """Randomly drop cells with probability p_drop, keeping coordinate/label alignment."""
    keep = np.random.rand(len(labels)) > p_drop
    return coords[keep], labels[keep]


# ─── TEST 1: Gap vs distortion strength ────────────────────────────────────────

def test_gap_vs_strength(frames, max_frames=50):
    """Sweep jitter (1-20px) and rotation (1-45°), measure gap."""
    rows = []
    # Baseline (no distortion)
    gaps = []
    for mp, ip in frames[:max_frames]:
        r = load_frame(mp, ip)
        if r is None: continue
        c, lbl, img = r
        if len(lbl) < 2: continue
        patches = extract_patches(img, c)
        emb = compute_dino_embs(patches)
        s = compute_gap(emb, emb, lbl, lbl)
        if s: gaps.append(s["gap"])
    row = {"distortion": "none", "param": 0, "gap": float(np.mean(gaps))}
    rows.append(row)

    # Jitter sweep
    for std in [1, 2, 4, 8, 12, 16, 20]:
        gaps = []
        for mp, ip in frames[:max_frames]:
            r = load_frame(mp, ip)
            if r is None: continue
            c, lbl, img = r
            if len(lbl) < 2: continue
            patches_a = extract_patches(img, c)
            emb_a = compute_dino_embs(patches_a)
            c_b = apply_jitter(c, std)
            patches_b = extract_patches(img, c_b)
            emb_b = compute_dino_embs(patches_b)
            s = compute_gap(emb_a, emb_b, lbl, lbl)
            if s: gaps.append(s["gap"])
        rows.append({"distortion": "jitter", "param": std, "gap": float(np.mean(gaps))})

    # Rotation sweep (affine with scale=1)
    for deg in [5, 10, 15, 25, 35, 45]:
        gaps = []
        for mp, ip in frames[:max_frames]:
            r = load_frame(mp, ip)
            if r is None: continue
            c, lbl, img = r
            if len(lbl) < 2: continue
            patches_a = extract_patches(img, c)
            emb_a = compute_dino_embs(patches_a)
            c_b = apply_affine(c, degrees=deg, scale_range=(0.95, 1.05))
            patches_b = extract_patches(img, c_b)
            emb_b = compute_dino_embs(patches_b)
            s = compute_gap(emb_a, emb_b, lbl, lbl)
            if s: gaps.append(s["gap"])
        rows.append({"distortion": "rotation", "param": deg, "gap": float(np.mean(gaps))})

    return pd.DataFrame(rows)


# ─── TEST 2a: Micro-SSL (same-frame eval) ───────────────────────────────────────

class MicroProjection(nn.Module):
    """Small MLP projection head used for micro-scale SSL training."""

    def __init__(self, in_dim=384, hidden=128, out_dim=64):
        """Build a 2-layer MLP with BatchNorm and L2-normalized output."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        """Project and L2-normalize the input features."""
        return F.normalize(self.net(x), dim=-1)


def nt_xent_loss(z, temperature=0.05):
    """NT-Xent loss over concatenated two-view embeddings (first half = view 1)."""
    B = z.shape[0] // 2
    sim = z @ z.T / temperature
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, targets)


def test_micro_ssl(frames, n_steps=200, lr=1e-3, max_frames=30):
    """Train a tiny projection head with NT-Xent on DINO features.
    Track gap improvement — predicts full-scale convergence."""
    rows = []
    proj = MicroProjection().to(device)
    opt = torch.optim.Adam(proj.parameters(), lr=lr)

    # Precompute DINO features for some frames
    all_feats, all_lbls = [], []
    for mp, ip in frames[:max_frames]:
        r = load_frame(mp, ip)
        if r is None: continue
        c, lbl, img = r
        if len(lbl) < 2: continue
        # View 1: original
        p1 = extract_patches(img, c)
        e1 = compute_dino_embs(p1)
        # View 2: jittered (std=4)
        c2 = apply_jitter(c, 4)
        p2 = extract_patches(img, c2)
        e2 = compute_dino_embs(p2)
        all_feats.append((torch.from_numpy(e1).float(), torch.from_numpy(e2).float()))
        all_lbls.append(lbl)

    if not all_feats:
        return pd.DataFrame(rows)

    for step in range(n_steps):
        losses = []
        proj.train()
        for (e1, e2), lbl in zip(all_feats, all_lbls):
            n = min(len(e1), len(e2))
            if n < 2: continue
            e1, e2 = e1[:n].to(device), e2[:n].to(device)
            z1 = proj(e1)
            z2 = proj(e2)
            z = torch.cat([z1, z2], dim=0)
            loss = nt_xent_loss(z)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        if step % 25 == 0 or step == n_steps - 1:
            proj.eval()
            with torch.no_grad():
                g_vals, intra_vals = [], []
                for (e1, e2), lbl in zip(all_feats, all_lbls):
                    n = min(len(e1), len(e2))
                    if n < 2: continue
                    z1 = proj(e1[:n].to(device)).cpu().numpy()
                    z2 = proj(e2[:n].to(device)).cpu().numpy()
                    lbl_n = lbl[:n]
                    s = compute_gap(z1, z2, lbl_n, lbl_n)
                    if s:
                        g_vals.append(s["gap"])
                        intra_vals.append(s["intra"])
                vgap = float(np.mean(g_vals)) if g_vals else np.nan
                vintra = float(np.mean(intra_vals)) if intra_vals else np.nan
            rows.append({
                "step": step,
                "train_loss": float(np.mean(losses)) if losses else np.nan,
                "val_gap": vgap,
                "val_intra": vintra,
            })

    return pd.DataFrame(rows)


# ─── TEST 2b: Micro-SSL with train/val generalization ──────────────────────────

def test_micro_ssl_generalization(frames, n_steps=200, lr=1e-3, n_train_frames=20, n_val_frames=10):
    """Train on N_train frames, eval on held-out N_val frames.
    If val gap improves too → genuine learning, not memorization."""
    rows = []
    proj = MicroProjection().to(device)
    opt = torch.optim.Adam(proj.parameters(), lr=lr)

    def _precompute(n_frames, offset=0):
        feats, lbls = [], []
        for mp, ip in frames[offset:offset + n_frames]:
            r = load_frame(mp, ip)
            if r is None: continue
            c, lbl, img = r
            if len(lbl) < 2: continue
            p1 = extract_patches(img, c)
            e1 = compute_dino_embs(p1)
            c2 = apply_jitter(c, 4)
            p2 = extract_patches(img, c2)
            e2 = compute_dino_embs(p2)
            feats.append((torch.from_numpy(e1).float(), torch.from_numpy(e2).float()))
            lbls.append(lbl)
        return feats, lbls

    train_feats, train_lbls = _precompute(n_train_frames, 0)
    val_feats, val_lbls = _precompute(n_val_frames, n_train_frames)

    if not train_feats or not val_feats:
        return pd.DataFrame(rows)

    for step in range(n_steps):
        losses = []
        proj.train()
        for (e1, e2), lbl in zip(train_feats, train_lbls):
            n = min(len(e1), len(e2))
            if n < 2: continue
            z = torch.cat([proj(e1[:n].to(device)), proj(e2[:n].to(device))], dim=0)
            loss = nt_xent_loss(z)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        if step % 25 == 0 or step == n_steps - 1:
            proj.eval()
            with torch.no_grad():
                def _eval_gap(feats, lbls):
                    gs = []
                    for (e1, e2), lbl in zip(feats, lbls):
                        n = min(len(e1), len(e2))
                        if n < 2: continue
                        z1 = proj(e1[:n].to(device)).cpu().numpy()
                        z2 = proj(e2[:n].to(device)).cpu().numpy()
                        s = compute_gap(z1, z2, lbl[:n], lbl[:n])
                        if s: gs.append(s["gap"])
                    return float(np.mean(gs)) if gs else np.nan

                train_gap = _eval_gap(train_feats, train_lbls)
                val_gap = _eval_gap(val_feats, val_lbls)
                gap_gap = (train_gap - val_gap) if not (np.isnan(train_gap) or np.isnan(val_gap)) else np.nan

            rows.append({
                "step": step,
                "train_loss": float(np.mean(losses)) if losses else np.nan,
                "train_gap": train_gap,
                "val_gap": val_gap,
                "generalization_gap": gap_gap,
            })

    return pd.DataFrame(rows)


# ─── TEST 2c: Micro-SSL with full distortion pipeline ──────────────────────────

def test_micro_ssl_hard(frames, n_steps=200, lr=1e-3, max_frames=30):
    """Micro-SSL with affine+jitter+dropout combined — hardest test before full training."""
    rows = []
    proj = MicroProjection().to(device)
    opt = torch.optim.Adam(proj.parameters(), lr=lr)

    all_feats, all_lbls = [], []
    for mp, ip in frames[:max_frames]:
        r = load_frame(mp, ip)
        if r is None: continue
        c, lbl, img = r
        if len(lbl) < 2: continue
        # View 1: original
        e1 = compute_dino_embs(extract_patches(img, c))
        # View 2: full pipeline (affine + jitter + dropout)
        c2 = apply_affine(c, degrees=10, scale_range=(0.9, 1.1))
        c2 = apply_jitter(c2, 4)
        c2, lbl2 = apply_dropout(c2, lbl.copy(), p_drop=0.1)
        if len(c2) < 2 or len(lbl2) < 2: continue
        e2 = compute_dino_embs(extract_patches(img, c2))
        all_feats.append((torch.from_numpy(e1).float(), torch.from_numpy(e2).float()))
        all_lbls.append(lbl)

    if not all_feats:
        return pd.DataFrame(rows)

    for step in range(n_steps):
        losses = []
        proj.train()
        for (e1, e2), lbl in zip(all_feats, all_lbls):
            n = min(len(e1), len(e2))
            if n < 2: continue
            z = torch.cat([proj(e1[:n].to(device)), proj(e2[:n].to(device))], dim=0)
            loss = nt_xent_loss(z)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        if step % 25 == 0 or step == n_steps - 1:
            proj.eval()
            with torch.no_grad():
                gs, inter_vals = [], []
                for (e1, e2), lbl in zip(all_feats, all_lbls):
                    n = min(len(e1), len(e2))
                    if n < 2: continue
                    z1 = proj(e1[:n].to(device)).cpu().numpy()
                    z2 = proj(e2[:n].to(device)).cpu().numpy()
                    s = compute_gap(z1, z2, lbl[:n], lbl[:n])
                    if s:
                        gs.append(s["gap"])
                        inter_vals.append(s["inter"])
            rows.append({
                "step": step,
                "train_loss": float(np.mean(losses)) if losses else np.nan,
                "val_gap": float(np.mean(gs)) if gs else np.nan,
                "inter_sim": float(np.mean(inter_vals)) if inter_vals else np.nan,
            })

    return pd.DataFrame(rows)


# ─── TEST 3: Distortion ranking ────────────────────────────────────────────────

def test_distortion_ranking(frames, max_frames=30):
    """Rank distortion types by gap degradation."""
    dist_configs = [
        ("none", lambda c, lbl: (c.copy(), lbl.copy())),
        ("jitter_2px", lambda c, lbl: (apply_jitter(c, 2), lbl.copy())),
        ("jitter_8px", lambda c, lbl: (apply_jitter(c, 8), lbl.copy())),
        ("rotation_10", lambda c, lbl: (apply_affine(c, 10, (0.95, 1.05)), lbl.copy())),
        ("rotation_30", lambda c, lbl: (apply_affine(c, 30, (0.95, 1.05)), lbl.copy())),
        ("affine_mild", lambda c, lbl: (apply_affine(c, 10, (0.9, 1.1)), lbl.copy())),
        ("affine_strong", lambda c, lbl: (apply_affine(c, 25, (0.8, 1.2)), lbl.copy())),
        ("dropout_0.1", lambda c, lbl: apply_dropout(c.copy(), lbl.copy(), 0.1)),
        ("dropout_0.3", lambda c, lbl: apply_dropout(c.copy(), lbl.copy(), 0.3)),
    ]
    rows = []
    for name, fn in dist_configs:
        gaps, recalls = [], []
        for mp, ip in frames[:max_frames]:
            r = load_frame(mp, ip)
            if r is None: continue
            c, lbl, img = r
            if len(lbl) < 2: continue
            e_a = compute_dino_embs(extract_patches(img, c))

            res = fn(c.copy(), lbl.copy())
            if isinstance(res, tuple) and len(res) == 2:
                c_b, lbl_b = res
            else:
                continue
            if len(c_b) < 2 or len(lbl_b) < 2:
                continue
            shared = set(lbl) & set(lbl_b)
            if len(shared) < 2: continue
            e_b = compute_dino_embs(extract_patches(img, c_b))
            s = compute_gap(e_a, e_b, lbl, lbl_b)
            if s: gaps.append(s["gap"])
            rk = compute_recall(e_a, e_b, lbl, lbl_b, topk=1)
            if rk: recalls.append(rk)
        rows.append({
            "distortion": name,
            "gap": float(np.mean(gaps)) if gaps else np.nan,
            "recall1": float(np.mean(recalls)) if recalls else np.nan,
        })
    return pd.DataFrame(rows)


# ─── TEST 4: PCA / effective rank ──────────────────────────────────────────────

def test_pca_analysis(frames, max_frames=50):
    """PCA on pooled DINO features. Effective dimensionality."""
    from sklearn.decomposition import PCA
    all_embs = []
    for mp, ip in frames[:max_frames]:
        r = load_frame(mp, ip)
        if r is None: continue
        c, lbl, img = r
        if len(lbl) < 2: continue
        emb = compute_dino_embs(extract_patches(img, c))
        all_embs.append(emb)
    if len(all_embs) < 2:
        return pd.DataFrame()
    X = np.concatenate(all_embs, axis=0)
    if len(X) < 3:
        return pd.DataFrame()
    Xc = X - X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=min(len(X), 384)).fit(Xc)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumvar, 0.95) + 1)
    rows = []
    for i in range(min(20, len(cumvar))):
        rows.append({"component": i+1, "explained_var": pca.explained_variance_ratio_[i],
                     "cumulative": cumvar[i]})
    return pd.DataFrame(rows), n95


# ─── HTML report ───────────────────────────────────────────────────────────────

def fig_to_b64(fig):
    """Render a matplotlib figure to a base64-encoded PNG string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_report(outdir, df_gap, df_micro, df_rank, df_pca, pca_n95, interpret):
    """Write the convergence-prediction HTML report with all figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    figs = []

    # Fig 1: Gap vs strength
    if not df_gap.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for dist in df_gap["distortion"].unique():
            sub = df_gap[df_gap["distortion"] == dist]
            ax.plot(sub["param"], sub["gap"], "o-", label=dist, linewidth=2)
        ax.axhline(0.15, color="gray", ls="--", alpha=0.5, label="borderline")
        ax.axhline(0.3, color="green", ls=":", alpha=0.5, label="strong signal")
        ax.set_xlabel("Distortion parameter")
        ax.set_ylabel("Separation gap")
        ax.set_title("Test 1: Gap vs Distortion Strength")
        ax.legend()
        figs.append(("Gap vs Distortion Strength", fig))
        plt.close(fig)

    # Fig 2: Micro-SSL convergence
    if not df_micro.empty:
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(df_micro["step"], df_micro["train_loss"], "b-", label="Train loss")
        ax1.set_ylabel("NT-Xent Loss", color="b")
        ax1.tick_params(axis="y", labelcolor="b")
        ax2 = ax1.twinx()
        ax2.plot(df_micro["step"], df_micro["val_gap"], "r-o", label="Val gap", linewidth=2)
        ax2.axhline(0.15, color="gray", ls="--", alpha=0.4)
        ax2.axhline(0.3, color="green", ls=":", alpha=0.4)
        ax2.set_ylabel("Separation gap", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        ax1.set_xlabel("Training step")
        ax1.set_title("Test 2: Micro-SSL Convergence (projection head only)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        figs.append(("Micro-SSL Convergence", fig))
        plt.close(fig)

    # Fig 3: Distortion ranking
    if not df_rank.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2ecc71" if r > 0.15 else ("#f39c12" if r > 0.05 else "#e74c3c")
                  for r in df_rank["gap"]]
        bars = ax.barh(range(len(df_rank)), df_rank["gap"], color=colors, alpha=0.8)
        ax.set_yticks(range(len(df_rank)))
        ax.set_yticklabels(df_rank["distortion"])
        ax.axvline(0.15, color="gray", ls="--", alpha=0.5, label="borderline")
        ax.set_xlabel("Separation gap")
        ax.set_title("Test 3: Distortion Difficulty Ranking")
        ax.legend()
        for bar, g in zip(bars, df_rank["gap"]):
            if not np.isnan(g):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                        f"{g:.4f}", va="center", fontsize=9)
        figs.append(("Distortion Ranking", fig))
        plt.close(fig)

    # Fig 4: PCA
    if not df_pca.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(df_pca["component"], df_pca["explained_var"], alpha=0.6, label="Per component")
        ax.plot(df_pca["component"], df_pca["cumulative"], "r-o", linewidth=2, label="Cumulative")
        ax.axhline(0.95, color="gray", ls="--", alpha=0.5, label="95% threshold")
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Explained variance ratio")
        ax.set_title(f"Test 4: DINO Embedding Effective Rank (n<0.95={pca_n95})")
        ax.legend()
        figs.append(("PCA Effective Rank", fig))
        plt.close(fig)

    # Assemble HTML
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>SSL Convergence Prediction</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:0 auto;padding:20px;background:#fafafa}",
        "h1{color:#333} h2{color:#555;border-bottom:2px solid #ddd;padding-bottom:5px}",
        "img{max-width:100%;margin:15px 0;border:1px solid #ddd;border-radius:6px}",
        ".pre{font-family:monospace;white-space:pre;background:#f8f8f8;padding:15px;border-radius:5px;overflow-x:auto;font-size:0.9em;line-height:1.5}",
        "</style></head><body>",
        "<h1>DINO + Contrastive SSL: Convergence Prediction</h1>",
        f"<p>Generated: {time.strftime('%Y-%m-%d %H:%M')}</p>",
        f"<div class='pre'>{interpret}</div>",
    ]
    for title, fig in figs:
        b64 = fig_to_b64(fig)
        html.append(f"<figure><figcaption><b>{title}</b></figcaption>"
                    f'<img src="data:image/png;base64,{b64}" /></figure>')
    html.append("</body></html>")

    with open(outdir / "convergence_prediction.html", "w") as f:
        f.write("\n".join(html))
    logger.info(f"Report: {outdir / 'convergence_prediction.html'}")


# ─── interpret ─────────────────────────────────────────────────────────────────

def interpret_results(micro_improvement, ranking_df, gap_df, pca_n95):
    """Build the textual convergence verdict from the micro-test results."""
    lines = ["=" * 65,
             "SSL CONVERGENCE PREDICTION",
             "=" * 65, ""]
    votes = []

    if micro_improvement > 0.05:
        votes.append(f"✓ Micro-SSL improved gap by {micro_improvement:.3f} — projection head learns")
    elif micro_improvement > 0.02:
        votes.append(f"~ Micro-SSL improved gap by {micro_improvement:.3f} — marginal")
    else:
        votes.append(f"✗ Micro-SSL gap change: {micro_improvement:.3f} — not learning")

    if not gap_df.empty:
        base = gap_df[gap_df["distortion"] == "none"]["gap"].values
        if len(base):
            lines.append(f"  Baseline (no distortion): gap={base[0]:.4f}")

    if pca_n95:
        lines.append(f"  DINO effective rank (95% var): {pca_n95}")
        if pca_n95 >= 10:
            votes.append("✓ High effective dimensionality — ample signal")
        elif pca_n95 >= 5:
            votes.append("~ Moderate effective dimensionality")
        else:
            votes.append("✗ Low effective dimensionality — risk of collapse")

    gap_at_jitter8 = None
    if not gap_df.empty:
        for _, r in gap_df[gap_df["distortion"] == "jitter"].iterrows():
            if r["param"] == 8:
                gap_at_jitter8 = r["gap"]
                if gap_at_jitter8 > 0.15:
                    votes.append(f"✓ Jitter8 gap={gap_at_jitter8:.3f} > 0.15 — robust to motion")
                else:
                    votes.append(f"~ Jitter8 gap={gap_at_jitter8:.3f} — sensitive to motion")

    # Strong yes conditions
    strong_yes = all([
        micro_improvement > 0.03,
        pca_n95 and pca_n95 >= 5,
        gap_at_jitter8 and gap_at_jitter8 > 0.1,
    ])
    if strong_yes:
        verdict = "CONVERGENCE LIKELY — DINO + NT-Xent should work on vanvliet. Proceed to full training."
    elif micro_improvement > 0:
        verdict = "CONVERGENCE POSSIBLE — Micro-SSL shows improvement. May need more epochs or lower temperature."
    else:
        verdict = "CONVERGENCE UNCLEAR — Consider deeper projection head or image patch augmentation."

    lines.extend(votes)
    lines.append(f"\n  VERDICT: {verdict}")
    lines.append("=" * 65)
    return "\n".join(lines)


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    """Run the micro-SSL convergence prediction tests and write CSVs + report."""
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="../data/vanvliet")
    p.add_argument("--conditions", default="rpsM,recA,pheA,metA,cib,trpL")
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--outdir", default="runs/convergence_prediction")
    p.add_argument("--test", default="all",
                   choices=["all", "gap_strength", "micro_ssl", "generalization", "hard", "ranking", "pca"])
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    conditions = [c.strip() for c in args.conditions.split(",")]
    frames = scan_frames(args.data_root, conditions, max_frames=args.max_frames)
    logger.info(f"Frames: {len(frames)}")

    results = {}
    tests_to_run = ["gap_strength", "micro_ssl", "generalization", "hard", "ranking", "pca"] if args.test == "all" else [args.test]

    if "gap_strength" in tests_to_run:
        logger.info("Test: Gap vs distortion strength...")
        df_gap = test_gap_vs_strength(frames, max_frames=50)
        df_gap.to_csv(outdir / "gap_vs_strength.csv", index=False)
        results["gap_df"] = df_gap
        if not df_gap.empty:
            base = df_gap[df_gap["distortion"] == "none"]["gap"].values
            if len(base):
                logger.info(f"  Baseline gap: {base[0]:.4f}")

    if "micro_ssl" in tests_to_run:
        logger.info("Test: Micro-SSL (same-frame eval)...")
        df_micro = test_micro_ssl(frames, n_steps=200, max_frames=30)
        df_micro.to_csv(outdir / "micro_ssl.csv", index=False)
        results["df_micro"] = df_micro
        if len(df_micro) > 1:
            improvement = df_micro["val_gap"].iloc[-1] - df_micro["val_gap"].iloc[0]
            logger.info(f"  Gap improvement (same-frame): {improvement:.4f}")

    if "generalization" in tests_to_run:
        logger.info("Test: Micro-SSL generalization (train/val split)...")
        df_gen = test_micro_ssl_generalization(frames, n_steps=200, n_train_frames=20, n_val_frames=10)
        df_gen.to_csv(outdir / "micro_ssl_generalization.csv", index=False)
        results["df_gen"] = df_gen
        if len(df_gen) > 1:
            train_last = df_gen["train_gap"].iloc[-1]
            val_last = df_gen["val_gap"].iloc[-1]
            train_first = df_gen["train_gap"].iloc[0]
            val_first = df_gen["val_gap"].iloc[0]
            logger.info(f"  Train gap: {train_first:.4f} → {train_last:.4f}")
            logger.info(f"  Val gap:   {val_first:.4f} → {val_last:.4f}")
            if not np.isnan(val_last - val_first):
                if val_last - val_first > 0.05:
                    logger.info(f"  ✓ GENERALIZES — val gap improved {val_last - val_first:.3f}")
                elif val_last - val_first > 0:
                    logger.info(f"  ~ Val gap improved marginally ({val_last - val_first:.3f})")
                else:
                    logger.info(f"  ✗ DOES NOT GENERALIZE — val gap dropped {val_first - val_last:.3f}")

    if "hard" in tests_to_run:
        logger.info("Test: Micro-SSL hard (affine+jitter+dropout)...")
        df_hard = test_micro_ssl_hard(frames, n_steps=200, max_frames=30)
        df_hard.to_csv(outdir / "micro_ssl_hard.csv", index=False)
        results["df_hard"] = df_hard
        if len(df_hard) > 1:
            imp = df_hard["val_gap"].iloc[-1] - df_hard["val_gap"].iloc[0]
            inter = df_hard["inter_sim"].iloc[-1]
            logger.info(f"  Gap improvement (hard): {imp:.4f}  final inter_sim={inter:.4f}")
            if inter > 0.8:
                logger.warning("  ⚠ High inter_sim — collapse risk")

    if "ranking" in tests_to_run:
        logger.info("Test: Distortion ranking...")
        df_rank = test_distortion_ranking(frames, max_frames=30)
        df_rank.to_csv(outdir / "distortion_ranking.csv", index=False)
        results["df_rank"] = df_rank

    if "pca" in tests_to_run:
        logger.info("Test: PCA analysis...")
        df_pca, pca_n95 = test_pca_analysis(frames, max_frames=50)
        if not df_pca.empty:
            df_pca.to_csv(outdir / "pca_analysis.csv", index=False)
        results["df_pca"] = df_pca
        results["pca_n95"] = pca_n95
        logger.info(f"  Effective rank: {pca_n95}")

    # Interpret only if we have key tests
    if "micro_ssl" in tests_to_run:
        improvement = (df_micro["val_gap"].iloc[-1] - df_micro["val_gap"].iloc[0]
                       if "df_micro" in results and len(results["df_micro"]) > 1 else 0)
    else:
        improvement = 0

    gap_df = results.get("gap_df", pd.DataFrame())
    rank_df = results.get("df_rank", pd.DataFrame())
    pca_n95 = results.get("pca_n95", None)

    if "generalization" in tests_to_run and "df_gen" in results:
        df_gen = results["df_gen"]
        if len(df_gen) > 1:
            vf, vl = df_gen["val_gap"].iloc[0], df_gen["val_gap"].iloc[-1]
            print(f"\n  GENERALIZATION: val_gap {vf:.4f} → {vl:.4f} (Δ={vl - vf:+.4f})")
            print(f"  {'✓ LEARNS' if vl > vf + 0.03 else '~ BORDERLINE' if vl > vf else '✗ MEMORIZES'}")
    if "hard" in tests_to_run and "df_hard" in results:
        df_hard = results["df_hard"]
        if len(df_hard) > 1:
            hf, hl = df_hard["val_gap"].iloc[0], df_hard["val_gap"].iloc[-1]
            hi = df_hard["inter_sim"].iloc[-1]
            print(f"\n  HARD PIPELINE: gap {hf:.4f} → {hl:.4f} (Δ={hl - hf:+.4f})  inter={hi:.4f}")

    interpret = interpret_results(improvement, rank_df, gap_df, pca_n95)
    print(interpret)

    generate_report(outdir, gap_df, results.get("df_micro", pd.DataFrame()),
                    rank_df, results.get("df_pca", pd.DataFrame()), pca_n95, interpret)
    logger.info(f"All results in {outdir}")


if __name__ == "__main__":
    main()
