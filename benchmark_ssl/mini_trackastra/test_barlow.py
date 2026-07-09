#!/usr/bin/env python3
"""Compare Barlow Twins vs NT-Xent loss for SSL pretraining.

Mode B only (NoPE + DINO): trains two ASCENT encoders with identical
architecture, data, and seed — one with NT-Xent, one with Barlow Twins.

Measures:
  1. Gap (intra − inter cosine similarity) — higher = better separation
  2. Effective rank (PCA 95% variance) — higher = more spread out
  3. Inter-cell cosine sim — lower = more discriminative
  4. Training loss — convergence behavior

Usage:
    cd benchmark_ssl/mini_trackastra
    ../.venv/bin/python test_barlow.py
    ../.venv/bin/python test_barlow.py --steps 10 --max-frames 5
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import regionprops_table
from tifffile import imread

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("barlow_vs_ntxent")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── DINO backbone ─────────────────────────────────────────────────────────────

logger.info("Loading DINOv2 vits14 backbone...")
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
DINO_DIM = 384
PATCH_SIZE = 64


def extract_patches(img, centroids):
    """Extract 64×64 patches around each centroid."""
    h, w = img.shape[-2:]
    half = PATCH_SIZE // 2
    patches = []
    for cy, cx in centroids:
        cy_i, cx_i = int(round(float(cy))), int(round(float(cx)))
        cy_i, cx_i = np.clip(cy_i, 0, h - 1), np.clip(cx_i, 0, w - 1)
        y1, x1 = cy_i - half, cx_i - half
        y2, x2 = cy_i + half, cx_i + half
        if y2 <= 0 or x2 <= 0:
            patches.append(np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32))
            continue
        pt, pb = max(0, -y1), max(0, y2 - h)
        pl, pr = max(0, -x1), max(0, x2 - w)
        y1c, y2c = max(0, y1), min(h, y2)
        x1c, x2c = max(0, x1), min(w, x2)
        if y2c <= y1c or x2c <= x1c:
            patches.append(np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32))
            continue
        crop = img[y1c:y2c, x1c:x2c]
        if pt or pb or pl or pr:
            crop = np.pad(crop, ((pt, pb), (pl, pr)), mode="reflect")
        if crop.shape != (PATCH_SIZE, PATCH_SIZE):
            crop = np.pad(
                crop,
                ((0, max(0, PATCH_SIZE - crop.shape[0])),
                 (0, max(0, PATCH_SIZE - crop.shape[1]))),
                mode="reflect",
            )[:PATCH_SIZE, :PATCH_SIZE]
        patches.append(crop)
    if patches:
        return np.stack(patches).astype(np.float32)
    return np.zeros((0, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)


@torch.no_grad()
def compute_dino_embs(patches_np):
    """Compute DINOv2 embeddings for a batch of patches."""
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


# ─── Free DINO from GPU ──────────────────────────────────────────────────────

def free_dino():
    """Free DINO backbone from GPU after precomputation."""
    global dino
    dino = dino.cpu()
    torch.cuda.empty_cache()
    logger.info("DINO freed from GPU memory")


# ─── Data loading ──────────────────────────────────────────────────────────────

def load_frame(mask_path, img_path):
    """Load a single frame: mask -> regionprops -> centroids + labels + image."""
    mask = imread(mask_path)
    img = imread(img_path).astype(np.float32)
    p1, p998 = np.percentile(img, (1, 99.8))
    img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)
    props = regionprops_table(mask, properties=("label", "centroid"))
    if not props or len(props["label"]) < 2:
        return None
    coords = np.stack(
        [props["centroid-0"], props["centroid-1"]], axis=-1
    ).astype(np.float32)
    return coords, props["label"].astype(np.int32), img


def scan_frames(data_root, conditions, max_frames):
    """Scan data directory for mask/image frame pairs."""
    dr = Path(data_root)
    frames = []
    for cond in conditions:
        for exp in sorted((dr / cond).iterdir()):
            if not exp.is_dir():
                continue
            tra, img_dir = exp / "TRA", exp / "img"
            if not tra.exists() or not img_dir.exists():
                continue
            for m in sorted(tra.glob("man_track*.tif")):
                stem = m.stem.replace("man_track", "")
                try:
                    fi = int(stem)
                except ValueError:
                    continue
                ip = img_dir / f"t{fi:06d}.tif"
                if ip.exists():
                    frames.append((str(m), str(ip)))
                    if len(frames) >= max_frames:
                        return frames
    return frames


# ─── Distortions (SSL view generation) ───────────────────────────────────────

def apply_jitter(c, std):
    """Add Gaussian jitter to coordinates."""
    return c + np.random.randn(*c.shape).astype(np.float32) * std


def distort(coords, labels, std=4):
    """Create second view: jitter coordinates by std=4."""
    c = apply_jitter(coords.copy(), std)
    return c, labels.copy()


# ─── Positional encoding ──────────────────────────────────────────────────────

class NoPE(nn.Module):
    """Learned constant positional encoding (no coordinate info)."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    def forward(self, coords):
        return self.token.expand(coords.shape[0], coords.shape[1], -1)


# ─── Encoder ────────────────────────────────────────────────────────────────────

class ASCENTEncoder(nn.Module):
    """ASCENT-style encoder: DINO + PE -> project -> add -> LayerNorm -> MLP -> embedding.

    Same architecture used in diagnose_coord_shortcut.py Mode B.
    """

    def __init__(self, pe_dim, dino_dim=384, d_model=256, out_dim=64, normalize=True):
        super().__init__()
        self.normalize_output = normalize
        self.pe_proj = nn.Linear(pe_dim, d_model)
        self.dino_proj = nn.Linear(dino_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, dino_feats, pe):
        p = F.normalize(self.pe_proj(pe), dim=-1)
        d = F.normalize(self.dino_proj(dino_feats), dim=-1)
        x = self.norm(p + d)
        x = self.mlp(x)
        if self.normalize_output:
            x = F.normalize(x, dim=-1)
        return x


# ─── Loss functions ─────────────────────────────────────────────────────────────

def barlow_twins_loss(z1, z2, lambd=0.0051):
    """Barlow Twins redundancy-reduction loss.

    Args:
        z1, z2: (N, D) — per-cell embeddings for two views.
        lambd: weight for off-diagonal decorrelation term.

    Returns:
        Scalar loss.
    """
    N, D = z1.shape
    # Center and normalize each dimension across the batch
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)

    # Cross-correlation matrix (D, D)
    c = z1.T @ z2 / N

    # On-diagonal: push to 1 (invariance)
    on_diag = (torch.diagonal(c) - 1).pow(2).sum()

    # Off-diagonal: push to 0 (decorrelation)
    off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=c.device)
    off_diag = c[off_diag_mask].pow(2).sum()

    return on_diag + lambd * off_diag


def nt_xent_loss(z, temperature=0.05):
    """NT-Xent contrastive loss for paired views.

    Args:
        z: (2*N, D) where first N = view1, last N = view2 in corresponding order.
        temperature: softmax temperature.

    Returns:
        Scalar loss.
    """
    B = z.shape[0] // 2
    sim = z @ z.T / temperature
    sim.fill_diagonal_(-1e9)
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, labels)


# ─── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(z1, z2, labels1, labels2):
    """Compute gap, intra, inter cosine similarity metrics.

    Args:
        z1, z2: (N, D) numpy arrays of embeddings for view 1 and view 2.
        labels1, labels2: (N,) integer labels.

    Returns:
        dict: gap, intra, inter, intra_frame.
    """
    z1n = z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + 1e-12)
    z2n = z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + 1e-12)
    sim = z1n @ z2n.T
    intra, inter = [], []
    for i, lbl in enumerate(labels1):
        matches = set(np.where(labels2 == lbl)[0])
        for j in range(len(labels2)):
            (intra if j in matches else inter).append(sim[i, j])
    # Intra-frame: cosine sim between different cells in same view
    intra_f = [float((z1n[i] * z1n[j]).sum())
               for i in range(len(z1n))
               for j in range(i + 1, len(z1n))]
    return {
        "intra": float(np.mean(intra)) if intra else np.nan,
        "inter": float(np.mean(inter)) if inter else np.nan,
        "gap": float(np.mean(intra) - np.mean(inter)) if intra and inter else np.nan,
        "intra_frame": float(np.mean(intra_f)) if intra_f else np.nan,
    }


def effective_rank(embeddings, variance_threshold=0.95):
    """Number of PCA components needed to explain given fraction of variance.

    Higher = more spread out / higher-dimensional representation.
    """
    if len(embeddings) < 2:
        return 1
    e = embeddings - embeddings.mean(axis=0, keepdims=True)
    # SVD on centered embeddings
    U, S, Vt = np.linalg.svd(e, full_matrices=False)
    var_exp = S ** 2 / (S ** 2).sum()
    cumvar = np.cumsum(var_exp)
    return int(np.searchsorted(cumvar, variance_threshold) + 1)


# ─── Precompute data ───────────────────────────────────────────────────────────

class PrecomputedData:
    """Holds precomputed (dino_e1, dino_e2, coords1, coords2, labels) for frames."""

    def __init__(self, frames, jitter_std=4, val_split=0.2):
        all_items = []
        for mp, ip in frames:
            r = load_frame(mp, ip)
            if r is None:
                continue
            c1, lbl, img = r
            c2, _ = distort(c1.copy(), lbl.copy(), std=jitter_std)
            p1 = extract_patches(img, c1)
            e1 = compute_dino_embs(p1)
            p2 = extract_patches(img, c2)
            e2 = compute_dino_embs(p2)
            all_items.append((e1, e2, c1, c2, lbl))

        n_val = max(1, int(len(all_items) * val_split))
        self.train = all_items[:-n_val]
        self.val = all_items[-n_val:]
        logger.info(f"Data: {len(self.train)} train + {len(self.val)} val frames")

    def to_gpu(self, subset):
        gpu = []
        for e1, e2, c1, c2, lbl in subset:
            gpu.append((
                torch.from_numpy(e1).float().to(device),
                torch.from_numpy(e2).float().to(device),
                torch.from_numpy(c1).float().to(device),
                torch.from_numpy(c2).float().to(device),
                lbl,
            ))
        return gpu


# ─── Training ───────────────────────────────────────────────────────────────────

def train_encoder(name, loss_fn, pos_enc, data, steps, lr, d_model, out_dim,
                  eval_every=50, is_barlow=False, normalize=True):
    """Train an ASCENTEncoder with a given loss function.

    Args:
        name: label for logging ("BT" or "NTX").
        loss_fn: callable that accepts either (z) for NT-Xent or (z1, z2) for BT.
        pos_enc: positional encoding module.
        data: PrecomputedData instance.
        steps: number of training steps.
        lr: learning rate.
        d_model: projection dimension.
        out_dim: output embedding dimension.
        eval_every: log metrics every N steps.
        is_barlow: if True, loss_fn takes (z1, z2); else it takes (z) where
                   z = cat([z1, z2]).

    Returns:
        pd.DataFrame of per-eval-step metrics.
        encoder: trained ASCENTEncoder.
    """
    pe_dim = pos_enc.d
    encoder = ASCENTEncoder(pe_dim, DINO_DIM, d_model, out_dim=out_dim, normalize=normalize).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)

    train_gpu = data.to_gpu(data.train)
    val_gpu = data.to_gpu(data.val)

    rows = []

    for step in range(steps):
        encoder.train()
        losses = []
        for de1, de2, c1, c2, lbl in train_gpu:
            n = min(len(de1), len(de2))
            if n < 2:
                continue
            # Add batch dim for PE
            pe1 = pos_enc(c1.unsqueeze(0)).squeeze(0)
            pe2 = pos_enc(c2.unsqueeze(0)).squeeze(0)
            z1 = encoder(de1[:n], pe1[:n])
            z2 = encoder(de2[:n], pe2[:n])

            if is_barlow:
                loss = loss_fn(z1, z2)
            else:
                z = torch.cat([z1, z2])
                loss = loss_fn(z)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else np.nan

        if (step + 1) % eval_every == 0 or step == steps - 1 or step == 0:
            with torch.no_grad():
                encoder.eval()

                # Evaluate on training data
                tr_gaps, tr_intras, tr_inters, tr_effranks = [], [], [], []
                for de1, de2, c1, c2, lbl in train_gpu:
                    n = min(len(de1), len(de2))
                    if n < 2:
                        continue
                    pe1 = pos_enc(c1.unsqueeze(0)).squeeze(0)
                    pe2 = pos_enc(c2.unsqueeze(0)).squeeze(0)
                    z1_np = encoder(de1[:n], pe1[:n]).cpu().numpy()
                    z2_np = encoder(de2[:n], pe2[:n]).cpu().numpy()
                    m = compute_metrics(z1_np, z2_np, lbl[:n], lbl[:n])
                    if not np.isnan(m["gap"]):
                        tr_gaps.append(m["gap"])
                        tr_intras.append(m["intra"])
                        tr_inters.append(m["inter"])
                    # Effective rank on concatenated embeddings from both views
                    z_all = np.concatenate([z1_np, z2_np], axis=0)
                    tr_effranks.append(effective_rank(z_all))

                # Evaluate on validation data
                vl_gaps, vl_intras, vl_inters, vl_effranks = [], [], [], []
                for de1, de2, c1, c2, lbl in val_gpu:
                    n = min(len(de1), len(de2))
                    if n < 2:
                        continue
                    pe1 = pos_enc(c1.unsqueeze(0)).squeeze(0)
                    pe2 = pos_enc(c2.unsqueeze(0)).squeeze(0)
                    z1_np = encoder(de1[:n], pe1[:n]).cpu().numpy()
                    z2_np = encoder(de2[:n], pe2[:n]).cpu().numpy()
                    m = compute_metrics(z1_np, z2_np, lbl[:n], lbl[:n])
                    if not np.isnan(m["gap"]):
                        vl_gaps.append(m["gap"])
                        vl_intras.append(m["intra"])
                        vl_inters.append(m["inter"])
                    z_all = np.concatenate([z1_np, z2_np], axis=0)
                    vl_effranks.append(effective_rank(z_all))

                row = {
                    "step": step,
                    "loss": avg_loss,
                    "train_gap": np.mean(tr_gaps) if tr_gaps else np.nan,
                    "train_intra": np.mean(tr_intras) if tr_intras else np.nan,
                    "train_inter": np.mean(tr_inters) if tr_inters else np.nan,
                    "train_eff_rank": np.mean(tr_effranks) if tr_effranks else np.nan,
                    "val_gap": np.mean(vl_gaps) if vl_gaps else np.nan,
                    "val_intra": np.mean(vl_intras) if vl_intras else np.nan,
                    "val_inter": np.mean(vl_inters) if vl_inters else np.nan,
                    "val_eff_rank": np.mean(vl_effranks) if vl_effranks else np.nan,
                }
                rows.append(row)

                logger.info(
                    f"  [{name}] step {step:4d}: loss={avg_loss:.4f}  "
                    f"gap={row['train_gap']:.4f}  "
                    f"inter={row['train_inter']:.4f}  "
                    f"eff_rank={row['train_eff_rank']:.0f}"
                )

    return rows, encoder


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Compare Barlow Twins vs NT-Xent for SSL pretraining"
    )
    p.add_argument("--data-root", default="../../data/vanvliet")
    p.add_argument("--conditions", default="rpsM")
    p.add_argument("--max-frames", type=int, default=30)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=256,
                    help="Projection dimension (d_model)")
    p.add_argument("--out-dim", type=int, default=64,
                    help="Output embedding dimension")
    p.add_argument("--pe-dim", type=int, default=128,
                    help="Positional encoding dimension")
    p.add_argument("--jitter-std", type=float, default=4.0,
                    help="Jitter std for second view creation")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--bt-lambd", type=float, default=0.0051,
                    help="Barlow Twins off-diagonal weight")
    p.add_argument("--ntx-temp", type=float, default=0.05,
                    help="NT-Xent temperature")
    args = p.parse_args()

    # Re-seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ─── Load data ──────────────────────────────────────────────────────────
    conditions = [c.strip() for c in args.conditions.split(",")]
    frames = scan_frames(args.data_root, conditions, args.max_frames)
    logger.info(f"Loaded {len(frames)} frames from {conditions}")

    if len(frames) < 4:
        logger.error("Need at least 4 frames for train/val split.")
        sys.exit(1)

    data = PrecomputedData(frames, jitter_std=args.jitter_std, val_split=0.2)

    # ─── Free DINO from GPU ───────────────────────────────────────────────
    free_dino()

    # ─── Build common positional encoding ─────────────────────────────────
    pe = NoPE(args.pe_dim).to(device)
    logger.info(f"NoPE dim: {args.pe_dim}")

    # ─── Train both encoders ──────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Training NT-Xent encoder (Mode B — contrastive)")
    logger.info("=" * 60)

    # Reset seeds before each training so both start with same initial weights
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ntx_rows, ntx_encoder = train_encoder(
        "NTX",
        lambda z: nt_xent_loss(z, temperature=args.ntx_temp),
        pe, data, args.steps, args.lr, args.d_model, args.out_dim,
        eval_every=args.eval_every, is_barlow=False, normalize=True,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Training Barlow Twins encoder (Mode B — decorrelation)")
    logger.info("=" * 60)

    # Reset seeds again for fair comparison
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    bt_rows, bt_encoder = train_encoder(
        "BT",
        lambda z1, z2: barlow_twins_loss(z1, z2, lambd=args.bt_lambd),
        pe, data, args.steps, args.lr, args.d_model, args.out_dim,
        eval_every=args.eval_every, is_barlow=True, normalize=False,
    )

    # ─── Final comparison ──────────────────────────────────────────────────
    ntx_final = ntx_rows[-1] if ntx_rows else {}
    bt_final = bt_rows[-1] if bt_rows else {}

    print("\n" + "=" * 65)
    print("BARLOW TWINS vs NT-XENT — FINAL COMPARISON")
    print("=" * 65)
    print(f"")
    print(f"Setup: {args.steps} steps, {args.max_frames} frames, "
          f"d_model={args.d_model}, out_dim={args.out_dim}")
    print(f"")

    # Training loss
    print(f"Training Loss:")
    print(f"  BT:  step {bt_final.get('step', 0):4d} loss={bt_final.get('loss', np.nan):.4f}")
    print(f"  NTX: step {ntx_final.get('step', 0):4d} loss={ntx_final.get('loss', np.nan):.4f}")
    print(f"")

    # Gap (intra - inter)
    print(f"Gap (intra − inter, higher = better):")
    print(f"  BT:  train_gap={bt_final.get('train_gap', np.nan):.4f}  "
          f"val_gap={bt_final.get('val_gap', np.nan):.4f}")
    print(f"  NTX: train_gap={ntx_final.get('train_gap', np.nan):.4f}  "
          f"val_gap={ntx_final.get('val_gap', np.nan):.4f}")
    print(f"")

    # Inter-cell cosine sim
    print(f"Inter-cell cosine sim (lower = more discriminative):")
    print(f"  BT:  train_inter={bt_final.get('train_inter', np.nan):.4f}  "
          f"val_inter={bt_final.get('val_inter', np.nan):.4f}")
    print(f"  NTX: train_inter={ntx_final.get('train_inter', np.nan):.4f}  "
          f"val_inter={ntx_final.get('val_inter', np.nan):.4f}")
    print(f"")

    # Effective rank
    print(f"Effective rank (PCA 95% var, higher = more spread out):")
    print(f"  BT:  train_eff_rank={bt_final.get('train_eff_rank', np.nan):.0f}  "
          f"val_eff_rank={bt_final.get('val_eff_rank', np.nan):.0f}")
    print(f"  NTX: train_eff_rank={ntx_final.get('train_eff_rank', np.nan):.0f}  "
          f"val_eff_rank={ntx_final.get('val_eff_rank', np.nan):.0f}")
    print(f"")

    # ─── Conclusion ────────────────────────────────────────────────────────
    print("─" * 65)
    print("CONCLUSION")
    print("─" * 65)

    bt_eff = bt_final.get('val_eff_rank', 0) or 0
    ntx_eff = ntx_final.get('val_eff_rank', 0) or 0
    bt_gap = bt_final.get('val_gap', np.nan)
    ntx_gap = ntx_final.get('val_gap', np.nan)
    bt_inter = bt_final.get('val_inter', np.nan)
    ntx_inter = ntx_final.get('val_inter', np.nan)

    eff_winner = "BT" if bt_eff > ntx_eff else "NTX" if ntx_eff > bt_eff else "TIE"
    gap_winner = "BT" if bt_gap > ntx_gap else "NTX" if ntx_gap > bt_gap else "TIE"
    inter_winner = "BT" if bt_inter < ntx_inter else "NTX" if ntx_inter < bt_inter else "TIE"

    print(f"")
    print(f"  Effective rank:     BT={bt_eff:.0f} vs NTX={ntx_eff:.0f}  → {eff_winner} wins")
    print(f"  Gap (val):          BT={bt_gap:.4f} vs NTX={ntx_gap:.4f}  → {gap_winner} wins")
    print(f"  Inter (val, lower): BT={bt_inter:.4f} vs NTX={ntx_inter:.4f}  → {inter_winner} wins")

    # For cross-attention, we want high effective rank (spread out) AND
    # low inter-cell similarity (discriminative).
    if bt_eff > ntx_eff and bt_inter < ntx_inter:
        print(f"")
        print(f"  *** Barlow Twins produces more spread-out, cross-attention-friendly embeddings. ***")
        print(f"      Higher effective rank ({bt_eff:.0f} vs {ntx_eff:.0f}) means the representation")
        print(f"      occupies more dimensions — better for dot-product attention.")
        print(f"      Lower inter-cell similarity ({bt_inter:.4f} vs {ntx_inter:.4f}) means")
        print(f"      cells are more discriminable.")
    elif ntx_eff > bt_eff and ntx_inter < bt_inter:
        print(f"")
        print(f"  *** NT-Xent produces more spread-out, cross-attention-friendly embeddings. ***")
    else:
        print(f"")
        print(f"  ~ Mixed signals. Check effective rank vs inter for your specific use case.")

    print(f"")
    print(f"Full metrics logged every {args.eval_every} steps above.")
    print("=" * 65)


if __name__ == "__main__":
    main()
