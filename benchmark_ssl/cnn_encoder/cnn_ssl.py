#!/usr/bin/env python3
"""NT-Xent pretraining of ScaledCNN on distorted bacteria patches.

Usage:
    python cnn_ssl.py --scale small --steps 20 --max-frames 5                  # quick test
    python cnn_ssl.py --scale large --steps 1000 --data-dirs "../../data/vanvliet" --distortion-strength strong  # full run
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
logger = logging.getLogger("cnn_ssl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

PATCH_SIZE = 64


# ═══════════════════════════════════════════════════════════════════════════════
#  Architecture
# ═══════════════════════════════════════════════════════════════════════════════

class ScaledCNN(nn.Module):
    """ConvNet for 64×64 grayscale cell patches. Output: 128-dim embedding.

    Args:
        scale: 'small' (~20K params), 'medium' (~100K params), 'large' (~200K params)
        out_dim: embedding dimension (default 128)
    """
    def __init__(self, scale='large', out_dim=128):
        super().__init__()
        if scale == 'small':
            ch = [8, 16, 32]           # 3 conv layers
        elif scale == 'medium':
            ch = [16, 32, 64]          # 3 conv layers
        else:  # large
            ch = [32, 64, 128, 256]    # 4 conv layers

        layers = []
        in_ch = 1
        for c in ch:
            layers += [nn.Conv2d(in_ch, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = c
        self.conv = nn.Sequential(*layers)
        # After pooling: small/medium: 64 → 32 → 16 → 8,  large: 64 → 32 → 16 → 8 → 4
        spatial = PATCH_SIZE // (2 ** len(ch))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch[-1] * spatial * spatial, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, patches):
        """patches: (N, 1, 64, 64) → embeddings: (N, out_dim)"""
        return self.fc(self.conv(patches))


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading & patch extraction
# ═══════════════════════════════════════════════════════════════════════════════

def load_frame(mask_path, img_path):
    """Load a single mask+image frame. Returns (coords, labels, img) or None."""
    mask = imread(mask_path)
    img = imread(img_path).astype(np.float32)
    p1, p998 = np.percentile(img, (1, 99.8))
    img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)
    props = regionprops_table(mask, properties=("label", "centroid"))
    if not props or len(props["label"]) < 2:
        return None
    coords = np.stack([props["centroid-0"], props["centroid-1"]], axis=-1).astype(np.float32)
    return coords, props["label"].astype(np.int32), img


def scan_frames(data_root, conditions, max_frames):
    """Scan for single frames across conditions."""
    dr = Path(data_root)
    frames = []
    for cond in conditions:
        for exp in sorted(dr.glob(f"{cond}/*")):
            if not exp.is_dir():
                continue
            tra = exp / "TRA"
            img_dir = exp / "img"
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Distortions (for SSL view generation)
# ═══════════════════════════════════════════════════════════════════════════════

def get_distortion_params(strength):
    """Return distortion parameter dict for given strength level.

    Args:
        strength: 'mild', 'medium', or 'strong'
    Returns:
        dict with keys: degrees, scale_range, jitter_std, dropout_p
    """
    if strength == 'mild':
        return {
            'degrees': 5,
            'scale_range': (0.95, 1.05),
            'jitter_std': 2,
            'dropout_p': 0.05,
        }
    elif strength == 'medium':
        return {
            'degrees': 10,
            'scale_range': (0.9, 1.1),
            'jitter_std': 4,
            'dropout_p': 0.1,
        }
    else:  # strong
        return {
            'degrees': 25,
            'scale_range': (0.7, 1.3),
            'jitter_std': 8,
            'dropout_p': 0.2,
        }


def apply_affine(coords, degrees=10, scale_range=(0.9, 1.1)):
    """Apply random rotation + scaling to coordinates."""
    theta = np.random.uniform(-degrees, degrees) / 180 * np.pi
    sx = np.random.uniform(*scale_range)
    sy = np.random.uniform(*scale_range)
    M = np.array([[sx * np.cos(theta), -sx * np.sin(theta)],
                  [sy * np.sin(theta),  sy * np.cos(theta)]])
    return coords @ M.T


def apply_jitter(coords, std=4):
    """Add per-cell Gaussian jitter."""
    return coords + np.random.randn(*coords.shape).astype(np.float32) * std


def apply_dropout(coords, labels, p=0.1):
    """Drop random subset of cells."""
    keep = np.random.rand(len(labels)) > p
    return coords[keep], labels[keep]


def distort_coords(coords, labels, dist_params):
    """Apply full distortion pipeline: affine → jitter → dropout.

    Args:
        coords: (N, 2) array of cell centroid coordinates
        labels: (N,) array of cell labels
        dist_params: dict from get_distortion_params()
    Returns:
        (coords_distorted, labels_distorted)
    """
    c = apply_affine(coords.copy(),
                     degrees=dist_params['degrees'],
                     scale_range=dist_params['scale_range'])
    c = apply_jitter(c, std=dist_params['jitter_std'])
    c, l = apply_dropout(c, labels.copy(), p=dist_params['dropout_p'])
    return c, l


# ═══════════════════════════════════════════════════════════════════════════════
#  NT-Xent Loss & Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def nt_xent_loss(z, temperature=0.05):
    """NT-Xent contrastive loss for paired views.

    Args:
        z: (2*N, D) where first N = view1, last N = view2, corresponding order.
    Returns:
        Scalar loss.
    """
    B = z.shape[0] // 2
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / temperature
    sim = sim - torch.eye(2 * B, device=sim.device) * 1e9
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, labels)


def compute_gap(z):
    """Compute intra-class vs inter-class cosine similarity gap.

    Intra: cosine sim between matching pairs (first B pairs).
    Inter: cosine sim between non-matching pairs.
    Returns (intra_mean, inter_mean, gap).
    """
    z = F.normalize(z, dim=-1)
    B = z.shape[0] // 2
    z1, z2 = z[:B], z[B:]

    # Intra: diagonal of z1 @ z2.T
    intra = (z1 * z2).sum(dim=-1).mean().item()

    # Inter: off-diagonal elements (excluding self-pairs)
    sim = z1 @ z2.T  # (B, B)
    n = B * B
    off_diag_sum = sim.sum() - sim.trace()
    inter = off_diag_sum.item() / max(n - B, 1)

    return intra, inter, intra - inter


def compute_effective_rank(z):
    """Compute entropy-based effective rank of the embedding matrix.

    Uses singular value entropy: eff_rank = exp(-sum(p_i * log(p_i)))
    where p_i are normalized singular values. This measures the
    dimensionality of the embedding distribution.

    Args:
        z: (N, D) embedding matrix (will be L2-normalized internally).
    Returns:
        Effective rank (scalar).
    """
    z = F.normalize(z, dim=-1)
    with torch.no_grad():
        s = torch.linalg.svd(z, full_matrices=False)[1]
        p = s / (s.sum() + 1e-10)
        p = p[p > 0]
        entropy = -(p * torch.log(p + 1e-10)).sum()
        eff_rank = torch.exp(entropy).item()
    return eff_rank


# ═══════════════════════════════════════════════════════════════════════════════
#  SSL Pretraining
# ═══════════════════════════════════════════════════════════════════════════════

def run_ssl(frames, cnn, args):
    """Train ScaledCNN with NT-Xent on distorted patches."""
    cnn = cnn.to(device)
    opt = torch.optim.Adam(cnn.parameters(), lr=args.lr)

    # Get distortion parameters for the chosen strength
    dist_params = get_distortion_params(args.distortion_strength)
    logger.info(f"Distortion strength: {args.distortion_strength} "
                f"(degrees={dist_params['degrees']}, "
                f"scale={dist_params['scale_range']}, "
                f"jitter={dist_params['jitter_std']}, "
                f"dropout={dist_params['dropout_p']})")

    # Pre-extract all data
    logger.info("Pre-extracting frame data...")
    ssl_data = []
    n_loaded = 0
    for mp, ip in frames:
        if n_loaded >= args.max_frames:
            break
        r = load_frame(mp, ip)
        if r is None:
            continue
        coords, labels, img = r

        # Original patches at cell centroids
        patches_orig = extract_patches(img, coords)

        # Distorted coordinates → patches at new positions
        coords_dist, labels_dist = distort_coords(coords, labels, dist_params)
        patches_dist = extract_patches(img, coords_dist)

        # Also get original patches for the distorted-view cells (for matching)
        ssl_data.append({
            "patches_orig": torch.from_numpy(patches_orig).float(),
            "patches_dist": torch.from_numpy(patches_dist).float(),
            "labels_orig": labels,
            "labels_dist": labels_dist,
        })
        n_loaded += 1
        logger.info(f"  Loaded frame {n_loaded}/{min(args.max_frames, len(frames))}: "
                     f"{len(labels)} cells")

    if n_loaded == 0:
        logger.error("No frames loaded!")
        sys.exit(1)

    # Split into train and validation sets
    n_val = min(10, max(1, len(ssl_data) // 5))
    val_data = ssl_data[:n_val]
    train_data = ssl_data[n_val:]
    logger.info(f"Train frames: {len(train_data)}, Validation frames: {len(val_data)}")

    logger.info(f"\nStarting SSL pretraining: {args.steps} steps, "
                f"lr={args.lr}, scale={args.scale}")
    logger.info(f"  CNN params: {sum(p.numel() for p in cnn.parameters()):,}")

    history = []
    for step in range(args.steps):
        cnn.train()
        losses = []
        intra_list, inter_list = [], []

        # Shuffle data each step
        perm = np.random.permutation(len(train_data))
        for idx in perm:
            item = train_data[idx]
            p_orig = item["patches_orig"]
            p_dist = item["patches_dist"]
            l_orig = item["labels_orig"]
            l_dist = item["labels_dist"]

            # Find matching cells between original and distorted views
            # Build matching based on shared labels
            shared = set(l_orig) & set(l_dist)
            if len(shared) < 2:
                continue

            idx_orig = [i for i, l in enumerate(l_orig) if l in shared]
            idx_dist = [i for i, l in enumerate(l_dist) if l in shared]

            # Keep first min(len) cells
            n = min(len(idx_orig), len(idx_dist))

            # Sort by label so matching cells are aligned
            labels_orig_shared = l_orig[idx_orig]
            labels_dist_shared = l_dist[idx_dist]
            sort_orig = np.argsort(labels_orig_shared)
            sort_dist = np.argsort(labels_dist_shared)

            po = p_orig[idx_orig][sort_orig][:n].unsqueeze(1).to(device)   # (N, 1, 64, 64)
            pd = p_dist[idx_dist][sort_dist][:n].unsqueeze(1).to(device)

            # Forward
            e_orig = cnn(po)      # (N, out_dim)
            e_dist = cnn(pd)      # (N, out_dim)

            z = torch.cat([e_orig, e_dist], dim=0)  # (2N, out_dim)
            loss = nt_xent_loss(z)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn.parameters(), 1.0)
            opt.step()

            losses.append(loss.item())

            # Compute gap metrics on this frame
            with torch.no_grad():
                ze = torch.cat([F.normalize(e_orig, dim=-1),
                                F.normalize(e_dist, dim=-1)], dim=0)
                intra, inter, gap = compute_gap(ze)
                intra_list.append(intra)
                inter_list.append(inter)

        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_intra = float(np.mean(intra_list)) if intra_list else 0.0
        avg_inter = float(np.mean(inter_list)) if inter_list else 0.0
        avg_gap = avg_intra - avg_inter

        # Validation metrics every 100 steps (detect collapse early)
        val_eff_rank = -1.0
        val_gap = -1.0
        if (step + 1) % 100 == 0 or step == 0:
            cnn.eval()
            with torch.no_grad():
                all_e_orig = []
                all_e_dist = []
                for item in val_data:
                    p_orig = item["patches_orig"]
                    p_dist = item["patches_dist"]
                    l_orig = item["labels_orig"]
                    l_dist = item["labels_dist"]

                    shared = set(l_orig) & set(l_dist)
                    if len(shared) < 2:
                        continue

                    idx_orig = [i for i, l in enumerate(l_orig) if l in shared]
                    idx_dist = [i for i, l in enumerate(l_dist) if l in shared]
                    n = min(len(idx_orig), len(idx_dist))

                    labels_orig_shared = l_orig[idx_orig]
                    labels_dist_shared = l_dist[idx_dist]
                    sort_orig = np.argsort(labels_orig_shared)
                    sort_dist = np.argsort(labels_dist_shared)

                    po = p_orig[idx_orig][sort_orig][:n].unsqueeze(1).to(device)
                    pd = p_dist[idx_dist][sort_dist][:n].unsqueeze(1).to(device)

                    e_orig = cnn(po)
                    e_dist = cnn(pd)
                    all_e_orig.append(e_orig)
                    all_e_dist.append(e_dist)

                if all_e_orig:
                    all_e_orig = torch.cat(all_e_orig, dim=0)
                    all_e_dist = torch.cat(all_e_dist, dim=0)
                    z_val = torch.cat([all_e_orig, all_e_dist], dim=0)
                    _, _, val_gap = compute_gap(z_val)
                    val_eff_rank = compute_effective_rank(z_val)

        history.append({
            "step": step,
            "loss": avg_loss,
            "intra": avg_intra,
            "inter": avg_inter,
            "gap": avg_gap,
            "val_gap": val_gap,
            "val_eff_rank": val_eff_rank,
        })

        if (step + 1) % 25 == 0 or step == 0:
            log_msg = (f"  step {step+1:4d}/{args.steps}: "
                       f"loss={avg_loss:.4f}  intra={avg_intra:.4f}  "
                       f"inter={avg_inter:.4f}  gap={avg_gap:.4f}")
            if val_eff_rank > 0:
                log_msg += (f"  val_gap={val_gap:.4f}  "
                            f"val_eff_rank={val_eff_rank:.2f}")
            logger.info(log_msg)

    # Save checkpoint (scale-specific name to avoid overwriting)
    ckpt_dir = Path("probe")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"cnn_ntxent_{args.scale}.pt"
    torch.save({
        "model_state_dict": cnn.state_dict(),
        "scale": args.scale,
        "out_dim": 128,
        "args": vars(args),
        "history": history,
        "distortion_params": dist_params,
    }, ckpt_path)
    logger.info(f"Checkpoint saved: {ckpt_path}")

    # Print final summary
    print()
    print("=" * 60)
    print("SSL PRETRAINING SUMMARY")
    print("=" * 60)
    print(f"  Scale:       {args.scale}")
    print(f"  Params:      {sum(p.numel() for p in cnn.parameters()):,}")
    print(f"  Steps:       {args.steps}")
    print(f"  Distortion:  {args.distortion_strength}")
    print(f"  Final loss:  {history[-1]['loss']:.4f}")
    print(f"  Intra sim:   {history[-1]['intra']:.4f}")
    print(f"  Inter sim:   {history[-1]['inter']:.4f}")
    print(f"  Gap:         {history[-1]['gap']:.4f}")
    if history[-1]['val_gap'] > 0:
        print(f"  Val gap:     {history[-1]['val_gap']:.4f}")
        print(f"  Val eff_rank:{history[-1]['val_eff_rank']:.2f}")
    print("=" * 60)

    return cnn, history


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NT-Xent pretraining of ScaledCNN on distorted bacteria patches"
    )
    parser.add_argument("--data-dirs", default="../../data/vanvliet",
                        help="Comma-separated list of data directories")
    parser.add_argument("--conditions", default="rpsM,recA,pheA,metA,cib,trpL")
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="large")
    parser.add_argument("--distortion-strength", choices=["mild", "medium", "strong"],
                        default="strong",
                        help="Strength of data augmentations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    global SEED
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Parse multiple data directories
    data_dirs = [d.strip() for d in args.data_dirs.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]

    logger.info(f"Data dirs: {data_dirs}")
    logger.info(f"Conditions: {conditions}")
    logger.info(f"Max frames: {args.max_frames}, Steps: {args.steps}")
    logger.info(f"Scale: {args.scale}, Distortion: {args.distortion_strength}, Device: {device}")

    # Scan frames from all data directories, pool together
    logger.info("Scanning frames...")
    all_frames = []
    for data_dir in data_dirs:
        dr = Path(data_dir)
        if not dr.exists():
            logger.warning(f"Data directory not found, skipping: {data_dir}")
            continue
        frames = scan_frames(data_dir, conditions, max_frames=10**9)
        logger.info(f"  {data_dir}: found {len(frames)} frames")
        all_frames.extend(frames)

    # Cap total frames
    if len(all_frames) > args.max_frames:
        logger.info(f"Capping to {args.max_frames} frames (from {len(all_frames)} found)")
        all_frames = all_frames[:args.max_frames]

    logger.info(f"Total frames: {len(all_frames)}")

    if len(all_frames) < 2:
        logger.error("Need at least 2 frames. Check data.")
        sys.exit(1)

    # Build CNN
    cnn = ScaledCNN(scale=args.scale, out_dim=128)
    logger.info(f"CNN built: {args.scale}, "
                f"{sum(p.numel() for p in cnn.parameters()):,} params")

    # Train SSL
    run_ssl(all_frames, cnn, args)

    logger.info("Done.")


if __name__ == "__main__":
    main()
