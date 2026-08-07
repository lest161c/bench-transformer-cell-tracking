"""Coordinate Shortcut Diagnostic — reproducible evidence for SSL convergence failure.

Tests THREE modes with identical architecture, data, and seed:
  Mode A: FourierPE(coords) + DINO(patches)  → MLP → NT-Xent  (current broken setup)
  Mode B: RandomNoise         + DINO(patches)  → MLP → NT-Xent  (fix: no coords)
  Mode C: FourierPE(coords)   + ZEROS          → MLP → NT-Xent  (smoking gun)

Outputs:
  results/diagnose_coord_shortcut/
    ├── mode_A.csv, mode_B.csv, mode_C.csv     per-step training metrics
    ├── summary.csv                             key metrics comparison
    ├── coordinate_shortcut.png                 3-panel figure
    └── summary.txt                             verdict text

Usage:
    cd benchmark_ssl
    .venv/bin/python diagnose_coord_shortcut.py
    .venv/bin/python diagnose_coord_shortcut.py --distortion full --steps 500 --max-frames 40
"""

import argparse
import logging
import time
import sys
import io
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tifffile import imread
from skimage.measure import regionprops_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("coord_diag")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# ─── DINO backbone ─────────────────────────────────────────────────────────────

logger.info("Loading DINOv2 (this may take a moment)...")
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
DINO_DIM = 384
PATCH_SIZE = 64


def extract_patches(img, centroids):
    """Extract square patches (PATCH_SIZE x PATCH_SIZE) centered on each centroid.

    Out-of-bounds regions are padded by reflection; returns a stacked float32
    array of shape (N, PATCH_SIZE, PATCH_SIZE).
    """
    h, w = img.shape[-2:]
    half = PATCH_SIZE // 2
    patches = []
    for cy, cx in centroids:
        cy_i, cx_i = int(round(float(cy))), int(round(float(cx)))
        cy_i = np.clip(cy_i, 0, h - 1)
        cx_i = np.clip(cx_i, 0, w - 1)
        y1, y2 = cy_i - half, cy_i + half
        x1, x2 = cx_i - half, cx_i + half
        if y2 <= 0 or x2 <= 0:
            patches.append(np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32))
            continue
        pt = max(0, -y1); pb = max(0, y2 - h)
        pl = max(0, -x1); pr = max(0, x2 - w)
        y1c, y2c = max(0, y1), min(h, y2)
        x1c, x2c = max(0, x1), min(w, x2)
        if y2c <= y1c or x2c <= x1c:
            patches.append(np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32))
            continue
        crop = img[y1c:y2c, x1c:x2c]
        if pt or pb or pl or pr:
            crop = np.pad(crop, ((pt, pb), (pl, pr)), mode="reflect")
        if crop.shape != (PATCH_SIZE, PATCH_SIZE):
            crop = np.pad(crop, ((0, max(0, PATCH_SIZE - crop.shape[0])),
                                  (0, max(0, PATCH_SIZE - crop.shape[1]))),
                           mode="reflect")[:PATCH_SIZE, :PATCH_SIZE]
        patches.append(crop)
    return np.stack(patches).astype(np.float32) if patches else np.zeros((0, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)


@torch.no_grad()
def compute_dino_embs(patches_np):
    """Compute 384D DINOv2 embeddings for a batch of normalized patches."""
    if len(patches_np) == 0:
        return np.zeros((0, DINO_DIM), dtype=np.float32)
    pmin = patches_np.min(axis=(1, 2), keepdims=True)
    pmax = patches_np.max(axis=(1, 2), keepdims=True)
    pn = (patches_np - pmin) / (pmax - pmin + 1e-8)
    patch_tensor = torch.from_numpy(pn).float().unsqueeze(1).to(device)
    patch_tensor = F.interpolate(patch_tensor, size=(224, 224), mode="bilinear", align_corners=False)
    patch_tensor = patch_tensor.expand(-1, 3, -1, -1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    patch_tensor = (patch_tensor - mean) / std
    return dino(patch_tensor).cpu().numpy()


# ─── Positional encodings ──────────────────────────────────────────────────────

class FourierPE(nn.Module):
    """Fourier positional encoding: sin/cos of coords at logarithmically spaced frequencies."""

    def __init__(self, coord_dim=2, per_dim=32):
        """Store per-dimension frequencies; output dim is coord_dim * per_dim * 2."""
        super().__init__()
        self.pe_dim = coord_dim * per_dim * 2
        freqs = 2.0 ** torch.linspace(0.0, 10.0, per_dim)
        self.register_buffer("freqs", freqs)

    def forward(self, coords):
        """Map coords (B, N, coord_dim) to (B, N, pe_dim) sin/cos features."""
        batch_size, seq_len, coord_dim = coords.shape
        parts = []
        for dim_idx in range(coord_dim):
            arg = coords[:, :, dim_idx].unsqueeze(-1) * self.freqs.view(1, 1, -1)
            parts.append(torch.sin(arg))
            parts.append(torch.cos(arg))
        return torch.cat(parts, dim=-1)


class NoPE(nn.Module):
    """Learned constant token used as a placeholder instead of positional encoding."""

    def __init__(self, pe_dim):
        """Initialize a small learnable token broadcast over all positions."""
        super().__init__()
        self.pe_dim = pe_dim
        self.token = nn.Parameter(torch.randn(1, 1, pe_dim) * 0.02)

    def forward(self, coords):
        return self.token.expand(coords.shape[0], coords.shape[1], -1)


# ─── Encoder ────────────────────────────────────────────────────────────────────

class ASCENTEncoder(nn.Module):
    """Encoder combining positional encoding (or noise) with optional DINO features."""

    def __init__(self, pe_dim, dino_dim=384, d_model=256, out_dim=64, use_dino=True):
        """Project PE and DINO features to d_model, fuse, and MLP to out_dim."""
        super().__init__()
        self.use_dino = use_dino
        self.pe_proj = nn.Linear(pe_dim, d_model)
        if use_dino:
            self.dino_proj = nn.Linear(dino_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, out_dim))

    def forward(self, dino_feats, pe):
        """Fuse normalized PE and DINO projections, normalize the MLP output."""
        pe_proj_out = F.normalize(self.pe_proj(pe), dim=-1)
        combined = self.norm(pe_proj_out + F.normalize(self.dino_proj(dino_feats), dim=-1)) if self.use_dino else self.norm(pe_proj_out)
        return F.normalize(self.mlp(combined), dim=-1)


# ─── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(z1, z2, labels1, labels2):
    """Intra/inter cosine similarity, their gap, and intra-frame similarity."""
    z1n = z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + 1e-12)
    z2n = z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + 1e-12)
    sim = z1n @ z2n.T
    intra, inter = [], []
    for i, lbl in enumerate(labels1):
        matches = set(np.where(labels2 == lbl)[0])
        for j in range(len(labels2)):
            (intra if j in matches else inter).append(sim[i, j])
    intra_f = [float((z1n[i] * z1n[j]).sum()) for i in range(len(z1n)) for j in range(i + 1, len(z1n))]
    return {
        "intra": float(np.mean(intra)) if intra else np.nan,
        "inter": float(np.mean(inter)) if inter else np.nan,
        "gap": float(np.mean(intra) - np.mean(inter)) if intra and inter else np.nan,
        "intra_frame": float(np.mean(intra_f)) if intra_f else np.nan,
    }


def nt_xent_loss(embeddings, temperature=0.05):
    """NT-Xent loss over concatenated two-view embeddings (first half = view 1)."""
    batch_size = embeddings.shape[0] // 2
    sim = embeddings @ embeddings.T / temperature
    sim.fill_diagonal_(-1e9)
    return F.cross_entropy(sim, torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(batch_size)]).to(embeddings.device))


# ─── Data loading ──────────────────────────────────────────────────────────────

def load_frame(mask_path, img_path):
    """Load a mask+image frame and return cell centroids, labels, and normalized image.

    Returns (coords, labels, img) or None if the mask contains fewer than 2 cells.
    """
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


# ─── Distortions ────────────────────────────────────────────────────────────────

def distort(coords, labels, mode):
    """Apply the requested distortion family to a frame's cell coordinates."""
    if mode == "jitter4":    return apply_jitter(coords, 4), labels.copy()
    if mode == "full":
        coords_dist = apply_affine(coords.copy(), 10, (0.9, 1.1))
        coords_dist = apply_jitter(coords_dist, 4)
        coords_dist, labels_dist = apply_dropout(coords_dist, labels.copy(), 0.1)
        return coords_dist, labels_dist
    return coords.copy(), labels.copy()

def apply_jitter(coords, std):
    """Add Gaussian noise of the given std to coordinates."""
    return coords + np.random.randn(*coords.shape).astype(np.float32) * std

def apply_affine(coords, degrees, scale_range):
    """Apply a random rotation (degrees) and scaling (scale_range) to coordinates."""
    angle = np.random.uniform(-degrees, degrees) / 180 * np.pi
    sx, sy = np.random.uniform(*scale_range), np.random.uniform(*scale_range)
    M = np.array([[sx*np.cos(angle), -sx*np.sin(angle)], [sy*np.sin(angle), sy*np.cos(angle)]])
    return coords @ M.T

def apply_dropout(coords, labels, drop_prob):
    """Randomly drop cells with probability drop_prob, keeping coordinate/label alignment."""
    keep_mask = np.random.rand(len(labels)) > drop_prob
    return coords[keep_mask], labels[keep_mask]


# ─── Precompute data ───────────────────────────────────────────────────────────

class PrecomputedData:
    """Holds train/val splits of (dino_e1, dino_e2, coords1, coords2, labels) for each frame."""
    def __init__(self, frames, distortion_mode, val_split=0.2):
        """Precompute DINO embeddings and distorted coords for each frame, then split."""
        all_pairs = []
        for mp, ip in frames:
            frame_data = load_frame(mp, ip)
            if frame_data is None: continue
            c1, lbl, img = frame_data
            c2, _ = distort(c1.copy(), lbl.copy(), distortion_mode)
            p1 = extract_patches(img, c1); e1 = compute_dino_embs(p1)
            p2 = extract_patches(img, c2); e2 = compute_dino_embs(p2)
            all_pairs.append((e1, e2, c1, c2, lbl))

        n_val = max(1, int(len(all_pairs) * val_split))
        self.train = all_pairs[:-n_val]
        self.val   = all_pairs[-n_val:]
        logger.info(f"Data: {len(self.train)} train + {len(self.val)} val frames")

    def to_gpu(self, subset):
        """Move a subset of precomputed pairs to GPU tensors (labels stay as numpy)."""
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


# ─── Run one experiment ────────────────────────────────────────────────────────

def run_experiment(name, pos_enc, data, steps, lr, d_model, use_dino, eval_every=25):
    """Train one SSL encoder variant and log per-step loss and gap metrics."""
    pe_dim = pos_enc.pe_dim
    encoder = ASCENTEncoder(pe_dim, DINO_DIM, d_model, use_dino=use_dino).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    train_gpu = data.to_gpu(data.train)
    val_gpu   = data.to_gpu(data.val)
    rows = []
    conv_step = steps

    for step in range(steps):
        encoder.train()
        losses = []
        for de1, de2, c1, c2, lbl in train_gpu:
            n_cells = min(len(de1), len(de2))
            if n_cells < 2: continue
            c1n, c2n = c1[:n_cells].unsqueeze(0), c2[:n_cells].unsqueeze(0)
            pe1 = pos_enc(c1n).squeeze(0); pe2 = pos_enc(c2n).squeeze(0)
            embeddings = torch.cat([encoder(de1[:n_cells], pe1), encoder(de2[:n_cells], pe2)])
            loss = nt_xent_loss(embeddings)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else np.nan
        if conv_step == steps and avg_loss and not np.isnan(avg_loss) and avg_loss < 0.5:
            conv_step = step

        if step % eval_every == 0 or step == steps - 1:
            with torch.no_grad():
                encoder.eval()
                def eval_gap(gpu_subset):
                    gaps, intras, inters, intraf = [], [], [], []
                    for de1, de2, c1, c2, lbl in gpu_subset:
                        n_cells = min(len(de1), len(de2))
                        if n_cells < 2: continue
                        c1n, c2n = c1[:n_cells].unsqueeze(0), c2[:n_cells].unsqueeze(0)
                        pe1 = pos_enc(c1n).squeeze(0); pe2 = pos_enc(c2n).squeeze(0)
                        z1 = encoder(de1[:n_cells], pe1).cpu().numpy()
                        z2 = encoder(de2[:n_cells], pe2).cpu().numpy()
                        metrics = compute_metrics(z1, z2, lbl[:n_cells], lbl[:n_cells])
                        if not np.isnan(metrics["gap"]):
                            gaps.append(metrics["gap"]); intras.append(metrics["intra"])
                            inters.append(metrics["inter"]); intraf.append(metrics["intra_frame"])
                    return gaps, intras, inters, intraf

                tr_gaps, tr_intras, tr_inters, tr_intraf = eval_gap(train_gpu)
                vl_gaps, vl_intras, vl_inters, vl_intraf = eval_gap(val_gpu)

                rows.append({
                    "step": step,
                    "loss": avg_loss,
                    "train_gap": np.mean(tr_gaps) if tr_gaps else np.nan,
                    "train_intra": np.mean(tr_intras) if tr_intras else np.nan,
                    "train_inter": np.mean(tr_inters) if tr_inters else np.nan,
                    "train_intra_frame": np.mean(tr_intraf) if tr_intraf else np.nan,
                    "val_gap": np.mean(vl_gaps) if vl_gaps else np.nan,
                    "val_intra": np.mean(vl_intras) if vl_intras else np.nan,
                    "val_inter": np.mean(vl_inters) if vl_inters else np.nan,
                    "val_intra_frame": np.mean(vl_intraf) if vl_intraf else np.nan,
                })
                logger.info(f"  [{name}] step {step:4d}: loss={avg_loss:.4f}  tr_gap={rows[-1]['train_gap']:.4f}  val_gap={rows[-1]['val_gap']:.4f}")

    return pd.DataFrame(rows), conv_step, encoder


# ─── Visualization ─────────────────────────────────────────────────────────────

def make_figure(results_dfs, results_meta, outdir, args):
    """Render a 6-panel figure summarizing loss, gap, and convergence across modes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {"A": "#e74c3c", "B": "#2ecc71", "C": "#3498db"}
    labels = {"A": "PE+coords + DINO (current)", "B": "PE+noise + DINO (fix)", "C": "PE+coords ONLY (smoking gun)"}
    styles = {"A": "-", "B": "--", "C": ":"}

    # Panel 1: Training loss
    ax = axes[0, 0]
    for mode in ["A", "B", "C"]:
        if mode not in results_dfs: continue
        df = results_dfs[mode]
        ax.plot(df["step"], df["loss"], styles[mode], color=colors[mode], label=labels[mode], linewidth=2)
    ax.set_ylabel("NT-Xent Loss"); ax.set_xlabel("Step"); ax.set_title("Training Loss")
    ax.set_yscale("log")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # Panel 2: Train gap (separation quality)
    ax = axes[0, 1]
    for mode in ["A", "B", "C"]:
        if mode not in results_dfs: continue
        df = results_dfs[mode]
        ax.plot(df["step"], df["train_gap"], styles[mode], color=colors[mode], linewidth=2)
    ax.set_ylabel("Separation Gap"); ax.set_xlabel("Step"); ax.set_title("Train Gap (intra − inter)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # Panel 3: Val gap (generalization)
    ax = axes[0, 2]
    for mode in ["A", "B", "C"]:
        if mode not in results_dfs: continue
        df = results_dfs[mode]
        ax.plot(df["step"], df["val_gap"], styles[mode], color=colors[mode], linewidth=2)
    ax.set_ylabel("Val Gap"); ax.set_xlabel("Step"); ax.set_title("Val Gap (generalization to unseen frames)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # Panel 4: Inter-cell similarity (collapse monitor)
    ax = axes[1, 0]
    for mode in ["A", "B", "C"]:
        if mode not in results_dfs: continue
        df = results_dfs[mode]
        ax.plot(df["step"], df["val_inter"], styles[mode], color=colors[mode], linewidth=2)
    ax.set_ylabel("Inter-cell cos sim"); ax.set_xlabel("Step"); ax.set_title("Val Inter-Cell Similarity (lower = better)")
    ax.axhline(0.0, color="gray", ls=":", alpha=0.5)
    ax.axhline(0.8, color="red", ls=":", alpha=0.3, label="collapse threshold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 5: Convergence speed bar chart
    ax = axes[1, 1]
    mode_names = []
    conv_steps = []
    conv_colors = []
    for mode in ["A", "B", "C"]:
        if mode not in results_meta: continue
        mode_names.append(labels[mode].split("(")[0].strip())
        conv_steps.append(results_meta[mode]["conv_step"])
        conv_colors.append(colors[mode])
    bars = ax.bar(mode_names, conv_steps, color=conv_colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Steps to loss < 0.5"); ax.set_title("Convergence Speed (lower = faster)")
    for bar, value in zip(bars, conv_steps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(value), ha="center", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 6: Final quality comparison (val gap + train gap)
    ax = axes[1, 2]
    x_pos = np.arange(len(mode_names))
    width = 0.35
    tr_final = [results_dfs[mode]["train_gap"].iloc[-1] for mode in ["A","B","C"] if mode in results_dfs]
    vl_final = [results_dfs[mode]["val_gap"].iloc[-1] for mode in ["A","B","C"] if mode in results_dfs]
    b1 = ax.bar(x_pos - width/2, tr_final, width, label="Train gap", color="#95a5a6", alpha=0.7)
    b2 = ax.bar(x_pos + width/2, vl_final, width, label="Val gap", color="#2c3e50", alpha=0.7)
    ax.set_ylabel("Final Gap"); ax.set_title("Final Representation Quality")
    ax.set_xticks(x_pos); ax.set_xticklabels(mode_names)
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(b1, tr_final):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", fontsize=8)
    for bar, value in zip(b2, vl_final):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", fontsize=8)

    plt.suptitle(f"Coordinate Shortcut Diagnostic — {args.distortion} distortion, {args.max_frames} frames, d_model={args.d_model}",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = outdir / "coordinate_shortcut.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved: {path}")
    return path


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    """Run the three-mode coordinate shortcut diagnostic and write CSV/figure/verdict outputs.

    Sets the RNG seed from --seed, loads frames, runs modes A (PE+coords+DINO),
    B (PE+noise+DINO), C (PE+coords only), and emits mode_*.csv, summary.csv,
    coordinate_shortcut.png, and summary.txt into --outdir.
    """
    parser = argparse.ArgumentParser(description="Coordinate shortcut diagnostic for DINO+SSL")
    parser.add_argument("--data-root", default="../data/vanvliet")
    parser.add_argument("--conditions", default="rpsM")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--pos-per-dim", type=int, default=32,
                       help="Fourier PE frequencies per coordinate dimension")
    parser.add_argument("--distortion", default="full",
                       choices=["jitter4", "full"])
    parser.add_argument("--eval-every", type=int, default=25,
                       help="Evaluate metrics every N steps")
    parser.add_argument("--outdir", default="runs/diagnose_coord_shortcut")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ─── Load data ──────────────────────────────────────────────────────────

    conditions = [cond.strip() for cond in args.conditions.split(",")]
    frames = scan_frames(args.data_root, conditions, args.max_frames)
    logger.info(f"Loaded {len(frames)} frames from {conditions}")

    if len(frames) < 4:
        logger.error("Need at least 4 frames for train/val split.")
        sys.exit(1)

    data = PrecomputedData(frames, args.distortion, val_split=0.2)

    # ─── PE variants ────────────────────────────────────────────────────────

    coord_dim = 2
    pe_dim = coord_dim * args.pos_per_dim * 2
    logger.info(f"Fourier PE dim: {pe_dim} ({args.pos_per_dim} freqs × {coord_dim} coords × 2)")

    fourier_pe = FourierPE(coord_dim, args.pos_per_dim).to(device)
    noise_pe   = NoPE(pe_dim).to(device)

    # ─── Run experiments ────────────────────────────────────────────────────

    results_dfs, results_meta = {}, {}

    for mode, pe_mod, use_dino, desc in [
        ("A", fourier_pe, True,  "PE+coords + DINO"),
        ("B", noise_pe,   True,  "PE+noise  + DINO"),
        ("C", fourier_pe, False, "PE+coords ONLY"),
    ]:
        logger.info(f"\n{'='*60}")
        logger.info(f"MODE {mode}: {desc}")
        logger.info(f"{'='*60}")
        t0 = time.time()
        df, conv, enc = run_experiment(mode, pe_mod, data, args.steps, args.lr, args.d_model, use_dino, args.eval_every)
        elapsed = time.time() - t0
        df.to_csv(outdir / f"mode_{mode}.csv", index=False)
        results_dfs[mode] = df
        results_meta[mode] = {"conv_step": conv, "time": elapsed, "description": desc}
        logger.info(f"  Saved: mode_{mode}.csv  ({elapsed:.1f}s)")

    # ─── Summary CSV ────────────────────────────────────────────────────────

    summary_rows = []
    for mode in ["A", "B", "C"]:
        if mode not in results_dfs: continue
        df = results_dfs[mode]
        meta = results_meta[mode]
        r0 = df.iloc[0]
        rf = df.iloc[-1]
        summary_rows.append({
            "mode": mode,
            "description": meta["description"],
            "epochs": args.steps,
            "conv_step_0.5": meta["conv_step"],
            "final_loss": rf["loss"],
            "init_train_gap": r0["train_gap"],
            "final_train_gap": rf["train_gap"],
            "train_gap_delta": rf["train_gap"] - r0["train_gap"],
            "init_val_gap": r0["val_gap"],
            "final_val_gap": rf["val_gap"],
            "val_gap_delta": rf["val_gap"] - r0["val_gap"],
            "final_val_inter": rf["val_inter"],
            "final_val_intra": rf["val_intra"],
            "wall_time_s": meta["time"],
            "distortion": args.distortion,
            "d_model": args.d_model,
            "pe_dim": pe_dim,
            "frames": len(data.train) + len(data.val),
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(outdir / "summary.csv", index=False)
    logger.info(f"Saved: summary.csv")

    # ─── Figure ─────────────────────────────────────────────────────────────

    fig_path = make_figure(results_dfs, results_meta, outdir, args)

    # ─── Verdict ────────────────────────────────────────────────────────────

    rows_a, rows_b = summary_rows[0], summary_rows[1]
    rows_c = summary_rows[2] if len(summary_rows) > 2 else None
    conv_c = results_meta["C"]["conv_step"] if "C" in results_meta else args.steps

    lines = []
    lines.append("=" * 65)
    lines.append("COORDINATE SHORTCUT DIAGNOSTIC — VERDICT")
    lines.append("=" * 65)
    lines.append(f"")
    lines.append(f"Setup: {args.max_frames} frames, {args.steps} steps, {args.distortion}, d_model={args.d_model}")
    lines.append(f"PE: Fourier {pe_dim}d, DINO: vits14 {DINO_DIM}d")
    lines.append(f"")

    # Key numbers
    lines.append("KEY METRICS:")
    lines.append(f"  Mode C (PE only):      converged={conv_c < args.steps}  final_gap={rows_c['final_train_gap']:.3f}  conv_step={conv_c}")
    lines.append(f"  Mode A (PE+DINO):       conv_step={rows_a['conv_step_0.5']}  final_val_gap={rows_a['final_val_gap']:.3f}")
    lines.append(f"  Mode B (noise+DINO):    conv_step={rows_b['conv_step_0.5']}  final_val_gap={rows_b['final_val_gap']:.3f}")
    lines.append(f"")

    # Smoking gun
    if conv_c < args.steps:
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║  SMOKING GUN: COORDINATE SHORTCUT EXISTS                    ║")
        lines.append(f"║  Mode C (PE only, ZERO visual input) converges at step {conv_c}.  ║")
        lines.append("║  The model solves NT-Xent using POSITION ALONE.              ║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        lines.append("")
    else:
        lines.append(f"  Mode C (PE only) does NOT converge → shortcut is NOT primary")
        lines.append("")

    # A vs B
    conv_diff = rows_b['conv_step_0.5'] - rows_a['conv_step_0.5']
    gap_diff  = rows_b['final_val_gap'] - rows_a['final_val_gap']

    if conv_diff > 5:
        lines.append(f"  ⚠  Mode A converges {conv_diff} steps FASTER than Mode B")
        lines.append(f"     → Coordinates provide a convergence shortcut.")
    elif conv_diff < -5:
        lines.append(f"  ✓  Mode B converges {-conv_diff} steps FASTER than Mode A")
        lines.append(f"     → Coordinates hurt convergence. Removing them helps.")
    else:
        lines.append(f"  ~  Similar convergence speed (Δ={conv_diff} steps)")

    if gap_diff > 0.02:
        lines.append(f"  ✓  Mode B achieves BETTER final val gap (+{gap_diff:.3f})")
        lines.append(f"     → Removing coordinates produces higher-quality embeddings.")
    elif gap_diff < -0.02:
        lines.append(f"  ⚠  Mode A achieves better val gap. Unexpected.")
    else:
        lines.append(f"  ~  Similar final val gap (Δ={gap_diff:.3f})")

    lines.append(f"")

    # Conclusions
    if conv_c < args.steps:
        lines.append("CONCLUSION:")
        lines.append("  ╔══════════════════════════════════════════════════════════════╗")
        lines.append("  ║  COORDINATE SHORTCUT IS REAL AND DOMINANT.                  ║")
        lines.append(f"  ║  Proved by Mode C: PE alone solves NT-Xent in {conv_c} steps.     ║")
        if gap_diff > 0.02:
            lines.append(f"  ║  Removing coords improves val gap by +{gap_diff:.3f}.          ║")
        lines.append("  ║                                                              ║")
        lines.append("  ║  Fix: NoPositionalEncoding during SSL.                       ║")
        lines.append("  ║  Code exists in model_parts.py:79-94 (commented out).        ║")
        lines.append("  ╚══════════════════════════════════════════════════════════════╝")
    else:
        lines.append("CONCLUSION: Coordinate shortcut not dominant at this scale.")
        lines.append("  The SSL failure may have another cause (decoder gap, distortion mismatch).")

    lines.append("")
    lines.append(f"Full results: {outdir}")
    lines.append(f"  {outdir}/mode_A.csv, mode_B.csv, mode_C.csv")
    lines.append(f"  {outdir}/summary.csv")
    lines.append(f"  {outdir}/coordinate_shortcut.png")
    lines.append("=" * 65)

    verdict_text = "\n".join(lines)
    print(verdict_text)
    with open(outdir / "summary.txt", "w") as file_handle:
        file_handle.write(verdict_text)

    logger.info(f"Done. Results in {outdir}/")
    return results_dfs, results_meta


if __name__ == "__main__":
    main()
