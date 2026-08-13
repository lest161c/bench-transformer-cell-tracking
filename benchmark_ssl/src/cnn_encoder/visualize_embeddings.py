#!/usr/bin/env python3
"""Extract embeddings from H100 convergence race models, log to W&B for interactive viz.

Usage:
    # From repo root (bench-transformer-cell-tracking/):
    python benchmark_ssl/cnn_encoder/visualize_embeddings.py

    # Dry run (no W&B, CPU-friendly):
    python benchmark_ssl/cnn_encoder/visualize_embeddings.py --dry-run --max-pairs 5

    # Override paths for local testing:
    python benchmark_ssl/cnn_encoder/visualize_embeddings.py \
        --data-root ../data/vanvliet \
        --ckpt-dir checkpoints/h100_conv_race \
        --checkpoint benchmark_ssl/cnn_encoder/probe/cnn_ntxent_large.pt
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from skimage.measure import regionprops as sk_regionprops
from tifffile import imread

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("embedding_viz")

# ─── Import model code from h100_convergence ──────────────────────────────
_SCRIPT_DIR = Path(__file__).parent.absolute()

from src.cnn_encoder.h100_convergence import (
    MiniEncoder,
    MiniDecoder,
    MiniTrackingTransformer,
    FourierPE,
    CNNPatchExtractor,
    ScaledCNN,
    scan_consecutive_pairs,
    load_frame,
    extract_patches,
    extract_regionprops_7d_by_label,
    CNN_FEAT_DIM,
    PE_DIM,
    PE_DIM_PER_COORD,
    PATCH_SIZE,
    COORD_DIM,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════════

def build_model(mode, device):
    """Build MiniTrackingTransformer for mode 'C' or 'R'."""
    enc = MiniEncoder(
        d_model=320, nhead=8, num_layers=6, pe_dim=PE_DIM,
        feat_dim=7,
        cnn_feat_dim=CNN_FEAT_DIM if mode == "C" else None,
    ).to(device)
    dec = MiniDecoder(d_model=320, nhead=8, num_layers=6).to(device)
    model = MiniTrackingTransformer(encoder=enc, decoder=dec, d_head=32).to(device)
    return model


def load_checkpoint(ckpt_path, device):
    """Load checkpoint and return state dict."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return None
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    return ckpt


def try_load_model(mode, ckpt_dir, device):
    """Try to load best checkpoint for given mode. Returns (model, epoch) or (None, None)."""
    ckpt_dir = Path(ckpt_dir)
    # Try best checkpoint first, then latest
    candidates = [
        ckpt_dir / f"model_{mode}_best.pt",
        ckpt_dir / f"model_{mode}_latest.pt",
    ]
    ckpt = None
    ckpt_path = None
    for cp in candidates:
        ckpt = load_checkpoint(cp, device)
        if ckpt is not None:
            ckpt_path = cp
            break

    if ckpt is None:
        logger.warning(f"No checkpoint found for Mode {mode} in {ckpt_dir}")
        return None, None

    model = build_model(mode, device)
    sd = ckpt.get("model_state_dict", ckpt)
    # Strip 'module.' prefix if saved with DataParallel
    if all(k.startswith("module.") for k in sd.keys()):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    epoch = ckpt.get("epoch", "unknown")
    logger.info(f"Mode {mode} model loaded from {ckpt_path} (epoch={epoch})")
    return model, epoch


# ═══════════════════════════════════════════════════════════════════════════
#  Embedding extraction
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings_from_pairs(model, pairs, pe_mod, mode_name,
                                  cnn_extractor=None, min_cells=4,
                                  max_pairs=None):
    """Extract per-cell embeddings from frame pairs.

    Iterates over pairs, loads each frame, computes features,
    and runs through model.encoder() to get (N, 320) embeddings.

    Returns:
        dict with keys: embedding, cell_id, frame_id, model_type, spatial_x, spatial_y
        Each value is a list.
    """
    model.eval()
    all_embs = {
        "embedding": [],
        "cell_id": [],
        "frame_id": [],
        "model_type": [],
        "spatial_x": [],
        "spatial_y": [],
    }

    n_pairs = len(pairs) if max_pairs is None else min(max_pairs, len(pairs))
    skipped = 0

    for pair_idx in range(n_pairs):
        mt, mn, img_t_path, img_n_path = pairs[pair_idx]

        # Load both frames
        rt = load_frame(mt, img_t_path)
        rn = load_frame(mn, img_n_path)
        if rt is None or rn is None:
            skipped += 1
            continue

        ct, lt, imgt = rt   # centroids (N_t, 2), labels (N_t,), image (H, W)
        cn, ln, imgn = rn

        shared = set(lt) & set(ln)
        if len(shared) < min_cells:
            skipped += 1
            continue
        if len(lt) < min_cells or len(ln) < min_cells:
            skipped += 1
            continue

        # Filter to shared cells only
        idx_t = [i for i, l in enumerate(lt) if l in shared]
        idx_n = [i for i, l in enumerate(ln) if l in shared]
        ct_s, lt_s = ct[idx_t], lt[idx_t]
        cn_s, ln_s = cn[idx_n], ln[idx_n]

        # Load masks for regionprops
        mask_t = imread(mt)
        mask_n = imread(mn)

        # 7D regionprops features
        feat_7d_t = extract_regionprops_7d_by_label(mask_t, imgt, lt_s)
        feat_7d_n = extract_regionprops_7d_by_label(mask_n, imgn, ln_s)

        if len(feat_7d_t) < 2 or len(feat_7d_n) < 2:
            skipped += 1
            continue

        # CNN features (needed for Mode C, computed for both to keep data pipeline uniform)
        if cnn_extractor is not None:
            pt_s = extract_patches(imgt, ct_s)[:, None, :, :]   # (N, 1, 64, 64)
            pn_s = extract_patches(imgn, cn_s)[:, None, :, :]
            cnn_t = torch.from_numpy(cnn_extractor.extract(pt_s)).float()
            cnn_n = torch.from_numpy(cnn_extractor.extract(pn_s)).float()
        else:
            cnn_t = cnn_n = None

        # Convert to tensors
        feat_t = torch.from_numpy(feat_7d_t).float().to(DEVICE)
        feat_n = torch.from_numpy(feat_7d_n).float().to(DEVICE)
        coords_t = torch.from_numpy(ct_s).float().unsqueeze(0).to(DEVICE)
        coords_n = torch.from_numpy(cn_s).float().unsqueeze(0).to(DEVICE)

        if cnn_t is not None:
            cnn_t = cnn_t.to(DEVICE)
            cnn_n = cnn_n.to(DEVICE)

        # Positional encoding
        pe_t = pe_mod(coords_t).squeeze(0)
        pe_n = pe_mod(coords_n).squeeze(0)

        # Encoder → embeddings
        enc_t = model.encoder(feat_t, pe_t,
                              cnn_t if mode_name == "C" else None)  # (N_t, 320)
        enc_n = model.encoder(feat_n, pe_n,
                              cnn_n if mode_name == "C" else None)  # (N_n, 320)

        emb_t_np = enc_t.cpu().numpy()
        emb_n_np = enc_n.cpu().numpy()

        # Store frame t
        for i in range(emb_t_np.shape[0]):
            all_embs["embedding"].append(emb_t_np[i].tolist())
            all_embs["cell_id"].append(int(lt_s[i]))
            all_embs["frame_id"].append(f"pair{pair_idx}_t")
            all_embs["model_type"].append(f"Mode_{mode_name}")
            all_embs["spatial_x"].append(float(ct_s[i, 1]))  # centroid-1 = x (col)
            all_embs["spatial_y"].append(float(ct_s[i, 0]))  # centroid-0 = y (row)

        # Store frame n
        for i in range(emb_n_np.shape[0]):
            all_embs["embedding"].append(emb_n_np[i].tolist())
            all_embs["cell_id"].append(int(ln_s[i]))
            all_embs["frame_id"].append(f"pair{pair_idx}_n")
            all_embs["model_type"].append(f"Mode_{mode_name}")
            all_embs["spatial_x"].append(float(cn_s[i, 1]))
            all_embs["spatial_y"].append(float(cn_s[i, 0]))

        if (pair_idx + 1) % 10 == 0:
            logger.info(f"  Processed {pair_idx + 1}/{n_pairs} pairs "
                        f"({len(all_embs['embedding'])} embeddings so far)")

    logger.info(f"Extracted {len(all_embs['embedding'])} embeddings from "
                f"{n_pairs} pairs (skipped {skipped})")
    return all_embs


# ═══════════════════════════════════════════════════════════════════════════
#  CKA metric
# ═══════════════════════════════════════════════════════════════════════════

def compute_cka(X, Y):
    """Linear Centered Kernel Alignment between two representation matrices.

    Args:
        X: (n, d1) numpy array
        Y: (n, d2) numpy array

    Returns:
        CKA score in [0, 1]
    """
    n = X.shape[0]
    if n != Y.shape[0]:
        logger.warning(f"CKA: sample count mismatch ({n} vs {Y.shape[0]}), truncating")
        m = min(n, Y.shape[0])
        X, Y = X[:m], Y[:m]

    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # HSIC via linear kernel (X X^T and Y Y^T)
    K = X @ X.T   # (n, n)
    L = Y @ Y.T

    # Centered Gram matrices
    H = np.eye(n) - np.ones((n, n)) / n
    K_c = H @ K @ H
    L_c = H @ L @ H

    hsic_xy = float(np.sum(K_c * L_c))
    hsic_xx = float(np.sum(K_c * K_c))
    hsic_yy = float(np.sum(L_c * L_c))

    cka = hsic_xy / np.sqrt(hsic_xx * hsic_yy + 1e-12)
    return cka


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualize H100 convergence race embeddings via W&B"
    )
    parser.add_argument("--data-root",
                        default="../../data/vanvliet"
                        if os.path.basename(os.getcwd()) == "benchmark_ssl"
                        else "../data/vanvliet")
    parser.add_argument("--conditions",
                        default="rpsM,recA,pheA,metA,cib,trpL")
    parser.add_argument("--max-pairs", type=int, default=50,
                        help="Number of frame pairs to load")
    parser.add_argument("--min-cells", type=int, default=4)
    parser.add_argument("--checkpoint", type=str,
                        default="probe/cnn_ntxent_large.pt"
                        if os.path.basename(os.getcwd()) == "cnn_encoder"
                        else "benchmark_ssl/cnn_encoder/probe/cnn_ntxent_large.pt",
                        help="Path to NT-Xent pretrained CNN for feature extraction")
    parser.add_argument("--scale", choices=["small", "medium", "large"],
                        default="large",
                        help="CNN architecture scale matching checkpoint")
    parser.add_argument("--ckpt-dir", type=str,
                        default="checkpoints/h100_conv_race",
                        help="Directory with model_{C,R}_{best,latest}.pt")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip W&B logging, just print stats")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    conditions = [c.strip() for c in args.conditions.split(",")]

    logger.info("=" * 60)
    logger.info("EMBEDDING VISUALIZATION — H100 Convergence Race")
    logger.info("=" * 60)
    logger.info(f"  Data root: {args.data_root}")
    logger.info(f"  Conditions: {conditions}")
    logger.info(f"  Max pairs: {args.max_pairs}")
    logger.info(f"  CNN checkpoint: {args.checkpoint} (scale={args.scale})")
    logger.info(f"  Model checkpoint dir: {args.ckpt_dir}")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Dry run: {args.dry_run}")

    # ── 1. Load frozen CNN feature extractor ────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        # Try relative to script dir
        alt_path = _SCRIPT_DIR / args.checkpoint
        if alt_path.exists():
            args.checkpoint = str(alt_path)
        else:
            logger.error(f"CNN checkpoint not found: {args.checkpoint} "
                         f"(also tried {alt_path})")
            sys.exit(1)

    logger.info("[1/5] Loading frozen CNN extractor...")
    cnn_extractor = CNNPatchExtractor(args.checkpoint, scale=args.scale)

    # ── 2. Scan pairs ───────────────────────────────────────────────────
    logger.info("[2/5] Scanning consecutive frame pairs...")
    pairs = scan_consecutive_pairs(
        args.data_root, conditions, max_pairs=args.max_pairs
    )
    logger.info(f"  Found {len(pairs)} pairs")
    if len(pairs) < 3:
        logger.error("Need at least 3 pairs. Aborting.")
        sys.exit(1)

    # ── 3. Load model checkpoints ───────────────────────────────────────
    logger.info("[3/5] Loading model checkpoints...")

    # Determine absolute checkpoint dir
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        # Try relative to repo root (research-proj)
        alt_dir = _SCRIPT_DIR.parent.parent.parent.parent / args.ckpt_dir
        if alt_dir.exists():
            ckpt_dir = alt_dir
        else:
            logger.warning(f"Checkpoint dir not found: {args.ckpt_dir} "
                           f"(tried {alt_dir}) — will skip model loading")

    models = {}
    for mode in ["C", "R"]:
        model, epoch = try_load_model(mode, ckpt_dir, DEVICE)
        if model is not None:
            models[mode] = model
        else:
            logger.warning(f"Mode {mode} will be skipped")

    if not models:
        logger.error("No checkpoints loaded. Nothing to visualize.")
        logger.error("Run h100_convergence.py first to train models.")
        sys.exit(1)

    # ── 4. Extract embeddings ──────────────────────────────────────────
    logger.info("[4/5] Extracting embeddings...")
    pe_mod = FourierPE(pos_per_dim=PE_DIM_PER_COORD).to(DEVICE)

    all_embeddings = {}
    for mode_name, model in models.items():
        logger.info(f"  Extracting Mode {mode_name} embeddings...")
        embs = extract_embeddings_from_pairs(
            model, pairs, pe_mod, mode_name,
            cnn_extractor=cnn_extractor,
            min_cells=args.min_cells,
            max_pairs=args.max_pairs,
        )
        all_embeddings[mode_name] = embs

    # ── 5. Compute CKA metric ──────────────────────────────────────────
    logger.info("[5/5] Computing metrics & logging...")

    cka_score = None
    if len(models) == 2:
        e_c = np.array(all_embeddings["C"]["embedding"])
        e_r = np.array(all_embeddings["R"]["embedding"])

        # For CKA we need matching samples. Truncate to minimum count.
        min_n = min(e_c.shape[0], e_r.shape[0])
        cka_score = compute_cka(e_c[:min_n], e_r[:min_n])
        logger.info(f"  CKA between Mode C and Mode R: {cka_score:.4f}")

    # ── Logging ─────────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("  Dry run — skipping W&B logging")
        for mode_name, embs in all_embeddings.items():
            n = len(embs["embedding"])
            logger.info(f"  Mode {mode_name}: {n} embeddings, "
                        f"dim={len(embs['embedding'][0]) if n else 'N/A'}")
        print("\nEmbedding stats:")
        for mode_name, embs in all_embeddings.items():
            arr = np.array(embs["embedding"])
            print(f"  Mode {mode_name}: shape={arr.shape}, "
                  f"mean={arr.mean():.4f}, std={arr.std():.4f}, "
                  f"norm={np.linalg.norm(arr, axis=-1).mean():.4f}")
        if cka_score is not None:
            print(f"  CKA: {cka_score:.4f}")
        return

    # W&B logging
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Install via: pip install wandb")
        sys.exit(1)

    # Check for W&B API key
    if not os.environ.get("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY not set — W&B may prompt for login")

    logger.info("  Initializing W&B run...")
    wandb_run = wandb.init(
        project="trackastra-cnn-convergence",
        name="embedding_viz",
        config={
            "max_pairs": args.max_pairs,
            "min_cells": args.min_cells,
            "scale": args.scale,
            "models_loaded": list(models.keys()),
            "seed": args.seed,
        },
    )

    # Log per-model embedding tables
    for mode_name, embs in all_embeddings.items():
        n = len(embs["embedding"])
        if n == 0:
            logger.warning(f"  No embeddings for Mode {mode_name}, skipping table")
            continue

        table = wandb.Table(
            columns=["embedding", "cell_id", "frame_id", "model_type",
                     "spatial_x", "spatial_y"]
        )
        for i in range(n):
            table.add_data(
                embs["embedding"][i],   # list of floats → auto-embedding in W&B
                embs["cell_id"][i],
                embs["frame_id"][i],
                embs["model_type"][i],
                embs["spatial_x"][i],
                embs["spatial_y"][i],
            )

        embed_dim = len(embs["embedding"][0])
        wandb_run.log({
            f"embeddings/Mode_{mode_name}": table,
            f"embeddings/Mode_{mode_name}_count": n,
            f"embeddings/Mode_{mode_name}_dim": embed_dim,
        })
        logger.info(f"  Logged Mode {mode_name} table: {n} rows × {embed_dim}D")

    # Log CKA scalar
    if cka_score is not None:
        wandb_run.log({"metrics/cka_C_vs_R": cka_score})

    # Log aggregate embedding stats
    for mode_name, embs in all_embeddings.items():
        arr = np.array(embs["embedding"])
        wandb_run.log({
            f"stats/{mode_name}_mean": float(arr.mean()),
            f"stats/{mode_name}_std": float(arr.std()),
            f"stats/{mode_name}_norm": float(np.linalg.norm(arr, axis=-1).mean()),
        })

    wandb_run.finish()
    logger.info("  W&B run finished. Dashboard:")
    logger.info("  https://wandb.ai/leonard-starke-tu-dresden/trackastra-cnn-convergence")

    # Print summary
    print()
    print("=" * 60)
    print("EMBEDDING VISUALIZATION — SUMMARY")
    print("=" * 60)
    for mode_name, embs in all_embeddings.items():
        arr = np.array(embs["embedding"])
        print(f"  Mode {mode_name}: {arr.shape[0]} embeddings, "
              f"dim={arr.shape[1]}, norm={np.linalg.norm(arr, axis=-1).mean():.3f}")
    if cka_score is not None:
        print(f"  CKA(Mode_C, Mode_R) = {cka_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
