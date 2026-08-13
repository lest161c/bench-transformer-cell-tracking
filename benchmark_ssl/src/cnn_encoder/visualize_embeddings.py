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

Anchor / query naming convention
--------------------------------
Anchor (frame t) is the reference; Query (frame t+1) is the candidate.
Each frame pair yields per-cell embeddings from the MiniTrackingTransformer
encoder, logged to W&B as interactive embedding tables.
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
    """Build MiniTrackingTransformer for mode 'C' or 'R'.

    Args:
        mode: 'C' for CNN features, 'R' for regionprops features.
        device: torch device to place the model on.

    Returns:
        MiniTrackingTransformer model instance.
    """
    encoder = MiniEncoder(
        d_model=320, nhead=8, num_layers=6, pe_dim=PE_DIM,
        feat_dim=7,
        cnn_feat_dim=CNN_FEAT_DIM if mode == "C" else None,
    ).to(device)
    decoder = MiniDecoder(d_model=320, nhead=8, num_layers=6).to(device)
    model = MiniTrackingTransformer(encoder=encoder, decoder=decoder, d_head=32).to(device)
    return model


def load_checkpoint(checkpoint_path, device):
    """Load checkpoint and return state dict.

    Args:
        checkpoint_path: path to the checkpoint file.
        device: torch device for mapping tensors.

    Returns:
        Loaded checkpoint dict, or None if the file does not exist.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return None
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    return checkpoint


def try_load_model(mode, checkpoint_dir, device):
    """Try to load best checkpoint for given mode. Returns (model, epoch) or (None, None).

    Args:
        mode: 'C' for CNN features, 'R' for regionprops features.
        checkpoint_dir: directory containing model_{mode}_{best,latest}.pt.
        device: torch device to place the model on.

    Returns:
        (model, epoch) tuple, or (None, None) if no checkpoint found.
    """
    checkpoint_dir = Path(checkpoint_dir)
    candidates = [
        checkpoint_dir / f"model_{mode}_best.pt",
        checkpoint_dir / f"model_{mode}_latest.pt",
    ]
    checkpoint = None
    checkpoint_path = None
    for candidate_path in candidates:
        checkpoint = load_checkpoint(candidate_path, device)
        if checkpoint is not None:
            checkpoint_path = candidate_path
            break

    if checkpoint is None:
        logger.warning(f"No checkpoint found for Mode {mode} in {checkpoint_dir}")
        return None, None

    model = build_model(mode, device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if all(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    logger.info(f"Mode {mode} model loaded from {checkpoint_path} (epoch={epoch})")
    return model, epoch


# ═══════════════════════════════════════════════════════════════════════════
#  Embedding extraction
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings_from_pairs(model, pairs, positional_encoder, mode_name,
                                  cnn_extractor=None, min_cells=4,
                                  max_pairs=None):
    """Extract per-cell embeddings from frame pairs.

    Iterates over pairs, loads each frame, computes features,
    and runs through model.encoder() to get (N, 320) embeddings.

    Args:
        model: MiniTrackingTransformer in eval mode.
        pairs: list of 4-tuples (mask_path_anchor, mask_path_query,
            img_path_anchor, img_path_query) from ``scan_consecutive_pairs``.
        positional_encoder: positional-encoding module (e.g. FourierPE).
        mode_name: 'C' or 'R', controls whether CNN features are passed to encoder.
        cnn_extractor: optional CNNPatchExtractor for CNN-based Mode C.
        min_cells: minimum number of shared cells required to keep a pair.
        max_pairs: maximum number of pairs to process.

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
        mask_path_anchor, mask_path_query, img_path_anchor, img_path_query = pairs[pair_idx]

        frame_anchor = load_frame(mask_path_anchor, img_path_anchor)
        frame_query = load_frame(mask_path_query, img_path_query)
        if frame_anchor is None or frame_query is None:
            skipped += 1
            continue

        coords_anchor, labels_anchor, img_anchor = frame_anchor   # centroids (N_anchor, 2), labels (N_anchor,), image (H, W)
        coords_query, labels_query, img_query = frame_query

        shared = set(labels_anchor) & set(labels_query)
        if len(shared) < min_cells:
            skipped += 1
            continue
        if len(labels_anchor) < min_cells or len(labels_query) < min_cells:
            skipped += 1
            continue

        # Filter to shared cells only
        idx_anchor = [i for i, label in enumerate(labels_anchor) if label in shared]
        idx_query = [i for i, label in enumerate(labels_query) if label in shared]
        coords_anchor_shared, labels_anchor_shared = coords_anchor[idx_anchor], labels_anchor[idx_anchor]
        coords_query_shared, labels_query_shared = coords_query[idx_query], labels_query[idx_query]

        # Load masks for regionprops
        mask_anchor = imread(mask_path_anchor)
        mask_query = imread(mask_path_query)

        # 7D regionprops features
        feat_7d_anchor = extract_regionprops_7d_by_label(mask_anchor, img_anchor, labels_anchor_shared)
        feat_7d_query = extract_regionprops_7d_by_label(mask_query, img_query, labels_query_shared)

        if len(feat_7d_anchor) < 2 or len(feat_7d_query) < 2:
            skipped += 1
            continue

        # CNN features (needed for Mode C, computed for both to keep data pipeline uniform)
        if cnn_extractor is not None:
            patches_anchor = extract_patches(img_anchor, coords_anchor_shared)[:, None, :, :]   # (N, 1, 64, 64)
            patches_query = extract_patches(img_query, coords_query_shared)[:, None, :, :]
            cnn_feat_anchor = torch.from_numpy(cnn_extractor.extract(patches_anchor)).float()
            cnn_feat_query = torch.from_numpy(cnn_extractor.extract(patches_query)).float()
        else:
            cnn_feat_anchor = cnn_feat_query = None

        # Convert to tensors
        feats_anchor = torch.from_numpy(feat_7d_anchor).float().to(DEVICE)
        feats_query = torch.from_numpy(feat_7d_query).float().to(DEVICE)
        coords_tensor_anchor = torch.from_numpy(coords_anchor_shared).float().unsqueeze(0).to(DEVICE)
        coords_tensor_query = torch.from_numpy(coords_query_shared).float().unsqueeze(0).to(DEVICE)

        if cnn_feat_anchor is not None:
            cnn_feat_anchor = cnn_feat_anchor.to(DEVICE)
            cnn_feat_query = cnn_feat_query.to(DEVICE)

        # Positional encoding
        pe_anchor = positional_encoder(coords_tensor_anchor).squeeze(0)
        pe_query = positional_encoder(coords_tensor_query).squeeze(0)

        # Encoder -> embeddings
        embeddings_anchor = model.encoder(feats_anchor, pe_anchor,
                                          cnn_feat_anchor if mode_name == "C" else None)  # (N_anchor, 320)
        embeddings_query = model.encoder(feats_query, pe_query,
                                         cnn_feat_query if mode_name == "C" else None)  # (N_query, 320)

        embeddings_anchor_np = embeddings_anchor.cpu().numpy()
        embeddings_query_np = embeddings_query.cpu().numpy()

        # Store anchor embeddings
        for i in range(embeddings_anchor_np.shape[0]):
            all_embs["embedding"].append(embeddings_anchor_np[i].tolist())
            all_embs["cell_id"].append(int(labels_anchor_shared[i]))
            all_embs["frame_id"].append(f"pair{pair_idx}_anchor")
            all_embs["model_type"].append(f"Mode_{mode_name}")
            all_embs["spatial_x"].append(float(coords_anchor_shared[i, 1]))  # centroid-1 = x (col)
            all_embs["spatial_y"].append(float(coords_anchor_shared[i, 0]))  # centroid-0 = y (row)

        # Store query embeddings
        for i in range(embeddings_query_np.shape[0]):
            all_embs["embedding"].append(embeddings_query_np[i].tolist())
            all_embs["cell_id"].append(int(labels_query_shared[i]))
            all_embs["frame_id"].append(f"pair{pair_idx}_query")
            all_embs["model_type"].append(f"Mode_{mode_name}")
            all_embs["spatial_x"].append(float(coords_query_shared[i, 1]))
            all_embs["spatial_y"].append(float(coords_query_shared[i, 0]))

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
    num_samples = X.shape[0]
    if num_samples != Y.shape[0]:
        logger.warning(f"CKA: sample count mismatch ({num_samples} vs {Y.shape[0]}), truncating")
        min_samples = min(num_samples, Y.shape[0])
        X, Y = X[:min_samples], Y[:min_samples]
        num_samples = min_samples

    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # HSIC via linear kernel (X X^T and Y Y^T)
    gram_matrix_x = X @ X.T   # (n, n)
    gram_matrix_y = Y @ Y.T

    # Centered Gram matrices
    centering_matrix = np.eye(num_samples) - np.ones((num_samples, num_samples)) / num_samples
    gram_matrix_x_centered = centering_matrix @ gram_matrix_x @ centering_matrix
    gram_matrix_y_centered = centering_matrix @ gram_matrix_y @ centering_matrix

    hsic_xy = float(np.sum(gram_matrix_x_centered * gram_matrix_y_centered))
    hsic_xx = float(np.sum(gram_matrix_x_centered * gram_matrix_x_centered))
    hsic_yy = float(np.sum(gram_matrix_y_centered * gram_matrix_y_centered))

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
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        # Try relative to script dir
        alt_checkpoint_path = _SCRIPT_DIR / args.checkpoint
        if alt_checkpoint_path.exists():
            args.checkpoint = str(alt_checkpoint_path)
        else:
            logger.error(f"CNN checkpoint not found: {args.checkpoint} "
                         f"(also tried {alt_checkpoint_path})")
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
    checkpoint_dir = Path(args.ckpt_dir)
    if not checkpoint_dir.exists():
        # Try relative to repo root (research-proj)
        alt_checkpoint_dir = _SCRIPT_DIR.parent.parent.parent.parent / args.ckpt_dir
        if alt_checkpoint_dir.exists():
            checkpoint_dir = alt_checkpoint_dir
        else:
            logger.warning(f"Checkpoint dir not found: {args.ckpt_dir} "
                           f"(tried {alt_checkpoint_dir}) — will skip model loading")

    models = {}
    for mode in ["C", "R"]:
        model, epoch = try_load_model(mode, checkpoint_dir, DEVICE)
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
    fourier_pe = FourierPE(pos_per_dim=PE_DIM_PER_COORD).to(DEVICE)

    all_embeddings = {}
    for mode_name, model in models.items():
        logger.info(f"  Extracting Mode {mode_name} embeddings...")
        mode_embeddings = extract_embeddings_from_pairs(
            model, pairs, fourier_pe, mode_name,
            cnn_extractor=cnn_extractor,
            min_cells=args.min_cells,
            max_pairs=args.max_pairs,
        )
        all_embeddings[mode_name] = mode_embeddings

    # ── 5. Compute CKA metric ──────────────────────────────────────────
    logger.info("[5/5] Computing metrics & logging...")

    cka_score = None
    if len(models) == 2:
        embeddings_mode_c = np.array(all_embeddings["C"]["embedding"])
        embeddings_mode_r = np.array(all_embeddings["R"]["embedding"])

        # For CKA we need matching samples. Truncate to minimum count.
        min_samples = min(embeddings_mode_c.shape[0], embeddings_mode_r.shape[0])
        cka_score = compute_cka(embeddings_mode_c[:min_samples], embeddings_mode_r[:min_samples])
        logger.info(f"  CKA between Mode C and Mode R: {cka_score:.4f}")

    # ── Logging ─────────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("  Dry run — skipping W&B logging")
        for mode_name, mode_embeddings in all_embeddings.items():
            num_embeddings = len(mode_embeddings["embedding"])
            logger.info(f"  Mode {mode_name}: {num_embeddings} embeddings, "
                        f"dim={len(mode_embeddings['embedding'][0]) if num_embeddings else 'N/A'}")
        print("\nEmbedding stats:")
        for mode_name, mode_embeddings in all_embeddings.items():
            embeddings_array = np.array(mode_embeddings["embedding"])
            print(f"  Mode {mode_name}: shape={embeddings_array.shape}, "
                  f"mean={embeddings_array.mean():.4f}, std={embeddings_array.std():.4f}, "
                  f"norm={np.linalg.norm(embeddings_array, axis=-1).mean():.4f}")
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
    for mode_name, mode_embeddings in all_embeddings.items():
        num_embeddings = len(mode_embeddings["embedding"])
        if num_embeddings == 0:
            logger.warning(f"  No embeddings for Mode {mode_name}, skipping table")
            continue

        table = wandb.Table(
            columns=["embedding", "cell_id", "frame_id", "model_type",
                     "spatial_x", "spatial_y"]
        )
        for i in range(num_embeddings):
            table.add_data(
                mode_embeddings["embedding"][i],   # list of floats → auto-embedding in W&B
                mode_embeddings["cell_id"][i],
                mode_embeddings["frame_id"][i],
                mode_embeddings["model_type"][i],
                mode_embeddings["spatial_x"][i],
                mode_embeddings["spatial_y"][i],
            )

        embedding_dim = len(mode_embeddings["embedding"][0])
        wandb_run.log({
            f"embeddings/Mode_{mode_name}": table,
            f"embeddings/Mode_{mode_name}_count": num_embeddings,
            f"embeddings/Mode_{mode_name}_dim": embedding_dim,
        })
        logger.info(f"  Logged Mode {mode_name} table: {num_embeddings} rows × {embedding_dim}D")

    # Log CKA scalar
    if cka_score is not None:
        wandb_run.log({"metrics/cka_C_vs_R": cka_score})

    # Log aggregate embedding stats
    for mode_name, mode_embeddings in all_embeddings.items():
        embeddings_array = np.array(mode_embeddings["embedding"])
        wandb_run.log({
            f"stats/{mode_name}_mean": float(embeddings_array.mean()),
            f"stats/{mode_name}_std": float(embeddings_array.std()),
            f"stats/{mode_name}_norm": float(np.linalg.norm(embeddings_array, axis=-1).mean()),
        })

    wandb_run.finish()
    logger.info("  W&B run finished. Dashboard:")
    logger.info("  https://wandb.ai/leonard-starke-tu-dresden/trackastra-cnn-convergence")

    # Print summary
    print()
    print("=" * 60)
    print("EMBEDDING VISUALIZATION — SUMMARY")
    print("=" * 60)
    for mode_name, mode_embeddings in all_embeddings.items():
        embeddings_array = np.array(mode_embeddings["embedding"])
        print(f"  Mode {mode_name}: {embeddings_array.shape[0]} embeddings, "
              f"dim={embeddings_array.shape[1]}, norm={np.linalg.norm(embeddings_array, axis=-1).mean():.3f}")
    if cka_score is not None:
        print(f"  CKA(Mode_C, Mode_R) = {cka_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
