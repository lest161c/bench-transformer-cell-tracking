"""Downstream comparison: SSL-pretrained vs random init for cell tracking.

Uses pretrained CellEmbedder to produce per-cell embeddings for each frame.
Tracks cells by computing cosine similarity between consecutive-frame
embeddings and solving bipartite matching via Hungarian algorithm.

Replaces old approach: training a BCE association head on pairwise logits.
"""

import csv, logging, sys, yaml, time, os
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from ssl_pipeline import load_experiment_frames, features_from_frame
from track_encoder import CellEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def embed_frame(model, mask, img, ndim, device):
    """Extract features and produce embeddings for all cells in a frame."""
    result = features_from_frame(mask, img)
    if result is None:
        return None, None, None

    coords, labels, feats_dict = result
    feats = np.concatenate(list(feats_dict.values()), axis=-1).astype(np.float32)

    c = torch.from_numpy(coords).float().unsqueeze(0).to(device)
    f = torch.from_numpy(feats).float().unsqueeze(0).to(device)
    pm = torch.zeros(1, len(labels), dtype=torch.bool, device=device)

    with torch.no_grad():
        z = model.encode(c, f, pm)

    return z.squeeze(0).cpu().numpy(), coords, labels


def track_via_embedding(model, frame_pairs, ndim, device, max_distance=50.0):
    """Track cells across frame pairs using cosine similarity + Hungarian.

    Args:
        model: CellEmbedder
        frame_pairs: list of (mask_src, img_src, mask_tgt, img_tgt) paths
        ndim: 2 or 3
        device: torch device
        max_distance: spatial distance cutoff for valid matches

    Returns:
        matches: list of (src_idx, tgt_idx) per frame pair
        accuracy stats
    """
    from tifffile import imread

    all_matches = []
    total_matched = 0
    total_gt = 0
    total_correct = 0
    total_mismatches = 0
    total_misses = 0

    model.eval()

    for pair_idx, (mask_path_src, img_path_src, mask_path_tgt, img_path_tgt) in enumerate(tqdm(frame_pairs, desc="Tracking")):
        mask_src = imread(mask_path_src)
        mask_tgt = imread(mask_path_tgt)
        img_src = np.clip((imread(img_path_src).astype(np.float32) - np.percentile(imread(img_path_src).astype(np.float32), 1)) /
                           (np.percentile(imread(img_path_src).astype(np.float32), 99.8) - np.percentile(imread(img_path_src).astype(np.float32), 1) + 1e-8), 0, 1)
        img_tgt = np.clip((imread(img_path_tgt).astype(np.float32) - np.percentile(imread(img_path_tgt).astype(np.float32), 1)) /
                           (np.percentile(imread(img_path_tgt).astype(np.float32), 99.8) - np.percentile(imread(img_path_tgt).astype(np.float32), 1) + 1e-8), 0, 1)

        z_src, coords_src, labels_src = embed_frame(model, mask_src, img_src, ndim, device)
        z_tgt, coords_tgt, labels_tgt = embed_frame(model, mask_tgt, img_tgt, ndim, device)

        if z_src is None or z_tgt is None:
            all_matches.append([])
            continue

        # Cosine similarity cost matrix: -cos_sim (lower = better match)
        sim = z_src @ z_tgt.T  # (N_src, N_tgt)
        cost = -sim

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost)

        matches = []
        for r, c in zip(row_ind, col_ind):
            dist = np.linalg.norm(coords_src[r] - coords_tgt[c])
            if dist <= max_distance:
                matches.append((int(labels_src[r]), int(labels_tgt[c]), r, c))
            else:
                # Reject matches over distance threshold
                pass

        all_matches.append(matches)

        # Ground truth: labels that persist across frames
        src_labels = set(int(l) for l in labels_src)
        tgt_labels = set(int(l) for l in labels_tgt)
        gt_persistent = src_labels & tgt_labels
        total_gt += len(gt_persistent)

        predicted = set()
        for _, tgt_l, _, _ in matches:
            predicted.add(tgt_l)

        correct = predicted & gt_persistent
        total_correct += len(correct)
        total_matched += len(matches)

        mismatches = len(predicted - gt_persistent)
        misses = len(gt_persistent - predicted)
        total_mismatches += mismatches
        total_misses += misses

    accuracy = total_correct / max(total_gt, 1)
    return all_matches, accuracy, total_correct, total_gt, total_mismatches, total_misses


def build_frame_pairs(frames):
    """Build consecutive frame pairs from loaded frames list.

    Pairs frames within the same experiment, sorted by frame index.
    """
    from collections import defaultdict
    by_exp = defaultdict(list)
    for cond, exp, fi, mp, ip in frames:
        by_exp[(cond, exp)].append((fi, mp, ip))
    pairs = []
    for (cond, exp), entries in by_exp.items():
        entries.sort(key=lambda x: x[0])
        for i in range(len(entries) - 1):
            pairs.append((entries[i][1], entries[i][2], entries[i + 1][1], entries[i + 1][2]))
    return pairs


def train_compare(ssl_ckpt_path, config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    frames = load_experiment_frames(cfg["data_root"], cfg.get("conditions"))
    frame_pairs = build_frame_pairs(frames)
    logger.info(f"Total frame pairs: {len(frame_pairs)}")

    # Split
    np.random.seed(42)
    idx = np.random.permutation(len(frame_pairs))
    n_val = max(1, int(len(idx) * 0.1))
    val_idx, test_idx = idx[:n_val], idx[n_val:]

    val_pairs = [frame_pairs[i] for i in val_idx]
    test_pairs = [frame_pairs[i] for i in test_idx]
    logger.info(f"Val pairs: {len(val_pairs)}, Test pairs: {len(test_pairs)}")

    enc_cfg = cfg.get("encoder", {})
    ndim = cfg.get("ndim", 2)

    results = {}

    for label in ["pretrained", "random"]:
        logger.info(f"\n{'='*60}\nEvaluating {label.upper()} model\n{'='*60}")

        model = CellEmbedder(
            feat_dim=7, coord_dim=ndim,
            d_model=enc_cfg.get("d_model", 128),
            nhead=enc_cfg.get("nhead", 4),
            num_layers=enc_cfg.get("num_layers", 4),
            dim_feedforward=enc_cfg.get("dim_feedforward", 256),
            dropout=enc_cfg.get("dropout", 0.1),
        ).to(device)

        if label == "pretrained" and ssl_ckpt_path and os.path.exists(ssl_ckpt_path):
            ckpt = torch.load(ssl_ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded SSL checkpoint from {ssl_ckpt_path}")
        else:
            logger.info("Using random initialization")

        _, val_acc, val_correct, val_gt, val_mis, val_miss = track_via_embedding(
            model, val_pairs, ndim, device
        )

        _, test_acc, test_correct, test_gt, test_mis, test_miss = track_via_embedding(
            model, test_pairs, ndim, device
        )

        results[label] = {
            "val_accuracy": val_acc,
            "val_correct": val_correct,
            "val_total": val_gt,
            "test_accuracy": test_acc,
            "test_correct": test_correct,
            "test_total": test_gt,
            "test_mismatches": test_mis,
            "test_misses": test_miss,
        }

        logger.info(f"{label.upper()} — Val: {val_acc:.4f} ({val_correct}/{val_gt}), "
                     f"Test: {test_acc:.4f} ({test_correct}/{test_gt})")

    # Save results
    outdir = Path("runs") / "downstream_compare"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "val_accuracy", "val_correct", "val_total",
                    "test_accuracy", "test_correct", "test_total",
                    "test_mismatches", "test_misses"])
        for label, r in results.items():
            w.writerow([label, f"{r['val_accuracy']:.4f}", r['val_correct'], r['val_total'],
                       f"{r['test_accuracy']:.4f}", r['test_correct'], r['test_total'],
                       r['test_mismatches'], r['test_misses']])

    logger.info(f"Results saved to {csv_path}")
    return results


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/ssl_contrastive/best_model.pt"
    train_compare(ckpt)
