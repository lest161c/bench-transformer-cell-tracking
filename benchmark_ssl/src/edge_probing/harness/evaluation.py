"""Feature evaluation orchestration and CV protocol.

Provides the high-level evaluation functions that tie together
frame-pair data building, the K-fold CV protocol (or the legacy
condition-split fallback), the training loop, and the shuffle
baseline. These are the functions called by the
``unified_edge_probe`` orchestrator script.

Anchor / query naming convention
-----------------------------------
Anchor (frame t) is the reference; Query (frame t+1) is the candidate.
Each datum carries features from two consecutive microscopy frames.
These are used for the link-prediction task.

"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from src.edge_probing.harness.feature_extractors import (
    compute_dino_embs,
    extract_regionprops_7d, extract_regionprops_7d_fourier,
    extract_hoct19, extract_hoct19_fourier, ScaledCNN,
    load_cached_features, save_cached_features,
    device,
)
from src.edge_probing.harness.frame_data import (
    load_frame, load_tracklets, extract_patches,
)
from src.edge_probing.harness.edge_datasets import (
    fit_feature_standardizer, apply_feature_standardizer,
    make_balanced_dataloader, flatten_to_pairs, concatenate_datasets,
    shuffle_edge_data_features,
)
from src.edge_probing.harness.probes import LinearProbe, MLPProbe, CNNProbeE2E
from src.edge_probing.harness.training import compute_metrics, train_probe
from src.edge_probing.harness.registry import FEATURE_REGISTRY

logger = logging.getLogger("edge_probing.evaluation")
TRAIN_CONDITIONS = ["rpsM", "recA", "pheA", "metA"]
VAL_CONDITIONS = ["cib", "trpL"]

# Per-feature run configuration consumed by evaluate_feature() (and by
# print_results_table() in results.py). Mirrors the canonical catalog in
# ``registry.FEATURE_REGISTRY``; kept as a dict for the exact display strings
# and dimensionalities used by the benchmark output.
FEATURE_CONFIGS = {
    'rp': {
        'feat_dim': 7,
        'display': 'Regionprops 7D',
        'needs_patches': False,
    },
    'rp_fourier': {
        'feat_dim': 55,
        'display': 'Regionprops 7D + Fourier PE',
        'needs_patches': False,
    },
    'cnn_frozen': {
        'feat_dim': 128,
        'display': 'CNN NT-Xent (frozen)',
        'needs_patches': False,
    },
    'cnn_e2e': {
        'feat_dim': 128,
        'display': 'CNN end-to-end',
        'needs_patches': True,
    },
    'dino': {
        'feat_dim': 384,
        'display': 'DINOv2 (frozen)',
        'needs_patches': False,
    },
    'hoct19': {
        'feat_dim': 13,
        'display': 'HOCT 19D (2D → 13D)',
        'needs_patches': False,
    },
    'hoct19_fourier': {
        'feat_dim': 43,
        'display': 'HOCT 13D + Fourier PE',
        'needs_patches': False,
    },
}


def build_frame_pairs(pairs, max_pairs, feature_type, checkpoint_path=None):
    """
    Build per-frame-pair data for a given feature type.

    For every consecutive frame pair found by ``scan_consecutive_pairs``,
    load the anchor (frame t) and query (frame t+1) images and
    segmentation masks, extract per-cell features of the requested
    ``feature_type``, align the two frames by label, and assemble the
    (anchor features, query features, edge target) tuples used by
    the probe training loop. See the module docstring for the full
    anchor / query naming convention.

    Returns list of dicts, each containing:
      - feat_anchor: (n_cells_anchor, D) or
        patches_anchor: (n_cells_anchor, 1, 64, 64)
      - feat_query: (n_cells_query, D) or
        patches_query: (n_cells_query, 1, 64, 64)
      - target: (n_cells_anchor, n_cells_query) binary matrix
      - n_pos, n_neg: counts
      - condition: string

    Args:
        pairs: list of consecutive frame-pair tuples from
            ``scan_consecutive_pairs`` (mask_anchor, mask_query,
            img_anchor, img_query, man_track_path, condition,
            experiment).
        max_pairs: maximum number of frame pairs to process.
        feature_type: one of 'rp', 'hoct19', 'cnn_frozen', 'dino',
            'cnn_e2e' (see FEATURE_CONFIGS).
        checkpoint_path: path to the NT-Xent checkpoint used to initialize
            the frozen ScaledCNN for 'cnn_frozen'; ignored otherwise.

    Returns:
        (edge_data, cnn_frozen_model): edge_data is the list of per-frame-pair
        dicts; cnn_frozen_model is the loaded frozen ScaledCNN (only for
        'cnn_frozen', else None).
    """
    edge_data = []
    total = 0

    # For CNN frozen: load model once
    cnn_frozen_model = None
    if feature_type == 'cnn_frozen':
        logger.info("  Loading ScaledCNN (large) with NT-Xent checkpoint...")
        cnn_frozen_model = ScaledCNN(scale='large', out_dim=128).to(device)
        checkpoint_state = torch.load(checkpoint_path, map_location=device)
        cnn_frozen_model.load_state_dict(checkpoint_state["model_state_dict"])
        cnn_frozen_model.eval()
        logger.info(f"  Loaded checkpoint: {checkpoint_path}")

    for mask_path_anchor, mask_path_query, img_path_anchor, img_path_query, man_track_path, condition, experiment in pairs[:max_pairs]:
        frame_anchor = load_frame(mask_path_anchor, img_path_anchor)
        frame_query = load_frame(mask_path_query, img_path_query)
        if frame_anchor is None or frame_query is None:
            continue
        coords_anchor, labels_anchor, img_anchor, mask_anchor = frame_anchor
        coords_query, labels_query, img_query, mask_query = frame_query

        if len(labels_anchor) < 3 or len(labels_query) < 3:
            continue

        tracklets = load_tracklets(man_track_path)

        # Extract features based on type
        if feature_type == 'rp':
            # Regionprops 7D
            _, labels_rp_anchor, feats_anchor = extract_regionprops_7d(mask_anchor, img_anchor)
            _, labels_rp_query, feats_query = extract_regionprops_7d(mask_query, img_query)
            if feats_anchor is None or feats_query is None:
                continue
            # Align by label
            label_map_anchor = {label: i for i, label in enumerate(labels_rp_anchor)}
            label_map_query = {label: i for i, label in enumerate(labels_rp_query)}
            idx_anchor = [label_map_anchor[label] for label in labels_anchor if label in label_map_anchor]
            idx_query = [label_map_query[label] for label in labels_query if label in label_map_query]
            if len(idx_anchor) < 2 or len(idx_query) < 2:
                continue
            feats_anchor = torch.from_numpy(feats_anchor[idx_anchor]).float()
            feats_query = torch.from_numpy(feats_query[idx_query]).float()
            labels_anchor_aligned = labels_anchor[[label in label_map_anchor for label in labels_anchor]]
            labels_query_aligned = labels_query[[label in label_map_query for label in labels_query]]

        elif feature_type == 'rp_fourier':
            # Regionprops 7D + Fourier PE of positions (t, y, x)
            frame_anchor_num = int(Path(mask_path_anchor).stem.replace("man_track", ""))
            frame_query_num = int(Path(mask_path_query).stem.replace("man_track", ""))
            _, labels_rp_anchor, feats_anchor = extract_regionprops_7d_fourier(
                mask_anchor, img_anchor, frame_idx=frame_anchor_num,
            )
            _, labels_rp_query, feats_query = extract_regionprops_7d_fourier(
                mask_query, img_query, frame_idx=frame_query_num,
            )
            if feats_anchor is None or feats_query is None:
                continue
            # Align by label
            label_map_anchor = {label: i for i, label in enumerate(labels_rp_anchor)}
            label_map_query = {label: i for i, label in enumerate(labels_rp_query)}
            idx_anchor = [label_map_anchor[label] for label in labels_anchor if label in label_map_anchor]
            idx_query = [label_map_query[label] for label in labels_query if label in label_map_query]
            if len(idx_anchor) < 2 or len(idx_query) < 2:
                continue
            feats_anchor = torch.from_numpy(feats_anchor[idx_anchor]).float()
            feats_query = torch.from_numpy(feats_query[idx_query]).float()
            labels_anchor_aligned = labels_anchor[[label in label_map_anchor for label in labels_anchor]]
            labels_query_aligned = labels_query[[label in label_map_query for label in labels_query]]

        elif feature_type == 'hoct19':
            # HOCT 13D (adapted from 19D): keep t, drop z, inertia 3×3→2×2
            frame_anchor_num = int(Path(mask_path_anchor).stem.replace("man_track", ""))
            frame_query_num = int(Path(mask_path_query).stem.replace("man_track", ""))
            _, labels_h_anchor, feats_anchor = extract_hoct19(mask_anchor, img_anchor, frame_idx=frame_anchor_num)
            _, labels_h_query, feats_query = extract_hoct19(mask_query, img_query, frame_idx=frame_query_num)
            if feats_anchor is None or feats_query is None:
                continue
            # Align by label
            label_map_anchor = {label: i for i, label in enumerate(labels_h_anchor)}
            label_map_query = {label: i for i, label in enumerate(labels_h_query)}
            idx_anchor = [label_map_anchor[label] for label in labels_anchor if label in label_map_anchor]
            idx_query = [label_map_query[label] for label in labels_query if label in label_map_query]
            if len(idx_anchor) < 2 or len(idx_query) < 2:
                continue
            feats_anchor = torch.from_numpy(feats_anchor[idx_anchor]).float()
            feats_query = torch.from_numpy(feats_query[idx_query]).float()
            labels_anchor_aligned = labels_anchor[[label in label_map_anchor for label in labels_anchor]]
            labels_query_aligned = labels_query[[label in label_map_query for label in labels_query]]

        elif feature_type == 'hoct19_fourier':
            # HOCT 13D + Fourier PE: keep t as scalar, replace raw
            # centroid coords with Fourier PE of (y, x)
            frame_anchor_num = int(Path(mask_path_anchor).stem.replace("man_track", ""))
            frame_query_num = int(Path(mask_path_query).stem.replace("man_track", ""))
            _, labels_h_anchor, feats_anchor = extract_hoct19_fourier(
                mask_anchor, img_anchor, frame_idx=frame_anchor_num,
            )
            _, labels_h_query, feats_query = extract_hoct19_fourier(
                mask_query, img_query, frame_idx=frame_query_num,
            )
            if feats_anchor is None or feats_query is None:
                continue
            # Align by label
            label_map_anchor = {label: i for i, label in enumerate(labels_h_anchor)}
            label_map_query = {label: i for i, label in enumerate(labels_h_query)}
            idx_anchor = [label_map_anchor[label] for label in labels_anchor if label in label_map_anchor]
            idx_query = [label_map_query[label] for label in labels_query if label in label_map_query]
            if len(idx_anchor) < 2 or len(idx_query) < 2:
                continue
            feats_anchor = torch.from_numpy(feats_anchor[idx_anchor]).float()
            feats_query = torch.from_numpy(feats_query[idx_query]).float()
            labels_anchor_aligned = labels_anchor[[label in label_map_anchor for label in labels_anchor]]
            labels_query_aligned = labels_query[[label in label_map_query for label in labels_query]]

        elif feature_type in ('cnn_frozen', 'dino'):
            # Check cache first
            frame_anchor_num = int(Path(mask_path_anchor).stem.replace("man_track", ""))
            frame_query_num = int(Path(mask_path_query).stem.replace("man_track", ""))
            cached_anchor, cached_labels_anchor = load_cached_features(condition, experiment, frame_anchor_num, feature_type, labels_anchor)
            cached_query, cached_labels_query = load_cached_features(condition, experiment, frame_query_num, feature_type, labels_query)

            if cached_anchor is not None and cached_query is not None:
                feats_anchor = torch.from_numpy(cached_anchor).float()
                feats_query = torch.from_numpy(cached_query).float()
                labels_anchor_aligned = cached_labels_anchor
                labels_query_aligned = cached_labels_query
            else:
                # Extract patches and compute features
                patches_anchor = extract_patches(img_anchor, coords_anchor)
                patches_query = extract_patches(img_query, coords_query)

                if feature_type == 'cnn_frozen':
                    patches_tensor_anchor = torch.from_numpy(patches_anchor).float().unsqueeze(1).to(device)
                    patches_tensor_query = torch.from_numpy(patches_query).float().unsqueeze(1).to(device)
                    with torch.no_grad():
                        feats_anchor_np = cnn_frozen_model(patches_tensor_anchor).cpu().numpy()
                        feats_query_np = cnn_frozen_model(patches_tensor_query).cpu().numpy()
                else:  # dino
                    feats_anchor_np = compute_dino_embs(patches_anchor)
                    feats_query_np = compute_dino_embs(patches_query)

                feats_anchor = torch.from_numpy(feats_anchor_np).float()
                feats_query = torch.from_numpy(feats_query_np).float()
                labels_anchor_aligned = labels_anchor
                labels_query_aligned = labels_query

                # Cache
                save_cached_features(condition, experiment, frame_anchor_num, feature_type, feats_anchor_np, labels_anchor)
                save_cached_features(condition, experiment, frame_query_num, feature_type, feats_query_np, labels_query)

        elif feature_type == 'cnn_e2e':
            # Just store patches; model is trained jointly
            patches_anchor = extract_patches(img_anchor, coords_anchor)
            patches_query = extract_patches(img_query, coords_query)
            n_cells_anchor, n_cells_query = len(labels_anchor), len(labels_query)
            target = torch.zeros(n_cells_anchor, n_cells_query, dtype=torch.float32)
            n_pos = 0
            for i, label_anchor in enumerate(labels_anchor):
                label_anchor_int = int(label_anchor)
                for j, label_query in enumerate(labels_query):
                    label_query_int = int(label_query)
                    if label_anchor_int == label_query_int:
                        target[i, j] = 1.0
                        n_pos += 1
                    elif label_query_int in tracklets and tracklets[label_query_int]['parent'] == label_anchor_int:
                        target[i, j] = 1.0
                        n_pos += 1
            if target.sum() < 1:
                continue
            edge_data.append({
                "patches_anchor": torch.from_numpy(patches_anchor).float().unsqueeze(1),  # (N, 1, H, W)
                "patches_query": torch.from_numpy(patches_query).float().unsqueeze(1),
                "target": target,
                "n_pos": n_pos,
                "n_neg": n_cells_anchor * n_cells_query - n_pos,
                "condition": condition,
                "experiment": experiment,
            })
            total += 1
            continue
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        # Build target matrix
        if feature_type != 'cnn_e2e':
            n_cells_anchor, n_cells_query = len(labels_anchor_aligned), len(labels_query_aligned)
            target = torch.zeros(n_cells_anchor, n_cells_query, dtype=torch.float32)
            n_pos = 0
            for i, label_anchor in enumerate(labels_anchor_aligned):
                label_anchor_int = int(label_anchor)
                for j, label_query in enumerate(labels_query_aligned):
                    label_query_int = int(label_query)
                    if label_anchor_int == label_query_int:
                        target[i, j] = 1.0
                        n_pos += 1
                    elif label_query_int in tracklets and tracklets[label_query_int]['parent'] == label_anchor_int:
                        target[i, j] = 1.0
                        n_pos += 1

            if target.sum() < 1:
                continue

            edge_data.append({
                "feat_anchor": feats_anchor,
                "feat_query": feats_query,
                "target": target,
                "n_pos": n_pos,
                "n_neg": n_cells_anchor * n_cells_query - n_pos,
                "condition": condition,
                "experiment": experiment,
            })
            total += 1

    logger.info(f"  Built {total} frame pairs for {feature_type}")
    return edge_data, cnn_frozen_model


def split_by_condition(edge_data, train_conditions, val_conditions):
    """Split edge data by condition (experimental group).

    Used by ``evaluate_feature`` in the legacy non-CV mode to partition
    frame pairs into a training set (majority conditions) and a validation
    set (held-out conditions).

    Args:
        edge_data: list of per-frame-pair dicts from ``build_frame_pairs``.
        train_conditions: condition names to use for training.
        val_conditions: condition names to use for validation.

    Returns:
        (train_data, val_data): the frame-pair dicts whose "condition" key
        matches each condition list.
    """
    train_data = [item for item in edge_data if item["condition"] in train_conditions]
    val_data = [item for item in edge_data if item["condition"] in val_conditions]
    return train_data, val_data


def run_cross_validation(frame_pair_datasets, probe_name, feat_dim, args,
                         n_folds=5, is_e2e=False):
    """
    Run K-fold cross-validation on frame-pair-level datasets.

    Each fold: train on K-1 folds, validate on 1 held-out fold.
    Returns dict with mean, std, min, max of metrics across folds.

    Args:
        frame_pair_datasets: list of EdgePairDataset / EdgePairDatasetPatches,
            one per frame pair.
        probe_name: 'Linear' or 'MLP'.
        feat_dim: dimensionality of each cell's feature vector.
        args: argparse Namespace with ``standardize``, ``lr``,
            ``batch_size_linear``, ``batch_size_mlp``, ``epochs``,
            ``patience``, ``eval_every``.
        n_folds: number of CV folds.
        is_e2e: whether the probe is a CNNProbeE2E trained on patches.

    Returns:
        Dict with fold_results and aggregate bal_acc/f1 statistics.
    """
    if len(frame_pair_datasets) < n_folds:
        n_folds = len(frame_pair_datasets)
        logger.warning(f"    Reducing folds to {n_folds} (not enough frame pairs)")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    all_fold_results = []

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(frame_pair_datasets)):
        logger.info(f"    Fold {fold_idx + 1}/{n_folds}: "
                    f"{len(train_indices)} train / {len(val_indices)} val frame pairs")

        train_datasets = [frame_pair_datasets[i] for i in train_indices]
        val_datasets = [frame_pair_datasets[i] for i in val_indices]

        # Leakage-safe standardization: fit on train folds, transform val fold
        if not is_e2e and getattr(args, "standardize", True):
            feat_mean, feat_std = fit_feature_standardizer(train_datasets)
            train_datasets = apply_feature_standardizer(train_datasets, feat_mean, feat_std)
            val_datasets = apply_feature_standardizer(val_datasets, feat_mean, feat_std)

        train_dataset = concatenate_datasets(train_datasets)
        val_dataset = concatenate_datasets(val_datasets)

        lr = args.lr if probe_name == "Linear" else args.lr / 10
        batch_size = args.batch_size_linear if probe_name == "Linear" else args.batch_size_mlp

        if is_e2e:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                collate_fn=lambda batch: (
                    torch.stack([sample[0] for sample in batch]),
                    torch.stack([sample[1] for sample in batch]),
                    torch.stack([sample[2] for sample in batch]),
                )
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=lambda batch: (
                    torch.stack([sample[0] for sample in batch]),
                    torch.stack([sample[1] for sample in batch]),
                    torch.stack([sample[2] for sample in batch]),
                )
            )
            use_probe = "linear" if probe_name == "Linear" else "mlp"
            model = CNNProbeE2E(scale='large', out_dim=128, probe_type=use_probe)
            result = train_probe(
                model, train_loader, val_loader,
                epochs=args.epochs, lr=lr,
                patience=args.patience, eval_every=args.eval_every,
                is_e2e=True,
            )
        else:
            train_loader = make_balanced_dataloader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
            )

            if probe_name == "Linear":
                probe = LinearProbe(feat_dim)
            else:
                probe = MLPProbe(feat_dim)

            result = train_probe(
                probe, train_loader, val_loader,
                epochs=args.epochs, lr=lr,
                patience=args.patience, eval_every=args.eval_every,
                is_e2e=False,
            )

        all_fold_results.append(result)

    bal_accs = [result["final_bal_acc"] for result in all_fold_results]
    f1s = [result["final_f1"] for result in all_fold_results]

    return {
        "fold_results": all_fold_results,
        "bal_acc_mean": float(np.mean(bal_accs)),
        "bal_acc_std": float(np.std(bal_accs)),
        "bal_acc_min": float(np.min(bal_accs)),
        "bal_acc_max": float(np.max(bal_accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "f1_min": float(np.min(f1s)),
        "f1_max": float(np.max(f1s)),
        "is_cv": True,
        "standardized": bool(not is_e2e and getattr(args, "standardize", True)),
    }


def evaluate_feature(feature_type, all_pairs, args, checkpoint_path):
    """Run full evaluation for a single feature type. Returns dict of results.

    In CV mode (``args.cv_folds > 0``) runs K-fold cross-validation over all
    frame pairs (plus the optional shuffled-feature baseline). Otherwise falls
    back to the legacy condition-based train/val split.

    Args:
        feature_type: feature key from FEATURE_CONFIGS / FEATURE_REGISTRY.
        all_pairs: list of consecutive frame-pair tuples from
            ``scan_consecutive_pairs``.
        args: argparse Namespace (see ``parse_args`` in the orchestrator).
        checkpoint_path: NT-Xent checkpoint path for 'cnn_frozen' (or None).

    Returns:
        Dict mapping probe key -> metrics (possibly empty), or None if the
        feature was skipped due to insufficient data.
    """
    if feature_type not in FEATURE_REGISTRY:
        raise ValueError(f"Unknown feature_type: {feature_type}")
    feature_config = FEATURE_CONFIGS[feature_type]
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Feature: {feature_config['display']} ({feature_config['feat_dim']}D)")
    logger.info(f"{'=' * 60}")

    # ── Build frame-pair data ────────────────────────────────────────────
    start_time = time.time()
    edge_data, cnn_model = build_frame_pairs(
        all_pairs, args.max_pairs, feature_type, checkpoint_path
    )
    logger.info(f"  Built {len(edge_data)} frame pairs in {time.time() - start_time:.1f}s")

    if len(edge_data) < 2:
        logger.warning(f"  Not enough data ({len(edge_data)}), skipping")
        return None

    # ═══════════════════════════════════════════════════════════════════════
    #  CV MODE (overrides train/val split)
    # ═══════════════════════════════════════════════════════════════════════
    if args.cv_folds > 0:
        logger.info(f"  Running {args.cv_folds}-fold CV on {len(edge_data)} frame pairs (all conditions)")
        if feature_type != 'cnn_e2e' and getattr(args, "standardize", True):
            logger.info("  Features z-score standardized per fold (fit on train folds)")
        all_datasets = flatten_to_pairs(edge_data)
        logger.info(f"  Total frame-pair datasets: {len(all_datasets)}")
        if not all_datasets:
            logger.warning("  No pairs after flattening, skipping")
            return None

        results = {}

        def _run_cv(probe_name, feat_dim, is_e2e):
            """Run k-fold cross-validation for one probe type on all frame-pair datasets."""
            return run_cross_validation(
                all_datasets, probe_name, feat_dim, args,
                n_folds=args.cv_folds, is_e2e=is_e2e,
            )

        # Linear probe
        if args.probe in ("linear", "both"):
            probe_start_time = time.time()
            cv_result = _run_cv("Linear", feature_config['feat_dim'],
                                is_e2e=(feature_type == 'cnn_e2e'))
            elapsed = time.time() - probe_start_time
            if cv_result:
                logger.info(f"    Linear CV: bal_acc={cv_result['bal_acc_mean']:.4f}\u00b1{cv_result['bal_acc_std']:.4f}, "
                            f"f1={cv_result['f1_mean']:.4f}\u00b1{cv_result['f1_std']:.4f} ({elapsed:.1f}s)")
                results["linear"] = cv_result

        # MLP probe
        if args.probe in ("mlp", "both"):
            probe_start_time = time.time()
            cv_result = _run_cv("MLP", feature_config['feat_dim'],
                                is_e2e=(feature_type == 'cnn_e2e'))
            elapsed = time.time() - probe_start_time
            if cv_result:
                logger.info(f"    MLP CV: bal_acc={cv_result['bal_acc_mean']:.4f}\u00b1{cv_result['bal_acc_std']:.4f}, "
                            f"f1={cv_result['f1_mean']:.4f}\u00b1{cv_result['f1_std']:.4f} ({elapsed:.1f}s)")
                results["mlp"] = cv_result

        # Shuffle baseline
        if args.shuffle_baseline:
            logger.info(f"  Running shuffled feature baseline ({args.cv_folds}-fold CV)...")
            shuffled_edge_data = shuffle_edge_data_features(edge_data, seed=args.seed)
            shuffled_datasets = flatten_to_pairs(shuffled_edge_data)
            if shuffled_datasets:
                if args.probe in ("linear", "both"):
                    probe_start_time = time.time()
                    shuffle_result = run_cross_validation(
                        shuffled_datasets, "Linear", feature_config['feat_dim'], args,
                        n_folds=args.cv_folds, is_e2e=(feature_type == 'cnn_e2e'),
                    )
                    elapsed = time.time() - probe_start_time
                    if shuffle_result:
                        logger.info(f"    Shuffled Linear CV: bal_acc={shuffle_result['bal_acc_mean']:.4f}\u00b1{shuffle_result['bal_acc_std']:.4f} "
                                    f"({elapsed:.1f}s)")
                        results["linear_shuffled"] = shuffle_result

                if args.probe in ("mlp", "both"):
                    probe_start_time = time.time()
                    shuffle_result = run_cross_validation(
                        shuffled_datasets, "MLP", feature_config['feat_dim'], args,
                        n_folds=args.cv_folds, is_e2e=(feature_type == 'cnn_e2e'),
                    )
                    elapsed = time.time() - probe_start_time
                    if shuffle_result:
                        logger.info(f"    Shuffled MLP CV: bal_acc={shuffle_result['bal_acc_mean']:.4f}\u00b1{shuffle_result['bal_acc_std']:.4f} "
                                    f"({elapsed:.1f}s)")
                        results["mlp_shuffled"] = shuffle_result
            else:
                logger.warning("  No shuffled datasets, skipping shuffle baseline")

        return results

    # ── Split by condition (non-CV mode) ──────────────────────────────────
    train_data, val_data = split_by_condition(
        edge_data, TRAIN_CONDITIONS, VAL_CONDITIONS
    )
    logger.info(f"  Train conditions: {TRAIN_CONDITIONS} -> {len(train_data)} pairs")
    logger.info(f"  Val conditions:   {VAL_CONDITIONS} -> {len(val_data)} pairs")

    if len(val_data) < 1:
        train_frac = 1.0 - args.val_frac
        logger.warning(f"  Val empty ({VAL_CONDITIONS} have 0 pairs), falling back to random "
                       f"{train_frac:.0%}/{args.val_frac:.0%} split")
        import random
        random.shuffle(train_data)
        split = max(1, int(train_frac * len(train_data)))
        val_data = train_data[split:]
        train_data = train_data[:split]
        logger.info(f"  Random split: {len(train_data)} train, {len(val_data)} val")
    if len(train_data) < 1 or len(val_data) < 1:
        logger.warning("  Need at least 1 train and 1 val pair, skipping")
        return None

    # ── Flatten to pair-level datasets ────────────────────────────────────
    train_datasets = flatten_to_pairs(train_data)
    val_datasets = flatten_to_pairs(val_data)

    if not train_datasets or not val_datasets:
        logger.warning("  No pairs after flattening, skipping")
        return None

    # ── Leakage-safe standardization: fit on train split, transform val ───
    if feature_type != 'cnn_e2e' and getattr(args, "standardize", True):
        feat_mean, feat_std = fit_feature_standardizer(train_datasets)
        train_datasets = apply_feature_standardizer(train_datasets, feat_mean, feat_std)
        val_datasets = apply_feature_standardizer(val_datasets, feat_mean, feat_std)
        logger.info("  Features z-score standardized (fit on train split)")

    train_dataset = concatenate_datasets(train_datasets)
    val_dataset = concatenate_datasets(val_datasets)

    n_train_pos = sum(int(dataset.labels.sum()) for dataset in train_datasets)
    n_train_neg = sum(len(dataset) - int(dataset.labels.sum()) for dataset in train_datasets)
    logger.info(f"  Train pairs: {len(train_dataset)} ({n_train_pos} pos / {n_train_neg} neg)")
    n_val_pos = sum(int(dataset.labels.sum()) for dataset in val_datasets)
    n_val_neg = sum(len(dataset) - int(dataset.labels.sum()) for dataset in val_datasets)
    logger.info(f"  Val pairs:   {len(val_dataset)} ({n_val_pos} pos / {n_val_neg} neg)")

    results = {}

    # ── Helper to run one probe type ──────────────────────────────────────
    def _run_probe(probe_name, probe_lr, batch_size):
        """Train and evaluate one probe architecture on this feature type.

        Args:
            probe_name: 'Linear' or 'MLP'.
            probe_lr: learning rate for the probe.
            batch_size: batch size for training and validation loaders.

        Returns:
            Metrics dict from train_probe, or None.
        """
        logger.info(f"  Training {probe_name} probe (lr={probe_lr}, bs={batch_size})...")

        if feature_type == 'cnn_e2e':
            # End-to-end: patches stored in datasets, build loaders specially
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                collate_fn=lambda batch: (
                    torch.stack([sample[0] for sample in batch]),
                    torch.stack([sample[1] for sample in batch]),
                    torch.stack([sample[2] for sample in batch]),
                )
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=lambda batch: (
                    torch.stack([sample[0] for sample in batch]),
                    torch.stack([sample[1] for sample in batch]),
                    torch.stack([sample[2] for sample in batch]),
                )
            )

            use_probe = "linear" if probe_name == "Linear" else "mlp"
            model = CNNProbeE2E(scale='large', out_dim=128, probe_type=use_probe)
            result = train_probe(
                model, train_loader, val_loader,
                epochs=args.epochs, lr=probe_lr,
                patience=args.patience, eval_every=args.eval_every,
                is_e2e=True,
            )
        else:
            # Feature-based: use balanced sampling
            train_loader = make_balanced_dataloader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
            )

            probe_feat_dim = feature_config['feat_dim']
            if probe_name == "Linear":
                probe = LinearProbe(probe_feat_dim)
            else:
                probe = MLPProbe(probe_feat_dim)

            result = train_probe(
                probe, train_loader, val_loader,
                epochs=args.epochs, lr=probe_lr,
                patience=args.patience, eval_every=args.eval_every,
                is_e2e=False,
            )

        return result

    # ── Linear probe ─────────────────────────────────────────────────────
    if args.probe in ("linear", "both"):
        probe_start_time = time.time()
        result = _run_probe("Linear", args.lr, args.batch_size_linear)
        elapsed = time.time() - probe_start_time
        if result:
            logger.info(f"    Linear: bal_acc={result['final_bal_acc']:.4f}, "
                        f"f1={result['final_f1']:.4f} ({elapsed:.1f}s)")
            results["linear"] = result

    # ── MLP probe ────────────────────────────────────────────────────────
    if args.probe in ("mlp", "both"):
        probe_start_time = time.time()
        result = _run_probe("MLP", args.lr / 10, args.batch_size_mlp)
        elapsed = time.time() - probe_start_time
        if result:
            logger.info(f"    MLP: bal_acc={result['final_bal_acc']:.4f}, "
                        f"f1={result['final_f1']:.4f} ({elapsed:.1f}s)")
            results["mlp"] = result

    return results
