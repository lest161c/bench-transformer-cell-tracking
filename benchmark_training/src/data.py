"""Data loading helpers for benchmark_training.

Provides three entry points used by the benchmark scripts:

- :func:`make_synthetic_batch` — single synthetic ``(cs, fs, ct, ft, a)``
  batch with controllable ``N`` and ``B``.
- :func:`create_tracking_pairs` — pairs of adjacent frames from real
  vanvliet images, with features extracted via ``benchmark_ssl``.
- :func:`load_real_pairs` — alias of :func:`create_tracking_pairs` kept
  for scripts that imported the older name.

The benchmark scripts call ``sys.path.insert`` for ``benchmark_attn``
and ``benchmark_ssl`` before importing this module so that the
``features_from_frame`` and ``load_experiment_frames`` helpers resolve.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Synthetic data
# ------------------------------------------------------------------ #

def make_synthetic_batch(
    num_cells: int,
    batch_size: int = 2,
    feat_dim: int = 7,
    coord_dim: int = 2,
    cell_scale: float = 100.0,
    jitter_scale: float = 3.0,
    device: Optional[torch.device] = None,
) -> dict:
    """Generate a synthetic src→tgt batch with identity association.

    Cells are placed uniformly on a ``[-cell_scale, +cell_scale]`` square
    in 2-D; the target frame is the source frame plus per-cell Gaussian
    jitter of scale ``jitter_scale``.  The association label is the
    identity matrix (cell ``i`` in source matches cell ``i`` in target).

    Parameters
    ----------
    num_cells : int
        ``N`` — number of cells per frame.
    batch_size : int, default 2
        ``B`` — number of independent frame pairs in the batch.
    feat_dim : int, default 7
        Feature dimensionality (matches ``benchmark_ssl``'s
        ``regionprops2`` output).
    coord_dim : int, default 2
        Coordinate dimensionality (x, y).
    cell_scale : float, default 100.0
        Standard deviation of the cell coordinates.
    jitter_scale : float, default 3.0
        Standard deviation of the per-cell displacement.
    device : torch.device, optional
        Move the returned tensors to this device if given.

    Returns
    -------
    dict
        Keys: ``cs``, ``fs``, ``ct``, ``ft``, ``a``, ``ps``, ``pt``.
        All values are :class:`torch.Tensor`.
    """
    generator = torch.Generator().manual_seed(0)
    cs = torch.randn(batch_size, num_cells, coord_dim, generator=generator) * cell_scale
    ct = cs + torch.randn(batch_size, num_cells, coord_dim, generator=generator) * jitter_scale
    fs = torch.randn(batch_size, num_cells, feat_dim, generator=generator)
    ft = torch.randn(batch_size, num_cells, feat_dim, generator=generator)
    a = torch.eye(num_cells).unsqueeze(0).expand(batch_size, -1, -1).float()
    ps = torch.zeros(batch_size, num_cells, dtype=torch.bool)
    pt = torch.zeros(batch_size, num_cells, dtype=torch.bool)
    batch = {
        "cs": cs,
        "fs": fs,
        "ct": ct,
        "ft": ft,
        "a": a,
        "ps": ps,
        "pt": pt,
    }
    if device is not None:
        batch = {key: value.to(device) for key, value in batch.items()}
    return batch


def make_synthetic_pairs(
    num_cells: int,
    num_pairs: int = 200,
    batch_size: int = 2,
) -> List[dict]:
    """Generate a list of synthetic pairs (no batching).

    Each element is a dict identical to the output of
    :func:`make_synthetic_batch` but without a batch dimension.

    Parameters
    ----------
    num_cells : int
        ``N`` — cells per frame.
    num_pairs : int, default 200
        Number of pairs to generate.
    batch_size : int, default 2
        Unused; kept for backward-compatibility with earlier scripts
        that passed it through.

    Returns
    -------
    list[dict]
        Each dict has the same keys as :func:`make_synthetic_batch`
        except with leading batch dimension 1.
    """
    generator = torch.Generator().manual_seed(0)
    pairs = []
    for _ in range(num_pairs):
        cs = torch.randn(num_cells, 2, generator=generator) * 100.0
        ct = cs + torch.randn(num_cells, 2, generator=generator) * 3.0
        fs = torch.randn(num_cells, 7, generator=generator)
        ft = torch.randn(num_cells, 7, generator=generator)
        a = torch.eye(num_cells).float()
        ps = torch.zeros(num_cells, dtype=torch.bool)
        pt = torch.zeros(num_cells, dtype=torch.bool)
        pairs.append({"cs": cs, "fs": fs, "ct": ct, "ft": ft, "a": a, "ps": ps, "pt": pt})
    return pairs


# ------------------------------------------------------------------ #
# Real vanvliet pairs
# ------------------------------------------------------------------ #

def _load_image_normalised(image_path: str) -> np.ndarray:
    """Load and percentile-normalise a microscopy image to ``[0, 1]``."""
    from tifffile import imread  # heavy import; keep local
    image = imread(image_path).astype(np.float32)
    low, high = np.percentile(image, (1, 99.8))
    return np.clip((image - low) / (high - low + 1e-8), 0, 1)


def _extract_pair_from_frames(frames: list, index: int) -> Optional[dict]:
    """Extract a single ``(cs, fs, ct, ft, a)`` dict from adjacent frames.

    Returns ``None`` if the pair cannot be constructed (missing masks,
    empty cells, read errors).  The caller is responsible for skipping
    ``None`` returns.
    """
    from ssl_pipeline import features_from_frame  # heavy import; keep local

    try:
        _, _, _, mask_src_path, image_src_path = frames[index]
        _, _, _, mask_tgt_path, image_tgt_path = frames[index + 1]
    except (IndexError, ValueError):
        return None

    from tifffile import imread
    try:
        mask_src = imread(mask_src_path)
        mask_tgt = imread(mask_tgt_path)
        image_src = _load_image_normalised(image_src_path)
        image_tgt = _load_image_normalised(image_tgt_path)
    except Exception as exc:
        logger.debug("Pair %d: image read failed (%s)", index, exc)
        return None

    src_features = features_from_frame(mask_src, image_src)
    tgt_features = features_from_frame(mask_tgt, image_tgt)
    if src_features is None or tgt_features is None:
        return None

    coords_src, labels_src, features_src_dict = src_features
    coords_tgt, labels_tgt, features_tgt_dict = tgt_features
    if len(labels_src) == 0 or len(labels_tgt) == 0:
        return None

    features_src = np.concatenate(list(features_src_dict.values()), axis=-1).astype(np.float32)
    features_tgt = np.concatenate(list(features_tgt_dict.values()), axis=-1).astype(np.float32)

    label_to_tgt_index = {int(label): tgt_index for tgt_index, label in enumerate(labels_tgt)}
    association = np.zeros((len(labels_src), len(labels_tgt)), dtype=np.float32)
    for src_index, src_label in enumerate(labels_src):
        tgt_match = label_to_tgt_index.get(int(src_label))
        if tgt_match is not None:
            association[src_index, tgt_match] = 1.0

    return {
        "cs": torch.from_numpy(coords_src).float(),
        "ct": torch.from_numpy(coords_tgt).float(),
        "fs": torch.from_numpy(features_src).float(),
        "ft": torch.from_numpy(features_tgt).float(),
        "a": torch.from_numpy(association).float(),
        "ns": len(labels_src),
        "nt": len(labels_tgt),
    }


def create_tracking_pairs(
    frames: list,
    max_pairs: int = 200,
) -> List[dict]:
    """Create tracking pairs from a list of adjacent frames.

    Walks ``frames`` two-at-a-time and builds a list of
    ``(src_features, tgt_features, association_matrix)`` triples.

    Parameters
    ----------
    frames : list
        Output of ``ssl_pipeline.load_experiment_frames``.
    max_pairs : int, default 200
        Stop after this many successfully constructed pairs.

    Returns
    -------
    list[dict]
        One dict per pair, with keys ``cs``, ``fs``, ``ct``, ``ft``,
        ``a``, ``ns``, ``nt``.
    """
    pairs = []
    for index in range(0, len(frames) - 1, 2):
        pair = _extract_pair_from_frames(frames, index)
        if pair is not None:
            pairs.append(pair)
            if len(pairs) >= max_pairs:
                break
    return pairs


# Backward-compat alias used by ``downstream_matched.py``.
load_real_pairs = create_tracking_pairs
