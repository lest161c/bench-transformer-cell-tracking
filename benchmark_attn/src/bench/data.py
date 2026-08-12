"""Shared input generation and KNN computation helpers for the benchmark harness.

Extracted from the duplicated ``knn_indices()`` helpers and inline input
creation code in ``benchmarks/benchmark_full.py``,
``benchmarks/benchmark_gather_v3.py`` and ``benchmarks/benchmark_cached_dist.py``.

The input layout is kept exactly as in those scripts: the coordinate tensor
has ``coord_dim + 1`` columns whose first (non-spatial) column is scaled by 4,
and the KNN computation drops that first column before computing pairwise
distances.
"""

import torch


def make_inputs(batch_size: int, seq_len: int, d_model: int, coord_dim: int,
                device: torch.device, dtype: torch.dtype, seed: int = 42
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic (query, coords) tensors for benchmarking.

    Sets ``torch.manual_seed(seed)`` and then draws the query as a standard
    normal tensor of shape (batch_size, seq_len, d_model) and the coords as a
    standard normal tensor of shape (batch_size, seq_len, coord_dim + 1) whose
    first column is multiplied by 4 (matching the existing scripts, where the
    leading column is a non-spatial scale and the trailing ``coord_dim``
    columns carry the spatial coordinates consumed by
    :func:`compute_knn_indices`).

    Args:
        batch_size: Batch dimension.
        seq_len: Sequence length.
        d_model: Embedding dimension.
        coord_dim: Number of spatial coordinate columns (excluding the leading
            scaled column).
        device: torch device on which to allocate the tensors.
        dtype: torch dtype of the tensors.
        seed: Random seed passed to ``torch.manual_seed`` before generation.

    Returns:
        Tuple (query, coords) with shapes (batch_size, seq_len, d_model) and
        (batch_size, seq_len, coord_dim + 1).
    """
    torch.manual_seed(seed)
    query = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    coords = torch.randn(batch_size, seq_len, coord_dim + 1, device=device, dtype=dtype)
    coords[..., 0] *= 4
    return query, coords


def compute_knn_indices(coords: torch.Tensor, knn_neighbors: int) -> torch.Tensor:
    """Compute K-nearest-neighbor indices from spatial coordinates.

    Identical to the ``knn_indices()`` helpers in the existing scripts: drops
    the first (non-spatial) coordinate column, computes pairwise distances
    with ``torch.cdist`` in float32, and keeps the ``knn_neighbors`` closest
    tokens per token via ``torch.topk``.

    Args:
        coords: Coordinate tensor of shape (batch_size, seq_len, coord_dim + 1).
        knn_neighbors: Number of nearest neighbors to select.

    Returns:
        KNN index tensor of shape (batch_size, seq_len, knn_neighbors).
    """
    yx = coords[..., 1:].float()
    dist = torch.cdist(yx, yx)
    _, knn = torch.topk(dist, k=knn_neighbors, dim=-1, largest=False)
    return knn
