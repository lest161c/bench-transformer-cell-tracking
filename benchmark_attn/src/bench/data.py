"""Shared input generation and KNN computation helpers for the benchmark harness.

The input layout is kept exactly as in the benchmark scripts: the coordinate
tensor has ``coord_dim + 1`` columns whose first (non-spatial) column is
scaled by 4, and the KNN computation drops that first column before
computing pairwise distances.
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


def compute_knn_indices(coords: torch.Tensor, knn_neighbors: int,
                        query_chunk: int = 512) -> torch.Tensor:
    """Compute exact K-nearest-neighbor indices in query blocks.

    Drops the first (non-spatial) coordinate column, casts the remaining
    coordinates to float32, and processes ``query_chunk`` queries at a time.
    For each block it materialises only a ``(B, chunk, N)`` distance block
    with ``torch.cdist`` and selects the ``knn_neighbors`` closest keys with
    ``torch.topk``. The selected indices are written into an ``int64``
    ``(B, N, K)`` output, yielding the same nearest-neighbor selection as the
    dense ``compute_knn_indices_dense`` implementation on no-tie data.

    The memory advantage is the reason this function exists: the working
    distance block is ``O(B * query_chunk * N)`` instead of the dense
    ``O(B * N * N)`` distance matrix, which is what previously caused large-N
    OOMs. Chunking only blocks the query dimension, so the computation is
    exact: every query is still compared against all ``N`` keys.

    Args:
        coords: Coordinate tensor of shape (batch_size, seq_len, coord_dim + 1).
        knn_neighbors: Number of nearest neighbors to select per query.
        query_chunk: Number of queries processed per block. Must be positive;
            values larger than ``seq_len`` behave like a single dense block.

    Returns:
        KNN index tensor of dtype ``int64`` and shape
        (batch_size, seq_len, knn_neighbors).

    Raises:
        ValueError: If ``query_chunk`` is not positive.
    """
    if query_chunk <= 0:
        raise ValueError(f"query_chunk must be positive, got {query_chunk}")

    spatial_coords = coords[..., 1:].float()
    batch_size, seq_len, _ = spatial_coords.shape
    knn_indices = torch.empty(
        (batch_size, seq_len, knn_neighbors),
        dtype=torch.int64,
        device=spatial_coords.device,
    )
    for query_start in range(0, seq_len, query_chunk):
        query_end = min(query_start + query_chunk, seq_len)
        distance_block = torch.cdist(
            spatial_coords[:, query_start:query_end, :], spatial_coords
        )
        _, block_indices = torch.topk(
            distance_block, k=knn_neighbors, dim=-1, largest=False
        )
        knn_indices[:, query_start:query_end, :] = block_indices
    return knn_indices


def compute_knn_indices_dense(coords: torch.Tensor,
                              knn_neighbors: int) -> torch.Tensor:
    """Compute K-nearest-neighbor indices with the dense reference algorithm.

    Kept as the reference implementation for tests only: it materialises the
    full ``O(B * N * N)`` distance matrix with ``torch.cdist`` and then runs
    ``torch.topk``. Production callers must use
    :func:`compute_knn_indices`, which is exact but memory-bounded.

    Args:
        coords: Coordinate tensor of shape (batch_size, seq_len, coord_dim + 1).
        knn_neighbors: Number of nearest neighbors to select per query.

    Returns:
        KNN index tensor of dtype ``int64`` and shape
        (batch_size, seq_len, knn_neighbors).
    """
    spatial_coords = coords[..., 1:].float()
    distance_matrix = torch.cdist(spatial_coords, spatial_coords)
    _, knn_indices = torch.topk(
        distance_matrix, k=knn_neighbors, dim=-1, largest=False
    )
    return knn_indices
