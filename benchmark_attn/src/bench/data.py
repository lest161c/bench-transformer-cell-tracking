"""Shared input generation and KNN computation helpers for the benchmark harness.

The input layout is kept exactly as in the benchmark scripts: the coordinate
tensor has ``coord_dim + 1`` columns whose first (non-spatial) column is
scaled by 4, and the KNN computation drops that first column before
computing pairwise distances.
"""

import math

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


def _make_spatial_coordinates(
    seq_len: int,
    window: int,
    coord_layout: str,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create the two spatial coordinate columns for one batch element.

    The square side is ``10 * sqrt(ceil(seq_len / window))`` so the nominal
    spatial density stays roughly constant as ``seq_len`` grows. The
    ``uniform`` layout draws independent positions from a seeded generator;
    ``tiled`` builds a deterministic jittered lattice with the same side so
    its points remain spatially comparable to the uniform layout.

    Args:
        seq_len: Number of tokens in the synthetic sequence.
        window: Number of time frames over which tokens are distributed.
        coord_layout: Either ``"uniform"`` or ``"tiled"``.
        generator: Seeded CPU generator controlling all random draws.
        dtype: Floating dtype of the returned coordinates.

    Returns:
        Spatial coordinate tensor of shape ``(seq_len, 2)``.

    Raises:
        ValueError: If ``coord_layout`` is not ``"uniform"`` or ``"tiled"``.
    """
    side_length = 10.0 * math.sqrt(math.ceil(seq_len / window))

    if coord_layout == "uniform":
        return torch.rand(
            (seq_len, 2), generator=generator, dtype=dtype
        ) * side_length

    if coord_layout == "tiled":
        points_per_frame = math.ceil(seq_len / window)
        lattice_size = math.ceil(math.sqrt(points_per_frame))
        spacing = side_length / lattice_size
        axis_centers = (
            torch.arange(lattice_size, dtype=dtype) + 0.5
        ) * spacing
        grid_y, grid_x = torch.meshgrid(
            axis_centers, axis_centers, indexing="ij"
        )
        lattice_centers = torch.stack(
            (grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1
        )[:points_per_frame]
        jitter = (
            torch.rand(
                lattice_centers.shape, generator=generator, dtype=dtype
            )
            - 0.5
        ) * (0.5 * spacing)
        jittered_lattice = lattice_centers + jitter
        lattice_indices = torch.arange(seq_len) % points_per_frame
        return jittered_lattice[lattice_indices]

    raise ValueError(
        f"Unknown coord_layout {coord_layout!r}; "
        "expected 'uniform' or 'tiled'."
    )


def make_full_model_inputs(
    batch_size: int,
    seq_len: int,
    *,
    window: int = 10,
    knn_neighbors: int = 16,
    coord_layout: str = "uniform",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
    query_chunk: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the synthetic inputs consumed by the full Trackastra model.

    One batch element is generated deterministically from ``seed`` and then
    replicated when ``batch_size > 1``, so batch scaling measures the same
    synthetic sample rather than different random data. The leading
    coordinate column is the time frame (``torch.arange(seq_len) % window``);
    the trailing two columns are spatial coordinates generated by
    :func:`_make_spatial_coordinates`. Features are standard-normal
    ``wrfeat``-shaped vectors. KNN indices are computed once with
    :func:`compute_knn_indices` on the batch-one sample and broadcast to the
    requested batch.

    Random draws happen on CPU through an explicit ``torch.Generator`` and
    the finished tensors are moved to ``device``, so a given ``seed``
    produces identical values regardless of the selected device. KNN
    construction uses only the spatial columns and includes each query
    itself, matching the existing benchmark helper.

    Args:
        batch_size: Batch dimension of the returned tensors. Must be positive.
        seq_len: Number of tokens per batch element. Must be at least
            ``knn_neighbors``.
        window: Number of time frames; the time column cycles over
            ``[0, window)``.
        knn_neighbors: Number of nearest neighbours stored per query.
        coord_layout: Spatial layout, either ``"uniform"`` or ``"tiled"``.
        device: torch device for the returned tensors.
        dtype: Floating dtype for coordinates and features. Timepoints and
            KNN indices are always ``int64``.
        seed: Seed for the CPU ``torch.Generator`` used for spatial
            coordinates and features.
        query_chunk: Query block size forwarded to
            :func:`compute_knn_indices`.

    Returns:
        Tuple ``(coords, features, timepoints, knn_indices)`` with shapes
        ``(batch_size, seq_len, 3)``, ``(batch_size, seq_len, 7)``,
        ``(batch_size, seq_len)`` and
        ``(batch_size, seq_len, knn_neighbors)`` respectively. ``coords`` and
        ``features`` use ``dtype``; ``timepoints`` and ``knn_indices`` are
        ``int64``. For ``batch_size > 1`` the KNN tensor is an expanded
        broadcast view of the batch-one indices.

    Raises:
        ValueError: If ``batch_size`` is not positive, ``seq_len`` is smaller
            than ``knn_neighbors``, ``window`` is not positive, or
            ``coord_layout`` is unknown.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if seq_len < knn_neighbors:
        raise ValueError(
            f"seq_len ({seq_len}) must be at least knn_neighbors "
            f"({knn_neighbors})"
        )
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")

    generator = torch.Generator().manual_seed(seed)
    time_column = (torch.arange(seq_len) % window).to(dtype=dtype)
    spatial_coords = _make_spatial_coordinates(
        seq_len, window, coord_layout, generator, dtype
    )
    coords_batch_one = torch.cat(
        (time_column.unsqueeze(-1), spatial_coords), dim=-1
    ).unsqueeze(0)

    features_batch_one = torch.randn(
        (1, seq_len, 7), generator=generator, dtype=dtype
    )
    timepoints_batch_one = coords_batch_one[..., 0].long()
    knn_indices_batch_one = compute_knn_indices(
        coords_batch_one, knn_neighbors, query_chunk=query_chunk
    )

    coords = coords_batch_one.repeat(batch_size, 1, 1).to(device)
    features = features_batch_one.repeat(batch_size, 1, 1).to(device)
    timepoints = timepoints_batch_one.repeat(batch_size, 1).to(device)
    knn_indices = knn_indices_batch_one.to(device).expand(
        batch_size, -1, -1
    )
    return coords, features, timepoints, knn_indices
