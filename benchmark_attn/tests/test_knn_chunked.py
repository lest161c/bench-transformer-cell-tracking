"""Tests for the chunked KNN implementation and harness allocation isolation.

The tests compare the memory-bounded :func:`compute_knn_indices` against the
dense reference on random no-tie coordinates, and verify that
``build_closure`` does not allocate a dense distance matrix for a gather
method that never consumes it.
"""

import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT / "scripts" / "benchmarks"))

import pytest
import torch

import benchmark_sweep
from src.bench.data import compute_knn_indices, compute_knn_indices_dense


@pytest.mark.parametrize("seq_len", [64, 256])
@pytest.mark.parametrize("knn_neighbors", [4, 16])
@pytest.mark.parametrize("query_chunk", [32, 64])
def test_chunked_knn_matches_dense(seq_len, knn_neighbors, query_chunk):
    """Chunked top-K is exactly equal to the dense reference on no-tie data.

    Args:
        seq_len: Number of tokens per batch element.
        knn_neighbors: Number of nearest neighbors to select.
        query_chunk: Number of queries processed per block.
    """
    batch_size = 2
    generator = torch.Generator().manual_seed(1234)
    # Three columns = leading non-spatial column + two spatial coordinates.
    coords = torch.randn(batch_size, seq_len, 3, generator=generator)

    chunked_indices = compute_knn_indices(
        coords, knn_neighbors, query_chunk=query_chunk
    )
    dense_indices = compute_knn_indices_dense(coords, knn_neighbors)

    assert chunked_indices.shape == (batch_size, seq_len, knn_neighbors)
    assert chunked_indices.dtype == torch.int64
    assert dense_indices.dtype == torch.int64
    assert torch.equal(chunked_indices, dense_indices)


def test_gather_closure_with_dist_cache_does_not_call_cdist(monkeypatch):
    """A gather closure with dist_cache='on' must not allocate an N x N matrix.

    ``sparse_gather_sdpa`` needs KNN indices, so the production
    ``compute_knn_indices`` is patched out; the only remaining potential
    ``torch.cdist`` caller would be the unused 2D distance matrix that H1
    removed. The closure is both built and run so a later regression inside
    the gather forward would also be caught.

    Args:
        monkeypatch: pytest fixture for replacing module attributes.
    """

    def fail_if_cdist_called(*args, **kwargs):
        """Fail the test if any code path allocates a distance matrix."""
        raise AssertionError(
            "torch.cdist was called for a gather method with dist_cache='on'; "
            "the dense distance matrix must only be allocated for masked methods"
        )

    def fake_knn_indices(coords, knn_neighbors):
        """Return valid-shaped dummy KNN indices without calling cdist."""
        batch_size, seq_len = coords.shape[:2]
        return torch.zeros(
            (batch_size, seq_len, knn_neighbors), dtype=torch.int64
        )

    monkeypatch.setattr(torch, "cdist", fail_if_cdist_called)
    monkeypatch.setattr(
        benchmark_sweep, "compute_knn_indices", fake_knn_indices
    )

    closure = benchmark_sweep.build_closure(
        method_key="sparse_gather_sdpa",
        layer_count=1,
        seq_len=32,
        knn_neighbors=4,
        batch_size=2,
        d_model=32,
        n_head=4,
        coord_dim=2,
        mode="none",
        dist_mode="v1",
        device=torch.device("cpu"),
        dtype=torch.float32,
        seed=42,
        dist_cache="on",
    )

    with torch.no_grad():
        closure()


def test_inside_closure_knn_uses_shared_batch_one_graph(monkeypatch):
    """With KNN inside the closure, it still runs on one shared batch graph.

    Contract B applies to both the precomputed path and the training-realistic
    ``with_knn``/``dist_cache='off'`` path: the KNN helper must receive only
    the first batch element's coordinates and return a ``(1, N, K)`` graph
    that ``build_closure`` expands to the batch.

    Args:
        monkeypatch: pytest fixture for replacing the KNN helper.
    """
    captured_coords = []

    def capture_knn_coords(coords, knn_neighbors):
        """Record the KNN input shape and return real indices for it."""
        captured_coords.append(coords)
        return compute_knn_indices(coords, knn_neighbors)

    monkeypatch.setattr(
        benchmark_sweep, "compute_knn_indices", capture_knn_coords
    )

    closure = benchmark_sweep.build_closure(
        method_key="sparse_gather_sdpa",
        layer_count=1,
        seq_len=32,
        knn_neighbors=4,
        batch_size=2,
        d_model=32,
        n_head=4,
        coord_dim=2,
        mode="none",
        dist_mode="v1",
        device=torch.device("cpu"),
        dtype=torch.float32,
        seed=42,
        dist_cache="off",
        with_knn=True,
    )

    assert captured_coords == []
    with torch.no_grad():
        closure()

    assert len(captured_coords) == 1
    assert captured_coords[0].shape[0] == 1
