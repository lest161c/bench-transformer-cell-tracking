"""Tests for contract B: one shared spatial graph across the batch.

Contract B makes the benchmark use a single batch-1 spatial graph for every
batch element so Gather and Flex receive the same KNN/block-mask setup. The
Flex block mask is therefore built once with ``B=1, H=1`` and reused, rather
than materialising a ``(B, H, N, N)`` mask. These tests run on CPU to keep the
unit suite fast and GPU-free; they skip on torch builds without
``torch.nn.attention.flex_attention`` (torch < 2.5).
"""

import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT / "scripts" / "benchmarks"))

import pytest
import torch

pytest.importorskip(
    "torch.nn.attention.flex_attention",
    reason="FlexAttention requires torch>=2.5",
)

import benchmark_sweep
import src.attention_modules as attention_modules
from src.attention_modules import FlexAttentionModule
from src.bench.data import compute_knn_indices

BATCH_SIZE = 2
SEQ_LEN = 64
N_HEAD = 4
D_MODEL = 32
CUTOFF_SPATIAL = 1.0


def _make_module(cutoff_spatial: float = CUTOFF_SPATIAL) -> FlexAttentionModule:
    """Create a deterministic CPU FlexAttention module for the tests.

    Args:
        cutoff_spatial: Spatial distance cutoff used by the mask/score mods.

    Returns:
        Freshly initialised :class:`FlexAttentionModule` on CPU.
    """
    torch.manual_seed(0)
    return FlexAttentionModule(
        embed_dim=D_MODEL,
        n_head=N_HEAD,
        cutoff_spatial=cutoff_spatial,
    ).to("cpu")


def _make_shared_coords() -> torch.Tensor:
    """Create one tightly clustered spatial graph expanded to the batch.

    Every spatial coordinate is within ``[0, 0.5]`` so all pairs fall inside
    the default cutoff and the forward pass has a well-defined softmax.

    Returns:
        Coordinate tensor ``(batch_size, seq_len, 3)`` whose batch entries are
        identical views of the same single-batch graph.
    """
    generator = torch.Generator().manual_seed(1234)
    shared_coords = torch.rand(1, SEQ_LEN, 3, generator=generator)
    shared_coords[0, :, 1:] *= 0.5
    return shared_coords.expand(BATCH_SIZE, -1, -1)


def _assert_no_dense_pairwise_tensor(module: FlexAttentionModule) -> None:
    """Fail if the module stores a dense ``(B, N, N)`` tensor.

    Args:
        module: Module whose direct tensor attributes are inspected.
    """
    dense_shape = (BATCH_SIZE, SEQ_LEN, SEQ_LEN)
    for attribute_name, attribute_value in vars(module).items():
        if isinstance(attribute_value, torch.Tensor):
            assert tuple(attribute_value.shape) != dense_shape, (
                f"module attribute {attribute_name!r} stores a dense "
                f"{dense_shape} tensor; distances must be computed on the fly"
            )


def _build_gather_closure(seq_len: int, knn_neighbors: int):
    """Build a one-layer gather closure used by the shared-graph test.

    Args:
        seq_len: Sequence length.
        knn_neighbors: Number of KNN neighbours.

    Returns:
        The closure returned by :func:`benchmark_sweep.build_closure`.
    """
    return benchmark_sweep.build_closure(
        method_key="sparse_gather_sdpa",
        layer_count=1,
        seq_len=seq_len,
        knn_neighbors=knn_neighbors,
        batch_size=BATCH_SIZE,
        d_model=D_MODEL,
        n_head=N_HEAD,
        coord_dim=2,
        mode="none",
        dist_mode="v1",
        device=torch.device("cpu"),
        dtype=torch.float32,
        seed=42,
        dist_cache="on",
    )


def test_shared_graph_coords_are_identical_across_batch(monkeypatch):
    """``build_closure`` keeps batch 0's graph and expands it to the batch.

    The synthetic inputs are given visibly distinct per-batch coordinate
    clouds. If the harness still consumed per-batch graphs, the KNN call would
    receive ``B=2`` coordinates and the forward would see differing batch
    entries. Contract B instead computes KNN on ``coords[0:1]`` and passes a
    single-graph ``expand`` view to every layer.

    Args:
        monkeypatch: pytest fixture for wrapping harness functions.
    """
    seq_len = 32
    knn_neighbors = 4
    captured = {}

    original_make_inputs = benchmark_sweep.make_inputs
    original_forward_layer = benchmark_sweep._forward_layer

    def make_inputs_with_distinct_batches(*args, **kwargs):
        """Return coords whose batch 1 cloud is shifted far from batch 0."""
        query, coords = original_make_inputs(*args, **kwargs)
        coords = coords.clone()
        coords[1, :, 1:] += 100.0
        captured["input_coords"] = coords
        return query, coords

    def capture_knn_input(coords, knn_neighbors):
        """Record the coordinates passed to the KNN helper."""
        captured["knn_coords"] = coords
        return compute_knn_indices(coords, knn_neighbors)

    def capture_forward(layer, x, coords, knn_idx, forward_style, dist_2d=None):
        """Record the coordinates and indices passed to the layer."""
        captured["forward_coords"] = coords
        captured["forward_knn_idx"] = knn_idx
        return original_forward_layer(
            layer, x, coords, knn_idx, forward_style, dist_2d
        )

    monkeypatch.setattr(
        benchmark_sweep, "make_inputs", make_inputs_with_distinct_batches
    )
    monkeypatch.setattr(
        benchmark_sweep, "compute_knn_indices", capture_knn_input
    )
    monkeypatch.setattr(benchmark_sweep, "_forward_layer", capture_forward)

    closure = _build_gather_closure(seq_len, knn_neighbors)
    with torch.no_grad():
        closure()

    input_coords = captured["input_coords"]
    knn_coords = captured["knn_coords"]
    assert knn_coords.shape[0] == 1
    assert torch.equal(knn_coords, input_coords[0:1])

    forward_coords = captured["forward_coords"]
    expected_forward_coords = input_coords[0:1].expand(BATCH_SIZE, -1, -1)
    assert forward_coords.shape == (
        BATCH_SIZE,
        seq_len,
        input_coords.shape[-1],
    )
    assert forward_coords.stride(0) == 0
    assert torch.equal(forward_coords, expected_forward_coords)

    forward_knn_idx = captured["forward_knn_idx"]
    assert forward_knn_idx.shape == (BATCH_SIZE, seq_len, knn_neighbors)
    assert torch.equal(forward_knn_idx[0], forward_knn_idx[1])


def test_flex_shared_block_mask_uses_b1_h1(monkeypatch):
    """The Flex block mask is requested once with ``B=1, H=1``.

    Per-batch masks would make the stock builder materialise a
    ``(B, H, N, N)`` boolean tensor; contract B builds a single shared mask
    sized only by ``N``.

    Args:
        monkeypatch: pytest fixture for replacing ``create_block_mask``.
    """
    captured = {}
    sentinel_block_mask = object()

    def fake_create_block_mask(
        mask_mod, B, H, Q_LEN, KV_LEN, device=None, **kwargs
    ):
        """Record block-mask arguments and return a sentinel."""
        captured.update(
            mask_mod=mask_mod,
            B=B,
            H=H,
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            device=device,
        )
        return sentinel_block_mask

    monkeypatch.setattr(
        attention_modules, "create_block_mask", fake_create_block_mask
    )

    shared_coords = torch.zeros(1, SEQ_LEN, 3)
    shared_coords[0, : SEQ_LEN // 2, 1:] = 10.0
    shared_coords[0, SEQ_LEN // 2 :, 1:] = 100.0
    coords = shared_coords.expand(BATCH_SIZE, -1, -1)

    module = _make_module()
    module._build_block_mask(coords)

    assert captured["B"] == 1
    assert captured["H"] == 1
    assert captured["Q_LEN"] == SEQ_LEN
    assert captured["KV_LEN"] == SEQ_LEN
    assert module._block_mask is sentinel_block_mask

    mask_mod = captured["mask_mod"]
    same_cluster = mask_mod(
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor(0),
    )
    cross_cluster_batch_zero = mask_mod(
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor(SEQ_LEN // 2),
    )
    cross_cluster_batch_one = mask_mod(
        torch.tensor(1),
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor(SEQ_LEN // 2),
    )
    assert bool(same_cluster) is True
    assert bool(cross_cluster_batch_zero) is False
    assert bool(cross_cluster_batch_one) is False


def test_prepare_then_forward_shared_graph():
    """Prepare is idempotent and forward runs all batches on one graph.

    The token queries differ per batch element while the spatial graph is
    shared. The output must be finite, and the module must not store a dense
    ``(B, N, N)`` distance tensor.
    """
    coords = _make_shared_coords()
    module = _make_module()

    module.prepare(coords)
    block_mask = module._block_mask
    compiled_flex = module._compiled_flex
    assert block_mask is not None
    assert compiled_flex is not None

    module.prepare(coords)
    assert module._block_mask is block_mask
    assert module._compiled_flex is compiled_flex
    _assert_no_dense_pairwise_tensor(module)

    generator = torch.Generator().manual_seed(7)
    query = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, generator=generator)
    key = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, generator=generator)
    value = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, generator=generator)

    # FlexAttention has no CPU backward; this test checks the forward shape
    # and finiteness only, so run without autograd.
    with torch.no_grad():
        output = module(query, key, value, coords=coords)

    assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
    assert torch.isfinite(output).all()
