"""Smoke tests for attention modules."""

import pytest
import torch

from src.models.attention import (
    GatherSparseAttention,
    KNNRelativePositionalBias,
    RelativePositionalAttention,
    RotaryPositionalEncoding,
    SpatialReorder,
)


def test_rotary_positional_encoding_shape():
    rope = RotaryPositionalEncoding(cutoffs=(256, 256), n_pos=(32, 32))
    coords = torch.randn(2, 10, 2)
    cos, sin = rope.get_co_si(coords)
    assert cos.shape == sin.shape
    assert cos.dim() == 3


def test_knn_relative_positional_bias_shape():
    bias = KNNRelativePositionalBias(n_head=4, cutoff_spatial=256, cutoff_temporal=16)
    coords = torch.randn(2, 10, 3)
    out = bias(coords)
    assert out.shape == (2, 4, 10, 10)


def test_gather_sparse_attention_shape():
    attn = GatherSparseAttention(embed_dim=64, n_head=4, knn_neighbors=8, mode="none")
    batch_size, seq_len = 2, 10
    query = torch.randn(batch_size, seq_len, 64)
    key = torch.randn(batch_size, seq_len, 64)
    value = torch.randn(batch_size, seq_len, 64)
    knn_indices = torch.randint(0, seq_len, (batch_size, seq_len, 8))
    out = attn(query, key, value, knn_indices)
    assert out.shape == (batch_size, seq_len, 64)


def test_relative_positional_attention_shape():
    attn = RelativePositionalAttention(coord_dim=3, embed_dim=64, n_head=4, mode="bias")
    query = torch.randn(2, 10, 64)
    key = torch.randn(2, 10, 64)
    value = torch.randn(2, 10, 64)
    coords = torch.randn(2, 10, 3)
    out = attn(query, key, value, coords)
    assert out.shape == (2, 10, 64)


def test_spatial_reorder_indices():
    reorder = SpatialReorder(n_bins=16)
    coords = torch.randn(2, 20, 2)
    reorder_idx, unreorder_idx = reorder.compute_idx(coords)
    assert reorder_idx.shape == (2, 20)
    assert unreorder_idx.shape == (2, 20)
    assert torch.allclose(reorder_idx.gather(1, reorder_idx.argsort(1)), torch.arange(20).expand(2, -1))
