"""Smoke tests for CellEmbedder."""

import pytest
import torch

from src.models import CellEmbedder


def test_cell_embedder_forward_shape():
    embedder = CellEmbedder(feat_dim=7, coord_dim=2, d_model=128)
    batch_size, n_cells = 2, 10
    coords = torch.randn(batch_size, n_cells, 2)
    features = torch.randn(batch_size, n_cells, 7)
    embeddings = embedder(coords, features)
    assert embeddings.shape == (batch_size, n_cells, 128)


def test_cell_embedder_encode_normalizes():
    embedder = CellEmbedder(feat_dim=7, coord_dim=2, d_model=64)
    coords = torch.randn(1, 5, 2)
    features = torch.randn(1, 5, 7)
    encoded = embedder.encode(coords, features)
    norms = torch.norm(encoded, dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_cell_embedder_with_padding_mask():
    embedder = CellEmbedder(feat_dim=7, coord_dim=2, d_model=64, num_layers=2)
    coords = torch.randn(1, 6, 2)
    features = torch.randn(1, 6, 7)
    padding_mask = torch.tensor([[False, False, False, True, True, True]])
    embeddings = embedder(coords, features, padding_mask=padding_mask)
    assert embeddings.shape == (1, 6, 64)
