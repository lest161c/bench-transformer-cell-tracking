"""Smoke tests for ScaledCNN."""

import pytest
import torch

from src.models import ScaledCNN


def test_scaledcnn_small_forward_shape():
    model = ScaledCNN(scale="small", out_dim=128)
    batch_size = 4
    x = torch.randn(batch_size, 1, 64, 64)
    out = model(x)
    assert out.shape == (batch_size, 128)


def test_scaledcnn_medium_forward_shape():
    model = ScaledCNN(scale="medium", out_dim=128)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 128)


def test_scaledcnn_large_forward_shape():
    model = ScaledCNN(scale="large", out_dim=128)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 128)
