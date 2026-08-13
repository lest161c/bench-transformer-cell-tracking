"""Tests for the edge-probing harness: registries, probes, datasets, training."""

import numpy as np
import pytest
import torch

from src.edge_probing.harness import (
    FEATURE_REGISTRY,
    PROBE_REGISTRY,
    FEATURE_ALIASES,
    EdgePairDataset,
    EdgePairDatasetPatches,
    LinearProbe,
    MLPProbe,
    CNNProbeE2E,
    compute_metrics,
    train_probe,
    list_features,
    list_probes,
)


# -- Registry tests --

def test_feature_registry_has_all_types():
    """All 5 feature types are registered."""
    expected = {"rp", "cnn_frozen", "cnn_e2e", "dino", "hoct19"}
    assert set(FEATURE_REGISTRY.keys()) == expected


def test_probe_registry_has_linear_and_mlp():
    """Both probe architectures are registered."""
    assert set(PROBE_REGISTRY.keys()) == {"linear", "mlp"}


def test_feature_aliases():
    """Feature aliases map to the correct feature lists."""
    assert set(FEATURE_ALIASES["all"]) == set(FEATURE_REGISTRY.keys())
    assert FEATURE_ALIASES["shallow"] == ["rp"]
    assert set(FEATURE_ALIASES["deep"]) == {"cnn_frozen", "cnn_e2e", "dino"}


def test_list_features_and_probes():
    """list_features and list_probes return the registry keys."""
    assert list_features() == list(FEATURE_REGISTRY.keys())
    assert list_probes() == list(PROBE_REGISTRY.keys())


# -- Probe architecture tests --

def test_linear_probe_forward():
    """LinearProbe produces (N,) logits from (N, D) pairs."""
    probe = LinearProbe(feat_dim=7)
    feat_t = torch.randn(4, 7)
    feat_n = torch.randn(4, 7)
    scores = probe(feat_t, feat_n)
    assert scores.shape == (4,)


def test_mlp_probe_forward():
    """MLPProbe produces (N,) logits from (N, D) pairs."""
    probe = MLPProbe(feat_dim=7, hidden=128)
    feat_t = torch.randn(4, 7)
    feat_n = torch.randn(4, 7)
    scores = probe(feat_t, feat_n)
    assert scores.shape == (4,)


def test_cnn_probe_e2e_forward():
    """CNNProbeE2E produces (N,) logits from (N, 1, 64, 64) patch pairs."""
    probe = CNNProbeE2E(scale="small", out_dim=64, probe_type="linear")
    patches_t = torch.randn(2, 1, 64, 64)
    patches_n = torch.randn(2, 1, 64, 64)
    scores = probe(patches_t, patches_n)
    assert scores.shape == (2,)


# -- Edge pair dataset tests --

def test_edge_pair_dataset():
    """EdgePairDataset flattens (N, M) target into N*M pairs."""
    feat_t = torch.randn(5, 7)
    feat_n = torch.randn(4, 7)
    target = torch.zeros(5, 4)
    target[0, 0] = 1.0
    target[1, 1] = 1.0
    ds = EdgePairDataset(feat_t, feat_n, target)
    assert len(ds) == 20
    ft, fn, lbl = ds[0]
    assert ft.shape == (7,)
    assert fn.shape == (7,)
    assert lbl.item() == 1.0


def test_edge_pair_dataset_patches():
    """EdgePairDatasetPatches stores patches for cnn_e2e training."""
    patches_t = torch.randn(3, 1, 64, 64)
    patches_n = torch.randn(2, 1, 64, 64)
    target = torch.zeros(3, 2)
    target[0, 0] = 1.0
    ds = EdgePairDatasetPatches(patches_t, patches_n, target)
    assert len(ds) == 6


# -- Metrics tests --

def test_compute_metrics_perfect():
    """compute_metrics returns 1.0 for perfect predictions."""
    scores = torch.tensor([2.0, -1.0, 0.5, -0.5])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    bal_acc, f1, prec, rec = compute_metrics(scores, targets)
    assert bal_acc == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)


def test_compute_metrics_random():
    """compute_metrics returns ~0.5 for random predictions."""
    rng = np.random.RandomState(42)
    scores = torch.from_numpy(rng.randn(100)).float()
    targets = torch.from_numpy(rng.randint(0, 2, 100)).float()
    bal_acc, f1, prec, rec = compute_metrics(scores, targets)
    assert 0.3 < bal_acc < 0.7


# -- Training loop tests --

def test_train_probe_runs():
    """train_probe runs for a few epochs and returns a metrics dict."""
    feat_dim = 7
    probe = LinearProbe(feat_dim)

    # Create dummy train/val loaders
    feat_t = torch.randn(20, feat_dim)
    feat_n = torch.randn(20, feat_dim)
    target = torch.zeros(20, 20)
    for i in range(20):
        target[i, i] = 1.0

    train_ds = EdgePairDataset(feat_t, feat_n, target)
    val_ds = EdgePairDataset(feat_t[:10], feat_n[:10], target[:10, :10])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

    result = train_probe(
        probe, train_loader, val_loader,
        epochs=3, lr=1e-3, patience=10, eval_every=1,
    )

    assert "final_bal_acc" in result
    assert "final_f1" in result
    assert isinstance(result["final_bal_acc"], float)
