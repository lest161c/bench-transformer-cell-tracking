"""Probe architectures for edge classification.

Three probe heads that take ``(feat_t, feat_n)`` pairs and produce a
single edge logit per pair:

- ``LinearProbe`` — single linear layer over the concatenated
  pair features.
- ``MLPProbe`` — 2-layer MLP (default hidden=128).
- ``CNNProbeE2E`` — ``ScaledCNN`` + linear probe, trained jointly
  on raw patches (the CNN learns its features end-to-end).
"""

import torch
import torch.nn as nn

from src.models.scaled_cnn import ScaledCNN


class LinearProbe(nn.Module):
    """Linear probe: concat(feat_t[i], feat_n[j]) -> score."""
    def __init__(self, feat_dim):
        """Initialize the linear probe.

        Args:
            feat_dim: dimensionality of each cell's feature vector; the layer
                maps the concatenation of the two cells' features (2*feat_dim)
                to a single logit.
        """
        super().__init__()
        self.fc = nn.Linear(2 * feat_dim, 1)

    def forward(self, feat_t, feat_n):
        """feat_t: (N, D), feat_n: (N, D) -> scores: (N,)"""
        pairs = torch.cat([feat_t, feat_n], dim=-1)  # (N, 2D)
        return self.fc(pairs).squeeze(-1)  # (N,)


class MLPProbe(nn.Module):
    """2-layer MLP probe: concat(feat_t[i], feat_n[j]) -> hidden -> score."""
    def __init__(self, feat_dim, hidden=128):
        """Initialize the 2-layer MLP probe.

        Args:
            feat_dim: dimensionality of each cell's feature vector; the input
                layer maps the concatenation of the two cells' features
                (2*feat_dim) to `hidden` units.
            hidden: number of hidden units between the two linear layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat_t, feat_n):
        """Score a batch of cell-cell pairs.

        Args:
            feat_t: (N, feat_dim) features of cells in frame t.
            feat_n: (N, feat_dim) features of cells in frame t+1.

        Returns:
            (N,) logit scores, one per pair.
        """
        pairs = torch.cat([feat_t, feat_n], dim=-1)  # (N, 2D)
        return self.net(pairs).squeeze(-1)  # (N,)


class CNNProbeE2E(nn.Module):
    """ScaledCNN + probe, trained end-to-end."""
    def __init__(self, scale='large', out_dim=128, probe_type='linear'):
        """Initialize the end-to-end CNN + probe model.

        Args:
            scale: ScaledCNN architecture size ('small'/'medium'/'large').
            out_dim: dimensionality of the CNN patch embedding.
            probe_type: 'linear' for a single linear layer or 'mlp' for a
                2-layer MLP on the concatenated embedding pair.
        """
        super().__init__()
        self.cnn = ScaledCNN(scale=scale, out_dim=out_dim)
        if probe_type == 'linear':
            self.probe = nn.Linear(2 * out_dim, 1)
        else:
            self.probe = nn.Sequential(
                nn.Linear(2 * out_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

    def forward(self, patches_t, patches_n):
        """patches_t: (N, 1, 64, 64), patches_n: (N, 1, 64, 64) -> scores: (N,)"""
        feat_t = self.cnn(patches_t)  # (N, D)
        feat_n = self.cnn(patches_n)  # (N, D)
        pairs = torch.cat([feat_t, feat_n], dim=-1)  # (N, 2D)
        return self.probe(pairs).squeeze(-1)  # (N,)
