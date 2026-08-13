"""Shared model components for benchmark_training.

This module collects the positional-encoding, normalization, encoder,
and KNN-helper classes that were previously duplicated across the
benchmark scripts.

The classes here are thin wrappers around PyTorch primitives and the
``GatherSparseAttention`` imported from ``benchmark_attn``.  They exist
so that every benchmark script can ``from src.models import ...``
instead of re-defining the same code.

Expected import path (from a script under ``scripts/``)::

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.models import SinusoidalPositionalEncoding, DenseEncoder, SparseEncoder

Dependencies
------------
- ``torch`` ≥ 2.0
- ``benchmark_attn.model_parts.GatherSparseAttention`` (imported lazily
  inside ``SparseEncoder.__init__`` so that the module can be imported
  even when ``benchmark_attn`` is not on the path).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
# Positional encoding
# ------------------------------------------------------------------ #

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding.

    Adds a fixed (non-learned) sinusoidal positional signal to the
    input embeddings.  The encoding is pre-computed up to ``max_len``
    positions and stored as a buffer.

    Parameters
    ----------
    d_model : int
        Model dimension.  Must be even.
    max_len : int, default 2048
        Maximum sequence length for which to pre-compute the encoding.

    Shape
    -----
    Input:  ``(B, N, d_model)``
    Output: ``(B, N, d_model)``
    """

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ------------------------------------------------------------------ #
# Normalization
# ------------------------------------------------------------------ #

class AgentCentricNormalization(nn.Module):
    """Agent-centric pair normalization.

    Given source and target coordinates, computes the displacement,
    direction, and distance between every (src, tgt) pair.  The result
    is a ``(B, N_src, N_tgt, 2*ndim+1)`` tensor that can be projected
    into the model dimension.

    Parameters
    ----------
    ndim : int, default 2
        Spatial dimensionality of the coordinates.

    Shape
    -----
    Input:  ``coords_src`` ``(B, N_src, ndim)``,
            ``coords_tgt`` ``(B, N_tgt, ndim)``
    Output: ``(B, N_src, N_tgt, 2*ndim+1)``
    """

    def __init__(self, ndim: int = 2) -> None:
        super().__init__()
        self.ndim = ndim

    def forward(
        self, coords_src: torch.Tensor, coords_tgt: torch.Tensor
    ) -> torch.Tensor:
        disp = coords_tgt[:, None, :, :] - coords_src[:, :, None, :]
        dist = torch.norm(disp, dim=-1, keepdim=True)
        dir_vec = F.normalize(disp + 1e-8, dim=-1)
        return torch.cat([disp, dir_vec, dist], dim=-1)


# ------------------------------------------------------------------ #
# KNN helper
# ------------------------------------------------------------------ #

def compute_knn_indices(coords: torch.Tensor, k: int) -> torch.Tensor:
    """Compute k-nearest-neighbour indices from spatial coordinates.

    Uses ``torch.cdist`` + ``torch.topk`` on the last two dimensions
    of ``coords`` (assumed to be 2-D y, x).

    If the sequence length ``N`` is smaller than ``k``, the result is
    padded by repeating the last valid index.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinate tensor of shape ``(B, N, D)`` where ``D >= 2``.
    k : int
        Number of neighbours to retrieve per query point.

    Returns
    -------
    torch.Tensor
        Index tensor of shape ``(B, N, k)`` (int64).
    """
    batch_size, seq_len, _ = coords.shape
    yx = coords[..., -2:]
    dist = torch.cdist(yx, yx)
    effective_k = min(k, seq_len)
    _, indices = torch.topk(dist, k=effective_k, dim=-1, largest=False)
    if indices.shape[-1] < k:
        pad = indices[:, :, -1:].expand(-1, -1, k - indices.shape[-1])
        indices = torch.cat([indices, pad], dim=-1)
    return indices


# ------------------------------------------------------------------ #
# Encoders
# ------------------------------------------------------------------ #

class DenseEncoder(nn.Module):
    """Standard Transformer encoder with dense O(N²) attention.

    Wraps ``nn.TransformerEncoder`` with sinusoidal positional encoding
    and a final ``LayerNorm``.

    Parameters
    ----------
    d_model : int, default 128
    nhead : int, default 4
    num_layers : int, default 4
    dim_feedforward : int, default 256
    dropout : float, default 0.1
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.pos_enc(x)
        return self.norm(self.encoder(x, src_key_padding_mask=mask))


class SparseEncoder(nn.Module):
    """Transformer encoder using ``GatherSparseAttention`` (O(Nk)).

    Each layer replaces the standard self-attention with
    ``GatherSparseAttention``, which gathers the ``knn_k`` nearest
    neighbours per query point before computing attention.

    Parameters
    ----------
    d_model : int, default 128
    nhead : int, default 4
    num_layers : int, default 4
    dim_feedforward : int, default 256
    dropout : float, default 0.1
    knn_k : int, default 16
        Number of nearest neighbours to attend to.

    Note
    ----
    ``GatherSparseAttention`` is imported from ``benchmark_attn`` at
    construction time.  If ``benchmark_attn`` is not importable, the
    ``ImportError`` propagates to the caller.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        knn_k: int = 16,
    ) -> None:
        super().__init__()
        from model_parts import GatherSparseAttention

        self.knn_k = knn_k
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = GatherSparseAttention(
                d_model, nhead, knn_k, dropout=dropout, mode="none"
            )
            layer = nn.TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.layers.append(
                nn.ModuleDict(
                    {
                        "attn": attn,
                        "linear1": layer.linear1,
                        "linear2": layer.linear2,
                        "norm1": layer.norm1,
                        "norm2": layer.norm2,
                        "dropout": layer.dropout,
                        "dropout1": layer.dropout1,
                        "dropout2": layer.dropout2,
                    }
                )
            )
        self.norm = nn.LayerNorm(d_model)

    def compute_knn(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute KNN indices from coordinates. Delegates to
        :func:`compute_knn_indices`."""
        return compute_knn_indices(coords, self.knn_k)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.pos_enc(x)
        if coords is None:
            return self.norm(x)
        knn_idx = self.compute_knn(coords)
        for layer in self.layers:
            attn_out = layer["attn"](x, x, x, knn_idx, coords)
            x = layer["norm1"](x + layer["dropout1"](attn_out))
            ff = layer["linear2"](layer["dropout"](F.gelu(layer["linear1"](x))))
            x = layer["norm2"](x + layer["dropout2"](ff))
        return self.norm(x)
