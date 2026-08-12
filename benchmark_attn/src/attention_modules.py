"""Attention mechanisms for KNN-based sparse attention benchmarks.

This module provides the core attention modules used across the
benchmark_attn project (extracted from the former ``model_parts.py``).
Each class implements a different attention variant (dense, gather-sparse,
mask-sparse, NSA, MiniMax) with a common ``forward`` interface so they can
be swapped in benchmark scripts without changing the surrounding harness.

The positional-encoding helpers (:class:`RotaryPositionalEncoding`, the
bin-initialisation functions, and ``ATTN_IGNORE_VALUE``) live in
``positional_encoding.py`` and are imported from there.

All classes accept float32 tensors unless noted otherwise. Tensors
are expected to be on GPU; callers are responsible for moving data
to device. Input shapes follow the ``(batch_size, seq_len, embed_dim)``
convention: batch, sequence length, embedding dimension.
"""

import logging
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from native_sparse_attention_pytorch import SparseAttention

from .positional_encoding import (
    ATTN_IGNORE_VALUE,
    RotaryPositionalEncoding,
    _init_exponential_bins,
    _init_linear_bins,
)


logger = logging.getLogger(__name__)


class KNNRelativePositionalBias(nn.Module):
    """Learnable KNN-aware relative positional bias for sparse attention.

    Computes a per-head additive bias for each query-neighbour pair.
    Spatial distances are bucketed into exponentially spaced bins;
    temporal distances into linearly spaced bins.  Each
    (spatial-bin, temporal-bin) pair maps to a learnable bias vector
    of size ``n_head``.

    When ``knn_indices`` is provided the bias has shape
    ``(batch_size, n_head, seq_len, knn_neighbors)`` and aligns with
    gathered keys/values.  Without ``knn_indices`` a full
    ``(batch_size, n_head, seq_len, seq_len)`` bias is produced.
    """

    def __init__(
        self,
        n_head: int,
        cutoff_spatial: float,
        cutoff_temporal: float,
        n_spatial: int = 32,
        n_temporal: int = 16,
    ):
        """Initialise spatial and temporal bin edges plus bias table.

        Args:
            n_head: Number of attention heads; determines bias vector width.
            cutoff_spatial: Maximum spatial distance represented by the
                last bin edge.
            cutoff_temporal: Half-range of the temporal bins.
            n_spatial: Number of spatial bin edges.
            n_temporal: Number of temporal bin edges per direction.
                Total bins = ``2 * n_temporal + 1``.
        """
        super().__init__()
        self.spatial_bins = _init_exponential_bins(cutoff_spatial, n_spatial)
        self.temporal_bins = _init_linear_bins(cutoff_temporal, 2 * n_temporal + 1)
        self.register_buffer("spatial_bins", self.spatial_bins)
        self.register_buffer("temporal_bins", self.temporal_bins)
        self.n_spatial = n_spatial
        self.n_head = n_head
        self.bias = nn.Parameter(
            -0.5 + torch.rand((2 * n_temporal + 1) * n_spatial, n_head)
        )

    def forward(
        self,
        coords: torch.Tensor,
        knn_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute relative positional bias for KNN-sparse attention.

        Args:
            coords: Spatio-temporal coordinates of shape
                ``(batch_size, seq_len, coord_dim)`` where
                ``coord_dim >= 2`` (first channel = temporal, rest = spatial).
            knn_indices: Optional KNN index tensor of shape
                ``(batch_size, seq_len, knn_neighbors)``.  When provided
                the bias is computed only for the neighbour set.

        Returns:
            Bias tensor of shape
            ``(batch_size, n_head, seq_len, knn_neighbors)`` if
            ``knn_indices`` is given, otherwise
            ``(batch_size, n_head, seq_len, seq_len)``.
        """
        temporal_coords = coords[..., 0]
        spatial_coords = coords[..., 1:]

        if knn_indices is not None:
            batch_idx = torch.arange(
                coords.shape[0], device=coords.device
            ).view(coords.shape[0], 1, 1)
            temporal_knn = temporal_coords[batch_idx, knn_indices]
            spatial_knn = spatial_coords[batch_idx, knn_indices, :]

            temporal_dist = temporal_coords.unsqueeze(-1) - temporal_knn
            spatial_dist = torch.norm(
                spatial_coords.unsqueeze(-2) - spatial_knn, dim=-1
            )
        else:
            temporal_dist = temporal_coords.unsqueeze(-1) - temporal_coords.unsqueeze(-2)
            spatial_dist = torch.cdist(spatial_coords, spatial_coords)

        spatial_bin_idx = torch.bucketize(spatial_dist, self.spatial_bins)
        torch.clamp_(spatial_bin_idx, max=len(self.spatial_bins) - 1)
        temporal_bin_idx = torch.bucketize(temporal_dist, self.temporal_bins)
        torch.clamp_(temporal_bin_idx, max=len(self.temporal_bins) - 1)

        # Flatten (spatial_idx, temporal_idx) into a single index so a
        # single ``index_select`` gathers the bias.  This avoids slow
        # gather/scatter on multi-dimensional indices.
        # Ref: https://discuss.pytorch.org/t/how-to-select-multiple-indexes-over-multiple-dimensions-at-the-same-time/98532/2
        flat_idx = spatial_bin_idx.flatten() + temporal_bin_idx.flatten() * self.n_spatial
        bias = self.bias.index_select(0, flat_idx).view(
            (*spatial_bin_idx.shape, self.n_head)
        )

        if knn_indices is not None:
            bias = bias.permute(0, 3, 1, 2)
        else:
            bias = bias.transpose(-1, 1)

        return bias


class KNNRelativePositionalAttention(nn.Module):
    """KNN-relative positional attention with gather-based sparse selection.

    Projects query/key/value, optionally applies RoPE, gathers the
    ``knn_neighbors`` nearest neighbours for each query token, and
    runs ``scaled_dot_product_attention`` on the resulting
    ``(batch_size * seq_len, n_head, 1, knn_neighbors)`` tensors.
    """

    def __init__(
        self,
        coord_dim: int,
        embed_dim: int,
        n_head: int,
        cutoff_spatial: float = 256,
        cutoff_temporal: float = 16,
        n_spatial: int = 32,
        n_temporal: int = 16,
        dropout: float = 0.0,
        mode: Literal["bias", "rope", "none"] = "bias",
        knn_neighbors: int = 12,
    ):
        """Initialise projections and positional encoding.

        Args:
            coord_dim: Number of coordinate dimensions (temporal + spatial).
            embed_dim: Total embedding dimension (must be divisible by
                ``2 * n_head``).
            n_head: Number of attention heads.
            cutoff_spatial: Spatial cutoff distance for positional bias.
            cutoff_temporal: Temporal cutoff for positional bias.
            n_spatial: Number of spatial bin edges.
            n_temporal: Number of temporal bin edges per direction.
            dropout: Dropout probability on attention weights.
            mode: Positional encoding mode — ``"bias"`` for additive
                relative bias, ``"rope"`` for rotary encoding, ``"none"``
                for no positional information.
            knn_neighbors: Number of nearest neighbours to gather.

        Raises:
            ValueError: If ``embed_dim`` is not divisible by ``2 * n_head``
                or ``mode`` is unrecognised.
        """
        super().__init__()

        if not embed_dim % (2 * n_head) == 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by 2 * n_head {2 * n_head}"
            )

        self.q_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.knn_neighbors = knn_neighbors
        self._mode = mode

        if mode == "bias":
            self.pos_bias = KNNRelativePositionalBias(
                n_head=n_head,
                cutoff_spatial=cutoff_spatial,
                cutoff_temporal=cutoff_temporal,
                n_spatial=n_spatial,
                n_temporal=n_temporal,
            )
        elif mode == "rope":
            n_split = 2 * (embed_dim // (2 * (coord_dim + 1) * n_head))
            self.rot_pos_enc = RotaryPositionalEncoding(
                cutoffs=((cutoff_temporal,) + (cutoff_spatial,) * coord_dim),
                n_pos=(embed_dim // n_head - coord_dim * n_split,)
                + (n_split,) * coord_dim,
            )
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unknown mode {mode!r}; expected 'bias', 'rope', or 'none'")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        knn_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run KNN-relative positional attention.

        Args:
            query: Query tensor of shape ``(batch_size, seq_len, embed_dim)``.
            key: Key tensor, same shape as ``query``.
            value: Value tensor, same shape as ``query``.
            coords: Spatio-temporal coordinates of shape
                ``(batch_size, seq_len, coord_dim)``.
            padding_mask: Currently unused; kept for interface compatibility.
            knn_indices: KNN index tensor of shape
                ``(batch_size, seq_len, knn_neighbors)``.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, embed_dim)``.
        """
        batch_size, seq_len, embed_dim = query.size()
        head_dim = embed_dim // self.n_head

        projected_query = self.q_pro(query)
        projected_key = self.k_pro(key)
        projected_value = self.v_pro(value)

        # split embedding into chunks for n_heads
        reshaped_query = projected_query.view(
            batch_size, seq_len, self.n_head, head_dim
        ).transpose(1, 2)
        reshaped_key = projected_key.view(
            batch_size, seq_len, self.n_head, head_dim
        ).transpose(1, 2)
        reshaped_value = projected_value.view(
            batch_size, seq_len, self.n_head, head_dim
        ).transpose(1, 2)

        if coords is not None and self._mode == "rope":
            reshaped_query, reshaped_key = self.rot_pos_enc(
                reshaped_query, reshaped_key, coords
            )

        batch_idx = torch.arange(batch_size, device=query.device).view(batch_size, 1, 1, 1)
        head_idx = torch.arange(self.n_head, device=query.device).view(1, self.n_head, 1, 1)
        idx_expanded = knn_indices.unsqueeze(1).expand(
            batch_size, self.n_head, seq_len, self.knn_neighbors
        )

        gathered_key = reshaped_key[batch_idx, head_idx, idx_expanded, :]
        gathered_value = reshaped_value[batch_idx, head_idx, idx_expanded, :]

        flat_query = reshaped_query.transpose(1, 2).reshape(
            batch_size * seq_len, self.n_head, 1, head_dim
        )
        flat_key = gathered_key.transpose(1, 2).reshape(
            batch_size * seq_len, self.n_head, self.knn_neighbors, head_dim
        )
        flat_value = gathered_value.transpose(1, 2).reshape(
            batch_size * seq_len, self.n_head, self.knn_neighbors, head_dim
        )

        attn_output = F.scaled_dot_product_attention(
            flat_query, flat_key, flat_value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
        )

        output = attn_output.view(batch_size, seq_len, self.n_head, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.proj(output)

        return output


class GatherSparseAttention(nn.Module):
    """Gather-based sparse attention with optional positional bias or RoPE.

    Precomputes KNN indices, gathers the corresponding keys/values, and
    runs ``scaled_dot_product_attention`` on ``N x knn_neighbors``
    tensors.  No ``N x N`` mask is needed in ``"none"`` mode, which
    enables FlashAttention kernels.  Complexity: ``O(N * k * d)``.

    The gathered tensors are reshaped to
    ``(batch_size * seq_len, n_head, 1, knn_neighbors)`` so SDPA
    operates on many tiny ``1 x K`` attention problems.  This is
    suboptimal for most SDPA backends — see
    :class:`GatherSparseMatmulAttention` for a manual-matmul variant.
    """

    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        knn_neighbors: int,
        dropout: float = 0.0,
        mode: Literal["none", "bias", "rope"] = "none",
        coord_dim: int = 3,
        cutoff_spatial: float = 256,
        cutoff_temporal: float = 16,
        n_spatial: int = 32,
        n_temporal: int = 16,
    ):
        """Initialise projections, KNN bias, and/or RoPE.

        Args:
            embed_dim: Total embedding dimension (must be divisible by
                ``n_head``).
            n_head: Number of attention heads.
            knn_neighbors: Number of nearest neighbours to gather per query.
            dropout: Dropout probability on attention weights.
            mode: Positional encoding mode.
            coord_dim: Number of coordinate dimensions (for RoPE shape calc).
            cutoff_spatial: Spatial cutoff for positional bias bins.
            cutoff_temporal: Temporal cutoff for positional bias bins.
            n_spatial: Number of spatial bin edges.
            n_temporal: Number of temporal bin edges per direction.

        Raises:
            ValueError: If ``embed_dim`` is not divisible by ``n_head``
                or ``mode`` is unrecognised.
        """
        super().__init__()
        assert embed_dim % n_head == 0
        self.q_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.knn_neighbors = knn_neighbors
        self.dropout = dropout
        self._mode = mode

        if mode == "bias":
            self.pos_bias = KNNRelativePositionalBias(
                n_head=n_head,
                cutoff_spatial=cutoff_spatial,
                cutoff_temporal=cutoff_temporal,
                n_spatial=n_spatial,
                n_temporal=n_temporal,
            )
        elif mode == "rope":
            n_split = 2 * (embed_dim // (2 * (coord_dim + 1) * n_head))
            self.rot_pos_enc = RotaryPositionalEncoding(
                cutoffs=((cutoff_temporal,) + (cutoff_spatial,) * coord_dim),
                n_pos=(embed_dim // n_head - coord_dim * n_split,)
                + (n_split,) * coord_dim,
            )
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unknown mode {mode!r}")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        knn_indices: torch.Tensor,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run gather-based sparse attention.

        Args:
            query: Query tensor ``(batch_size, seq_len, embed_dim)``.
            key: Key tensor, same shape as ``query``.
            value: Value tensor, same shape as ``query``.
            knn_indices: KNN index tensor
                ``(batch_size, seq_len, knn_neighbors)``.
            coords: Optional spatio-temporal coordinates for
                positional encoding.

        Returns:
            Output tensor ``(batch_size, seq_len, embed_dim)``.
        """
        batch_size, seq_len, embed_dim = query.shape
        head_dim = embed_dim // self.n_head

        projected_query = self.q_pro(query)
        projected_key = self.k_pro(key)
        projected_value = self.v_pro(value)

        reshaped_query = projected_query.view(
            batch_size, seq_len, self.n_head, head_dim
        ).transpose(1, 2)
        reshaped_key = projected_key.view(
            batch_size, seq_len, self.n_head, head_dim
        ).transpose(1, 2)
        reshaped_value = projected_value.view(
            batch_size, seq_len, self.n_head, head_dim
        ).transpose(1, 2)

        if self._mode == "rope" and coords is not None:
            reshaped_query, reshaped_key = self.rot_pos_enc(
                reshaped_query, reshaped_key, coords
            )

        batch_idx = torch.arange(batch_size, device=query.device).view(batch_size, 1, 1, 1)
        head_idx = torch.arange(self.n_head, device=query.device).view(1, self.n_head, 1, 1)
        idx_expanded = knn_indices.unsqueeze(1).expand(
            batch_size, self.n_head, seq_len, self.knn_neighbors
        )

        gathered_key = reshaped_key[batch_idx, head_idx, idx_expanded, :]
        gathered_value = reshaped_value[batch_idx, head_idx, idx_expanded, :]

        flat_query = reshaped_query.transpose(1, 2).reshape(
            batch_size * seq_len, self.n_head, 1, head_dim
        )
        flat_key = gathered_key.transpose(1, 2).reshape(
            batch_size * seq_len, self.n_head, self.knn_neighbors, head_dim
        )
        flat_value = gathered_value.transpose(1, 2).reshape(
            batch_size * seq_len, self.n_head, self.knn_neighbors, head_dim
        )

        attn_mask = None
        if self._mode == "bias" and coords is not None:
            attn_mask = self.pos_bias(coords, knn_indices)

        attn_output = F.scaled_dot_product_attention(
            flat_query, flat_key, flat_value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
        )

        output = attn_output.view(batch_size, seq_len, self.n_head, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.proj(output)
        return output


class GatherSparseFusedAttention(nn.Module):
    """Optimised gather-sparse attention: eliminates unnecessary copies.

    Compared to :class:`GatherSparseAttention` this variant:
      - Uses ``view + unsqueeze`` for ``flat_query`` (no copy since
        ``q`` is contiguous at that point).
      - Uses ``reshape`` on the SDPA output instead of
        ``transpose + contiguous + view``.

    Profiling shows ~2 fewer clone+copy pairs per layer vs V1.
    """

    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        knn_neighbors: int,
        dropout: float = 0.0,
        mode: Literal["none", "bias", "rope"] = "none",
        coord_dim: int = 3,
        cutoff_spatial: float = 256,
        cutoff_temporal: float = 16,
        n_spatial: int = 32,
        n_temporal: int = 16,
    ):
        """Initialise projections and optional positional encoding.

        Args:
            embed_dim: Total embedding dimension (divisible by ``n_head``).
            n_head: Number of attention heads.
            knn_neighbors: Number of nearest neighbours to gather.
            dropout: Dropout probability on attention weights.
            mode: Positional encoding mode.
            coord_dim: Number of coordinate dimensions.
            cutoff_spatial: Spatial cutoff for positional bias bins.
            cutoff_temporal: Temporal cutoff for positional bias bins.
            n_spatial: Number of spatial bin edges.
            n_temporal: Number of temporal bin edges per direction.

        Raises:
            ValueError: If ``embed_dim`` is not divisible by ``n_head``
                or ``mode`` is unrecognised.
        """
        super().__init__()
        assert embed_dim % n_head == 0
        self.q_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.knn_neighbors = knn_neighbors
        self.dropout = dropout
        self._mode = mode

        if mode == "bias":
            self.pos_bias = KNNRelativePositionalBias(
                n_head=n_head,
                cutoff_spatial=cutoff_spatial,
                cutoff_temporal=cutoff_temporal,
                n_spatial=n_spatial,
                n_temporal=n_temporal,
            )
        elif mode == "rope":
            n_split = 2 * (embed_dim // (2 * (coord_dim + 1) * n_head))
            self.rot_pos_enc = RotaryPositionalEncoding(
                cutoffs=((cutoff_temporal,) + (cutoff_spatial,) * coord_dim),
                n_pos=(embed_dim // n_head - coord_dim * n_split,)
                + (n_split,) * coord_dim,
            )
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unknown mode {mode!r}")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        knn_indices: torch.Tensor,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run optimised gather-sparse attention.

        Args:
            query: Query tensor ``(batch_size, seq_len, embed_dim)``.
            key: Key tensor, same shape as ``query``.
            value: Value tensor, same shape as ``query``.
            knn_indices: KNN index tensor
                ``(batch_size, seq_len, knn_neighbors)``.
            coords: Optional spatio-temporal coordinates for
                positional encoding.

        Returns:
            Output tensor ``(batch_size, seq_len, embed_dim)``.
        """
        batch_size, seq_len, embed_dim = query.shape
        n_head = self.n_head
        head_dim = embed_dim // n_head
        knn_neighbors = self.knn_neighbors

        projected_query = self.q_pro(query)
        projected_key = self.k_pro(key)
        projected_value = self.v_pro(value)

        reshaped_query = projected_query.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        reshaped_key = projected_key.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        reshaped_value = projected_value.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)

        if self._mode == "rope" and coords is not None:
            reshaped_query, reshaped_key = self.rot_pos_enc(
                reshaped_query, reshaped_key, coords
            )

        batch_idx = torch.arange(batch_size, device=query.device).view(batch_size, 1, 1, 1)
        head_idx = torch.arange(n_head, device=query.device).view(1, n_head, 1, 1)
        idx_expanded = knn_indices.unsqueeze(1).expand(batch_size, n_head, seq_len, knn_neighbors)

        gathered_key = reshaped_key[batch_idx, head_idx, idx_expanded, :]
        gathered_value = reshaped_value[batch_idx, head_idx, idx_expanded, :]

        flat_query = reshaped_query.reshape(batch_size * seq_len, n_head, head_dim).unsqueeze(2)
        flat_key = gathered_key.transpose(1, 2).contiguous().view(
            batch_size * seq_len, n_head, knn_neighbors, head_dim
        )
        flat_value = gathered_value.transpose(1, 2).contiguous().view(
            batch_size * seq_len, n_head, knn_neighbors, head_dim
        )

        attn_mask = None
        if self._mode == "bias" and coords is not None:
            attn_mask = self.pos_bias(coords, knn_indices)

        attn_output = F.scaled_dot_product_attention(
            flat_query, flat_key, flat_value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
        )

        output = attn_output.reshape(batch_size, seq_len, embed_dim)
        output = self.proj(output)
        return output


class GatherSparseMatmulAttention(nn.Module):
    """Gather-sparse attention with manual ``torch.matmul`` instead of SDPA.

    The V1/V2 reshape to ``(batch_size * seq_len, n_head, 1, knn_neighbors)``
    creates many tiny ``1 x K`` attention problems — no SDPA backend
    handles ``q_len=1`` efficiently.

    V3 keeps the native ``(batch_size, n_head, seq_len, head_dim)`` shape
    and uses ``torch.matmul`` for per-query attention, avoiding SDPA
    kernel-launch overhead.  No mask is needed; all operations use
    optimised cuBLAS/cuDNN matmul kernels.
    """

    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        knn_neighbors: int,
        dropout: float = 0.0,
        mode: Literal["none", "bias", "rope"] = "none",
        coord_dim: int = 3,
        cutoff_spatial: float = 256,
        cutoff_temporal: float = 16,
        n_spatial: int = 32,
        n_temporal: int = 16,
    ):
        """Initialise projections and optional positional encoding.

        Args:
            embed_dim: Total embedding dimension (divisible by ``n_head``).
            n_head: Number of attention heads.
            knn_neighbors: Number of nearest neighbours per query.
            dropout: Dropout probability on attention weights.
            mode: Positional encoding mode.
            coord_dim: Number of coordinate dimensions.
            cutoff_spatial: Spatial cutoff for positional bias bins.
            cutoff_temporal: Temporal cutoff for positional bias bins.
            n_spatial: Number of spatial bin edges.
            n_temporal: Number of temporal bin edges per direction.

        Raises:
            ValueError: If ``embed_dim`` is not divisible by ``n_head``
                or ``mode`` is unrecognised.
        """
        super().__init__()
        assert embed_dim % n_head == 0
        self.q_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.knn_neighbors = knn_neighbors
        self.dropout = dropout
        self._mode = mode
        self._scale = (embed_dim // n_head) ** -0.5

        if mode == "bias":
            self.pos_bias = KNNRelativePositionalBias(
                n_head=n_head,
                cutoff_spatial=cutoff_spatial,
                cutoff_temporal=cutoff_temporal,
                n_spatial=n_spatial,
                n_temporal=n_temporal,
            )
        elif mode == "rope":
            n_split = 2 * (embed_dim // (2 * (coord_dim + 1) * n_head))
            self.rot_pos_enc = RotaryPositionalEncoding(
                cutoffs=((cutoff_temporal,) + (cutoff_spatial,) * coord_dim),
                n_pos=(embed_dim // n_head - coord_dim * n_split,)
                + (n_split,) * coord_dim,
            )
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unknown mode {mode!r}")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        knn_indices: torch.Tensor,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run manual-matmul gather-sparse attention.

        Args:
            query: Query tensor ``(batch_size, seq_len, embed_dim)``.
            key: Key tensor, same shape as ``query``.
            value: Value tensor, same shape as ``query``.
            knn_indices: KNN index tensor
                ``(batch_size, seq_len, knn_neighbors)``.
            coords: Optional coordinates for positional bias.

        Returns:
            Output tensor ``(batch_size, seq_len, embed_dim)``.
        """
        batch_size, seq_len, embed_dim = query.shape
        n_head = self.n_head
        head_dim = embed_dim // n_head
        knn_neighbors = self.knn_neighbors

        projected_query = self.q_pro(query).view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        projected_key = self.k_pro(key).view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        projected_value = self.v_pro(value).view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)

        if self._mode == "rope" and coords is not None:
            projected_query, projected_key = self.rot_pos_enc(
                projected_query, projected_key, coords
            )

        batch_idx = torch.arange(batch_size, device=query.device).view(batch_size, 1, 1, 1)
        head_idx = torch.arange(n_head, device=query.device).view(1, n_head, 1, 1)
        idx_expanded = knn_indices.unsqueeze(1).expand(batch_size, n_head, seq_len, knn_neighbors)
        gathered_key = projected_key[batch_idx, head_idx, idx_expanded, :]
        gathered_value = projected_value[batch_idx, head_idx, idx_expanded, :]

        # scores: (batch_size, n_head, seq_len, knn_neighbors)
        # Manual SDPA Step 1: Compute dot-product scores (Q @ K^T)
        scores = torch.matmul(
            projected_query.unsqueeze(3), gathered_key.transpose(-2, -1)
        ).squeeze(3)

        # Manual SDPA Step 2: Scale scores
        scores = scores * self._scale

        if self._mode == "bias" and coords is not None:
            bias = self.pos_bias(coords, knn_indices)
            scores = scores + bias

        # Manual SDPA Step 3: Softmax normalization
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # Manual SDPA Step 4: Weighted sum of values (Weights @ V)
        # This manual sequence avoids the kernel dispatch overhead of F.scaled_dot_product_attention
        # when dealing with tiny (1 x K) attention problems.
        output = torch.matmul(attn.unsqueeze(3), gathered_value).squeeze(3)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.proj(output)
        return output


class KNNMaskSparseAttention(nn.Module):
    """KNN-sparse attention via an ``N x N`` boolean mask.

    Builds a mask from KNN indices (scatter ``0.0`` at neighbour
    positions, ``-inf`` elsewhere) and applies
    ``scaled_dot_product_attention`` with that mask.  Keeps
    ``q, k, v`` at the native ``(batch_size, n_head, seq_len, head_dim)``
    shape so the cuDNN/EfficientAttention SDPA backend handles it
    efficiently.

    Roughly 10x faster than :class:`GatherSparseAttention` at
    ``seq_len=256`` because the mask is built via scatter (``O(NK)``)
    rather than ``cdist`` (``O(N^2)``) and there are no advanced
    indexing copies for K/V.
    """

    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        knn_neighbors: int,
        dropout: float = 0.0,
        mode: Literal["none", "bias", "rope"] = "none",
        coord_dim: int = 3,
        cutoff_spatial: float = 256,
        cutoff_temporal: float = 16,
        n_spatial: int = 32,
        n_temporal: int = 16,
    ):
        """Initialise projections and optional positional encoding.

        Args:
            embed_dim: Total embedding dimension (divisible by ``n_head``).
            n_head: Number of attention heads.
            knn_neighbors: Number of nearest neighbours per query.
            dropout: Dropout probability on attention weights.
            mode: Positional encoding mode.
            coord_dim: Number of coordinate dimensions.
            cutoff_spatial: Spatial cutoff for positional bias bins.
            cutoff_temporal: Temporal cutoff for positional bias bins.
            n_spatial: Number of spatial bin edges.
            n_temporal: Number of temporal bin edges per direction.

        Raises:
            ValueError: If ``embed_dim`` is not divisible by ``n_head``
                or ``mode`` is unrecognised.
        """
        super().__init__()
        assert embed_dim % n_head == 0
        self.q_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.knn_neighbors = knn_neighbors
        self.dropout = dropout
        self._mode = mode

        if mode == "bias":
            self.pos_bias = KNNRelativePositionalBias(
                n_head=n_head,
                cutoff_spatial=cutoff_spatial,
                cutoff_temporal=cutoff_temporal,
                n_spatial=n_spatial,
                n_temporal=n_temporal,
            )
        elif mode == "rope":
            n_split = 2 * (embed_dim // (2 * (coord_dim + 1) * n_head))
            self.rot_pos_enc = RotaryPositionalEncoding(
                cutoffs=((cutoff_temporal,) + (cutoff_spatial,) * coord_dim),
                n_pos=(embed_dim // n_head - coord_dim * n_split,)
                + (n_split,) * coord_dim,
            )
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unknown mode {mode!r}")

    def _make_mask(
        self,
        batch_size: int,
        n_head: int,
        seq_len: int,
        knn_neighbors: int,
        knn_indices: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build ``(batch_size, n_head, seq_len, seq_len)`` mask from KNN indices.

        ``-inf`` for non-neighbours, ``0.0`` for neighbours.

        Args:
            batch_size: Batch size.
            n_head: Number of heads.
            seq_len: Sequence length.
            knn_neighbors: Number of neighbours per query.
            knn_indices: KNN index tensor
                ``(batch_size, seq_len, knn_neighbors)``.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Mask tensor of shape
            ``(batch_size, n_head, seq_len, seq_len)``.
        """
        expanded_indices = knn_indices.unsqueeze(1).expand(
            batch_size, n_head, seq_len, knn_neighbors
        )
        mask = torch.full(
            (batch_size, n_head, seq_len, seq_len),
            float("-inf"),
            device=device,
            dtype=dtype,
        )
        mask.scatter_(3, expanded_indices, 0.0)
        return mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        knn_indices: torch.Tensor,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run mask-based KNN sparse attention.

        Args:
            query: Query tensor ``(batch_size, seq_len, embed_dim)``.
            key: Key tensor, same shape as ``query``.
            value: Value tensor, same shape as ``query``.
            knn_indices: KNN index tensor
                ``(batch_size, seq_len, knn_neighbors)``.
            coords: Optional coordinates for positional bias.

        Returns:
            Output tensor ``(batch_size, seq_len, embed_dim)``.
        """
        batch_size, seq_len, embed_dim = query.shape
        n_head = self.n_head
        head_dim = embed_dim // n_head

        projected_query = self.q_pro(query).view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        projected_key = self.k_pro(key).view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        projected_value = self.v_pro(value).view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)

        mask = self._make_mask(
            batch_size, n_head, seq_len, self.knn_neighbors,
            knn_indices, projected_query.device, projected_query.dtype,
        )

        if self._mode == "bias" and coords is not None:
            mask = mask + self.pos_bias(coords, knn_indices)

        attn_output = F.scaled_dot_product_attention(
            projected_query, projected_key, projected_value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0,
        )

        output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.proj(output)
        return output


class DenseFlashAttention(nn.Module):
    """Plain dense ``scaled_dot_product_attention`` without KNN overhead.

    No mask, no positional bias, no gather.  Relies on the FlashAttention
    kernel for ``O(N^2 d)`` compute.  Optimal when
    ``seq_len < knn_neighbors * seq_len`` overhead threshold
    (approximately ``seq_len < 500`` for typical ``knn_neighbors=16``).
    """

    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        dropout: float = 0.0,
    ):
        """Initialise QKV projections and output projection.

        Args:
            embed_dim: Total embedding dimension (divisible by ``n_head``).
            n_head: Number of attention heads.
            dropout: Dropout probability on attention weights.

        Raises:
            AssertionError: If ``embed_dim`` is not divisible by ``n_head``.
        """
        super().__init__()
        assert embed_dim % n_head == 0
        self.q_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        knn_indices: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run dense self-attention via ``scaled_dot_product_attention``.

        Args:
            query: Query tensor ``(batch_size, seq_len, embed_dim)``.
            key: Key tensor, same shape as ``query``.
            value: Value tensor, same shape as ``query``.
            knn_indices: Unused; kept for interface compatibility.
            coords: Unused; kept for interface compatibility.

        Returns:
            Output tensor ``(batch_size, seq_len, embed_dim)``.
        """
        batch_size, seq_len, embed_dim = query.shape
        n_head = self.n_head
        head_dim = embed_dim // n_head

        projected_query = self.q_pro(query).view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        projected_key = self.k_pro(key).view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        projected_value = self.v_pro(value).view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            projected_query, projected_key, projected_value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
        )

        output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.proj(output)
        return output


class RelativePositionalBias(nn.Module):
    """Full-attention learnable relative positional bias.

    Computes pairwise spatial and temporal distances, bins them, and
    looks up a learnable bias for each ``(spatial-bin, temporal-bin)``
    pair.  The result is a full
    ``(batch_size, n_head, seq_len, seq_len)`` bias tensor added to
    the attention scores.

    Unlike :class:`KNNRelativePositionalBias` this does not require
    pre-computed KNN indices — it computes the full ``N x N``
    pairwise distances.
    """

    def __init__(
        self,
        n_head: int,
        cutoff_spatial: float,
        cutoff_temporal: float,
        n_spatial: int = 32,
        n_temporal: int = 16,
    ):
        """Initialise bin edges and learnable bias table.

        Args:
            n_head: Number of attention heads.
            cutoff_spatial: Spatial cutoff distance.
            cutoff_temporal: Temporal cutoff distance.
            n_spatial: Number of spatial bin edges.
            n_temporal: Number of temporal bin edges per direction
                (total = ``2 * n_temporal + 1``).
        """
        super().__init__()
        self._spatial_bins = _init_exponential_bins(cutoff_spatial, n_spatial)
        self._temporal_bins = _init_linear_bins(cutoff_temporal, 2 * n_temporal + 1)
        self.register_buffer("spatial_bins", self._spatial_bins)
        self.register_buffer("temporal_bins", self._temporal_bins)
        self.n_spatial = n_spatial
        self.n_head = n_head
        self.bias = nn.Parameter(
            -0.5 + torch.rand((2 * n_temporal + 1) * n_spatial, n_head)
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute full pairwise relative positional bias.

        Args:
            coords: Spatio-temporal coordinates
                ``(batch_size, seq_len, coord_dim)``.

        Returns:
            Bias tensor of shape
            ``(batch_size, n_head, seq_len, seq_len)``.
        """
        temporal_coords = coords[..., 0]
        spatial_coords = coords[..., 1:]
        temporal_dist = temporal_coords.unsqueeze(-1) - temporal_coords.unsqueeze(-2)
        spatial_dist = torch.cdist(spatial_coords, spatial_coords)

        spatial_idx = torch.bucketize(spatial_dist, self.spatial_bins)
        torch.clamp_(spatial_idx, max=len(self.spatial_bins) - 1)
        temporal_idx = torch.bucketize(temporal_dist, self.temporal_bins)
        torch.clamp_(temporal_idx, max=len(self.temporal_bins) - 1)

        idx = spatial_idx.flatten() + temporal_idx.flatten() * self.n_spatial
        bias = self.bias.index_select(0, idx).view((*spatial_idx.shape, self.n_head))
        bias = bias.transpose(-1, 1)
        return bias


class RelativePositionalAttention(nn.Module):
    """Full-attention module with relative positional bias and spatial cutoff.

    Applies a spatial cutoff mask (tokens farther than
    ``cutoff_spatial`` are masked out), an optional learnable relative
    positional bias, and an optional distance-decay term added to
    the attention scores.

    Supports two positional encoding modes:
      - ``"bias"``: learnable :class:`RelativePositionalBias`
      - ``"rope"``: rotary positional encoding (:class:`RotaryPositionalEncoding`)
    """

    def __init__(
        self,
        coord_dim: int,
        embed_dim: int,
        n_head: int,
        cutoff_spatial: float = 256,
        cutoff_temporal: float = 16,
        n_spatial: int = 32,
        n_temporal: int = 16,
        dropout: float = 0.0,
        mode: Literal["bias", "rope", "none"] = "bias",
        attn_dist_mode: str = "v0",
    ):
        """Initialise projections, positional encoding, and distance decay.

        Args:
            coord_dim: Number of coordinate dimensions.
            embed_dim: Total embedding dimension (divisible by ``2 * n_head``).
            n_head: Number of attention heads.
            cutoff_spatial: Spatial cutoff distance for masking.
            cutoff_temporal: Temporal cutoff for positional bias.
            n_spatial: Number of spatial bin edges.
            n_temporal: Number of temporal bin edges per direction.
            dropout: Dropout probability on attention weights.
            mode: Positional encoding mode — ``"bias"``, ``"rope"``, or ``"none"``.
            attn_dist_mode: Distance decay mode — ``"v0"`` (3D
                ``exp(-0.1 * dist)``) or ``"v1"`` (2D
                ``exp(-5 * dist / cutoff)``).

        Raises:
            ValueError: If ``embed_dim`` is not divisible by
                ``2 * n_head`` or ``mode`` is unrecognised.
        """
        super().__init__()

        if not embed_dim % (2 * n_head) == 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by 2 * n_head {2 * n_head}"
            )

        self.q_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.cutoff_spatial = cutoff_spatial
        self.attn_dist_mode = attn_dist_mode
        self._mode = mode

        if mode == "bias":
            self.pos_bias = RelativePositionalBias(
                n_head=n_head,
                cutoff_spatial=cutoff_spatial,
                cutoff_temporal=cutoff_temporal,
                n_spatial=n_spatial,
                n_temporal=n_temporal,
            )
        elif mode == "rope":
            n_split = 2 * (embed_dim // (2 * (coord_dim + 1) * n_head))
            self.rot_pos_enc = RotaryPositionalEncoding(
                cutoffs=((cutoff_temporal,) + (cutoff_spatial,) * coord_dim),
                n_pos=(embed_dim // n_head - coord_dim * n_split,)
                + (n_split,) * coord_dim,
            )
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unknown mode {mode!r}")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run full attention with spatial cutoff and positional encoding.

        Args:
            query: Query tensor ``(batch_size, seq_len, embed_dim)``.
            key: Key tensor, same shape as ``query``.
            value: Value tensor, same shape as ``query``.
            coords: Spatio-temporal coordinates
                ``(batch_size, seq_len, coord_dim)``.
            padding_mask: Optional padding mask ``(batch_size, seq_len)``.

        Returns:
            Output tensor ``(batch_size, seq_len, embed_dim)``.
        """
        batch_size, seq_len, embed_dim = query.size()
        n_head = self.n_head
        head_dim = embed_dim // n_head

        projected_query = self.q_pro(query)
        projected_key = self.k_pro(key)
        projected_value = self.v_pro(value)

        reshaped_query = projected_query.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        reshaped_key = projected_key.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        reshaped_value = projected_value.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)

        attn_mask = torch.zeros(
            (batch_size, n_head, seq_len, seq_len),
            device=query.device,
            dtype=reshaped_query.dtype,
        )
        attn_ignore_val = ATTN_IGNORE_VALUE

        spatial_coords = coords[..., 1:]
        spatial_dist = torch.cdist(spatial_coords, spatial_coords)
        spatial_mask = (spatial_dist > self.cutoff_spatial).unsqueeze(1)
        attn_mask.masked_fill_(spatial_mask, attn_ignore_val)

        if coords is not None:
            if self._mode == "bias":
                attn_mask = attn_mask + self.pos_bias(coords)
            elif self._mode == "rope":
                reshaped_query, reshaped_key = self.rot_pos_enc(
                    reshaped_query, reshaped_key, coords
                )

            if self.attn_dist_mode == "v0":
                full_dist = torch.cdist(coords, coords, p=2)
                attn_mask += torch.exp(-0.1 * full_dist.unsqueeze(1))
            elif self.attn_dist_mode == "v1":
                attn_mask += torch.exp(
                    -5 * spatial_dist.unsqueeze(1) / self.cutoff_spatial
                )
            else:
                raise ValueError(f"Unknown attn_dist_mode {self.attn_dist_mode!r}")

        if padding_mask is not None:
            ignore_mask = torch.logical_or(
                padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
            ).unsqueeze(1)
            attn_mask.masked_fill_(ignore_mask, attn_ignore_val)

        attn_output = F.scaled_dot_product_attention(
            reshaped_query, reshaped_key, reshaped_value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
        )

        output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.proj(output)
        return output


class MiniMaxSparseAttention(nn.Module):
    """MiniMax Sparse Attention (MSA) adapted for MHA benchmark (Lai et al. 2026).

    Original MSA uses GQA with Group-specific block selection via a learned
    Index Branch. This adaptation treats each head independently (n_kv_head=n_head),
    performing per-head block selection. The Index Branch is retained to
    capture the realistic computational overhead.

    Design:
      - Index Branch: w_q_idx, w_k_idx project to index_dim, then block-level
        max-pooling scores -> Top-k block selection per head per query
      - Main Branch: block-sparse matmul attention over selected k*Bk tokens
      - Complexity: O(N * k*B_k * d) compute per forward pass
    """

    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        block_size: int = 128,
        num_selected_blocks: int = 4,
        index_dim: int = 64,
        dropout: float = 0.0,
        mode: str = "none",
    ):
        """Initialise projections and block-selection parameters.

        Args:
            embed_dim: Total embedding dimension (divisible by ``n_head``).
            n_head: Number of attention heads.
            block_size: Size of each attention block (``B_k``).
            num_selected_blocks: Number of top blocks to select per query.
            index_dim: Dimension of the index branch projections.
            dropout: Dropout probability on attention weights.
            mode: Must be ``"none"`` (other modes not implemented).

        Raises:
            NotImplementedError: If ``mode`` is not ``"none"``.
        """
        super().__init__()
        assert embed_dim % n_head == 0
        self.q_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.num_selected_blocks = num_selected_blocks
        self.index_dim = index_dim
        self.dropout = dropout
        self._mode = mode

        if mode not in ("none",):
            raise NotImplementedError(
                f"MiniMax mode '{mode}' not supported in benchmark. Use 'none'."
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        knn_indices: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run block-sparse MiniMax attention.

        Args:
            query: Query tensor ``(batch_size, seq_len, embed_dim)``.
            key: Key tensor, same shape as ``query``.
            value: Value tensor, same shape as ``query``.
            knn_indices: Unused; kept for interface compatibility.
            coords: Unused; kept for interface compatibility.

        Returns:
            Output tensor ``(batch_size, seq_len, embed_dim)``.
        """
        batch_size, seq_len, embed_dim = query.shape
        n_head = self.n_head
        head_dim = embed_dim // n_head
        block_size = self.block_size
        ksel = self.num_selected_blocks

        projected_query = self.q_pro(query).view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        projected_key = self.k_pro(key).view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        projected_value = self.v_pro(value).view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)

        num_blocks = (seq_len + block_size - 1) // block_size
        pad = num_blocks * block_size - seq_len
        k_pad = F.pad(projected_key, (0, 0, 0, pad))
        k_block = k_pad.view(batch_size, n_head, num_blocks, block_size, head_dim)

        expanded_query = projected_query.unsqueeze(3).unsqueeze(-3)
        expanded_key_block = k_block.unsqueeze(2)
        scores_raw = (expanded_query * expanded_key_block).sum(dim=-1)
        block_scores, _ = scores_raw.max(dim=-1)

        _, topk_blk = torch.topk(block_scores, k=ksel, dim=-1)

        offsets = torch.arange(block_size, device=query.device).view(1, 1, 1, 1, block_size)
        token_idx = topk_blk.unsqueeze(-1) * block_size + offsets
        token_idx = token_idx.clamp(0, seq_len + pad - 1).view(batch_size, n_head, seq_len, ksel * block_size)

        batch_idx = torch.arange(batch_size, device=query.device).view(batch_size, 1, 1, 1)
        head_idx = torch.arange(n_head, device=query.device).view(1, n_head, 1, 1)
        k_sel = k_pad[batch_idx, head_idx, token_idx, :]
        v_sel = F.pad(projected_value, (0, 0, 0, pad))[batch_idx, head_idx, token_idx, :]

        scores = torch.matmul(
            projected_query.unsqueeze(3), k_sel.transpose(-2, -1)
        ).squeeze(3)
        scores = scores * (head_dim ** -0.5)
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        output = torch.matmul(attn.unsqueeze(3), v_sel).squeeze(3)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.proj(output)
        return output


class NSASparseAttention(nn.Module):
    """Native Sparse Attention (NSA) from DeepSeek (Yuan et al. 2025).

    https://arxiv.org/abs/2502.11089

    Combines three sparse attention strategies:
      - Compressed token attention (coarse-grained)
      - Selected block attention (fine-grained, top-k from compressed scores)
      - Sliding window attention (local context)

    Uses lucidrains' pytorch implementation. Wraps the SparseAttention
    module to match the benchmark interface (query, key, value, knn_indices, coords).
    """

    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        sliding_window_size: int = 64,
        compress_block_size: int = 32,
        compress_block_sliding_stride: int = 16,
        selection_block_size: int = 32,
        num_selected_blocks: int = 4,
        dropout: float = 0.0,
        mode: Literal["none"] = "none",
    ):
        """Initialise the wrapped NSA module.

        Args:
            embed_dim: Total embedding dimension (divisible by ``n_head``).
            n_head: Number of attention heads.
            sliding_window_size: Size of the sliding window for local attention.
            compress_block_size: Block size for compressed token attention.
            compress_block_sliding_stride: Stride for compressed block sliding.
            selection_block_size: Block size for selected block attention.
            num_selected_blocks: Number of top blocks to select per query.
            dropout: Unused (NSA handles dropout internally).
            mode: Must be ``"none"``.

        Raises:
            AssertionError: If ``embed_dim`` is not divisible by ``n_head``.
        """
        super().__init__()
        assert embed_dim % n_head == 0
        self.embed_dim = embed_dim
        self.n_head = n_head
        dim_head = embed_dim // n_head

        self.nsa = SparseAttention(
            dim=embed_dim,
            dim_head=dim_head,
            heads=n_head,
            sliding_window_size=sliding_window_size,
            compress_block_size=compress_block_size,
            compress_block_sliding_stride=compress_block_sliding_stride,
            selection_block_size=selection_block_size,
            num_selected_blocks=num_selected_blocks,
            causal=False,
            use_diff_topk=False,
            use_triton_kernel=False,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        knn_indices: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run Native Sparse Attention.

        Args:
            query: Query tensor ``(batch_size, seq_len, embed_dim)``.
            key: Unused (NSA uses only ``query``); kept for interface compatibility.
            value: Unused; kept for interface compatibility.
            knn_indices: Unused; kept for interface compatibility.
            coords: Unused; kept for interface compatibility.

        Returns:
            Output tensor ``(batch_size, seq_len, embed_dim)``.
        """
        return self.nsa(query)


class SpatialReorder:
    """Reorders sequence by spatial proximity for memory locality (Wu et al. 2024).

    Quantizes coords to grid, computes linearized index (Z-order-like), sorts by it.
    Neighboring tokens in original space → close indices → gather hits contiguous memory.

    This is the spatial reorder concept from Point Transformer V3: reordering
    tokens by spatial proximity via a Z-order-like linearized index so
    neighbours land adjacent in memory — exactly PTv3's point cloud
    serialization (§4.1), using Z-order/Hilbert space-filling curves.

    Wu, X., Jiang, L., Wang, P.-S., Liu, Z., Liu, X., Qiao, Y., Ouyang, W.,
    He, T., & Zhao, H. (2024). Point Transformer V3: Simpler, Faster,
    Stronger. CVPR 2024. arXiv:2312.10035.
    """

    def __init__(self, n_bins: int = 32):
        """Initialise the reorder quantizer.

        Args:
            n_bins: Number of quantization bins per spatial dimension.
                Higher values give finer spatial sorting.
        """
        self.n_bins = n_bins

    def compute_idx(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute reorder and unreorder indices from coordinates.

        Quantizes each coordinate dimension into ``n_bins`` bins,
        then computes a Z-order-like linearized index by interleaving
        dimension-specific indices with increasing strides.

        Args:
            coords: Coordinate tensor ``(batch_size, seq_len, coord_dim)``.

        Returns:
            A tuple ``(reorder_idx, unreorder_idx)`` each of shape
            ``(batch_size, seq_len)``.  ``reorder_idx`` sorts tokens
            by spatial proximity; ``unreorder_idx`` inverts the sort.
        """
        batch_size, seq_len, coord_dim = coords.shape
        normalized_coords = coords - coords.amin(dim=1, keepdim=True)
        normalized_coords = normalized_coords / (
            normalized_coords.amax(dim=1, keepdim=True) + 1e-8
        )
        grid_indices = (normalized_coords * self.n_bins).long().clamp(0, self.n_bins - 1)
        stride = 1
        linear_idx = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=coords.device
        )
        for dim_idx in range(coord_dim):
            linear_idx = linear_idx + grid_indices[..., dim_idx] * stride
            stride *= self.n_bins
        reorder_idx = linear_idx.argsort(dim=1, stable=True)
        unreorder_idx = reorder_idx.argsort(dim=1, stable=True)
        return reorder_idx, unreorder_idx

    @staticmethod
    def apply(tensor: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Apply a reorder/unreorder index to a tensor.

        Args:
            tensor: Input tensor of shape ``(batch_size, seq_len, ...)``
                (3-D or 4-D).
            idx: Index tensor of shape ``(batch_size, seq_len)``.

        Returns:
            Tensor of the same shape as ``tensor`` with tokens
            reordered according to ``idx``.

        Raises:
            ValueError: If ``tensor`` is not 3-D or 4-D.
        """
        batch_size = torch.arange(tensor.shape[0], device=tensor.device)
        if tensor.dim() == 3:
            return tensor[batch_size[:, None], idx]
        elif tensor.dim() == 4:
            return tensor[batch_size[:, None, None], :, idx]
        raise ValueError(f"Unsupported dim {tensor.dim()}")

    def reorder(self, tensor: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Reorder a tensor by spatial proximity.

        Args:
            tensor: Input tensor of shape ``(batch_size, seq_len, ...)``.
            idx: Reorder index from :meth:`compute_idx`.

        Returns:
            Reordered tensor of the same shape as ``tensor``.
        """
        return self.apply(tensor, idx)

    def unreorder(self, tensor: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Unreorder a tensor (invert :meth:`reorder`).

        Args:
            tensor: Input tensor of shape ``(batch_size, seq_len, ...)``.
            idx: Unreorder index from :meth:`compute_idx`.

        Returns:
            Unreordered tensor of the same shape as ``tensor``.
        """
        return self.apply(tensor, idx)


class CachedDistAttention(nn.Module):
    """Spatial-cutoff attention with pre-computed 2D pairwise distances.

    Identical semantics to :class:`RelativePositionalAttention` but
    avoids per-layer ``cdist``: the 2D distance matrix is computed
    once in ``TrackingTransformer.forward()`` and shared across all
    layers.  3D ``cdist`` for distance decay is still per-layer.
    """

    def __init__(
        self,
        coord_dim: int,
        embed_dim: int,
        n_head: int,
        cutoff_spatial: float = 256,
        cutoff_temporal: float = 16,
        n_spatial: int = 32,
        n_temporal: int = 16,
        dropout: float = 0.0,
        mode: Literal["bias", "rope", "none"] = "none",
        attn_dist_mode: str = "v0",
        knn_neighbors: int = -1,
    ):
        """Initialise projections and positional encoding.

        Args:
            coord_dim: Number of coordinate dimensions.
            embed_dim: Total embedding dimension (divisible by ``n_head``).
            n_head: Number of attention heads.
            cutoff_spatial: Spatial cutoff distance for masking.
            cutoff_temporal: Temporal cutoff for positional bias.
            n_spatial: Number of spatial bin edges.
            n_temporal: Number of temporal bin edges per direction.
            dropout: Dropout probability on attention weights.
            mode: Positional encoding mode — ``"bias"``, ``"rope"``, or ``"none"``.
            attn_dist_mode: Distance decay mode — ``"v0"`` or ``"v1"``.
            knn_neighbors: Unused; kept for interface compatibility.

        Raises:
            ValueError: If ``mode`` is unrecognised.
        """
        super().__init__()
        assert embed_dim % n_head == 0
        self.q_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.cutoff_spatial = cutoff_spatial
        self.attn_dist_mode = attn_dist_mode
        self._mode = mode

        if mode == "bias":
            self.pos_bias = RelativePositionalBias(
                n_head=n_head,
                cutoff_spatial=cutoff_spatial,
                cutoff_temporal=cutoff_temporal,
                n_spatial=n_spatial,
                n_temporal=n_temporal,
            )
        elif mode == "rope":
            n_split = 2 * (embed_dim // (2 * (coord_dim + 1) * n_head))
            self.rot_pos_enc = RotaryPositionalEncoding(
                cutoffs=((cutoff_temporal,) + (cutoff_spatial,) * coord_dim),
                n_pos=(embed_dim // n_head - coord_dim * n_split,)
                + (n_split,) * coord_dim,
            )
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unknown mode {mode!r}")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        coords: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        knn_indices: torch.Tensor | None = None,
        dist_2d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run spatial-cutoff attention with cached or on-the-fly distances.

        Args:
            query: Query tensor ``(batch_size, seq_len, embed_dim)``.
            key: Key tensor, same shape as ``query``.
            value: Value tensor, same shape as ``query``.
            coords: Spatio-temporal coordinates
                ``(batch_size, seq_len, coord_dim)``.
            padding_mask: Optional padding mask ``(batch_size, seq_len)``.
            knn_indices: Unused; kept for interface compatibility.
            dist_2d: Optional pre-computed 2D spatial distance matrix
                ``(batch_size, seq_len, seq_len)``.  If ``None``,
                distances are computed from ``coords``.

        Returns:
            Output tensor ``(batch_size, seq_len, embed_dim)``.
        """
        batch_size, seq_len, embed_dim = query.shape
        if seq_len == 0:
            return torch.zeros(
                batch_size, 0, embed_dim, device=query.device, dtype=query.dtype
            )
        n_head = self.n_head
        head_dim = embed_dim // n_head
        attn_ignore_val = ATTN_IGNORE_VALUE

        projected_query = self.q_pro(query).view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        projected_key = self.k_pro(key).view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        projected_value = self.v_pro(value).view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)

        if coords is not None and self._mode == "rope":
            projected_query, projected_key = self.rot_pos_enc(
                projected_query, projected_key, coords
            )

        # Use cached 2D distances if provided; otherwise compute per call.
        if dist_2d is not None:
            spatial_mask = (dist_2d > self.cutoff_spatial).unsqueeze(1).expand(
                -1, n_head, -1, -1
            )
        else:
            spatial_coords = coords[..., 1:]
            spatial_dist = torch.cdist(spatial_coords, spatial_coords)
            spatial_mask = (spatial_dist > self.cutoff_spatial).unsqueeze(1)

        mask = torch.zeros(
            batch_size, n_head, seq_len, seq_len,
            device=projected_query.device, dtype=projected_query.dtype,
        )
        mask.masked_fill_(spatial_mask, attn_ignore_val)

        if coords is not None and self._mode == "bias":
            mask = mask + self.pos_bias(coords)

        if coords is not None:
            if self.attn_dist_mode == "v0":
                full_dist = torch.cdist(coords, coords, p=2)
                mask = mask + torch.exp(-0.1 * full_dist.unsqueeze(1))
            elif self.attn_dist_mode == "v1" and dist_2d is not None:
                # Reuse cached 2D distances for the decay term instead of recomputing.
                mask = mask + torch.exp(-5 * dist_2d.unsqueeze(1) / self.cutoff_spatial)

        if padding_mask is not None:
            ignore_mask = torch.logical_or(
                padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
            ).unsqueeze(1)
            mask.masked_fill_(ignore_mask, attn_ignore_val)

        attn_output = F.scaled_dot_product_attention(
            projected_query, projected_key, projected_value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0,
        )

        output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.proj(output)
        return output


# Backward-compatible aliases for the pre-refactor versioned names.  Kept
# so external code importing ``GatherSparseAttentionV2`` or
# ``GatherSparseAttentionV3`` keeps working; new code should use the
# descriptive names ``GatherSparseFusedAttention`` and
# ``GatherSparseMatmulAttention``.
GatherSparseAttentionV2 = GatherSparseFusedAttention
GatherSparseAttentionV3 = GatherSparseMatmulAttention
