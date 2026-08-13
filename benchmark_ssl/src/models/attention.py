"""Attention building blocks for SSL tracking benchmarks.

This module provides the core attention modules used by the
benchmark_ssl project: rotary positional encoding, KNN-relative
positional bias and attention, gather-based sparse attention, dense
relative-positional attention, and a spatial reordering helper. All
modules expose a common ``forward`` interface so they can be swapped
in SSL training / benchmark scripts without changing the surrounding
harness.

All classes accept float32 tensors unless noted otherwise; callers are
responsible for moving data to the target device. Input shapes follow
the ``(batch_size, seq_len, embed_dim)`` convention: batch size,
sequence length, embedding dimension. Coordinates use
``(batch_size, seq_len, coord_dim)`` where the first channel is
temporal and the remaining channels are spatial.
"""

import logging
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


logger = logging.getLogger(__name__)


def _init_exponential_bins(cutoff: float, n: int) -> torch.Tensor:
    """Build exponentially spaced bin edges from 0 to ``cutoff``.

    Used as the right-edges of ``torch.bucketize`` for spatial distance
    binning in the relative positional bias modules.

    Args:
        cutoff: Maximum distance represented by the last bin edge.
        n: Number of bin edges to generate.

    Returns:
        1-D tensor of length ``n`` with values growing from 1 to
        ``cutoff + 1``.
    """
    return torch.exp(torch.linspace(0, math.log(cutoff + 1), n))


def _init_linear_bins(cutoff: float, n: int) -> torch.Tensor:
    """Build linearly spaced bin edges from ``-cutoff`` to ``+cutoff``.

    Used for temporal distance binning where uniform resolution across
    the time window is preferred.

    Args:
        cutoff: Half-range of the bins (edges span ``[-cutoff, +cutoff]``).
        n: Number of bin edges to generate.

    Returns:
        1-D tensor of length ``n``.
    """
    return torch.linspace(-cutoff, cutoff, n)


def _init_fourier_frequencies(cutoff: float = 128, n: int = 32) -> torch.Tensor:
    """Initialise geometrically decaying Fourier frequencies.

    Produces a 1-D tensor of ``n`` frequencies starting at 1 (Nyquist)
    and decaying to ``1/cutoff``. The frequency vector is reshaped to
    ``(1, 1, n)`` so it broadcasts cleanly when multiplied with
    coordinates.

    Args:
        cutoff: Controls the decay rate; larger values give slower decay.
        n: Number of frequency components.

    Returns:
        Tensor of shape ``(1, 1, n)``.
    """
    return torch.exp(torch.linspace(0, -math.log(cutoff), n)).unsqueeze(0).unsqueeze(0)


# Implementation adapted from LightGlue:
# https://github.com/cvg/LightGlue/blob/b1cd942fc4a3a824b6aedff059d84f5c31c297f6/lightglue/lightglue.py#L51
def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of scalars as 2-D vectors by ``pi/2``.

    Implements eq. 34 in `RoFormer <https://arxiv.org/pdf/2104.09864.pdf>`_.
    The last dimension is split into consecutive pairs ``(first, second)``
    and each pair ``(first, second)`` is replaced with ``(-second, first)``.

    Args:
        tensor: Input tensor whose last dimension is even.

    Returns:
        Tensor of the same shape as ``tensor``.
    """
    tensor = tensor.unflatten(-1, (-1, 2))
    first, second = tensor.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(start_dim=-2)


class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding (RoPE) for multi-dimensional coordinates.

    Each coordinate dimension gets its own set of learnable frequencies.
    Frequencies are initialised so that the maximum frequency is 1
    (Nyquist) and the minimum is ``1/cutoff``. See
    `RoFormer <https://arxiv.org/pdf/2104.09864.pdf>`_ for the
    mathematical background.

    The encoding rotates pairs of channels in the query/key tensors
    by an angle proportional to the coordinate value, providing a
    smooth inductive bias for spatial proximity.
    """

    def __init__(
        self,
        cutoffs: tuple[float, ...] = (256,),
        n_pos: tuple[int, ...] = (32,),
    ):
        """Initialise RoPE with per-dimension frequency vectors.

        Args:
            cutoffs: Per-dimension frequency decay constants. Larger
                values give slower decay (wider spatial reach).
            n_pos: Per-dimension number of frequency components. Must
                be even. The number of coordinate dimensions is inferred
                from ``len(cutoffs)``.

        Raises:
            AssertionError: If ``len(cutoffs) != len(n_pos)``.
            ValueError: If any element of ``n_pos`` is odd.
        """
        super().__init__()
        assert len(cutoffs) == len(n_pos)
        if not all(n % 2 == 0 for n in n_pos):
            raise ValueError("n_pos must be even")

        self.n_dim = len(cutoffs)
        # theta in RoFormer https://arxiv.org/pdf/2104.09864.pdf
        self.freqs = nn.ParameterList([
            nn.Parameter(_init_fourier_frequencies(cutoff, n // 2))
            for cutoff, n in zip(cutoffs, n_pos)
        ])

    def get_co_si(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cosine and sine encodings for each coordinate dimension.

        For each dimension ``d`` the encoding is
        ``cos(0.5 * pi * freq * coord_d)`` (and the sine analogue),
        normalised by ``1/sqrt(len(freq))``.

        Args:
            coords: Tensor of shape ``(batch_size, seq_len, n_dim)``.

        Returns:
            A tuple ``(cosine_encoding, sine_encoding)`` each of shape
            ``(batch_size, seq_len, total_freqs)`` where
            ``total_freqs = sum(len(freq) for freq in self.freqs)``.
        """
        coord_dim = coords.shape[-1]
        assert coord_dim == len(self.freqs)
        cosine_encoding = torch.cat(
            tuple(
                torch.cos(0.5 * math.pi * coord.unsqueeze(-1) * freq.to(coord.device))
                / math.sqrt(len(freq))
                for coord, freq in zip(coords.moveaxis(-1, 0), self.freqs)
            ),
            axis=-1,
        )
        sine_encoding = torch.cat(
            tuple(
                torch.sin(0.5 * math.pi * coord.unsqueeze(-1) * freq.to(coord.device))
                / math.sqrt(len(freq))
                for coord, freq in zip(coords.moveaxis(-1, 0), self.freqs)
            ),
            axis=-1,
        )

        return cosine_encoding, sine_encoding

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary positional encoding to query and key tensors.

        Args:
            query: Tensor of shape ``(batch_size, n_head, seq_len, head_dim)``.
            key: Tensor of same shape as ``query``.
            coords: Tensor of shape ``(batch_size, seq_len, n_dim)``.

        Returns:
            Tuple ``(rotated_query, rotated_key)`` with the same shapes
            as the inputs.

        Raises:
            ValueError: If the last dimension of ``coords`` does not
                match the number of configured frequency vectors.
        """
        coord_dim = coords.shape[-1]
        if coord_dim != self.n_dim:
            raise ValueError(f"coords must have {self.n_dim} dimensions, got {coord_dim}")

        cosine_encoding, sine_encoding = self.get_co_si(coords)

        cosine_encoding = cosine_encoding.unsqueeze(1).repeat_interleave(2, dim=-1)
        sine_encoding = sine_encoding.unsqueeze(1).repeat_interleave(2, dim=-1)

        rotated_query = query * cosine_encoding + _rotate_half(query) * sine_encoding
        rotated_key = key * cosine_encoding + _rotate_half(key) * sine_encoding

        return rotated_query, rotated_key


class KNNRelativePositionalBias(nn.Module):
    """Learnable KNN-aware relative positional bias for sparse attention.

    Computes a per-head additive bias for each query-neighbour pair.
    Spatial distances are bucketed into exponentially spaced bins;
    temporal distances into linearly spaced bins. Each
    (spatial-bin, temporal-bin) pair maps to a learnable bias vector
    of size ``n_head``.

    When ``knn_indices`` is provided the bias has shape
    ``(batch_size, n_head, seq_len, knn_neighbors)`` and aligns with
    gathered keys/values. Without ``knn_indices`` a full
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
        self._spatial_bins = _init_exponential_bins(cutoff_spatial, n_spatial)
        self._temporal_bins = _init_linear_bins(cutoff_temporal, 2 * n_temporal + 1)
        self.register_buffer("spatial_bins", self._spatial_bins)
        self.register_buffer("temporal_bins", self._temporal_bins)
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
                ``(batch_size, seq_len, knn_neighbors)``. When provided
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
            spatial_dist = torch.norm(spatial_coords.unsqueeze(-2) - spatial_knn, dim=-1)
        else:
            temporal_dist = temporal_coords.unsqueeze(-1) - temporal_coords.unsqueeze(-2)
            spatial_dist = torch.cdist(spatial_coords, spatial_coords)

        spatial_bin_idx = torch.bucketize(spatial_dist, self.spatial_bins)
        torch.clamp_(spatial_bin_idx, max=len(self.spatial_bins) - 1)
        temporal_bin_idx = torch.bucketize(temporal_dist, self.temporal_bins)
        torch.clamp_(temporal_bin_idx, max=len(self.temporal_bins) - 1)

        # Flatten (spatial_idx, temporal_idx) into a single index so a
        # single ``index_select`` gathers the bias. This avoids slow
        # gather/scatter on multi-dimensional indices.
        # Ref: https://discuss.pytorch.org/t/how-to-select-multiple-indexes-over-multiple-dimensions-at-the-same-time/98532/2
        flat_idx = spatial_bin_idx.flatten() + temporal_bin_idx.flatten() * self.n_spatial
        bias = self.bias.index_select(0, flat_idx).view((*spatial_bin_idx.shape, self.n_head))

        if knn_indices is not None:
            # -> B, nH, N, K
            bias = bias.permute(0, 3, 1, 2)
        else:
            # -> B, nH, N, N
            bias = bias.transpose(-1, 1)

        return bias


class KNNRelativePositionalAttention(nn.Module):
    """KNN-relative positional attention with gather-based sparse selection.

    The naive KNN-sparse adaptation of :class:`RelativePositionalAttention`.
    Projects query/key/value, optionally applies RoPE, gathers the
    ``knn_neighbors`` nearest neighbours for each query token, applies an
    optional per-head additive positional bias (when ``mode="bias"``), and
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
        attn_dist_mode: str = "v0",
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
            attn_dist_mode: Distance decay mode. Stored for interface
                compatibility; not used by this module.
            knn_neighbors: Number of nearest neighbours to gather.

        Raises:
            ValueError: If ``embed_dim`` is not divisible by
                ``2 * n_head`` or ``mode`` is unrecognised.
        """
        super().__init__()

        if not embed_dim % (2 * n_head) == 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by 2 times n_head {2 * n_head}"
            )

        # qkv projection
        self.q_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_pro = nn.Linear(embed_dim, embed_dim, bias=True)

        # output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        # regularization
        self._mode = mode
        self.attn_dist_mode = attn_dist_mode
        self.cutoff_spatial = cutoff_spatial

        self.dropout = dropout
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.knn_neighbors = knn_neighbors

        if mode == "bias" or mode is True:
            self.pos_bias = KNNRelativePositionalBias(
                n_head=n_head,
                cutoff_spatial=cutoff_spatial,
                cutoff_temporal=cutoff_temporal,
                n_spatial=n_spatial,
                n_temporal=n_temporal,
            )
        elif mode == "rope":
            # each part needs to be divisible by 2
            n_split = 2 * (embed_dim // (2 * (coord_dim + 1) * n_head))

            self.rot_pos_enc = RotaryPositionalEncoding(
                cutoffs=((cutoff_temporal,) + (cutoff_spatial,) * coord_dim),
                n_pos=(embed_dim // n_head - coord_dim * n_split,)
                + (n_split,) * coord_dim,
            )
        elif mode == "none":
            pass
        elif mode is None or mode is False:
            logger.warning(
                "attn_positional_bias is not set (None or False), no positional bias."
            )
            pass
        else:
            raise ValueError(f"Unknown mode {mode}")

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

        projected_query = self.q_pro(query)  # (B, N, D)
        projected_key = self.k_pro(key)  # (B, N, D)
        projected_value = self.v_pro(value)  # (B, N, D)
        # (B, nh, N, hs)
        reshaped_key = projected_key.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        reshaped_query = projected_query.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        reshaped_value = projected_value.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)

        if coords is not None and self._mode == "rope":
            reshaped_query, reshaped_key = self.rot_pos_enc(reshaped_query, reshaped_key, coords)

        batch_idx = torch.arange(batch_size, device=query.device).view(batch_size, 1, 1, 1)
        head_idx = torch.arange(self.n_head, device=query.device).view(1, self.n_head, 1, 1)
        idx_expanded = knn_indices.unsqueeze(1).expand(batch_size, self.n_head, seq_len, self.knn_neighbors)

        gathered_key = reshaped_key[batch_idx, head_idx, idx_expanded, :]
        gathered_value = reshaped_value[batch_idx, head_idx, idx_expanded, :]

        flat_query = reshaped_query.transpose(1, 2).reshape(batch_size * seq_len, self.n_head, 1, head_dim)
        flat_key = gathered_key.transpose(1, 2).reshape(batch_size * seq_len, self.n_head, self.knn_neighbors, head_dim)
        flat_value = gathered_value.transpose(1, 2).reshape(batch_size * seq_len, self.n_head, self.knn_neighbors, head_dim)

        attn_output = F.scaled_dot_product_attention(
            flat_query, flat_key, flat_value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
        )

        output = attn_output.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.proj(output)

        return output


class GatherSparseAttention(nn.Module):
    """Gather-based sparse attention with optional positional bias (KNN-aware)
    or RoPE.

    Precomputes KNN indices, gathers the corresponding keys/values, and
    runs ``scaled_dot_product_attention`` on ``N x knn_neighbors``
    tensors.  No ``N x N`` mask is needed in ``"none"`` mode, which
    enables FlashAttention kernels. Complexity: ``O(N * k * d)``.

    The gathered tensors are reshaped to
    ``(batch_size * seq_len, n_head, 1, knn_neighbors)`` so SDPA
    operates on many tiny ``1 x K`` attention problems.
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
            AssertionError: If ``embed_dim`` is not divisible by ``n_head``.
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
            raise ValueError(f"Unknown mode {mode}")

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

        reshaped_query = projected_query.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        reshaped_key = projected_key.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        reshaped_value = projected_value.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)

        # apply rope before gather
        if self._mode == "rope" and coords is not None:
            reshaped_query, reshaped_key = self.rot_pos_enc(reshaped_query, reshaped_key, coords)

        batch_idx = torch.arange(batch_size, device=query.device).view(batch_size, 1, 1, 1)
        head_idx = torch.arange(self.n_head, device=query.device).view(1, self.n_head, 1, 1)
        idx_expanded = knn_indices.unsqueeze(1).expand(batch_size, self.n_head, seq_len, self.knn_neighbors)

        gathered_key = reshaped_key[batch_idx, head_idx, idx_expanded, :]
        gathered_value = reshaped_value[batch_idx, head_idx, idx_expanded, :]

        flat_query = reshaped_query.transpose(1, 2).reshape(batch_size * seq_len, self.n_head, 1, -1)
        # transpose(1,2) creates a view; reshape on non-contiguous triggers a copy.
        # Explicit contiguous + del reduces peak memory: gathered_key is freed
        # before gathered_value is allocated.
        flat_key = gathered_key.transpose(1, 2).contiguous().view(batch_size * seq_len, self.n_head, self.knn_neighbors, -1)
        del gathered_key
        flat_value = gathered_value.transpose(1, 2).contiguous().view(batch_size * seq_len, self.n_head, self.knn_neighbors, -1)
        del gathered_value

        # bias mode: compute KNN-aware positional bias → shape (B, nH, N, K)
        attn_mask = None
        if self._mode == "bias" and coords is not None:
            attn_mask = self.pos_bias(coords, knn_indices).reshape(
                batch_size * seq_len, self.n_head, 1, self.knn_neighbors
            )

        attn_output = F.scaled_dot_product_attention(
            flat_query, flat_key, flat_value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
        )

        output = attn_output.view(batch_size, seq_len, self.n_head, -1).transpose(1, 2)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.proj(output)
        return output


class RelativePositionalBias(nn.Module):
    """Full-attention learnable relative positional bias.

    Computes pairwise spatial and temporal distances, bins them, and
    looks up a learnable bias for each ``(spatial-bin, temporal-bin)``
    pair. The result is a full
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

        spatial_bin_idx = torch.bucketize(spatial_dist, self.spatial_bins)
        torch.clamp_(spatial_bin_idx, max=len(self.spatial_bins) - 1)
        temporal_bin_idx = torch.bucketize(temporal_dist, self.temporal_bins)
        torch.clamp_(temporal_bin_idx, max=len(self.temporal_bins) - 1)

        # Flatten (spatial_idx, temporal_idx) into a single index so a
        # single ``index_select`` gathers the bias. This avoids slow
        # gather/scatter on multi-dimensional indices.
        # Ref: https://discuss.pytorch.org/t/how-to-select-multiple-indexes-over-multiple-dimensions-at-the-same-time/98532/2
        flat_idx = spatial_bin_idx.flatten() + temporal_bin_idx.flatten() * self.n_spatial
        bias = self.bias.index_select(0, flat_idx).view((*spatial_bin_idx.shape, self.n_head))
        # -> B, nH, N, N
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
                f"embed_dim {embed_dim} must be divisible by 2 times n_head {2 * n_head}"
            )

        # qkv projection
        self.q_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_pro = nn.Linear(embed_dim, embed_dim, bias=True)

        # output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        # regularization
        self.dropout = dropout
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.cutoff_spatial = cutoff_spatial
        self.attn_dist_mode = attn_dist_mode

        if mode == "bias" or mode is True:
            self.pos_bias = RelativePositionalBias(
                n_head=n_head,
                cutoff_spatial=cutoff_spatial,
                cutoff_temporal=cutoff_temporal,
                n_spatial=n_spatial,
                n_temporal=n_temporal,
            )
        elif mode == "rope":
            # each part needs to be divisible by 2
            n_split = 2 * (embed_dim // (2 * (coord_dim + 1) * n_head))

            self.rot_pos_enc = RotaryPositionalEncoding(
                cutoffs=((cutoff_temporal,) + (cutoff_spatial,) * coord_dim),
                n_pos=(embed_dim // n_head - coord_dim * n_split,)
                + (n_split,) * coord_dim,
            )
        elif mode == "none":
            pass
        elif mode is None or mode is False:
            logger.warning(
                "attn_positional_bias is not set (None or False), no positional bias."
            )
            pass
        else:
            raise ValueError(f"Unknown mode {mode}")

        self._mode = mode

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

        Raises:
            ValueError: If ``attn_dist_mode`` is unrecognised.
        """
        batch_size, seq_len, embed_dim = query.size()
        head_dim = embed_dim // self.n_head

        projected_query = self.q_pro(query)  # (B, N, D)
        projected_key = self.k_pro(key)  # (B, N, D)
        projected_value = self.v_pro(value)  # (B, N, D)
        # (B, nh, N, hs)
        reshaped_key = projected_key.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        reshaped_query = projected_query.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        reshaped_value = projected_value.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)

        attn_mask = torch.zeros(
            (batch_size, self.n_head, seq_len, seq_len),
            device=query.device,
            dtype=reshaped_query.dtype,
        )

        # add negative value but not too large to keep mixed precision loss from becoming nan
        attn_ignore_val = -1e3

        # spatial cutoff
        spatial_coords = coords[..., 1:]
        spatial_dist = torch.cdist(spatial_coords, spatial_coords)
        spatial_mask = (spatial_dist > self.cutoff_spatial).unsqueeze(1)
        attn_mask.masked_fill_(spatial_mask, attn_ignore_val)

        # dont add positional bias to self-attention if coords is None
        if coords is not None:
            if self._mode == "bias":
                attn_mask = attn_mask + self.pos_bias(coords)
            elif self._mode == "rope":
                reshaped_query, reshaped_key = self.rot_pos_enc(reshaped_query, reshaped_key, coords)
            else:
                pass

            if self.attn_dist_mode == "v0":
                full_dist = torch.cdist(coords, coords, p=2)
                attn_mask += torch.exp(-0.1 * full_dist.unsqueeze(1))
            elif self.attn_dist_mode == "v1":
                attn_mask += torch.exp(-5 * spatial_dist.unsqueeze(1) / self.cutoff_spatial)
            else:
                raise ValueError(f"Unknown attn_dist_mode {self.attn_dist_mode}")

        # if given key_padding_mask = (B,N) then ignore those tokens (e.g. padding tokens)
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
        # output projection
        output = self.proj(output)

        return output


class SpatialReorder:
    """Reorders sequence by spatial proximity for memory locality (Hassani et al. 2024).

    Quantizes coords to grid, computes linearized index (Z-order-like), sorts by it.
    Neighboring tokens in original space → close indices → gather hits contiguous memory.
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
            ``(batch_size, seq_len)``. ``reorder_idx`` sorts tokens
            by spatial proximity; ``unreorder_idx`` inverts the sort.
        """
        batch_size, seq_len, coord_dim = coords.shape
        normalized_coords = coords - coords.amin(dim=1, keepdim=True)
        normalized_coords = normalized_coords / (
            normalized_coords.amax(dim=1, keepdim=True) + 1e-8
        )
        grid_indices = (normalized_coords * self.n_bins).long().clamp(0, self.n_bins - 1)
        stride = 1
        linear_idx = torch.zeros(batch_size, seq_len, dtype=torch.long, device=coords.device)
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
        batch_idx = torch.arange(tensor.shape[0], device=tensor.device)
        if tensor.dim() == 3:
            return tensor[batch_idx[:, None], idx]
        elif tensor.dim() == 4:
            return tensor[batch_idx[:, None, None], :, idx]
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
