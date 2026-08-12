"""Positional-encoding helpers for the attention modules.

Extracted from the former ``model_parts.py`` so the attention classes in
``attention_modules.py`` can import them without carrying the bin/encoding
machinery inline.  This module provides:

  - ``ATTN_IGNORE_VALUE``: sentinel value used to mask out non-neighbour
    tokens in attention masks.
  - ``_init_exponential_bins`` / ``_init_linear_bins``: build the bin
    edges used by the relative positional bias modules for spatial and
    temporal distance bucketing.
  - ``_init_fourier_frequencies``: geometrically decaying frequency
    vectors for the rotary encoder.
  - ``_rotate_half``: the RoPE pair-rotation primitive.
  - :class:`RotaryPositionalEncoding`: rotary positional encoding for
    multi-dimensional coordinates.

All helpers are pure tensor functions; callers are responsible for device
placement.  Tensors are float32 unless noted otherwise.
"""

import math

import torch
from torch import nn

# Sentinel value used to mask out non-neighbour tokens in attention masks.
# Kept moderate (not ``-inf``) to avoid NaN gradients in mixed-precision
# training with autocast.
ATTN_IGNORE_VALUE: float = -1e3


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
    The last dimension is split into consecutive pairs ``(x1, x2)`` and
    replaced with ``(-x2, x1)``.

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
    (Nyquist) and the minimum is ``1/cutoff``.  See
    `RoFormer <https://arxiv.org/pdf/2104.09864.pdf>`_ for the
    mathematical background.

    The encoding rotates pairs of channels in the query/key tensors
    by an angle proportional to the coordinate value, providing a
    smooth inductive bias for spatial proximity.
    """

    def __init__(
        self,
        cutoffs: tuple[float, ...] = (256.0,),
        n_pos: tuple[int, ...] = (32,),
    ):
        """Initialise RoPE with per-dimension frequency vectors.

        Args:
            cutoffs: Per-dimension frequency decay constants.  Larger
                values give slower decay (wider spatial reach).
            n_pos: Per-dimension number of frequency components.  Must
                be even.  The number of coordinate dimensions is
                inferred from ``len(cutoffs)``.

        Raises:
            ValueError: If ``len(cutoffs) != len(n_pos)`` or any
                element of ``n_pos`` is odd.
        """
        super().__init__()
        if len(cutoffs) != len(n_pos):
            raise ValueError(
                f"cutoffs ({len(cutoffs)}) and n_pos ({len(n_pos)}) "
                "must have the same length"
            )
        if not all(n % 2 == 0 for n in n_pos):
            raise ValueError("n_pos must be even")

        self.n_dim = len(cutoffs)
        self.freqs = nn.ParameterList([
            nn.Parameter(_init_fourier_frequencies(cutoff, n // 2))
            for cutoff, n in zip(cutoffs, n_pos)
        ])

    def get_cosine_sine(
        self, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cosine and sine encodings for each coordinate dimension.

        For each dimension ``d`` the encoding is
        ``cos(0.5 * pi * freq * coord_d)`` (and the sine analogue),
        normalised by ``1/sqrt(n_dim)``.

        Args:
            coords: Tensor of shape ``(batch_size, seq_len, n_dim)``.

        Returns:
            A tuple ``(cosine_encoding, sine_encoding)`` each of shape
            ``(batch_size, seq_len, total_freqs)`` where
            ``total_freqs = sum(len(freq) for freq in self.freqs)``.
        """
        n_dim = coords.shape[-1]
        if n_dim != len(self.freqs):
            raise ValueError(
                f"coords last dimension ({n_dim}) must match "
                f"number of frequency vectors ({len(self.freqs)})"
            )

        cos_parts = []
        sin_parts = []
        for dim_idx, freq in enumerate(self.freqs):
            coord_dim = coords[..., dim_idx].unsqueeze(-1)
            cos_parts.append(
                torch.cos(0.5 * math.pi * coord_dim * freq)
                / math.sqrt(len(self.freqs))
            )
            sin_parts.append(
                torch.sin(0.5 * math.pi * coord_dim * freq)
                / math.sqrt(len(self.freqs))
            )

        return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)

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
        """
        cosine_encoding, sine_encoding = self.get_cosine_sine(coords)

        cosine_encoding = cosine_encoding.unsqueeze(1).repeat_interleave(2, dim=-1)
        sine_encoding = sine_encoding.unsqueeze(1).repeat_interleave(2, dim=-1)

        rotated_query = query * cosine_encoding + _rotate_half(query) * sine_encoding
        rotated_key = key * cosine_encoding + _rotate_half(key) * sine_encoding

        return rotated_query, rotated_key
