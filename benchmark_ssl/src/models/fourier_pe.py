"""Fourier positional encoding for probe experiments.

Maps spatial coordinates (t, y, x) to a higher-dimensional
sin/cos representation using geometrically decaying frequencies.

This replaces raw coordinate features with a smooth, bounded
encoding that captures both absolute position and relative
proximity through the frequency structure.
"""

import math

import torch
from torch import nn


def _init_fourier_frequencies(cutoff: float = 128, n: int = 16) -> torch.Tensor:
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


class FourierPE(nn.Module):
    """Fourier positional encoding: sin/cos of coords at
    geometrically decaying frequencies.

    Each coordinate dimension gets its own set of frequency
    vectors, initialised geometrically from 1 (Nyquist) to
    ``1/cutoff``. The output is the concatenation of sin and cos
    features for each dimension, giving ``coord_dim * n_freqs * 2``
    output dimensions.

    Args:
        coord_dim: Number of coordinate dimensions (e.g. 3 for
            t, y, x).
        n_freqs: Number of frequency components per dimension.
        cutoff: Controls the decay rate; larger values give
            slower decay (wider spatial reach).
    """

    def __init__(
        self,
        coord_dim: int = 3,
        n_freqs: int = 16,
        cutoff: float = 128.0,
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.n_freqs = n_freqs
        self.pe_dim = coord_dim * n_freqs * 2
        self.freqs = nn.Parameter(
            _init_fourier_frequencies(cutoff, n_freqs),
            requires_grad=False,
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Map coordinates to Fourier PE features.

        Args:
            coords: Tensor of shape ``(seq_len, coord_dim)`` or
                ``(batch_size, seq_len, coord_dim)``.

        Returns:
            Tensor of shape ``(seq_len, pe_dim)`` or
            ``(batch_size, seq_len, pe_dim)`` where
            ``pe_dim = coord_dim * n_freqs * 2``.
        """
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)

        _, _, coord_dim = coords.shape # batch_size, seq_len, coord_dim
        assert coord_dim == self.coord_dim

        parts = []
        for dim_idx in range(coord_dim):
            coord = coords[:, :, dim_idx].unsqueeze(-1)
            arg = coord * self.freqs
            parts.append(torch.sin(arg))
            parts.append(torch.cos(arg))

        result = torch.cat(parts, dim=-1)
        return result.squeeze(0) if result.shape[0] == 1 else result
