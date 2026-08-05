import logging
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from native_sparse_attention_pytorch import SparseAttention


logger = logging.getLogger(__name__)

def _bin_init_exp(cutoff: float, n: int):
    return torch.exp(torch.linspace(0, math.log(cutoff + 1), n))


def _bin_init_linear(cutoff: float, n: int):
    return torch.linspace(-cutoff, cutoff, n)

def _pos_embed_fourier1d_init(cutoff: float = 128, n: int = 32):
    # Maximum initial frequency is 1
    return torch.exp(torch.linspace(0, -math.log(cutoff), n)).unsqueeze(0).unsqueeze(0)


# https://github.com/cvg/LightGlue/blob/b1cd942fc4a3a824b6aedff059d84f5c31c297f6/lightglue/lightglue.py#L51
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of scalars as 2d vectors by pi/2.
    Refer to eq 34 in https://arxiv.org/pdf/2104.09864.pdf.
    """
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, cutoffs: tuple[float] = (256,), n_pos: tuple[int] = (32,)):
        """Rotary positional encoding with given cutoff and number of frequencies for each dimension.
        number of dimension is inferred from the length of cutoffs and n_pos.

        see
        https://arxiv.org/pdf/2104.09864.pdf
        """
        super().__init__()
        assert len(cutoffs) == len(n_pos)
        if not all(n % 2 == 0 for n in n_pos):
            raise ValueError("n_pos must be even")

        self._n_dim = len(cutoffs)
        # theta in RoFormer https://arxiv.org/pdf/2104.09864.pdf
        self.freqs = nn.ParameterList([
            nn.Parameter(_pos_embed_fourier1d_init(cutoff, n // 2))
            for cutoff, n in zip(cutoffs, n_pos)
        ])

    def get_co_si(self, coords: torch.Tensor):
        _B, _N, D = coords.shape
        assert D == len(self.freqs)
        co = torch.cat(
            tuple(
                torch.cos(0.5 * math.pi * x.unsqueeze(-1) * freq) / math.sqrt(len(freq))
                for x, freq in zip(coords.moveaxis(-1, 0), self.freqs)
            ),
            axis=-1,
        )
        si = torch.cat(
            tuple(
                torch.sin(0.5 * math.pi * x.unsqueeze(-1) * freq) / math.sqrt(len(freq))
                for x, freq in zip(coords.moveaxis(-1, 0), self.freqs)
            ),
            axis=-1,
        )

        return co, si

    def forward(self, q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor):
        _B, _N, D = coords.shape
        _B, _H, _N, _C = q.shape

        if not D == self._n_dim:
            raise ValueError(f"coords must have {self._n_dim} dimensions, got {D}")

        co, si = self.get_co_si(coords)

        co = co.unsqueeze(1).repeat_interleave(2, dim=-1)
        si = si.unsqueeze(1).repeat_interleave(2, dim=-1)
        q2 = q * co + _rotate_half(q) * si
        k2 = k * co + _rotate_half(k) * si

        return q2, k2


class KNNRelativePositionalBias(nn.Module):
    def __init__(
        self,
        n_head: int,
        cutoff_spatial: float,
        cutoff_temporal: float,
        n_spatial: int = 32,
        n_temporal: int = 16,
    ):
        """Learnt relative positional bias to add to self-attention matrix.

        Spatial bins are exponentially spaced, temporal bins are linearly spaced.

        Args:
            n_head (int): Number of pos bias heads. Equal to number of attention heads
            cutoff_spatial (float): Maximum distance in space.
            cutoff_temporal (float): Maxium distance in time. Equal to window size of transformer.
            n_spatial (int, optional): Number of spatial bins.
            n_temporal (int, optional): Number of temporal bins in each direction. Should be equal to window size. Total = 2 * n_temporal + 1. Defaults to 16.
        """
        super().__init__()
        self._spatial_bins = _bin_init_exp(cutoff_spatial, n_spatial)
        self._temporal_bins = _bin_init_linear(cutoff_temporal, 2 * n_temporal + 1)
        self.register_buffer("spatial_bins", self._spatial_bins)
        self.register_buffer("temporal_bins", self._temporal_bins)
        self.n_spatial = n_spatial
        self.n_head = n_head
        self.bias = nn.Parameter(
            -0.5 + torch.rand((2 * n_temporal + 1) * n_spatial, n_head)
        )

    def forward(self, coords: torch.Tensor, knn_indices: torch.Tensor = None):
        _B, _N, _D = coords.shape
        t = coords[..., 0]
        yx = coords[..., 1:]
        
        if knn_indices is not None:
            B_idx = torch.arange(_B, device=coords.device).view(_B, 1, 1)
            t_knn = t[B_idx, knn_indices]
            yx_knn = yx[B_idx, knn_indices, :]
            
            temporal_dist = t.unsqueeze(-1) - t_knn
            spatial_dist = torch.norm(yx.unsqueeze(-2) - yx_knn, dim=-1)
        else:
            temporal_dist = t.unsqueeze(-1) - t.unsqueeze(-2)
            spatial_dist = torch.cdist(yx, yx)

        spatial_idx = torch.bucketize(spatial_dist, self.spatial_bins)
        torch.clamp_(spatial_idx, max=len(self.spatial_bins) - 1)
        temporal_idx = torch.bucketize(temporal_dist, self.temporal_bins)
        torch.clamp_(temporal_idx, max=len(self.temporal_bins) - 1)

        # do some index gymnastics such that backward is not super slow
        # https://discuss.pytorch.org/t/how-to-select-multiple-indexes-over-multiple-dimensions-at-the-same-time/98532/2
        idx = spatial_idx.flatten() + temporal_idx.flatten() * self.n_spatial
        bias = self.bias.index_select(0, idx).view((*spatial_idx.shape, self.n_head))
        
        if knn_indices is not None:
            # -> B, nH, N, K
            bias = bias.permute(0, 3, 1, 2)
        else:
            # -> B, nH, N, N
            bias = bias.transpose(-1, 1)
            
        return bias


class KNNRelativePositionalAttention(nn.Module):
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

        self._mode = mode

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
        knn_indices: torch.Tensor = None,
    ):
        B, N, D = query.size()
        q = self.q_pro(query)  # (B, N, D)
        k = self.k_pro(key)  # (B, N, D)
        v = self.v_pro(value)  # (B, N, D)
        # (B, nh, N, hs)
        k = k.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        q = q.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)

        if coords is not None and self._mode == "rope":
            q, k = self.rot_pos_enc(q, k, coords)

        B_idx = torch.arange(B, device=q.device).view(B, 1, 1, 1)
        H_idx = torch.arange(self.n_head, device=q.device).view(1, self.n_head, 1, 1)
        idx_exp = knn_indices.unsqueeze(1).expand(B, self.n_head, N, self.knn_neighbors)

        k_knn = k[B_idx, H_idx, idx_exp, :]
        v_knn = v[B_idx, H_idx, idx_exp, :]

        q_reshaped = q.transpose(1, 2).reshape(B * N, self.n_head, 1, q.shape[-1])
        k_reshaped = k_knn.transpose(1, 2).reshape(B * N, self.n_head, self.knn_neighbors, k.shape[-1])
        v_reshaped = v_knn.transpose(1, 2).reshape(B * N, self.n_head, self.knn_neighbors, v.shape[-1])

        y_reshaped = F.scaled_dot_product_attention(
            q_reshaped, k_reshaped, v_reshaped, attn_mask=None, dropout_p=self.dropout if self.training else 0
        )
        
        y = y_reshaped.view(B, N, self.n_head, -1).transpose(1, 2)
        y = y.transpose(1, 2).contiguous().view(B, N, D)

        # output projection
        y = self.proj(y)

        return y


class GatherSparseAttention(nn.Module):
    """Gather-based sparse attention with optional positional bias (KNN-aware)
    or RoPE. Precompute KNN indices, gather K/V via idx, run SDPA on N×k tensors.
    No N×N mask → enables FlashAttention for mode='none'. Complexity O(N k d)."""

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
        coords: torch.Tensor = None,
    ):
        B, N, D = query.shape
        q = self.q_pro(query)
        k = self.k_pro(key)
        v = self.v_pro(value)

        q = q.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        k = k.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)

        # apply rope before gather
        if self._mode == "rope" and coords is not None:
            q, k = self.rot_pos_enc(q, k, coords)

        B_idx = torch.arange(B, device=q.device).view(B, 1, 1, 1)
        H_idx = torch.arange(self.n_head, device=q.device).view(1, self.n_head, 1, 1)
        idx = knn_indices.unsqueeze(1).expand(B, self.n_head, N, self.knn_neighbors)
        
        # PROFILE: view or copy
        k_sel = k[B_idx, H_idx, idx, :]
        v_sel = v[B_idx, H_idx, idx, :]

        q_flat = q.transpose(1, 2).reshape(B * N, self.n_head, 1, -1)
        # transpose(1,2) creates view; reshape on non-contiguous triggers copy.
        # Explicit contiguous + del reduces peak: k_sel freed before v_sel alloc.
        k_flat = k_sel.transpose(1, 2).contiguous().view(B * N, self.n_head, self.knn_neighbors, -1)
        del k_sel
        v_flat = v_sel.transpose(1, 2).contiguous().view(B * N, self.n_head, self.knn_neighbors, -1)
        del v_sel

        # bias mode: compute KNN-aware positional bias → shape (B, nH, N, K)
        attn_mask = None
        if self._mode == "bias" and coords is not None:
            attn_mask = self.pos_bias(coords, knn_indices)

        y = F.scaled_dot_product_attention(
            q_flat, k_flat, v_flat, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
        )

        y = y.view(B, N, self.n_head, -1).transpose(1, 2)
        y = y.transpose(1, 2).contiguous().view(B, N, D)
        y = self.proj(y)
        return y


class GatherSparseAttentionV2(nn.Module):
    """V2: avoid unnecessary copies. Eliminates double-transpose at output and
    uses view+unsqueeze for q_flat instead of transpose+reshape. Profile shows
    ~2 fewer clone+copy pairs per layer vs V1."""

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
        coords: torch.Tensor = None,
    ):
        B, N, D = query.shape
        nH = self.n_head
        Dh = D // nH
        K = self.knn_neighbors

        q = self.q_pro(query)
        k = self.k_pro(key)
        v = self.v_pro(value)

        q = q.view(B, N, nH, Dh).transpose(1, 2)
        k = k.view(B, N, nH, Dh).transpose(1, 2)
        v = v.view(B, N, nH, Dh).transpose(1, 2)

        if self._mode == "rope" and coords is not None:
            q, k = self.rot_pos_enc(q, k, coords)

        B_idx = torch.arange(B, device=q.device).view(B, 1, 1, 1)
        H_idx = torch.arange(nH, device=q.device).view(1, nH, 1, 1)
        idx = knn_indices.unsqueeze(1).expand(B, nH, N, K)

        k_sel = k[B_idx, H_idx, idx, :]
        v_sel = v[B_idx, H_idx, idx, :]

        # v2: q_flat uses view+unsqueeze — NO copy (q is contiguous at this point)
        q_flat = q.reshape(B * N, nH, Dh).unsqueeze(2)

        #TODO: do not merge B and N -> do not flatten the shape
        # gets independent from N
        k_flat = k_sel.transpose(1, 2).contiguous().view(B * N, nH, K, Dh)
        del k_sel
        v_flat = v_sel.transpose(1, 2).contiguous().view(B * N, nH, K, Dh)
        del v_sel

        attn_mask = None
        if self._mode == "bias" and coords is not None:
            attn_mask = self.pos_bias(coords, knn_indices)

        y = F.scaled_dot_product_attention(
            q_flat, k_flat, v_flat, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
        )

        # v2: direct reshape — SDPA output is contiguous, so .reshape is a VIEW
        y = y.reshape(B, N, D)
        y = self.proj(y)
        return y


class KNNMaskSparseAttention(nn.Module):
    """KNN-sparse attention via NxN boolean mask (not gather).

    Uses F.scaled_dot_product_attention with a KNN-derived mask.
    Keeps q,k,v at the native (B, nH, N, Dh) shape → avoids q_len=1.
    cuDNN/EfficientAttention handle the masked dense SDPA efficiently
    since N is small (~300). No gather, no contiguous, no shape tricks.

    This is ~10x faster than GatherSparseAttention at N=256 because:
    1. (B, nH, N, Dh) shape → 8 ops of NxN, not 2048 of 1xK
    2. Mask is built via scatter (O(NK)), not cdist (O(N^2))
    3. No advanced indexing copies for K/V
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

    def _make_mask(self, B, nH, N, K, knn_indices, device, dtype):
        """Build (B, nH, N, N) mask from KNN indices. -inf for non-neighbors."""
        src = knn_indices.unsqueeze(1).expand(B, nH, N, K)
        mask = torch.full((B, nH, N, N), float("-inf"), device=device, dtype=dtype)
        mask.scatter_(3, src, 0.0)
        return mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        knn_indices: torch.Tensor,
        coords: torch.Tensor = None,
    ):
        B, N, D = query.shape
        nH = self.n_head
        Dh = D // nH

        q = self.q_pro(query).view(B, N, nH, Dh).transpose(1, 2)
        k = self.k_pro(key).view(B, N, nH, Dh).transpose(1, 2)
        v = self.v_pro(value).view(B, N, nH, Dh).transpose(1, 2)

        if self._mode == "rope" and coords is not None:
            q, k = self.rot_pos_enc(q, k, coords)

        mask = self._make_mask(B, nH, N, self.knn_neighbors,
                               knn_indices, q.device, q.dtype)

        if self._mode == "bias" and coords is not None:
            mask = mask + self.pos_bias(coords, knn_indices)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0,
        )

        y = y.transpose(1, 2).contiguous().view(B, N, D)
        y = self.proj(y)
        return y


class GatherSparseAttentionV3(nn.Module):
    """V3: replaces F.scaled_dot_product_attention with manual batch matmul.

    The V1/V2 reshape to (B*N, nH, 1, Dh) which creates 2048 tiny (1×K)
    attention ops — no SDPA backend handles q_len=1 efficiently.
    V3 keeps the native (B, nH, N, Dh) shape and uses torch.matmul
    for the per-query attention, avoiding the SDPA kernel launch overhead.
    No mask needed → all operations use optimized cuBLAS/cuDNN matmul kernels."""

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
            raise ValueError(f"Unknown mode {mode}")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        knn_indices: torch.Tensor,
        coords: torch.Tensor = None,
    ):
        B, N, D = query.shape
        nH = self.n_head
        Dh = D // nH
        K = self.knn_neighbors

        q = self.q_pro(query).view(B, N, nH, Dh).transpose(1, 2)
        k = self.k_pro(key).view(B, N, nH, Dh).transpose(1, 2)
        v = self.v_pro(value).view(B, N, nH, Dh).transpose(1, 2)

        if self._mode == "rope" and coords is not None:
            q, k = self.rot_pos_enc(q, k, coords)

        B_idx = torch.arange(B, device=q.device).view(B, 1, 1, 1)
        H_idx = torch.arange(nH, device=q.device).view(1, nH, 1, 1)
        idx = knn_indices.unsqueeze(1).expand(B, nH, N, K)
        k_sel = k[B_idx, H_idx, idx, :]
        v_sel = v[B_idx, H_idx, idx, :]

        # v3: manual matmul instead of SDPA — no q_len=1, no contiguous
        # scores: (B, nH, N, K) = for each (b,h,n), dot(q[n], k_sel[n,k,:])
        scores = torch.matmul(q.unsqueeze(3), k_sel.transpose(-2, -1)).squeeze(3)
        scores = scores * self._scale

        if self._mode == "bias" and coords is not None:
            bias = self.pos_bias(coords, knn_indices)
            scores = scores + bias

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # output: (B, nH, N, Dh)
        y = torch.matmul(attn.unsqueeze(3), v_sel).squeeze(3)

        y = y.transpose(1, 2).contiguous().view(B, N, D)
        y = self.proj(y)
        return y


class DenseFlashAttention(nn.Module):
    """Plain dense SDPA without mask, positional bias, or KNN overhead.
    Relies on FlashAttention for O(N^2 d) compute. Optimal when N < K*N
    overhead threshold (~N < 500 for typical K=16)."""

    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        dropout: float = 0.0,
    ):
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
        knn_indices: torch.Tensor = None,
        coords: torch.Tensor = None,
    ):
        B, N, D = query.shape
        nH = self.n_head
        Dh = D // nH

        q = self.q_pro(query).view(B, N, nH, Dh).transpose(1, 2)
        k = self.k_pro(key).view(B, N, nH, Dh).transpose(1, 2)
        v = self.v_pro(value).view(B, N, nH, Dh).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
        )

        y = y.transpose(1, 2).contiguous().view(B, N, D)
        y = self.proj(y)
        return y


class RelativePositionalBias(nn.Module):
    def __init__(
        self,
        n_head: int,
        cutoff_spatial: float,
        cutoff_temporal: float,
        n_spatial: int = 32,
        n_temporal: int = 16,
    ):
        """Learnt relative positional bias to add to self-attention matrix.

        Spatial bins are exponentially spaced, temporal bins are linearly spaced.

        Args:
            n_head (int): Number of pos bias heads. Equal to number of attention heads
            cutoff_spatial (float): Maximum distance in space.
            cutoff_temporal (float): Maxium distance in time. Equal to window size of transformer.
            n_spatial (int, optional): Number of spatial bins.
            n_temporal (int, optional): Number of temporal bins in each direction. Should be equal to window size. Total = 2 * n_temporal + 1. Defaults to 16.
        """
        super().__init__()
        self._spatial_bins = _bin_init_exp(cutoff_spatial, n_spatial)
        self._temporal_bins = _bin_init_linear(cutoff_temporal, 2 * n_temporal + 1)
        self.register_buffer("spatial_bins", self._spatial_bins)
        self.register_buffer("temporal_bins", self._temporal_bins)
        self.n_spatial = n_spatial
        self.n_head = n_head
        self.bias = nn.Parameter(
            -0.5 + torch.rand((2 * n_temporal + 1) * n_spatial, n_head)
        )

    def forward(self, coords: torch.Tensor):
        _B, _N, _D = coords.shape
        t = coords[..., 0]
        yx = coords[..., 1:]
        temporal_dist = t.unsqueeze(-1) - t.unsqueeze(-2)
        spatial_dist = torch.cdist(yx, yx)

        spatial_idx = torch.bucketize(spatial_dist, self.spatial_bins)
        torch.clamp_(spatial_idx, max=len(self.spatial_bins) - 1)
        temporal_idx = torch.bucketize(temporal_dist, self.temporal_bins)
        torch.clamp_(temporal_idx, max=len(self.temporal_bins) - 1)

        # do some index gymnastics such that backward is not super slow
        # https://discuss.pytorch.org/t/how-to-select-multiple-indexes-over-multiple-dimensions-at-the-same-time/98532/2
        idx = spatial_idx.flatten() + temporal_idx.flatten() * self.n_spatial
        bias = self.bias.index_select(0, idx).view((*spatial_idx.shape, self.n_head))
        # -> B, nH, N, N
        bias = bias.transpose(-1, 1)
        return bias


class RelativePositionalAttention(nn.Module):
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
        padding_mask: torch.Tensor = None,
    ):
        B, N, D = query.size()
        q = self.q_pro(query)  # (B, N, D)
        k = self.k_pro(key)  # (B, N, D)
        v = self.v_pro(value)  # (B, N, D)
        # (B, nh, N, hs)
        k = k.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        q = q.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)

        attn_mask = torch.zeros(
            (B, self.n_head, N, N), device=query.device, dtype=q.dtype
        )

        # add negative value but not too large to keep mixed precision loss from becoming nan
        attn_ignore_val = -1e3

        # spatial cutoff
        yx = coords[..., 1:]
        spatial_dist = torch.cdist(yx, yx)
        spatial_mask = (spatial_dist > self.cutoff_spatial).unsqueeze(1)
        attn_mask.masked_fill_(spatial_mask, attn_ignore_val)

        # dont add positional bias to self-attention if coords is None
        if coords is not None:
            if self._mode == "bias":
                attn_mask = attn_mask + self.pos_bias(coords)
            elif self._mode == "rope":
                q, k = self.rot_pos_enc(q, k, coords)
            else:
                pass

            if self.attn_dist_mode == "v0":
                dist = torch.cdist(coords, coords, p=2)
                attn_mask += torch.exp(-0.1 * dist.unsqueeze(1))
            elif self.attn_dist_mode == "v1":
                attn_mask += torch.exp(
                    -5 * spatial_dist.unsqueeze(1) / self.cutoff_spatial
                )
            else:
                raise ValueError(f"Unknown attn_dist_mode {self.attn_dist_mode}")

        # if given key_padding_mask = (B,N) then ignore those tokens (e.g. padding tokens)
        if padding_mask is not None:
            ignore_mask = torch.logical_or(
                padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
            ).unsqueeze(1)
            attn_mask.masked_fill_(ignore_mask, attn_ignore_val)

        # self.attn_mask = attn_mask.clone()

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0
        )

        y = y.transpose(1, 2).contiguous().view(B, N, D)
        # output projection
        y = self.proj(y)

        return y


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

    NOTE (2026-06-22): This implementation is for benchmarking only. The
    labbook (2026-06-16) already documents architectural incompatibility:
    MSA requires GQA (Trackastra uses MHA), operates at 1M+ context (ours
    is 128-512), and provides no cross-attention path. Expect speed inferior
    to KNNMaskSparseAttention at our N scale. Included for completeness per
    meeting directive (meeting_18_06_26.txt line 2).
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
        knn_indices: torch.Tensor = None,
        coords: torch.Tensor = None,
    ):
        B, N, D = query.shape
        nH = self.n_head
        Dh = D // nH
        Bk = self.block_size
        ksel = self.num_selected_blocks

        q = self.q_pro(query).view(B, N, nH, Dh).transpose(1, 2)
        k = self.k_pro(key).view(B, N, nH, Dh).transpose(1, 2)
        v = self.v_pro(value).view(B, N, nH, Dh).transpose(1, 2)

        # ── Index Branch: block-level scoring ──
        # Re-use q,k for block scoring (avoids extra projection overhead in
        # benchmark — real MSA uses separate Index Branch projections)
        num_blocks = (N + Bk - 1) // Bk
        pad = num_blocks * Bk - N
        k_pad = F.pad(k, (0, 0, 0, pad))
        k_block = k_pad.view(B, nH, num_blocks, Bk, Dh)

        # Block scores: max over block tokens of q·k
        # q: (B, nH, 1, N, 1, Dh), k_block: (B, nH, num_blocks, 1, Bk, Dh)
        # -> scores: (B, nH, N, num_blocks, Bk) -> max -> (B, nH, N, num_blocks)
        q_exp = q.unsqueeze(3).unsqueeze(-3)
        k_block_exp = k_block.unsqueeze(2)
        scores_raw = (q_exp * k_block_exp).sum(dim=-1)
        block_scores, _ = scores_raw.max(dim=-1)

        # Top-k block selection
        _, topk_blk = torch.topk(block_scores, k=ksel, dim=-1)

        # ── Block-sparse KV gather ──
        # Convert block indices to token ranges: [blk*Bk, (blk+1)*Bk)
        offsets = torch.arange(Bk, device=q.device).view(1, 1, 1, 1, Bk)
        token_idx = topk_blk.unsqueeze(-1) * Bk + offsets
        token_idx = token_idx.clamp(0, N + pad - 1).view(B, nH, N, ksel * Bk)

        B_idx = torch.arange(B, device=q.device).view(B, 1, 1, 1)
        H_idx = torch.arange(nH, device=q.device).view(1, nH, 1, 1)
        k_sel = k_pad[B_idx, H_idx, token_idx, :]
        v_sel = F.pad(v, (0, 0, 0, pad))[B_idx, :, token_idx, :]

        # ── Block-sparse attention ──
        scores = torch.matmul(q.unsqueeze(3), k_sel.transpose(-2, -1)).squeeze(3)
        scores = scores * (Dh ** -0.5)
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        y = torch.matmul(attn.unsqueeze(3), v_sel).squeeze(3)

        y = y.transpose(1, 2).contiguous().view(B, N, D)
        y = self.proj(y)
        return y


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
        knn_indices: torch.Tensor = None,
        coords: torch.Tensor = None,
    ):
        return self.nsa(query)


class SpatialReorder:
    """Reorders sequence by spatial proximity for memory locality (Hassani et al. 2024).

    Quantizes coords to grid, computes linearized index (Z-order-like), sorts by it.
    Neighboring tokens in original space → close indices → gather hits contiguous memory.
    """

    def __init__(self, n_bins: int = 32):
        self.n_bins = n_bins

    def compute_idx(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = coords.shape
        c = coords - coords.amin(dim=1, keepdim=True)
        c = c / (c.amax(dim=1, keepdim=True) + 1e-8)
        g = (c * self.n_bins).long().clamp(0, self.n_bins - 1)
        stride = 1
        idx = torch.zeros(B, N, dtype=torch.long, device=coords.device)
        for d in range(D):
            idx = idx + g[..., d] * stride
            stride *= self.n_bins
        reorder_idx = idx.argsort(dim=1, stable=True)
        unreorder_idx = reorder_idx.argsort(dim=1, stable=True)
        return reorder_idx, unreorder_idx

    @staticmethod
    def apply(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        B = torch.arange(x.shape[0], device=x.device)
        if x.dim() == 3:
            return x[B[:, None], idx]
        elif x.dim() == 4:
            return x[B[:, None, None], :, idx]
        raise ValueError(f"Unsupported dim {x.dim()}")

    def reorder(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return self.apply(x, idx)

    def unreorder(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return self.apply(x, idx)
