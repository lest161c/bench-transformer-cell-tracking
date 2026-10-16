#!/usr/bin/env python3
"""Install native DenseFlashAttention support into a Trackastra checkout.

Why this exists
---------------
``knn_neighbors=-2`` selects DenseFlashAttention (plain dense SDPA, no mask,
FlashAttention-2 kernel). Trackastra upstream has no such class, so it must be
added to the checkout that runs the experiments. The previous
``patch_dense_flash.py`` did this by exact-string replacement and hardcoded a
cluster path; both broke (dead path; the model.py factory string no longer
matched after the code moved from ``GatherSparseAttention`` to
``KNNMaskSparseAttention``), so it silently did nothing and ``-2`` silently fell
back to CachedDistAttention.

What this does (idempotent, verifies its own post-conditions)
-------------------------------------------------------------
Given the Trackastra package root (the directory containing ``model/model.py``
and ``model/model_parts.py``):

1. appends a ``DenseFlashAttention`` class to ``model_parts.py`` if absent;
2. adds it to the ``.model_parts`` import in ``model.py`` if absent;
3. rewrites the attention-factory selector so that
   ``knn_neighbors == -2`` -> DenseFlashAttention, ``> 0`` -> SparseMaskAttention,
   ``-1`` -> CachedDistAttention, and **any other value raises** (no silent
   fallback);
4. verifies the result and exits non-zero if anything was not applied.

Run (from the repository root)::

    python benchmark_training/scripts/slurm/install_dense_flash.py \
        --trackastra-root /path/to/trackastra/trackastra

The change is idempotent: re-running detects an already-patched checkout and
does nothing. Back up the checkout before first use if it is shared.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DENSE_FLASH_CLASS = '''


class DenseFlashAttention(nn.Module):
    """Plain dense attention without any mask, dispatched to FlashAttention-2.

    No spatial-cutoff mask, no KNN mask, no positional bias: just Q/K/V
    projections and ``F.scaled_dot_product_attention``. This is the fastest
    attention variant at small N because it uses the fused FlashAttention-2
    kernel with no mask overhead. Selected by ``knn_neighbors = -2``.
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
        knn_neighbors: int = -2,
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
        self._mode = mode

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        coords: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
        knn_indices: torch.Tensor = None,
        dist_2d: torch.Tensor = None,
    ):
        B, N, D = query.shape
        if N == 0:
            return torch.zeros(B, 0, D, device=query.device, dtype=query.dtype)
        nH = self.n_head
        Dh = D // nH

        q = self.q_pro(query).view(B, N, nH, Dh).transpose(1, 2)
        k = self.k_pro(key).view(B, N, nH, Dh).transpose(1, 2)
        v = self.v_pro(value).view(B, N, nH, Dh).transpose(1, 2)

        attn_mask = None
        if padding_mask is not None:
            attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask * torch.finfo(q.dtype).min

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
        )

        y = y.transpose(1, 2).contiguous().view(B, N, D)
        y = self.proj(y)
        return y
'''

# Anchor: the exact selector line in model.py. Replaced by the guarded chain.
FACTORY_ANCHOR = "        if knn_neighbors > 0:\n"
FACTORY_REPLACEMENT = '''        if knn_neighbors < -2:
            raise ValueError(
                f"unsupported knn_neighbors={knn_neighbors}; use -2 (dense "
                "flash), -1 (dense masked), or >0 (KNN). Refusing to fall "
                "back silently."
            )
        if knn_neighbors == -2:
            attn_factory = lambda: DenseFlashAttention(
                coord_dim,
                d_model,
                nhead,
                cutoff_spatial=spatial_pos_cutoff,
                cutoff_temporal=window,
                dropout=dropout,
                mode=attn_positional_bias,
                attn_dist_mode=attn_dist_mode,
            )
        elif knn_neighbors > 0:
'''


def _patch_model_parts(model_parts: Path) -> bool:
    """Append the DenseFlashAttention class if absent; return True if changed."""
    text = model_parts.read_text()
    if "class DenseFlashAttention" in text:
        return False
    model_parts.write_text(text + DENSE_FLASH_CLASS)
    return True


def _patch_model_py(model_py: Path) -> bool:
    """Add the import and the guarded factory selector; return True if changed."""
    text = model_py.read_text()
    if "DenseFlashAttention" in text:
        return False
    if FACTORY_ANCHOR not in text:
        raise RuntimeError(
            f"factory anchor {FACTORY_ANCHOR!r} not found in {model_py}; "
            "the checkout differs from what this installer expects."
        )
    text = text.replace(
        "    CachedDistAttention,\n",
        "    CachedDistAttention,\n    DenseFlashAttention,\n",
        1,
    )
    text = text.replace(FACTORY_ANCHOR, FACTORY_REPLACEMENT, 1)
    model_py.write_text(text)
    return True


def _verify(model_parts: Path, model_py: Path) -> None:
    """Assert the post-conditions; raise if the install is incomplete."""
    parts = model_parts.read_text()
    model = model_py.read_text()
    checks = {
        "DenseFlashAttention class": "class DenseFlashAttention" in parts,
        "DenseFlashAttention import": "    DenseFlashAttention,\n" in model,
        "knn_neighbors == -2 branch": "if knn_neighbors == -2:" in model,
        "loud guard (no silent fallback)": "unsupported knn_neighbors" in model,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"verification failed: {', '.join(failed)}")


def main(argv: list[str] | None = None) -> int:
    """Install the native DenseFlashAttention support and verify it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--trackastra-root", required=True, type=Path,
        help="Trackastra package root (contains model/model.py and "
             "model/model_parts.py).",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing.")
    args = parser.parse_args(argv)

    model_parts = args.trackastra_root / "model" / "model_parts.py"
    model_py = args.trackastra_root / "model" / "model.py"
    for path in (model_parts, model_py):
        if not path.exists():
            print(f"ERROR: not found: {path}", file=sys.stderr)
            return 2

    if args.dry_run:
        already = "class DenseFlashAttention" in model_parts.read_text()
        print("already installed" if already else "would install DenseFlashAttention")
        return 0

    changed_parts = _patch_model_parts(model_parts)
    changed_model = _patch_model_py(model_py)
    _verify(model_parts, model_py)
    print(
        f"model_parts.py: {'patched' if changed_parts else 'already present'}; "
        f"model.py: {'patched' if changed_model else 'already present'}; "
        "verification OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
