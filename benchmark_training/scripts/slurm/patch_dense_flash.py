"""Patch: Add DenseFlashAttention to trackastra.

This adds DenseFlashAttention class to model_parts.py and wires it into
the model factory. DenseFlashAttention is plain dense attention without
any mask — it gets the FlashAttention-2 kernel, making it the fastest
attention method at small N.

Usage: set knn_neighbors=-2 to select DenseFlashAttention.
"""

# Add DenseFlashAttention class after CachedDistAttention in model_parts.py
MODEL_PARTS_ADDITION = '''


class DenseFlashAttention(nn.Module):
    """Plain dense attention without any mask — gets FlashAttention-2 kernel.

    No spatial cutoff mask, no KNN mask, no positional bias. Just Q/K/V
    projections and F.scaled_dot_product_attention. This is the fastest
    attention method at small N because it uses the fused FlashAttention-2
    kernel without any mask overhead.

    Use knn_neighbors=-2 in the config to select this attention type.
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

# Replace the attn_factory block in model.py to include DenseFlashAttention
MODEL_PY_REPLACEMENT = '''        if knn_neighbors == -2:
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
            attn_factory = lambda: GatherSparseAttention(
                coord_dim,
                d_model,
                nhead,
                cutoff_spatial=spatial_pos_cutoff,
                cutoff_temporal=window,
                dropout=dropout,
                mode=attn_positional_bias,
                attn_dist_mode=attn_dist_mode,
                knn_neighbors=knn_neighbors,
            )
        else:
            attn_factory = lambda: CachedDistAttention('''

MODEL_PARTS_PATH = (
    "/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/"
    "trackastra/trackastra/model/model_parts.py"
)
MODEL_PATH = (
    "/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/"
    "trackastra/trackastra/model/model.py"
)


def apply_patch():
    """Apply the DenseFlashAttention patch to trackastra files."""
    # 1. Add DenseFlashAttention to model_parts.py
    with open(MODEL_PARTS_PATH) as f:
        parts_content = f.read()
    if "class DenseFlashAttention" not in parts_content:
        with open(MODEL_PARTS_PATH, "a") as f:
            f.write(MODEL_PARTS_ADDITION)
        print("Added DenseFlashAttention to model_parts.py")
    else:
        print("DenseFlashAttention already in model_parts.py")

    # 2. Add DenseFlashAttention import and factory in model.py
    with open(MODEL_PATH) as f:
        model_content = f.read()
    if "DenseFlashAttention" not in model_content:
        # Add import
        model_content = model_content.replace(
            "CachedDistAttention,",
            "CachedDistAttention,\n    DenseFlashAttention,",
        )
        # Replace attn_factory block
        old_factory = """        if knn_neighbors > 0:
            attn_factory = lambda: GatherSparseAttention(
                coord_dim,
                d_model,
                nhead,
                cutoff_spatial=spatial_pos_cutoff,
                cutoff_temporal=window,
                dropout=dropout,
                mode=attn_positional_bias,
                attn_dist_mode=attn_dist_mode,
                knn_neighbors=knn_neighbors,
            )
        else:
            attn_factory = lambda: CachedDistAttention("""
        model_content = model_content.replace(old_factory, MODEL_PY_REPLACEMENT)
        with open(MODEL_PATH, "w") as f:
            f.write(model_content)
        print("Added DenseFlashAttention import and factory in model.py")
    else:
        print("DenseFlashAttention already in model.py")


if __name__ == "__main__":
    apply_patch()
