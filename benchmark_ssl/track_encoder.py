"""Contrastive cell embedding encoder — ASCENT NETr-inspired.

Produces per-cell embeddings from coords + shallow features.
Processes each frame independently via transformer self-attention.
No pairwise comparisons during encoding. Embeddings are compared post-hoc
via cosine similarity for tracking and contrastive learning.

Removed: AgentCentricNormalization, ModeQueryDecoder (from wrong ASCENT paper).
Implemented: CellEmbedder — dual-encoder that maps (coords, features) → z.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CellEmbedder(nn.Module):
    """Produces per-cell embeddings from shallow features + coordinates.

    Implements core of ASCENT NETr architecture:
    1. Feature MLP projection  → d_model
    2. Coordinate MLP encoding → d_model
    3. Element-wise addition to fuse modalities
    4. Transformer encoder for context-aware refinement
    5. Output projection head → per-cell embedding vector

    The output embeddings are used in contrastive learning (NT-Xent)
    during SSL pretraining. For downstream tracking, cosine similarity
    between embeddings from consecutive frames drives bipartite matching.
    """

    def __init__(
        self,
        feat_dim=7,
        coord_dim=2,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
    ):
        """Build the feature/coordinate encoders, transformer, and head.

        Args:
            feat_dim: dimensionality of input features per cell.
            coord_dim: spatial dimensionality of the coordinates.
            d_model: transformer embedding dimension.
            nhead: number of attention heads per transformer layer.
            num_layers: number of transformer encoder layers.
            dim_feedforward: hidden dimension of the transformer feedforward.
            dropout: dropout probability inside the transformer layers.
        """
        super().__init__()

        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.coord_enc = nn.Sequential(
            nn.Linear(coord_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        self.embedding_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize all weight matrices with Xavier uniform (gain 0.5)."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.5)

    def forward(self, coords, features, padding_mask=None):
        """Encode a frame of cells into per-cell embeddings.

        Args:
            coords:     (B, N, coord_dim) — spatial coordinates
            features:   (B, N, feat_dim)  — regionprops features
            padding_mask: (B, N) — True for padded positions

        Returns:
            embeddings: (B, N, d_model) — per-cell embeddings
        """
        feature_tokens = self.feat_proj(features)
        coord_tokens = self.coord_enc(coords)
        hidden = feature_tokens + coord_tokens

        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=padding_mask)
        hidden = self.enc_norm(hidden)

        embeddings = self.embedding_head(hidden)
        return embeddings

    def encode(self, coords, features, padding_mask=None):
        """Alias for forward — produces L2-normalized embeddings for matching."""
        embeddings = self.forward(coords, features, padding_mask)
        return F.normalize(embeddings, dim=-1)


if __name__ == "__main__":
    pass
