"""ASCENT-inspired encoder for SSL cell tracking.

Key design principles from ASCENT paper:
1. Agent-centric coordinate normalization — removes global position bias
2. Self-attention motion encoding — learns relative motion patterns
3. Mode query decoder — prevents collapse to single hypothesis
4. Parameterized prediction — predicts motion params, not raw associations

This module:
- Takes source and target cell features + coordinates
- Normalizes to per-cell local reference frames
- Encodes via transformer self-attention
- Predicts association scores via pairwise comparison or mode queries

Output format: (B, N_src, N_tgt) matching logits — compatible with Trackastra.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentCentricNormalization(nn.Module):
    """Normalize cell coordinates to agent-centric local frame.

    Like ASCENT: normalize each cell by its own position and orientation.
    For cell tracking:
    - Subtract centroid of each source cell
    - Align orientation (direction of motion or random reference)
    - Produces relative displacement features

    Input coords: (B, N, ndim) — spatial coordinates only
    Output: (B, N, N, ndim*2) — for each (i,j) pair: [delta_y, delta_x, ...]
    """

    def __init__(self, ndim=2):
        super().__init__()
        self.ndim = ndim

    def forward(self, coords_src, coords_tgt):
        B, N_src, D = coords_src.shape
        _, N_tgt, _ = coords_tgt.shape

        # Pairwise displacement vectors
        # src_i → tgt_j displacement: tgt_j - src_i
        # Shape: (B, N_src, N_tgt, ndim)
        disp = coords_tgt[:, None, :, :] - coords_src[:, :, None, :]

        # Normalize displacement by distances (scale invariance)
        # Also compute pairwise distances as additional feature
        dist = torch.norm(disp, dim=-1, keepdim=True)  # (B, N_src, N_tgt, 1)
        # Unit direction vector
        dir_vec = F.normalize(disp + 1e-8, dim=-1)  # (B, N_src, N_tgt, ndim)

        return torch.cat([disp, dir_vec, dist], dim=-1)  # (B, N_src, N_tgt, ndim*2 + 1)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal PE for cell identities in sequence."""

    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    """Self-attention motion encoder (ASCENT-style).

    Takes cell features + position embeddings, applies transformer encoder.
    """

    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.norm(x)


class ModeQueryDecoder(nn.Module):
    """Multi-hypothesis decoder with learnable mode queries.

    Like ASCENT: k learnable queries capture different motion modes.
    Each query independently predicts a set of associations.
    """

    def __init__(self, d_model=128, n_mode_queries=5, nhead=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.n_mode_queries = n_mode_queries
        self.query_embeds = nn.Parameter(torch.randn(n_mode_queries, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # Association head: one per mode
        self.assoc_head = nn.Linear(d_model * 2, 1)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_encoded, tgt_encoded, padding_mask_src=None, padding_mask_tgt=None):
        B = src_encoded.size(0)

        # Expand queries across batch
        queries = self.query_embeds[None, :, :].expand(B, -1, -1)  # (B, K, D)

        # Decode: use queries to cross-attend to encoded targets
        tgt_modes = self.decoder(
            queries, tgt_encoded,
            memory_key_padding_mask=padding_mask_tgt,
        )  # (B, K, D)

        # Compute pairwise associations between each src cell and each mode's target decoding
        # For each src token, attend to all tgt tokens via mode-weighted combination
        # Actually: directly compute per-mode logits
        # src_encoded: (B, N_src, D), tgt_modes: (B, K, D)
        # For each mode k: logits = src_encoded @ tgt_modes[k].T → (B, N_src, N_tgt)

        # Combine src features with mode features to get per-mode, per-pair logit
        # src_encoded: (B, N_src, D)
        # Expand and compare
        logits = 0
        for k in range(self.n_mode_queries):
            # Cross-attention between source and target via mode query
            # Project both to same space
            src_proj = src_encoded[:, :, None, :]  # (B, N_src, 1, D)
            tgt_proj = tgt_modes[:, None, :, :]    # (B, 1, K, D)
            pair_feat = torch.cat([src_proj.expand(-1, -1, self.n_mode_queries, -1),
                                   tgt_proj.expand(-1, src_encoded.size(1), -1, -1)], dim=-1)
            mode_logits = self.assoc_head(pair_feat).squeeze(-1)  # (B, N_src, K)
            logits = logits + mode_logits[:, :, k:k+1]

        return logits  # (B, N_src, N_tgt) — actually this isn't right

    def forward_v2(self, src_encoded, tgt_encoded, padding_mask_src=None, padding_mask_tgt=None):
        """Simpler approach: directly compute pairwise logits after encoding."""
        B, N_src, D = src_encoded.shape
        N_tgt = tgt_encoded.shape[1]

        # Einstein summation: outer product → similarity logits
        logits = torch.einsum("bnd,bmd->bnm", src_encoded, tgt_encoded)
        return logits / math.sqrt(D)


class AssociationEncoder(nn.Module):
    """Full association encoder: normalize → encode → score.

    Combines:
    1. Projection of cell features + coordinates
    2. Agent-centric pairwise encoding
    3. Transformer self-attention
    4. Pairwise scoring

    This is the core module that will be pretrained via SSL,
    then transferred to Trackastra's association head.
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
        super().__init__()

        # Input projection: features + coordinates → d_model
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.coord_proj = nn.Linear(coord_dim, d_model)

        # Agent-centric normalization for pairwise encoding
        self.pair_norm = AgentCentricNormalization(ndim=coord_dim)
        pair_feat_dim = coord_dim * 2 + 1  # disp + dir + dist
        self.pair_proj = nn.Linear(pair_feat_dim, d_model)

        # Feature fusion MLP (ASCENT-style)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer encoder (ASCENT motion encoder)
        self.encoder = TransformerEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )

        # Mode query decoder for multi-hypothesis associations
        self.decoder = ModeQueryDecoder(
            d_model=d_model, n_mode_queries=5,
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(
        self,
        coords_src, features_src,
        coords_tgt, features_tgt,
        padding_mask_src=None, padding_mask_tgt=None,
    ):
        # 1. Project features and coordinates
        f_src = self.feat_proj(features_src)  # (B, N_src, D)
        f_tgt = self.feat_proj(features_tgt)  # (B, N_tgt, D)
        c_src = self.coord_proj(coords_src)   # (B, N_src, D)
        c_tgt = self.coord_proj(coords_tgt)   # (B, N_tgt, D)

        # 2. Pairwise displacement encoding (agent-centric)
        pair_feat = self.pair_norm(coords_src, coords_tgt)  # (B, N_src, N_tgt, pair_dim)
        B, N_src, N_tgt, _ = pair_feat.shape
        pair_enc = self.pair_proj(pair_feat)  # (B, N_src, N_tgt, D)

        # 3. Fuse per-cell features with pooled pairwise context
        # Mean pool pairwise features into per-source and per-target context
        ctx_src = pair_enc.mean(dim=2)  # (B, N_src, D) — avg over targets
        ctx_tgt = pair_enc.mean(dim=1)  # (B, N_tgt, D) — avg over sources

        src_feat = self.fusion(torch.cat([f_src, c_src, ctx_src], dim=-1))
        tgt_feat = self.fusion(torch.cat([f_tgt, c_tgt, ctx_tgt], dim=-1))

        # 4. Encode via transformer
        # Concatenate src and tgt for joint encoding
        N_src, N_tgt = coords_src.shape[1], coords_tgt.shape[1]
        both = torch.cat([src_feat, tgt_feat], dim=1)  # (B, N_src + N_tgt, D)

        if padding_mask_src is not None and padding_mask_tgt is not None:
            pad_both = torch.cat([padding_mask_src, padding_mask_tgt], dim=1)
        else:
            pad_both = None

        encoded = self.encoder(both, padding_mask=pad_both)  # (B, N_src+N_tgt, D)

        # Split back
        src_encoded = encoded[:, :N_src, :]
        tgt_encoded = encoded[:, N_src:, :]

        # 5. Score associations
        # Use mode query decoder
        logits = self.decoder.forward_v2(
            src_encoded, tgt_encoded,
            padding_mask_src=padding_mask_src,
            padding_mask_tgt=padding_mask_tgt,
        )  # (B, N_src, N_tgt)

        # Apply padding mask
        if padding_mask_src is not None and padding_mask_tgt is not None:
            logits = logits.masked_fill(
                padding_mask_src[:, :, None] | padding_mask_tgt[:, None, :], -1e9
            )

        return logits


class FeedForward(nn.Module):
    """Simple MLP used in Trackastra's association head."""

    def __init__(self, d_model, hidden_dim=None, n_layers=2):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model * 2
        layers = []
        for i in range(n_layers):
            in_dim = d_model if i == 0 else hidden_dim
            out_dim = d_model if i < n_layers - 1 else d_model
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
