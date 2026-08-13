"""ScaledCNN: contrastive CNN encoder for cell patch embeddings."""

import torch.nn as nn

# Patch size the CNN is designed for (64×64 grayscale cell patches).
PATCH_SIZE = 64


class ScaledCNN(nn.Module):
    """ConvNet for 64×64 grayscale cell patches. Output: 128-dim embedding.

    Used as the frozen feature extractor in the CNN-augmented tracking
    experiments and as the online encoder in NT-Xent/BYOL pretraining.
    Checkpoint loading relies on this architecture matching the saved
    state dict, so keep the layer layout stable.

    Args:
        scale: 'small' (~20K params), 'medium' (~100K params), 'large' (~200K params)
        out_dim: embedding dimension (default 128)
    """
    def __init__(self, scale='large', out_dim=128):
        """Build the CNN; `scale` selects the channel widths per layer."""
        super().__init__()
        if scale == 'small':
            ch = [8, 16, 32]           # 3 conv layers
        elif scale == 'medium':
            ch = [16, 32, 64]          # 3 conv layers
        else:  # large
            ch = [32, 64, 128, 256]    # 4 conv layers

        layers = []
        in_ch = 1
        for channel_dim in ch:
            layers += [nn.Conv2d(in_ch, channel_dim, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = channel_dim
        self.conv = nn.Sequential(*layers)
        # After pooling: small/medium: 64 → 32 → 16 → 8,  large: 64 → 32 → 16 → 8 → 4
        spatial = PATCH_SIZE // (2 ** len(ch))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch[-1] * spatial * spatial, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier-initialize all 2D weight tensors with reduced gain."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.5)

    def forward(self, patches):
        """patches: (N, 1, 64, 64) → embeddings: (N, out_dim)"""
        return self.fc(self.conv(patches))
