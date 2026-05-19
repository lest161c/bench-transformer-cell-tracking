# SSL Pretraining for Cell Tracking via Geometric Distortion

Benchmarking self-supervised pretraining strategies for Trackastra's association head.
Inspired by ASCENT paper (agent-centric normalization, self-attention encoder, mode query decoder).

## Setup

```bash
uv sync           # install deps
mkdir -p runs     # for checkpoints + logs
```

## Pipeline

```
Data (masks + images) → WRFeatures → Distortion → Synthetic pair → Encoder → Association loss
                                          ↑
                                   5 distortion families:
                                   affine, elastic, jitter, dropout, photometric
```

## Structure

```
├── distortions.py      # 5 distortion families + DistortionPipeline
├── ssl_pipeline.py     # Data loading + SSLDataset + collator
├── track_encoder.py    # AssociationEncoder (ASCENT-inspired)
├── pretrain.py         # Training loop + validation
├── config.yaml         # Hyperparameters
├── README.md
└── pyproject.toml
```

## Usage

```bash
# Test data pipeline
python ssl_pipeline.py

# Train SSL (requires GPU for full 50 epochs)
python pretrain.py config.yaml

# View logs
tensorboard --logdir runs/
```

## Key Design (why this differs from previous failed SSL)

Previous attempt: shallow augmentation pretraining → model learned warp geometry, not cell identity.

This approach (ASCENT-inspired):
1. **Agent-centric normalization** — each cell in its own local reference frame
2. **Self-attention encoder** — 4-layer transformer over all cells jointly
3. **Multimodal prediction** — mode query decoder prevents collapse
4. **Distortion diversity** — 5 families prevent shortcut memorization
5. **Hardest pretext** — per-cell jitter (independent motion) forces learning cell identity features

## Distortion Families

| Family | Effect | Why needed |
|--------|--------|------------|
| Affine | Global rotation, scale, shear | Tests global warp invariance |
| Elastic | Non-linear local deformation | Tests local deformation invariance |
| Jitter | Per-cell random displacement | *Hardest* — forces cell identity learning |
| Dropout | Random cell removal | Teaches robustness to detection failures |
| Photometric | Intensity shifts | Only relevant for image-based features |

## Integration with Trackastra

After SSL pretraining:
```python
from track_encoder import AssociationEncoder
model = AssociationEncoder(feat_dim=7, coord_dim=2)
ckpt = torch.load("runs/ssl_v1/best_model.pt")
model.load_state_dict(ckpt["model_state_dict"])
# Extract encoder weights → Trackastra association head
```

## Config

See `config.yaml`. Key params:
- `distortions`: list of active distortion families
- Per-distortion params (degrees, scale, sigma, etc.)
- `encoder`: d_model, nhead, num_layers
- `training`: lr, batch_size, epochs
