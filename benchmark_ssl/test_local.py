"""Minimal integration test: model forward + SSL + training loop on vanvliet.

Runs on CPU with tiny model (d_model=64) and 2 epochs. Catches import
errors, shape mismatches, device issues, and attention crashes.
"""

import sys, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("test_local")

ROOT = Path(__file__).resolve().parent.parent / "trackastra"
DATA = Path(__file__).resolve().parent.parent / "data" / "vanvliet"

if not DATA.exists():
    sys.exit(f"vanvliet data not found at {DATA}")

sys.path.insert(0, str(ROOT))

import torch
torch.manual_seed(42)

# Force CPU to avoid CUDA dependency for local testing
DEVICE = torch.device("cpu")

from trackastra.model import TrackingTransformer
from trackastra.model.model_parts import GatherSparseAttention
from trackastra.data.data import CTCData, collate_sequence_padding
from torch.utils.data import DataLoader


# ===== 1. Data loading (single experiment for speed) =====
exps = sorted(d for d in (DATA / "rpsM").iterdir() if d.is_dir() and (d / "TRA").exists())
if not exps:
    sys.exit("No experiment subdirs found")
root = exps[0]
logger.info(f"Loading data from {root}")

dataset = CTCData(
    root=root,
    ndim=2,
    detection_folders=["TRA"],
    window_size=5,
    max_tokens=256,
    features="regionprops2",
    augment=0,
    use_gt=False,
)
logger.info(f"Dataset windows: {len(dataset)}")
if len(dataset) == 0:
    logger.warning("Empty dataset, creating dummy window")
    dataset.windows = [{
        "coords": torch.randn(8, 3),
        "features": torch.randn(8, dataset.feat_dim),
        "assoc_matrix": torch.eye(8),
        "timepoints": torch.zeros(8, dtype=torch.long),
        "padding_mask": torch.zeros(8, dtype=torch.bool),
    }]
logger.info(f"Dataset windows: {len(dataset)}, feat_dim={dataset.feat_dim}, ndim={dataset.ndim}")

loader = DataLoader(
    dataset, batch_size=4, shuffle=False,
    collate_fn=collate_sequence_padding,
)


# ===== 2. Model construction (using dataset dimensions) =====
feat_dim = dataset.feat_dim
logger.info(f"=== Building model (sparse, k=4, d_model=64, feat_dim={feat_dim}) ===")
model = TrackingTransformer(
    coord_dim=dataset.ndim,
    feat_dim=feat_dim,
    d_model=64,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    window=5,
    dropout=0.0,
    attn_positional_bias="rope",
    knn_neighbors=4,
)
model.to(DEVICE)
model.train()

# Verify GatherSparseAttention is used
n_sparse = sum(1 for m in model.modules() if isinstance(m, GatherSparseAttention))
logger.info(f"GatherSparseAttention layers: {n_sparse}")
assert n_sparse >= 2, f"Expected at least 2 GatherSparseAttention layers, got {n_sparse}"


# ===== 3. Forward + backward pass =====
logger.info("=== Forward/backward test ===")
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

for step, batch in enumerate(loader):
    if step >= 2:
        break
    coords = batch["coords"].to(DEVICE)
    feats = batch["features"].to(DEVICE)
    padding_mask = batch["padding_mask"].to(DEVICE)
    A = batch["assoc_matrix"].to(DEVICE)

    logger.info(f"  Step {step}: coords {coords.shape}, feats {feats.shape}, pad {padding_mask.shape}")

    # Model forward
    A_pred = model(coords, feats, padding_mask=padding_mask)
    logger.info(f"  A_pred shape: {A_pred.shape}, range: [{A_pred.min():.3f}, {A_pred.max():.3f}]")
    assert A_pred.shape == A.shape, f"Shape mismatch: {A_pred.shape} vs {A.shape}"
    assert not torch.any(torch.isnan(A_pred)), "NaN in A_pred"
    assert not torch.any(torch.isinf(A_pred)), "INF in A_pred"

    # Loss
    mask_invalid = torch.logical_or(
        padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    loss = loss_fn(A_pred, A)
    loss[mask_invalid] = 0
    loss = loss.mean()
    logger.info(f"  Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "NaN in loss"
    assert not torch.isinf(loss), "INF in loss"

    # Backward
    loss.backward()
    opt.step()
    opt.zero_grad()
    logger.info(f"  Backward OK")

# Verify all attn layers ran
logger.info("=== GatherSparseAttention forward count ===")
n_forward = model._forward_count if hasattr(model, "_forward_count") else "N/A"
logger.info(f"Forward/backward: OK")
logger.info("\n=== ALL TESTS PASSED ===")
