#!/usr/bin/env python3
"""Smoke test: CNN-augmented Trackastra on A500. Catches import bugs, path
issues, shape mismatches before H100 submission."""

import sys, warnings, torch
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent  # research-proj/
TRACKASTRA = PROJECT_ROOT / "trackastra"
sys.path.insert(0, str(TRACKASTRA))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = PROJECT_ROOT / "data" / "vanvliet"
CKPT = Path(__file__).parent / "probe" / "cnn_ntxent_large.pt"


def test_import():
    """1. Circular import: import Trackastra with use_cnn=True."""
    from trackastra.model import TrackingTransformer
    from trackastra.data import CTCData, collate_sequence_padding
    print("✓ Import OK")
    return TrackingTransformer, CTCData, collate_sequence_padding


def test_model_creation(TrackingTransformer):
    """2. CNN checkpoint loading: verify shapes."""
    assert CKPT.exists(), f"Checkpoint not found: {CKPT}"
    m = TrackingTransformer(coord_dim=2, feat_dim=7, d_model=128, nhead=4,
        num_encoder_layers=2, num_decoder_layers=2, window=3,
        feat_embed_per_dim=8, use_cnn=True, cnn_checkpoint=str(CKPT)).to(device)
    assert hasattr(m, "cnn_encoder") and hasattr(m, "cnn_proj")
    dummy = torch.randn(4, 1, 64, 64, device=device)
    with torch.no_grad():
        emb = m.cnn_encoder(dummy)
    assert emb.shape == (4, 128), f"Expected (4,128), got {emb.shape}"
    print(f"✓ Model creation OK ({sum(p.numel() for p in m.parameters()):,} params)")
    return m


def test_data_pipeline(CTCData, collate):
    """3. Data pipeline: load vanvliet, verify patches_cnn in batch."""
    exp = sorted(p for p in (DATA_ROOT / "rpsM").iterdir()
                 if p.is_dir() and (p / "TRA").exists())
    assert exp, f"No TRA/ experiments in {DATA_ROOT}/rpsM"
    data = CTCData(root=str(exp[0]), ndim=2, features="wrfeat",
                   window_size=3, use_cnn=True, compress=False)
    assert len(data) > 0
    sample = data[0]
    assert "patches_cnn" in sample, f"Missing patches_cnn in {list(sample.keys())}"
    assert sample["patches_cnn"].shape[-3:] == (1, 64, 64), \
        f"Expected (N,1,64,64), got {sample['patches_cnn'].shape}"
    print(f"✓ Data pipeline OK ({len(data)} windows, {sample['patches_cnn'].shape[0]} cells)")
    return data


def test_forward_pass(m, data, collate):
    """4. Forward pass with CNN patches, verify output shape + finiteness."""
    batch = collate([data[0]])
    for k in ("coords", "features", "patches_cnn", "padding_mask"):
        if k in batch and batch[k] is not None:
            batch[k] = batch[k].to(device)
    m.eval()
    with torch.no_grad():
        A = m(coords=batch["coords"], features=batch["features"],
              padding_mask=batch["padding_mask"],
              patches_cnn=batch["patches_cnn"])
    B, N = batch["coords"].shape[:2]
    assert A.shape == (B, N, N), f"Expected ({B},{N},{N}), got {A.shape}"
    assert torch.isfinite(A).all(), "Non-finite values in output"
    print(f"✓ Forward pass OK (output {tuple(A.shape)})")


def test_training_step(m, data, collate):
    """5. One training step, verify loss is finite."""
    batch = collate([data[0]] * 2)  # batch size 2
    for k in ("coords", "features", "patches_cnn", "assoc_matrix",
              "timepoints", "padding_mask"):
        if k in batch and batch[k] is not None:
            batch[k] = batch[k].to(device)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    opt.zero_grad()
    A = m(coords=batch["coords"], features=batch["features"],
          padding_mask=batch["padding_mask"],
          patches_cnn=batch["patches_cnn"])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(A, batch["assoc_matrix"])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    assert torch.isfinite(loss), "Loss is not finite!"
    print(f"✓ Training step OK (loss={loss.item():.4f})")


if __name__ == "__main__":
    print(f"Device: {device}\nCheckpoint: {CKPT}\n")
    T, CTC, collate = test_import()
    model = test_model_creation(T)
    dataset = test_data_pipeline(CTC, collate)
    test_forward_pass(model, dataset, collate)
    test_training_step(model, dataset, collate)
    print("\n=== ALL TESTS PASSED ===")
