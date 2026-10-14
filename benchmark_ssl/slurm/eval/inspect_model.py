"""Inspect a trained Trackastra model's pretrained-feature branch (SPEC 0033 S4).

Why: to argue whether the DynaCLR features are actually consumed, inspect the
learned ``ptfeat_proj``/``ptfeat_norm`` weights and the input projection.

Usage:
    python inspect_model.py <model.pt>
"""

import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    """Print projection-branch statistics for one trained model."""
    state = torch.load(Path(sys.argv[1]), map_location="cpu", weights_only=False)
    weight = state["ptfeat_proj.0.weight"].numpy()
    singular_values = np.linalg.svd(weight, compute_uv=False)
    participation_rank = float(singular_values.sum() ** 2 / (singular_values ** 2).sum())
    print("ptfeat_proj.weight shape:", weight.shape)
    print("ptfeat_proj.weight fro norm:", float(np.linalg.norm(weight)))
    print("ptfeat_proj.bias norm:", float(state["ptfeat_proj.0.bias"].norm()))
    print("singular values:", np.round(singular_values, 4).tolist())
    print("effective rank (participation ratio):", round(participation_rank, 2))
    norm_w = state["ptfeat_norm.weight"].numpy()
    print("ptfeat_norm.weight mean/std:", round(float(norm_w.mean()), 4), round(float(norm_w.std()), 4))
    print("input proj.weight shape:", tuple(state["proj.weight"].shape))


if __name__ == "__main__":
    main()
