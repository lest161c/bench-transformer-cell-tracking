#!/usr/bin/env python3
"""Training convergence ablation: attention × K × L × N × init.

Measures train/val loss over multiple epochs for every
(attention, init) pair.  This is the **convergence** phase of the
ablation — the speed-scaling phase belongs in ``benchmark_attn``.

The model is a simplified association model (no O(N²) pair norm)
that projects ``(features, coords)`` into ``d_model`` and encodes
the concatenated src+tgt sequence.

Hardware notes
--------------
- Designed for the local A500 (4 GB) GPU.
- All variants OOM at N ≥ 1024 on 4 GB.

Output CSV columns: ``phase, attn, L, K, N, init, epoch, time_s,
mem_mb, train_loss, val_loss, train_acc, val_acc, status``.

Usage
-----
::

    python scripts/benchmarks/benchmark_ablation.py
    python scripts/benchmarks/benchmark_ablation.py --epochs 20
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARK_TRAINING = _WORKSPACE_ROOT / "benchmark_training"
_BENCH_ATTN = _WORKSPACE_ROOT / "benchmark_attn"
for _p in (_BENCHMARK_TRAINING, _BENCH_ATTN):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.bench_utils import get_device, log_device_info
from src.models import (
    AgentCentricNormalization,
    DenseEncoder,
    SinusoidalPositionalEncoding,
    SparseEncoder,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Model
# ------------------------------------------------------------------ #

class AblationModel(nn.Module):
    """Simplified association model for ablation: no O(N²) pair norm.

    Projects ``(features, coords)`` into ``d_model`` via a single
    linear layer, then encodes the concatenated src+tgt sequence.
    A linear head scores every src-tgt pair.

    Parameters
    ----------
    encoder : nn.Module
        A :class:`DenseEncoder` or :class:`SparseEncoder`.
    feat_dim : int, default 7
    coord_dim : int, default 2
    d_model : int, default 128
    """

    def __init__(
        self,
        encoder: nn.Module,
        feat_dim: int = 7,
        coord_dim: int = 2,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(feat_dim + coord_dim, d_model)
        self.encoder = encoder
        self.head = nn.Linear(d_model * 2, 1)

    def forward(
        self,
        cs: torch.Tensor,
        fs: torch.Tensor,
        ct: torch.Tensor,
        ft: torch.Tensor,
    ) -> torch.Tensor:
        src_in = torch.cat([fs, cs], dim=-1)
        tgt_in = torch.cat([ft, ct], dim=-1)
        src = self.proj(src_in)
        tgt = self.proj(tgt_in)
        both = torch.cat([src, tgt], dim=1)
        coords = torch.cat([cs, ct], dim=1)
        encoded = self.encoder(both, coords=coords)
        n_src = cs.shape[1]
        src_enc = encoded[:, :n_src]
        tgt_enc = encoded[:, n_src:]
        b, n1, d = src_enc.shape
        n2 = tgt_enc.shape[1]
        src_e = src_enc[:, :, None, :].expand(-1, -1, n2, -1)
        tgt_e = tgt_enc[:, None, :, :].expand(-1, n1, -1, -1)
        return self.head(torch.cat([src_e, tgt_e], dim=-1)).squeeze(-1)


# ------------------------------------------------------------------ #
# Synthetic data
# ------------------------------------------------------------------ #

def make_synthetic_batch(
    num_cells: int,
    batch_size: int = 2,
    feat_dim: int = 7,
    coord_dim: int = 2,
) -> dict:
    """Generate a synthetic src→tgt batch with identity association."""
    cs = torch.randn(batch_size, num_cells, coord_dim) * 100
    ct = cs + torch.randn(batch_size, num_cells, coord_dim) * 3
    fs = torch.randn(batch_size, num_cells, feat_dim)
    ft = torch.randn(batch_size, num_cells, feat_dim)
    a = torch.eye(num_cells).unsqueeze(0).expand(batch_size, -1, -1).float()
    return {"cs": cs, "fs": fs, "ct": ct, "ft": ft, "a": a}


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #

def main() -> None:
    """Run the convergence ablation benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Number of convergence epochs (default: 15).",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging.")
    args = parser.parse_args()

    device = get_device()
    log_device_info(device)

    results_dir = _WORKSPACE_ROOT / "benchmark_training" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "full_attention_ablation_results.csv"

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "phase", "attn", "L", "K", "N", "init", "epoch",
        "time_s", "mem_mb", "train_loss", "val_loss", "train_acc", "val_acc", "status",
    ])

    D, H = 128, 4
    L_vals = [1, 4]
    conv_N = 512

    attn_configs = [
        ("dense", lambda L: DenseEncoder(D, H, L)),
        ("sparseK4", lambda L: SparseEncoder(D, H, L, knn_k=4)),
        ("sparseK16", lambda L: SparseEncoder(D, H, L, knn_k=16)),
    ]

    for L in L_vals:
        for attn_name, attn_fn in attn_configs:
            K = 0 if attn_name == "dense" else (4 if "K4" in attn_name else 16)
            model = AblationModel(attn_fn(L), d_model=D).to(device)
            batch = make_synthetic_batch(conv_N, B=2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            pos_weight = torch.tensor(10.0, device=device)
            moved = {k: v.to(device) for k, v in batch.items()}

            for epoch in range(1, args.epochs + 1):
                try:
                    t0 = time.perf_counter()
                    model.train()
                    optimizer.zero_grad()
                    logits = model(moved["cs"], moved["fs"], moved["ct"], moved["ft"])
                    loss = F.binary_cross_entropy_with_logits(logits, moved["a"], pos_weight=pos_weight)
                    loss.backward()
                    optimizer.step()
                    tl = loss.item()
                    ta = ((logits.detach() > 0).float() == moved["a"]).float().mean().item()

                    model.eval()
                    with torch.no_grad():
                        vl = F.binary_cross_entropy_with_logits(
                            model(moved["cs"], moved["fs"], moved["ct"], moved["ft"]),
                            moved["a"], pos_weight=pos_weight
                        ).item()
                        va = ((logits > 0).float() == moved["a"]).float().mean().item()

                    et = time.perf_counter() - t0
                    status = "ok"
                except RuntimeError:
                    tl, ta, vl, va, et = -1, -1, -1, -1, -1
                    status = "oom"

                csv_writer.writerow(["converge", attn_name, L, K, conv_N, "rand", epoch,
                                     f"{et:.6f}" if et >= 0 else "",
                                     "",
                                     f"{tl:.6f}" if tl >= 0 else "",
                                     f"{vl:.6f}" if vl >= 0 else "",
                                     f"{ta:.4f}" if ta >= 0 else "",
                                     f"{va:.4f}" if va >= 0 else "",
                                     status])
                csv_file.flush()

                if status == "oom":
                    break

            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    csv_file.close()
    logger.info("Done. Results: %s", csv_path)


if __name__ == "__main__":
    main()
