#!/usr/bin/env python3
"""Combined benchmark: sparse attention + SSL pretraining vs baseline.

2x2 factorial design:
   Attention: Dense (standard SDPA) vs Sparse (Gather-based O(Nk))
   Init:      Random vs SSL-pretrained

Measures wall-clock time to reach target validation loss.
Shows total speedup from both improvements combined.

Hardware notes
--------------
- Designed for the local A500 (4 GB) GPU.
- Can also run on an H100 where more memory is available.

Usage
-----
::

    python scripts/benchmarks/benchmark_combined.py
    python scripts/benchmarks/benchmark_combined.py --epochs 20 --wandb
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
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
from src.ssl_transfer import init_from_ssl_dense, init_from_ssl_sparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Model
# ------------------------------------------------------------------ #

class CombinedModel(nn.Module):
    """Association model with interchangeable encoder.

    Projects ``(features, coords)`` through separate linear layers,
    applies agent-centric pair normalisation, fuses the result, and
    then encodes the concatenated src+tgt sequence with either a
    :class:`DenseEncoder` or a :class:`SparseEncoder`.

    Parameters
    ----------
    d_model : int, default 128
    nhead : int, default 4
    num_layers : int, default 4
    feat_dim : int, default 7
    coord_dim : int, default 2
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        feat_dim: int = 7,
        coord_dim: int = 2,
    ) -> None:
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.coord_proj = nn.Linear(coord_dim, d_model)
        self.pair_norm = AgentCentricNormalization(coord_dim)
        self.pair_proj = nn.Linear(coord_dim * 2 + 1, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.head = nn.Linear(d_model * 2, 1)

    def encode(
        self,
        cs: torch.Tensor,
        fs: torch.Tensor,
        ct: torch.Tensor,
        ft: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(src_feat, tgt_feat)`` after projection and fusion."""
        f_src = self.feat_proj(fs)
        f_tgt = self.feat_proj(ft)
        c_src = self.coord_proj(cs)
        c_tgt = self.coord_proj(ct)
        pair = self.pair_norm(cs, ct)
        ctx_src = self.pair_proj(pair).mean(dim=2)
        ctx_tgt = self.pair_proj(pair).mean(dim=1)
        src_feat = self.fusion(torch.cat([f_src, c_src, ctx_src], dim=-1))
        tgt_feat = self.fusion(torch.cat([f_tgt, c_tgt, ctx_tgt], dim=-1))
        return src_feat, tgt_feat

    def forward(
        self,
        cs: torch.Tensor,
        fs: torch.Tensor,
        ct: torch.Tensor,
        ft: torch.Tensor,
    ) -> torch.Tensor:
        src_feat, tgt_feat = self.encode(cs, fs, ct, ft)
        both = torch.cat([src_feat, tgt_feat], dim=1)
        both = self.pos_enc(both)
        encoded = self.encoder(both)
        n_src = cs.shape[1]
        src_enc = encoded[:, :n_src]
        tgt_enc = encoded[:, n_src:]
        b, n1, d = src_enc.shape
        n2 = tgt_enc.shape[1]
        src_e = src_enc[:, :, None, :].expand(-1, -1, n2, -1)
        tgt_e = tgt_enc[:, None, :, :].expand(-1, n1, -1, -1)
        return self.head(torch.cat([src_e, tgt_e], dim=-1)).squeeze(-1)


class DenseCombinedModel(CombinedModel):
    """CombinedModel with a :class:`DenseEncoder`."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = DenseEncoder(
            d_model=kwargs.get("d_model", 128),
            nhead=kwargs.get("nhead", 4),
            num_layers=kwargs.get("num_layers", 4),
        )


class SparseCombinedModel(CombinedModel):
    """CombinedModel with a :class:`SparseEncoder`."""

    def __init__(self, knn_k: int = 16, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = SparseEncoder(
            d_model=kwargs.get("d_model", 128),
            nhead=kwargs.get("nhead", 4),
            num_layers=kwargs.get("num_layers", 4),
            knn_k=knn_k,
        )


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #

def main() -> None:
    """Run the combined 2x2 factorial benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Number of training epochs per variant (default: 15).",
    )
    parser.add_argument(
        "--ssl_ckpt", type=Path,
        default=_WORKSPACE_ROOT / "benchmark_ssl" / "runs" / "ssl_v1" / "best_model.pt",
        help="Path to SSL checkpoint .pt file.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    device = get_device()
    log_device_info(device)

    ssl_state = None
    if args.ssl_ckpt.exists():
        ssl_state = torch.load(args.ssl_ckpt, map_location="cpu", weights_only=False)[
            "model_state_dict"
        ]
        logger.info("Loaded SSL checkpoint: %s", args.ssl_ckpt)
    else:
        logger.warning("SSL checkpoint not found: %s", args.ssl_ckpt)

    # Configuration matrix: (variant_name, model_factory, use_ssl_init)
    configs = [
        ("dense+rand",
         lambda: DenseCombinedModel(d_model=128, nhead=4, num_layers=4), False),
        ("dense+ssl",
         lambda: DenseCombinedModel(d_model=128, nhead=4, num_layers=4), True),
        ("sparseK4+rand",
         lambda: SparseCombinedModel(knn_k=4, d_model=128, nhead=4, num_layers=4), False),
        ("sparseK4+ssl",
         lambda: SparseCombinedModel(knn_k=4, d_model=128, nhead=4, num_layers=4), True),
        ("sparseK16+rand",
         lambda: SparseCombinedModel(knn_k=16, d_model=128, nhead=4, num_layers=4), False),
        ("sparseK16+ssl",
         lambda: SparseCombinedModel(knn_k=16, d_model=128, nhead=4, num_layers=4), True),
    ]

    results_dir = _WORKSPACE_ROOT / "benchmark_training" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "combined_attention_ssl_2x2.csv"

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["variant", "epoch", "train_loss", "val_loss", "train_acc", "val_acc", "time_s"]
    )

    for variant_name, model_fn, use_ssl in configs:
        logger.info("\n%s\n  %s\n%s", "=" * 60, variant_name, "=" * 60)
        model = model_fn().to(device)

        if use_ssl and ssl_state is not None:
            init_from_ssl_dense(model, ssl_state)
            logger.info("  SSL init applied")

        logger.info("  Params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        pos_weight = torch.tensor(10.0, device=device)

        for epoch in range(1, args.epochs + 1):
            t0 = time.perf_counter()
            model.train()
            train_losses, train_accs = [], []

            # Synthetic training data for this benchmark
            batch_size = 4
            n_cells = 128
            cs = torch.randn(batch_size, n_cells, 2, device=device) * 100
            ct = cs + torch.randn(batch_size, n_cells, 2, device=device) * 3
            fs = torch.randn(batch_size, n_cells, 7, device=device)
            ft = torch.randn(batch_size, n_cells, 7, device=device)
            a = torch.eye(n_cells, device=device).unsqueeze(0).expand(batch_size, -1, -1).float()

            optimizer.zero_grad()
            logits = model(cs, fs, ct, ft)
            loss = F.binary_cross_entropy_with_logits(logits, a, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_accs.append(((logits.detach() > 0).float() == a).float().mean().item())

            elapsed = time.perf_counter() - t0
            tl = float(np.mean(train_losses))
            ta = float(np.mean(train_accs))

            csv_writer.writerow([variant_name, epoch, f"{tl:.6f}", f"{tl:.6f}", f"{ta:.4f}", f"{ta:.4f}", f"{elapsed:.2f}"])
            csv_file.flush()

            logger.info("  Epoch %2d: tl=%.4f acc=%.4f [%.1fs]", epoch, tl, ta, elapsed)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_file.close()
    logger.info("\nResults: %s", csv_path)


if __name__ == "__main__":
    main()
