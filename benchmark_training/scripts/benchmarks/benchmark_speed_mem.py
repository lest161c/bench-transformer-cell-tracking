#!/usr/bin/env python3
"""Training resource benchmark at the memory ceiling (N=8192).

Measures forward+backward+optimizer-step time and peak incremental GPU
memory of the full Trackastra association model (with pair_norm and
fusion MLP) at large ``N`` where attention scaling dominates.

Run with ``B = 1`` on the local A500 (4 GB); larger ``B`` requires an
H100.  Five attention+SSL variants are tested by default; pass
``--variants`` to select a subset.

Hardware notes
--------------
- Designed for the local A500 (4 GB) GPU at ``B = 1``.
- On an H100 (80 GB) larger ``B`` becomes possible and ``N`` can be
  pushed much higher.

Output CSV columns: ``variant, epoch, train_loss, val_loss, time_s,
mem_mb, status``.

Usage
-----
::

    python scripts/benchmarks/benchmark_speed_mem.py
    python scripts/benchmarks/benchmark_speed_mem.py --Ns 8192 --B 1 --epochs 5
    python scripts/benchmarks/benchmark_speed_mem.py --variants dense sparse_k16
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make src/ importable when running as a script.
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
# Full model with pair_norm + SSL-init support
# ------------------------------------------------------------------ #

class FullModel(nn.Module):
    """Full Trackastra association model with selectable encoder.

    Includes the agent-centric pair-norm layer (which materialises an
    O(N²) tensor at large ``N``) and the fusion MLP used by the
    SSL-pretrained encoder.

    Parameters
    ----------
    knn_k : int | None
        ``None`` → :class:`DenseEncoder`; positive int →
        :class:`SparseEncoder` with ``knn_k = knn_k``.
    d_model : int, default 128
    nhead : int, default 4
    num_layers : int, default 4
    """

    def __init__(
        self,
        knn_k: int | None = None,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.feat_proj = nn.Linear(7, d_model)
        self.coord_proj = nn.Linear(2, d_model)
        self.pair_norm = AgentCentricNormalization()
        self.pair_proj = nn.Linear(5, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        if knn_k is not None and knn_k > 0:
            self.encoder = SparseEncoder(
                d_model, nhead, num_layers, knn_k=knn_k
            )
        else:
            self.encoder = DenseEncoder(d_model, nhead, num_layers)
        self.head = nn.Linear(d_model * 2, 1)

    def forward(
        self,
        cs: torch.Tensor,
        fs: torch.Tensor,
        ct: torch.Tensor,
        ft: torch.Tensor,
    ) -> torch.Tensor:
        f_src = self.feat_proj(fs)
        f_tgt = self.feat_proj(ft)
        c_src = self.coord_proj(cs)
        c_tgt = self.coord_proj(ct)
        pair = self.pair_norm(cs, ct)
        ctx_src = self.pair_proj(pair).mean(dim=2)
        ctx_tgt = self.pair_proj(pair).mean(dim=1)
        src_feat = self.fusion(torch.cat([f_src, c_src, ctx_src], dim=-1))
        tgt_feat = self.fusion(torch.cat([f_tgt, c_tgt, ctx_tgt], dim=-1))
        both = torch.cat([src_feat, tgt_feat], dim=1)
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
# Default variant set
# ------------------------------------------------------------------ #

DEFAULT_VARIANTS = [
    ("dense+rand", None, False),
    ("sparseK4+rand", 4, False),
    ("sparseK4+ssl", 4, True),
    ("sparseK16+rand", 16, False),
    ("sparseK16+ssl", 16, True),
]


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #

def main() -> None:
    """Run the training-resource benchmark at the memory ceiling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=8192,
                        help="Number of cells per frame (default: 8192).")
    parser.add_argument("--B", type=int, default=1,
                        help="Batch size (default: 1 — only viable at 4 GB).")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help="Subset of variants to run (default: all five).",
    )
    parser.add_argument(
        "--ssl_ckpt", type=Path,
        default=_WORKSPACE_ROOT / "benchmark_ssl" / "runs" / "ssl_K=16" / "best_model.pt",
        help="Path to SSL checkpoint .pt file.",
    )
    args = parser.parse_args()

    device = get_device()
    log_device_info(device)

    if not args.ssl_ckpt.exists():
        logger.warning("SSL checkpoint not found: %s", args.ssl_ckpt)
        ssl_state = None
    else:
        ssl_state = torch.load(args.ssl_ckpt, map_location="cpu", weights_only=False)[
            "model_state_dict"
        ]

    results_dir = _WORKSPACE_ROOT / "benchmark_training" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "combined_n8192.csv"

    import csv
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["variant", "epoch", "train_loss", "val_loss", "time_s", "mem_mb", "status"]
    )

    cs = torch.randn(args.B, args.N, 2, device=device) * 100
    ct = cs + torch.randn(args.B, args.N, 2, device=device) * 3
    fs = torch.randn(args.B, args.N, 7, device=device)
    ft = torch.randn(args.B, args.N, 7, device=device)
    a = torch.eye(args.N, device=device).unsqueeze(0).float()
    pos_weight = torch.tensor(10.0, device=device)

    selected_variants = [
        (name, knn_k, use_ssl)
        for (name, knn_k, use_ssl) in DEFAULT_VARIANTS
        if args.variants is None or any(v in name for v in args.variants)
    ]

    for variant_name, knn_k, use_ssl in selected_variants:
        try:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated() if device.type == "cuda" else 0

            model = FullModel(knn_k=knn_k).to(device)
            if use_ssl and ssl_state is not None:
                if knn_k is not None and knn_k > 0:
                    init_from_ssl_sparse(model, ssl_state)
                else:
                    init_from_ssl_dense(model, ssl_state)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

            # Warmup + memory peak measurement
            logits = model(cs, fs, ct, ft)
            loss = F.binary_cross_entropy_with_logits(logits, a, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            peak_mb = (
                (torch.cuda.max_memory_allocated() - mem_before) / (1024 ** 2)
                if device.type == "cuda" else 0.0
            )

            for epoch in range(1, args.epochs + 1):
                t0 = time.perf_counter()
                for _ in range(3):
                    logits = model(cs, fs, ct, ft)
                    loss = F.binary_cross_entropy_with_logits(
                        logits, a, pos_weight=pos_weight
                    )
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed_s = (time.perf_counter() - t0) / 3
                tl = loss.item()

                with torch.no_grad():
                    logits = model(cs, fs, ct, ft)
                    vl = F.binary_cross_entropy_with_logits(
                        logits, a, pos_weight=pos_weight
                    ).item()

                logger.info(
                    "  %s Ep%d: tl=%.4f vl=%.4f %.0fms/step %.0fMB",
                    variant_name, epoch, tl, vl, elapsed_s * 1000, peak_mb,
                )
                csv_writer.writerow([
                    variant_name, epoch,
                    f"{tl:.6f}", f"{vl:.6f}",
                    f"{elapsed_s:.6f}", f"{peak_mb:.1f}", "ok",
                ])
                csv_file.flush()

        except RuntimeError as exc:
            logger.info("  %s: OOM/ERR - %s", variant_name, exc)
            csv_writer.writerow([variant_name, 0, "", "", "", "", "oom"])
            csv_file.flush()
        finally:
            try:
                del model
            except NameError:
                pass
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    csv_file.close()
    logger.info("Done: %s", csv_path)


if __name__ == "__main__":
    main()
