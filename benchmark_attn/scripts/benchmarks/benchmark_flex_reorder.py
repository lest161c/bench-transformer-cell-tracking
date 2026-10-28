#!/usr/bin/env python3
"""FlexAttention at production batch: does spatial token ordering restore
linear scaling?

Question (owner, 2026-10-28): the B=64 campaign measured FlexAttention's
steady-state time growing ~4x per N-doubling (5.7 -> 21.4 -> 83.2 -> 327.3 ->
1374.1 ms at N = 2048..32768) despite linear memory — a quadratic-in-N
signature. Hypothesis: with tokens in unsorted (segmentation-label) order,
each 128-token query-block's K=16 scattered nearest neighbours cover nearly
every key-block, so the block mask is DENSE at block level (measured density
1.000 at every N, numpy check, seed 0) and the compiled kernel does O(N^2)
block work. Spatially sorted tokens should make the block mask banded
(measured density 0.148 -> 0.036; active key-blocks per query-block constant
~5-9), restoring ~linear time.

Pre-registered verdict rules (fixed before running):
- Consistency: the identity-order steady times must reproduce the published
  ``flex_setup_h100_b64.csv`` steady_ms within 35% (different process,
  autorange noise) — otherwise the harness diverged and nothing may be
  concluded.
- Linear-scaling: for the sorted order, the median per-doubling growth
  exponent log2(t(2N)/t(N)) <= 1.4 across N >= 4096 (unsorted reference ~2).
- Material effect: sorted/unsorted steady-time ratio <= 0.5 at N >= 16384.

Design: B=64, d=320, nhead=8, coord_dim=3, cutoff=256, mode=none,
dist_mode=v1, fp16, no_grad (inference mode, matching the 32k serving
argument), L=1, K=16-equivalent spatial graph (the module builds its own
block mask from coords). Orders: identity vs lexicographic (x, y) spatial
sort applied identically to query rows and coords (permutation-equivalent
attention with absolute PEs — accuracy untouched, no retraining). The
reorder itself is excluded from the timed region (computed once per
sequence, shared across layers — deployment intent, same convention as the
2026-10 SparseGather reordering experiment).

Usage (cluster, H100)::

    python benchmark_attn/scripts/benchmarks/benchmark_flex_reorder.py \
        --out benchmark_attn/results/flex_reorder_b64.csv

Output: per-(N, order) steady_ms/std/mem, the analytic block-mask density,
the consistency check against the published values, and the verdict.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

BENCH_ROOT = Path(__file__).resolve().parents[2]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from src.attention_modules import FlexAttentionModule  # noqa: E402
from src.bench.data import make_inputs  # noqa: E402
from src.bench.timing import measure  # noqa: E402

BATCH = 64
D_MODEL = 320
N_HEAD = 8
COORD_DIM = 3
CUTOFF_SPATIAL = 256.0
MODE = "none"
DIST_MODE = "v1"
SEED = 42
DTYPE = torch.float16
SEQ_LENS = (4096, 8192, 16384, 32768)
BLOCK = 128  # flex default block size, for the analytic density column
KNN = 16

PUBLISHED_STEADY_MS = {4096: 21.4, 8192: 83.2, 16384: 327.3, 32768: 1374.1}


def block_density(coords_2d: torch.Tensor) -> float:
    """Fraction of active (query-block, key-block) pairs for the KNN graph.

    Exact chunked KNN in torch (no scipy — the cluster benchmark venv does
    not carry it); ties are irrelevant for a density metric.
    """
    import numpy as np

    pts = coords_2d.detach().float()
    n = pts.shape[0]
    nb = n // BLOCK
    kidx = torch.empty(n, KNN, dtype=torch.long, device=pts.device)
    for i in range(0, n, 2048):
        d2 = torch.cdist(pts[i:i + 2048], pts)
        kidx[i:i + 2048] = d2.topk(KNN + 1, largest=False).indices[:, 1:]
    kblock = (kidx // BLOCK).reshape(-1).cpu().numpy()
    qblock = np.repeat(np.arange(n) // BLOCK, KNN)
    active = set(zip(qblock.tolist(), kblock.tolist()))
    return len(active) / (nb * nb)


def run_shape(seq_len: int, sort: bool, device: torch.device) -> dict:
    torch._dynamo.reset()
    torch.manual_seed(SEED)
    query, coords = make_inputs(BATCH, seq_len, D_MODEL, COORD_DIM, device, DTYPE,
                                seed=SEED)
    # coords: (B, N, coord_dim+1); column 0 is a non-spatial scale, columns
    # 1.. are the spatial coordinates consumed by the module.
    if sort:
        spatial = coords[0, :, 1:3]          # (x, y) of the shared graph
        order = torch.argsort(spatial[:, 0] * 4096.0 + spatial[:, 1], stable=True)
        coords = coords[:, order, :]
        query = query[:, order, :]
    module = FlexAttentionModule(
        embed_dim=D_MODEL, n_head=N_HEAD, cutoff_spatial=CUTOFF_SPATIAL,
        mode=MODE, dist_mode=DIST_MODE,
    ).to(device, DTYPE)
    module.prepare(coords)

    def timed():
        with torch.no_grad():
            return module(query, query, query, coords=coords)

    steady_ms, steady_std_ms, steady_mem_mb = (lambda t: (t[0] * 1e3, t[1] * 1e3, t[2]))(
        measure(timed, warmup=5, min_run_time=0.3, repeats=3))
    density = block_density(coords[0][:, 1:3])
    del module, query, coords
    torch.cuda.empty_cache()
    return {"N": seq_len, "order": "sorted" if sort else "identity",
            "steady_ms": steady_ms, "steady_std_ms": steady_std_ms,
            "steady_mem_mb": steady_mem_mb, "block_density": density}


def main() -> int:
    global BATCH, SEQ_LENS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark_attn/results/flex_reorder_b64.csv")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny local config (B=8, N<=2048) to validate mechanics")
    args = ap.parse_args()
    if args.smoke:
        BATCH, SEQ_LENS = 8, (512, 1024, 2048)
    device = torch.device("cuda")

    rows = []
    for seq_len in SEQ_LENS:
        for sort in (False, True):
            r = run_shape(seq_len, sort, device)
            r["published_ms"] = None if (sort or seq_len not in PUBLISHED_STEADY_MS) \
                else PUBLISHED_STEADY_MS[seq_len]
            rows.append(r)
            print(f"N={seq_len:6d} {r['order']:>8}: {r['steady_ms']:9.1f} ms "
                  f"(std {r['steady_std_ms']:.2f})  {r['steady_mem_mb']:7.0f} MB  "
                  f"block density {r['block_density']:.3f}")

    # consistency check (identity order vs published campaign)
    ok = True
    for r in rows:
        if r["order"] != "identity" or args.smoke or r["published_ms"] is None:
            continue
        dev = abs(r["steady_ms"] - r["published_ms"]) / r["published_ms"]
        status = "PASS" if dev <= 0.35 else "FAIL"
        ok &= dev <= 0.35
        print(f"consistency N={r['N']}: {r['steady_ms']:.1f} vs published "
              f"{r['published_ms']:.1f} ({dev*100:.1f}%) {status}")

    # scaling exponents per order
    verdict = {"consistency_pass": ok, "exponents": {}, "ratios": {}}
    for order in ("identity", "sorted"):
        xs = [r["steady_ms"] for r in rows if r["order"] == order]
        exps = [math.log2(xs[i + 1] / xs[i]) for i in range(len(xs) - 1)]
        verdict["exponents"][order] = sum(exps) / len(exps)
        print(f"median doubling exponent ({order}): {verdict['exponents'][order]:.2f}")
    for r in rows:
        if r["order"] == "sorted":
            ident = next(x for x in rows if x["N"] == r["N"] and x["order"] == "identity")
            verdict["ratios"][r["N"]] = r["steady_ms"] / ident["steady_ms"]
            print(f"sorted/identity N={r['N']}: {verdict['ratios'][r['N']]:.3f}")
    big = [verdict["ratios"][n] for n in (16384, 32768) if n in verdict["ratios"]]
    verdict["linear_scaling"] = verdict["exponents"]["sorted"] <= 1.4
    verdict["material_effect"] = bool(big) and max(big) <= 0.5
    verdict["VERDICT"] = ("REORDERING WORKS" if (ok and verdict["linear_scaling"]
                          and verdict["material_effect"]) else "DOES NOT WORK")
    print("\n=== PRE-REGISTERED VERDICT ===")
    print(json.dumps(verdict, indent=2, default=float))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out.parent / "flex_reorder_b64_verdict.json").write_text(
        json.dumps(verdict, indent=2, default=float))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
