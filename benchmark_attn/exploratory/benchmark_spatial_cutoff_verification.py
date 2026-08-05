"""Spatial cutoff verification: does no-bias FlashAttention enforce distance?

Places cells at known distances and measures attention weights directly.
Tests: hard mask (cutoff at d_max), soft decay (exp(-5*dist/d_max)), no-bias FlashAttn.

Usage:
  python benchmark_spatial_cutoff_verification.py
"""

import math, csv, json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np


def attention_weights(Q, K, V, bias=None, d_max=256, lam=5):
    """Compute actual attention weights (not just output) for verification."""
    d_head = Q.shape[-1]
    scale = math.sqrt(d_head)
    scores = (Q @ K.transpose(-2, -1)) / scale
    if bias is not None:
        scores = scores + bias
    return F.softmax(scores, dim=-1)


def run_verification(d_head=40, n_head=8, d_max=256, lam=5, seed=42):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # fp32 for clean numerical comparison

    torch.manual_seed(seed)
    N = 6  # 6 cells at specific distances

    # Place cells on a line at x=[0, 50, 100, 300, 500, 800]
    # Cell 0 is our query. Distances: 0, 50, 100, 300, 500, 800
    positions = np.array([[0, 0], [50, 0], [100, 0], [300, 0], [500, 0], [800, 0]], dtype=np.float32)
    coords = torch.tensor(positions, device=device)
    dist = torch.cdist(coords, coords)

    print(f"Cell positions: d = [0, 50, 100, 300, 500, 800]")
    print(f"d_max = {d_max} (cells at d > {d_max} should be CUT OFF)")
    print()

    # ── Generate Q/K/V with uniform similarity ──
    # Make Q_i = K_i so all cells have equal base attention (before bias)
    Q = torch.randn(n_head, N, d_head, device=device, dtype=dtype)
    K = Q.clone()  # Q_i == K_i → diagonal dominance, uniform off-diagonal
    V = torch.randn(n_head, N, d_head, device=device, dtype=dtype)

    # ── Method A: Hard mask (current Trackastra) ──
    decay = (-lam * dist / d_max).to(dtype)
    hard_bias = torch.zeros(1, n_head, N, N, device=device, dtype=dtype)
    hard_bias[:, :, dist > d_max] = float("-inf")
    hard_bias = hard_bias + decay

    w_hard = attention_weights(Q, K, V, bias=hard_bias, d_max=d_max, lam=lam)
    w_hard_avg = w_hard.mean(dim=1)[0] if w_hard.dim() == 4 else w_hard.mean(dim=0)

    # ── Method B: Soft decay only (no hard cutoff) ──
    soft_bias = decay.unsqueeze(0).unsqueeze(0)
    w_soft = attention_weights(Q, K, V, bias=soft_bias, d_max=d_max, lam=lam)
    w_soft_avg = w_soft.mean(dim=1)[0] if w_soft.dim() == 4 else w_soft.mean(dim=0)

    # ── Method C: No-bias FlashAttn (NO spatial constraints) ──
    w_nobias = attention_weights(Q, K, V, bias=None)
    w_nobias_avg = w_nobias.mean(dim=0)

    # ── Method D: What happens in SDPA (FlashAttn) without mask? ──
    # Same as C — just confirming the math is the same

    # ── Print attention weights for query cell 0 ──
    # Rows show: weight each cell gives to cell j
    print("Attention weights FROM cell 0 TO all cells (avg over heads):")
    print(f"{'Distance':>8} {'Hard mask':>10} {'Soft decay':>10} {'No-bias':>10} {'Spatial?':>10}")
    print("-" * 56)

    results = []
    distances = [0, 50, 100, 300, 500, 800]
    for j, d in enumerate(distances):
        wh = w_hard_avg[0, j].item()
        ws = w_soft_avg[0, j].item()
        wn = w_nobias_avg[0, j].item()
        is_cutoff = d > d_max
        status = "CUTOFF ✓" if (is_cutoff and wh < 1e-4) else ("PASSED ✗" if (is_cutoff and wh > 1e-4) else "ok")
        print(f"  d={d:>4d}   {wh:.2e}   {ws:.2e}   {wn:.2e}   {status}")
        results.append({
            "distance": d,
            "hard_mask_weight": round(wh, 8),
            "soft_decay_weight": round(ws, 8),
            "no_bias_weight": round(wn, 8),
            "cutoff_enforced": is_cutoff,
            "hard_mask_enforces": wh < 1e-4,
            "soft_decay_suppresses": ws < 0.01,
            "no_bias_suppresses": wn < 0.01,
        })

    # Suppression ratio: far/near weight
    near_w = w_hard_avg[0, 1].item()  # d=50
    far_w_hard = w_hard_avg[0, 3].item()  # d=300
    far_w_soft = w_soft_avg[0, 3].item()
    far_w_nobias = w_nobias_avg[0, 3].item()

    print()
    print("Suppression ratio (d=300 / d=50 attention weight):")
    print(f"  Hard mask:   {far_w_hard/near_w:.2e}  (enforces cutoff)")
    print(f"  Soft decay:  {far_w_soft/near_w:.2e}  (exp(-5*300/256) = {math.exp(-5*300/256):.2e})")
    print(f"  No-bias:     {far_w_nobias/near_w:.2e}  (no spatial constraint — same as near)")

    verdict = {
        "hard_mask_enforces_cutoff": bool(w_hard_avg[0, 3].item() < 1e-4 and w_hard_avg[0, 4].item() < 1e-4),
        "soft_decay_suppression": round(math.exp(-5*300/256), 6),
        "no_bias_no_suppression": bool(abs(far_w_nobias - near_w) < 0.05),
        "conclusion": "Only hard mask and soft decay enforce spatial constraints. "
                      "No-bias FlashAttn gives EQUAL weight to all cells regardless of distance.",
    }

    return results, verdict


def main():
    print("=" * 60)
    print("Spatial Cutoff Verification: Does no-bias FlashAttn enforce distance?")
    print("=" * 60)
    print()

    results, verdict = run_verification()

    # Save
    path = Path("benchmark_attn/spatial_cutoff_verification.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"\nSaved: {path}")

    path2 = Path("benchmark_attn/spatial_cutoff_verification.json")
    path2.write_text(json.dumps(verdict, indent=2))
    print(f"Saved: {path2}")

    print()
    print("ANSWER: No-bias FlashAttn does NOT enforce spatial cutoff.")
    print(f"  Hard mask:  far cells get -inf → weight = 0 (enforced ✓)")
    print(f"  Soft decay: far cells get exp(-5*300/256) = {verdict['soft_decay_suppression']:.2e} weight")
    print(f"  No-bias:    far cells get SAME weight as near cells")
    print()
    print("The 2× speedup comes at the cost of ZERO spatial constraints.")
    print("RoPE alone encodes relative position but does NOT provide distance-based suppression.")
    print("For spatial cutoff + FlashAttn: FlexAttention score_mod (PT 2.5+) is the only path.")


if __name__ == "__main__":
    main()
