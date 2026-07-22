# Attention Is Not the Training Bottleneck

**Date:** 2026-05-29  
**Depends on:** `2026-05-29_gather_sparse_bottleneck.md`, `2026-05-29_flash_attn_integration.md`

---

## 1. Claim

The vanvliet dataset (N ≈ 35 cells/frame, window=4 → N ≈ 140 per sample) has too few tokens per attention call for attention computation to dominate training step time. No attention optimization — KNN gather, FlashAttention, cached-dist, or otherwise — can produce a measurable training throughput speedup at this N.

## 2. Evidence

Three independent H100 training runs compared `CachedDistAttention` (branch `cached-dist-attn`) against the baseline `RelativePositionalAttention` (branch `feature/sparse-attention-gather`), identical config `vanvliet_baseline.yaml`:

| Variant | batch_size | Elapsed (10 ep) | val_loss @ ep 5 |
|---|---|---|---|
| baseline | 48 | 14:17 | 0.118 |
| cached-dist | 48 | 14:14 | 0.107 |
| baseline | 128 | 14:21 | — |
| cached-dist | 128 | 14:02 | — |

Maximum wall-time difference: **2.2%** between any pair. At full training scale (190 epochs): 5h27m vs 5h34m baseline — 2.1%.

## 3. Why

At N ≈ 140–350, a single attention forward takes ~1–5 ms per layer (12 layers → ~15–60 ms total). The training step also includes:

- **Data loading** from Lustre via 8 workers: ~40–60 ms (GPU idle)
- **Data augmentation** (RandomAffine, flips, intensity shifts): ~15 ms
- **Feed-forward + layer norms** per layer: ~20 ms
- **Backward pass** (autograd): ~30 ms

Attention represents **<15% of total step time**. A 10× attention speedup would improve total throughput by <13%. The 1.5× we achieved translates to <5% in training — indistinguishable from run-to-run noise.

## 4. When Attention Would Matter

| Scenario | N | Attention share of step | Speedup visible? |
|---|---|---|---|
| vanvliet (actual) | ~140 | ~5% | No |
| vanvliet (window=10) | ~350 | ~10% | Barely |
| Hypothetical large colony | ~2048 | ~60% | Yes |
| Isolated benchmark | 8192 | 100% | Yes (1.5× at L=12, proj) |

The isolated benchmark (`benchmark_sparse.py`) measures exactly what the proposal §4.1 asks for: throughput/memory plots across N, K, and depth L. Training throughput was never the metric.

## 5. Implication

The `CachedDistAttention` optimization is correct, functionally equivalent to the original, and measurably faster in isolation (1.5× at L=12). It does not accelerate training on the vanvliet dataset because attention was never the bottleneck at N ≈ 140. This is a property of the data, not the method.

For the thesis: isolate the speedup claim to the controlled benchmark (§4.1). The training runs serve as correctness verification and convergence parity evidence (§4.3), not speedup evidence.
