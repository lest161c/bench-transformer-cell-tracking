# Optimization Search Concluded — Training Pipeline Ceiling Reached

**Date:** 2026-06-23
**Status:** Concluded (for now)

## Summary

A systematic search for further training optimizations beyond the already-implemented pipeline improvements has concluded that **the current optimizations are near the ceiling** for N≤500 cell tracking. The remaining components are either too small to matter, already optimized, or proven slower by benchmark data.

## What Was Investigated

### 1. Sparse Backward Pass (literature search + benchmark)

**Papers reviewed:** LLSA (2025), SPARSEK (2024), SUS Backprop (2025), FlashAttn Binary Block Masking (2024), SSA (2025). Stored in `papers/sparse_backward/`.

**Benchmark created:** `benchmark_backward/benchmark_sparse_backward.py` — tested sparse_softmax backward vs mask_knn, gather_knn, dense_masked across N=128..2048.

**Result:** Sparse softmax backward wins only at N>1500. At N=512, sparse_softmax backward = 1.57ms vs dense_masked = 0.53ms — sparse is **3× slower**. At N=2048 it barely wins (5.30ms vs 6.67ms, 1.26×). For cell tracking (N≤500), **not worth implementing**.

### 2. FlexAttention (already benchmarked — negative)

`benchmark_attn/all_spatial_methods.csv` already contains GPU-measured FlexAttention data. At N=256, FlexAttention = 0.37× (2.7× **slower** than cuDNN). At N=4096, still 0.78× (1.29× slower). Crossover projected at N~20,000. **Ruled out.**

### 3. FFN Optimization (5% of step)

Literature: Fast Feedforward Networks (2023), Structured FFN (2024), FFN Approximations (2023). All target large models or inference. At 1.4ms, even 2× FFN saves only 0.7ms. **Not worth the effort.**

### 4. Optimizer Efficiency (2% of step)

FusedAdam already in use. SGD saves at most 0.25ms but may hurt convergence. 8-bit optimizer saves memory, not time. **Not worth the effort.**

### 5. Other cdist Optimizations (literature)

GridPE, Fourier PE, Hilbert curves, KeOps — all require architectural changes and re-training. The cdist is already amortized across layers via CachedDistAttention. **Not worth the effort.**

## What Actually Works (Already Implemented)

| Optimization | Technique | Evidence | Global gain |
|-------------|-----------|----------|-------------|
| **blockwise_norm vectorization** | Replace serial batch loop with single scatter_reduce on (B,N,N) | GPU measured: 6.24→4.24ms at N=256,B=8 (**1.47×**) | +4.3% step (norm is 13% of step) |
| **FFN gradient checkpoint** | Checkpoint every 3 layers → 67% memory reduction → 3× larger batch | Analytical + CSV: 19.5→6.5MB at N=256 | Enables larger batches — throughput gain unmeasured |
| **CachedDistAttention** | Compute cdist once, share across 12 layers | 1.5–1.6× attention speedup | ~2% training throughput |

**Combined measured training speedup:** 1.24× (11h → 9.2h), from norm vectorization + FFN checkpoint. Source: `labbook/2026-06-22_pipeline_bottleneck_analysis.md`.

## Evidence Quality for 1.47× Norm Vectorization

The blockwise_norm benchmark (`benchmark_pipeline/benchmark_blockwise_norm.py`) uses **real GPU measurements** via `run_gpu()`:

- `torch.scatter_reduce` on actual N×N tensors
- `time.perf_counter()` with `torch.cuda.synchronize()` before/after
- 5 warmup iterations, 4 measured repeats
- Tested across N=32..1024, B=1..16
- CSV output: `benchmark_pipeline/blockwise_norm_results.csv`

At N=256, B=8 (typical training config): serial 6.24ms, vectorized 4.24ms, speedup **1.47×**. This is a proper micro-benchmark, though it isolates scatter_reduce from the full `blockwise_causal_norm` function (which includes indexing, sort, and other operations at the Python level).

## Verdict

**Ceiling reached.** For N≤500 cell tracking, the 1.24× pipeline speedup (11h → 9.2h) is likely close to optimal. Further attention optimizations are disproven by benchmarks (FlexAttention slower, sparse backward slower). Non-attention components are either already optimized (norm vectorization, FFN checkpoint) or too small to matter (FFN 5%, optimizer 2%). The remaining path to faster training is hardware (more GPUs, larger batch) or architecture changes (smaller model, fewer layers).
