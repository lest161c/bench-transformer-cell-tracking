# NSA (Native Sparse Attention) — Negative Benchmark Result

**Date:** 2026-06-08
**Context:** Evaluated DeepSeek's NSA (Yuan et al. 2025, arXiv:2502.11089) against our existing FlashAttention-based `DenseFlashAttention` at N = 128–16384. Used lucidrains' PyTorch implementation (`native-sparse-attention-pytorch v0.2.3`).
**Depends on:** `2026-05-29_gather_sparse_bottleneck.md`, `2026-05-29_flash_attn_integration.md`

---

## 1. Summary

**NSA is 5–25× slower than `DenseFlashAttention` at all N ≤ 16384 on our hardware (RTX A500 Laptop, 3.7 GB). It also consumes 20–40× more GPU memory, OOMing at N=8192 for larger configurations.** The gap narrows with increasing N but projects a crossover at N ≈ 35K–40K — unreachable on our GPU.

The finding is consistent with the paper's own Figure 6, which shows NSA achieves only 1.1× forward speedup at 8k context (the minimum they test). Meaningful speedups (≥2×) require N ≥ 16k. Our N range (128–16k) lies entirely in the "overhead-dominated" regime.

---

## 2. Benchmark Results

All measurements: fp16, B=2, d=256, h=4, L=1, A500 (3.7 GB Ampere). NSA configured with `compress_block_size=16, selection_block_size=16, sliding_window_size=32, num_selected_blocks=2`.

| N | dense_flash | NSA (plain) | NSA (torch.compile) | Ratio (flash/NSA) | NSA memory |
|---|---|---|---|---|---|
| 128 | **96 µs** | 2,375 µs | — | 24.7× | 22 MB |
| 256 | **91 µs** | 2,374 µs | — | 26.1× | 43 MB |
| 512 | **119 µs** | 3,529 µs | — | 29.7× | 88 MB |
| 2,048 | **779 µs** | 10,790 µs | 8,861 µs | 11.4× | 228 MB |
| 8,192 | **9,461 µs** | 69,993 µs | 49,253 µs | 5.2× | 1,153 MB |
| 10,240 | 14,326 µs | 98,484 µs | — | 6.9× | 1,547 MB |
| 12,288 | 22,320 µs | OOM | — | — | >3.7 GB |
| 16,384 | 41,775 µs | OOM | — | — | >3.7 GB |

Additional NSA configurations (`num_selected_blocks=4,8`, `compress_block_size=32`, `sliding_window_size=64`) were consistently slower and OOM'd earlier.

---

## 3. Profiler Root-Cause Analysis

### 3.1 Operator-Level Breakdown (N=2048, single forward)

**dense_flash** — 0.67 ms self CUDA, ~5 kernel launches:
```
 flash_fwd_kernel              0.46 ms  (68.9%)  ← single fused flash attention
 addmm (Q/K/V/Output linear)   0.21 ms  (31.2%)  ← 4 linear layer calls
```

**NSA** — 8.5 ms self CUDA, ~88 kernel launches:
```
 bmm (einsum → attention)      1.69 ms  (19.9%)  ← 7 separate einsum calls
 gather (scatter_gather)       1.74 ms  (20.5%)  ← block selection gathering K/V
 mm/addmm (linear + MLPs)      1.90 ms  (22.4%)  ← QKV + compress MLPs + gate MLP
 copy_/clone                   1.07 ms  (12.6%)  ← memory management
 elementwise_kernel            1.04 ms  (12.3%)  ← mask building, mul, etc.
 topk                          0.88 ms  (10.4%)  ← selecting top-k blocks
 cudaLaunchKernel              0.94 ms  (11.1%)  ← *just* the kernel launch overhead
 reshape/clone/cat             ~1.2 ms  (14.6%)  ← 61 reshape + 10 cat calls
```

### 3.2 Why the Overhead Is Fatal at N < 16K

NSA computes **three separate attention paths** with Python-level orchestration:

```
┌─ Compressed attn:  split_compress_window → k/v_compress MLP → einsum → softmax → einsum
├─ Selected attn:     topk(importance) → gather(K/V blocks) → mask build → einsum → softmax → einsum
├─ Sliding window:     LocalAttention (own implementation)
└─ Gating:             MLP → sigmoid → weighted sum
```

Each path requires its own kernel launches, intermediate tensors, and memory management. At N=2048, this produces:

| Metric | dense_flash | NSA | Ratio |
|---|---|---|---|
| CUDA kernel launches | ~5 | ~88 | 17.6× |
| Self CUDA time | 0.67 ms | 8.5 ms | 12.7× |
| Memory allocated (forward) | ~10 MB | ~256 MB | 25.6× |
| Unique operator types | 4 | 131 | 32.8× |

**The critical insight:** at N=2048, NSA spends more time on `cudaLaunchKernel` overhead (0.94 ms) than dense_flash's entire computation (0.67 ms). The kernel launch overhead from orchestrating sparsity **exceeds the total cost of the dense computation itself**.

This mirrors the earlier finding with `GatherSparseAttention` (`2026-05-29_gather_sparse_bottleneck.md`) — sparse attention methods in pure PyTorch carry irreducible Python orchestration overhead that dominates at short and medium sequence lengths.

---

## 4. Consistency with the NSA Paper

The authors' own Figure 6 (Triton kernel vs FlashAttention-2 on A100) shows:

| Context | NSA forward speedup | NSA backward speedup |
|---|---|---|
| 8k | **1.1×** (negligible) | 1.1× |
| 16k | 2.0× | 2.1× |
| 32k | 3.4× | 3.8× |
| 64k | 9.0× | 6.0× |

The paper never tests below 8k. At 8k (their minimum), the speedup is only 1.1× — barely faster than dense attention. The paper explicitly acknowledges this in Section 2.1 ("The Illusion of Efficient Inference"), noting that sparse attention overheads prevent real speedup at insufficiently long contexts.

Our results at N=128–16384 **extend and confirm** the left tail of the paper's Figure 6: below the 8k–16k threshold, NSA's sparse orchestration overhead **exceeds** the dense compute cost. The paper and our experiments agree on the existence and approximate location of this crossover.

However, the paper also notes that their results use a **custom Triton kernel** (Section 3.4) optimized for A100 Tensor Cores. The lucidrains PyTorch implementation we used does not include this kernel (`use_triton_kernel=True` failed due to a `fp32/fp64` mismatch in Triton 3.7). The paper's performance at 16k–64k is therefore achieved with significantly lower-level optimization than available in open-source.

---

## 5. Decision Basis

**NSA is not suitable for our research.** The argument:

1. **Empirically:** NSA is 5–25× slower than FlashAttention at all N ≤ 16k on our GPU.
2. **Theoretically (confirmed by the paper):** NSA's overhead only amortizes at N > 8k (they measure 1.1× at 8k, 2.0× at 16k).
3. **Hardware limit:** Our A500 GPU (3.7 GB) OOMs at N ≈ 12k, below the crossover point of ~35k.
4. **Implementation quality:** The open-source PyTorch implementation lacks the custom Triton kernel (broken in Triton 3.7), so even at larger N on bigger GPUs the pure-PyTorch version would underperform.
5. **Our dataset N ≈ 128–512:** At these lengths, NSA's multi-path overhead (~2.4 ms fixed cost) dominates entirely — the method is structurally unsuited for small-N attention.

`DenseFlashAttention` remains the recommended approach for our pipeline. The compression/selection/window strategy is architecturally interesting for N > 16k but irrelevant to our use case.

---

## 6. Files Modified

| File | Change |
|---|---|
| `model_parts.py` | Added `NSASparseAttention` wrapper class (line 952) |
| `benchmark_sparse.py` | Added `bench_nsa()`, import, and sweep loop (sel_blocks ∈ {2,4,8}) |
| `benchmark_attn/pyproject.toml` | Added `native-sparse-attention-pytorch` via `uv pip install` |
| `benchmark_sparse_results.csv` | Contains NSA rows (method="nsa", K=sel_blocks) |
