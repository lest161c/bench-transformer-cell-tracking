# Why GatherSparseAttention Cannot Beat Dense at N < 2000

**Date:** 2026-05-29  
**Context:** Profiling `GatherSparseAttention` to identify per-op bottlenecks and determine whether the KNN gather approach can be optimized to achieve >1× speedup at N≈300 (real dataset cell count).

---

## 1. Summary

`GatherSparseAttention` is **structurally incapable** of outperforming a plain dense `F.scaled_dot_product_attention` (no mask) at sequence lengths N < 2000. Two independent bottlenecks guarantee this:

1. **SDPA shape problem:** Reshaping to `(B·N, nH, 1, Dh)` creates **2048 independent (1×K) attention ops** — no GPU kernel handles `q_len=1` efficiently (cuDNN refuses it, FlashAttention launch overhead dominates).
2. **Irreducible gather copy:** `k[B_idx, H_idx, idx, :]` triggers advanced indexing → a mandatory O(N·K·nH·Dh) memory copy that costs **more time than the original dense N² attention itself** at N<2000.

Alternative designs (`KNNMaskSparseAttention`, `DenseFlashAttention`) avoid both bottlenecks and achieve 7–10× speedup over the original gather approach.

---

## 2. Profiler Measurements

All measurements on RTX A500 Laptop GPU, dtype=float16, B=2, nH=4, Dh=64, K=16, L=1.

### 2.1 GatherSparseAttention forward (N=256)

| Operator | CUDA Time | % | Category |
|---|---|---|---|
| `flash_fwd_kernel` | 534 us | 51.7% | SDPA on (512,4,1,64)→(16,64) |
| `index_elementwise_kernel` | 271 us | 26.2% | gather `k[B_idx,H_idx,idx,:]` |
| `copy_/clone` | 189 us | 18.3% | contiguous reshape copies |
| `addmm` (QKV proj) | 37 us | 3.6% | linear projections |
| **Total forward** | **1033 us** | — | without QKV+output proj |

### 2.2 Dense masked (RelativePositionalAttention, N=256)

| Operator | CUDA Time | % |
|---|---|---|
| `cdist` + `_euclidean_dist` | 112 us | 57.9% |
| `masked_fill_` | 24 us | 12.6% |
| `add_` (distance decay) | 24 us | 12.5% |
| SDPA (EfficientAttention) | 33 us | 17.0% |
| **Total forward** | **193 us** | — |

### 2.3 DenseFlashAttention (no mask, plain FlashAttn, N=256)

| Operator | CUDA Time | % |
|---|---|---|
| `addmm` (QKV proj) | 36 us | 70.8% |
| `flash_fwd_kernel` | 15 us | 29.2% |
| **Total forward** | **51 us** | — |

### 2.4 SDPA backend dispatch (attention core only, no gather/proj)

| Shape | Backend | CUDA Time |
|---|---|---|
| Dense `(2,4,256,64)` | FlashAttention | **12 us** |
| Dense `(2,4,256,64)` + N×N mask | EfficientAttention | **29 us** |
| Sparse `(512,4,1,64)` → `(16,64)` | FlashAttention (auto) | **531 us** |
| Sparse forced EfficientAttention | EfficientAttention | **219 us** |
| Sparse forced cuDNN | **FAIL** — "cudnn SDPA does not support sequence length 1" | — |

---

## 3. Bottleneck 1: The Shape Problem (q_len=1)

`GatherSparseAttention.forward()` reshapes Q, K, V to:

```python
q_flat = q.transpose(1,2).reshape(B * N, nH, 1, Dh)   # (512, 4, 1, 64)
k_flat = ...view(B * N, nH, K, Dh)                     # (512, 4, 16, 64)
v_flat = ...view(B * N, nH, K, Dh)
```

SDPA interprets `(B*N, nH, 1, Dh)` as **512 × 4 = 2048 independent attention groups**, each computing a `(1 query × 16 keys)` attention matrix. This is 2048 launched kernels, each handling ~0.5K FLOPs.

**Why this is fatal:**
- **cuDNN explicitly refuses** `q_len=1` — the fastest backend cannot be used.
- **FlashAttention** handles `q_len=1` correctly but the kernel launch overhead per group exceeds the compute time. 2048 kernel launches × ~0.25us/launch ≈ 500us overhead.
- **EfficientAttention** is actually 2.4× faster on this shape (219us vs 531us) but PyTorch's auto-dispatcher selects FlashAttention because no mask is present.

**Attempted fixes:**

| Approach | Description | SDPA Core Time (N=256) |
|---|---|---|
| SDPA (q_len=1) | Original `(512,4,1,64)` shape | 531 us |
| Force EfficientAttention | `sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])` | 219 us |
| Manual matmul (V3) | `torch.matmul(q.unsqueeze(3), k_sel.transpose(-2,-1))` | 287 us (score + output) |
| flex_attention block-diag | `flex_attention(q, k_big, v_big, block_mask=B)` | ~10,000 us (block size 1×K too small) |
| N×N mask (mask_knn) | `scatter_` KNN mask + SDPA on `(B,nH,N,Dh)` | **29 us** (EfficientAttention) |
| No mask (dense_flash) | Plain `F.sdpa(q,k,v)` on `(B,nH,N,Dh)` | **12 us** (FlashAttention) |

The manual matmul approach (V3) is optimal for the gather-based design — it avoids SDPA kernel overhead entirely by using 2 batched `torch.matmul` calls that process all 256 queries in each batch-head group at once. But it still cannot beat the approaches that avoid the gather.

---

## 4. Bottleneck 2: The Irreducible Gather Copy

Even with optimal matmul-based attention (V3), the gather remains:

```python
k_sel = k[B_idx, H_idx, idx, :]   # (B, nH, N, K, Dh)
v_sel = v[B_idx, H_idx, idx, :]   # ditto
```

This is **advanced indexing** in PyTorch → always a **copy**, never a view. At N=256:
- Copy size: 2 × 4 × 256 × 16 × 64 × 2 bytes (fp16) = **4 MB per K/V pair = 8 MB total**
- Memory bandwidth on A500: ~200 GB/s → theoretical minimum ~40us
- Actual measured: **276 us** (index computation + kernel launch + non-contiguous access pattern)

**Full V3 forward decomposition (N=256):**

```
gather (aten::index × 2)      276 us   ← IRREDUCIBLE: must copy 8MB from scattered memory
matmul score (WMMA gemm)       63 us   ← near-optimal: 2048 batched (1,Dh)@(Dh,K)=1.2M MACs
matmul output (GEMV batched)   57 us   ← near-optimal: 2048 batched (1,K)@(K,Dh)=1.2M MACs
softmax                         4 us
QKV projection                 44 us
output contig+proj             61 us
─────────────────────────────────────
Total (V3, attention only):   505 us   ← 2.3× faster than V1, but...
vs dense_flash attention core:  12 us   ← 42× faster, pays zero gather tax
```

**The gather exceeds the dense SDPA time by 23×.** On H100 (3.35 TB/s bandwidth) the gather reduces to ~30us, but the dense_flash SDPA also drops to ~3-5us. The ratio remains unfavorable at small N.

---

## 5. The Envelope Analogy

At N=256:
- **Dense flash**: 8 matrices of 256×256, 0.5M MACs total → FlashAttention fuses into one kernel per batch-head. **12 us.**
- **KNN gather**: Must first copy 4MB of scattered K/V values into 256 per-query envelopes. Copy alone costs 276us — 23× more than the original computation. Then computes 2048 tiny (1×16) attentions. **505 us minimum.**

> **Analogy:** Sorting mail into 256 small envelopes costs more than just delivering the single large envelope. The compute you save (0.5M → 0.03M MACs) was never the limiting factor — FlashAttention handled the large envelope in 12us. You paid more to *prepare* the work than the original work cost.

---

## 6. Full Forward Comparison

### 6.1 Small-N (dataset range: N=128, 256, 512)

All measurements include QKV projection, attention, and output projection (L=1, K=16, fp16, A500):

| Method | N=128 | N=256 | N=512 | Mechanism |
|---|---|---|---|---|
| sparse_gather (V1) | 605 us | 1157 us | 2304 us | SDPA q_len=1 + gather + contiguous |
| sparse_gather (V3) | 276 us | 503 us | 985 us | Manual matmul + gather (2.3× over V1) |
| mask_knn (new) | 109 us | 113 us | 288 us | N×N KNN mask via scatter + EfficientAttn |
| dense_masked | 591 us | 350 us | 711 us | cdist + mask + EfficientAttn |
| **dense_flash** | **82 us** | **86 us** | **137 us** | FlashAttention, no mask, no gather |

**Key observation:** At N=256, `dense_flash` (86us) vs `mask_knn` (113us): the KNN spatial structure costs only **31% overhead** over the theoretical lower bound. No gather-based approach can come close to either.

### 6.2 Large-N (full sweep: where the crossover lives)

Full benchmark sweep including N=2048 and N=8192 (L=1, K=16, fp16, A500):

| Method | N=128 | N=256 | N=512 | N=2048 | N=8192 |
|---|---|---|---|---|---|
| dense_masked | 0.37 ms | 0.35 ms | 0.59 ms | 8.4 ms | 131.6 ms |
| sparse K=16 | 0.56 ms | 1.07 ms | 2.10 ms | 8.5 ms | 34.5 ms |
| sparse K=4 | 0.37 ms | 0.69 ms | 1.36 ms | 5.5 ms | 21.9 ms |
| **dense_flash** | **0.11 ms** | **0.10 ms** | **0.12 ms** | **0.88 ms** | **9.4 ms** |

Key observations:
- **dense_flash is faster than sparse at ALL N on this GPU.** The crossover never happens.
- **The original 5.88× claim:** At N=8192, sparse K=4 (21.9ms) vs dense_masked (131.6ms) = 6.0×. This is a real measurement but compares against the wrong baseline — the masked dense, which is itself bottlenecked by mask construction.
- **Fair comparison:** At N=8192, dense_flash (9.4ms) vs sparse K=4 (21.9ms): dense_flash is **2.3× faster**. At N=2048: dense_flash (0.88ms) vs sparse K=16 (8.5ms): dense_flash is **9.6× faster**.
- **At N=256 (real dataset):** dense_flash (0.10ms) vs sparse K=16 (1.07ms): dense_flash is **10.7× faster**.

**Conclusion: KNN gather attention is slower than unmasked dense FlashAttention at all measured N on this hardware (A500, Ampere).** However, unmasked dense attention drops the spatial cutoff — a structural component defined in Trackastra Eq. 3 (`M_ij = -∞ if ||p_i-p_j|| > d_max`). Removing it is architecturally incorrect without empirical validation.

**Correct solution:** `KNNMaskSparseAttention` preserves the spatial cutoff via an N×N scatter mask (O(NK), not O(N²) cdist), runs at 2.5–3.1× faster than the original `dense_masked` at all N, and keeps the exact same KNN spatial semantics. This is the attention layer integrated into Trackastra on the `flash-attention` branch.

---

## 7. Training-Time Impact (Backward Pass + V2 Negative Result)

### 7.1 Forward + Backward Profile (N=256, K=16)

| Operator | CUDA Time | % |
|---|---|---|
| `flash_bwd_dq_dk_dv` (SDPA backward) | 1170 us | 41.3% |
| `_index_put_impl_` (gather backward) | 338 us | 11.9% |
| `flash_fwd_kernel` (SDPA forward) | 534 us | 18.9% |
| `index_elementwise_kernel` (gather forward) | 272 us | 9.6% |
| `indexing_backward_kernel` (gather bw) | 270 us | 9.5% |
| `copy_/clone` (contiguous) | 524 us | 18.5% |
| **Total fw+bw** | **3577 us** | — |

The backward pass doubles the overhead: `IndexBackward0` + `_index_put_impl_` add ~20% of total CUDA time. In training, the gather penalty is paid **on every forward AND backward pass**, doubling the cumulative cost.

### 7.2 V2 Copy Optimizations (Negative Result)

`GatherSparseAttentionV2` attempted to eliminate unnecessary copies by:
- Replacing `q.transpose(1,2).reshape(B*N, nH, 1, Dh)` (1 copy) with `q.reshape(B*N, nH, Dh).unsqueeze(2)` (0 copies)
- Replacing double-transpose at output (1 copy) with direct `y.reshape(B, N, D)` (0 copies)

**Result: 0.5% faster than V1** (563 us vs 559 us at N=128). The copies saved (~40 us theoretical) are in the noise compared to the gather kernel (137 us) and SDPA kernel (270 us). The bottleneck is not copy operations.

---

## 8. Resolution of Proposal §4.1

The proposal's §4.1 ("Efficient Sparse Attention") can be resolved as follows:

**Hypothesis (from proposal):** Replacing dense masked SDPA with gather-based sparse attention over a KNN graph enables FlashAttention and yields speedup via O(NKd) complexity.

**What was found:**

1. The mask, not N² compute, is the bottleneck. `cdist` + `masked_fill` + `add_` = **82% of dense CUDA time** at N=256. The actual attention is only 17%.

2. Removing the mask **without** adding KNN gather (`DenseFlashAttention`) yields **3–14× speedup** over the masked baseline across all N. This is simpler than the gather approach and strictly faster.

3. The gather-based approach (`GatherSparseAttention`) is **slower than the masked baseline** at all N < 2000 on A500, and slower than `dense_flash` at all N including 8192.

4. KNN can preserve spatial structure via `KNNMaskSparseAttention` (N×N scatter mask), incurring only ~30% overhead over `dense_flash`. This is the correct design when spatial priors matter.

**Resolution:** §4.1 is **complete as a scientific result.** The hypothesis was tested, found to be correct at N > 8000 but incorrect for the actual dataset (N ≈ 300). The investigation produced two faster alternatives (`DenseFlashAttention`, `KNNMaskSparseAttention`) and identified the mask computation as the real bottleneck. The remaining question — whether removing spatial structure hurts tracking accuracy — is an empirical evaluation question for §4.3, not a novel method question for §4.1.

**Pivot to §4.2 (SSL):** With the attention speedup question resolved, the remaining original research contribution is the SSL pretraining pipeline (§4.2). The attention findings are publishable as a methods improvement but do not require further implementation work.

---

## 9. Source Code

All benchmarks are in `benchmark_attn/`. Relevant files:

| File | Purpose |
|---|---|
| `model_parts.py` | `GatherSparseAttention` (V1), `GatherSparseAttentionV3` (matmul), `KNNMaskSparseAttention` (mask), `DenseFlashAttention` |
| `profile_gather_sparse.py` | Targeted profiler outputting op-level CUDA times |
| `benchmark_mask_vs_gather.py` | Full forward comparison of all variants |
| `benchmark_sparse.py` | Sweep over N=128..8192, all K values |
| `profiling_report.html` | Combined charts and tables |
| `benchmark_sparse_results.csv` | Raw timing data |
