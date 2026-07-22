# The Irreducible Mask Indirection Principle

**Date:** 2026-06-08
**Context:** Crystallizing the fundamental tradeoff behind all sparse attention designs at sequence lengths N < 2000.
**Depends on:** `2026-05-29_gather_sparse_bottleneck.md`, `2026-06-08_nsa_benchmark_negative.md`, `2026-06-08_attention_optimization_landscape.md`

---

## 1. Statement

Let \(\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times H \times N \times d_h}\) be batched multi-head query, key, and value tensors. A sparse attention pattern defines a per-query neighbor set:

\[\mathcal{N}(b, h, i) \subseteq \{0, \dots, N-1\}, \quad |\mathcal{N}(b, h, i)| = K \ll N\]

such that query \((b, h, i)\) attends only to keys and values indexed by \(\mathcal{N}(b, h, i)\).

**The Principle:** In any batched SDPA kernel, every query in the batch must share the same K/V tensor. Heterogeneous per-query neighbor sets can only be communicated to the kernel through exactly one of two mechanisms, and choosing one forces the other:

\[\text{Mask indirection} \; \Longleftrightarrow \; \text{1 kernel launch, } N^2 \text{ mask entries}\]
\[\text{Gather specialization} \; \Longleftrightarrow \; N \text{ kernel launches, } 0 \text{ mask entries}\]

No mechanism exists to have both **1 kernel launch** and **0 mask entries** with heterogeneous neighbor sets.

---

## 2. Formal Derivation

### 2.1 The SDPA Contract

`F.scaled_dot_product_attention(q, k, v, attn_mask)` imposes:

\[
\mathbf{y}_{b,h,i,:} = \sum_{j=0}^{N-1} \alpha_{ij}^{(b,h)} \cdot \mathbf{V}_{b,h,j,:},
\quad
\alpha_{ij}^{(b,h)} = \frac{\exp\left(\frac{\langle\mathbf{Q}_{b,h,i,:}, \mathbf{K}_{b,h,j,:}\rangle}{\sqrt{d_h}} + \mathbf{M}_{b,h,i,j}\right)}{\sum_{j'=0}^{N-1} \exp\left(\frac{\langle\mathbf{Q}_{b,h,i,:}, \mathbf{K}_{b,h,j',:}\rangle}{\sqrt{d_h}} + \mathbf{M}_{b,h,i,j'}\right)}
\]

All \(N\) key tokens \(\mathbf{K}_{b,h,:,:}\) are materialized in the denominator for every query \((b, h, i)\). The mask \(\mathbf{M} \in \mathbb{R}^{B \times H \times N \times N}\) is the **only** mechanism to exclude individual key-query pairs from contributing to the softmax.

### 2.2 The Mask Approach (KNNMaskSparseAttention)

Define the mask from KNN neighbor sets:

\[
\mathbf{M}_{b,h,i,j} = \begin{cases}
0 & \text{if } j \in \mathcal{N}(b, h, i) \\
-\infty & \text{otherwise}
\end{cases}
\]

Then a single call suffices:

```python
y = F.scaled_dot_product_attention(q, k, v, attn_mask=M)  # 1 kernel, (B,H,N,N) mask
```

**Cost structure:**
- Kernel launches: \(1\) (EfficientAttention or cuDNN masked SDPA)
- Mask construction: \(O(NK)\) via scatter (KNN indices → N×N boolean)
- Memory: \(O(N^2)\) for mask tensor (128 KB at N=256 in fp16)
- Attention compute: \(O(N^2 d_h)\) — dense FLOPs, but masked arithmetic optimizes internally
- **Fixed overhead:** mask dispatch penalty ≈ 21 µs at N=256 (2.7× slower than unmasked flash)

### 2.3 The Gather Approach (GatherSparseAttention)

Construct per-query private K/V tensors by advanced indexing:

\[
\mathbf{K}^{(b,h,i)} = \mathbf{K}_{b,h,\mathcal{N}(b,h,i),:} \in \mathbb{R}^{K \times d_h},
\quad
\mathbf{V}^{(b,h,i)} = \mathbf{V}_{b,h,\mathcal{N}(b,h,i),:} \in \mathbb{R}^{K \times d_h}
\]

Then issue \(N\) independent SDPA calls with no mask:

```python
for i in range(N):            # N iterations, each launching SDPA
    y_i = F.scaled_dot_product_attention(
        q[:,:,i:i+1,:],         # (B, H, 1, d_h)  — q_len = 1
        k_gathered[:,:,i,:,:],  # (B, H, K, d_h)  — private K slice
        v_gathered[:,:,i,:,:],  # (B, H, K, d_h)  — private V slice
    )
```

In practice this is vectorized to `(B·N, H, 1, d_h)` but the semantics are identical — SDPA sees \(B·N\) independent groups each with \(q_{\text{len}} = 1\).

**Cost structure:**
- Kernel launches: \(B \cdot N\) (each head-group-query generates a launch)
- Gather copies: \(2 \cdot B \cdot H \cdot N \cdot K \cdot d_h\) bytes of scattered memory reads
- Attention compute: \(O(BNK d_h)\) — sparse FLOPs
- **Kernel launch overhead:** \(B \cdot N \cdot \tau_{\text{launch}}\) where \(\tau_{\text{launch}} \approx 10\;\mu s\)
- **No mask dispatch penalty** (mask = None, but \(q_{\text{len}} = 1\) prevents fast dispatch anyway)

### 2.4 Why They Cannot Be Combined

The SDPA API has three arguments for data: `(q, k, v)`. The mask is a fourth, optional argument. There is no fifth argument for "per-query K/V index sets." The sparse access pattern \(\mathcal{N}(b, h, i)\) must be encoded **either** in the data layout (gather, fragmenting the batch) **or** in the mask argument (mask, paying the dispatch penalty). No hybrid API exists in cuDNN, FlashAttention, or PyTorch's SDPA interface.

This is not a PyTorch limitation — it reflects the GPU execution model. A single kernel grid processes all queries by partitioning them across thread blocks. If different queries point to different regions of K/V memory, those regions must be specified. The mask encodes this as a per-element boolean override; gather encodes it by pre-rearranging memory. Both are valid encodings, but each pays its own cost. There is no free encoding.

---

## 3. Quantitative Consequence (N=256, fp16, A500)

| Mechanism | Kernel launches | Attention compute | Overhead dominated by | CUDA time |
|---|---|---|---|---|
| Mask (KNN scatter) | ~1 | \(O(N^2 d_h)\) | Mask dispatch penalty (+21 µs) | 33 µs |
| Gather (q_len=1) | ~2048 | \(O(NK d_h)\) | Launch overhead (2048 × 10 µs) | 994 µs |
| Unmasked dense | 1 | \(O(N^2 d_h)\) | None | 12 µs |

**The mask dispatch penalty (21 µs) is a single fixed cost paid once. The gather tries to avoid it but substitutes 2048 kernel launches, each costing ~10 µs overhead — 2048 × 10 µs = 20,480 µs just for launch overhead.** The gather approach spends ~600× more on kernel launch overhead than the masked approach spends on its entire computation.

---

## 4. Practical Resolution

For spatial-cutoff attention at N < 2000:

1. **Accept the mask.** It is the cheapest mechanism to encode heterogeneous per-query access in a single kernel launch. The alternative (fragmentation) is structurally worse at all N where \(\tau_{\text{launch}} \cdot N > \tau_{\text{mask}}\).

2. **Make the mask cheap to construct.** Cache `cdist(yx, yx)` once (CachedDistAttention). Build the N×N mask via scatter from KNN indices — \(O(NK)\), not \(O(N^2)\).

3. **The remaining 21 µs overhead is irreducible under the current SDPA API.** The only escape route is `flex_attention` with a `score_mod` callback that folds the spatial cutoff into a single compiled Triton kernel — no separate mask tensor, no dispatch fork.

At the dataset scale (N ≈ 140): a single 21 µs mask dispatch penalty is 0.005% of a 400 ms training step. The entire mask construction + masked SDPA is ~50 µs — 0.01% of step time. Attention optimization at this N is a solved problem; further effort belongs in the benchmark lab, not the training pipeline.
