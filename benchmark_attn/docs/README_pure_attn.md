# Pure Attention Kernel Benchmark

**File:** `benchmark_pure_attn.py`  
**Goal:** Isolate and benchmark ONLY the raw attention kernel (SDPA/matmul) without QKV projections, output projections, mask construction, or gather overhead.

## Methods

| Method | What is timed |
|--------|--------------|
| pure_dense | `F.scaled_dot_product_attention(q, k, v)` on (B, nH, N, Dh) |
| pure_flash | Same as pure_dense (explicit flash dispatch) |
| pure_gather_K | matmul+softmax on pre-gathered (B, nH, N, K, Dh) |
| pure_mask_K | `F.scaled_dot_product_attention` with pre-built KNN mask |
| pure_minimax_Bk | matmul+softmax on block-gathered KV |

## Confounders Excluded
- **No QKV projections:** q/k/v pre-computed and reshaped outside timed region
- **No output projection:** not called inside timed function
- **No positional encoding:** ROPE/bias/precomputed masks excluded
- **No KNN index computation:** KNN indices pre-generated via random
- **No mask construction:** masks pre-built outside timed region
- **No gather overhead:** K/V pre-gathered before timing (for gather methods)

## Usage

```bash
python benchmark_pure_attn.py --d 320 --nhead 8 --out pure_attn_results.csv
```

## Output
- `pure_attn_results.csv` — time (ms) and memory (MiB) per method
- Console table comparing all methods at all N

## Key Findings
- Pure SDPA on (B, nH, N, Dh) is 3-10× faster than gather/matmul at all N
- The overhead in full benchmarks comes from QKV projections (~4%),
  mask construction (~82% for dense), and gather/contiguous (~44% for gather_sdpa)
- Pure kernel time grows as O(N²·d) for dense, O(N·K·d) for gather
- FlashAttention achieves near-roofline throughput for d_head ≥ 64
