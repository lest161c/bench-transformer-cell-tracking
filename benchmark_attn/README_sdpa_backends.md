# SDPA Backend Dispatch Benchmark

**File:** `benchmark_sdpa_backends.py`  
**Goal:** Map which SDPA backend PyTorch selects for each (N, d_head) combination.

## Backends

| Backend | Conditions | Characteristics |
|---------|-----------|-----------------|
| FlashAttention (F) | d_head ∈ {16,32,64,128,256} ∧ N ≥ 64 | Fastest, tile-based O(N) SRAM usage |
| Memory-Efficient (M) | d_head % 8 = 0 ∧ N ≥ 16 | Lower memory, good at medium N |
| cuDNN/Math (C) | Everything else | Best at very small N, no SRAM constraints |

## Confounders Excluded
- No QKV projections, no positional encoding, no mask
- Single SDPA call per measurement
- Pre-computed q/k/v tensors outside timed region

## Usage

```bash
python benchmark_sdpa_backends.py --out sdpa_backend_results.csv
```

## Output
- `sdpa_backend_results.csv` — backend prediction per (d_head, N, backend, time_ms)
- `sdpa_backend_dispatch.png` — heatmap of backend dispatch
- `sdpa_timing_d_head_40.png` — timing for Trackastra's d_head=40
- `sdpa_head_dim_analysis.png` — flash-compatible d_heads + GFLOPS efficiency

## Key Findings
- Trackastra d_head=40: math/cuDNN at N<16, **mem_efficient at N≥16**
- **d_head=40 NEVER dispatches FlashAttention** (flash kernels only exist for {16,32,64,128,256})
- PyTorch 2.6 falls back to mem_efficient for head dimensions outside the hardcoded set
- At dataset N (~100-300): backend choice has <20% impact on SDPA time
- **Recommendation:** d_head=64 would guarantee FlashAttention dispatch
