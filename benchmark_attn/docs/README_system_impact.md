# System Impact Benchmark

**File:** `benchmark_system_impact.py`  
**Goal:** Measure FLOP and memory breakdown across all Trackastra training components to identify the true bottlenecks.

## Components Analyzed

| Component | What is measured | Scaling |
|-----------|-----------------|---------|
| Attention (12 layers) | QKV, QK^T, softmax, attn·V, outproj | O(N²·d) |
| FFN (12 layers) | Linear(320,640) + GELU + Linear(640,320) | O(N·d²) |
| Einsum | Outer product "bnd,bmd→bnm" | O(N²·d) |
| blockwise_causal_norm | 4× scatter_reduce per sample | O(N²·n_blocks) |

## Confounders Excluded
- **No GPU kernel overhead:** Analytical FLOP model, not measured
- **No data loading:** Regionprops/augmentation excluded (CPU-bound)
- **No optimizer:** AdamW step excluded (constant overhead)
- **Components isolated:** Each measured independently

## Usage

```bash
python benchmark_system_impact.py --out system_impact_results.csv
```

## Output
- `system_impact_results.csv` — FLOPs per component at each N
- `system_impact_breakdown.png` — 4-panel: FLOP stacked bar, % distribution, ratio, memory
- `blockwise_norm_overhead.png` — normalization cost vs window size
- `optimization_priority.png` — which component to optimize first

## Key Findings (analytical MACs model, d_model=320)

| N | Attention % | FFN % | Total GMACs | Memory (MiB) |
|---|------------|-------|-------------|---------------|
| 64 | 52% | 48% | 0.66 | 0.75 |
| 128 | 54% | 45% | 1.39 | 3.0 |
| 256 | 58% | 41% | 3.04 | 12.0 |
| 512 | 64% | 35% | 7.13 | 48.0 |
| 1024 | 71% | 27% | 18.46 | 192.0 |
| 2048 | 79% | 19% | 53.7 | 768.0 |

- **Low N (~50-100): FFN ≈ Attention** — linear terms dominate
- **Typical N (~200-300): Attention ~55-60%** — N² term growing
- **Large N (>1000): Attention dominates** — N² overtakes linear
- **Memory grows O(N²)** — attention masks stored across 12 layers for backprop
- **Primary bottleneck: memory from N² masks** (not compute)
- **Secondary: blockwise_causal_norm** serial loop over batch samples
