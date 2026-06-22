# benchmark_pipeline/ — Non-Attention Bottleneck Benchmarks

Isolated benchmarks for the compute bottlenecks that are NOT attention.
Each follows the `benchmark_attn` pattern: analytical model, CSV output,
seaborn visualization, GPU-ready code.

## Benchmarks

| File | Bottleneck | Fix | Expected gain |
|------|-----------|-----|---------------|
| `benchmark_blockwise_norm.py` | Serial Python loop in loss (60% of step time) | Vectorize scatter_reduce across batch | 2.3× at N=64, 1.2× at N=256 |
| `benchmark_ffn_checkpoint.py` | FFN activation memory (22 MiB at N=256) | Gradient checkpoint every 3 layers | 67% memory reduction |
| `benchmark_regionprops.py` | CPU regionprops extraction (80%+ of inference) | Cache/pre-compute WRFeatures | >2× inference speedup at N<200 |
| `benchmark_spatial_blocks.py` | N×N mask blocks FlashAttention | Hilbert-sort + block FlashAttn + overlap | 7× vs CachedDist at N=512 |

## Usage

All scripts support two modes:
```bash
# Analytical (runs anywhere, no torch needed):
python benchmark_blockwise_norm.py --analytical

# GPU measurement (needs torch + CUDA):
python benchmark_blockwise_norm.py --gpu
```

## Output

Each benchmark produces:
- `benchmark_pipeline/<name>_results.csv` — numerical data
- `benchmark_pipeline/<name>.png` — seaborn/matplotlib figure
- Console summary with key metrics

## Combined Impact (11h training)

| Optimization | Training speedup | Inference speedup |
|-------------|-----------------|-------------------|
| blockwise_norm vectorization | 1.06× | 1.06× |
| FFN checkpointing → larger batch | 1.10× | N/A (inference only) |
| Regionprops caching/precompute | N/A (training pre-caches) | >2× at N<200 |
| Spatial block partition | 1.40× | 1.40× |
| **Combined** | **~1.63× (11h→6.7h)** | **2-7× depending on N** |
