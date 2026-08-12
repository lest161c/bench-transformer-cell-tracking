# Gather-Sparse Attention Benchmark

Standalone benchmarks for sparse attention mechanisms in transformer-based cell tracking, part of the **Trackastra** research project.

**Core idea:** Replace N×N dense attention with KNN-gathered N×K sparse attention. Precompute KNN indices, gather K/V, run SDPA on N×K tensors. Complexity drops from O(N²) to O(NK).

## Key Results

Tested on NVIDIA A500 (fp16, B=2, d=256, h=4). All numbers single-layer forward pass.

| N | Method | Time (s) | GPU Mem (MB) | vs Dense |
|---|--------|----------|-------------|----------|
| 2048 | Dense | 0.0085 | 142 | 1× |
| 2048 | Sparse K=16 | 0.0089 | 102 | 0.96× speed, 1.4× mem savings |
| 8192 | Dense | 0.1391 | 2200 | 1× |
| 8192 | Sparse K=4 | 0.0237 | 120 | **5.9× faster**, 18× less mem |
| 8192 | Sparse K=16 | 0.0361 | 408 | **3.9× faster**, 5.4× less mem |
| 8192 L=4 | Dense | OOM | OOM | — |
| 8192 L=4 | Sparse K=4 | 0.0935 | 385 | Fits when dense OOMs |

**Spatial reorder** (Hassani et al. 2024): negligible speedup on random data (~1-3%). Real data may differ.

**Attention backends:** FlashAttention/CuDNN vs Math backend ~1.5-2× throughput difference at N≥512.

## Setup

```bash
uv sync
```

Requires Python ≥3.14, PyTorch ≥2.11 with CUDA.

## Usage

All scripts run from the `benchmark_attn/` directory with the project venv:
`cd benchmark_attn && .venv/bin/python <script>`.

### Core benchmarks

```bash
# Main sparse vs dense sweep (N up to 8192, K 4-64; extend via --Ns/--Ks flags)
python scripts/benchmarks/benchmark_sweep.py --methods gather-sdpa,dense_flash,dense_masked --Ks 4,16,64

# Unified benchmark: all methods × all N, forward + peak memory (mask-knn at N>=2048)
python scripts/benchmarks/benchmark_sweep.py --methods gather-sdpa,gather-fused,gather-matmul,mask-knn,knn-relpos,minimax,nsa,dense_flash,dense_masked

# CachedDistAttention real measurement (classes in local src/attention_modules.py)
python benchmarks/benchmark_cached_dist.py

# Gather variants: gather-sdpa (GatherSparseAttention), gather-fused (GatherSparseFusedAttention), gather-matmul (GatherSparseMatmulAttention)
python scripts/benchmarks/benchmark_sweep.py --methods gather-sdpa,gather-fused,gather-matmul

# Profiler (per-operation CUDA time breakdown) → profiler_out/profile_gather_results.json
python benchmarks/profile_gather_sparse.py

# Mask vs gather numerical equivalence across 18 configs (console output)
python benchmarks/validate_mask_vs_gather.py

# Plot main results → benchmark_sparse.html
python analysis/plot_sparse.py
```

### Additional benchmarks

```bash
# KNN method comparison (analytical model)
python benchmarks/benchmark_knn_methods.py

# Mask vs gather at small N (N ≤ 512)
python benchmarks/benchmark_mask_vs_gather.py

# Blockwise norm / FFN vs attention ratio (analytical)
python benchmarks/benchmark_blockwise_norm.py
python benchmarks/benchmark_ffn_attention_ratio.py

# Generic profiler wrapper
python benchmarks/benchmark_profiler.py

# All spatial methods (NOTE: gather_knn_ms column is not populated; do not cite gather)
python benchmarks/benchmark_all_spatial_methods.py

# Small-N behavior (N ≤ 512)
python scripts/benchmarks/benchmark_sweep.py --methods gather-sdpa,gather-fused,gather-matmul --Ns 128,256,512 --Ks 16 --layers 1

# FlashAttention with learned cutoff
python benchmarks/benchmark_flash_with_cutoff.py
```

### Exploratory / one-off

```bash
python exploratory/benchmark_pure_attn.py
python exploratory/benchmark_soft_decay_measure.py
python exploratory/benchmark_no_bias_flashattn.py
python exploratory/benchmark_no_mask_soft_decay.py
python exploratory/benchmark_flex_spatial.py
python exploratory/benchmark_spatial_block_partition.py
python exploratory/benchmark_spatial_flash.py
python exploratory/benchmark_spatial_cutoff_verification.py
python exploratory/benchmark_data_pipeline.py
python exploratory/benchmark_sdpa_backends.py
python exploratory/benchmark_system_impact.py
python exploratory/benchmark_trackastra_inference.py
python exploratory/benchmark_dinov3_comparison_v2.py
python exploratory/benchmark_reorder_real_vs_random.py
python exploratory/profiler_recipe.py
```

### Analysis / visualization

```bash
# Tile size analysis
python analysis/tile_size_analysis.py

# Cosine histogram analysis
python analysis/cosine_histogram.py

# Verify CUDA attention backends
python analysis/verify_cudnn.py

# Tra-AOGM visualization
python analysis/tra_aogm_visualization.py

# Collapse visualization
python analysis/vis_collapse.py

# Generate the HTML dashboard (aggregates results/ artifacts)
python analysis/make_report.py
```

## Structure

```
├── src/                            # library / implementation code
│   ├── attention_modules.py        # core attention classes
│   ├── positional_encoding.py      # RoPE + bin helpers
│   └── bench/                      # shared benchmark harness
│       ├── timing.py               # measure(), try_bench()
│       ├── data.py                 # make_inputs(), compute_knn_indices()
│       ├── registry.py             # METHOD_REGISTRY (9 methods)
│       ├── io.py                   # write_results_csv()
│       └── cli.py                  # add_common_args(), parse_int_list()
│
├── scripts/                        # runnable entry points
│   └── benchmarks/
│       └── benchmark_sweep.py      # consolidated method × L × N × K sweep
│
├── docs/                           # SPEC.md, REPRODUCTION.md, sidecar READMEs, REFACTOR_REMARKS.md
├── exploratory/                    # one-off research scripts
└── results/                        # generated artifacts
```

## Reference

- Gallusser & Weigert, *Trackastra*, ECCV 2024 — [github.com/weigertlab/trackastra](https://github.com/weigertlab/trackastra)
- Hassani et al., *Neighborhood Attention*, ECCV 2024
