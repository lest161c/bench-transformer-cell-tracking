# Gather-Sparse Attention Benchmark

Standalone benchmarks for sparse attention mechanisms in transformer-based cell tracking, part of the **Trackastra** research project.

**Core idea:** Replace N×N dense attention with KNN-gathered N×K sparse attention. Precompute KNN indices, gather K/V, run SDPA on N×K tensors. Complexity drops from O(N²) to O(NK).

## Key Results

Tested on NVIDIA RTX 4090 (fp16, B=2, d=256, h=4). All numbers single-layer forward pass.

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

### Core benchmarks (load-bearing — feed the report)

```bash
# Main sparse vs dense sweep (N up to 8192, K 4-64; edit N_vals/K_vals at top to extend)
python benchmarks/benchmark_sparse.py

# Unified benchmark: all methods × all N, forward + peak memory (mask-KNN at N>=2048)
python benchmarks/benchmark_full.py

# CachedDistAttention real measurement (imports trackastra model_parts via shim)
python benchmarks/benchmark_cached_dist.py

# GatherSparseAttention V1/V2/V3 comparison
python benchmarks/benchmark_gather_v3.py

# Profiler (per-operation CUDA time breakdown) → profiler_out/profile_gather_results.json
python benchmarks/profile_gather_sparse.py

# Mask vs gather numerical equivalence across 18 configs (console output)
python benchmarks/validate_mask_vs_gather.py

# Plot main results → benchmark_sparse.html
python analysis/plot_sparse.py
```

### Superseded / historical (kept for provenance; see REPRODUCTION.md)

```bash
# KNN method comparison (analytical; superseded by benchmark_cached_dist.py)
python benchmarks/benchmark_knn_methods.py

# Mask vs gather at small N (superseded by benchmark_full.py for N>=2048)
python benchmarks/benchmark_mask_vs_gather.py

# Blockwise norm / FFN-ratio (analytical; superseded by benchmark_pipeline/*)
python benchmarks/benchmark_blockwise_norm.py
python benchmarks/benchmark_ffn_attention_ratio.py

# Generic profiler wrapper (superseded by profile_gather_sparse.py)
python benchmarks/benchmark_profiler.py

# All spatial methods (NOTE: gather_knn_ms column is broken (-1); do not cite gather)
python benchmarks/benchmark_all_spatial_methods.py

# Small-N behavior (feeds figure set 02)
python benchmarks/benchmark_small_n.py

# FlashAttention with learned cutoff
python benchmarks/benchmark_flash_with_cutoff.py
```

### Exploratory / one-off (moved to `exploratory/`; NOT cited in the report)

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

# Generate the internal HTML dashboard (aggregates results/ artifacts)
python analysis/make_report.py
```

## Structure

```
├── model_parts.py               # Core attention modules (GatherSparseAttention, RelativePositionalAttention, etc.)
├── config.yaml                  # Default hyperparameters
├── pyproject.toml               # Dependencies
├── README.md                    # Project overview
├── README_knn_methods.md        # KNN methods documentation
├── README_pure_attn.md          # Pure attention documentation
├── README_sdpa_backends.md      # SDPA backends documentation
├── README_system_impact.md      # System impact documentation
├── REPRODUCTION.md              # Benchmark reproduction guide
├── results_analysis.md          # Results analysis notes
├── uv.lock                      # Lockfile (gitignored)
├── validate_prompt.md           # Prompt validation notes (pending removal)
│
├── benchmarks/                           # Load-bearing + superseded benchmark scripts
│   ├── benchmark_all_spatial_methods.py            # Unified spatial methods (gather_knn_ms column broken)
│   ├── benchmark_blockwise_norm.py                 # Blockwise norm (superseded by benchmark_pipeline/*)
│   ├── benchmark_cached_dist.py                    # CachedDistAttention real measurement
│   ├── benchmark_ffn_attention_ratio.py            # FFN vs attention ratio (superseded)
│   ├── benchmark_flash_with_cutoff.py              # FlashAttention with learned cutoff
│   ├── benchmark_full.py                           # Unified benchmark: all methods × all N
│   ├── benchmark_gather_v3.py                      # Gather v3 benchmark
│   ├── benchmark_knn_methods.py                    # KNN method comparison (analytical)
│   ├── benchmark_mask_vs_gather.py                 # Mask vs gather at small N (superseded)
│   ├── benchmark_profiler.py                       # Generic profiler wrapper (superseded)
│   ├── benchmark_small_n.py                        # Small-N behavior
│   ├── benchmark_sparse.py                         # Dense vs sparse sweep (N up to 8192, K 4-64)
│   ├── profile_gather_sparse.py                    # Gather sparse profiler
│   └── validate_mask_vs_gather.py                  # Equivalence verification (18 configs)
│
├── exploratory/                        # Exploratory / one-off scripts (not cited in the report)
│   ├── benchmark_data_pipeline.py                  # Data pipeline
│   ├── benchmark_dinov3_comparison_v2.py           # Dinov3 comparison v2
│   ├── benchmark_flex_spatial.py                   # Flexible spatial
│   ├── benchmark_no_bias_flashattn.py              # No-bias FlashAttention
│   ├── benchmark_no_mask_soft_decay.py             # No-mask soft decay
│   ├── benchmark_pure_attn.py                      # Pure attention variants
│   ├── benchmark_reorder_real_vs_random.py         # Reorder speedup (no artifact)
│   ├── benchmark_sdpa_backends.py                  # SDPA backend comparison
│   ├── benchmark_soft_decay_measure.py             # Soft decay measurement
│   ├── benchmark_spatial_block_partition.py        # Block partition strategies
│   ├── benchmark_spatial_cutoff_verification.py    # Cutoff verification
│   ├── benchmark_spatial_flash.py                  # Spatial FlashAttention
│   ├── benchmark_system_impact.py                  # System impact breakdown
│   ├── benchmark_trackastra_inference.py           # Trackastra inference
│   └── profiler_recipe.py                          # Profiler recipe
│
├── analysis/                             # Plotting / report / verification scripts
│   ├── plot_sparse.py                  # Visualization → HTML
│   ├── make_report.py                  # HTML report generator
│   ├── verify_cudnn.py                 # CUDA backend verification
│   ├── tile_size_analysis.py           # Tile size analysis
│   ├── cosine_histogram.py             # Cosine histogram
│   ├── tra_aogm_visualization.py       # AOGM visualization
│   └── vis_collapse.py                 # Collapse visualization
│
├── slurm/                                # SLURM batch scripts
│   └── run_full_bench.slurm       # SLURM batch script
│
└── results/                              # Generated artifacts (gitignored)
    ├── benchmark_sparse_results.csv       # Benchmark output data
    ├── ..._results.csv                    # Various benchmark result CSVs
    ├── ..._analysis.png                   # Benchmark visualization PNGs
    ├── comprehensive_report.html          # Full HTML report
    └── profiling_report.html              # Profiling HTML report
```

## Reference

- Gallusser & Weigert, *Trackastra*, ECCV 2024 — [github.com/weigertlab/trackastra](https://github.com/weigertlab/trackastra)
- Hassani et al., *Neighborhood Attention*, ECCV 2024
