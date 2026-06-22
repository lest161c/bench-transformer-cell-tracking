# Meeting 2026-06-18 — Work Session: Implementation

**Date:** 2026-06-22
**Context:** Implementation of tasks from meeting_18_06_26.txt.
**Depends on:** `2026-06-16_msa_paper_review_negative.md`, `2026-06-15_dino_contrastive_ssl.md`, `2026-06-08_attention_optimization_landscape.md`

---

## 1. Tasks Completed

### 1.1 MiniMax Sparse Attention — Benchmark Implementation

Added `MiniMaxSparseAttention` to `benchmark_attn/model_parts.py`. Adapted from GQA to MHA (each head independently selects blocks). Architecture:
- Index Branch: block-level max-pooling scores from Q/K dot products
- Top-k block selection per head per query (block_size=32/64/128, k=1-4)
- Block-level KV gather + manual matmul attention

Integrated into `benchmark_full.py` sweep with block sizes [32, 64, 128].

**Verdict:** Per labbook 2026-06-16, MSA is architecturally incompatible at our scale. This implementation is for benchmark completeness and citation in the report. Expected to be slower than KNNMaskSparseAttention at N<2048.

### 1.2 Pure Attention Kernel Benchmark

Created `benchmark_attn/benchmark_pure_attn.py` — isolates ONLY the SDPA/matmul kernel:
- Pre-computes Q, K, V tensors outside timed region
- No QKV projections, output projections, mask construction, or gather overhead
- Tests: pure_dense, pure_flash, pure_gather (K=4,16,32,64,128), pure_mask (K=4,16), pure_minimax (Bk=32,64,128)
- Measures raw attention compute scaling with respect to N and K

### 1.3 Cosine Similarity Histogram — Intra + Inter

Created `benchmark_attn/cosine_histogram.py` — full distributions from real data:
- Computes both intra-cell AND inter-cell cosine similarity histograms (was only intra before)
- Processes vanvliet frames through distortion pipeline
- Supports both regionprops (7D) and DINOv2/DINOv3 (384D) features
- Output: multi-panel histogram figure + CDF + CSV summary
- Auto-interprets results (PASS/BORDERLINE/FAIL for contrastive SSL)

Key finding: Regionprops 7D features show inter-cell cosine sim ~0.89 (collapse). Negative similarity is not observed because all regionprops are non-negative scalars (area, intensity, etc.) — L2-normalized dot product of non-negative vectors is always >= 0.

### 1.4 DINOv3 Support

Updated `trackastra/trackastra/model/dino_encoder.py`:
- `DINOBackbone` now supports `version="v2"` (dinov2_vits14, 384D) and `version="v3"` (dinov3_vits16, 384D)
- `DINOProjection` accepts `version` parameter, auto-adjusts Linear input dim
- Backward compatible — existing code using `DINOBackbone()` or `DINOProjection()` defaults to v2
- DINO_SPECS dict maps version → {repo, model, dim}

### 1.5 Trackastra Inference Benchmark

Created `benchmark_attn/benchmark_trackastra_inference.py`:
- Measures `get_features()` and `predict_windows()` separately
- Tests on both example bacteria data and synthetic masks at N=[10,25,50,100,200,500,1000]
- Outputs: CSV with time/memory per stage, scaling plots (time vs N, memory vs N, percent breakdown, per-cell cost)
- Isolates CPU-bound feature extraction from GPU-bound transformer inference

### 1.6 TRA/AOGM Visualization — Gather vs Scatter + 1-TRA

Created `benchmark_attn/tra_aogm_visualization.py`:
- 6-panel figure: 1-TRA error rate, AOGM, TRA vs AOGM scatter + Pareto front, gather vs scatter speed, Edge/Div F1, speed-accuracy tradeoff
- Explicitly labels gather-KNN vs mask-KNN (scatter) in speed panel
- Uses 1-TRA (error rate) for better visual separation (0.0028-0.0043 range vs 0.9957-0.9972)
- Detailed K graphs: time vs N, time vs K, memory vs N, speedup vs K for K=4,16,32,64,128

### 1.7 K=128 Added to Benchmarks

Updated `benchmark_full.py` (Ks=[4,16,32,64,128]) and `benchmark_pure_attn.py` (Ks=[4,16,32,64,128]).

---

## 2. SSL Distortion Plausibility Analysis

Compared DynaCLR vs ASCENT for the cell tracking SSL task:
- **ASCENT** (our approach): Spatial-only, single-frame distortions (jitter, affine, elastic, dropout), NT-Xent loss, operates on feature vectors
- **DynaCLR**: Temporal contrastive learning, uses cell tracking for positive pairs across time, operates on raw 3D image patches with ConvNeXt encoder

**Verdict:** ASCENT is correct for annotation-free cell tracking pretraining. DynaCLR requires tracking labels (defeating SSL purpose) and targets cell state dynamics (not tracking). All distortions in our pipeline are biologically plausible (jitter = Brownian motion, dropout = segmentation failure, etc.).

---

## 3. Why No Negative Similarity

Regionprops features (area, intensity, inertia, border_dist) are all non-negative scalars. After L2-normalization, the dot product of two non-negative vectors is always >= 0. DINO features, while not strictly non-negative, mostly land in the positive orthant due to ViT architecture (GELU activations, self-distillation pretraining). No code absolute value is taken; the all-positive feature space is inherent.

---

## 4. Files Modified / Created

| File | Change |
|------|--------|
| `benchmark_attn/model_parts.py` | Added `MiniMaxSparseAttention` class (~100 lines, MHA-adapted) |
| `benchmark_attn/benchmark_full.py` | Added MiniMax sweep, expanded Ks to [4,16,32,64,128] |
| `benchmark_attn/benchmark_pure_attn.py` | **New** — pure attention kernel benchmark |
| `benchmark_attn/cosine_histogram.py` | **New** — intra+inter cosine similarity with real data |
| `benchmark_attn/benchmark_trackastra_inference.py` | **New** — Trackastra inference timing breakdown |
| `benchmark_attn/tra_aogm_visualization.py` | **New** — gather vs scatter, 1-TRA, K=4..128 graphs |
| `trackastra/trackastra/model/dino_encoder.py` | DINOv3 support (version parameter, DINO_SPECS dict) |

---

## 5. Next Steps (Remaining Meeting Tasks)

| Priority | Task | 
|----------|------|
| Medium | Tile size influence on flash attention (min 4→128) |
| Medium | Identify most compute-intensive task in Trackastra training+inference |
| Low | Distribution analysis of cosine similarity (Xi, Poisson, etc.) |
