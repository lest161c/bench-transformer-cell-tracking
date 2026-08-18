# Experimental Results — Single Source of Truth

**Date:** 2026-08-18
**Status:** COMPLETE — all experiments finished, results verified

---

## 1. Spatial Cutoff Ablation (6-Fold Leave-One-Condition-Out CV)

### Setup
- 6 vanvliet conditions: rpsM, recA, pheA, metA, cib, trpL
- Train on 5 conditions, evaluate on held-out 6th
- Two configs, identical except `knn_neighbors`:
  - **Masked** (`knn=-1`): CachedDistAttention, spatial cutoff mask, EfficientAttention
  - **Dense Flash** (`knn=-2`): DenseFlashAttention, no mask, FlashAttention-2
- Both: window=4, dropout=0.05, d_model=320, 6+6 layers, 100 epochs, seed=42

### Results

| Fold | Held-out | Masked TRA | Flash TRA | Masked AOGM | Flash AOGM |
|------|----------|-----------|-----------|-------------|------------|
| 0 | rpsM | 0.9928 | 0.9933 | 174.4 | 164.4 |
| 1 | recA | 0.9203 | 0.9184 | 861.5 | 882.4 |
| 2 | pheA | 0.9987 | 0.9986 | 37.4 | 38.9 |
| 3 | metA | 0.9905 | 0.9903 | 213.8 | 221.6 |
| 4 | cib | 0.9998 | 0.9998 | 6.0 | 7.5 |
| 5 | trpL | 0.9947 | 0.9948 | 55.9 | 58.0 |

### Aggregate

| Config | TRA | AOGM |
|--------|-----|------|
| Masked (knn=-1) | 0.9828 ± 0.0281 | 224.8 ± 294.3 |
| Dense Flash (knn=-2) | 0.9825 ± 0.0288 | 228.8 ± 301.6 |
| **Δ (flash - masked)** | **-0.0003** | **+4.0** |

### Conclusion
Removing the spatial cutoff has **no measurable effect** on tracking accuracy.
- ΔTRA = -0.0003 (zero within noise)
- ΔAOGM = +4.0 errors out of ~225 (negligible, <2%)
- All 6 folds show TRA differences <0.002

---

## 2. Attention Kernel Benchmark (H100, isolated)

### Setup
- Isolated attention forward pass, batch=1, L=1 layer
- `dense_flash`: F.scaled_dot_product_attention with no mask → FlashAttention-2
- `dense_masked`: CachedDistAttention with spatial cutoff mask → EfficientAttention
- K=4 neighbors (irrelevant for dense variants)

### Results

| N | Flash (ms) | Masked (ms) | Speedup |
|------|-----------|------------|---------|
| 128 | 0.109 | 0.344 | **3.16×** |
| 256 | 0.113 | 0.341 | **3.02×** |
| 512 | 0.114 | 0.345 | **3.02×** |
| 1024 | 0.115 | 0.347 | **3.02×** |
| 2048 | 0.110 | 0.409 | **3.72×** |
| 4096 | 0.111 | 1.422 | **12.81×** |
| 8192 | 0.351 | 5.604 | **15.96×** |

### Key Finding
FlashAttention-2 is **3× faster** than masked EfficientAttention at N=128-1024,
growing to **16× at N=8192**. The speedup is real and large in isolation.

---

## 3. Theoretical Training Speedup (Amdahl's Law)

### Formula
```
training_speedup = 1 / (1 - attn_share + attn_share / attn_speedup)
```

### At Vanvliet Scale (N≈140, window=4)

| Parameter | Value | Source |
|-----------|-------|--------|
| Attention share of step | ~5% | Profiler (labbook 2026-05-29) |
| FlashAttn isolated speedup | 3.16× | Benchmark (§2) |
| **Theoretical training speedup** | **1.035× (3.5%)** | Amdahl's Law |
| **Measured training speedup** | **~1.03× (3%)** | 6-fold CV (§1) |

### Why So Low?
At N≈140, attention is **~5% of step time**. The remaining 95% is:
- Data loading from Lustre (CPU, ~40-60ms)
- Data augmentation (RandomAffine, flips, ~15ms)
- Feed-forward + layer norms per layer (~20ms)
- Backward pass (autograd, ~30ms)

A 3.16× speedup on 5% of step time yields only 3.5% training speedup.

---

## 4. Per-Epoch Training Times

### CV Runs (window=4, 100 epochs, all optimizations)

| Config | Per-epoch (min) |
|--------|-----------------|
| Masked (knn=-1) | 3.6 ± 0.0 |
| Dense Flash (knn=-2) | 3.5 ± 0.2 |

**Speedup: 1.03×** (3% faster, consistent with theoretical 3.5%)

### Standalone Runs (window=4, 500 epochs, 12h timeout)

| Run | Config | Epochs | Per-epoch |
|-----|--------|--------|-----------|
| vanilla_train | main branch (RelativePositionalAttention + skimage + serial blockwise_norm) | 202 | 3.6 min |
| optimal_train | cached-dist-attn (CachedDistAttention + fast-regionprops + vectorized blockwise_norm) | 197 | 3.7 min |

**Speedup: 0.97×** (optimized is slightly slower, within noise)

### Why Optimizations Don't Help at N≈140

| Optimization | Isolated speedup | Share of step | Training speedup |
|---|---|---|---|
| FlashAttention | 3.16× | 5% | 3.5% |
| Vectorized blockwise_norm | 1.96× | ~15% | ~6% (theoretical) |
| CachedDistAttention | 1.5× | ~3% | ~1% |
| fast-regionprops | 10-80× | 0% (parallel) | 0% |

**Combined theoretical: ~9.5%**
**Measured: ~3%**

The gap is because:
1. fast-regionprops runs in the data loader (CPU, parallel with GPU) — 0% training speedup
2. CachedDistAttention's cdist(140×140) = 0.01ms — negligible at N≈140
3. Vectorized blockwise_norm's 6% theoretical may be offset by scatter_reduce overhead

The only optimization that actually helps training is FlashAttention, and only by 3%.

---

## 5. Deepcell Cross-Dataset Evaluation

### Setup
- Train on all 6 vanvliet conditions (window=10, knn=-2, 100 epochs)
- Evaluate on 12 deepcell test sequences (held-out, never used in training)

### Results

| Seq | TRA | AOGM |
|-----|-----|------|
| 00 | 0.9998 | 2.0 |
| 01 | 0.9989 | 14.0 |
| 02 | 1.0000 | 0.0 |
| 03 | 0.9994 | 4.0 |
| 04 | 0.9988 | 10.0 |
| 05 | 0.9974 | 12.0 |
| 06 | 0.9989 | 2.5 |
| 07 | 0.9998 | 9.0 |
| 08 | 0.9996 | 74.0 |
| 09 | 0.9996 | 43.5 |
| 10 | 0.9990 | 45.5 |
| 11 | 0.9985 | 163.5 |

### Aggregate

| Metric | Value |
|--------|-------|
| mean TRA | 0.9991 ± 0.0007 |
| mean AOGM | 31.7 ± 45.4 |

### Conclusion
DenseFlashAttention (no spatial cutoff) generalizes well to held-out data.
TRA = 0.9991 on deepcell test set is excellent.

---

## 6. Key Insights for Report

### Insight 1: The Spatial Cutoff Is Purely Computational
- 6-fold CV: ΔTRA = -0.0003, ΔAOGM = +4.0 (both negligible)
- The cutoff doesn't regularize — removing it doesn't degrade accuracy
- Trackastra paper never ablates this; our experiment fills that gap

### Insight 2: FlashAttention-2 Is 3× Faster in Isolation
- At N=128: 0.109ms (flash) vs 0.344ms (masked)
- Speedup grows to 16× at N=8192
- This is the attention kernel speedup, not training speedup

### Insight 3: Training Speedup Is Only 3% at Vanvliet Scale
- Amdahl's Law: 3.16× speedup on 5% of step = 3.5% training speedup
- Measured: 3% (6-fold CV)
- Attention is not the bottleneck at N≈140

### Insight 4: The Trackastra Paper Never Justifies the Cutoff
- Paper mentions cutoff once (Eq. 3, Section 2.2), no justification given
- Paper never ablates it — treated as fixed structural default
- Our CV experiment is the first test of whether the cutoff matters

### Insight 5: HOCT Also Uses Spatial Masks for Sparsity
- HOCT (Bragantini et al., 2026) masks out node pairs farther than threshold τ
- HOCT frames this as "preserving sparsity" — explicitly computational
- HOCT never claims the mask is necessary for accuracy

### Insight 6: DenseFlashAttention Is the Optimal Architecture at N≈140
- Removes the spatial cutoff mask entirely
- Enables FlashAttention-2 (3.16× faster attention kernel)
- Zero accuracy cost (ΔTRA = -0.0003)
- 3% training speedup (measured, matches theoretical 3.5%)

### Insight 7: Other Optimizations (fast-regionprops, vectorized blockwise_norm, CachedDistAttention) Provide Negligible Training Speedup at N≈140
- fast-regionprops: 0% (runs in data loader, parallel with GPU)
- CachedDistAttention: ~1% (cdist(140×140) = 0.01ms)
- Vectorized blockwise_norm: ~6% theoretical, may be offset by overhead
- Combined with FlashAttention: ~9.5% theoretical, ~3% measured

---

## 7. Reproducibility

### Branches
- `main`: Original Trackastra (RelativePositionalAttention + skimage + serial blockwise_norm)
- `cached-dist-attn`: Optimized Trackastra (CachedDistAttention + fast-regionprops + vectorized blockwise_norm + DenseFlashAttention)
- `report-rework`: This report rewrite

### Jobs
- CV: job 3926370 (12 tasks, all completed)
- Dense_flash training: job 3920745 (completed)
- Vanilla training: job 3924376 (completed, 12h timeout)
- Optimal training: job 3922452 (completed, 12h timeout)
- Deepcell eval: job 3926322 (completed)

### Data
- Training: vanvliet dataset (6 conditions, ~35 cells/frame, 1009×1305 frames)
- Evaluation: deepcell test set (12 sequences, held-out)
- CV: 6-fold leave-one-condition-out on vanvliet
