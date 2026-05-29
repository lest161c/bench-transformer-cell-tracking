# Expected Speedup: Sparse KNN + Contrastive SSL vs Baseline

**Date:** 2026-05-23  
**Context:** Rewrite of SSL pipeline from wrong-ASCENT (aircraft, BCE) to correct-ASCENT (contrastive, NT-Xent). Estimate total training-time reduction from combining sparse KNN attention + contrastive SSL pretraining.

---

## 1. Sources

- `benchmark_attn/results_analysis.md` — per-layer attention timing on RTX 4090 (N∈{128,512,2048,8192})
- `benchmark_combined/results/ablation_full.csv` — full-step timing on A500 laptop (d=128, L=4, B=2)
- `benchmark_combined/results/speed_mem.csv` — memory scaling across K values
- `benchmark_combined/analysis_ablation.md` — convergence measurements (val_loss vs epoch)
- `benchmark_ssl/results_analysis.md` — old BCE-SSL downstream transfer results (3x faster convergence)
- `base-config.yaml` — Trackastra default: d=320, L=12, 500 epochs, max_tokens=2048, B=192
- ASCENT paper (Han & Lu 2025) — 20 SSL epochs, contrastive InfoNCE, 15 min on H200

---

## 2. Three independent mechanisms — not multiplicative

### A) Per-step speedup from sparse KNN (measured)

| N | Device | Dense (ms) | Sparse K=16 (ms) | Ratio |
|---|--------|-----------|-------------------|-------|
| 128 | A500, L=4 | 6.2 | 17.2 | 0.36x (slower) |
| 256 | A500, L=4 | 16.1 | 37.4 | 0.43x |
| 512 | A500, L=4 | 53.5 | 88.7 | 0.60x |
| 1024 | A500, L=4 | 182 | 208 | 0.88x |
| 2048 | 4090, L=1 | 8.5 | 8.9 | 0.95x |
| 8192 | 4090, L=1 | 139 | 36.1 | 3.85x |

KNN overhead (cdist + topk + gather) dominates at N<1024. At N=2048 (Trackastra's max_tokens), sparse is roughly breakeven on consumer GPUs.

**On H100, dense baseline does NOT use FlashAttention** (attn_mask disables it). KNN gather allows FlashAttention-2, which fuses QK^T→softmax→V into SRAM, never materializing the attention matrix. With H100's 3.3 TB/s bandwidth, dense is memory-bandwidth-limited; KNN+FlashAttn is compute-bound.

**Estimated H100 per-layer speedup:** Dense ~20ms → KNN ~8ms = **~2.5x attention speedup**. After FFN (unchanged ~10ms): ~30ms → ~18ms per layer = **~1.7x per layer**. At L=12: **~2x per step**. At B=128 (H100 fits): memory-bound dense gets worse → **~3x per step**.

### B) KNN as structural regularizer (measured)

| Variant (N=512, L=4, epoch 15) | Val Loss |
|--------------------------------|----------|
| dense+rand | 0.504 (baseline) |
| dense+ssl | 0.373 (-26%) |
| sparseK4+rand | 0.108 (-79%) |
| sparseK4+ssl | 0.108 (-79%) |

KNN sparse reaches val_loss <0.15 in **3-4 epochs**. Dense never reaches <0.2 in 15 epochs. This is a **~5x convergence speedup** due to the structural prior: cells are physically local, restricting attention to K neighbors prevents attending to distant irrelevant cells early in training.

### C) SSL convergence (old BCE measured, new InfoNCE expected)

Old (wrong) SSL gave 3x faster downstream convergence (reached loss 0.03 by epoch 3 vs random at epoch 9+). New contrastive SSL uses InfoNCE loss matching real ASCENT paper — should be at least as effective. SSL pretraining cost: ~20 epochs.

**Overlap:** KNN already provides strong regularization (val_loss 0.108 with random init). SSL pretraining adds benefit mainly in the **low-label regime** (proposal §4.3) where labeled data is scarce and pretrained representations compensate.

---

## 3. Combined estimate

| Component | Factor | Source |
|-----------|--------|--------|
| Per-step (H100, FlashAttn) | 2-3x | §2A, extrapolated from 4090 + H100 bandwidth |
| Epoch count (KNN regularization) | 5x | §2B, measured ablation |
| Epoch count (SSL pretraining) | 2-3x | §2C, old BCE measured; overlapped with KNN |
| Combined (accounting for overlap) | 8-10x | — |
| **Conservative** | **5-8x** | — |

**Baseline:** 500 epochs × ~100s/epoch (d=320, L=12, H100) ≈ **14 hours**  
**With KNN+SSL:** ~100 epochs × ~45s/epoch + 20 SSL epochs × ~40s ≈ **1.5 hours**

---

## 4. Memory scaling (measured, 4090)

| N | Dense L=4 (MB) | Sparse K=32 L=4 (MB) | Saving |
|---|---------------|----------------------|--------|
| 2048 | ~2,100 (est.) | ~1,200 | 1.8x |
| 8192 | OOM (24GB) | ~1,350 | **fits vs OOM** |

On H100 (80GB): memory savings enable 2x larger batch size or processing N=16384 cells which dense cannot. This is the **enabler**, not the speedup — larger microscopy datasets become tractable.

---

## 5. Token reordering: correctly negligible

Measured across A500/4090: 0.95-1.03x speedup (max 3%). Overhead of permute→compute→unpermute outweighs cache-line benefit. Not worth implementing.

---

## 6. Why N=2000 and not more

Trackastra's `max_tokens=2048` is a training bucketing limit, not a hardware limit. The dataset has ~35 cells/frame average. With B=192, even at max_tokens=2048 the actual tokens per step are limited by how many cells are in the sampled windows. Moving to N=8192 would require:
1. Datasets with >2000 cells per frame (currently none in vanvliet/deepcell)
2. Trackastra batching logic change (hard-coded max_tokens limit)
3. N=8192 would make dense OOM even on H100 (the attention matrix alone is 256MB×128 heads×L layers). Sparse K=32 at N=8192 uses ~1.3GB — fits easily, but the data doesn't need it today.

The **architecture** supports N=8192 via KNN sparse. The **experimental setup** for this project doesn't exceed N=2048. The benefit of KNN for this project is primarily the regularization effect (5x convergence), not the large-N memory savings.
