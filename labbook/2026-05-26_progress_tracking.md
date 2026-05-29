# Progress Tracking: Week 2-3 (May 19-26)

**Week 1 (May 12):** Isolated KNN attention benchmark + token reordering.
Token reordering conclusively dead (<1.03x). KNN gather implemented.

---

## Week 2 (May 19-23) — Architecture + Planning

### Done before this conversation
- Standalone KNN sparse attention benchmark (A500/4090: N∈[128,8192], K∈[4,64], L∈[1,12])
- Combined KNN vs dense ablation (A500, synthetic, d=128, L=4)
- Initial SSL pipeline using **wrong** ASCENT (aircraft BCE architecture)
- Downstream comparison: old SSL gave 3x faster convergence, 26% lower val loss
- Speedup analysis: estimated 5-8x total with KNN+SSL on H100

### Key findings from Week 2
- KNN sparse is SLOWER than dense at N<1024 (overhead dominates)
- KNN K=4 gives 4.7x lower val loss than dense on synthetic task → structural regularizer
- Token reordering: <3% benefit → dropped
- Old SSL (BCE) gave 3x downstream convergence but architecture was wrong

---

## Week 3 (May 23-24) — This Conversation

### SSL pipeline rewrite
- Replaced entire SSL pipeline: wrong-ASCENT (aircraft BCE) → correct-ASCENT (contrastive InfoNCE)
- New files: `CellEmbedder` (track_encoder.py), `nt_xent_loss` (train_ssl.py), two-view distortions, label-sorted collation
- Fixed transformer NaN bug (LayerNorm on custom block → replaced with stock TransformerEncoderLayer)

### H100 experiments (cluster)
- **Scaling benchmark:** N=128..8192, L=1/6/12, K=4/16/32/64 on H100
  - KNN K=4 is 1.6x faster at N=2048, 1.5x at N=4096
  - KNN K=4 is FASTER at ALL N (no crossover like on A500)
  - Dense excluded at N>4096; KNN fits to N=8192
- **SSL pretraining:** 50 epochs, all 6 conditions, d=128, L=4
  - Loss: 3.46→2.97 (plateau at epoch 10)
  - Consistency: 0.97→0.92 (high from start, barely changes)

### SSL collapse discovery
- **Root cause:** Representational collapse — all cells map to nearly identical embeddings (cosine sim ~0.89)
- **Diagnostic:** Inter-cell sim = 0.85-0.90, pos-pair sim = 0.91-0.94, gap = 0.05
- **Causes identified:**
  1. 7-dim regionprops have small inter-cell variance → no discriminative signal
  2. Coordinate features dominate → model uses positional shortcut
  3. Augmentations too mild to break shortcut
- **Ablation results:**
  - Stronger augmentations → WORSE collapse (inter-sim 0.90→0.96)
  - Remove coords → dimension mismatch crash
  - Smaller model → same collapse

### Architecture verification
- **Synthetic test:** 16-dim unique cell signatures + 4-dim shared noise
  - Loss: 1.41→**0.03** in 20 epochs
  - Inter-cell sim: 0.68→**0.17**
  - Generalization: pos_sim=0.80 vs neg_sim=0.13 (**gap=0.67**)
  - **The NT-Xent loss and CellEmbedder architecture are correct**

### Image feature exploration (May 24)
- Tested frozen ResNet18, frozen DINOv2 ViT-S, scratch CNN on 64×64 patches
- **ImageNet backbones do NOT transfer** to bacterial microscopy (loss flat, inter-sim 0.8-0.95)
- **Trainable scratch CNN shows direction** (loss 3.58→2.88) but undertrained at 200 frames × 20 epochs

### Queued cluster jobs (May 24)
- **Distortion ablation:** 7 variants × 20 epochs SSL (full, -jitter, -dropout, -noise, -photo, -elastic, -affine)
- **Trackastra KNN training:** baseline (dense), K=4, K=16, K=32 — all 100 epochs on H100

---

## Timeline summary

| Week | What | Status |
|------|------|--------|
| May 12 | KNN attention benchmark, token reorder evaluation | Done |
| May 19-23 | Combined ablation, old SSL pipeline, speedup analysis | Done |
| May 23-24 | SSL rewrite (correct ASCENT), H100 benchmarks, collapse discovery, image feature tests | Done |
| May 24-26 | Distortion ablation, Trackastra KNN training | Running |
| After | Compile results, write report | Pending |
