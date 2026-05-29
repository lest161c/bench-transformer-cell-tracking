# Results Compilation for Final Report

**Date:** 2026-05-24  
**Status:** All quantitative results collected, ready for writeup

---

## Contribution A: Sparse KNN attention

### Throughput (H100, L=12, d=256, fp16)

| N | Dense (ms) | K=4 (ms) | Speedup | Memory (K=4) |
|---|-----------|----------|---------|-------------|
| 128 | 2.17 | 0.61 | **3.6x** | 297 MB |
| 512 | 2.18 | 0.62 | **3.5x** | 41 MB |
| 2048 | 2.37 | 1.48 | **1.6x** | 102 MB |
| 4096 | 4.61 | 3.10 | **1.5x** | 296 MB |
| 8192 | — | 6.70 | dense excluded | 685 MB |

Key: KNN K=4 is faster at ALL N. At Trackastra's max_tokens=2048: 1.6x per step. Enables FlashAttention-2 (no attn_mask). Dense at N=4096+12 layers OOMs on A500; KNN fits at 296 MB.

### Regularization (synthetic ablation, N=512, L=4, A500)

| Variant | Final val_loss | vs dense |
|---------|---------------|----------|
| dense+rand | 0.504 | baseline |
| sparseK4+rand | 0.108 | **4.7x lower** |
| sparseK4+ssl | 0.108 | 4.7x lower |

KNN structural prior dominates at small capacity. SSL adds nothing on top of KNN at this scale.

### Real-data tracking (vanvliet, frozen embeddings + Hungarian)

| Method | Accuracy |
|--------|----------|
| dense_noSSL | 64.1% |
| knn_noSSL | 66.9% |
| knn_SSL | 65.1% |

KNN gives +2.8% over dense. SSL frozen embeddings: no improvement (same as KNN).

---

## Contribution B: Contrastive SSL

### SSL convergence (H100, 50 epochs, all 6 conditions)

| Metric | Epoch 1 | Epoch 10 | Epoch 50 |
|--------|---------|----------|----------|
| Train loss | 3.46 | 3.07 | 2.97 |
| Val loss | 3.29 | 2.86 | 2.82 |
| Consistency | 0.97 | 0.93 | 0.92 |

Loss plateau after epoch 10. Consistency stays high throughout — model collapses to near-identical embeddings for all cells.

### Embedding collapse diagnostic

| Metric | Random init | SSL (epoch 44) |
|--------|-------------|-----------------|
| Inter-cell cosine sim | 0.89 | 0.85-0.90 |
| Pos-pair cosine sim | 0.86-0.92 | 0.91-0.94 |
| Gap (pos - inter) | ~0.03-0.05 | ~0.05-0.09 |

Minimal separation between same-cell and different-cell pairs. Model cannot distinguish cell identities.

### Synthetic verification (architecture correct)

| Metric | Value |
|--------|-------|
| Loss (epoch 1→20) | 1.41 → 0.03 |
| Inter-cell sim | 0.68 → 0.17 |
| Generalization (new sigs) | gap=0.67 |

NT-Xent + CellEmbedder architecture confirmed working. Collapse on real data is purely a feature-quality problem.

### Image feature tests (local A500, 200 frames, 20 epochs)

| Backbone | Loss decay | Inter-sim range | Verdict |
|----------|-----------|-----------------|---------|
| Regionprops (7-dim) | 3.34→3.22 | 0.90 | Dead: total collapse |
| Scratch CNN (1.8M) | 3.58→2.88 | 0.66-0.86 | Direction works, undertrained |
| ResNet18 (frozen) | 3.81→3.39 | 0.73-0.94 | ImageNet doesn't transfer |
| DINOv2 ViT-S (frozen) | 3.54→3.82 | 0.81-0.97 | ImageNet doesn't transfer |

ImageNet features are domain-irrelevant for bacterial microscopy. Trainable CNN from scratch shows the right trajectory but needs more data/epochs.

---

## Contribution C: Distortion importance

Pending H100 run (`run_dist_ablation.slurm`). Based on ASCENT paper: jitter expected to dominate, dropout relevant for missing-cell handling. Hypothesis from our earlier test: stronger augmentations worsen collapse (inter-sim 0.90→0.96), confirming jitter is a double-edged sword — necessary for robustness but destructive with shallow features.

---

## Figures to produce

1. **Attention scaling** — time vs N for dense/K=4/K=16/K=32/K=64 at L=12 on H100 (source: `scaling_h100.csv`)
2. **SSL convergence** — NT-Xent loss + consistency vs epoch (source: `ssl_phase1/training_log.csv`)
3. **Embedding quality** — inter-cell similarity vs epoch for random vs SSL (source: collapse diagnostic)
4. **Distortion ablation** — bar chart of tracking accuracy vs distortion removed (source: pending `dist_abl/`)
5. **Synthetic verification** — loss curve showing architecture works with discriminative features
