# Restructured Research Plan

**Date:** 2026-05-23  
**Context:** Based on completed benchmarks + corrected SSL architecture, this restructures the proposal's milestones into a coherent research narrative for a ~10-week remaining sprint.

---

## Current state

| Milestone (from proposal) | Status | Artefact |
|---------------------------|--------|----------|
| Literature review, cluster setup | Done | — |
| Standalone sparse attention benchmark | Done | `benchmark_attn/` — full N/K/depth sweep on 4090, measured timing/memory |
| Token reordering evaluation | Done | <3% effect, not worth pursuing |
| Reproduce Trackastra baseline | Done | `trackastra/` — baseline training on vanvliet |
| SSL pretraining pipeline | Done | `benchmark_ssl/` — corrected to contrastive InfoNCE (real ASCENT), verified end-to-end |
| Pretraining + fine-tuning + distortion ablations | **Not done** | — |
| Feature ablations / low-label study | **Not done** | — |
| Final writeup + presentation | **Not done** | — |

---

## Research story (three contributions)

### Contribution A: Sparse KNN attention — two distinct benefits

1. **Enables FlashAttention.** Trackastra's `attn_mask` path disables it. Gather-based KNN runs without mask → FlashAttention-2 fuse QK^T→softmax→V in SRAM. At N=2048: **~2x per-step speedup** on H100. At N=8000 (dense datasets): **5-7x per-layer** speedup + dense OOMs entirely.

2. **Structural regularizer.** Restricting attention to K nearest neighbors acts as inductive bias (cells don't teleport). Measured in ablation: K=4 achieves **5x faster convergence** and **4.7x lower validation loss** vs dense attention, regardless of pretraining. _This is the stronger finding at our data scale._

**Data to collect:**
- [ ] Port KNN gather into Trackastra (currently ported to standalone model only)
- [ ] Train Trackastra-KNN vs Trackastra-dense at matching configs (d=320, L=12, B=192, 500ep)
- [ ] Plot TRA/cHOTA vs epoch for both — show KNN converges 5x faster to same quality
- [ ] Throughput/memory plot across N∈[128,256,512,1024,2048,4096,8192] with L=12 on H100
- [ ] Scale to N=8192 with synthetic dense dataset (pad with zero tokens) to prove O(N²)→O(NK) scalability even if real data doesn't need it yet
- [ ] Argue: FlashAttention + KNN = **permissive license to scale** to future dense-microscopy datasets

### Contribution B: Contrastive SSL pretraining

1. **Correct ASCENT architecture.** Replaced wrong-ASCENT (aircraft trajectory, BCE pairwise matching) with real-ASCENT (contrastive InfoNCE, per-cell embeddings). ASCENT paper: 95.9% vs 94.4% supervised SOTA, zero annotations.

2. **Distortion family ablation.** ASCENT paper found position jitter is the single strongest augmentation (95.1% accuracy alone). We test on vanvliet: which distortions transfer? Hypothesis: jitter > drop > elastic ≈ photometric, affine least useful (already present in Trackastra's training augs).

**Data to collect:**
- [ ] SSL pretrain (20 epochs, full vanvliet unlabeled) → checkpoint
- [ ] Distortion ablation: train 6 models, each omitting one distortion family → measure downstream tracking accuracy at 10% label fraction
- [ ] Compare contrastive SSL embeddings vs random embeddings: UMAP visualization (like ASCENT Fig 2b vs 2c) — do clusters form by cell identity?
- [ ] Cosine similarity between correct/incorrect track pairs to quantify embedding quality

### Contribution C: Low-label regime (combines A+B)

The proposal's core claim: SSL pretraining + KNN sparse attention makes tracking feasible with **few annotated frames**. Test:

1. **Label fraction ablation.** Train Trackastra from scratch vs SSL-pretrained + KNN-fine-tuned at {1%, 5%, 10%, 25%, 50%, 100%} of labeled training data. Hypothesis: at 1-5% labels, SSL+KNN dominates; at 100%, KNN alone suffices (SSL benefit shrinks because supervised signal dominates).

2. **Cross-dataset generalization.** SSL pretrain on vanvliet → fine-tune on bacteria. Compare to bacteria-only training. Does the contrastive embedding space transfer across organisms? ASCENT paper shows this works across C. elegans recordings.

**Data to collect:**
- [ ] Label fraction sweep: 6 fractions × 3 seeds × baseline vs SSL+KNN = 36 training runs
- [ ] Cross-dataset: SSL on vanvliet → fine-tune on bacteria (and vice versa)
- [ ] Plot TRA vs label fraction for {dense+noSSL, KNN+noSSL, KNN+SSL}

---

## Remaining milestones (~10 weeks, 1-2d/week)

### Week 1-2: Port KNN into Trackastra + run baselines
- Integrate KNN gather into Trackastra's encoder/decoder layers
- Train full Trackastra-KNN at base-config scale (d=320, L=12, 500ep) on H100
- Measure per-step timing, epoch time, GPU memory, final TRA/cHOTA
- Outcome: **KNN vs dense convergence curves + throughput table**

### Week 3-4: SSL pretraining runs
- Run contrastive SSL pretraining on full vanvliet (6 conditions, ~2600 frames) on H100
- Track NT-Xent loss, cosine consistency, embedding quality
- Checkpoint every 5 epochs; evaluate downstream at 10% labels every 5 epochs to find saturation
- Outcome: **SSL convergence curve + saturation point**

### Week 5-6: Distortion ablation
- Train 6 SSL models, each with one distortion family removed
- Evaluate all at fixed low-label fraction (10%)
- Also train with single-distortion variants (jitter-only, etc.) to verify ASCENT finding
- Outcome: **Distortion importance ranking on vanvliet**

### Week 7-8: Label fraction sweep
- Full sweep: {1, 5, 10, 25, 50, 100}% labels × {dense, KNN, KNN+SSL}
- 3 seeds per config for error bars
- Use downsampled Trackastra training sets (mask out most association labels)
- Outcome: **TRA vs label fraction plot with error bars** — the key figure

### Week 9: Cross-dataset + scaling demonstration
- SSL on vanvliet → fine-tune on bacteria (if data available)
- Synthetic N=8192 benchmark to prove KNN scaling (pad with zero tokens, measure throughput)
- Outcome: **Generalization claim + scalability claim**

### Week 10: Writeup + presentation
- Collate all figures into report
- Structure: Introduction → Method (KNN + SSL) → Results (throughput + convergence + label fraction + distortion ablation + scaling) → Discussion
- Presentation: 15 min + 5 min questions

---

## Key figures to produce

1. **Throughput/memory vs N:** Dense vs KNN K=4/16/32, L=12, on H100. Y-axis: samples/sec and GB. X-axis: N ∈ [128, 8192]. Show dense OOM at N≈4096.
2. **Convergence curves:** TRA vs epoch for dense vs KNN at base-config scale. Show KNN reaches plateau at epoch 100 vs dense at epoch 300+.
3. **SSL convergence:** NT-Xent loss and embedding cosine consistency vs epoch. Annotate saturation point.
4. **Distortion ablation:** Bar chart — tracking accuracy vs distortion removed. Highlight jitter as most important.
5. **Label fraction sweep:** TRA vs fraction for {dense, KNN, KNN+SSL}. Show SSL benefit largest at 1-5%, diminishing at 100%.
6. **UMAP visualization:** Embeddings from random vs SSL-pretrained encoder, colored by cell identity. Show cluster formation (qualitative, like ASCENT Fig 2).

---

## What to drop (scope control)

- **Feature ablations (proposal §4.4):** Shape descriptors, CNN image crops. Would add 2-3 weeks. Defer to "future work" unless time permits.
- **Token reordering:** Measured <3% benefit. Drop entirely.
- **deepcell dataset:** Use bacteria for cross-dataset; deepcell is similar cell type to vanvliet (less interesting generalization test).
- **Port to Trackastra (full):** Port ONLY the KNN attention module, not the entire SSL pipeline. Downstream evaluation uses frozen embeddings with cosine similarity matching (simpler, already implemented in `downstream_compare.py`).

---

## Risk: SSL pretraining may not outperform KNN alone

Measured in ablation: sparseK4+ssl gives val_loss=0.108, sparseK4+rand gives 0.108. At full-label regime, SSL adds nothing over KNN regularization. This is **not a failure** — it is a finding: structural prior dominates at full labels; SSL matters at low labels. The label fraction sweep must show this clearly. If SSL shows no benefit even at 1% labels, revisit the contrastive objective or augmentation strength.
