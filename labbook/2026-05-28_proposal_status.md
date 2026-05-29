# Proposal Restatement — Status, Tradeoffs, and Implementation State

**Date:** 2026-05-28  
**Purpose:** Structured restatement of the CMS-PRO proposal [1] with honest completion status per section, literature references, explicit tradeoffs, and optional branches. Acts as ground-truth reference for remaining work.

---

## Project Overview

| Field | Value |
|-------|-------|
| Title | Efficient sparse attention and self-supervised pretraining for transformer-based cell tracking |
| Program | CMS-PRO (Computational Modeling and Simulation, Research Project) |
| Duration | 1 semester (summer), ~1-2 days/week |
| Deliverables | Written report + final presentation |
| Supervisors | Ziwen Liu, Martin Weigert (TU Dresden, Weigert Lab) |

---

## Research Question

> Can we make transformer-based cell tracking more efficient and more data-efficient, by **(a)** exploiting the sparsity of the spatial-cutoff attention pattern and **(b)** pretraining the association head from unlabeled data via geometric distortion?

Sub-questions:

1. How much throughput/memory can be recovered by replacing dense masked SDPA with gather-based sparse attention over the cutoff graph? Does token reordering for memory locality yield further gains?
2. Can we construct SSL association labels for free by geometrically distorting single frames (identity correspondence through the warp)?
3. Which families of distortions produce the most transferable representations?
4. Does SSL pretraining help more in the low-label regime than in the full-label regime?

---

## Baselines

- **Primary:** Trackastra [2] without pretraining, fully supervised on labeled tracking data
- **Datasets:** vanvliet (bacteria) [3], deepcell

---

## §4.1 — Efficient Sparse Attention

### Proposed method

Trackastra uses `F.scaled_dot_product_attention` with an explicit additive `attn_mask` encoding a spatial-distance cutoff [2]. Passing any `attn_mask` disables the FlashAttention backend [4] in PyTorch and materializes the full N×N score matrix, making a structurally O(Nk) problem run in O(N²) time and memory.

**Approach:** Replace dense masked SDPA with gather-based sparse attention:
- Precompute `idx ∈ Z^{N×k}` from the cutoff graph (KNN)
- Gather `K_sel = K[idx]`, `V_sel = V[idx]`
- Run SDPA on (N, k) tensors without a mask → enables FlashAttention-2 [5]
- Scales as O(Nkd)

**Token reordering variant** [6]: reorder tokens so gather loads hit contiguous memory across layers.

### Current state: DONE ✓

| Task | Status | Date | Notes |
|------|--------|------|-------|
| Standalone benchmark (dense vs KNN gather) | **Done** | May 19 | H100 scaling at N=128…8192, L=1/6/12. Results: `runs/phase1/scaling_h100.csv` |
| KNN throughput advantage | **Done** | May 19 | K=4: 1.6× faster at N=2048 L=12. Faster at ALL N. FA-2 enabled since no `attn_mask`. |
| Port to Trackastra | **Done** | May 26 | `GatherSparseAttention` in `trackastra/model/model_parts.py`. `knn_neighbors` config parameter. |
| Dense baseline reproduced | **Done** | May 26 | 234 epochs, best val_loss=0.002 (epoch 151). Run time: 6h51m. Job 3576409. |
| KNN K=4/16/32 training | **Done** (running) | May 26 | Jobs 3576410-3576412. Dense still best (0.002), K=16 approaching (0.0023), K=32 approaching (0.0021), K=4 worse (0.016). Naming conflict noted (K4/K16 share run dir `sparse_k16`). |
| Edge accuracy evaluation | **Done** | May 27 | K=16 (0.9981), K=32 (0.9992) > dense (0.9769). Evidence: KNN acts as structural regularizer. |
| Token reordering | **Dropped** | May 12 | <3% throughput benefit after overhead. Documented in `labbook/2026-05-12_token_reordering_analysis.md`. |

### Remaining for §4.1

- [ ] TRA/AOGM evaluation on all K variants (needs `traccuracy`) — **priority: high**
- [ ] Convergence curves: val_loss over epochs for dense vs K=4/16/32
- [ ] Throughput/memory plots across K (already have raw data, needs figure)

### Tradeoffs

| Decision | Why | Risk |
|----------|-----|------|
| KNN K=4 too sparse | 0.016 val_loss — cells need more context | Underperforms, may need K≥8 |
| KNN K=16/32 near dense | 0.0023/0.0021 vs 0.002 — close to parity | Dense still slightly better; need TRA to confirm |
| Token reordering dropped | <3% gain, added complexity | Correct decision per benchmarks |

---

## §4.2 — SSL Pretraining via Geometric Distortion

### Proposed method [1, §4.2]

> Given a single frame with segmentations {s_i}, apply a random transformation T to produce a synthetic "next frame" with segmentations {T(s_i)}. The ground-truth association is then the identity i → i (modulo dropped/occluded cells). **Train Trackastra's association head** on these synthetic pairs.

**Key design, verbatim from proposal:**
- Assemble a multi-frame window: [original frame, distorted frame(s)]
- Ground truth: identity association matrix (cell i → cell i across frames, minus dropped cells)
- Train the **full** TrackingTransformer (encoder + decoder + association head) — same model as downstream [2]
- Loss: BCE on association matrix (same loss as supervised training)
- Distortion families: affine, elastic deformations, per-cell jitter, detection dropout, photometric variations

**Distinction from supervised augmentations** [1, lines 71-78]: Trackastra already applies geometric augmentations during supervised training (RandomAffine, RandomTemporalAffine). The SSL difference is **where supervision comes from**: existing training uses real tracking labels; SSL derives identity labels from the warp itself (i → T(i)), requiring **no annotations**.

### Literature context

| Work | Relationship | Key difference |
|------|-------------|----------------|
| ASCENT [7] | Also uses geometric distortions of single frames for self-supervised learning | ASCENT: contrastive NT-Xent on NETr encoder → direct Hungarian tracking. Proposal: identity BCE on full Trackastra → fine-tune decoder. ASCENT relies on rich image features (64×64×3 patches → ChannelViT); proposal uses shallow regionprops (7-dim). |
| DynaCLR [8] | Contrastive learning of cellular dynamics with temporal regularization | Uses real temporal sequences, not synthetic distortions. More aligned with tracking dynamics but requires annotations. |
| Lalit & Funke 2025 [9] | Unsupervised cell tracking and interactive fine-tuning | Explores zero-shot vs few-shot tracking regimes; different pretext tasks. |

### Current state: PARTIAL — architecture misaligned ✓✗

#### What was built

| Component | Status | Implementation | File |
|-----------|--------|---------------|------|
| DistortionPipeline (6 types) | **Done** | Affine, elastic, jitter, dropout, photometric, feature_noise | `benchmark_ssl/distortions.py`, `trackastra/data/ssl_distortions.py` |
| Two augmented views per frame | **Done** | Shared-RNG dropout, label-sorted collation | `ssl_pipeline.py`, `ssl_trainer.py` |
| NT-Xent contrastive loss | **Done** | Per-frame InfoNCE, τ=0.05, unit-sphere normalization | `train_ssl.py`, `ssl_trainer.py` |
| Standalone CellEmbedder SSL | **Done** | 4-layer TransformerEncoder, converges (no_dropout: val 0.23, cons 0.98) | `benchmark_ssl/track_encoder.py` |
| Trackastra-integrated SSL | **Done** | `model.encode()` → NT-Xent, converges instantly (loss~0 in 2 ep) | `trackastra/model/ssl_trainer.py` |
| Distortion ablation (7 variants) | **Done** (buggy), **queued** (fixed) | Job 3578124 pending. Buggy results: only `no_dropout` converged. | SLURM queue |

#### What is WRONG (critical)

The implementation builds a **contrastive SSL pipeline** (NT-Xent loss on `model.encode()`), NOT the proposed **identity BCE pipeline** (BCE loss on `model.forward()`). This is a fundamental mismatch:

| | Proposal §4.2 | Current implementation |
|---|---|---|
| Model trained | Full TrackingTransformer (encoder + decoder + heads) | Encoder only (`encode()`, no decoder) |
| Loss | BCE on association matrix | NT-Xent (InfoNCE) on embeddings |
| Window structure | Multi-frame (original + distorted treated as separate timepoints) | Single frame, two views at same timepoint |
| Transfer | Same model, fine-tune all components | Encoder weight init, decoder random (~50% of params untrained) |
| Dropout | "modulo dropped cells" (cells may appear in one frame only) | Identical dropout in both views (no missing-cell challenge) |

#### Known fundamental problems

1. **Feature poverty** [documented May 24, `labbook/2026-05-24_ssl_collapse_findings.md`]: 7-dim regionprops have inter-cell cosine sim ~0.89. ASCENT uses 64×64×3 image patches → 256-dim visual features. Without richer features, SSL signal is fundamentally starved.

2. **Loss → 0 trivially**: 7-dim input → 320-dim encoder → NT-Xent → 0 in 2 epochs. Massive overparameterization. Model memorizes, doesn't learn. A healthy SSL should plateau at non-zero irreducible loss.

3. **Coordinate shortcut**: Fourier positional encoding (96-dim) dominates feature encoding (56-dim). Model matches cells across views by spatial proximity, not cell identity. Downstream: cells move between frames → spatial shortcut actively harmful.

4. **Distortions ≠ real motion**: Affine rotation, elastic deformation, per-cell jitter are random geometric transforms. None simulate real biological motion (persistent direction, Brownian motion, cell growth, division). Model learns to invert synthetic transforms, not to track cells.

5. **Decoder untrained**: Even if encoder learned useful representations, the decoder (cross-attention, head_y, association outer product) starts from random init. These components are what actually produce tracking associations.

6. **Image feature negative result** [May 24, `labbook/2026-05-24_imgfeat_negative_result.md`]: Frozen ResNet18/DINOv2 ViT on 64×64 bacterial patches — loss flat, no separation (ImageNet domain irrelevant). Scratch CNN shows direction (loss 3.58→2.88) but undertrained. ASCENT's success with ChannelViT depends on C. elegans neurons looking sufficiently like natural image patches.

#### SSL label-sweep: zero improvement [May 23-24]

Standalone CellEmbedder + frozen SSL embeddings → Hungarian tracking:

| Method | Accuracy |
|--------|----------|
| Dense (random init) | 64.3% |
| KNN (random init, avg) | 64.8% |
| KNN + SSL (frozen) | 65.1% |

SSL provides **no statistically meaningful improvement** over random embeddings for downstream tracking. Corroborates the feature-poverty collapse diagnosis.

### Tradeoffs — Path forward for §4.2

#### Option A: Continue contrastive SSL (current path) — **NOT RECOMMENDED**

**What:** Fix convergence issues, add image features, run downstream fine-tuning.
**Risk:** Architectural misalignment (encoder-only pretraining, decoder starts from scratch). Feature poverty unsolved. Evidence already negative (zero improvement in label-sweep). Would require image patches + trainable encoder → ASCENT-level compute.
**Verdict:** Whac-a-mole. Fixes symptoms, not root cause.

#### Option B: Identity BCE on full model (proposal's original design) — **RECOMMENDED as primary**

**What:** Implement as described in §4.2:
1. Single frame → distortion → 2-frame window [frame 0, frame 1]
2. Train full `TrackingTransformer.forward()` — encoder + decoder + association head
3. BCE loss on identity association matrix
4. Fine-tune on real data (§4.3)

**Pros:**
- Matches proposal verbatim
- Trains decoder + heads (currently 0% trained)
- Same loss as downstream → direct weight transfer
- Leverages existing Trackastra training infrastructure
- Testable in 1-2 days of implementation

**Cons:**
- Still suffers "distortions ≠ real motion" problem
- May confuse model (learns synthetic transform inversion, not biological tracking)
- No guarantee of convergence benefit

**Verdict:** Correct architecture/loss alignment per proposal. Motion-distortion gap is real but secondary — this path tests the proposal AS WRITTEN. If it fails, that's a valid scientific result (the proposal's core hypothesis was wrong). If it succeeds, that's a contribution.

#### Option C: Cross-frame SSL (conceptually better, logistically harder) — **STRETCH GOAL**

**What:** Use real consecutive frames from unlabeled videos. Pretext task from simple overlap heuristics or motion priors. Trains model on real cell motion.

**Pros:** Addresses "distortions ≠ real motion." Uses real temporal dynamics.
**Cons:** Requires unlabeled video data, not just single frames. Pseudo-label quality depends on heuristics. Deeper implementation. Outside proposal scope.
**Verdict:** Better science, less faithful to proposal. Keep as discussion topic.

### Remaining for §4.2

- [ ] Implement identity BCE on full TrackingTransformer (Option B) — **priority: critical**
- [ ] Run 20-epoch SSL → fine-tune pipeline
- [ ] Compare convergence: SSL init vs random init at matched downstream epochs
- [ ] Distortion ablation: which families transfer best under identity BCE?
- [ ] Label-fraction sweep: does SSL help more at 1%, 10%, 50%, 100% labels?
- [ ] Wait for `dist_abl2` job 3578124 (queued) — provides contrastive distortion ablation results even if path changes

---

## §4.3 — Fine-Tuning

### Proposed method

Fine-tune the pretrained model on real annotated tracking data. Compare to from-scratch Trackastra trained on the same labeled data [1].

### Current state: PARTIAL — baseline done, SSL not integrated ✗

| Task | Status | Date | Notes |
|------|--------|------|-------|
| Dense baseline | **Done** | May 26 | val_loss=0.002, epoch 151. Job 3576409. |
| K=4 KNN training | **Done** (running) | May 26 | Job 3576410. val_loss=0.016 — too sparse. |
| K=16 KNN training | **Done** (running) | May 26 | Job 3576411. val_loss=0.0023 — near dense. |
| K=32 KNN training | **Done** (running) | May 26 | Job 3576412. val_loss=0.0021 — near dense. |
| SSL-pretrained fine-tuning | **Not started** | — | Depends on §4.2 resolution. |
| Low-label experiments | **Not started** | — | Depends on §4.2 fine-tuning. |
| TRA/AOGM evaluation | **Not started** | — | `traccuracy` installation needed. |
| Deepcell dataset | **Not started** | — | vanvliet only so far. |

### Remaining for §4.3

- [ ] Install `traccuracy` for TRA/AOGM metrics
- [ ] Run tracking evaluation on dense baseline + K=4/16/32 models (once jobs complete)
- [ ] SSL-pretrained fine-tuning (once §4.2 identity BCE is implemented)
- [ ] Convergence comparison plots
- [ ] Low-label regime experiments

---

## §4.4 — Feature Ablations (Optional, per proposal)

### Proposed candidates [1]

- Shape descriptors: eccentricity, perimeter, Hu moments
- Local image crops around each cell with small CNN encoder

### Current state: EXPLORED with negative result

| Task | Status | Date | Notes |
|------|--------|------|-------|
| ImageNet backbones (frozen) | **Tested** — negative | May 24 | ResNet18, DINOv2 ViT: loss flat, no separation. Domain gap too large (ImageNet → bacteria). |
| Scratch CNN (32×32 patches) | **Tested** — negative | May 24 | Shows direction (loss 3.58→2.88) but undertrained. 32×32 too small for bacteria morphology. |
| Shape descriptors | **Not tested** | — | Eccentricity, perimeter, Hu moments never implemented. |
| 64×64 patches + deeper CNN | **Not tested** | — | ASCENT-level patch size. Would require ~days of training. |

### Verdict

Image features are the correct direction per ASCENT [7] but substantially increase complexity. Proposal marks §4.4 optional. Recommendation: **skip for now**, note as future work. SSL with regionprops is sufficient to test the core hypothesis. If SSL shows any benefit on regionprops, image features can amplify it later.

---

## Datasets

| Dataset | Status | Notes |
|---------|--------|-------|
| vanvliet (bacteria, 6 conditions) | **Active** | rpsM, recA, pheA, metA, cib, trpL. All benchmarks + Trackastra training. |
| deepcell | **Not started** | Expected for generalization evaluation. |

---

## Evaluation Plan

| Metric | Tool | Status |
|--------|------|--------|
| val_loss (BCE) | Built-in (Trackastra training loop) | Done for dense, K=4/16/32 |
| Edge accuracy | `eval_model.py` or custom | Done for KNN variants (May 27) |
| TRA (Tracking Accuracy) | `traccuracy` [10] | Not installed |
| cHOTA / AOGM | `traccuracy` [10] | Not installed |
| Throughput (steps/sec) | Custom benchmark | Done for scaling, needs figure |
| Convergence curves | Training logs | Data exists, needs plotting |

---

## Literature References

[1] This proposal (CMS-PRO, 2026).  
[2] Gallusser, B. & Weigert, M. (2024). Trackastra: Transformer-based cell tracking for live-cell microscopy. *ECCV 2024*.  
[3] Van Vliet, S. et al. (2022). Spatiotemporally resolved protein synthesis in single cells. *Nature*.  
[4] Dao, T. et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS 2022*.  
[5] Dao, T. (2024). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. *ICLR 2024*.  
[6] Hassani, A. et al. (2024). Faster Neighborhood Attention: Reducing the O(n²) Cost of Self Attention at the Threadblock Level.  
[7] Han, L. & Lu, Z. (2025). ASCENT: Annotation-free Self-supervised Contrastive Embeddings for 3D Neuron Tracking in Fluorescence Microscopy. *bioRxiv*.  
[8] Hirata-Miyasaki, E. et al. (2025). DynaCLR: Contrastive Learning of Cellular Dynamics with Temporal Regularization.  
[9] Lalit, M. & Funke, J. (2025). An Investigation of Unsupervised Cell Tracking and Interactive Fine-Tuning.  
[10] Bragantini, J. et al. (2025). Ultrack: pushing the limits of cell tracking across biological scales. *Nature Methods*.  
[11] Maška, M. et al. (2023). The cell tracking challenge: 10 years of objective benchmarking. *Nature Methods*.

---

## Milestone Status

| Milestone | Status | Completed | Notes |
|-----------|--------|-----------|-------|
| Literature review, cluster setup | Done | Week 1 | Capella HPC, Trackastra env, data paths |
| Benchmark dense vs sparse+gather attention | Done | Week 3 (May 19) | H100 scaling, KNN attention in Trackastra |
| Reproduce Trackastra baseline | Done | Week 6 (May 26) | dense baseline 0.002 val_loss, K4/16/32 running |
| Implement geometric-distortion SSL pipeline | **Wrong** | Week 4-7 | Built contrastive NT-Xent, not proposed identity BCE |
| Pretraining + fine-tuning experiments | **Not started** | — | Depends on SSL pipeline fix |
| Distortion ablations | Partial | Week 5-6 | Buggy run done, fixed run queued (3578124) |
| Feature ablations (optional) | Explored | Week 5 (May 24) | Image features negative, shape descriptors untested |
| Low-label regime study | **Not started** | — | Depends on fine-tuning |
| Final experiments, figures, writeup | **Not started** | — | — |

---

## Current Priority Order

1. **Fix SSL pipeline** — implement identity BCE on full model (proposal §4.2 as written)
2. **Complete KNN evaluation** — wait for K=4/16/32 jobs, run TRA/AOGM
3. **SSL pretraining + fine-tuning experiment** — does identity BCE pretraining help convergence?
4. **Distortion ablation** — under identity BCE framework
5. **Low-label sweep** — if SSL shows any benefit
6. **Figures and writeup**
