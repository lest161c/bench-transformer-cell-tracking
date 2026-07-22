# Why Contrastive SSL Fails on Trackastra — Architectural Analysis

**Date:** 2026-06-08
**Context:** Explain why contrastive NT-Xent pretraining on Trackastra performs worse than random init, and why full-model contrastive learning is architecturally infeasible.
**Depends on:** `2026-05-24_ssl_collapse_findings.md`, `2026-05-29_ssl_memorization_analysis.md`, `2026-06-02_ssl_post_mortem_and_way_forward.md`

---

## 1. Why Full-Model Contrastive Learning Cannot Work

### 1.1 Architectural Incompatibility

Contrastive learning (NT-Xent / InfoNCE) requires a **per-sample embedding** \(\mathbf{z}_i \in \mathbb{R}^d\):

\[
\mathcal{L}_{\text{NT-Xent}} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_i^+)/\tau)}{\sum_{j} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau)}
\]

Trackastra's full forward pass produces an **association matrix**:

\[
\hat{A} = \text{MLP}_Y(\mathbf{Y}) \cdot \text{MLP}_Z(\mathbf{Z})^T \in \mathbb{R}^{N \times N}
\]

where \(\mathbf{Y} = f_{\text{enc}}(\mathbf{X})\) and \(\mathbf{Z} = g_{\text{dec}}(\mathbf{X}, \mathbf{Y})\) are functions of cross-attention between all cells.

There is **no point in the full forward pass** where a single cell gets a 1D embedding suitable for contrastive comparison. The decoder's output \(\mathbf{Z}\) is a set of N per-cell vectors, but each \(\mathbf{z}_i\) encodes cell i's relationship to ALL other cells via cross-attention — it's not a self-contained identity embedding.

| Component | Output shape | Suitable for NT-Xent? |
|---|---|---|
| `model.encode()` | \((B,N,d)\) per-cell features after self-attention | **Yes** — used in current (broken) pipeline |
| `model.forward()` | \((B,N,N)\) association logits | **No** — pairwise scores, not per-sample |
| `MLP_Y(encode())` | \((B,N,d/2)\) | **Maybe** — but trained for outer product, not contrastive loss |
| Decoder cross-attn output | \((B,N,d)\) context-aware | **No** — encodes pair-wise context, not cell identity |

The only component that can be contrastively trained is the encoder. The decoder + association heads are structurally incompatible with per-sample contrastive objectives.

### 1.2 Why Encoder-Only Pretraining Hurts

Training only `model.encode()` with NT-Xent creates a **representational mismatch**:

```
SSL:    encode(cell_i)               → 1D embedding → contrastive loss (per-sample)
Track:  encode(cell_i) → decoder → cross-attention → association scores (per-pair)
```

The encoder learns to produce embeddings that are **cosine-separable** (good for contrastive ranking). The decoder expects embeddings that are **cross-attendable** (good for estimating pairwise associations). These are different geometric properties of the latent space. A contrastively-optimal embedding may place cells in a configuration the randomly-initialized decoder finds harder to process than random noise.

Evidence: downstream tracking accuracy with SSL-pretrained encoder + random decoder = 65.1%. Random encoder + random decoder = 64.8% avg. Difference: 0.3% — statistically zero. The pretrained encoder provides **no usable information** to the decoder.

---

## 2. Why It Performs Worse Than Random

Three independent mechanisms converge on the same outcome:

### 2.1 Feature Poverty (7-dim Regionprops)

Trackastra uses 7 hand-crafted per-cell features:
```
[area, perimeter, eccentricity, solidity, extent, mean_intensity, std_intensity]
```

Two bacteria in the same colony have **near-identical** values for all 7. Raw inter-cell cosine similarity: ~0.89. The contrastive loss is tasked with discriminating between points that are essentially indistinguishable by input:

```
Cell A: [1.82, 2.24, 0.96, 0.88, 0.72, 0.45, 0.12]
Cell B: [1.79, 2.19, 0.95, 0.87, 0.71, 0.44, 0.11]
        ↑ input difference: ~2%  →  encoder must amplify to meaningful separation
```

The encoder expands 7 → 320 dimensions. But expansion from a low-variance input just spreads noise across more dimensions — it doesn't create information. A random embedding (320-dim Gaussian) has more useful variance than a trained embedding of identical inputs.

### 2.2 Coordinate Shortcut (Position → Identity)

The encoder concatenates positional encoding with feature encoding:

```
pos_embed(coords):  96-dim  (Fourier encoding of t,y,x)
feat_embed(features): 56-dim (linear projection of 7-dim regionprops)
─────────────────────────────────
total input to encoder: 152-dim  (63% position, 37% features)
```

Under mild distortions (jitter std=2-8), coordinates barely change: `T(pos) ≈ pos`. The contrastive objective is solved by:

```
sim(encode(cell_i, pos_i), encode(cell_i, T(pos_i))) ≈ 1
```

The encoder learns: "if positions are similar → embeddings should be similar." Features contribute nothing. During downstream tracking:

```
sim(encode(cell_t, pos_t), encode(cell_{t+1}, pos_{t+1}))
```

Cells MOVE between frames. `pos_t ≠ pos_{t+1}`. The encoder produces **different embeddings for the same cell at different timepoints** — the exact opposite of what tracking requires. The positional shortcut that minimized SSL loss actively degrades tracking performance.

### 2.3 The Decoder Gap (~50% of Weights Untrained)

Trackastra's architecture by component:

| Component | # Params (est.) | % Total | Trained in SSL? |
|---|---|---|---|
| pos_embed | 2K | 2% | Yes (but learned wrong thing) |
| feat_embed + input proj | 20K | 8% | Yes |
| Encoder (L=6 self-attn + MLP) | 1.2M | 45% | Yes |
| Decoder (L=6 cross-attn + MLP) | 1.2M | 45% | **No** — starts random |
| head_y (MLP before outer product) | 50K | <1% | **No** |
| head_z (MLP before outer product) | 50K | <1% | **No** |

**~50% of the model starts from random init** during fine-tuning. The encoder has been optimized for a different objective (cosine similarity in embedding space) under a different data distribution (geometric distortions, no cell division). The decoder must both (a) learn to interpret the encoder's output from scratch, and (b) learn the tracking task from scratch — simultaneously.

This is **worse** than training from scratch with random init, because:
- From scratch: encoder and decoder co-adapt. The encoder's weights are free to move to whatever representation the decoder needs.
- With frozen encoder: decoder must adapt to a fixed, potentially unhelpful representation.
- With fine-tuned encoder: encoder must unlearn SSL task before learning tracking task → gradient interference.

### 2.4 Combined Effect

```
SSL objective:    "Match cell_i at pos_i to cell_i at T(pos_i)"
                  → encoder learns: pos_i ≈ pos_j ⇒ encode(cell_i) ≈ encode(cell_j)

Real tracking:    "Match cell_i at pos_t to cell_i at pos_{t+1}"  
                  → encoder should learn: identity(cell_i) ≈ identity(cell_i) regardless of position

SSL → tracking:   NEGATIVE TRANSFER. The SSL-trained encoder produces actively misleading
                  information. The model would perform BETTER with an untrained encoder
                  because the untrained encoder provides random noise (no systematic bias),
                  while the SSL encoder provides systematically wrong position-based embeddings.
```

This is why TRA is 65.1% (SSL) vs 64.8% (random) — effectively identical. The SSL encoder provides noise indistinguishable from random for the downstream task.

---

## 3. What Identity BCE Fixes (and What It Doesn't)

The proposal's original §4.2 design uses **identity BCE** on the full model, not contrastive NT-Xent on the encoder:

```
SSL (BCE):    model.forward(frame0, frame1) → A_ij  where A = I (identity matrix)
BCE loss:     BCE(A_hat, A_identity, weight=W)
```

**What this fixes:** The decoder + association heads are trained during SSL (no untrained weights). The loss is the same as downstream tracking (no loss function gap). The full model is optimized end-to-end (no representational mismatch).

**What this doesn't fix:**
- **Feature poverty** — still 7-dim regionprops, still near-identical inputs for different cells. Identity BCE asks the model to discriminate between cells in the same frame — impossible if their features are identical. Result: model collapses to predicting all associations ≈ 0 (or memorizes position-based shortcuts, see below).
- **Coordinate shortcut** — identity BCE with real coordinates is solvable by learning the distortion inverse function T⁻¹. `A_hat[i,i] = score(pos_i, T(pos_i))` → model learns `score(pos, p) = exp(-||p - T(pos)||²)`. Features irrelevant. Downstream: `score(pos_t, pos_{t+1})` — learns to score cells based on proximity, which is just Euclidean-distance tracking (the trivial baseline, TRA ~94%, Trackastra achieves 99.9%).
- **Distortions ≠ biological motion** — affine rotation, elastic warping, per-cell jitter are random geometric transforms. Real cells exhibit persistent Brownian motion, division dynamics, and colony growth. The model learns to invert synthetic geometric transforms, not to track cells through biological motion.

---

## 4. The Coordinate-Free Fix (Only Hope for SSL)

The coordinate shortcut is the primary failure mode for both contrastive and BCE pretraining. The proposed fix is to **randomize or remove coordinates during SSL**:

```python
# During SSL pretraining only:
rand_coords = coords[:, torch.randperm(N), :]  # shuffle positions across cells
x = model(features=features, coords=rand_coords)
```

This forces the model to learn feature-based cell identity. Each cell gets a random position → the encoder must use regionprops to distinguish cells. The pos_embed fires but carries no information about which cell is which — it becomes a learnable perturbation, not a shortcut.

**Downside during fine-tuning:** The pos_embed weights were perturbed during SSL (they adapted to random positions). During fine-tuning with real coordinates, these weights must be re-aligned. This takes ~1-2 epochs of fine-tuning (the pos_embed is only 2K params, very fast to adapt).

**Open question:** Even with coordinate-free pretraining, can the 7-dim regionprops provide enough signal for identity BCE? The feature-poverty analysis suggests no — all cells look the same in regionprots. The minimum viable fix requires **image features** (CNN on 64×64 cell crops) AND coordinate-free pretraining AND identity BCE on the full model. That's a 2-3 week implementation commitment with no guarantee of success.

---

## 5. Summary

| Problem | Contrastive NT-Xent | Identity BCE | Identity BCE + coord-free |
|---|---|---|---|
| Full model trained? | No (encoder only) | Yes | Yes |
| Loss matches downstream? | No (contrastive vs BCE) | Yes | Yes |
| Coordinate shortcut? | Yes (dominates) | Yes (can learn T⁻¹) | **Partially fixed** (pos_embed sees noise) |
| Feature poverty? | Yes (7-dim, all cells identical) | Yes | **Unresolved** — needs image features |
| Distortions ≠ real motion? | Yes | Yes | Yes |
| TRA vs random (actual) | 65.1% vs 64.8% | 0.404 val_loss vs 0.416 (no improvement) | Not tested |

The fundamental problem is not the loss function or the pretraining method — it's that **7-dim regionprops contain no per-cell identity information** for discriminating bacteria in a dense colony. Every SSL method on these features ultimately trains a model to do what Euclidean distance already does: match cells by proximity.
