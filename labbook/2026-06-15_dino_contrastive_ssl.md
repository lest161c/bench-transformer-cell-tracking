# DINO + Contrastive SSL: Argumentation for Full-Scale Training

**Date:** 2026-06-15
**Author:** Auto-generated
**Status:** Ready for HPC verification

## 1. Prior: Why regionprops contrastive SSL failed

Two prior SSL attempts on vanvliet both failed:

| Attempt | Method | Result |
|---------|--------|--------|
| NT-Xent (ssl_trainer.py) | Contrastive on 7D regionprops | Collapse: inter-cell cosine sim = 0.89 |
| Identity BCE (ssl_pretrain.py) | BCE on full model | No improvement over from-scratch |

Signal analysis of 7D regionprops (`benchmark_ssl/analyze_signal.py`):
- Separation gap (intra - inter): **0.047** — essentially noise
- Effective dimensionality: **3** — only 3/7 dims carry signal
- Recall@1 (feature-only matching): **0.215** — near random
- Adding shape descriptors (+7D), Hu moments (+7D), local patches (+16D PCA) did NOT improve gap. All gave gap ≈ 0.047, effective dim = 3.
- Conclusion: **Mask-derived features are fundamentally insufficient** for contrastive SSL on bacterial data, regardless of feature engineering.

## 2. Hypothesis: DINOv2 patch features provide sufficient visual signal

ASCENT (Han & Lu 2025) succeeds on C. elegans neuron tracking using:
- 64×64×3 image patches → DINO-pretrained ChannelViT → 256D visual features
- NT-Xent contrastive loss on encoder outputs
- Their architecture ablation showed visual-only model (no positional encoding) achieves near-peak accuracy

Key difference from our regionprops: DINOv2 produces **384D learned visual features** capturing pixel-level patterns (texture, local background, intensity gradients) that distinguish cells even within the same frame.

## 3. Evidence: Local A500 tests

### Test A: DINO feature separation (no distortion)
- DINO embeddings of different cells in same frame have gap = **0.292**
- Regionprops equivalent: **0.047** (6.2x improvement)
- Effective dimensionality: **108** (vs 3 for regionprops → 36x improvement)
- *Significance:* DINO naturally separates cells — contrastive SSL has a strong foundation

### Test B: Micro-SSL generalization (train/val split)
- Trained 1-layer projection head (384→128→64) with NT-Xent on 20 frames
- Evaluated gap on **10 held-out frames**
- Val gap improved: **0.481 → 0.577 (+0.095)**
- *Significance:* **Genuine learning, not memorization.** The projection head learns invariances that transfer to unseen frames.

### Test C: Micro-SSL hard pipeline (affine + jitter + dropout)
- Full distortion pipeline collapses DINO features to gap = **0.040** (same as regionprops baseline)
- After 200 steps of micro-SSL: gap = **0.359 (+0.319)**
- Inter-cell similarity after training: **0.044** (no collapse)
- *Significance:* The hardest test — starting from near-zero signal, a 1-layer MLP learns distortion invariance in 200 steps. Full 6L transformer will be far more powerful.

### Test D: Gap vs distortion strength
| Distortion | Param | Gap |
|-----------|-------|-----|
| None | — | 0.292 |
| Jitter | 4px | 0.217 |
| Jitter | 8px | 0.214 |
| Jitter | 16px | 0.072 |
| Rotation | 10° | 0.137 |
| Rotation | 25° | 0.031 |

*Significance:* DINO is robust to small shifts (jitter ≤8px) but sensitive to rotation — exactly the invariance that SSL must learn.

## 4. Implementation plan

### Architecture
```
image (H,W) + centroids → extract 64×64 patches → DINOv2 (frozen) → 384D → Linear(384, d_model)
coords → FourierPE → proj to d_model
add(pos, dino_feats) → LayerNorm → encoder + decoder (untouched)
```

### What stays the same
- `TrackingTransformer` encoder+decoder architecture: **zero changes** to transformer layers
- `nt_xent_loss()`: **identical**
- Fine-tuning procedure: **identical** (DINO projection unused during SL)
- Downstream tracking: **identical**

### What changes
1. **New module:** `dino_encoder.py` — DINOv2 + projection head
2. **Model change:** `model.py` — accept `patches` kwarg in `forward()`/`encode()`
3. **New trainer:** `ssl_dino_trainer.py` — distortion pipeline → patch extraction → DINO → NT-Xent
4. **Config + slurm:** for HPC submission

### Files changed/added
- `trackastra/trackastra/model/dino_encoder.py` (NEW)
- `trackastra/trackastra/model/model.py` (MODIFY)
- `trackastra/trackastra/model/ssl_dino_trainer.py` (NEW)
- `configs/ssl_dino_pretrain.yaml` (NEW)
- `benchmark_ssl/run_ssl_dino_pretrain.slurm` (NEW)
- `labbook/2026-06-15_dino_contrastive_ssl.md` (NEW)

## 5. Files (detailed)

### `trackastra/trackastra/model/dino_encoder.py`
- `DINOBackbone`: singleton DINOv2 model, frozen, forward pass with normalization
- `DINOProjection`: `DINOBackbone → Linear(384, hidden) → ReLU → Linear(hidden, d_model)`
- `extract_patches(img, centroids)`: utility returning `(N, 1, 64, 64)` tensor

### `trackastra/trackastra/model/model.py` changes
In `TrackingTransformer.__init__`:
- Add `use_dino: bool = False` parameter
- When `use_dino`: create `self.dino_proj = DINOProjection(d_model)`
- Add `pos_proj = nn.Linear(pos_embed_dim, d_model)` for ASCENT-style addition

In `forward()` and `encode()`:
- Accept `patches: Optional[torch.Tensor] = None` kwarg
- When `patches is not None`:
  - Run `dino_feats = self.dino_proj(patches)` → `(B, N, d_model)`
  - Project pos to d_model: `pos_proj = self.pos_proj(pos)`
  - `features = self.norm(pos_proj + dino_feats)` (ASCENT-style)
- Otherwise: existing regionprops path (unchanged)

### `ssl_dino_trainer.py`
Pattern: duplicate of `ssl_trainer.py` but:
- Uses `DistortionPipeline` to warp coordinates
- Extracts image patches at warped coords
- Passes patches to `model.encode(patches=patches)`
- Same NT-Xent loss
- Saves encoder weights for downstream fine-tuning

### Config
```yaml
use_dino: true
d_model: 320
ssl:
  epochs: 20
  lr: 3e-4
  temperature: 0.05
  batch_size: 4
distortions: [affine, jitter, dropout, elastic, photometric, feature_noise]
```

### Slurm
Based on `run_baseline.slurm`, 6h time limit (SSL only, no SL).

## 6. HPC verification

Submit via:
```bash
sbatch benchmark_ssl/run_ssl_dino_pretrain.slurm
```

Expected runtime: ~2-6h on H100.

### Success criteria (from SSL log)
- Training loss decreases monotonically (starting high, ending <0.5)
- Inter-cell cosine similarity stays < 0.3 (no collapse)
- Embedding consistency > 0.5 (same cell across views)
- After fine-tuning: faster convergence than from-scratch baseline (compare val_loss curves)

### If failure
- **Collapse (inter_sim > 0.8):** Lower temperature (0.01), increase projection head size, add stop-gradient
- **No learning (loss flat):** Low LR, increase distortion diversity, train more epochs
- **Poor fine-tuning transfer:** Fine-tune entire model (not just head), use lower fine-tuning LR

## 7. References
- Han & Lu 2025. ASCENT. CVF.
- Moutakanni et al. 2025. Cell-DINO. PLOS Comp Bio.
- Gallusser & Weigert 2025. Trackastra. ECCV.
