# SSL Implementation Gap vs Proposal §4.2

## What §4.2 requires

From the proposal (verbatim):

> Given a single frame with segmentations {s_i}, apply a random transformation T to produce
> a synthetic "next frame" with segmentations {T(s_i)}. The ground-truth association is then the
> identity i → i (modulo dropped/occluded cells). Train Trackastra's association head on these
> synthetic pairs.

Key elements:

1. **Single-frame input** — no tracking labels needed, only segmentations
2. **Random geometric distortion T** — produce synthetic paired views
3. **Identity association** — same cell across views = positive, different cells = negative
4. **Train Trackastra directly** — same model as downstream, no transfer needed
5. **Rich distortion set**: affine, elastic, per-cell jitter, detection dropout, photometric
6. **Contrastive embedding learning** (per ASCENT citation) — not simple BCE identity matching

## What we currently have

### Approach A: benchmark_ssl (good)

- Model: `CellEmbedder` (separate 4-layer transformer encoder)
- Loss: NT-Xent contrastive (matches ASCENT)
- Distortions: Full `DistortionPipeline` with 6 types (affine, elastic, jitter, dropout, photometric, feature_noise)
- Two independently augmented views per frame
- Convergence proven after bug fixes: val_loss 0.05, consistency 1.0 at epoch 20
- **Fails §4.2:** Wrong model (CellEmbedder ≠ Trackastra). Needs weight transfer.

### Approach B: Trackastra `--ssl_pretrain` (bad)

- Model: `TrackingTransformer` directly (matches §4.2)
- Loss: BCE identity association on a single distorted view — NOT contrastive
- Distortions: `WRAugmentationPipeline` — only 5 simple geometric ops (flip, affine, offset, movement, brightness). NO dropout, NO elastic, NO jitter, NO feature_noise
- Only ONE distorted view created from original
- **Fails §4.2:** Wrong loss (BCE identity, not contrastive). Wrong distortions (too simple, risk memorization). Wrong pairing (single view, not two).

### Summary

| Requirement | benchmark_ssl | Trackastra SSL | Matches §4.2? |
|-------------|:--:|:--:|:--:|
| NT-Xent contrastive loss | ✓ ASCENT-style | ✗ BCE identity | benchmark_ssl |
| 6-distortion pipeline | ✓ Rich pipelin | ✗ 5 simple ops | benchmark_ssl |
| Shared-RNG dropout fix | ✓ Fixed | N/A (no dropout) | benchmark_ssl |
| Two augmented views | ✓ | ✗ Single view | benchmark_ssl |
| Train on Trackastra directly | ✗ CellEmbedder | ✓ TrackingTransformer | Trackastra |
| Zero-transfer to downstream | ✗ Needs mapping | ✓ Same model | Trackastra |
| Proven convergence (20ep) | ✓ val=0.05, cons=1.0 | ✗ Untested with rich distortions | benchmark_ssl |

**Neither implementation satisfies §4.2.** benchmark_ssl has the right learning signal but wrong model. Trackastra SSL has the right model but wrong learning signal.

## Rework TODO

### 1. SSL pretraining module (new file in trackastra)

- [ ] Create `trackastra/model/ssl_encoder.py`
- [ ] Takes `TrackingTransformer` as backbone
- [ ] Forward: run encoder only (stop before decoder), expose `(B,N,d_model)` embeddings
- [ ] Supports both dense and KNN modes (`knn_neighbors` config)
- [ ] No decoder involvement during SSL training
- [ ] Embedding head: optional projection layer for NT-Xent (or use raw (B,N,d_model))

### 2. Port fixed DistortionPipeline into Trackastra

- [ ] Copy `distortions.py` fixes into Trackastra (or import from benchmark_ssl)
- [ ] DistortionPipeline with 6 types: affine, elastic, jitter, dropout, photometric, feature_noise
- [ ] Shared-RNG dropout (both views get identical drops) — already fixed
- [ ] Correct label filtering with `keep` mask — already fixed
- [ ] All distortions return `(coords, features, labels)` — already fixed

### 3. SSL data pipeline for Trackastra

- [ ] Create `SSLDataset2` or extend `SSLPretrainDataset`
- [ ] For each single frame: extract WRFeatures → apply DistortionPipeline → two views
- [ ] Label-sort both views for positive-pair alignment
- [ ] Collation: pad both views to max N, produce view1, view2 batches
- [ ] Each batch: coords1, feats1, labels1, coords2, feats2, labels2
- [ ] Target association: labels1[i] == labels2[j] → positive pair

### 4. NT-Xent loss on Trackastra encoder

- [ ] Create training loop: for each frame pair
  1. View1 → encoder → x1 (B, N1, d_model)
  2. View2 → encoder → x2 (B, N2, d_model)
  3. Normalize x1, x2 to unit sphere
  4. Compute NT-Xent: cosine similarity / temperature → cross-entropy
- [ ] Per-frame InfoNCE (not batch-level, matches frame structure)
- [ ] Mask padding positions
- [ ] Log train/val loss, consistency, epoch time

### 5. SSL training script

- [ ] `scripts/train_ssl.py` or extend `train.py`
- [ ] CLI: `--ssl_epochs 20 --ssl_conditions rpsM,recA,... --knn_neighbors K`
- [ ] Save checkpoint: encoder weights + config
- [ ] Optional: `--ssl_only` flag for pure SSL eval
- [ ] TensorBoard logging (loss, consistency)

### 6. Distortion ablation (proposal §4.2)

- [ ] 7 variants: full, no_jitter, no_dropout, no_feature_noise, no_photometric, no_elastic, no_affine
- [ ] Each: 20 epochs SSL pretraining
- [ ] Compare SSL val_loss / consistency at matched epochs
- [ ] Identify which distortions transfer best (via downstream §4.3)

### 7. Downstream fine-tuning (proposal §4.3)

- [ ] Load SSL-pretrained encoder weights into TrackingTransformer
- [ ] Decoder weights: random init
- [ ] Train on real tracking data (vanvliet, same splits as baseline)
- [ ] Compare convergence: SSL init vs random init (baseline, K=16, K=32)
- [ ] Metrics: val_loss per epoch, edge accuracy, final TRA/AOGM
- [ ] Low-label ablation: 10%, 50%, 100% of tracking data

### 8. KNN + SSL combined

- [ ] Run §4.2 SSL pretraining with `--knn_neighbors 16` or `32`
- [ ] Train encoder with KNN attention during SSL
- [ ] §4.3 fine-tuning with same KNN config
- [ ] Max_tokens=8192+ for large-N KNN benefit
- [ ] Compare: SSL+KNN vs KNN-only vs dense-only vs SSL+dense

### 9. Evaluation

- [ ] Edge accuracy on val data (existing eval_model.py)
- [ ] Install traccuracy for TRA/AOGM
- [ ] Run full tracking eval on dense vs KNN vs SSL+KNN
- [ ] Produce comparison tables and convergence plots

### 10. Report figures

- [ ] Throughput: steps/sec vs N for dense, K=4,16,32
- [ ] SSL convergence: val_loss per epoch per distortion variant
- [ ] Downstream convergence: SSL init vs random init at matched epochs
- [ ] Distortion ranking: which transfers best
- [ ] KNN accuracy: edge accuracy bar chart
