# Current State — 2026-05-28

## Active Problem

**Build ASCENT-style contrastive SSL pretraining directly on Trackastra's encoder** (per proposal §4.2), then fine-tune on real tracking data (§4.3). Compare SSL init vs random init for downstream convergence.

**Why:** Proposal §4.2 requires training Trackastra's association head on synthetic pairs from geometric distortion of single frames. The identity association comes "for free" from the warp: i → T(i).

## What's Running

| Job ID | Name | Config | Status |
|--------|------|--------|--------|
| 3588978 | ssl_d2d | SSL(20ep,6dist,NT-Xent) → Trackastra(K16,max_tok=8192,100ep) | SSL ep13/20 |

## Implementation (3 new/modified files)

- `trackastra/model/model.py` (+44): `encode()` method — runs encoder only, returns `(B,N,d_model)` per-cell embeddings. Used for NT-Xent contrastive loss during SSL. No decoder involvement.
- `trackastra/data/ssl_distortions.py` (275): Fixed `DistortionPipeline` ported from benchmark_ssl. 6 distortion types (affine, elastic, jitter, dropout, photometric, feature_noise). Both label filtering (`labels[keep]`) and shared-RNG dropout fixes applied.
- `trackastra/model/ssl_trainer.py` (410): NT-Xent SSL training loop. Per-frame InfoNCE with temperature τ=0.05. Two augmented views → encoder each → NT-Xent. Inter-cell similarity metric detects embedding collapse.

## Key Findings So Far

### KNN edge accuracy (no SSL)
| Model | Edge Accuracy | BCE |
|-------|:---:|:---:|
| Dense baseline | 0.9769 | 0.099 |
| KNN K=16 | **0.9981** | 0.0036 |
| KNN K=32 | **0.9992** | 0.0013 |

KNN beats dense by 2+% edge accuracy. K=4 cancelled (7.5x worse val_loss).

### SSL convergence (fixed distortions)
- Consistency: **1.0** (encoder perfectly distortion-invariant across views)
- inter-cell similarity: **0.13→0.02** (different cells have near-zero cosine sim → discriminative embeddings, NO collapse)
- NT-Xent loss: **∼0.0** after 2 epochs (320-dim 6-layer encoder has high capacity for 7-dim regionprops — converges instantly)
- Without the distortion fix (label truncation + independent dropout): loss stuck at 3.2 (SSL fails completely → collapse)

### KNN throughput (A500, fwd+bwd)
| N | Dense (s/s) | K=4 (s/s) | K=16 (s/s) | K=32 (s/s) |
|---|:---:|:---:|:---:|:---:|
| 2048 | 3.0 | **3.5** | 2.7 | 2.1 |
| 4096 | 0.8 | **1.8** | 1.4 | 1.1 |
| 6144 | OOM | — | — | — |

KNN advantage increases with N. At N=4096, K=4 is 2.1x faster. Dense OOMs at N≥6144 on 4GB A500. On H100 (80GB), KNN enables N=8192+ where dense can't run.

## Bugs Found & Fixed

| Bug | Where | Fix |
|-----|-------|-----|
| DropoutDistortion label truncation `l[:len(c)]` | distortions.py | `labels[keep]` |
| Independent per-view dropout | DistortionPipeline | Shared RNG via `__getstate__`/`__setstate__` |
| Distortions not returning labels | Multiple classes | All return `(coords, features, labels)` |
| K4/K16 slurm name collision | slurm scripts | Distinct `--name sparse_k4`/`sparse_k32` |
| Empty-frame crash `coords.max()` | model.py encode() | `coords.amax(dim=(0,1))` + `padding_mask.any()` guard |
| train.py path wrong in slurm | run_ssl_d2d.slurm | `cd $TRK` before `python scripts/train.py` |
| Bench ssl recopy from wrong git branch | ssl_distortions.py | Checkout `feature/imgfeat-ssl` (has both fixes) |

## Remaining Issues

1. **SSL converges too easily** — 320-dim encoder + 6 layers + 7-dim features = trivial task. Model memorizes feature space instantly. May not transfer meaningful representations to tracking. Downstream fine-tuning is the real test.

2. **No TRA/AOGM evaluation** — all comparisons are edge prediction accuracy. Need CTC tracking metrics for proper evaluation. `traccuracy` not installed.

3. **Baseline vs KNN config mismatch** — baseline uses `window=4, max_tokens=2048, crop_size, compress, train_samples=32000` but KNN uses `window=10, max_tokens=4096, no crop, no compress, all data`. Makes throughput comparison unreliable.

4. **K4/K16 checkpoint collision** — both shared `2026-05-26_11-13-51_sparse_k16/` run dir. `model.pt` was overwritten. Slurm scripts fixed for future runs.

## Next Steps

1. Wait for 3588978 to complete — check downstream convergence (SSL init vs random)
2. Run local eval on SSL-pretrained downstream model → compare with K16 baseline
3. If SSL helps: run distortion ablation on Trackastra SSL
4. If SSL doesn't help: reduce model capacity for SSL, add feature noise magnitude, or use image patches instead of regionprops
5. Install traccuracy for TRA/AOGM evaluation
6. Write report
