# Session Context — Handover to Next Agent

**Date:** 2026-05-26  
**Use:** Reference for next conversation. Contains all active state, key findings, running jobs, architecture details, and bug fixes.

---

## Mode

caveman full. Drop articles. Fragments OK. Technical terms exact.

---

## Project

CMS-PRO semester project. Efficient sparse attention + SSL pretraining for transformer-based cell tracking (Trackastra). Evaluate KNN gather attention vs dense masked SDPA. Self-supervised contrastive pretraining (SimCLR/InfoNCE) from unlabeled segmentations via geometric distortion.

---

## Repos

| Repo | Branch | Contents |
|------|--------|----------|
| `lest161c/trackastra` | `feature/sparse-attention-gather` | KNN attention ported, Literal import fixed, SLURM training scripts |
| `lest161c/bench-transformer-cell-tracking` | `feature/imgfeat-ssl` (**NOTE:** not master) | All benchmarks: scaling, SSL, sweep, Trackastra training scripts. DropoutDistortion fix pushed to this branch. |
| `lest161c/benchmark-ssl` | `main` | Standalone SSL pipeline: CellEmbedder, NT-Xent, distortions |
| Labbook | N/A | `/home/leonard.starke@mediainterface.de/Dokumente/Uni/research-proj/labbook/` |

---

## SSH / Cluster

- **Cluster:** `capella` (HPC, H100 GPUs, 80GB)
- **SSH key for GitHub:** `eval $(ssh-agent) && ssh-add ~/.ssh/trackastra`
- **Data path:** `/data/cat/ws/mawe985g-data/data/celltracking/vanvliet/<condition>/`
- **Trackastra env:** `/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra/trackastra_env`
- **Benchmarks path:** `/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking/`
- **VPN required:** `eduvpn` (eduVPN GUI) — must connect to TU Dresden before SSH works

---

## Running Cluster Jobs (2026-05-26 21:30 update)

| Job ID | Name | What | K | Progress | Node | Status |
|--------|------|------|---|----------|------|--------|
| 3576409 | `tr_baseline` | Trackastra dense | — | 234ep, best val=0.002 | c69 | **COMPLETED** (6h51m) |
| 3576410 | `tr_sparse_k4` | Trackastra KNN | 4 | ~ep52, best val=0.0151 | c69 | RUNNING |
| 3576411 | `tr_sparse_k16` | Trackastra KNN | 16 | ~ep36, best val=0.0023 | c90 | RUNNING |
| 3576412 | `tr_sparse_k32` | Trackastra KNN | 32 | ~ep26, best val=**0.0014** | c116 | RUNNING |
| 3578124 | `dist_abl2` | 7 distortion variants (FIXED v3) | — | PENDING | — | QUEUED |

### Cancelled Jobs

| Job ID | Name | Reason |
|--------|------|--------|
| 3576397 | `dist_abl` (buggy) | FAILED exit 2:0, 43min |
| 3578086 | `dist_abl2` (label fix only) | CANCELLED — label fix insufficient. Dropout was still applied independently to each view, breaking sorted-collation alignment. Replaced by 3578124 with full fix. |

---

## Distortion Ablation Results (from BUGGY code — DropoutDistortion labels broken)

Extracted from TF events in `benchmark_ssl/runs/dist_abl_*/`:

| Variant | Train Loss (start→end) | Val Loss (start→end) | Train Cons | Val Cons |
|---------|------------------------|---------------------|------------|----------|
| full | 3.44 → 3.07 (ep20) | 3.43 → 2.88 | 0.931 | 0.932 |
| no_jitter | 3.47 → 3.02 (ep20) | 3.38 → 2.85 | 0.932 | 0.930 |
| **no_dropout** | **2.09 → 0.41 (ep20)** | **1.25 → 0.23** | 0.944 | **0.981** |
| no_feature_noise | 3.46 → 2.99 (ep20) | 3.33 → 2.95 | 0.929 | 0.930 |
| no_photometric | 3.46 → 3.03 (ep20) | 3.37 → 2.95 | 0.929 | 0.939 |
| no_elastic | 3.48 → 3.07 (ep9 only) | 3.41 → 2.93 | 0.933 | 0.945 |
| no_affine | _MISSING_ | _MISSING_ | — | — |

**Key finding:** `no_dropout` is only variant that converges. Val loss 0.23 vs 2.88 (full). Consistency 0.981 vs 0.932. All other variants flat at ~3.0.

### Why: Confirmed Bug

**DropoutDistortion** in `distortions.py` had `l = l[:len(c)]` (truncate labels to new cell count after dropout). Cells dropped randomly → remaining cells have FIRST len(c) labels, not the labels of the ACTUALLY kept cells. Labels misaligned → `valid_pair` mask rejects most pairs → SSL collapses.

**Also:** DropoutDistortion only returned `(coords, feats)` without labels. All other distortions returned `(coords, feats)` without labels too. `_apply_view` had special isinstance check to truncate labels after dropout.

**Fix (pushed to `feature/imgfeat-ssl`):**
- All distortions now return `(coords, features, labels)` consistently
- `DropoutDistortion` properly filters labels with `labels_t = labels[keep]`
- Removed `isinstance` check in `_apply_view`

### Collation Label-Sorting Issue

`SSLDataset` sorts cells by label in both views independently. With dropout removing cells from one view but not the other, label-sorted positions become misaligned. The `valid_pair` mask in collation filters mismatches, but with the bug above, even correctly-filtered labels would cause misalignment.

**This is fundamental design issue:** cell dropout + label-sorted collation = broken positive pair alignment. Fix is to either (a) apply dropout identically to both views, or (b) match by label instead of relying on index alignment.

---

## Trackastra Dense Baseline (3576409) — COMPLETED

- 234 epochs, early stopped at 83 epochs no improvement
- Best val_loss: **0.002** (epoch 151, step 101384)
- Final model: `/trackastra/runs/2026-05-26_11-13-51_vanvliet_baseline/model.pt`
- Training time: 6h51m

### Trackastra K4/K16 Naming Conflict

All sparse jobs use `--name sparse_k16` regardless of actual K. K4 (3576410) and K16 (3576411) share same run dir `2026-05-26_11-13-51_sparse_k16/`. Lightning auto-detects conflict and adds `-v1.ckpt` suffix for second job. Checkpoints distinguishable but `model.pt` overwritten. **Fix needed:** use distinct `--name` per K value in slurm scripts.

Current best val_loss:
| K | Job | Best val_loss | Epoch |
|---|-----|---------------|-------|
| 4 | 3576410 | 0.0159 | 50 |
| 16 | 3576411 | 0.0023 | 36 |
| 32 | 3576412 | 0.0021 | 25 |
| dense | 3576409 | 0.002 | 151 |

Dense still best so far. K=4 significantly worse (0.016 vs 0.002). K=16 and K=32 approaching dense quality.

---

## SSL Phase 1 Results (existing)

| File | Contents | Key numbers |
|------|----------|-------------|
| `runs/phase1/scaling_h100.csv` | H100 attention timing | K=4 1.6x faster at N=2048 L=12, faster at ALL N |
| `runs/ssl_phase1/training_log.csv` | SSL pretraining 50ep | loss 3.46→2.97 (plateau ep10), consistency 0.97→0.92 |
| `runs/phase1/label_sweep.csv` | Frozen embedding tracking | dense 64.1%, KNN 66.9%, SSL 65.1% — no SSL improvement |
| Collapse diagnostic | Inter-cell similarity | Random 0.89, SSL 0.85-0.90 (all cells near-identical) |

**NOTE:** SSL phase 1 (training_log.csv) had dropout → contaminated by label bug. Results unreliable.

---

## Key Technical Findings (UPDATED)

1. **KNN attention works.** 1.6x faster per step at N=2048 on H100. Enables FlashAttention-2 (no `attn_mask`). Dense OOMs at N=8192 L=12. Token reordering dead (<3%).
2. **SSL architecture proven correct.** Synthetic test: loss 1.41→0.03, inter-sim 0.68→0.17.
3. ~~**SSL on regionprops fails.**~~ **BUG FOUND.** DropoutDistortion label filtering broken. Without dropout, SSL converges (val 0.23, consistency 0.98). Awaiting fixed-distortion full re-run (3578086).
4. **ImageNet backbones don't transfer.** Frozen ResNet18/DINOv2 ViT on 64×64 bacterial patches: loss flat, no separation. Scratch CNN shows direction (loss 3.58→2.88) but undertrained.
5. **KNN structural regularizer.** On synthetic ablation: K=4 gives 4.7x lower val_loss than dense (0.108 vs 0.504). On real Trackastra: K=4 (0.016) worse than dense (0.002), K=16/32 approaching dense (0.0023/0.0021).

---

## Architecture Reference

### CellEmbedder (`benchmark_ssl/track_encoder.py`)
```
feat_proj(MLP) + coord_enc(MLP) → element-wise addition
→ TransformerEncoderLayer × L
→ enc_norm → embedding_head
→ per-cell embedding z (B, N, d_model)
```

### NT-Xent loss (`benchmark_ssl/train_ssl.py`)
```
Per-frame InfoNCE:
 - Normalize z1, z2 to unit sphere
 - Concatenate: [z1, z2] → (2N, D)
 - Cosine similarity matrix / temperature (0.05)
 - Cross-entropy: positive = z1[i]↔z2[i], negatives = all other 2N-1 embeddings
 - Average across frames in batch
```

### DistortionPipeline (`benchmark_ssl/distortions.py`)
```
__call__(coords, features, labels) → c1,f1,l1, c2,f2,l2
Two independently augmented views. Each distortion applied sequentially.
Distortions: affine, elastic, jitter, dropout, photometric, feature_noise.
Label-sorted in SSLDataset for positive-pair alignment.

FIXED: All distortions return (coords, features, labels). Dropout properly filters labels.
```

### Collation (`benchmark_ssl/ssl_pipeline.py`)
```
collate_ssl: pads both views to max N across batch.
valid_pair mask: label1 == label2 & ~padding1 & ~padding2.
Cells sorted by label → matching cells at same index positions.

ISSUE: Cell dropout breaks index alignment since each view drops cells independently.
```

---

## Known Bugs / Fixes

| Bug | Fix | Where | Status |
|-----|-----|-------|--------|
| `Literal` not imported | `from typing import Literal` | `trackastra/model/model_parts.py` | Pushed |
| Transformer NaN (custom block) | Replace with `nn.TransformerEncoderLayer` | `benchmark_ssl/track_encoder.py` | Done |
| Dense benchmark fp16 overflow | `-1e9` → `-65504.0` | `run_phase1_scaling.slurm` | Done |
| KNN gather shape mismatch | `q.unsqueeze(2)` for FA compatibility | `run_phase1_scaling.slurm` | Done |
| Cluster data path | `data_root: <data>/vanvliet` (not `<data>`) | `run_phase1_ssl.slurm` | Done |
| **DropoutDistortion labels** | `l[:len(c)]` → `labels[keep]`, return labels from all distortions | `benchmark_ssl/distortions.py` | **Pushed** (feature/imgfeat-ssl) |
| **Dropout independent per view** | Dropout rng state saved before view1, restored before view2 → identical cell drops both views | `distortions.py` DistortionPipeline | **Pushed** (feature/imgfeat-ssl) |
| K4/K16 naming conflict | Same `--name sparse_k16`, same timestamp → shared run dir | slurm scripts | **FIXED** — `--name sparse_k4`, `--name sparse_k32` |

---

## Next Steps (after cluster jobs complete)

1. Wait for `dist_abl2` (3578086) → check fixed distortion ablation. Expect ALL variants to converge better. Remove dropout from pipeline permanently.
2. Wait for K4/K16/K32 (3576410-3576412) to complete → TRA/AOGM evaluation. K4 likely underperforms due to too few neighbors.
3. Run downstream tracking evaluation (TRA, AOGM) on best models from each K.
4. Compile all CSVs → figures (throughput, convergence, distortion ranking, K vs val_loss).
5. Write report. Structure: Introduction → KNN Attention → SSL Architecture (incl. bug fix) → Results → Discussion.

---

## Proposal Status

| Section | Status |
|---------|--------|
| §4.1 Sparse attention benchmark | Done (standalone + H100 + port) |
| §4.1 Token reordering | Dropped (<3% benefit) |
| §4.2 SSL pretraining | **BUG FIXED.** DropoutDistortion was broken. Re-run pending (3578086). |
| §4.3 Fine-tuning | In progress (baseline done, K4/K16/K32 running) |
| §4.4 Feature ablations | Explored (image patches negative result; proposal marks §4.4 optional) |
