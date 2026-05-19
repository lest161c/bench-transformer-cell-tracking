# Research Project Plan

## Project: Efficient Sparse Attention + SSL Pretraining for Transformer Cell Tracking

Based on proposal: `proposal.txt`
Attention benchmark: `benchmark_attn/`
Trackastra fork: `trackastra/` (submodule, branch `feature/sparse-attention-trackastra`)
SSL exploration: `experimental/ssl-pretraining` branch (old, failed approach)

---

## Combined Benchmark (benchmark_combined/)

2x2 factorial: (Dense vs Sparse attention) × (Random vs SSL init).
PoC on vanvliet data (RTX A500 Laptop, 4GB).

**Key result:** sparse+ssl 43% lower epoch-1 val_loss than baseline.
Estimated ~8-78× total wall-time savings at scale (N=8192).

Files:
- `benchmark_combined.py` — Unified 4-variant benchmark
- `plot_combined.py` → `benchmark_combined.html` (10 figures)
- `results_analysis.md` — Full analysis

Next: run on GPU cluster at realistic cell counts (N=2000-8000).

## Part 1: Efficient Sparse Attention (benchmark_attn/)

### Status

`benchmark_attn/` has complete standalone benchmark with `GatherSparseAttention` vs dense masked SDPA.
Results in `benchmark_sparse_results.csv`: N ∈ {128,512,2048,8192}, K ∈ {4,16,64}, L ∈ {1,4}, with/without `SpatialReorder`.

**Key results** (RTX 4090, fp16, B=2, d=256, h=4):
- N=2048: sparse K=16 ≈ same speed as dense, 1.4× less memory
- N=8192 L=1: sparse K=16 is **3.9× faster**, **5.4× less memory** than dense
- N=8192 L=4: dense OOMs, sparse K=4 fits at 385MB
- SpatialReorder: negligible speedup (~1-3%) on random/uniform data. Real centroid data may differ.

### Tasks

- [x] **P0.1: Generate plots** — `plot_sparse.py` → `benchmark_sparse.html` (12 figures, 943KB)
- [ ] **P0.2: Real centroids reorder benchmark** — Centroids exist at `data/vanvliet/rpsM/151101_E4-12/centroids.npy` (2202×2, float32). Script path correct. **Needs CUDA** — run on GPU cluster
- [x] **P0.3: Analysis writeup** — `benchmark_attn/results_analysis.md` written
- [ ] **P0.4: Reorder benchmark run** — `benchmark_reorder_real_vs_random.py` needs CUDA. RTX 4090, ~5 min runtime
- [x] **P0.5: Config clarity** — `config.yaml` mirrors trackastra config schema (knn_neighbors, attn_positional_bias). Used as template, not directly consumed by benchmark scripts

### Trackastra port

`GatherSparseAttention` already ported to `trackastra/trackastra/model/model_parts.py:289`.
Wired in `model.py` via config key `knn_neighbors` (line 300-339).
Enable with `knn_neighbors: 12` in config (disabled if ≤ 0).

**Open tasks:**
- [ ] Verify parity: train trackastra with sparse attention, compare metrics to dense baseline
- [ ] Benchmark trackastra-level throughput (not just single layer)

---

## Part 2: SSL Pretraining via Geometric Distortion

### Motivation

Annotated tracking data is scarce. SSL pretext: take a single frame with segmentations {si},
apply random transformation T → synthetic "next frame" {T(si)}. Ground-truth association i→i (identity).
Train association head on synthetic pairs. No tracking labels needed — only segmentations.

### Previous attempt (branch `experimental/ssl-pretraining`)

Failed. Model learned shallow geometric features specific to augmentation pipeline,
overfitted rapidly, transferred worse than random init to real tracking.

**Root cause:** naive augmentation-based pretraining lets the model memorize the distortion
pattern itself rather than learning about cell identity and motion dynamics. The network
learns "if cells move like this, they must be the same cell" — a trivial shortcut.

### How ASCENT paper informs proper SSL approach

The ASCENT paper (aircraft trajectory prediction) succeeds through:
1. **Agent-centric coordinate normalization** — normalize trajectory to local frame,
   removing global position bias. For cells: normalize cell positions relative to each cell's
   own reference frame before encoding.
2. **Self-attention motion encoding** — encode relative motion patterns, not absolute positions.
3. **Mode query decoder** — multi-hypothesis prediction prevents collapse to single mode.
4. **Parameterized prediction** — predict kinematics (velocity, heading) not raw positions.

For SSL to work in cell tracking, the pretext task must force the network to learn
**cell identity and motion dynamics**, not warp geometry. Key design principles:
- Normalize coordinates to agent-centric frame before encoding
- Use self-attention over cell trajectories (not shallow MLP)
- Predict multiple association hypotheses (mode queries)
- Predict motion parameters rather than direct associations
- Distortion families must be diverse enough to prevent shortcut learning

### Repo: `benchmark_ssl/` (isolated, like `benchmark_attn/`)

```
benchmark_ssl/
├── distortions.py          # All distortion families
│   ├── affine (translation, rotation, scale, shear)
│   ├── elastic / thin-plate-spline
│   ├── per-cell jitter (independent cell motion — most realistic)
│   ├── detection dropout (simulate segmentation failures)
│   └── photometric (intensity, noise — only if image features)
├── ssl_pipeline.py         # Pretext: frame → T(frame) + identity labels
├── track_encoder.py        # Coordinate normalization + self-attention encoder
├── pretrain.py             # SSL training loop
├── finetune.py             # Downstream fine-tuning on real tracking data
├── evaluate.py             # TRA/cHOTA computation
├── ablations.py            # Distortion family comparison, label fraction study
├── config.yaml
├── plot_ssl.py
├── pyproject.toml
└── README.md
```

### Tasks

- [x] **P1.1: Repo setup** — `benchmark_ssl/` created with pyproject.toml, config.yaml
- [x] **P1.2: Distortion pipeline** — `distortions.py` + `ssl_pipeline.py`
  - 5 distortion families: affine, elastic, jitter, dropout, photometric
  - Loads vanvliet data (all conditions), extracts WRFeatures per frame
  - Generates identity association labels with correct label matching
  - Pipeline verified end-to-end — loads 924 frames from rpsM/recA/pheA
  - Collator produces padded batch tensors compatible with Trackastra format
- [x] **P1.3: Encoder** — `track_encoder.py`
  - `AgentCentricNormalization`: pairwise displacement + direction + distance
  - `AssociationEncoder`: feature proj + coordinate proj + pairwise encoding → transformer encoder → pairwise scoring
  - Mode query decoder for multi-hypothesis (ASCENT-inspired)
  - Verified: forward pass, backward pass, correct output shapes
- [x] **P1.4: Pretraining** — `pretrain.py`
  - Training loop with BCE loss + positive weighting (handles ~1% positive ratio)
  - Validation loop with accuracy/precision/recall/F1
  - TensorBoard logging, checkpoint saving
  - Verified: GPU training on RTX A500 Laptop — 5 epochs, 6 min runtime
- [x] **P1.5: Proof-of-concept training** — `train_ssl.py` standalone
  - 5 epochs on vanvliet (rpsM+recA+phea, 924 frames)
  - Loss 0.50→0.15 (-69%), Acc 89%→97%, no overfitting
  - CSV logging at `runs/ssl_v1/training_log.csv`
  - Best model at `runs/ssl_v1/best_model.pt`
- [x] **P1.6: Visualizations** — `plot_ssl.py` → `benchmark_ssl.html`
  - 6 figures: loss convergence, accuracy, F1, precision/recall, epoch times, improvement table
- [x] **P1.7: Analysis writeup** — `benchmark_ssl/results_analysis.md`
- [ ] **P1.8: Distortion ablation** — Which T gives best downstream transfer? (next)
- [ ] **P1.9: Label fraction study** — Fine-tune with 1%, 5%, 10%, 50%, 100% labels
- [x] **P1.10: Compare SSL vs random init** — On vanvliet data
  - SSL-pretrained encoder → 26% lower val loss on real tracking
  - SSL converges ~3× faster to low-loss regime
  - Full results in `results_analysis.md` Section 7

**Note:** See `benchmark_ssl/results_analysis.md` for full analysis.

## Part 2: Integration into Trackastra (pending GPU cluster)

---

## Part 3: Integration into Trackastra

### Tasks

- [ ] **P2.1: Sparse attention activation in Trackastra** — Verify config-driven sparse attention works end-to-end
- [ ] **P2.2: SSL weight init** — Load SSL-pretrained weights into Trackastra association head
- [ ] **P2.3: Full training** — Train Trackastra (sparse attention + SSL init) on bacteria + deepcell
- [ ] **P2.4: Baseline reproduction** — Original Trackastra (dense attention, no SSL) for comparison
- [ ] **P2.5: Metrics** — TRA, cHOTA on test sets

---

## Part 4: Report & Presentation

### Tasks

- [ ] Figures: speedup plot, memory plot, SSL ablation bar chart, label fraction curves
- [ ] Written report (proposal deliverable)
- [ ] Final presentation slides

---

## Timeline

| Week | Focus | Deliverables |
|------|-------|-------------|
| 1-2 | Phase 0 (close attention) + Phase 1 (SSL repo) | Plots, analysis, SSL pipeline code |
| 3-4 | Phase 2 (SSL experiments) | Pretraining results, distortion ablation |
| 5-6 | Phase 3 (Trackastra integration) | Sparse+SSL Trackastra, baseline comparison |
| 7-8 | Phase 4 (report) | Written report, figures, presentation |

---

## Branch Structure

```
main                          # Clean baseline
feature/sparse-attention-trackastra  # Current: sparse attention ported to trackastra
experimental/ssl-pretraining  # Old failed SSL attempt (reference only)
(SSL work will be on new branches from main or feature/sparse-attention-trackastra)
```

## Execution Environment

**Local machine:** No CUDA PyTorch (conflicting conda/pip installs). CPU-only for code, analysis, plotting.
**GPU cluster (SLURM):** All GPU work — benchmarks, reorder tests, SSL pretraining, Trackastra training.
**Constraint:** Benchmarks >4GB VRAM skipped locally (user note). Most sparse benchmarks fit this budget.

To run on cluster: `conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia`

## Key Files

| File | Purpose |
|------|---------|
| `proposal.txt` | Original project proposal |
| `benchmark_attn/` | Standalone sparse attention benchmark |
| `trackastra/trackastra/model/model_parts.py` | GatherSparseAttention impl (line 289) |
| `trackastra/trackastra/model/model.py` | Trackastra model with sparse attention wiring (line 300) |
| `trackastra/scripts/example_config.yaml` | Trackastra config template |
| `data/vanvliet/` | Cell microscopy data |
| `project_plan.md` | This file |
