# Node-Level Representation Probing — SPEC

## Goal
Train linear probes on **frozen encoder node embeddings** to measure how much biological information the Trackastra encoder captures. This is standard representation probing (like linear probes in BERT) adapted for cell tracking. Unlike HOCT's edge probing (which operates on edge tokens), we probe per-cell node embeddings — the only representation available in Trackastra's node-centric architecture.

## Prerequisites
- Base: `cached-dist-attn` branch, any trained Trackastra checkpoint (baseline, CNN, CNN+dropout, etc.)
- Probe code: `trackastra/trackastra/model/node_probing.py`
- CLI: `trackastra/scripts/probe_nodes.py`

## Probe Attributes

We define **5 node-level probes** (per-cell) and **1 pairwise edge probe**:

### P1: Division in Next Frame (binary classification)
- **Label**: Does this cell divide within Δt=1?
- **Source**: Ground truth association matrix — a cell at time t has ≥2 connections to cells at time t+1
- **Probe**: `Linear(d_model → 1)` with BCEWithLogitsLoss
- **Expected**: Encoders that capture cell morphology/state should predict division better than chance

### P2: Cell Displacement Magnitude (regression)
- **Label**: Euclidean distance (pixels) between centroid at time t and t+1 (persisting cells only)
- **Source**: Coordinate difference for cells with the same label across consecutive frames
- **Probe**: `Linear(d_model → 1)` with MSELoss
- **Expected**: Encoders that track cell position should embed displacement-relevant info

### P3: Cell Displacement Direction (multi-class, 8 octants)
- **Label**: Octant index 0–7 based on the angle of the displacement vector (persisting cells only)
- **Source**: `atan2(dy, dx)` mapped to 8 × 45° bins
- **Probe**: `Linear(d_model → 8)` with CrossEntropyLoss
- **Expected**: Directional movement bias (e.g., toward nutrients) should be detectable

### P4: Ancestor Count (regression)
- **Label**: Number of ancestor cells (earlier timepoints linked via association matrix)
- **Source**: Sum of `assoc_matrix[i, :]` for cells at earlier timepoints — proxy for lineage depth
- **Probe**: `Linear(d_model → 1)` with MSELoss
- **Expected**: Deeper-lineage cells may have distinct embedding patterns

### P5: Descendant Count (regression)
- **Label**: Number of descendant cells (later timepoints linked via association matrix)
- **Source**: Sum of `assoc_matrix[i, :]` for cells at later timepoints — proxy for track span
- **Probe**: `Linear(d_model → 1)` with MSELoss
- **Expected**: Cells early in long tracks vs. terminating cells should differ in embedding space

### P6: Edge Existence (binary classification, PAIRWISE)
- **Label**: Does an association edge exist between two cells (same track or parent→child)?
- **Source**: `assoc_matrix[i, j]` for pairs (i, j) with time[j] − time[i] ∈ [1, Δt_max]
- **Features**: `concat(e_i, e_j, e_i − e_j, e_i ⊙ e_j)` → 4·d_model
- **Probe**: `MLP(4·d_model → 64 → 1)` with BCEWithLogitsLoss
- **Expected**: This is the closest we can get to HOCT's edge probing (§4.2). If node embeddings carry association-relevant info, this probe should outperform random. A large gap vs. HOCT's 59% AOGM reduction would confirm their finding that edge-level representations are fundamentally more informative.

## Modifications

### M1: Node Probing Module (new file)
**File:** `trackastra/trackastra/model/node_probing.py`
**Components:**
- `NodeProbingHead`: Linear layer `d_model → output_dim` for per-cell attributes
- `EdgeProbingHead`: MLP on pairwise features for edge existence
- `train_node_probe()`: Runs frozen encoder, extracts (embedding, label) pairs, trains linear probe
- `evaluate_node_probe()`: Evaluates probe on held-out data
- Helper functions for attribute extraction from batches

### M2: Probe CLI Script (new file)
**File:** `trackastra/scripts/probe_nodes.py`
**Behavior:**
- Loads a trained model checkpoint (frozen)
- Loads data config via the same config system as `train.py`
- Extracts node embeddings for all cells in train/val splits
- Trains probe heads for each attribute (P1–P6)
- Prints a results table: probing accuracy per attribute per model variant
- Saves results to a JSON file

### M3: Probe SPEC (this file)
**File:** `benchmark_ssl/cnn_encoder/NODE_PROBING_SPEC.md`

## Expected Results

| Probe | Metric | Random Chance | Baseline (no CNN) | CNN Frozen | CNN+Dropout | CNN+Trainable |
|-------|:-----:|:-------------:|:-----------------:|:----------:|:-----------:|:-------------:|
| **P1: Division** | Balanced Accuracy | 0.50 | ~0.55 | ~0.58 | ~0.60 | ~0.62 |
| **P2: Disp. Mag** | R² | 0.00 | ~0.05 | ~0.08 | ~0.10 | ~0.12 |
| **P3: Disp. Dir** | Accuracy (8-class) | 0.125 | ~0.18 | ~0.20 | ~0.22 | ~0.25 |
| **P4: Ancestors** | R² | 0.00 | ~0.10 | ~0.12 | ~0.15 | ~0.18 |
| **P5: Descendants** | R² | 0.00 | ~0.08 | ~0.10 | ~0.13 | ~0.15 |
| **P6: Edge Exist** | Balanced Accuracy | 0.50 | ~0.55 | ~0.60 | ~0.62 | ~0.65 |

**Key comparisons:**
- **CNN vs. no-CNN**: If CNN features add tracking-relevant information, all probes should improve
- **Dropout vs. frozen**: If dropout forces more robust features, probing accuracy should improve
- **Trainable vs. frozen**: If fine-tuning the CNN for BCE loss preserves or improves SSL features, probes improve further
- **Edge probe (P6) vs. HOCT**: HOCT achieves 59% AOGM reduction with edge probing. Our pairwise node probe is expected to be weaker, providing a quantitative measure of the node-centric vs. edge-centric gap

## Confidence Assessment

| Outcome | Likelihood | Why |
|---------|:---------:|------|
| **Division probe beats random** | 90% | Division involves cell morphology and cycle state — encoder should capture this |
| **CNN improves all probes** | 75% | CNN adds visual features that correlate with cell state and position |
| **Edge probe beats random** | 70% | Node embeddings encode some pairwise information via positional encoding |
| **Edge probe << HOCT edge probe** | 95% | Node-centric dot-product is inherently less informative; 30× gap expected per HOCT analysis |
| **Probes identical across model variants** | 20% | If probe metrics don't change, it suggests the encoder learns the same representation regardless of CNN injection |

## Validation Checklist
- [ ] `probe_nodes.py` loads checkpoint without error
- [ ] `probe_nodes.py` runs encoder on data without error
- [ ] Per-cell attribute extraction produces correct labels (verify statistics)
- [ ] Probe training converges (loss decreases)
- [ ] Results table prints correctly
- [ ] Results JSON saves correctly
- [ ] Run on at least 2 model variants (baseline + CNN) to compare

## Implementation Order
1. Write `node_probing.py` — NodeProbingHead, EdgeProbingHead, attribute extraction, train/evaluate functions
2. Write `probe_nodes.py` — CLI wrapper following `train.py` style
3. Write this SPEC file
4. Smoke test: run `probe_nodes.py --checkpoint <path> --config <config.yaml> --dry`
5. Run on baseline (no-CNN) and CNN variants
6. Push results to repository
