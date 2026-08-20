# SPEC: Obsidian Knowledge Vault

**Date:** 2026-08-19
**Status:** Draft — awaiting approval

---

## 1. Goal

Create a concise Obsidian vault that serves as the **shared brain** of the research project. The vault is NOT a 1:1 mirror of all project knowledge. It is a **distilled synthesis** of the answered research questions, their evidence, and their conclusions.

The vault answers: **"What did we ask, what did we find, and what does it mean?"**

---

## 2. Vault Location

```
vault/                    # Obsidian vault root (gitignored from main repo)
├── .obsidian/            # Obsidian config (gitignored)
├── 00-MOC.md             # Map of Content — central entry point
├── questions/            # One note per research question
├── evidence/             # One note per experiment/result
├── context/              # Literature notes, architecture, data
├── conclusions/          # One note per conclusion
├── attachments/          # Figures, PDFs, CSVs
└── templates/            # Note templates
```

---

## 3. Content

### 3.1 `questions/` — Research Questions

Each question is a note with the question, the answer, and links to evidence.

| Note | Question | Status |
|------|----------|--------|
| `Q1-Sparse-KNN-Attention.md` | Can sparse KNN attention replace dense masked SDPA? | Answered |
| `Q2-SSL-Pretraining.md` | Can SSL pretraining improve data efficiency? | Answered (failed) |
| `Q3-Spatial-Cutoff-Ablation.md` | Does the spatial cutoff affect tracking accuracy? | Answered |
| `Q4-Attention-Bottleneck.md` | Is attention the training bottleneck at vanvliet scale? | Answered |
| `Q5-Feature-Engineering-Ceiling.md` | What is the feature-engineering ceiling for edge prediction? | Answered |
| `Q6-CNN-Feature-Injection.md` | Can learned visual embeddings bootstrap tracking? | Answered (failed) |
| `Q7-Cross-Dataset-Generalization.md` | Does the baseline generalize across datasets? | Answered |
| `Q8-Optimal-Architecture.md` | What is the optimal architecture at vanvliet scale? | Answered |

### 3.2 `evidence/` — Experimental Evidence

Each experiment is a note with setup, key results, figure embeds, and interpretation.

| Note | Content |
|------|---------|
| `E1-Attention-Benchmark.md` | H100 benchmark: 1.78× speedup at N=128 |
| `E2-KNN-K-Sweep.md` | K-sweep: K=16 best (TRA 0.9972), all K converge identically |
| `E3-Profiler-Analysis.md` | Profiler: cdist=82% of attention time, attention=5% of step time |
| `E4-Spatial-Cutoff-CV.md` | 6-fold CV: ΔTRA=-0.0003, ΔAOGM=+4.0 (spatial cutoff is purely computational) |
| `E5-SSL-Identity-BCE.md` | Identity BCE: val_loss 0.404 vs 0.416 baseline (no improvement) |
| `E6-SSL-NT-Xent.md` | NT-Xent: cosine similarity 0.89 (collapse) |
| `E7-Edge-Probing.md` | Edge probe: DINOv2 0.896, HOCT 19D 0.880, regionprops 7D 0.860 |
| `E8-CNN-Feature-Injection.md` | CNN injection: all 8 variants worse than baseline (3.8–25.7×) |
| `E9-Cross-Dataset-DeepCell.md` | DeepCell: baseline best (TRA 0.990, cHOTA 0.956, AOGM 592) |
| `E10-Amdahls-Law.md` | Amdahl: 1/(0.95+0.05/1.78)=1.022 → 2.2% training speedup |
| `E11-KNN-Equivalence.md` | 18-config forward/backward equivalence test |
| `E12-Blockwise-Norm.md` | Blockwise norm: 1.96× speedup on normalization component |

### 3.3 `context/` — Background Knowledge

Concise notes on the architecture, data, and literature that provide context for the questions.

| Note | Content |
|------|---------|
| `C1-TrackingTransformer.md` | Trackastra architecture: encoder-decoder, node-centric attention |
| `C2-Attention-Variants.md` | Dense masked, gather-KNN, mask-KNN, dense flash |
| `C3-VanVliet-Dataset.md` | 33 sequences, 1,690 frames, 75,828 cells; median N=62, mean N=166 |
| `C4-DeepCell-Dataset.md` | 12 sequences, fluorescence microscopy |
| `C5-Regionprops-Features.md` | 7D features: centroid, area, mean intensity, etc. |
| `C6-Literature-Trackastra.md` | Gallusser & Weigert, ECCV 2024 |
| `C7-Literature-HOCT.md` | Bragantini et al., 2026 — edge-centric architecture |
| `C8-Literature-FlashAttention.md` | Dao, 2023 — FlashAttention-2 |
| `C9-Literature-NSA.md` | Yuan et al., 2025 — Native Sparse Attention |
| `C10-Literature-DINOv2.md` | Oquab et al., 2023 — self-supervised ViT features |
| `C11-Capella-HPC.md` | TU Dresden Capella cluster, H100 GPUs, SLURM |

### 3.4 `conclusions/` — Project Conclusions

Each conclusion is a note that synthesizes evidence into a finding.

| Note | Conclusion |
|------|------------|
| `F1-Attention-Not-Bottleneck.md` | At vanvliet scale (median N=62, mean N=166), attention is <5% of step time. The bottleneck is data loading and feature extraction. |
| `F2-DenseFlash-Optimal.md` | DenseFlashAttention is optimal at vanvliet scale: 1.78× attention speedup with zero accuracy cost (ΔTRA=-0.0003). |
| `F3-Spatial-Cutoff-Purely-Computational.md` | Removing the spatial cutoff has no measurable effect on tracking accuracy (6-fold CV, ΔTRA=-0.0003, ΔAOGM=+4.0). |
| `F4-SSL-Failed.md` | Both identity BCE (val_loss 0.404 vs 0.416) and NT-Xent (cosine 0.89, collapse) fail to improve over from-scratch baseline. Root cause: 7D regionprops insufficient + coordinate shortcut. |
| `F5-Feature-Engineering-Ceiling.md` | Edge probing shows +0.036 balanced accuracy from 7D to 384D features. The ceiling is architectural, not representational. |
| `F6-CNN-Injection-Trap.md` | Post-LayerNorm additive injection creates an irreversible local minimum. All 8 CNN variants are 3.8–25.7× worse than baseline. |
| `F7-Node-Centric-Bottleneck.md` | The node-centric architecture collapses pairwise edge representations to a single scalar. HOCT's edge-centric paradigm achieves ~30× larger AOGM reduction. |
| `F8-Future-Work-Edge-Centric.md` | Integrate HOCT's edge-centric paradigm into the sparse-attention framework. Test whether edge-centric attention rescues SSL pretraining. |

### 3.5 `attachments/` — Binary Assets

All figures and data files referenced by evidence notes.

| Source | Content |
|--------|---------|
| `report/resources/figures/*.pdf` | Report figures |
| `report/resources/figures/*.svg` | Report figure sources |
| `benchmark_attn/results/*.csv` | Benchmark CSVs |
| `benchmark_attn/results/*.png` | Benchmark PNGs |
| `benchmark_ssl/results/*.json` | SSL probe results |
| `papers/*.pdf` | Literature PDFs |

### 3.6 `templates/`

| Template | Used For |
|----------|----------|
| `Template-Question.md` | Research question notes |
| `Template-Evidence.md` | Experiment result notes |
| `Template-Conclusion.md` | Conclusion notes |
| `Template-Context.md` | Background knowledge notes |

---

## 4. Note Templates

### 4.1 Question Template

```markdown
---
tags: [question]
status: answered | open | failed
date: YYYY-MM-DD
---

# [Question Title]

> [Question text, one sentence]

## Answer

[2-3 sentence answer]

## Evidence

- [[E1-...]] — [one-line summary of what this evidence shows]
- [[E2-...]] — [one-line summary]

## Related

- [[Q3-...]] — [how this question relates to another]
- [[F1-...]] — [which conclusion this question supports]
```

### 4.2 Evidence Template

```markdown
---
tags: [evidence]
date: YYYY-MM-DD
---

# [Experiment Title]

## Setup

- **Dataset:** [[C3-...]]
- **Hardware:** [[C11-...]]
- **Key parameters:** [comma-separated list]

## Results

| Metric | Value |
|--------|-------|
| [metric] | [value] |

![[attachment_name.png]]

## Interpretation

[2-3 sentences on what the results mean]

## Supports

- [[Q3-...]] — [how this evidence answers the question]
- [[F1-...]] — [how this evidence supports the conclusion]
```

### 4.3 Conclusion Template

```markdown
---
tags: [conclusion]
date: YYYY-MM-DD
---

# [Conclusion Title]

## Finding

[2-3 sentence synthesis]

## Evidence

- [[E1-...]] — [one-line summary]
- [[E2-...]] — [one-line summary]

## Implications

[1-2 sentences on what this means for the project]

## Related

- [[Q1-...]] — [which question this conclusion answers]
- [[F3-...]] — [how this conclusion relates to another]
```

### 4.4 Context Template

```markdown
---
tags: [context]
date: YYYY-MM-DD
---

# [Context Note Title]

## Summary

[2-3 sentence summary]

## Key Details

- [detail 1]
- [detail 2]

## Related

- [[C1-...]] — [how this context relates to another note]
- [[E1-...]] — [which evidence uses this context]
```

---

## 5. Map of Content (`00-MOC.md`)

The MOC is the central entry point. It includes:

1. **Project summary** (2-3 sentences)
2. **Mermaid graph** showing question → evidence → conclusion flow
3. **Links to all questions** grouped by theme (attention, SSL, features, generalization)
4. **Links to all conclusions** in priority order
5. **Links to all evidence** grouped by experiment type
6. **Links to all context** grouped by type (architecture, data, literature, infra)

---

## 6. Linking Strategy

### 6.1 Link Types

| Link Type | Format | Example |
|-----------|--------|---------|
| Question → Evidence | `[[E1-...]]` | `[[E4-Spatial-Cutoff-CV]]` |
| Question → Conclusion | `[[F1-...]]` | `[[F3-Spatial-Cutoff-Purely-Computational]]` |
| Evidence → Question | `[[Q3-...]]` | `[[Q3-Spatial-Cutoff-Ablation]]` |
| Evidence → Conclusion | `[[F1-...]]` | `[[F1-Attention-Not-Bottleneck]]` |
| Conclusion → Evidence | `[[E1-...]]` | `[[E10-Amdahls-Law]]` |
| Context → Evidence | `[[E1-...]]` | `[[E1-Attention-Benchmark]]` |
| Figure embed | `![[file.png]]` | `![[01_speed_vs_n.svg]]` |

### 6.2 Tags

Each note has YAML frontmatter with tags:

```yaml
---
tags: [question, attention, ssl]
status: answered | open | failed
date: 2026-08-19
---
```

### 6.3 Backlinks

Obsidian's backlink panel is the primary navigation mechanism. Each note ends with a `## Related` section linking to connected notes.

---

## 7. Key Numbers

These are the verified numbers that must appear consistently throughout the vault:

| Metric | Value | Source |
|--------|-------|--------|
| DenseFlash vs masked speedup | 1.78× | `full_bench_h100.csv` |
| Amdahl training speedup | 2.2% | `1/(0.95+0.05/1.78) = 1.022` |
| Attention share of step time | ~5% | Profiler, labbook 2026-05-29 |
| Vanvliet median N | 62 | Dataset statistics |
| Vanvliet mean N | 166 | Dataset statistics |
| CV ΔTRA | -0.0003 | `eval_results.json` from HPC |
| CV ΔAOGM | +4.0 | `eval_results.json` from HPC |
| Edge probe DINOv2 bal_acc | 0.896 ± 0.036 | `unified_probe_results_cv.json` |
| Edge probe regionprops 7D bal_acc | 0.860 ± 0.014 | Same |
| Edge probe HOCT 19D bal_acc | 0.880 ± 0.020 | Same |
| Identity BCE val_loss | 0.404 | Labbook |
| From-scratch baseline val_loss | 0.416 | Labbook |
| NT-Xent inter-cell cosine similarity | 0.89 | Labbook |
| Blockwise norm speedup | 1.96× | `blockwise_norm_results.csv` |
| K=16 TRA | 0.9972 | K-sweep results |
| K=16 AOGM | 37.7 | K-sweep results |
| DeepCell baseline TRA | 0.990 | DeepCell eval |
| DeepCell baseline cHOTA | 0.956 | DeepCell eval |
| DeepCell baseline AOGM | 592 | DeepCell eval |
| Per-epoch time (masked) | 3.6 min/epoch | sacct |
| Per-epoch time (dense flash) | 3.5 min/epoch | sacct |

---

## 8. Acceptance Criteria

The vault is complete when:

1. **All 8 research questions** are documented in `questions/` with answer, evidence links, and status.
2. **All 12 experiments** are documented in `evidence/` with setup, results, figures, and interpretation.
3. **All 8 conclusions** are documented in `conclusions/` with evidence links and implications.
4. **All 11 context notes** are documented in `context/` with summaries and key details.
5. **All attachments are in `attachments/`** — figures, PDFs, CSVs, PNGs.
6. **The MOC (`00-MOC.md`)** links to all sections and includes a Mermaid graph.
7. **All notes have YAML frontmatter** with tags, status, and date.
8. **All notes use `[[wikilinks]]`** to connect related concepts.
9. **All numbers are verified** — 1.78×, 2.2%, N≈166, ΔTRA=-0.0003, ΔAOGM=+4.0, etc.
10. **The vault opens in Obsidian** without errors and the graph view shows the full knowledge network.

---

## 9. Implementation Plan

### Phase 1: Vault Scaffolding (15 min)

1. Create `vault/` directory structure
2. Add `vault/` to `.gitignore`
3. Create note templates in `templates/`
4. Copy all attachments to `attachments/`

### Phase 2: Context Notes (20 min)

1. Create 11 context notes in `context/`
2. Pull from `knowledge-graph/nodes/` and report sections
3. Add YAML frontmatter and wikilinks

### Phase 3: Evidence Notes (30 min)

1. Create 12 evidence notes in `evidence/`
2. Pull from report tables, benchmark CSVs, and HPC data
3. Embed figures using `![[attachment_name.png]]`
4. Add interpretation linking to questions and conclusions

### Phase 4: Question Notes (20 min)

1. Create 8 question notes in `questions/`
2. Pull from `knowledge-graph/nodes/Research-Questions.md` and report sections
3. Add answer, evidence links, and status
4. Link to related questions and conclusions

### Phase 5: Conclusion Notes (20 min)

1. Create 8 conclusion notes in `conclusions/`
2. Pull from report conclusion section and key findings
3. Add finding, evidence links, implications
4. Link to related conclusions and questions

### Phase 6: MOC and Final Review (15 min)

1. Create `00-MOC.md` with project summary, Mermaid graph, and links to all sections
2. Verify all acceptance criteria
3. Verify all wikilinks resolve
4. Open in Obsidian and verify graph view

**Total estimated time: 2 hours**

---

## 10. Out of Scope

- The vault does not replace the LaTeX report.
- The vault does not replace the code repository.
- The vault does not include executable code.
- The vault does not include raw data (only metadata and statistics).
- The vault does not include the `third_party/` or `trackastra/` source code.
- The vault does not include every labbook entry (only distilled findings).
