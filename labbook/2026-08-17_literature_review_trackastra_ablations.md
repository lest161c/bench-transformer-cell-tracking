# Literature Review: Trackastra Window Size and Spatial Cutoff Sweeps

**Date:** 2026-08-17
**Status:** COMPLETE
**Source:** Gallusser & Weigert (2024), *Trackastra: Transformer-based cell tracking for live-cell microscopy.* ECCV 2024. arXiv:2405.15700v2.

---

## 1. Window Size Sweep

**Do the Trackastra authors sweep window size?** YES.

Section 3.6 ("Ablations"), subsection "Window size" (Fig. 4c):

> We run experiments with different window sizes s, i.e. different temporal
> context available to the model. When using only s=2 we find that the
> performance substantially deteriorates, whereas window sizes s∈(3,6) all
> lead to good results, demonstrating that a relatively small temporal context
> is already enough for decent tracking results.

**Key findings:**

| Window size s | TRA | Notes |
|:---:|:---:|:---|
| 2 | ~0.97 | Too short, performance deteriorates |
| 3 | ~0.99 | Good results |
| 4 | ~0.99 | Good results |
| 5 | ~0.99 | Good results |
| 6 | ~0.999 | Best (paper default) |

**Paper default:** s=6 (Section 2.5: "We set window size s=6")

**Code default:** window=10 (train.py line 1232)

**Discrepancy:** The paper uses s=6, but the code defaults to s=10. The
ablation only tests s=2 through s=6, so s=10 is beyond the tested range.
However, the paper also mentions "the maximum number of tokens per window
|D|=2048", which suggests the window is capped by max_tokens.

**Our experiments:** We use window=4 (in the good range s∈(3,6)).
This is consistent with the paper's finding that s=4 gives good results.

---

## 2. Spatial Cutoff Sweep

**Do the Trackastra authors sweep spatial cutoff?** NO.

The spatial cutoff d_max is mentioned in the attention mask (Eq. 3):

> M_ij = 0 if ||p_i - p_j||_2 ≤ d_max and M_ij = -∞ otherwise

The paper describes d_max as "a user defined threshold" but does not report
any sensitivity analysis for it. The spatial cutoff is used as a fixed
hyperparameter throughout all experiments.

**Default value:** d_max=256 pixels (from code: spatial_pos_cutoff=256)

**No ablation reported** for:
- Different d_max values (e.g., 128, 256, 512)
- Removing the spatial cutoff entirely (dense_flash)
- The interaction between d_max and window size

**This is a gap our research addresses:** The dense_flash ablation
(job 3920471) tests whether removing the spatial cutoff degrades TRA.

---

## 3. Transformer Size Ablation

**Do the Trackastra authors sweep transformer size?** YES.

Section 3.6, "Transformer size" (Fig. 4a,b):

> As expected, using no attention layers (L=0), i.e. directly predicting the
> association matrix with the two projection heads leads to poor performance,
> confirming the importance of attention across all detections. Furthermore,
> using more than L=6 layers does not lead to a notable improvement.

**Key findings:**

| Layers L | TRA | Notes |
|:---:|:---:|:---|
| 0 | ~0.90 | No attention, poor performance |
| 2 | ~0.98 | Good results |
| 4 | ~0.99 | Good results |
| 6 | ~0.999 | Best (paper default) |
| 8 | ~0.999 | No improvement beyond L=6 |
| 10 | ~0.999 | No improvement beyond L=6 |

**Paper default:** L=6 (6 encoder + 6 decoder = 12 attention layers)

**Our experiments:** We use 6 encoder + 6 decoder layers, matching the paper.

---

## 4. Other Ablations (Table 6)

The paper also ablates:

| Component | TRA without | TRA with | Notes |
|-----------|:-----------:|:--------:|-------|
| Relative pos. encoding | 0.989 | 0.995 | RoPE improves tracking |
| Object features | 0.992 | 0.999 | Shallow features (area, intensity) help |
| ILP linking | 0.995 (greedy) | 0.999 (ILP) | ILP improves over greedy |
| Parental softmax | 0.989 | 0.999 | Reduces AOGM by ~20% |

---

## 5. Implementation Details (Section 2.5)

From the paper:

> Trackastra can be trained on a single GPU (e.g. Nvidia RTX 4090) for
> prototypical 2D cell tracking datasets. The transformer is implemented in
> PyTorch. We set window size s=6, embedding dimension d=256, number of
> encoder and decoder attention layers L=6, the maximum number of tokens per
> window |D|=2048, and batch size 8.

**Our config:** d_model=320, nhead=8, 6+6 layers, window=4, max_tokens=2048, batch_size=48

**Differences:**
- d_model: 320 (ours) vs 256 (paper) — we use a slightly larger model
- window: 4 (ours) vs 6 (paper) — both in the good range
- batch_size: 48 (ours) vs 8 (paper) — we use larger batches
- max_tokens: 2048 (both) — same cap

---

## 6. Implications for Our Research

1. **Window size:** The paper confirms s=4 is in the good range. No need
   to increase window beyond s=6 for accuracy reasons.

2. **Spatial cutoff:** The paper does NOT sweep d_max. This is a genuine
   research gap. Our dense_flash ablation fills this gap.

3. **Transformer size:** The paper confirms L=6 is optimal. Our 6+6 layer
   choice is correct.

4. **Attention mechanism:** The paper uses dense attention with a spatial
   cutoff mask (d_max=256). It does NOT test sparse attention (KNN gather
   or KNN mask). This is our contribution.

5. **K-sweep:** The paper does NOT test different K values for sparse
   attention. Our K-sweep fills this gap.

---

## 7. Summary

| Question | Answer |
|----------|--------|
| Do Trackastra authors sweep window size? | YES — s=2,3,4,5,6 tested, s∈(3,6) all good |
| Do Trackastra authors sweep spatial cutoff? | NO — d_max=256 fixed, no sensitivity analysis |
| Do Trackastra authors sweep transformer size? | YES — L=0,2,4,6,8,10 tested, L=6 optimal |
| Do Trackastra authors test sparse attention? | NO — only dense attention with spatial cutoff |
| Do Trackastra authors test K values? | NO — no KNN sparse attention experiments |

**Our research contributions:**
1. **Sparse attention for cell tracking** — testing KNN mask + FlashAttention
2. **Spatial cutoff ablation** — does removing d_max degrade TRA?
3. **K-sweep** — which K (4, 16, 32, 64) gives best TRA/AOGM?
4. **Window size at high N** — multi-condition batching for N≥2048
