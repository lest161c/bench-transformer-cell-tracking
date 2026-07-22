# MiniMax Sparse Attention (MSA) — Negative, Excluded by Paper Analysis

**Date:** 2026-06-16
**Context:** Reviewed MiniMax MSA (arXiv:2506.xxx, MiniMax-01 team) — block-level learned sparse attention co-designed with GQA. Evaluated against our architecture requirements before any implementation.
**Depends on:** `2026-06-08_nsa_benchmark_negative.md`

---

## 1. Summary

**MSA is excluded without implementation. The architectural prerequisites are absent from Trackastra.** Unlike NSA (which at least supports MHA), MSA's core design decisions are structurally incompatible with our framework:

1. **No GQA** — MSA's per-GQA-group independent selection is its central innovation
2. **N range mismatch** — MSA's block-level selection (Bk=32–128) gives 1–16 blocks at our N=128–512
3. **No cross-attention support** — Decoder cross-attention cannot use MSA's causal sparse pattern
4. **Spatial KNN is the correct prior** — No learned indexer needed

Implementation would require: adding GQA to Trackastra, implementing the Index Branch, block-level selection, KL loss with stop-grad, two-stage warmup, and custom block-sparse kernel. All for a guaranteed negative result at our scale.

---

## 2. Architectural Incompatibility

| MSA component | Trackastra's reality | Gap |
|---|---|---|
| GQA groups (e.g., 32Q:4KV) | MHA: n_head=8, each head has own KV | **No GQA at all** |
| Block-level selection (Bk=32–128) | N=128–512 → 1–16 blocks | **Indexer selects 0 or 1 block** at N<Bk |
| Learned Index Branch (Q/K proj + KL loss) | Spatial KNN is known prior | **Learned indexer is unnecessary** |
| Two-stage warmup (full→sparse) | Standard training | **Extra complexity with zero benefit** |
| Causal sparse attention | Non-causal + cross-attention | **Pattern doesn't transfer** |
| Custom Triton kernel (block-sparse FA) | Pure PyTorch | **Kernel development effort for no gain** |
| 1M context, 109B MoE | d_model=320, N=128–512 | **3+ orders of magnitude scale gap** |

---

## 3. Why Inference from Paper Is Sufficient

The paper's own results confirm MSA operates in a completely different regime:

- **Figure 1/Table 2:** "28.4× reduction at 1M context" — N=1,000,000, not 512
- **Table 4 (ablation):** Tests block sizes Bk∈{32,64,128} — our entire N fits in 1–4 blocks
- **Section 7:** "MSA's core decisions compose with the GQA backbone shared by most current open-source frontier models" — explicitly GQA-only

No implementation is needed. The method's design domain (LLM pretraining at 100B+ scale with GQA) does not intersect with our problem (cell tracking at 5M-parameter scale with MHA and spatial priors).

NSA was at least testable (wraps `SparseAttention` with `causal=False`). MSA is not — there is no open-source implementation, and the GQA requirement means even a faithful reimplementation would need to first restructure the entire attention mechanism.

---

## 4. Decision Basis

1. **Architectural:** MSA requires GQA; Trackastra uses MHA. Core incompatibility.
2. **Scale:** At N=512 with Bk=32, only 16 blocks exist — the indexer's top-k selection is meaningless.
3. **Prior redundancy:** Spatial KNN encodes the correct inductive bias. No learned indexer needed.
4. **No cross-attention path:** Decoder cross-attention is fundamental to Trackastra.
5. **Precedent:** NSA was 29× slower than `DenseFlashAttention` at N=512 despite being architecturally compatible. MSA is even heavier (Index Branch + KL loss + warmup).
6. **Implementation cost:** Writing a faithful MSA module would take days (GQA conversion, Index Branch, block scatter/gather, KL loss). The result is predictable.

**Recommendation already covered by existing architecture guidance (architecture.md):** mask-KNN for N≤512, gather-KNN for N>2048. MSA joins NSA as not suitable for our use case.

---

## 5. Files Modified

| File | Change |
|---|---|
| `labbook/2026-06-16_msa_paper_review_negative.md` | This entry — paper review and exclusion decision |
