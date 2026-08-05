# Agent Prompt: Validate Mask-KNN vs Gather-KNN Output Equivalence

You are a fresh-context agent. Your task is to prove (or disprove) that
`KNNMaskSparseAttention` and `GatherSparseAttention` produce approximately identical outputs
and gradients, given the same inputs and KNN indices. If proven, mask-KNN inherits
gather-KNN's real-data tracking accuracy without retraining.

---

## STEP 1: Understand both implementations

Read the relevant classes in:
- `/home/leonard.starke@mediainterface.de/Dokumente/Uni/research-proj/benchmark_attn/model_parts.py`

Key classes:
- `KNNMaskSparseAttention` (line ~482) — scatter-based, builds Boolean mask, runs SDPA with `attn_mask`
- `GatherSparseAttention` (line ~273) — gather-based, gathers K_sel/V_sel, runs SDPA without mask
- Also read `GatherSparseAttentionV2` (line ~376) and `GatherSparseAttentionV3` (line ~583) for completeness

Understand:
1. How each constructs its attention arguments (Q, K, V shapes, mask, gather indices)
2. The KNN index format: `idx ∈ Z^{B × nH × N × K}`
3. The SDPA dispatch path each triggers (EFFICIENT_ATTENTION vs CUDNN_ATTENTION)
4. How the output is reshaped back to `(B, nH, N, Dh)`

---

## STEP 2: Write the validation script

Create a new file: `/home/leonard.starke@mediainterface.de/Dokumente/Uni/research-proj/benchmark_attn/validate_mask_vs_gather.py`

Requirements for the script:

### 2.1 Imports and setup
```python
import torch
import torch.nn.functional as F
from model_parts import KNNMaskSparseAttention, GatherSparseAttention

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16  # Use fp16 as that's what the benchmarks use
```

### 2.2 Test parameters
Test across a grid:
- `N ∈ [128, 256, 512]` (sequence lengths)
- `K ∈ [4, 16]` (KNN neighbors)
- `B = 2, nH = 4, Dh = 64, d_model = 256`
- At least 3 random seeds per config to check stability

### 2.3 Model construction
- Instantiate ONE `KNNMaskSparseAttention` and ONE `GatherSparseAttention` with the same `d_model, nH`
- Both MUST be in eval mode (`.eval()`) to disable dropout
- Both MUST use the same dtype and device
- **Important:** Initialize both models with the same weights. The simplest way:
  - Create one instance, then use its `state_dict()` to load into the other
  - Or manually set identical QKV projection weights via `copy_()`

### 2.4 Forward pass comparison
For each (N, K, seed):

1. Generate random input: `x = torch.randn(B, N, d_model, device=device, dtype=dtype)`
2. Generate KNN indices: `idx = torch.randint(0, N, (B, nH, N, K), device=device)`  
   (real KNN uses spatial cdist+topk, but for equivalence testing random indices suffice)
3. Run BOTH models on the SAME x and SAME idx
4. Compare outputs:
   - `torch.allclose(out_mask, out_gather, rtol=1e-2, atol=1e-3)` — relaxed for fp16
   - Report max absolute difference: `(out_mask - out_gather).abs().max().item()`
   - Report mean relative difference
   - Compute cosine similarity between flattened outputs
   - Print a clear PASS/FAIL per config

### 2.5 Backward pass comparison
For each (N, K, seed):

1. Use `x.requires_grad_(True)` as input (or create a separate learnable parameter)
2. Run both models forward
3. Sum the outputs and call `.backward()` on each
4. Compare input gradients (from the QKV projection):
   - The simplest: compare `.weight.grad` of the QKV projection layers
   - Or feed identical x with `requires_grad=True` and compare `x.grad`
5. Use relaxed tolerances for fp16 gradients (rtol=1e-1, atol=1e-2)
6. Report max absolute gradient difference, PASS/FAIL

### 2.6 Output format
Print a structured report:

```
========================================
Mask-KNN vs Gather-KNN Equivalence Test
========================================
dtype: float16, device: cuda

Configuration         Forward    Backward   Max |Δ|     Status
-----------------------------------------------------------------
N=128 K=4  seed=0     PASS       PASS       1.2e-04     EQUIVALENT
N=128 K=4  seed=1     PASS       PASS       8.9e-05     EQUIVALENT
N=128 K=16 seed=0     PASS       PASS       2.1e-04     EQUIVALENT
N=256 K=4  seed=0     PASS       PASS       1.8e-04     EQUIVALENT
N=256 K=16 seed=0     PASS       PASS       3.4e-04     EQUIVALENT
N=512 K=4  seed=0     PASS       PASS       2.7e-04     EQUIVALENT
N=512 K=16 seed=0     PASS       PASS       5.2e-04     EQUIVALENT

Verdict: ALL CONFIGURATIONS PASS - Mask-KNN and Gather-KNN are
approximately equivalent in both forward and backward passes.
Mask-KNN inherits Gather-KNN's tracking accuracy.
```

If any config fails, print detailed diagnostic info for that config.

---

## STEP 3: Run and verify

Execute the script:
```bash
python /home/leonard.starke@mediainterface.de/Dokumente/Uni/research-proj/benchmark_attn/validate_mask_vs_gather.py
```

If CUDA is unavailable, use CPU (but note fp16 may be less accurate on CPU — document this).

---

## STEP 4: Interpret and document

After running, check the output and write a brief analysis:

1. **Did all configs pass?** If yes, mask-KNN is validated as equivalent.
2. **Failure modes:** If some configs fail, is it at specific (N, K) combinations? Larger K? Larger N?
3. **Tolerance sensitivity:** If forward and backward use different tolerances, check whether failures are "borderline" (e.g., max diff is 1.5× atol) or "catastrophic" (e.g., orders of magnitude off).
4. **Theoretical expectation:** Both implementations should be identical up to fp16 kernel rounding. EFFICIENT_ATTENTION and CUDNN_ATTENTION might differ by a few ULPs. The `allclose` tolerances above should accommodate this.

---

## STEP 5: Report back

Return the COMPLETE output of the validation script and your analysis verdict:
- "EQUIVALENT" — mask-KNN inherits gather-KNN's accuracy
- "APPROXIMATELY EQUIVALENT" — borderline tolerances, recommend tighter check
- "NOT EQUIVALENT" — there's a real mathematical difference, cannot inherit accuracy

If NOT EQUIVALENT, investigate WHY and report the specific difference found.
