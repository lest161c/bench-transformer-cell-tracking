"""Validate that SparseMaskAttention and SparseGatherAttention are equivalent.

Tests forward pass equivalence (fp16, rtol=1e-2, atol=1e-3) and
backward pass gradient equivalence (rtol=1e-1, atol=1e-2) across
multiple seeds, sequence lengths, and KNN neighbor counts.

If all configurations pass, sparse_mask inherits gather_knn's tracking
accuracy and the two can be used interchangeably.

Note: the forward cosine-similarity metric is computed in fp32 (the
fp16 outputs are upcast first); an fp16 metric can round above 1.0
(Cauchy-Schwarz), overstating equivalence.

Usage::

    python tests/validate_equivalence.py

Requires CUDA.  Outputs a summary table to stdout.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from src.attention_modules import (
    SparseGatherAttention,
    SparseMaskAttention,
    SparseSoftmaxAttention,
    CachedDistAttention,
    ATTN_IGNORE_VALUE,
)

try:
    from torch.nn.attention.flex_attention import (
        flex_attention,
        create_block_mask,
    )
    _HAS_FLEX = True
except ImportError:
    _HAS_FLEX = False

# Tolerances for forward (output) and backward (gradient) comparison.
# Forward is tight: the two methods compute the same softmax over the
# same gathered/masked keys, so differences are purely fp16 rounding.
# Backward is looser: gradient computation involves different graph
# structures (gather vs scatter_mask), so more numerical drift accrues.
FORWARD_RTOL = 1e-2
FORWARD_ATOL = 1e-3
BACKWARD_RTOL = 1e-1
BACKWARD_ATOL = 1e-2

# Parameters with gradient norm below this threshold are skipped during
# backward comparison.  Some parameters (e.g. k_pro.bias) receive
# near-zero gradients because softmax cancels the key bias contribution;
# comparing noise against noise produces spurious failures.
GRAD_NORM_THRESHOLD = 1.0

SEEDS = [0, 1, 2]
SEQUENCE_LENGTHS = [128, 256, 512]
KNN_NEIGHBOR_COUNTS = [4, 16]
BATCH_SIZE = 2
NUM_HEADS = 4
HEAD_DIM = 64
EMBED_DIM = NUM_HEADS * HEAD_DIM  # 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16

print(f"device={DEVICE}, dtype={DTYPE}")
print(f"forward tolerances: rtol={FORWARD_RTOL}, atol={FORWARD_ATOL}")
print(f"backward tolerances: rtol={BACKWARD_RTOL}, atol={BACKWARD_ATOL}")

all_pass = True
results = []

for seq_len in SEQUENCE_LENGTHS:
    for knn_neighbors in KNN_NEIGHBOR_COUNTS:
        for seed in SEEDS:
            torch.manual_seed(seed)

            mask_model = SparseMaskAttention(
                embed_dim=EMBED_DIM, n_head=NUM_HEADS,
                knn_neighbors=knn_neighbors, mode="none",
            ).to(device=DEVICE, dtype=DTYPE).eval()

            gather_model = SparseGatherAttention(
                embed_dim=EMBED_DIM, n_head=NUM_HEADS,
                knn_neighbors=knn_neighbors, mode="none",
            ).to(device=DEVICE, dtype=DTYPE).eval()

            # Copy weights so both models start from identical parameters.
            gather_model.load_state_dict(mask_model.state_dict())

            input_tokens = torch.randn(
                BATCH_SIZE, seq_len, EMBED_DIM, device=DEVICE, dtype=DTYPE,
            )
            # Generate unique KNN indices per query.
            # randint produces duplicates which cause mask vs gather
            # divergence; argsort of rand gives guaranteed-unique rows.
            random_values = torch.rand(BATCH_SIZE, seq_len, seq_len, device=DEVICE)
            knn_indices = random_values.argsort(dim=-1)[..., :knn_neighbors]

            # ---- forward pass ----
            with torch.no_grad():
                output_mask = mask_model(
                    input_tokens, input_tokens, input_tokens,
                    knn_indices=knn_indices,
                )
                output_gather = gather_model(
                    input_tokens, input_tokens, input_tokens,
                    knn_indices=knn_indices,
                )

            forward_passes = torch.allclose(
                output_mask, output_gather,
                rtol=FORWARD_RTOL, atol=FORWARD_ATOL,
            )
            forward_max_delta = (
                output_mask - output_gather
            ).abs().max().item()
            forward_mean_delta = (
                output_mask - output_gather
            ).abs().mean().item()

            # Cosine is computed in fp32: an fp16 metric can round above 1.0
            # (Cauchy-Schwarz), overstating equivalence.  Upcast removes the
            # metric artifact without changing the fp16 outputs themselves.
            forward_cosine_similarity = torch.nn.functional.cosine_similarity(
                output_mask.flatten().float(), output_gather.flatten().float(), dim=0,
            ).item()

            # ---- backward pass ----
            # Re-instantiate with requires_grad inputs to compare gradients.
            # We re-seed and re-init to get the same starting weights as the
            # forward models above.
            backward_input = torch.randn(
                BATCH_SIZE, seq_len, EMBED_DIM,
                device=DEVICE, dtype=DTYPE, requires_grad=True,
            )

            torch.manual_seed(seed)
            mask_model_backward = SparseMaskAttention(
                embed_dim=EMBED_DIM, n_head=NUM_HEADS,
                knn_neighbors=knn_neighbors, mode="none",
            ).to(device=DEVICE, dtype=DTYPE)

            gather_model_backward = SparseGatherAttention(
                embed_dim=EMBED_DIM, n_head=NUM_HEADS,
                knn_neighbors=knn_neighbors, mode="none",
            ).to(device=DEVICE, dtype=DTYPE)

            # Copy weights so backward gradients start from same parameters.
            mask_model_backward.load_state_dict(
                mask_model.state_dict()
            )
            gather_model_backward.load_state_dict(
                gather_model.state_dict()
            )

            backward_random_values = torch.rand(
                BATCH_SIZE, seq_len, seq_len, device=DEVICE,
            )
            backward_knn_indices = backward_random_values.argsort(dim=-1)[
                ..., :knn_neighbors
            ]

            output_mask_backward = mask_model_backward(
                backward_input, backward_input, backward_input,
                knn_indices=backward_knn_indices,
            )
            output_gather_backward = gather_model_backward(
                backward_input, backward_input, backward_input,
                knn_indices=backward_knn_indices,
            )

            output_mask_backward.sum().backward()
            output_gather_backward.sum().backward()

            # Compare gradients parameter-by-parameter.
            # Only check parameters with significant gradient norm.
            backward_passes = True
            backward_max_rel_error = 0.0
            backward_max_grad_delta = 0.0
            for (mask_name, mask_param), (gather_name, gather_param) in zip(
                mask_model_backward.named_parameters(),
                gather_model_backward.named_parameters(),
            ):
                if mask_param.grad is None or gather_param.grad is None:
                    continue
                grad_difference = (
                    mask_param.grad.float() - gather_param.grad.float()
                )
                grad_diff_norm = grad_difference.norm().item()
                grad_norm = mask_param.grad.float().norm().item()
                relative_error = grad_diff_norm / max(grad_norm, 1e-8)
                if grad_norm > GRAD_NORM_THRESHOLD and relative_error > backward_max_rel_error:
                    backward_max_rel_error = relative_error
                    backward_max_grad_delta = grad_difference.abs().max().item()
                if grad_norm > GRAD_NORM_THRESHOLD and relative_error > BACKWARD_RTOL:
                    backward_passes = False

            forward_label = "PASS" if forward_passes else "FAIL"
            backward_label = "PASS" if backward_passes else "FAIL"
            status = "EQUIVALENT" if (forward_passes and backward_passes) else "MISMATCH"

            results.append((
                seq_len, knn_neighbors, seed,
                forward_label, backward_label,
                forward_max_delta, backward_max_grad_delta,
                backward_max_rel_error,
                forward_mean_delta, forward_cosine_similarity,
                status,
            ))

            # Cleanup
            del (
                mask_model, gather_model,
                mask_model_backward, gather_model_backward,
                input_tokens, backward_input,
                knn_indices, backward_knn_indices,
                output_mask, output_gather,
                output_mask_backward, output_gather_backward,
            )

torch.cuda.empty_cache()

# ═══ Flex vs dense masked SDPA equivalence (SPEC 0024 T4, demo Section 4) ═══
# Mirrors exploratory/flex_attention_demo.py Section 4: the flex path
# (create_block_mask + compiled flex_attention) vs the dense reference
# (masked SDPA with additive float mask: -lam*distance inside the spatial
# cutoff, -BIG beyond). Forward AND backward, tolerances fwd 2e-3 / bwd
# 1e-2. The constants LAM=0.1 and BIG=65504.0 mirror FlexAttentionModule's
# hardcodes (lines ~1836-1841 of src/attention_modules.py).

LAM = 0.1  # spatial decay strength (matches demo LAM)
BIG = 65504.0  # fp16 min finite; exp(-BIG) ≈ 0 in softmax


def check_flex_vs_dense():
    """Check flex (block-mask path) vs dense masked SDPA equivalence.

    Builds the toy problem from the demo (N=128, 2D coords in [0, 512),
    seed 42, fp16), runs the flex path via the dedicated flex API
    (create_block_mask + compiled flex_attention with score_mod and
    block_mask) and the dense path via F.scaled_dot_product_attention with
    an additive mask, then compares forward outputs and backward
    gradients. Tolerances: forward max|Δ| < 2e-3, backward max|Δ| < 1e-2.

    Returns:
        True if both forward and backward pass, False otherwise.
    """
    torch.manual_seed(42)

    n_tokens = 128
    d_head = 40
    cutoff = 256.0

    coords = torch.rand(n_tokens, 2, device=DEVICE) * 512.0
    pairwise = torch.cdist(coords, coords)

    def score_mod(score, b, h, q_idx, kv_idx):
        distance = pairwise[q_idx, kv_idx]
        return torch.where(distance <= cutoff, score - LAM * distance, score - BIG)

    def mask_mod(b, h, q_idx, kv_idx):
        return pairwise[q_idx, kv_idx] <= cutoff

    block_mask = create_block_mask(mask_mod, None, None, n_tokens, n_tokens, device=DEVICE)

    query = torch.randn(1, 8, n_tokens, d_head, device=DEVICE, dtype=DTYPE)
    key = torch.randn(1, 8, n_tokens, d_head, device=DEVICE, dtype=DTYPE)
    value = torch.randn(1, 8, n_tokens, d_head, device=DEVICE, dtype=DTYPE)

    query_flex = query.detach().clone().requires_grad_(True)
    key_flex = key.detach().clone().requires_grad_(True)
    value_flex = value.detach().clone().requires_grad_(True)

    compiled_flex = torch.compile(flex_attention, dynamic=False)
    # Warmup: triggers Triton JIT compilation before the measured path.
    for _ in range(3):
        _ = compiled_flex(query_flex, key_flex, value_flex,
                           score_mod=score_mod, block_mask=block_mask)
    torch.cuda.synchronize()

    output_flex = compiled_flex(query_flex, key_flex, value_flex,
                                score_mod=score_mod, block_mask=block_mask)
    output_flex.sum().backward()

    query_dense = query.detach().clone().requires_grad_(True)
    key_dense = key.detach().clone().requires_grad_(True)
    value_dense = value.detach().clone().requires_grad_(True)

    additive_mask = torch.where(
        pairwise <= cutoff,
        -LAM * pairwise,
        torch.full_like(pairwise, float("-inf")),
    ).to(DTYPE)

    output_dense = F.scaled_dot_product_attention(query_dense, key_dense, value_dense,
                                                    attn_mask=additive_mask)
    output_dense.sum().backward()

    diff_forward = (output_flex - output_dense).abs().max().item()
    diff_grad_query = (query_flex.grad - query_dense.grad).abs().max().item()
    diff_grad_key = (key_flex.grad - key_dense.grad).abs().max().item()
    diff_grad_value = (value_flex.grad - value_dense.grad).abs().max().item()

    forward_pass = diff_forward < 2e-3
    backward_pass = max(diff_grad_query, diff_grad_key, diff_grad_value) < 1e-2
    equivalence_pass = forward_pass and backward_pass

    print(f"  forward max|Δ| = {diff_forward:.6f}  (tol = 2e-3)")
    print(f"  backward q max|Δ| = {diff_grad_query:.6f}  (tol = 1e-2)")
    print(f"  backward k max|Δ| = {diff_grad_key:.6f}  (tol = 1e-2)")
    print(f"  backward v max|Δ| = {diff_grad_value:.6f}  (tol = 1e-2)")
    print(f"  equivalence: {'PASS' if equivalence_pass else 'FAIL'}")

    return equivalence_pass


def check_sparse_softmax_vs_sparse_mask():
    """Check SparseSoftmaxAttention vs SparseMaskAttention equivalence.

    N=128, K=16, fp16, seed 42.  Same knn_indices fed to both
    SparseSoftmaxAttention and SparseMaskAttention (identical
    weights via load_state_dict — both have q_pro/k_pro/v_pro/proj);
    compare forward output AND backward grads (q/k/v); tolerances fwd
    2e-3, bwd 1e-2; print PASS/FAIL with actual max|Δ|; return bool.

    They are mathematically identical: softmax over the same K set.

    Returns:
        True if both forward and backward pass, False otherwise.
    """
    torch.manual_seed(42)

    n_tokens = 128
    knn_neighbors = 16
    embed_dim = 64
    n_head = 4

    # Build both models with identical weights.
    mask_model = SparseMaskAttention(
        embed_dim, n_head, knn_neighbors, mode="none",
    ).to(device=DEVICE, dtype=DTYPE)
    sparse_model = SparseSoftmaxAttention(
        embed_dim, n_head, knn_neighbors, mode="none",
    ).to(device=DEVICE, dtype=DTYPE)
    # Copy weights so both models start from identical parameters.
    sparse_model.load_state_dict(mask_model.state_dict())

    # Shared inputs.
    query = torch.randn(1, n_tokens, embed_dim, device=DEVICE, dtype=DTYPE)
    key = torch.randn(1, n_tokens, embed_dim, device=DEVICE, dtype=DTYPE)
    value = torch.randn(1, n_tokens, embed_dim, device=DEVICE, dtype=DTYPE)

    # Generate unique KNN indices per query.
    # argsort of rand gives guaranteed-unique rows.
    random_values = torch.rand(1, n_tokens, n_tokens, device=DEVICE)
    knn_indices = random_values.argsort(dim=-1)[..., :knn_neighbors]

    # Forward through both models.
    output_mask = mask_model(query, key, value, knn_indices=knn_indices)
    output_sparse = sparse_model(query, key, value, knn_indices=knn_indices)

    # Compare forward outputs.
    diff_forward = (output_mask - output_sparse).abs().max().item()

    # Backward: separate requires_grad leaves for each model.
    # (If the same leaf were shared, gradients from both graphs would
    # accumulate and the per-model comparison would be impossible.)
    query_mask = query.detach().clone().requires_grad_(True)
    key_mask = key.detach().clone().requires_grad_(True)
    value_mask = value.detach().clone().requires_grad_(True)
    query_sparse = query.detach().clone().requires_grad_(True)
    key_sparse = key.detach().clone().requires_grad_(True)
    value_sparse = value.detach().clone().requires_grad_(True)

    output_mask_bwd = mask_model(
        query_mask, key_mask, value_mask, knn_indices=knn_indices,
    )
    output_sparse_bwd = sparse_model(
        query_sparse, key_sparse, value_sparse, knn_indices=knn_indices,
    )

    output_mask_bwd.sum().backward()
    output_sparse_bwd.sum().backward()

    # Compare gradients.
    diff_grad_query = (query_mask.grad - query_sparse.grad).abs().max().item()
    diff_grad_key = (key_mask.grad - key_sparse.grad).abs().max().item()
    diff_grad_value = (value_mask.grad - value_sparse.grad).abs().max().item()

    forward_pass = diff_forward < 2e-3
    backward_pass = max(diff_grad_query, diff_grad_key, diff_grad_value) < 1e-2
    equivalence_pass = forward_pass and backward_pass

    print(f"  forward max|Δ| = {diff_forward:.6f}  (tol = 2e-3)")
    print(f"  backward q max|Δ| = {diff_grad_query:.6f}  (tol = 1e-2)")
    print(f"  backward k max|Δ| = {diff_grad_key:.6f}  (tol = 1e-2)")
    print(f"  backward v max|Δ| = {diff_grad_value:.6f}  (tol = 1e-2)")
    print(f"  equivalence: {'PASS' if equivalence_pass else 'FAIL'}")

    return equivalence_pass


def check_cached_dist_mask_build():
    """Check CachedDistAttention._build_mask against a reference construction.

    SPEC 0045 guard: the mask-construction block was factored out of
    ``CachedDistAttention.forward`` into ``_build_mask`` so the
    ``benchmark_cudnn_vs_fa2_mask.py`` decomposition benchmark can time the
    production construction in isolation.  This check pins the refactor to
    zero behavioural change: the reference implementation below is a literal
    copy of the pre-refactor inline block and must produce bitwise-identical
    masks across dist_mode (v0/v1) x dist_cache (on/off) x positional mode
    (none/bias), including a padding-mask variant.

    Returns:
        True if every configuration is bitwise equal, False otherwise.
    """

    def reference_build_mask(module, batch_size, seq_len, coords, padding_mask, dist_2d, device, dtype):
        """Literal pre-refactor inline construction (ground truth copy)."""
        n_head = module.n_head
        attn_ignore_val = ATTN_IGNORE_VALUE

        if dist_2d is not None:
            spatial_mask = (dist_2d > module.cutoff_spatial).unsqueeze(1).expand(
                -1, n_head, -1, -1
            )
        else:
            spatial_coords = coords[..., 1:]
            spatial_dist = torch.cdist(spatial_coords, spatial_coords)
            spatial_mask = (spatial_dist > module.cutoff_spatial).unsqueeze(1)

        mask = torch.zeros(
            batch_size, n_head, seq_len, seq_len,
            device=device, dtype=dtype,
        )
        mask.masked_fill_(spatial_mask, attn_ignore_val)

        if coords is not None and module._mode == "bias":
            mask = mask + module.pos_bias(coords)

        if coords is not None:
            if module.attn_dist_mode == "v0":
                full_dist = torch.cdist(coords, coords, p=2)
                mask = mask + torch.exp(-0.1 * full_dist.unsqueeze(1))
            elif module.attn_dist_mode == "v1":
                decay_dist = dist_2d if dist_2d is not None else spatial_dist
                mask = mask + torch.exp(-5 * decay_dist.unsqueeze(1) / module.cutoff_spatial)

        if padding_mask is not None:
            ignore_mask = torch.logical_or(
                padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
            ).unsqueeze(1)
            mask.masked_fill_(ignore_mask, attn_ignore_val)

        return mask

    batch_size, seq_len, n_head, head_dim = 2, 128, 4, 64
    embed_dim = n_head * head_dim
    coord_dim = 2

    # (dist_mode, dist_cache, positional_mode, use_padding_mask) configurations.
    configurations = [
        ("v0", "on", "none", False),
        ("v0", "off", "none", False),
        ("v1", "on", "none", False),
        ("v1", "off", "none", False),
        ("v1", "on", "bias", False),
        ("v1", "on", "none", True),
    ]

    all_bitwise_equal = True
    for dist_mode, dist_cache, positional_mode, use_padding_mask in configurations:
        torch.manual_seed(42)
        cached_dist_model = CachedDistAttention(
            coord_dim, embed_dim, n_head, mode=positional_mode,
            attn_dist_mode=dist_mode,
        ).to(device=DEVICE, dtype=DTYPE).eval()

        coords = torch.randn(
            batch_size, seq_len, coord_dim + 1, device=DEVICE, dtype=DTYPE,
        )
        coords[..., 0] *= 4
        padding_mask = None
        if use_padding_mask:
            padding_mask = torch.zeros(
                batch_size, seq_len, dtype=torch.bool, device=DEVICE,
            )
            padding_mask[:, -8:] = True

        dist_2d = None
        if dist_cache == "on":
            # Same recipe as the sweep harness: fp32 cdist cast to the
            # working dtype, pre-computed outside the construction.
            dist_2d = torch.cdist(
                coords[..., 1:].float(), coords[..., 1:].float(),
            ).to(DTYPE)

        mask_factored = cached_dist_model._build_mask(
            batch_size, seq_len, coords, padding_mask, dist_2d,
            device=DEVICE, dtype=DTYPE,
        )
        mask_reference = reference_build_mask(
            cached_dist_model, batch_size, seq_len, coords, padding_mask,
            dist_2d, DEVICE, DTYPE,
        )

        bitwise_equal = torch.equal(mask_factored, mask_reference)
        all_bitwise_equal = all_bitwise_equal and bitwise_equal
        config_label = (
            f"dist={dist_mode} cache={dist_cache} mode={positional_mode}"
            f"{' pad' if use_padding_mask else ''}"
        )
        print(f"  {config_label:<44} mask max|d| = "
              f"{(mask_factored - mask_reference).abs().max().item():.2e}  "
              f"{'BITWISE-EQUAL' if bitwise_equal else 'MISMATCH'}")

    print(f"  mask-build refactor: {'PASS' if all_bitwise_equal else 'FAIL'}")
    return all_bitwise_equal


def main():
    """Run the mask vs gather equivalence validation and print results."""
    print()
    print("=" * 84)
    print("SparseMask vs SparseGather Equivalence Test")
    print("=" * 84)
    print(f"dtype: float16, device: {DEVICE}")
    print()

    header = (
        f"{'Configuration':<20} {'FW':<6} {'BW':<6} "
        f"{'Max |d|':<12} {'GradRel':<10} "
        f"{'Mean|d|':<12} {'CosSim':<8} {'Verdict'}"
    )
    print(header)
    print("-" * 88)
    for (seq_len, knn_neighbors, seed,
         forward_label, backward_label,
         forward_delta, backward_delta,
         backward_rel, mean_delta,
         cosine_sim, status) in results:
        config = f"N={seq_len:<4d} K={knn_neighbors:<2d} seed={seed}"
        print(
            f"{config:<20} {forward_label:<6} {backward_label:<6} "
            f"{forward_delta:<12.2e} {backward_rel:<10.2e} "
            f"{mean_delta:<12.2e} {cosine_sim:<8.4f} {status}"
        )

    print("-" * 88)

    pass_count = sum(1 for r in results if r[-1] == "EQUIVALENT")
    fail_count = sum(1 for r in results if r[-1] == "MISMATCH")
    print(f"\nPASS: {pass_count}/{len(results)}, FAIL: {fail_count}/{len(results)}")

    if fail_count == 0:
        print("\nVerdict: ALL CONFIGURATIONS PASS - SparseMask and SparseGather are")
        print("approximately equivalent in both forward and backward passes.")
        print("SparseMask inherits SparseGather's tracking accuracy.")
    else:
        print("\nVerdict: NOT ALL CONFIGURATIONS PASS - see details above.")
        print("SparseMask does NOT reliably inherit SparseGather's accuracy.")

    # Flex vs dense masked SDPA check (SPEC 0024 T4).
    flex_pass = check_flex_vs_dense() if _HAS_FLEX else False
    if not _HAS_FLEX:
        print("  flex API unavailable on this torch build; skipping flex check")

    # Sparse softmax vs mask KNN check.
    sparse_softmax_pass = check_sparse_softmax_vs_sparse_mask()

    # CachedDistAttention mask-build refactor guard (SPEC 0045).
    print()
    print("CachedDistAttention._build_mask refactor check (SPEC 0045)")
    print("-" * 88)
    mask_build_pass = check_cached_dist_mask_build()

    overall_pass = (fail_count == 0) and flex_pass and sparse_softmax_pass and mask_build_pass
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
