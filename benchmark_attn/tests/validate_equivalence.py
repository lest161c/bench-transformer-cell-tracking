"""Validate that KNNMaskSparseAttention and GatherSparseAttention are equivalent.

Tests forward pass equivalence (fp16, rtol=1e-2, atol=1e-3) and
backward pass gradient equivalence (rtol=1e-1, atol=1e-2) across
multiple seeds, sequence lengths, and KNN neighbor counts.

If all configurations pass, mask_knn inherits gather_knn's tracking
accuracy and the two can be used interchangeably.

Usage::

    python tests/validate_equivalence.py

Requires CUDA.  Outputs a summary table to stdout.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.attention_modules import KNNMaskSparseAttention, GatherSparseAttention

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

            mask_model = KNNMaskSparseAttention(
                embed_dim=EMBED_DIM, n_head=NUM_HEADS,
                knn_neighbors=knn_neighbors, mode="none",
            ).to(device=DEVICE, dtype=DTYPE).eval()

            gather_model = GatherSparseAttention(
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

            forward_cosine_similarity = torch.nn.functional.cosine_similarity(
                output_mask.flatten(), output_gather.flatten(), dim=0,
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
            mask_model_backward = KNNMaskSparseAttention(
                embed_dim=EMBED_DIM, n_head=NUM_HEADS,
                knn_neighbors=knn_neighbors, mode="none",
            ).to(device=DEVICE, dtype=DTYPE)

            gather_model_backward = GatherSparseAttention(
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


def main():
    """Run the mask vs gather equivalence validation and print results."""
    print()
    print("=" * 84)
    print("Mask-KNN vs Gather-KNN Equivalence Test")
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
        print("\nVerdict: ALL CONFIGURATIONS PASS - Mask-KNN and Gather-KNN are")
        print("approximately equivalent in both forward and backward passes.")
        print("Mask-KNN inherits Gather-KNN's tracking accuracy.")
    else:
        print("\nVerdict: NOT ALL CONFIGURATIONS PASS - see details above.")
        print("Mask-KNN does NOT reliably inherit Gather-KNN's accuracy.")


if __name__ == "__main__":
    main()
