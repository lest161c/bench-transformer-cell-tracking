"""Validate that KNNMaskSparseAttention and GatherSparseAttention are equivalent.

Tests forward pass equivalence (fp16, rtol=1e-2, atol=1e-3) and
backward pass gradient equivalence across multiple seeds, sequence
lengths, and KNN neighbor counts.

If all configurations pass, Mask-KNN inherits Gather-KNN's tracking
accuracy and the two can be used interchangeably.

Usage::

    python validate_mask_vs_gather.py

Requires CUDA.  Outputs a summary table to stdout.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.attention_modules import KNNMaskSparseAttention, GatherSparseAttention

FW_RTOL = 1e-2
FW_ATOL = 1e-3
BW_RTOL = 1e-1
BW_ATOL = 1e-2

SEEDS = [0, 1, 2]
NS = [128, 256, 512]
KS = [4, 16]
B = 2
NH = 4
DH = 64
D = NH * DH  # 256

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

print(f"device={device}, dtype={dtype}")
print(f"FW tolerances: rtol={FW_RTOL}, atol={FW_ATOL}")
print(f"BW tolerances: rtol={BW_RTOL}, atol={BW_ATOL}")

all_pass = True
results = []

for N in NS:
    for K in KS:
        for seed in SEEDS:
            torch.manual_seed(seed)

            mask_model = KNNMaskSparseAttention(
                embed_dim=D, n_head=NH, knn_neighbors=K, mode="none"
            ).to(device=device, dtype=dtype).eval()

            gather_model = GatherSparseAttention(
                embed_dim=D, n_head=NH, knn_neighbors=K, mode="none"
            ).to(device=device, dtype=dtype).eval()

            gather_model.load_state_dict(mask_model.state_dict())

            x = torch.randn(B, N, D, device=device, dtype=dtype)
            # Unique KNN indices (no duplicates per query row).
            # Real KNN via topk produces unique neighbors; randint allows
            # duplicates which causes mask vs gather divergence.
            rand_vals = torch.rand(B, N, N, device=device)
            idx = rand_vals.argsort(dim=-1)[..., :K]  # (B, N, K) unique

            # ---- forward pass ----
            with torch.no_grad():
                out_mask = mask_model(x, x, x, knn_indices=idx)
                out_gather = gather_model(x, x, x, knn_indices=idx)

            fw_ok = torch.allclose(out_mask, out_gather, rtol=FW_RTOL, atol=FW_ATOL)
            max_delta = (out_mask - out_gather).abs().max().item()
            mean_delta = (out_mask - out_gather).abs().mean().item()

            # cosine similarity
            cos = torch.nn.functional.cosine_similarity(
                out_mask.flatten(), out_gather.flatten(), dim=0
            ).item()

            # ---- backward pass ----
            x2 = torch.randn(B, N, D, device=device, dtype=dtype, requires_grad=True)

            mask_model2 = KNNMaskSparseAttention(
                embed_dim=D, n_head=NH, knn_neighbors=K, mode="none"
            ).to(device=device, dtype=dtype)

            gather_model2 = GatherSparseAttention(
                embed_dim=D, n_head=NH, knn_neighbors=K, mode="none"
            ).to(device=device, dtype=dtype)

            # copy weights from eval model (same seed gives same init)
            torch.manual_seed(seed)
            _tmp = KNNMaskSparseAttention(
                embed_dim=D, n_head=NH, knn_neighbors=K, mode="none"
            ).to(device=device, dtype=dtype)
            mask_model2.load_state_dict(_tmp.state_dict())
            gather_model2.load_state_dict(_tmp.state_dict())
            del _tmp

            rand_vals2 = torch.rand(B, N, N, device=device)
            idx2 = rand_vals2.argsort(dim=-1)[..., :K]  # (B, N, K) unique
            out_mask2 = mask_model2(x2, x2, x2, knn_indices=idx2)
            out_gather2 = gather_model2(x2, x2, x2, knn_indices=idx2)

            out_mask2.sum().backward()
            out_gather2.sum().backward()

            # compare gradients of QKV weights
            # Use norm-based: only check params with significant gradient.
            # Softmax cancels key bias so k_pro.bias has near-zero gradient.
            grad_pass = True
            grad_max_rel = 0.0
            grad_max_delta = 0.0
            GRAD_MIN_NORM = 1.0  # skip params with negligible gradient
            for (nm, pm), (ng, pg) in zip(
                mask_model2.named_parameters(), gather_model2.named_parameters()
            ):
                if pm.grad is None or pg.grad is None:
                    continue
                gd = pm.grad.float() - pg.grad.float()
                gdiff_norm = gd.norm().item()
                gnorm = pm.grad.float().norm().item()
                rel_err = gdiff_norm / max(gnorm, 1e-8)
                if gnorm > GRAD_MIN_NORM and rel_err > grad_max_rel:
                    grad_max_rel = rel_err
                    grad_max_delta = gd.abs().max().item()
                if gnorm > GRAD_MIN_NORM and rel_err > BW_RTOL:
                    grad_pass = False

            bw_label = "PASS" if grad_pass else "FAIL"
            fw_label = "PASS" if fw_ok else "FAIL"
            status = "EQUIVALENT" if (fw_ok and grad_pass) else "MISMATCH"

            results.append((N, K, seed, fw_label, bw_label, max_delta, grad_max_delta, grad_max_rel,
                            mean_delta, cos, status))

            # cleanup
            del mask_model, gather_model, mask_model2, gather_model2, x, x2, idx
            del out_mask, out_gather, out_mask2, out_gather2

torch.cuda.empty_cache()


def main():
    """Run the mask vs gather equivalence validation and print results."""
    print()
    print("=" * 84)
    print("Mask-KNN vs Gather-KNN Equivalence Test")
    print("=" * 84)
    print(f"dtype: float16, device: {device}")
    print()

    header = f"{'Configuration':<20} {'FW':<6} {'BW':<6} {'Max |d|':<12} {'GradRel':<10} {'Mean|d|':<12} {'CosSim':<8} {'Verdict'}"
    print(header)
    print("-" * 88)
    for (N, K, seed, fw, bw, fw_delta, bw_delta, bw_rel, mean_d, cos, status) in results:
        cfg = f"N={N:<4d} K={K:<2d} seed={seed}"
        print(f"{cfg:<20} {fw:<6} {bw:<6} {fw_delta:<12.2e} {bw_rel:<10.2e} {mean_d:<12.2e} {cos:<8.4f} {status}")

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
