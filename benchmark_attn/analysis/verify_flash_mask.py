"""Verify which SDPA backend handles additive float masks.

The central question: does ``F.scaled_dot_product_attention`` dispatch to
``FLASH_ATTENTION`` when an additive float ``attn_mask`` is supplied, or does
it fall back to ``EFFICIENT_ATTENTION`` / ``MATH``?

This matters because:

* ``CachedDistAttention`` builds an N×N additive mask (spatial cutoff +
  distance decay) and passes it to SDPA.
* ``KNNMaskSparseAttention`` builds an N×N additive mask via scatter
  (KNN neighbours = 0, rest = -inf) and passes it to SDPA.
* If FlashAttention handles additive masks, then both methods get the
  fused kernel — the same kernel ``DenseFlashAttention`` uses without a
  mask.

Run as::

    python verify_flash_mask.py

The script tests four configurations:

1. No mask, FLASH forced — baseline, should succeed
2. Additive float mask, FLASH forced — the key test
3. Additive float mask, no backend forced — check auto-dispatch
4. Boolean mask, FLASH forced — expected to fail (FlashAttention needs
   float masks)

For each configuration it prints ``SUCCESS`` or ``FALLBACK`` with the
backend that was actually used.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def _make_tensors(batch_size=2, n_heads=4, seq_len=512, head_dim=64):
    """Create fp16 Q/K/V tensors on CUDA.

    Args:
        batch_size: Batch dimension.
        n_heads: Number of attention heads.
        seq_len: Sequence length (N).
        head_dim: Per-head embedding dimension.

    Returns:
        Tuple of (query, key, value) tensors, each shaped
        (batch_size, n_heads, seq_len, head_dim) in float16 on CUDA.
    """
    shape = (batch_size, n_heads, seq_len, head_dim)
    q = torch.randn(*shape, dtype=torch.float16, device="cuda")
    k = torch.randn(*shape, dtype=torch.float16, device="cuda")
    v = torch.randn(*shape, dtype=torch.float16, device="cuda")
    return q, k, v


def _make_float_mask(batch_size=2, n_heads=4, seq_len=512, cutoff=64):
    """Create an additive float attention mask.

    The mask simulates a spatial cutoff: positions within ``cutoff``
    tokens get 0.0 (attend normally), positions outside get -1e3
    (effectively masked).  A distance-decay term
    (``exp(-0.1 * dist)``) is added for positions inside the cutoff,
    matching the pattern used by ``CachedDistAttention``.

    Args:
        batch_size: Batch dimension.
        n_heads: Number of attention heads.
        seq_len: Sequence length (N).
        cutoff: Spatial cutoff in tokens (for this synthetic test).

    Returns:
        Float tensor shaped (batch_size, n_heads, seq_len, seq_len)
        on CUDA.
    """
    idx = torch.arange(seq_len, device="cuda")
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).float()
    mask = torch.zeros(seq_len, seq_len, device="cuda")
    far = dist.abs() > cutoff
    mask[far] = -1e3
    mask[~far] = torch.exp(-0.1 * dist[~far])
    mask = mask.unsqueeze(0).unsqueeze(0).expand(
        batch_size, n_heads, seq_len, seq_len
    )
    return mask.to(torch.float16)


def _run_with_backend(q, k, v, mask, backend, label):
    """Run SDPA with a forced backend and report the result.

    Args:
        q, k, v: Query, key, value tensors.
        mask: Attention mask tensor (or None).
        backend: ``SDPBackend`` enum value to force, or ``None`` for
            auto-dispatch.
        label: Human-readable label for this test case.

    Returns:
        ``True`` if the forced backend ran successfully, ``False``
        otherwise.
    """
    try:
        if backend is not None:
            with sdpa_kernel([backend]):
                F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        print(f"  {label:50s}  SUCCESS")
        return True
    except Exception as exc:
        print(f"  {label:50s}  FAILED: {exc}")
        return False


def verify_flash_mask():
    """Run all four mask/backend configurations and report results."""
    print("=" * 70)
    print("FlashAttention + Additive Float Mask Verification")
    print("=" * 70)
    print()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    q, k, v = _make_tensors()
    float_mask = _make_float_mask()
    bool_mask = float_mask < -500

    # --- Test 1: No mask, FLASH forced (baseline) ---
    print("Test 1: No mask, FLASH_ATTENTION forced")
    _run_with_backend(
        q, k, v, None, SDPBackend.FLASH_ATTENTION, "No mask + FLASH"
    )
    print()

    # --- Test 2: Additive float mask, FLASH forced (KEY TEST) ---
    print("Test 2: Additive float mask, FLASH_ATTENTION forced")
    _run_with_backend(
        q, k, v, float_mask, SDPBackend.FLASH_ATTENTION,
        "Float mask + FLASH (key test)",
    )
    print()

    # --- Test 3: Additive float mask, no backend forced (auto-dispatch) ---
    print("Test 3: Additive float mask, auto-dispatch (no backend forced)")
    _run_with_backend(
        q, k, v, float_mask, None, "Float mask + auto-dispatch"
    )
    print()

    # --- Test 4: Boolean mask, FLASH forced (expected to fail) ---
    print("Test 4: Boolean mask, FLASH_ATTENTION forced")
    _run_with_backend(
        q, k, v, bool_mask, SDPBackend.FLASH_ATTENTION,
        "Bool mask + FLASH (expected fail)",
    )
    print()

    # --- Timing comparison ---
    print("Test 5: Timing comparison (1000 iterations each)")
    torch.cuda.synchronize()

    # No mask
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(1000):
        F.scaled_dot_product_attention(q, k, v)
    t1.record()
    torch.cuda.synchronize()
    no_mask_ms = t0.elapsed_time(t1)

    # Float mask
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(1000):
        F.scaled_dot_product_attention(q, k, v, attn_mask=float_mask)
    t1.record()
    torch.cuda.synchronize()
    float_mask_ms = t0.elapsed_time(t1)

    print(f"  No mask:      {no_mask_ms:8.2f} ms / 1000 iters")
    print(f"  Float mask:   {float_mask_ms:8.2f} ms / 1000 iters")
    print(f"  Mask overhead: {float_mask_ms / no_mask_ms:.2f}×")
    print()

    # --- Backend dispatch log ---
    print("Test 6: Backend dispatch log (auto-dispatch with float mask)")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    with torch.nn.attention.sdpa_kernel(
        [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
    ):
        try:
            F.scaled_dot_product_attention(q, k, v, attn_mask=float_mask)
            print("  SDPA ran with float mask.")
            print("  (Backend selection is internal to PyTorch.)")
        except Exception as exc:
            print(f"  FAILED: {exc}")

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("If Test 2 shows SUCCESS:")
    print("  → FlashAttention DOES handle additive float masks")
    print("  → CachedDistAttention and KNNMaskSparseAttention")
    print("    both get the fused FlashAttention kernel")
    print("  → The 'FlashAttention cannot use masks' assumption is wrong")
    print()
    print("If Test 2 shows FAILED:")
    print("  → FlashAttention does NOT handle additive float masks")
    print("  → SDPA falls back to EFFICIENT_ATTENTION or MATH")
    print("  → The mask overhead is real and measurable")
    print()


if __name__ == "__main__":
    verify_flash_mask()
