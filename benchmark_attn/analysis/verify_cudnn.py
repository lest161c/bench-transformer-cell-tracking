"""Verify that the cuDNN attention backend is available on the current GPU.

Run as::

    python verify_cudnn.py

The script creates small dummy Q/K/V tensors in float16 on CUDA and
attempts to run ``scaled_dot_product_attention`` with the
``CUDNN_ATTENTION`` backend forced.  It prints ``SUCCESS`` and exits 0
if the backend runs, or prints the crash error and exits 1 if not.

This is used to confirm the A500 / cuDNN environment before running
the full benchmark suite.
"""

import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def verify_cudnn() -> None:
    """Check CUDA availability and attempt a cuDNN-backed SDPA forward pass.

    Exits with code 1 if CUDA is unavailable or if the
    ``CUDNN_ATTENTION`` backend raises an exception.
    """
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("Error: CUDA not available.")
        sys.exit(1)

    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    batch_size, seq_len, n_heads, head_dim = 2, 1024, 4, 64
    query = torch.randn(
        batch_size, n_heads, seq_len, head_dim,
        dtype=torch.float16, device="cuda",
    )
    key = torch.randn(
        batch_size, n_heads, seq_len, head_dim,
        dtype=torch.float16, device="cuda",
    )
    value = torch.randn(
        batch_size, n_heads, seq_len, head_dim,
        dtype=torch.float16, device="cuda",
    )

    print("Attempting to run SDPA with CUDNN_ATTENTION only...")
    try:
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            F.scaled_dot_product_attention(query, key, value)
        print("SUCCESS! CUDNN_ATTENTION ran successfully.")
    except Exception as exc:
        print(f"HARD CRASH: Failed to run CUDNN_ATTENTION.\nError details:\n{exc}")
        sys.exit(1)


if __name__ == "__main__":
    verify_cudnn()
