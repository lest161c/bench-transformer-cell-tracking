"""Common argparse helpers for the benchmark scripts.

Provides the shared CLI flags used across the benchmark entry points and the
comma-separated integer list parser that several scripts use for sweep
parameters such as ``--Ns`` and ``--Ks``.
"""

import argparse


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common benchmark CLI flags: --d, --nhead, --warmup, --rep, --out, --seed, --device.

    The flag defaults mirror the values used by the existing benchmark scripts
    (embedding dimension 320, 8 heads, 5 warmup iterations, 30 ms minimum run
    time, seed 42). ``--device`` accepts ``"cuda"``, ``"cpu"``, or ``"auto"``,
    where ``"auto"`` selects CUDA when available and CPU otherwise.

    Args:
        parser: ArgumentParser to attach the flags to.
    """
    parser.add_argument("--d", type=int, default=320, help="Model embedding dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--rep", type=int, default=30, help="Minimum run time in milliseconds for the benchmark timer")
    parser.add_argument("--out", default="benchmark_results.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", default="auto",
                        help="Device to run on: 'cuda', 'cpu', or 'auto' (auto = cuda if available)")


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated string of integers into a list.

    Args:
        value: Comma-separated integer string, e.g. "128,256,512".

    Returns:
        List of parsed integers, e.g. [128, 256, 512].
    """
    return [int(item) for item in value.split(",")]
