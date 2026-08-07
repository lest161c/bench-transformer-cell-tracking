"""Pure attention kernel benchmark — isolates SDPA/matmul from overhead.

Only times the raw attention computation: Q*K^T + softmax + V multiply.
Excludes QKV projections, output projections, mask construction, gather operations,
positional encoding (ROPE/bias), and all other non-attention overhead.

Methods:
    pure_dense      — F.scaled_dot_product_attention(q, k, v) on (B, nH, N, Dh)
    pure_gather_K   — manual matmul + softmax on (B, nH, N, K, Dh) pre-gathered KV
    pure_mask_K     — F.scaled_dot_product_attention with pre-built N×N KNN mask
    pure_flash      — Same as pure_dense but explicitly to compare with flash kernel
    pure_minimax    — manual matmul + softmax on (B, nH, N, k*Bk, Dh) block-gathered

Usage:
    python benchmark_pure_attn.py [--d 320] [--nhead 8] [--out pure_attn_results.csv]
"""

import argparse
import csv
import gc
import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark


@torch.no_grad()
def measure(fn, warmup=5, min_run_time=0.5):
    """Measure mean execution time and peak GPU memory of *fn*.

    Args:
        fn: Callable to benchmark.
        warmup: Number of untimed warmup iterations.
        min_run_time: Minimum run time for ``torch.utils.benchmark``.

    Returns:
        Tuple of (mean_time_seconds, peak_memory_megabytes).
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    _ = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    memory_mb = (peak - baseline) / (1024 ** 2)
    timer = benchmark.Timer("fn()", globals={"fn": fn}, num_threads=1)
    time_mean = timer.blocked_autorange(min_run_time=min_run_time).mean
    return time_mean, memory_mb


def run_pure_benchmark(args):
    """Run pure attention kernel benchmark across all configurations.

    Benchmarks five attention strategies for each sequence length *N*:
      pure_dense, pure_flash, pure_gather_K, pure_mask_K, pure_minimax.

    Args:
        args: Parsed argparse namespace with fields ``Ns``, ``Ks``,
            ``block_sizes``, ``d``, ``nhead``, ``warmup``, ``rep``.

    Returns:
        List of rows ``[method, N, time_ms, memory_mb, status]``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}  dtype: {dtype}  d={args.d}  nhead={args.nhead}")
    print(f"Flash SDPA enabled: {torch.backends.cuda.flash_sdp_enabled() if device.type == 'cuda' else 'N/A'}")
    print(f"N = {args.Ns}  K = {args.Ks}  Bk = {args.block_sizes}\n")

    rows = []
    n_head, embed_dim, head_dim = args.nhead, args.d, args.d // args.nhead

    for N in args.Ns:
        torch.manual_seed(args.seed)
        query = torch.randn(1, n_head, N, head_dim, device=device, dtype=dtype)
        key = torch.randn(1, n_head, N, head_dim, device=device, dtype=dtype)
        value = torch.randn(1, n_head, N, head_dim, device=device, dtype=dtype)

        # ── A: pure_dense — SDPA on full N×N ──
        try:
            time_s, memory_mb = measure(lambda: F.scaled_dot_product_attention(query, key, value))
            rows.append(["pure_dense", N, time_s * 1000, memory_mb, "ok"])
        except RuntimeError as error:
            rows.append(["pure_dense", N, None, None, "oom" if "out of memory" in str(error).lower() else str(error)[:120]])

        # ── B: pure_flash (same as pure_dense, explicitly tracked) ──
        try:
            time_s, memory_mb = measure(lambda: F.scaled_dot_product_attention(query, key, value))
            rows.append(["pure_flash", N, time_s * 1000, memory_mb, "ok"])
        except RuntimeError as error:
            rows.append(["pure_flash", N, None, None, "oom" if "out of memory" in str(error).lower() else str(error)[:120]])

        # ── C: pure_gather for each K ──
        for K in args.Ks:
            try:
                batch_idx = torch.arange(1, device=device).view(1, 1, 1, 1)
                head_idx = torch.arange(n_head, device=device).view(1, n_head, 1, 1)
                idx = torch.randint(0, N, (1, N, K), device=device)
                idx_exp = idx.unsqueeze(1).expand(1, n_head, N, K)

                key_sel = key[batch_idx, head_idx, idx_exp, :]
                value_sel = value[batch_idx, head_idx, idx_exp, :]
                scale = head_dim ** -0.5

                def gather_attn_fn():
                    scores = torch.matmul(query.unsqueeze(3), key_sel.transpose(-2, -1)).squeeze(3)
                    scores.mul_(scale)
                    attn = F.softmax(scores, dim=-1)
                    return torch.matmul(attn.unsqueeze(3), value_sel).squeeze(3)

                time_s, memory_mb = measure(gather_attn_fn)
                rows.append([f"pure_gather_K={K}", N, time_s * 1000, memory_mb, "ok"])
            except RuntimeError as error:
                rows.append([f"pure_gather_K={K}", N, None, None,
                             "oom" if "out of memory" in str(error).lower() else str(error)[:120]])

        # ── D: pure_mask for each K ──
        for K in args.Ks:
            try:
                mask = torch.full((1, n_head, N, N), float("-inf"), device=device, dtype=dtype)
                idx = torch.randint(0, N, (1, N, K), device=device).unsqueeze(1).expand(1, n_head, N, K)
                mask.scatter_(3, idx, 0.0)

                def mask_attn_fn():
                    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)

                time_s, memory_mb = measure(mask_attn_fn)
                rows.append([f"pure_mask_K={K}", N, time_s * 1000, memory_mb, "ok"])
            except RuntimeError as error:
                rows.append([f"pure_mask_K={K}", N, None, None,
                             "oom" if "out of memory" in str(error).lower() else str(error)[:120]])

        # ── E: pure_minimax for each block_size/config ──
        for Bk in args.block_sizes:
            ksel = max(1, min(4, (N + Bk - 1) // Bk - 1))
            try:
                num_blocks = (N + Bk - 1) // Bk
                pad = num_blocks * Bk - N
                key_pad = F.pad(key, (0, 0, 0, pad))
                key_block = key_pad.view(1, n_head, num_blocks, Bk, head_dim)

                query_exp = query.unsqueeze(3).unsqueeze(-3)
                key_block_exp = key_block.unsqueeze(2)
                scores_raw = (query_exp * key_block_exp).sum(dim=-1)
                block_scores, _ = scores_raw.max(dim=-1)
                _, topk_blk = torch.topk(block_scores, k=ksel, dim=-1)

                offsets = torch.arange(Bk, device=device).view(1, 1, 1, 1, Bk)
                token_idx = (topk_blk.unsqueeze(-1) * Bk + offsets).clamp(0, N + pad - 1)
                token_idx = token_idx.view(1, n_head, N, ksel * Bk)

                key_sel = key_pad[torch.arange(1, device=device).view(1, 1, 1, 1),
                                  torch.arange(n_head, device=device).view(1, n_head, 1, 1),
                                  token_idx, :]
                value_pad = F.pad(value, (0, 0, 0, pad))
                value_sel = value_pad[torch.arange(1, device=device).view(1, 1, 1, 1),
                                      torch.arange(n_head, device=device).view(1, n_head, 1, 1),
                                      token_idx, :]

                def minimax_attn_fn():
                    scores = torch.matmul(query.unsqueeze(3), key_sel.transpose(-2, -1)).squeeze(3)
                    scores = scores * (head_dim ** -0.5)
                    attn = F.softmax(scores, dim=-1)
                    return torch.matmul(attn.unsqueeze(3), value_sel).squeeze(3)

                time_s, memory_mb = measure(minimax_attn_fn)
                label = f"pure_minimax_Bk={Bk}_k={ksel}"
                rows.append([label, N, time_s * 1000, memory_mb, "ok"])
            except RuntimeError as error:
                rows.append([f"pure_minimax_Bk={Bk}", N, None, None,
                             "oom" if "out of memory" in str(error).lower() else str(error)[:120]])

        print(f"  N={N} done", flush=True)

    return rows


def print_table(rows, Ns):
    """Print a formatted timing table for each method across *Ns*.

    Args:
        rows: List of ``[method, N, time_ms, memory_mb, status]``.
        Ns: Sequence lengths to include as columns.
    """
    methods = sorted(set(row[0] for row in rows))
    print(f"\n{'method':>30s}", end="")
    for N in Ns:
        print(f"  N={N:>5d}", end="")
    print()
    for m in methods:
        print(f"{m:>30s}", end="")
        for N in Ns:
            matching_rows = [row for row in rows if row[0] == m and row[1] == N]
            if matching_rows and matching_rows[0][2] is not None:
                print(f" {matching_rows[0][2]:7.2f}", end="")
            else:
                tag = "OOM" if matching_rows and "oom" in str(matching_rows[0][4]).lower() else "ERR"
                print(f" {tag:>7s}", end="")
        print()


def main():
    """Run the pure attention benchmark and save results to CSV."""
    parser = argparse.ArgumentParser(description="Pure attention kernel benchmark")
    parser.add_argument("--d", type=int, default=320)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--out", default="pure_attn_results.csv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.Ns = [128, 256, 512, 1024, 2048, 4096, 8192]
    args.Ks = [4, 16, 32, 64, 128]
    args.block_sizes = [32, 64, 128]

    rows = run_pure_benchmark(args)

    header = ["method", "N", "time_ms", "memory_mb", "status"]
    with open(args.out, "w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote {len(rows)} rows → {args.out}")
    print_table(rows, args.Ns)


if __name__ == "__main__":
    main()
