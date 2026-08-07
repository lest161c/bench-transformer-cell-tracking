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
    mem = (peak - baseline) / (1024 ** 2)
    t = benchmark.Timer("fn()", globals={"fn": fn}, num_threads=1)
    t_mean = t.blocked_autorange(min_run_time=min_run_time).mean
    return t_mean, mem


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
    nH, D, Dh = args.nhead, args.d, args.d // args.nhead

    for N in args.Ns:
        torch.manual_seed(42)
        q = torch.randn(1, nH, N, Dh, device=device, dtype=dtype)
        k = torch.randn(1, nH, N, Dh, device=device, dtype=dtype)
        v = torch.randn(1, nH, N, Dh, device=device, dtype=dtype)

        # ── A: pure_dense — SDPA on full N×N ──
        try:
            t, mem = measure(lambda: F.scaled_dot_product_attention(q, k, v))
            rows.append(["pure_dense", N, t * 1000, mem, "ok"])
        except RuntimeError as e:
            rows.append(["pure_dense", N, None, None, "oom" if "out of memory" in str(e).lower() else str(e)[:120]])

        # ── B: pure_flash (same as pure_dense, explicitly tracked) ──
        try:
            t, mem = measure(lambda: F.scaled_dot_product_attention(q, k, v))
            rows.append(["pure_flash", N, t * 1000, mem, "ok"])
        except RuntimeError as e:
            rows.append(["pure_flash", N, None, None, "oom" if "out of memory" in str(e).lower() else str(e)[:120]])

        # ── C: pure_gather for each K ──
        for K in args.Ks:
            try:
                B_idx = torch.arange(1, device=device).view(1, 1, 1, 1)
                H_idx = torch.arange(nH, device=device).view(1, nH, 1, 1)
                idx = torch.randint(0, N, (1, N, K), device=device)
                idx_exp = idx.unsqueeze(1).expand(1, nH, N, K)

                k_sel = k[B_idx, H_idx, idx_exp, :]
                v_sel = v[B_idx, H_idx, idx_exp, :]
                scale = Dh ** -0.5

                def gather_attn_fn():
                    scores = torch.matmul(q.unsqueeze(3), k_sel.transpose(-2, -1)).squeeze(3)
                    scores.mul_(scale)
                    attn = F.softmax(scores, dim=-1)
                    return torch.matmul(attn.unsqueeze(3), v_sel).squeeze(3)

                t, mem = measure(gather_attn_fn)
                rows.append([f"pure_gather_K={K}", N, t * 1000, mem, "ok"])
            except RuntimeError as e:
                rows.append([f"pure_gather_K={K}", N, None, None,
                             "oom" if "out of memory" in str(e).lower() else str(e)[:120]])

        # ── D: pure_mask for each K ──
        for K in args.Ks:
            try:
                mask = torch.full((1, nH, N, N), float("-inf"), device=device, dtype=dtype)
                idx = torch.randint(0, N, (1, N, K), device=device).unsqueeze(1).expand(1, nH, N, K)
                mask.scatter_(3, idx, 0.0)

                def mask_attn_fn():
                    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

                t, mem = measure(mask_attn_fn)
                rows.append([f"pure_mask_K={K}", N, t * 1000, mem, "ok"])
            except RuntimeError as e:
                rows.append([f"pure_mask_K={K}", N, None, None,
                             "oom" if "out of memory" in str(e).lower() else str(e)[:120]])

        # ── E: pure_minimax for each block_size/config ──
        for Bk in args.block_sizes:
            ksel = max(1, min(4, (N + Bk - 1) // Bk - 1))
            try:
                num_blocks = (N + Bk - 1) // Bk
                pad = num_blocks * Bk - N
                k_pad = F.pad(k, (0, 0, 0, pad))
                k_block = k_pad.view(1, nH, num_blocks, Bk, Dh)

                q_exp = q.unsqueeze(3).unsqueeze(-3)
                k_block_exp = k_block.unsqueeze(2)
                scores_raw = (q_exp * k_block_exp).sum(dim=-1)
                block_scores, _ = scores_raw.max(dim=-1)
                _, topk_blk = torch.topk(block_scores, k=ksel, dim=-1)

                offsets = torch.arange(Bk, device=device).view(1, 1, 1, 1, Bk)
                token_idx = (topk_blk.unsqueeze(-1) * Bk + offsets).clamp(0, N + pad - 1)
                token_idx = token_idx.view(1, nH, N, ksel * Bk)

                k_sel = k_pad[torch.arange(1, device=device).view(1, 1, 1, 1),
                              torch.arange(nH, device=device).view(1, nH, 1, 1),
                              token_idx, :]
                v_pad = F.pad(v, (0, 0, 0, pad))
                v_sel = v_pad[torch.arange(1, device=device).view(1, 1, 1, 1),
                              torch.arange(nH, device=device).view(1, nH, 1, 1),
                              token_idx, :]

                def minimax_attn_fn():
                    scores = torch.matmul(q.unsqueeze(3), k_sel.transpose(-2, -1)).squeeze(3)
                    scores = scores * (Dh ** -0.5)
                    attn = F.softmax(scores, dim=-1)
                    return torch.matmul(attn.unsqueeze(3), v_sel).squeeze(3)

                t, mem = measure(minimax_attn_fn)
                label = f"pure_minimax_Bk={Bk}_k={ksel}"
                rows.append([label, N, t * 1000, mem, "ok"])
            except RuntimeError as e:
                rows.append([f"pure_minimax_Bk={Bk}", N, None, None,
                             "oom" if "out of memory" in str(e).lower() else str(e)[:120]])

        print(f"  N={N} done", flush=True)

    return rows


def print_table(rows, Ns):
    """Print a formatted timing table for each method across *Ns*.

    Args:
        rows: List of ``[method, N, time_ms, memory_mb, status]``.
        Ns: Sequence lengths to include as columns.
    """
    methods = sorted(set(r[0] for r in rows))
    print(f"\n{'method':>30s}", end="")
    for N in Ns:
        print(f"  N={N:>5d}", end="")
    print()
    for m in methods:
        print(f"{m:>30s}", end="")
        for N in Ns:
            r = [row for row in rows if row[0] == m and row[1] == N]
            if r and r[0][2] is not None:
                print(f" {r[0][2]:7.2f}", end="")
            else:
                tag = "OOM" if r and "oom" in str(r[0][4]).lower() else "ERR"
                print(f" {tag:>7s}", end="")
        print()


def main():
    """Run the pure attention benchmark and save results to CSV."""
    p = argparse.ArgumentParser(description="Pure attention kernel benchmark")
    p.add_argument("--d", type=int, default=320)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=50)
    p.add_argument("--out", default="pure_attn_results.csv")
    args = p.parse_args()
    args.Ns = [128, 256, 512, 1024, 2048, 4096, 8192]
    args.Ks = [4, 16, 32, 64, 128]
    args.block_sizes = [32, 64, 128]

    rows = run_pure_benchmark(args)

    header = ["method", "N", "time_ms", "memory_mb", "status"]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

    print(f"\nWrote {len(rows)} rows → {args.out}")
    print_table(rows, args.Ns)


if __name__ == "__main__":
    main()
