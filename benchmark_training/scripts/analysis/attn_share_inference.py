#!/usr/bin/env python3
"""Attention-share attribution from PyTorch profiler traces (2026-10-28).

Question (owner): how big is the attention share s at INFERENCE (forward only,
no backward)? The report's s ~ 0.098 is the attention share of the TRAINING
step; at inference backward is absent and the share rises sharply.

Method: every GPU kernel is linked to its cudaLaunchKernel runtime event via
the correlation id (same attribution as audit_norm_span_trace.py A4), and the
launch timestamp is assigned to the innermost [pl][module] CachedDistAttention
span containing it. Kernels are classified as the fused attention kernel
(name contains cudnn/fmha/flash) vs everything else launched inside the
attention spans (the mask/decay construction chain, RoPE, reshapes).

Measured results (H100 traces, results/epoch_profile/dense_prof_s42/):
- validation_step trace (forward-only, no_grad, 2 profiled steps):
  attention modules hold ~59.5% of GPU kernel time (153.4 of 257.7 ms);
  the fused attention kernel itself only ~2.4% (6.1 ms); the remainder is
  the O(B*N^2) mask/decay construction chain.
- training_step trace: attention-span kernels 14.2 ms/step (~7.3% of the
  194 ms production step wall); with attention backward this reconciles with
  the published s ~ 0.098 (10.05% step share).

Usage:
    python benchmark_training/scripts/analysis/attn_share_inference.py \
        --trace results/epoch_profile/dense_prof_s42/fit-profile-validation_step.pt.trace.json \
        --label "validation (forward-only)"
"""

import argparse
import bisect
import json
from collections import defaultdict

ATTN_SPAN_TOKEN = "CachedDistAttention: model"
FUSED_TOKENS = ("cudnn", "fmha", "flash", "attention_kernel")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--label", default="trace")
    args = ap.parse_args()

    with open(args.trace) as f:
        events = json.load(f)["traceEvents"]

    spans = []           # (ts, dur) attention-module spans
    step_spans = defaultdict(float)   # name -> total dur
    launches = {}        # correlation -> launch ts (cudaLaunchKernel)
    kernels = []         # (ts, dur, name, correlation)

    for e in events:
        if e.get("ph") != "X":
            continue
        cat, name, ts, dur = e.get("cat"), e.get("name", ""), e.get("ts"), e.get("dur", 0)
        if cat == "user_annotation":
            if ATTN_SPAN_TOKEN in name:
                spans.append((ts, dur))
            elif "ProfilerStep#" in name or "Strategy.training_step" in name \
                    or "Strategy.backward" in name or "Strategy.validation_step" in name:
                step_spans[name.split("Strategy.")[-1]] += dur
        elif cat == "cuda_runtime" and name == "cudaLaunchKernel":
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None:
                launches[corr] = ts
        elif cat == "kernel":
            corr = (e.get("args") or {}).get("correlation")
            kernels.append((ts, dur, name, corr))

    spans.sort()
    starts = [s[0] for s in spans]
    total_kernel_us = sum(k[1] for k in kernels)
    wall_us = sum(step_spans.values())

    attn_kernel_us = 0.0
    fused_kernel_us = 0.0
    n_attn_kernels = 0
    for ts, dur, name, corr in kernels:
        lt = launches.get(corr, ts)          # fall back to kernel ts
        i = bisect.bisect_right(starts, lt) - 1
        if i >= 0 and lt <= spans[i][0] + spans[i][1]:
            attn_kernel_us += dur
            n_attn_kernels += 1
            if any(t in name.lower() for t in FUSED_TOKENS):
                fused_kernel_us += dur

    attn_span_us = sum(d for _, d in spans)
    print(f"[{args.label}] events: {len(kernels)} kernels, "
          f"{len(spans)} attention spans")
    print(f"  profiled wall (step spans):        {wall_us/1e3:10.1f} ms")
    print(f"  attention module CPU wall:         {attn_span_us/1e3:10.1f} ms "
          f"({attn_span_us/wall_us*100:5.1f}% of wall)")
    print(f"  total GPU kernel time:             {total_kernel_us/1e3:10.1f} ms")
    print(f"  kernels launched in attn spans:    {attn_kernel_us/1e3:10.1f} ms "
          f"({attn_kernel_us/total_kernel_us*100:5.1f}% of GPU kernel time, "
          f"{n_attn_kernels} kernels)")
    print(f"    of which fused attn kernel:      {fused_kernel_us/1e3:10.1f} ms "
          f"({fused_kernel_us/total_kernel_us*100:5.1f}%)")
    print(f"    remainder (mask build/PE/etc):   {(attn_kernel_us-fused_kernel_us)/1e3:10.1f} ms")
    for k in sorted(step_spans):
        print(f"  span {k}: {step_spans[k]/1e3:.1f} ms")


if __name__ == "__main__":
    main()
