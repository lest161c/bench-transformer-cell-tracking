#!/usr/bin/env python3
r"""Adversarial audit of the ``blockwise_causal_norm`` profiler span (SPEC 0042).

Why this script exists: labbook E21 published three numbers from a single
profiler run (job 4424548 task 4, node c30, shared) — the norm span covers
~42 % of the profiled training-step wall, kernels launched inside the span
account for ~34 % of the production step wall (65.5 ms of 192 ms), and the
span holds 48 calls/step. The published numbers rest on the profiler's own
table aggregation, which the trace shows to be internally inconsistent (the
table's Self CUDA column exceeds 100 % in three rows, so annotation-level
CUDA attribution overlaps). This script recomputes every published number
directly from the raw Chrome trace and runs the SPEC 0042 attack vectors
A1 (nesting audit), A2 (profiler-overhead bound), and A4 (correlation-id
kernel attribution):

  A1  nesting      classify every norm span as inside the TrackingTransformer
                   module span / inside training_step only / inside neither;
                   flag any parent span shorter than its child.
  A2  overhead     count aten ops inside vs outside the norm spans; bound the
                   true span time as span_measured − ops_inside × per-op tax
                   (tax from the active-phase step wall minus the production
                   step wall, spread uniformly over ops); report the corrected
                   range of the 42 % figure.
  A4  kernels      attribute every kernel to its launching cpu_op via the
                   cudaLaunchKernel correlation id, sum kernel time launched
                   inside norm spans, and compare against the GPU-side
                   annotation envelope (the source of the published 65.5 ms);
                   quantify the double counting between the two.

All numbers are recomputed from the trace, never copied from the profiler
report. The recomputed report numbers (span totals, call counts, step walls)
are cross-checked against ``results/epoch_profile/fit-profile_norm_span.txt``
by the caller.

Usage::

    # default: the SPEC 0042 trace pulled from the cluster (c30, shared)
    python benchmark_training/scripts/analysis/audit_norm_span_trace.py

    # explicit trace and output paths
    python benchmark_training/scripts/analysis/audit_norm_span_trace.py \
        --trace /path/to/fit-profile-*.pt.trace.json \
        --output results/epoch_profile/norm_span_trace_audit.json

The production step wall defaults to 192.0 ms — the median batch_compute of
the five exclusive num_workers=8 callback legs (198.94–201.96 s over 1041
batches, results/epoch_profile/w8_t*_s42/epoch_profile.json), i.e. the same
training configuration as the profile leg without the profiler attached.

Exit status: 0 always (the script reports measurements; verdicts are drawn
in the labbook addendum, not by exit code).
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import defaultdict
from pathlib import Path

# user_annotation names of the spans under test (full names as written by
# Lightning / the fork commit 155aa29).
MODULE_SPAN_PREFIX = "[pl][module]trackastra.model.model.TrackingTransformer"
TRAINING_STEP_SPAN = "[pl][profile][Strategy]SingleDeviceStrategy.training_step"
BACKWARD_SPAN = "[pl][profile][Strategy]SingleDeviceStrategy.backward"
NORM_SPAN_NAME = "blockwise_causal_norm"
PROFILER_STEP_PREFIX = "ProfilerStep#"


def load_trace(trace_path: Path) -> list[dict]:
    """Load a Chrome JSON trace and return its event list.

    Args:
        trace_path: path to a PyTorch ``export_chrome_trace`` JSON file.

    Returns:
        The list of trace events (dicts with ``ph``, ``cat``, ``name``,
        ``ts``, ``dur``, ``pid``, ``tid``, ``args`` keys).
    """
    with open(trace_path) as trace_file:
        data = json.load(trace_file)
    return data["traceEvents"]


def event_interval(event: dict) -> tuple[float, float]:
    """Return the (start, end) timestamp interval of a span event."""
    return event["ts"], event["ts"] + event["dur"]


def contains(outer: tuple[float, float], inner: tuple[float, float]) -> bool:
    """Test interval containment with a small tolerance for clock rounding."""
    tolerance_us = 1.0
    return (
        inner[0] >= outer[0] - tolerance_us and inner[1] <= outer[1] + tolerance_us
    )


def group_by_thread(
    events: list[dict],
) -> dict[tuple[int, int], list[dict]]:
    """Group span events by (pid, tid) and sort each timeline by start time.

    Args:
        events: trace events with span semantics (``ph == "X"``).

    Returns:
        Mapping (pid, tid) -> events sorted by ``ts``.
    """
    timelines: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for event in events:
        if event.get("ph") == "X":
            timelines[(event["pid"], event["tid"])].append(event)
    for timeline in timelines.values():
        timeline.sort(key=lambda event: event["ts"])
    return timelines


def innermost_containing(
    timeline: list[dict], timeline_starts: list[float], timestamp: float
) -> dict | None:
    """Return the innermost span on a timeline that contains a timestamp.

    Args:
        timeline: spans on one thread, sorted by ``ts``.
        timeline_starts: the sorted ``ts`` values of ``timeline``.
        timestamp: the point-in-time to locate (e.g. a kernel-launch
            runtime call).

    Returns:
        The span with the smallest duration whose interval contains the
        timestamp, or None when no span contains it.
    """
    position = bisect.bisect_right(timeline_starts, timestamp) - 1
    best: dict | None = None
    index = position
    while index >= 0:
        candidate = timeline[index]
        start, end = event_interval(candidate)
        if end < timestamp:
            break
        if start <= timestamp <= end:
            # the innermost container is the one with the latest start
            # (spans on one thread nest); prefer the shorter one on ties
            if best is None or candidate["ts"] > best["ts"] or (
                candidate["ts"] == best["ts"] and candidate["dur"] < best["dur"]
            ):
                best = candidate
        index -= 1
    return best


def is_inside(span: dict, container: dict) -> bool:
    """Test whether a span lies temporally inside a container span."""
    return contains(event_interval(container), event_interval(span))


def classify_norm_spans(
    norm_spans: list[dict],
    module_spans: list[dict],
    training_step_spans: list[dict],
) -> dict[str, int]:
    """A1: classify norm spans by containment in module / training_step spans.

    Args:
        norm_spans: CPU-side ``blockwise_causal_norm`` user annotations.
        module_spans: TrackingTransformer module spans.
        training_step_spans: Lightning training_step spans.

    Returns:
        Counts per class: ``inside_module``, ``inside_training_step_only``,
        ``inside_neither``.
    """
    counts = {
        "inside_module": 0,
        "inside_training_step_only": 0,
        "inside_neither": 0,
    }
    for norm_span in norm_spans:
        in_module = any(is_inside(norm_span, module) for module in module_spans)
        in_training_step = any(
            is_inside(norm_span, step) for step in training_step_spans
        )
        if in_module:
            counts["inside_module"] += 1
        elif in_training_step:
            counts["inside_training_step_only"] += 1
        else:
            counts["inside_neither"] += 1
    return counts


def find_nesting_violations(spans: list[dict]) -> list[dict]:
    """A1: find same-thread span pairs where a parent is shorter than a child.

    Args:
        spans: spans on one thread, sorted by ``ts``.

    Returns:
        List of violation records with parent/child names, durations, and
        the containment status (a child starting inside a parent but
        outlasting it is a violation; a properly nested shorter child is
        not).
    """
    violations = []
    for child_index, child in enumerate(spans):
        child_start, child_end = event_interval(child)
        for parent_index in range(child_index - 1, -1, -1):
            parent = spans[parent_index]
            parent_start, parent_end = event_interval(parent)
            if parent_end < child_start:
                break
            starts_inside = child_start >= parent_start
            ends_inside = child_end <= parent_end
            if starts_inside and not ends_inside:
                violations.append(
                    {
                        "parent_name": parent["name"][:80],
                        "parent_dur_ms": parent["dur"] / 1000.0,
                        "child_name": child["name"][:80],
                        "child_dur_ms": child["dur"] / 1000.0,
                    }
                )
    return violations


def main() -> int:
    """Run the A1/A2/A4 audit and write the JSON report.

    Returns:
        0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("/tmp/opencode/norm_span_trace.json"),
        help="Chrome trace of the norm-span profile leg (SPEC 0042).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/epoch_profile/norm_span_trace_audit.json"),
        help="JSON output path for the audit results.",
    )
    parser.add_argument(
        "--production-step-wall-ms",
        type=float,
        default=192.0,
        help=(
            "Production step wall in ms (w8 callback median batch_compute "
            "199.85 s / 1041 batches = 192.0 ms)."
        ),
    )
    arguments = parser.parse_args()

    events = load_trace(arguments.trace)

    span_events = [event for event in events if event.get("ph") == "X"]
    cpu_ops = [event for event in span_events if event.get("cat") == "cpu_op"]
    user_annotations = [
        event for event in span_events if event.get("cat") == "user_annotation"
    ]
    gpu_annotations = [
        event for event in span_events if event.get("cat") == "gpu_user_annotation"
    ]
    kernels = [event for event in span_events if event.get("cat") == "kernel"]
    memcpys = [event for event in span_events if event.get("cat") == "gpu_memcpy"]
    memsets = [event for event in span_events if event.get("cat") == "gpu_memset"]
    device_events = kernels + memcpys + memsets
    runtime_events = [
        event for event in span_events if event.get("cat") == "cuda_runtime"
    ]

    norm_spans_all = [
        event for event in user_annotations if event["name"] == NORM_SPAN_NAME
    ]
    module_spans_all = [
        event
        for event in user_annotations
        if event["name"].startswith(MODULE_SPAN_PREFIX)
    ]
    training_step_spans_all = [
        event for event in user_annotations if event["name"] == TRAINING_STEP_SPAN
    ]
    backward_spans_all = [
        event for event in user_annotations if event["name"] == BACKWARD_SPAN
    ]
    profiler_steps = sorted(
        (event for event in user_annotations if event["name"].startswith(
            PROFILER_STEP_PREFIX)),
        key=lambda event: event["ts"],
    )

    # ------------------------------------------------------------------
    # Identify profiler-active steps: steps containing kernel launches.
    # ------------------------------------------------------------------
    launches = [
        event for event in runtime_events if event["name"].startswith("cudaLaunch")
    ]
    launch_times = sorted(event["ts"] for event in launches)

    def kernels_launched_within(span: dict) -> int:
        start, end = event_interval(span)
        low = bisect.bisect_left(launch_times, start)
        high = bisect.bisect_right(launch_times, end)
        return high - low

    active_steps = [
        step for step in profiler_steps if kernels_launched_within(step) > 0
    ]
    inactive_steps = [step for step in profiler_steps if step not in active_steps]

    # ------------------------------------------------------------------
    # A1 — nesting audit (per step; norm spans vs module / training_step).
    # ------------------------------------------------------------------
    a1_steps = []
    for step in profiler_steps:
        step_start, step_end = event_interval(step)
        norm_spans = [
            span
            for span in norm_spans_all
            if contains((step_start, step_end), event_interval(span))
        ]
        module_spans = [
            span
            for span in module_spans_all
            if contains((step_start, step_end), event_interval(span))
        ]
        training_step_spans = [
            span
            for span in training_step_spans_all
            if contains((step_start, step_end), event_interval(span))
        ]
        backward_spans = [
            span
            for span in backward_spans_all
            if contains((step_start, step_end), event_interval(span))
        ]
        classification = classify_norm_spans(
            norm_spans, module_spans, training_step_spans
        )
        # nesting check on the main thread inside this step
        main_tid = norm_spans[0]["tid"] if norm_spans else None
        main_spans = [
            span
            for span in user_annotations
            if span["tid"] == main_tid
            and contains((step_start, step_end), event_interval(span))
        ]
        main_spans.sort(key=lambda event: event["ts"])
        violations = find_nesting_violations(main_spans)
        a1_steps.append(
            {
                "step_name": step["name"],
                "is_active": step in active_steps,
                "step_wall_ms": step["dur"] / 1000.0,
                "n_norm_spans": len(norm_spans),
                "n_module_spans": len(module_spans),
                "norm_classification": classification,
                "norm_cpu_total_ms": sum(s["dur"] for s in norm_spans) / 1000.0,
                "module_cpu_total_ms": sum(s["dur"] for s in module_spans) / 1000.0,
                "training_step_wall_ms": sum(
                    s["dur"] for s in training_step_spans
                )
                / 1000.0,
                "backward_wall_ms": sum(s["dur"] for s in backward_spans) / 1000.0,
                "nesting_violations": len(violations),
                "violation_examples": violations[:5],
            }
        )

    # ------------------------------------------------------------------
    # A2 — profiler-overhead bound on the norm-span share.
    # ------------------------------------------------------------------
    a2_steps = []
    for step in active_steps:
        step_start, step_end = event_interval(step)
        step_ops = [
            op for op in cpu_ops if contains((step_start, step_end), event_interval(op))
        ]
        step_norm_spans = [
            span
            for span in norm_spans_all
            if contains((step_start, step_end), event_interval(span))
        ]
        norm_intervals = [event_interval(span) for span in step_norm_spans]

        def op_in_norm(op: dict) -> bool:
            return any(
                contains(interval, event_interval(op)) for interval in norm_intervals
            )

        ops_inside = sum(1 for op in step_ops if op_in_norm(op))
        ops_outside = len(step_ops) - ops_inside
        norm_cpu_total = sum(span["dur"] for span in step_norm_spans) / 1000.0
        step_wall = step["dur"] / 1000.0
        production_wall = arguments.production_step_wall_ms
        total_ops = len(step_ops)
        tax_full_ms = (step_wall - production_wall) / total_ops * ops_inside
        corrected_full = norm_cpu_total - tax_full_ms
        a2_steps.append(
            {
                "step_name": step["name"],
                "step_wall_ms": step_wall,
                "production_wall_ms": production_wall,
                "profiler_tax_step_ms": step_wall - production_wall,
                "n_ops_step": total_ops,
                "n_ops_inside_norm": ops_inside,
                "n_ops_outside_norm": ops_outside,
                "tax_per_op_us": (step_wall - production_wall) / total_ops * 1000.0,
                "norm_cpu_total_ms": norm_cpu_total,
                "norm_calls": len(step_norm_spans),
                "tax_inside_norm_ms": tax_full_ms,
                "norm_corrected_lower_bound_ms": corrected_full,
                "share_of_profiled_step_pct": norm_cpu_total / step_wall * 100.0,
                "corrected_share_of_profiled_step_pct": corrected_full / step_wall
                * 100.0,
                "corrected_share_of_production_step_pct": corrected_full
                / production_wall
                * 100.0,
            }
        )

    # ------------------------------------------------------------------
    # A4 — correlation-id kernel attribution.
    # ------------------------------------------------------------------
    # correlation id -> launching runtime event (CPU side)
    correlation_to_launch: dict[int, dict] = {}
    for runtime_event in runtime_events:
        correlation = runtime_event.get("args", {}).get("correlation")
        if correlation is not None:
            correlation_to_launch[correlation] = runtime_event

    # per-(pid,tid) cpu_op timelines for innermost-op lookup
    op_timelines = group_by_thread(cpu_ops)
    op_timeline_starts = {
        key: [event["ts"] for event in timeline]
        for key, timeline in op_timelines.items()
    }

    a4_steps = []
    for step in active_steps:
        step_start, step_end = event_interval(step)
        step_norm_spans = [
            span
            for span in norm_spans_all
            if contains((step_start, step_end), event_interval(span))
        ]
        norm_cpu_intervals = [event_interval(span) for span in step_norm_spans]
        # GPU-side annotations of this step's norm calls; the GPU clock is
        # the same time base in PyTorch chrome traces, with async drain
        # pushing envelopes slightly past the step end (10 ms tolerance).
        step_window = (step_start - 10000.0, step_end + 10000.0)
        gpu_norm_spans = [
            span
            for span in gpu_annotations
            if span["name"] == NORM_SPAN_NAME
            and contains(step_window, event_interval(span))
        ]
        gpu_norm_intervals = [event_interval(span) for span in gpu_norm_spans]

        # normalize GPU/CPU clock offset: GPU annotations are shifted; align
        # by comparing the envelope to correlated kernels instead of raw
        # intervals (the offset cancels in the difference below only if the
        # same clock is used; PyTorch writes GPU events in the same time
        # base, so no shift is applied).
        correlated_norm_kernel_ms = 0.0
        envelope_kernel_ms = 0.0
        foreign_kernel_ms = 0.0
        total_kernel_ms = 0.0
        total_device_ms = 0.0
        n_kernels_step = 0
        n_kernels_norm_correlated = 0
        for device_event in device_events:
            kernel_start, kernel_end = event_interval(device_event)
            launch = correlation_to_launch.get(
                device_event.get("args", {}).get("correlation")
            )
            if launch is None:
                continue
            if not (step_start <= launch["ts"] <= step_end):
                continue
            total_device_ms += device_event["dur"] / 1000.0
            if device_event.get("cat") != "kernel":
                continue
            total_kernel_ms += device_event["dur"] / 1000.0
            n_kernels_step += 1
            in_envelope = any(
                kernel_start >= interval[0] and kernel_end <= interval[1]
                for interval in gpu_norm_intervals
            )
            if in_envelope:
                envelope_kernel_ms += device_event["dur"] / 1000.0
            # launching cpu_op lookup on the launch's thread
            timeline = op_timelines.get((launch["pid"], launch["tid"]))
            if timeline is None:
                continue
            starts = op_timeline_starts[(launch["pid"], launch["tid"])]
            innermost = innermost_containing(
                timeline, starts, launch["ts"]
            )
            if innermost is None:
                continue
            launched_by_norm = any(
                contains(interval, event_interval(innermost))
                for interval in norm_cpu_intervals
            )
            if launched_by_norm:
                correlated_norm_kernel_ms += device_event["dur"] / 1000.0
                n_kernels_norm_correlated += 1
            elif in_envelope:
                foreign_kernel_ms += device_event["dur"] / 1000.0

        a4_steps.append(
            {
                "step_name": step["name"],
                "n_kernels_step": n_kernels_step,
                "gpu_busy_kernels_ms": total_kernel_ms,
                "gpu_busy_incl_memcpy_ms": total_device_ms,
                "n_norm_gpu_annotations": len(gpu_norm_spans),
                "norm_correlated_kernel_ms": correlated_norm_kernel_ms,
                "n_kernels_norm_correlated": n_kernels_norm_correlated,
                "norm_envelope_kernel_ms": envelope_kernel_ms,
                "foreign_kernels_in_envelope_ms": foreign_kernel_ms,
                "envelope_minus_correlated_ms": envelope_kernel_ms
                - correlated_norm_kernel_ms,
                "norm_kernel_share_of_gpu_busy_pct": correlated_norm_kernel_ms
                / total_kernel_ms
                * 100.0
                if total_kernel_ms
                else 0.0,
                "norm_kernel_share_of_production_step_pct": correlated_norm_kernel_ms
                / arguments.production_step_wall_ms
                * 100.0,
            }
        )

    # ------------------------------------------------------------------
    # Sync-call census (aten::item forces a device synchronization).
    # ------------------------------------------------------------------
    item_audit = []
    for step in active_steps:
        step_start, step_end = event_interval(step)
        step_items = [
            event
            for event in cpu_ops
            if event["name"] == "aten::item"
            and contains((step_start, step_end), event_interval(event))
        ]
        step_norm_spans = [
            span
            for span in norm_spans_all
            if contains((step_start, step_end), event_interval(span))
        ]
        norm_intervals = [event_interval(span) for span in step_norm_spans]
        items_inside_norm = sum(
            1
            for item in step_items
            if any(contains(interval, event_interval(item)) for interval in norm_intervals)
        )
        item_audit.append(
            {
                "step_name": step["name"],
                "n_item_calls": len(step_items),
                "n_item_inside_norm_spans": items_inside_norm,
            }
        )

    result = {
        "description": (
            "SPEC 0042 adversarial audit of the blockwise_causal_norm span "
            "(A1 nesting, A2 overhead bound, A4 correlation attribution). "
            f"Trace: {arguments.trace}. Production step wall "
            f"{arguments.production_step_wall_ms} ms (w8 callback median)."
        ),
        "profiler_steps": [
            {
                "name": step["name"],
                "wall_ms": step["dur"] / 1000.0,
                "is_active": step in active_steps,
            }
            for step in profiler_steps
        ],
        "a1_nesting": a1_steps,
        "a2_overhead_bound": a2_steps,
        "a4_kernel_attribution": a4_steps,
        "item_sync_census": item_audit,
    }

    print(json.dumps(result, indent=2))

    output_path = arguments.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as output_file:
        json.dump(result, output_file, indent=2)
    print(f"\nWritten: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
