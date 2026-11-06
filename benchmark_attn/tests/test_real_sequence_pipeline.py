"""Unit tests for the real-sequence aggregate inference planner."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = BENCHMARK_ROOT / "scripts" / "benchmarks"

sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(BENCHMARKS_DIR))

from benchmark_real_sequence_pipeline import (  # noqa: E402
    SequenceInfo,
    plan_sequence_prefix,
    select_sequence_plans,
)


def _info(name: str, counts: list[int]) -> SequenceInfo:
    return SequenceInfo(
        condition="test",
        name=name,
        path=Path("/tmp") / name,
        frame_count=len(counts),
        frame_detections=np.asarray(counts, dtype=np.int64),
    )


def test_plan_sequence_prefix_exact_and_partial():
    counts = np.asarray([5, 5, 5], dtype=np.int64)

    assert plan_sequence_prefix(counts, 5) == (1, None, 5)
    assert plan_sequence_prefix(counts, 10) == (2, None, 10)
    assert plan_sequence_prefix(counts, 12) == (3, 2, 12)
    assert plan_sequence_prefix(counts, 20) == (3, None, 15)


def test_plan_sequence_prefix_handles_zero_count_frames():
    counts = np.asarray([0, 0, 4, 0, 4], dtype=np.int64)
    assert plan_sequence_prefix(counts, 4) == (3, None, 4)
    assert plan_sequence_prefix(counts, 5) == (5, 1, 5)


def test_select_sequence_plans_hits_target_exactly():
    infos = [
        _info("a", [5, 5, 5]),
        _info("b", [5, 5, 5]),
        _info("c", [5, 5, 5]),
    ]
    plans = select_sequence_plans(infos, target_detections=12, seed=42)

    assert sum(plan.detections for plan in plans) == 12
    assert len(plans) == 1
    assert plans[0].partial_last_frame == 2


def test_select_sequence_plans_can_cross_sequences():
    infos = [
        _info("a", [5]),
        _info("b", [7]),
        _info("c", [11]),
    ]
    plans = select_sequence_plans(infos, target_detections=12, seed=0)

    assert sum(plan.detections for plan in plans) == 12
    assert len(plans) >= 2


def test_select_sequence_plans_raises_when_target_unreachable():
    infos = [_info("a", [3]), _info("b", [4])]
    with pytest.raises(ValueError, match="short by"):
        select_sequence_plans(infos, target_detections=20, seed=42)
