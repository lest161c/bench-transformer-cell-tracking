"""Tests for SPEC 0062: inputs, registry, swap, sanity, timings, and linker.

The tests run on CPU and skip the checkpoint-backed construction, equivalence,
and grid-smoke checks when ``checkpoints/k16`` is absent. Linker tests cover
both the pre-existing iterable ``weights`` path and the additive lazy
``weight_lookup`` path in ``build_graph``.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import pytest
import torch

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BENCHMARK_ROOT.parent
BENCHMARKS_DIR = BENCHMARK_ROOT / "scripts" / "benchmarks"

sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(BENCHMARKS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "trackastra"))

import benchmark_full_model_inference_grid as full_grid  # noqa: E402
from src.bench.data import make_full_model_inputs  # noqa: E402
from src.bench.registry import (  # noqa: E402
    FULL_MODEL_VARIANT_REGISTRY,
    METHOD_REGISTRY,
    resolve_class,
)
from src.bench.timing import ComponentTimer  # noqa: E402
from trackastra.tracking import build_graph  # noqa: E402

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "k16"
EXPECTED_VARIANT_KEYS = {
    "relative_pos",
    "cached_dist",
    "sparse_mask_k16",
    "sparse_gather_k16",
}
EXPECTED_METHOD_KEYS = {
    "dense_masked",
    "dense_flash",
    "sparse_gather_sdpa",
    "sparse_gather_fused",
    "sparse_gather_matmul",
    "sparse_mask",
    "sparse_softmax",
    "sparse_gather_relpos",
    "nsa",
    "minimax",
    "flex_attn",
}
REQUIRED_WEIGHT_PREFIXES = (
    "q_pro.",
    "k_pro.",
    "v_pro.",
    "proj.",
    "rot_pos_enc.",
)
EXPECTED_RAW_CSV_HEADER = [
    "variant",
    "N",
    "B",
    "repeat_idx",
    "status",
    "checkpoint",
    "dtype",
    "seed",
    "knn_neighbors",
    "coord_layout",
    "linker_mode",
    "linker_status",
    "full_forward_ms",
    "embed_ms",
    "encoder_ms",
    "encoder_attn_ms",
    "decoder_ms",
    "decoder_attn_ms",
    "head_logits_ms",
    "head_x_ms",
    "head_y_ms",
    "einsum_ms",
    "embed_proj_ms",
    "normalize_ms",
    "linker_build_graph_ms",
    "linker_greedy_ms",
    "linker_ms",
    "total_ms",
    "peak_mem_mb",
    "dist_2d_standalone_ms",
    "forward_closure_error_ms",
    "error",
]
EXPECTED_OK_TIMING_COLUMNS = (
    "full_forward_ms",
    "embed_ms",
    "encoder_ms",
    "encoder_attn_ms",
    "decoder_ms",
    "decoder_attn_ms",
    "head_logits_ms",
    "head_x_ms",
    "head_y_ms",
    "einsum_ms",
    "embed_proj_ms",
    "normalize_ms",
    "total_ms",
    "peak_mem_mb",
    "dist_2d_standalone_ms",
    "forward_closure_error_ms",
)
EXPECTED_LINKER_TIMING_COLUMNS = (
    "linker_build_graph_ms",
    "linker_greedy_ms",
    "linker_ms",
)


def _linker_nodes(
    frame_count: int, nodes_per_frame: int
) -> list[dict[str, object]]:
    """Build simple integer-strided linker nodes for graph tests.

    Args:
        frame_count: Number of consecutive frames.
        nodes_per_frame: Number of nodes in every frame.

    Returns:
        Node dictionaries whose ``id`` is ``frame * nodes_per_frame + index``
        and whose spatial coordinates are a small deterministic spread.
    """
    nodes = []
    for frame in range(frame_count):
        for index in range(nodes_per_frame):
            node_id = frame * nodes_per_frame + index
            nodes.append(
                {
                    "id": node_id,
                    "time": frame,
                    "label": node_id,
                    "coords": [float(index), float(frame)],
                }
            )
    return nodes


def test_build_graph_iterable_weights_regression():
    """The pre-existing iterable ``weights`` path still builds its edges."""
    nodes = _linker_nodes(frame_count=2, nodes_per_frame=3)
    supplied_weights = [
        ((0, 3), 0.9),
        ((0, 4), 0.2),
        ((2, 5), 0.7),
    ]

    graph = build_graph(
        nodes,
        weights=supplied_weights,
        use_distance=False,
        max_distance=256,
        max_neighbors=10,
        delta_t=1,
    )

    assert set(graph.edges) == {
        (node_in, node_out) for (node_in, node_out), _ in supplied_weights
    }
    for (node_in, node_out), expected_weight in supplied_weights:
        assert graph.edges[node_in, node_out]["weight"] == pytest.approx(
            expected_weight
        )


def test_build_graph_lazy_lookup_matches_iterable_weights():
    """The lazy consecutive-frame lookup reproduces the iterable-weights graph."""
    nodes = _linker_nodes(frame_count=3, nodes_per_frame=3)
    supplied_weights = [
        ((0, 3), 0.9),
        ((1, 4), 0.6),
        ((2, 5), 0.3),
        ((3, 6), 0.8),
        ((4, 7), 0.4),
    ]
    weight_by_edge = dict(supplied_weights)

    iterable_graph = build_graph(
        nodes,
        weights=supplied_weights,
        use_distance=False,
        max_distance=256,
        max_neighbors=10,
        delta_t=1,
    )
    lazy_graph = build_graph(
        nodes,
        weight_lookup=lambda node_in, node_out: weight_by_edge.get(
            (node_in, node_out)
        ),
        use_distance=False,
        max_distance=256,
        max_neighbors=10,
        delta_t=1,
    )

    assert set(lazy_graph.edges) == set(iterable_graph.edges)
    for node_in, node_out in iterable_graph.edges:
        assert lazy_graph.edges[node_in, node_out]["weight"] == pytest.approx(
            iterable_graph.edges[node_in, node_out]["weight"]
        )


def test_make_full_model_inputs_shapes_dtypes_and_timepoints():
    """Inputs have the SPEC 0062 shapes, dtypes, and timepoint range."""
    sequence_length = 32
    window = 10
    knn_neighbors = 16

    coords, features, timepoints, knn_indices = make_full_model_inputs(
        batch_size=1,
        seq_len=sequence_length,
        window=window,
        knn_neighbors=knn_neighbors,
        device="cpu",
        dtype=torch.float32,
        seed=42,
    )

    assert coords.shape == (1, sequence_length, 3)
    assert features.shape == (1, sequence_length, 7)
    assert timepoints.shape == (1, sequence_length)
    assert knn_indices.shape == (1, sequence_length, knn_neighbors)

    assert coords.dtype == torch.float32
    assert features.dtype == torch.float32
    assert timepoints.dtype == torch.int64
    assert knn_indices.dtype == torch.int64

    expected_timepoints = torch.arange(sequence_length) % window
    assert torch.equal(timepoints[0], expected_timepoints)
    assert int(timepoints.min()) >= 0
    assert int(timepoints.max()) <= window - 1
    assert torch.equal(coords[0, :, 0], expected_timepoints.float())
    assert torch.isfinite(coords).all()
    assert torch.isfinite(features).all()


def test_make_full_model_inputs_replicates_batch_one_and_is_seeded():
    """B>1 replicates the B=1 sample and repeated seeds are identical."""
    batch_one = make_full_model_inputs(
        batch_size=1, seq_len=24, device="cpu", seed=123
    )
    batched = make_full_model_inputs(
        batch_size=3, seq_len=24, device="cpu", seed=123
    )
    repeated = make_full_model_inputs(
        batch_size=1, seq_len=24, device="cpu", seed=123
    )

    for batch_one_tensor, repeated_tensor in zip(batch_one, repeated):
        assert torch.equal(batch_one_tensor, repeated_tensor)

    coords_one, features_one, timepoints_one, knn_one = batch_one
    coords_batched, features_batched, timepoints_batched, knn_batched = batched

    for batch_index in range(3):
        assert torch.equal(coords_batched[batch_index], coords_one[0])
        assert torch.equal(features_batched[batch_index], features_one[0])
        assert torch.equal(timepoints_batched[batch_index], timepoints_one[0])
        assert torch.equal(knn_batched[batch_index], knn_one[0])


def test_make_full_model_inputs_tiled_layout_is_deterministic_in_bounds():
    """The optional tiled layout is deterministic and stays inside its square."""
    sequence_length = 32
    window = 10
    side_length = 10.0 * torch.tensor(
        (sequence_length + window - 1) // window
    ).sqrt()

    first = make_full_model_inputs(
        batch_size=1,
        seq_len=sequence_length,
        window=window,
        coord_layout="tiled",
        device="cpu",
        seed=7,
    )
    second = make_full_model_inputs(
        batch_size=1,
        seq_len=sequence_length,
        window=window,
        coord_layout="tiled",
        device="cpu",
        seed=7,
    )

    assert torch.equal(first[0], second[0])
    spatial_coords = first[0][..., 1:]
    assert float(spatial_coords.min()) >= 0.0
    assert float(spatial_coords.max()) <= float(side_length)


def test_full_model_registry_has_four_keys_and_methods_are_unchanged():
    """The additive registry exposes four variants without touching methods."""
    assert set(FULL_MODEL_VARIANT_REGISTRY) == EXPECTED_VARIANT_KEYS
    assert set(METHOD_REGISTRY) == EXPECTED_METHOD_KEYS

    for variant_key, variant_entry in FULL_MODEL_VARIANT_REGISTRY.items():
        assert variant_entry.registry_key == variant_key
        assert variant_entry.class_path.startswith(
            "trackastra.model.model_parts."
        )

    assert (
        FULL_MODEL_VARIANT_REGISTRY["relative_pos"].needs_dist_2d is False
    )
    assert FULL_MODEL_VARIANT_REGISTRY["cached_dist"].needs_dist_2d is True
    for sparse_key in ("sparse_mask_k16", "sparse_gather_k16"):
        sparse_entry = FULL_MODEL_VARIANT_REGISTRY[sparse_key]
        assert sparse_entry.needs_knn is True
        assert sparse_entry.knn_neighbors == 16


@pytest.mark.skipif(
    not CHECKPOINT_PATH.is_dir(),
    reason=f"Checkpoint not found: {CHECKPOINT_PATH}",
)
@pytest.mark.parametrize("variant_key", sorted(EXPECTED_VARIANT_KEYS))
def test_attention_swap_loads_every_required_weight(variant_key):
    """Each variant replaces all layers and copies every checkpoint weight."""
    model = full_grid.load_full_trackastra_model(
        CHECKPOINT_PATH, device=torch.device("cpu"), dtype=torch.float32
    )
    transformer = model.transformer
    transformer.eval()

    encoder_source_states = [
        layer.attn.state_dict() for layer in transformer.encoder
    ]
    decoder_source_states = [
        layer.attn.state_dict() for layer in transformer.decoder
    ]

    report = full_grid.swap_full_model_attention(transformer, variant_key)

    expected_encoder_layers = transformer.config["num_encoder_layers"]
    expected_decoder_layers = transformer.config["num_decoder_layers"]
    assert expected_encoder_layers == 6
    assert expected_decoder_layers == 6
    assert report.encoder_layers_swapped == expected_encoder_layers
    assert report.decoder_layers_swapped == expected_decoder_layers
    assert report.missing_weight_keys == ()
    assert report.unexpected_weight_keys == ()
    assert report.weight_keys_loaded > 0

    expected_class = resolve_class(
        FULL_MODEL_VARIANT_REGISTRY[variant_key].class_path
    )
    for layer in list(transformer.encoder) + list(transformer.decoder):
        assert isinstance(layer.attn, expected_class)
        assert layer.positional_bias == transformer.config[
            "attn_positional_bias"
        ]

    source_states = encoder_source_states + decoder_source_states
    target_modules = list(transformer.encoder) + list(transformer.decoder)
    for source_state, target_layer in zip(source_states, target_modules):
        target_state = target_layer.attn.state_dict()
        assert set(source_state) == set(target_state)
        for key, source_value in source_state.items():
            assert torch.equal(target_state[key], source_value)

        required_keys = full_grid.audit_required_weight_keys(
            target_layer.attn
        )
        for prefix in REQUIRED_WEIGHT_PREFIXES:
            assert any(
                required_key.startswith(prefix)
                for required_key in required_keys
            )


@pytest.mark.skipif(
    not CHECKPOINT_PATH.is_dir(),
    reason=f"Checkpoint not found: {CHECKPOINT_PATH}",
)
def test_full_model_fp32_sanity_equivalence():
    """SPEC 0062 F2: fp32 CPU outputs match for the attention swaps.

    ``run_sanity_checks`` reloads every model independently and raises with
    the measured differences if any comparison misses the required
    ``atol=1e-4``/``rtol=1e-3`` tolerance, so the assertions below also pin
    the expected comparison set and tolerance values.
    """
    sanity_report = full_grid.run_sanity_checks(
        checkpoint_path=CHECKPOINT_PATH,
        device=torch.device("cpu"),
        seq_len=128,
        batch_size=1,
        seed=42,
    )

    assert sanity_report.device.type == "cpu"
    assert sanity_report.seq_len == 128
    assert sanity_report.batch_size == 1

    expected_labels = {
        "relative_pos_vs_cached_dist.raw_A",
        "relative_pos_vs_cached_dist.normalized_A",
        "sparse_mask_k16_vs_sparse_gather_k16.raw_A",
    }
    comparisons_by_label = {
        comparison.label: comparison
        for comparison in sanity_report.comparisons
    }
    assert set(comparisons_by_label) == expected_labels

    for comparison in sanity_report.comparisons:
        assert comparison.passed, (
            f"{comparison.label} failed: "
            f"max_abs_diff={comparison.max_absolute_difference:.6e}, "
            f"max_rel_diff={comparison.max_relative_difference:.6e}"
        )
        assert comparison.atol == pytest.approx(1e-4)
        assert comparison.rtol == pytest.approx(1e-3)


def test_component_timer_cpu_start_stop_elapsed_is_sane():
    """ComponentTimer records CPU boundaries in milliseconds and can clear.

    The test forces the CPU backend so it is independent of CUDA availability.
    A short sleep supplies a lower bound that cannot be produced by an
    accidental nanosecond/second unit mix-up, and the upper bound catches a
    runaway or negative interval.
    """
    timer = ComponentTimer(use_cuda=False)
    assert timer.use_cuda is False

    timer.start("interval_start")
    time.sleep(0.02)
    timer.stop("interval_end")
    elapsed_ms = timer.elapsed_ms("interval_start", "interval_end")

    assert isinstance(elapsed_ms, float)
    assert 10.0 <= elapsed_ms < 5000.0

    timer.start("second_start")
    time.sleep(0.01)
    timer.start("second_end")
    start_to_start_ms = timer.elapsed_ms("second_start", "second_end")
    assert 5.0 <= start_to_start_ms < 5000.0

    timer.clear()
    with pytest.raises(KeyError):
        timer.elapsed_ms("interval_start", "interval_end")


def test_raw_csv_header_matches_spec_4_1_contract():
    """The raw CSV header is the complete SPEC 0062 §4.1 column contract."""
    assert full_grid.RAW_CSV_HEADER == EXPECTED_RAW_CSV_HEADER
    assert len(full_grid.RAW_CSV_HEADER) == len(
        set(full_grid.RAW_CSV_HEADER)
    )

    for column in EXPECTED_OK_TIMING_COLUMNS:
        assert column in full_grid.RAW_CSV_HEADER
    for column in EXPECTED_LINKER_TIMING_COLUMNS:
        assert column in full_grid.RAW_CSV_HEADER

    base_row = full_grid._base_raw_row(
        variant_key="relative_pos",
        sequence_length=128,
        batch_size=1,
        repeat_index=0,
        status="ok",
        checkpoint_path=Path("checkpoints/k16"),
        dtype=torch.float32,
        seed=42,
        knn_neighbors=16,
        coord_layout="uniform",
        linker_mode="greedy",
        error="",
    )
    assert set(base_row) == set(EXPECTED_RAW_CSV_HEADER)


@pytest.mark.skipif(
    not CHECKPOINT_PATH.is_dir(),
    reason=f"Checkpoint not found: {CHECKPOINT_PATH}",
)
def test_cpu_smoke_run_grid_writes_one_ok_row_per_variant(tmp_path):
    """CPU ``run_grid`` at N=128 writes one successful linker row per variant.

    This exercises the timing, CSV, and linker paths end-to-end without the
    primary output paths: one repeat, one warmup, ``B=1``, CPU fp32. Every
    row must carry ``linker_status="ok"``, populated linker stage cells, and a
    ``total_ms`` that includes the linker in addition to the forward and
    normalization intervals.
    """
    sequence_length = 128
    result = full_grid.run_grid(
        checkpoint_path=CHECKPOINT_PATH,
        variant_keys=sorted(EXPECTED_VARIANT_KEYS),
        sequence_lengths=[sequence_length],
        batch_sizes=[1],
        seed=42,
        device=torch.device("cpu"),
        dtype=torch.float32,
        coord_layout="uniform",
        repeats=1,
        warmup=1,
        linker_mode="greedy",
        out_path=tmp_path / "smoke.csv",
        env_json_path=tmp_path / "smoke_env.json",
        summary_path=tmp_path / "smoke_summary.csv",
        knn_neighbors=16,
        command="pytest smoke",
    )

    assert result.out_path == tmp_path / "smoke.csv"
    assert len(result.rows) == len(EXPECTED_VARIANT_KEYS)
    assert {row["variant"] for row in result.rows} == EXPECTED_VARIANT_KEYS

    for row in result.rows:
        assert row["N"] == sequence_length
        assert row["B"] == 1
        assert row["repeat_idx"] == 0
        assert row["status"] == "ok"
        assert row["error"] == ""
        assert row["linker_status"] == "ok"
        for column in EXPECTED_OK_TIMING_COLUMNS:
            assert row[column] not in ("", None)
            assert float(row[column]) >= 0.0
        for column in EXPECTED_LINKER_TIMING_COLUMNS:
            assert row[column] not in ("", None)
            assert float(row[column]) >= 0.0
        assert float(row["linker_ms"]) > 0.0
        assert float(row["linker_ms"]) == pytest.approx(
            float(row["linker_build_graph_ms"])
            + float(row["linker_greedy_ms"]),
            rel=1e-12,
            abs=1e-12,
        )
        assert float(row["total_ms"]) == pytest.approx(
            float(row["full_forward_ms"])
            + float(row["normalize_ms"])
            + float(row["linker_ms"]),
            rel=1e-12,
            abs=1e-12,
        )

    with open(result.out_path, newline="") as csv_file:
        reader = csv.reader(csv_file)
        written_header = next(reader)
        written_rows = list(reader)
    assert written_header == EXPECTED_RAW_CSV_HEADER
    assert len(written_rows) == len(EXPECTED_VARIANT_KEYS)
    status_column_index = EXPECTED_RAW_CSV_HEADER.index("status")
    assert {
        written_row[status_column_index] for written_row in written_rows
    } == {"ok"}

    with open(result.env_json_path) as env_file:
        environment_record = json.load(env_file)
    assert environment_record["variants"] == sorted(EXPECTED_VARIANT_KEYS)
    assert environment_record["Ns"] == [sequence_length]
    assert environment_record["batch_sizes"] == [1]
    assert environment_record["repeats"] == 1
    assert environment_record["warmup"] == 1
    assert environment_record["dtype"] == "fp32"
    assert environment_record["device"] == "cpu"
    assert environment_record["checkpoint_path"] == str(CHECKPOINT_PATH)
    assert environment_record["checkpoint_sha256"]
    assert environment_record["model_config"]["d_model"] == 320
    assert environment_record["linker_mode"] == "greedy"
