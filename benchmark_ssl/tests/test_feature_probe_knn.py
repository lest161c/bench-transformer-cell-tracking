"""Tests for the SPEC 0046 F05 feature-probe CLI slice.

The tests use only synthetic in-memory data.  They cover the CLI help path, a
complete pair -> candidate -> normalization -> training -> evaluation pipeline,
the automatic-K escalation path, the JSON/CSV artifact schemas, LOCO split and
summary construction, and the clear errors for deferred feature rows.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis import feature_probe_knn
from src.analysis.feature_probe_knn import (
    METRIC_KEYS,
    SUMMARY_FIELDNAMES,
    FeatureExtractorContext,
    InsufficientCoverageError,
    _extract_feature_frame,
    _loco_split_specs,
    _summary_row,
    aggregate_coverage_rows,
    build_argument_parser,
    build_candidate_dataset,
    select_protocol_k,
    standardize_datasets,
    train_and_evaluate_probe,
    validate_feature_names,
    validate_protocol,
    write_json,
    write_summary_csv,
)
from src.analysis.knn_candidates import coverage_vs_k

FEATURE_DIMENSION = 3
CANDIDATE_COUNT = 2


def _make_pair_record(pair_id: str, seed: int) -> dict:
    """Build one tiny synthetic feature pair with full K=2 coverage.

    Args:
        pair_id: Identifier used for the record and deterministic ordering.
        seed: NumPy RNG seed for the feature values.

    Returns:
        A feature pair record shaped like the records produced by
        ``extract_feature_pairs``.  Every anchor's true match is its nearest
        candidate, so the K=2 candidate set covers all anchors.
    """
    anchor_coords = np.array(
        [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float32
    )
    candidate_coords = np.array(
        [[0.1, 0.0], [10.1, 0.0], [20.1, 0.0], [50.0, 50.0]],
        dtype=np.float32,
    )
    anchor_labels = np.array([1, 2, 3], dtype=np.int64)
    candidate_labels = np.array([1, 2, 3, 99], dtype=np.int64)

    generator = np.random.default_rng(seed)
    return {
        "feature": "synthetic",
        "condition": "synthetic_condition",
        "experiment": "synthetic_experiment",
        "pair_id": pair_id,
        "anchor_features": generator.normal(
            size=(anchor_coords.shape[0], FEATURE_DIMENSION)
        ).astype(np.float32),
        "candidate_features": generator.normal(
            size=(candidate_coords.shape[0], FEATURE_DIMENSION)
        ).astype(np.float32),
        "anchor_coords": anchor_coords,
        "candidate_coords": candidate_coords,
        "anchor_labels": anchor_labels,
        "candidate_labels": candidate_labels,
    }


def test_help_exits_zero() -> None:
    """``--help`` is handled by argparse and exits successfully."""
    parser = build_argument_parser()
    with pytest.raises(SystemExit) as exit_information:
        parser.parse_args(["--help"])
    assert exit_information.value.code == 0


def test_synthetic_pipeline_pair_to_candidate_to_train_to_evaluate() -> None:
    """The full mid-level pipeline runs on tiny in-memory arrays."""
    train_pair = _make_pair_record("pair_train", seed=1)
    validation_pair = _make_pair_record("pair_validation", seed=2)
    evaluation_pair = _make_pair_record("pair_evaluation", seed=3)

    train_dataset = build_candidate_dataset([train_pair], k=CANDIDATE_COUNT)
    validation_dataset = build_candidate_dataset(
        [validation_pair], k=CANDIDATE_COUNT
    )
    evaluation_dataset = build_candidate_dataset(
        [evaluation_pair], k=CANDIDATE_COUNT
    )

    assert train_dataset["anchor_features"].shape == (
        3,
        CANDIDATE_COUNT,
        FEATURE_DIMENSION,
    )
    assert train_dataset["candidate_features"].shape == (
        3,
        CANDIDATE_COUNT,
        FEATURE_DIMENSION,
    )
    assert train_dataset["target_index"].tolist() == [0, 0, 0]

    (
        standardized_train,
        standardized_validation,
        standardized_evaluation,
        mean,
        standard_deviation,
    ) = standardize_datasets(
        train_dataset, validation_dataset, evaluation_dataset
    )
    assert mean.shape == (FEATURE_DIMENSION,)
    assert standard_deviation.shape == (FEATURE_DIMENSION,)
    assert standardized_train["anchor_features"].shape == (
        3,
        CANDIDATE_COUNT,
        FEATURE_DIMENSION,
    )

    result = train_and_evaluate_probe(
        standardized_train,
        standardized_validation,
        standardized_evaluation,
        probe_name="linear",
        seed=42,
        epochs=3,
        learning_rate=0.01,
        patience=2,
        batch_size=4,
        device="cpu",
    )

    assert 0.0 <= result["threshold"] <= 1.0
    for metric_name in ("top1_accuracy", "precision", "recall", "f1", "nll"):
        assert metric_name in result["metrics"]
    assert result["n_evaluation_anchors"] == 3
    assert result["n_validation_anchors"] == 3
    assert result["epochs_trained"] >= 1
    assert "balanced_accuracy" not in result["metrics"]
    assert "bal_acc" not in result["metrics"]


def test_automatic_k_selection_escalates_when_coverage_is_below_one() -> None:
    """A fold that never reaches full coverage raises instead of proceeding."""
    anchor_coords = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    candidate_coords = np.array([[100.0, 100.0], [200.0, 200.0]], dtype=np.float32)
    anchor_labels = np.array([1, 2], dtype=np.int64)
    candidate_labels = np.array([3, 4], dtype=np.int64)

    pair_rows = coverage_vs_k(
        anchor_coords,
        candidate_coords,
        anchor_labels,
        candidate_labels,
        k_max=2,
    )
    aggregated_rows = aggregate_coverage_rows(
        [pair_rows], [candidate_labels.shape[0]], k_max=2
    )
    assert aggregated_rows[-1]["coverage"] < 1.0

    with pytest.raises(InsufficientCoverageError, match="coverage did not reach"):
        select_protocol_k([aggregated_rows], ["fold_0"])


def test_json_and_csv_writer_schema(tmp_path: Path) -> None:
    """Run JSON, coverage JSON and summary CSV keep the documented schema."""
    required_run_keys = {
        "protocol",
        "feature",
        "probe",
        "seed",
        "k",
        "coverage",
        "threshold",
        "metrics",
        "data_scope",
        "checkpoint",
        "environment",
        "versions",
        "args",
        "per_fold",
    }
    run_payload = {
        "protocol": "frame_cv",
        "feature": "regionprops_7d",
        "probe": "linear",
        "seed": 42,
        "k": 3,
        "coverage": {"protocol_k": 3, "per_fold": []},
        "threshold": 0.5,
        "metrics": {
            "top1_accuracy": {"mean": 1.0, "std": 0.0},
            "precision": {"mean": 1.0, "std": 0.0},
            "recall": {"mean": 1.0, "std": 0.0},
            "f1": {"mean": 1.0, "std": 0.0},
            "nll": {"mean": 0.1, "std": 0.0},
        },
        "data_scope": {"conditions": ["rpsM"]},
        "checkpoint": {"status": "not_applicable"},
        "environment": {"device": "cpu"},
        "versions": {"run_schema": 1},
        "args": {},
        "per_fold": [],
    }
    run_path = tmp_path / "runs" / "regionprops_7d_linear_s42.json"
    write_json(run_path, run_payload)
    loaded_run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    assert required_run_keys <= set(loaded_run_payload)
    assert "bal_acc" not in loaded_run_payload
    assert "balanced_accuracy" not in loaded_run_payload["metrics"]

    coverage_payload = {
        "protocol": "frame_cv",
        "feature_agnostic": True,
        "k_max": 4,
        "protocol_k": 2,
        "folds": [
            {
                "fold": "fold_0",
                "chosen_k": 2,
                "chosen_coverage": 1.0,
                "coverage": [{"k": 1, "coverage": 0.5}, {"k": 2, "coverage": 1.0}],
            }
        ],
    }
    coverage_path = tmp_path / "coverage_vs_k.json"
    write_json(coverage_path, coverage_payload)
    loaded_coverage_payload = json.loads(coverage_path.read_text(encoding="utf-8"))
    assert {
        "protocol",
        "feature_agnostic",
        "k_max",
        "protocol_k",
        "folds",
    } <= set(loaded_coverage_payload)
    assert loaded_coverage_payload["folds"][0]["chosen_k"] == 2

    summary_row = {
        field: (
            "regionprops_7d"
            if field == "feature"
            else "linear"
            if field == "probe"
            else 42
            if field == "seed"
            else 3
            if field in ("n_folds", "protocol_k")
            else 0.5
        )
        for field in SUMMARY_FIELDNAMES
    }
    summary_path = tmp_path / "summary.csv"
    write_summary_csv(summary_path, [summary_row])
    with summary_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames == list(SUMMARY_FIELDNAMES)
        written_rows = list(reader)
    assert len(written_rows) == 1
    assert written_rows[0]["feature"] == "regionprops_7d"
    assert written_rows[0]["probe"] == "linear"
    assert "bal_acc" not in reader.fieldnames


def test_deferred_feature_and_protocol_raise_clear_errors() -> None:
    """SAM2.1 is supported; unknown feature and protocol names still error."""
    assert validate_feature_names(["sam21"]) == ["sam21"]
    assert validate_protocol("loco") == "loco"
    with pytest.raises(ValueError, match="unknown feature row"):
        validate_feature_names(["not_a_feature"])
    with pytest.raises(ValueError, match="unknown protocol"):
        validate_protocol("not_a_protocol")


def test_argument_parser_parses_heldout_conditions() -> None:
    """``--heldout-conditions`` is parsed into the raw comma-separated string."""
    parsed_arguments = build_argument_parser().parse_args(
        ["--heldout-conditions", "b,c"]
    )
    assert parsed_arguments.heldout_conditions == "b,c"


def test_loco_split_specs_hold_out_one_condition_at_a_time() -> None:
    """Each LOCO split evaluates one condition and trains on all others."""
    pairs_by_condition = {
        "a": ["a_pair_1", "a_pair_2"],
        "b": ["b_pair_1", "b_pair_2"],
        "c": ["c_pair_1", "c_pair_2"],
    }
    base_pairs = [
        {"pair_id": pair_id, "condition": condition}
        for condition, pair_ids in pairs_by_condition.items()
        for pair_id in pair_ids
    ]

    split_specs = _loco_split_specs(base_pairs, ["b", "c"])

    assert [split_spec["fold"] for split_spec in split_specs] == ["b", "c"]
    for split_spec in split_specs:
        heldout_condition = split_spec["fold"]
        heldout_pair_ids = set(pairs_by_condition[heldout_condition])
        other_pair_ids = {
            pair_id
            for condition, pair_ids in pairs_by_condition.items()
            if condition != heldout_condition
            for pair_id in pair_ids
        }

        assert set(split_spec["held_out_pair_ids"]) == heldout_pair_ids
        training_pair_ids = set(split_spec["training_pair_ids"])
        assert training_pair_ids.isdisjoint(heldout_pair_ids)
        assert training_pair_ids == other_pair_ids


def test_summary_row_records_loco_protocol_and_overall_condition() -> None:
    """A LOCO overall summary row has the fixed schema and names its protocol."""
    fold_results = [
        {
            "threshold": threshold,
            "metrics": {
                metric_name: float(metric_value)
                for metric_name, metric_value in zip(METRIC_KEYS, metric_values)
            },
        }
        for threshold, metric_values in (
            (0.4, (0.8, 0.7, 0.6, 0.5, 0.2)),
            (0.6, (0.9, 0.8, 0.7, 0.6, 0.1)),
        )
    ]

    summary_row = _summary_row(
        protocol="loco",
        feature_name="regionprops_7d",
        probe_name="linear",
        seed=42,
        protocol_k=3,
        fold_results=fold_results,
        heldout_condition="overall",
    )

    assert summary_row["protocol"] == "loco"
    assert summary_row["heldout_condition"] == "overall"
    assert tuple(summary_row) == SUMMARY_FIELDNAMES
    assert summary_row["n_folds"] == 2
    assert summary_row["threshold_mean"] == pytest.approx(0.5)
    assert summary_row["f1_mean"] == pytest.approx(0.55)


def test_extract_feature_frame_dispatches_dino(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``dino`` dispatches to the patch/embedding helpers without a KeyError."""
    recorded_arguments: dict[str, np.ndarray] = {}

    def fake_extract_patches(img: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """Record the input and return one synthetic patch per centroid."""
        recorded_arguments["img"] = img
        recorded_arguments["coords"] = coords
        return np.zeros((coords.shape[0], 64, 64), dtype=np.float32)

    def fake_compute_dino_embs(patches: np.ndarray) -> np.ndarray:
        """Record the patches and return a distinct synthetic embedding per row."""
        recorded_arguments["patches"] = patches
        return np.arange(
            patches.shape[0] * 3, dtype=np.float32
        ).reshape(patches.shape[0], 3)

    monkeypatch.setattr(
        feature_probe_knn, "extract_patches", fake_extract_patches
    )
    monkeypatch.setattr(
        feature_probe_knn, "compute_dino_embs", fake_compute_dino_embs
    )

    coords = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    labels = np.array([7, 8], dtype=np.int64)
    frame = {
        "img": np.zeros((16, 16), dtype=np.float32),
        "coords": coords,
        "labels": labels,
        "mask": np.zeros((16, 16), dtype=np.int64),
    }

    extracted_coords, extracted_labels, features = _extract_feature_frame(
        "dino", frame, FeatureExtractorContext("cpu")
    )

    assert np.array_equal(extracted_coords, coords)
    assert np.array_equal(extracted_labels, labels)
    assert np.array_equal(recorded_arguments["coords"], coords)
    assert recorded_arguments["patches"].shape == (2, 64, 64)
    assert features.shape == (2, 3)


def test_extract_feature_frame_dispatches_sam21_without_sam2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``sam21`` dispatches to the adapter method without importing ``sam2``."""
    from src.analysis import sam21_features

    recorded_arguments: dict[str, np.ndarray] = {}

    def fake_init(
        self: object,
        device: str = "cuda",
        mode: str = "nearest_patch",
        image_size: tuple[int, int] = (1024, 1024),
    ) -> None:
        """Replace checkpoint loading so the test never needs ``sam2``."""
        self.device = device
        self.mode = mode
        self.image_size = image_size

    def fake_extract_frame(
        self: object, image: np.ndarray, coords: np.ndarray
    ) -> np.ndarray:
        """Record the frame inputs and return one synthetic 256-D row per cell."""
        recorded_arguments["image"] = image
        recorded_arguments["coords"] = coords
        return np.arange(
            coords.shape[0] * 256, dtype=np.float32
        ).reshape(coords.shape[0], 256)

    monkeypatch.setattr(sam21_features.Sam21FeatureExtractor, "__init__", fake_init)
    monkeypatch.setattr(
        sam21_features.Sam21FeatureExtractor, "extract_frame", fake_extract_frame
    )

    coords = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    labels = np.array([7, 8], dtype=np.int64)
    image = np.zeros((16, 16), dtype=np.float32)
    frame = {
        "img": image,
        "coords": coords,
        "labels": labels,
        "mask": np.zeros((16, 16), dtype=np.int64),
    }

    extracted_coords, extracted_labels, features = _extract_feature_frame(
        "sam21", frame, FeatureExtractorContext("cpu")
    )

    assert np.array_equal(extracted_coords, coords)
    assert np.array_equal(extracted_labels, labels)
    assert recorded_arguments["image"] is image
    assert np.array_equal(recorded_arguments["coords"], coords)
    assert features.shape == (2, 256)
    assert features[0, 0] == 0.0


def test_checkpoint_record_sam21_reports_hub_identity() -> None:
    """The ``sam21`` checkpoint block names the Hugging Face hub model."""
    checkpoint_record = FeatureExtractorContext("cpu").checkpoint_record("sam21")

    assert checkpoint_record == {
        "status": "pretrained_hub",
        "feature": "sam21",
        "path": "facebook/sam2.1-hiera-base-plus",
        "md5": None,
        "note": (
            "weights are downloaded from the Hugging Face hub; no "
            "repository-owned checkpoint file is hashed"
        ),
    }
