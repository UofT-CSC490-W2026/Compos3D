from __future__ import annotations

import json
from pathlib import Path

import pytest

from compos3d.data.dataset import load_training_dataset
from compos3d.evaluation.service import EvaluateRequest, evaluate_run
from compos3d.hypothesis.engine import train_vertical_slice


_GOLDEN_DIR = Path(__file__).parent / "integration_test_training"
_MANIFEST_PATH_KEYS = (
    "bank_path",
    "predictions_path",
    "training_trace_path",
    "failed_scene_bank_path",
)


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _normalize_manifest_paths(payload: dict) -> dict:
    normalized = dict(payload)
    for key in _MANIFEST_PATH_KEYS:
        normalized[key] = Path(normalized[key]).name
    return normalized


def _normalize_result(payload: dict) -> dict:
    normalized = dict(payload)
    normalized["run_dir"] = Path(normalized["run_dir"]).name
    return normalized


def _run_training(
    *, dataset_path: Path, output_dir: Path, experiment_name: str
) -> tuple[dict, Path]:
    result = train_vertical_slice(
        dataset_path=dataset_path,
        output_dir=output_dir,
        experiment_name=experiment_name,
        llm_provider="mock",
        num_init_examples_per_room=1,
        init_hypotheses_per_room=3,
        top_k=2,
        alpha=0.5,
        max_num_hypotheses_per_room=8,
        num_wrong_scale=0.8,
        update_batch_size=1,
        num_hypotheses_to_update=1,
        update_hypotheses_per_batch=2,
        only_best_hypothesis=False,
        num_epochs=2,
        success_threshold=0.65,
        save_every_n_examples=5,
        selection_strategy="ucb",
        use_repair=True,
        baseline_mode=None,
        seed=42,
    )
    return result, Path(result["run_dir"])


@pytest.mark.integration
@pytest.mark.training
def test_training_smoke_matches_golden_artifacts(
    dummy_dataset_path: Path, tmp_path: Path
) -> None:
    result, run_dir = _run_training(
        dataset_path=dummy_dataset_path,
        output_dir=tmp_path / "training",
        experiment_name="smoke_training",
    )

    assert run_dir.exists()

    assert result["metrics"] == _read_json(_GOLDEN_DIR / "metrics.json")
    assert result["num_predictions"] == 6
    assert result["num_hypotheses"] == 6
    assert result["num_regeneration_events"] == 0

    assert _read_json(run_dir / "experiment_config.json") == _read_json(
        _GOLDEN_DIR / "experiment_config.json"
    )
    assert _read_json(run_dir / "metrics.json") == _read_json(
        _GOLDEN_DIR / "metrics.json"
    )
    assert _read_json(run_dir / "hypothesis_bank.json") == _read_json(
        _GOLDEN_DIR / "hypothesis_bank.json"
    )
    assert _read_jsonl(run_dir / "predictions.jsonl") == _read_jsonl(
        _GOLDEN_DIR / "predictions.jsonl"
    )
    assert _read_jsonl(run_dir / "training_trace.jsonl") == _read_jsonl(
        _GOLDEN_DIR / "training_trace.jsonl"
    )

    actual_manifest = _read_json(run_dir / "manifest.json")
    expected_manifest = _read_json(_GOLDEN_DIR / "manifest.json")
    assert _normalize_manifest_paths(actual_manifest) == _normalize_manifest_paths(
        expected_manifest
    )

    for key in _MANIFEST_PATH_KEYS:
        assert Path(actual_manifest[key]).parent == run_dir

    failed_scene_bank = run_dir / "failed_scene_bank.jsonl"
    assert failed_scene_bank.exists()
    assert failed_scene_bank.read_text() == ""

    program_files = sorted((run_dir / "programs").glob("*.json"))
    assert len(program_files) == result["num_predictions"]

    snapshots_dir = run_dir / "bank_snapshots"
    assert (snapshots_dir / "hypothesis_bank_initial.json").exists()
    assert (snapshots_dir / "hypothesis_bank_final.json").exists()


@pytest.mark.integration
@pytest.mark.training
def test_training_smoke_is_deterministic_across_reruns(
    dummy_dataset_path: Path, tmp_path: Path
) -> None:
    result_a, run_dir_a = _run_training(
        dataset_path=dummy_dataset_path,
        output_dir=tmp_path / "run_a",
        experiment_name="smoke_training",
    )
    result_b, run_dir_b = _run_training(
        dataset_path=dummy_dataset_path,
        output_dir=tmp_path / "run_b",
        experiment_name="smoke_training",
    )

    assert _normalize_result(result_a) == _normalize_result(result_b)
    assert _read_json(run_dir_a / "experiment_config.json") == _read_json(
        run_dir_b / "experiment_config.json"
    )
    assert _read_json(run_dir_a / "metrics.json") == _read_json(
        run_dir_b / "metrics.json"
    )
    assert _read_json(run_dir_a / "hypothesis_bank.json") == _read_json(
        run_dir_b / "hypothesis_bank.json"
    )
    assert _read_jsonl(run_dir_a / "predictions.jsonl") == _read_jsonl(
        run_dir_b / "predictions.jsonl"
    )
    assert _read_jsonl(run_dir_a / "training_trace.jsonl") == _read_jsonl(
        run_dir_b / "training_trace.jsonl"
    )
    assert _normalize_manifest_paths(_read_json(run_dir_a / "manifest.json")) == (
        _normalize_manifest_paths(_read_json(run_dir_b / "manifest.json"))
    )


@pytest.mark.integration
@pytest.mark.training
def test_training_smoke_artifacts_are_internally_consistent(
    dummy_dataset_path: Path, tmp_path: Path
) -> None:
    dataset = load_training_dataset(dummy_dataset_path)
    result, run_dir = _run_training(
        dataset_path=dummy_dataset_path,
        output_dir=tmp_path / "training",
        experiment_name="smoke_training",
    )

    metrics = _read_json(run_dir / "metrics.json")
    manifest = _read_json(run_dir / "manifest.json")
    bank = _read_json(run_dir / "hypothesis_bank.json")
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    training_trace = _read_jsonl(run_dir / "training_trace.jsonl")
    failed_scene_bank = _read_jsonl(run_dir / "failed_scene_bank.jsonl")
    program_files = sorted((run_dir / "programs").glob("*.json"))
    snapshot_files = sorted((run_dir / "bank_snapshots").glob("*.json"))

    by_room: dict[str, list] = {}
    for example in dataset.examples:
        by_room.setdefault(example.room_type, []).append(example)
    seed_skips = sum(
        max(
            1,
            min(manifest["hypothesis_loop_config"]["num_init_examples_per_room"], len(examples)),
        )
        for examples in by_room.values()
    )
    expected_predictions = (
        len(dataset.examples) * manifest["hypothesis_loop_config"]["num_epochs"]
    ) - seed_skips
    bank_ids = {record["hypothesis_id"] for record in bank}
    trace_example_ids = [row["example_id"] for row in training_trace]
    prediction_example_ids = [row["example_id"] for row in predictions]

    assert metrics["num_predictions"] == expected_predictions
    assert result["num_predictions"] == expected_predictions
    assert manifest["num_predictions"] == expected_predictions
    assert len(predictions) == expected_predictions
    assert len(training_trace) == expected_predictions
    assert len(program_files) == expected_predictions
    assert len(failed_scene_bank) == 0
    assert manifest["num_regeneration_events"] == 0
    assert result["num_regeneration_events"] == 0

    assert trace_example_ids == prediction_example_ids
    assert set(trace_example_ids) == {example.example_id for example in dataset.examples}
    assert {record["room_type"] for record in bank} == {
        example.room_type for example in dataset.examples
    }
    assert all(
        row["combined_score"] == pred["critic_score"]["overall"]
        for row, pred in zip(training_trace, predictions, strict=True)
    )
    assert all(
        set(trace_row["selected_hypothesis_ids"]).issubset(bank_ids)
        for trace_row in training_trace
    )
    assert all(
        len(trace_row["selected_hypothesis_ids"])
        == len(trace_row["selected_hypotheses"])
        for trace_row in training_trace
    )
    assert all(pred["image_paths"] == [] for pred in predictions)
    assert all(
        pred["critic_score"]["critic_mode"] == "heuristic" for pred in predictions
    )
    assert any(path.name == "hypothesis_bank_initial.json" for path in snapshot_files)
    assert any(path.name == "hypothesis_bank_final.json" for path in snapshot_files)


@pytest.mark.integration
@pytest.mark.training
def test_training_smoke_outputs_are_immediately_evaluable(
    dummy_dataset_path: Path, tmp_path: Path
) -> None:
    result, run_dir = _run_training(
        dataset_path=dummy_dataset_path,
        output_dir=tmp_path / "training",
        experiment_name="smoke_training",
    )

    evaluation = evaluate_run(
        EvaluateRequest(
            predictions_dir=run_dir,
            output_dir=tmp_path / "evaluation",
        )
    )

    assert evaluation == result["metrics"]
    assert _read_json(tmp_path / "evaluation" / "evaluation_summary.json") == _read_json(
        run_dir / "metrics.json"
    )
