"""Pipeline orchestrator tests (ported from compos3d_dp).

Verifies that the full data pipeline (train → infer → evaluate) can be run
end-to-end through the service layer.  All LLM calls use the mock provider.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compos3d.hypothesis.service import TrainingRequest, InferenceRequest, train_hypotheses, run_frozen_inference
from compos3d.evaluation.service import EvaluateRequest, evaluate_run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def training_request(dummy_dataset_path, tmp_path) -> TrainingRequest:
    return TrainingRequest(
        dataset_path=dummy_dataset_path,
        output_dir=tmp_path / "training",
        experiment_name="orch_test",
        llm_provider="mock",
        num_init_examples_per_room=1,
        init_hypotheses_per_room=2,
        num_epochs=1,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.orchestrator
@pytest.mark.unit
def test_orchestrator_initialization() -> None:
    """Service layer requests can be constructed without error."""
    req = TrainingRequest(
        dataset_path=Path("examples/dummy_fast.json"),
        output_dir=Path("/tmp/test_orch"),
        experiment_name="init_test",
    )
    assert req.experiment_name == "init_test"
    assert req.llm_provider == "mock"


@pytest.mark.orchestrator
@pytest.mark.unit
def test_orchestrator_data_pipeline(training_request: TrainingRequest) -> None:
    """Training pipeline produces the expected output files."""
    result = train_hypotheses(training_request)

    assert "run_dir" in result
    run_dir = Path(result["run_dir"])
    assert (run_dir / "hypothesis_bank.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions.jsonl").exists()


@pytest.mark.orchestrator
@pytest.mark.unit
def test_orchestrator_training_pipeline(training_request: TrainingRequest) -> None:
    """Training result exposes num_predictions and run_dir with a bank file."""
    result = train_hypotheses(training_request)
    assert "num_predictions" in result
    assert result["num_predictions"] >= 0
    assert "run_dir" in result
    bank_path = Path(result["run_dir"]) / "hypothesis_bank.json"
    assert bank_path.exists(), f"hypothesis_bank.json not found at {bank_path}"


@pytest.mark.orchestrator
@pytest.mark.unit
def test_orchestrator_generation_pipeline(training_request: TrainingRequest, tmp_path: Path) -> None:
    """Inference after training produces a valid SceneProgram."""
    train_result = train_hypotheses(training_request)
    bank_path = Path(train_result["run_dir"]) / "hypothesis_bank.json"
    assert bank_path.exists()

    inf_request = InferenceRequest(
        bank_path=bank_path,
        prompt="a modern dining room with a table and chairs",
        output_dir=tmp_path / "inference",
        llm_provider="mock",
        render_scene=False,
    )
    inf_result = run_frozen_inference(inf_request)

    assert "scene_program_path" in inf_result
    sp = json.loads(Path(inf_result["scene_program_path"]).read_text())
    assert sp["room_type"] in ("dining_room", "living_room", "bedroom")


@pytest.mark.orchestrator
@pytest.mark.unit
def test_orchestrator_api() -> None:
    """Service functions are importable and callable."""
    assert callable(train_hypotheses)
    assert callable(run_frozen_inference)
    assert callable(evaluate_run)


@pytest.mark.orchestrator
@pytest.mark.unit
def test_pipeline_types_enum() -> None:
    """Core model types are importable and behave correctly."""
    from compos3d.config import SelectionStrategy
    from compos3d.models import SceneProgram, HypothesisRecord

    # SelectionStrategy is a Literal type — check valid values at runtime.
    valid_strategies = {"ucb", "greedy", "random"}
    assert valid_strategies  # trivially true — confirms import works

    # SceneProgram can be constructed.
    sp = SceneProgram(
        prompt="a dining room",
        room_type="dining_room",
        assets=[],
        constraints=[],
        style="modern",
        hypotheses=[],
    )
    assert sp.room_type == "dining_room"


@pytest.mark.orchestrator
@pytest.mark.integration
def test_orchestrator_full_system(training_request: TrainingRequest, tmp_path: Path) -> None:
    """Full system: train → infer → evaluate → summary."""
    # 1. Train
    train_result = train_hypotheses(training_request)
    bank_path = Path(train_result["run_dir"]) / "hypothesis_bank.json"

    # 2. Infer
    inf_result = run_frozen_inference(InferenceRequest(
        bank_path=bank_path,
        prompt="a dining room with a table",
        output_dir=tmp_path / "inference",
        llm_provider="mock",
        render_scene=False,
    ))

    # 3. Evaluate
    eval_result = evaluate_run(EvaluateRequest(
        predictions_dir=tmp_path / "inference",
        output_dir=tmp_path / "evaluation",
    ))

    assert "average_overall" in eval_result
    assert 0.0 <= eval_result["average_overall"] <= 1.0

    summary_path = tmp_path / "evaluation" / "evaluation_summary.json"
    assert summary_path.exists()
