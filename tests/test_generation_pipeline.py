"""Generation pipeline tests (ported from compos3d_dp).

Tests the inference / scene generation pipeline: loading a hypothesis bank,
selecting hypotheses by UCB score, calling the LLM (mock), and scoring the
resulting SceneProgram with the heuristic critic.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from compos3d.config import GeneratorConfig, CriticConfig
from compos3d.llm.scene_llm import build_scene_llm
from compos3d.evaluation.critic import build_scene_critic, evaluate_scene_program
from compos3d.models import HypothesisRecord, SceneProgram, AssetSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    return build_scene_llm(GeneratorConfig(provider="mock"))


@pytest.fixture
def heuristic_critic():
    return build_scene_critic(CriticConfig(mode="heuristic"))


@pytest.fixture
def frozen_bank(tmp_path) -> Path:
    """Write a small frozen hypothesis bank to disk and return the path."""
    bank = [
        {
            "hypothesis_id": "h1",
            "text": "anchor the composition around dining_table in a dining_room",
            "room_type": "dining_room",
            "reward": 0.85,
            "accuracy": 0.90,
            "mean_score": 0.87,
            "num_visits": 8,
            "num_successes": 7,
            "generation_round": 1,
            "source_example_ids": ["dr_001"],
            "support_example_ids": ["dr_001"],
            "applicability_tags": [],
            "failure_tags": [],
        },
        {
            "hypothesis_id": "h2",
            "text": "surround the dining_table with chairs in a dining_room",
            "room_type": "dining_room",
            "reward": 0.75,
            "accuracy": 0.80,
            "mean_score": 0.77,
            "num_visits": 5,
            "num_successes": 4,
            "generation_round": 1,
            "source_example_ids": ["dr_002"],
            "support_example_ids": ["dr_002"],
            "applicability_tags": [],
            "failure_tags": [],
        },
    ]
    p = tmp_path / "hypothesis_bank.json"
    p.write_text(json.dumps(bank, indent=2))
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.generation
@pytest.mark.unit
def test_generator_initialization(mock_llm) -> None:
    """Mock LLM is initialised with the correct provider name."""
    assert mock_llm.provider_name == "mock"


@pytest.mark.generation
@pytest.mark.unit
def test_generator_blender_code_generation(mock_llm) -> None:
    """Mock LLM returns a valid SceneProgram for a simple prompt."""
    sp = mock_llm.generate_scene_program(
        prompt="a dining room with a table and two chairs",
        room_type="dining_room",
        selected_hypotheses=["anchor around dining_table"],
    )
    assert isinstance(sp, SceneProgram)
    assert sp.room_type == "dining_room"
    assert len(sp.assets) > 0


@pytest.mark.generation
@pytest.mark.unit
def test_generator_scene_evaluation(mock_llm, heuristic_critic) -> None:
    """Heuristic critic scores a generated SceneProgram in [0, 1]."""
    sp = mock_llm.generate_scene_program(
        prompt="a cozy living room with a sofa and a lamp",
        room_type="living_room",
        selected_hypotheses=[],
    )
    score = evaluate_scene_program(sp, critic=heuristic_critic)
    for field in ("validity", "prompt_adherence", "asset_precision", "asset_recall",
                  "room_match", "overall"):
        v = getattr(score, field)
        assert 0.0 <= v <= 1.0, f"{field} out of range: {v}"


@pytest.mark.generation
@pytest.mark.unit
def test_generator_api(mock_llm) -> None:
    """Mock LLM exposes the required interface methods."""
    assert callable(getattr(mock_llm, "generate_scene_program", None))
    assert callable(getattr(mock_llm, "generate_hypotheses", None))


@pytest.mark.generation
@pytest.mark.unit
def test_generator_respects_room_type(mock_llm) -> None:
    """The generated SceneProgram's room_type matches the requested one."""
    for room in ("dining_room", "living_room", "bedroom"):
        sp = mock_llm.generate_scene_program(
            prompt=f"a nice {room.replace('_', ' ')}",
            room_type=room,
            selected_hypotheses=[],
        )
        assert sp.room_type == room


@pytest.mark.generation
@pytest.mark.integration
def test_generation_end_to_end(frozen_bank: Path, tmp_path: Path) -> None:
    """Full inference: load bank → select hypotheses → generate → score."""
    from compos3d.hypothesis.engine import run_vertical_inference

    result = run_vertical_inference(
        bank_path=frozen_bank,
        prompt="a minimalist dining room with a glass table and two chairs",
        output_dir=tmp_path / "inference",
        llm_provider="mock",
        render_scene=False,
    )

    assert "scene_program_path" in result
    assert "critic_score_path" in result
    assert "selected_hypotheses" in result

    sp_path = Path(result["scene_program_path"])
    assert sp_path.exists()

    sp_data = json.loads(sp_path.read_text())
    assert sp_data["room_type"] == "dining_room"

    score_path = Path(result["critic_score_path"])
    assert score_path.exists()
    score_data = json.loads(score_path.read_text())
    assert 0.0 <= score_data["overall"] <= 1.0
