"""Edge-case tests for heuristic critic scoring behavior.

Covers:
- Empty predicted assets.
- Reference room-type mismatch.
- Repeated assets in scene program inputs.
- Unsupported room types.
- Contradictory constraints (robustness/no-crash expectation).

Expected outcomes:
- Scores remain bounded in [0, 1].
- Invalid room types and empty assets reduce validity with explanatory notes.
- Duplicate assets do not inflate set-based precision/recall calculations.
"""

from __future__ import annotations

from compos3d.evaluation.critic import _build_heuristic_score
from compos3d.models import AssetSpec, ConstraintSpec, SceneProgram, TrainingExample


def test_critic_handles_empty_assets_with_low_validity_and_note() -> None:
    scene = SceneProgram(prompt="simple bedroom", room_type="bedroom", assets=[])
    score = _build_heuristic_score(scene)
    assert score.validity == 0.2
    assert score.overall <= 1.0
    assert any("No assets predicted." in note for note in score.notes)


def test_critic_marks_mismatched_room_type_against_reference() -> None:
    scene = SceneProgram(
        prompt="bedroom with bed", room_type="living_room", assets=[AssetSpec(asset_type="sofa")]
    )
    ref = TrainingExample(
        example_id="e1",
        room_type="bedroom",
        prompt="bedroom with bed",
        required_assets=["bed"],
    )
    score = _build_heuristic_score(scene, reference_example=ref)
    assert score.room_match == 0.0


def test_critic_uses_unique_assets_for_precision_recall_calculation() -> None:
    scene = SceneProgram(
        prompt="bedroom with bed and lamp",
        room_type="bedroom",
        assets=[AssetSpec(asset_type="bed"), AssetSpec(asset_type="bed"), AssetSpec(asset_type="lamp")],
    )
    ref = TrainingExample(
        example_id="e1",
        room_type="bedroom",
        prompt="x",
        required_assets=["bed", "lamp"],
    )
    score = _build_heuristic_score(scene, reference_example=ref)
    assert score.asset_recall == 1.0
    assert score.asset_precision == 1.0


def test_critic_invalid_room_type_forces_zero_validity() -> None:
    scene = SceneProgram(prompt="odd room", room_type="garage", assets=[AssetSpec(asset_type="bed")])
    score = _build_heuristic_score(scene)
    assert score.validity == 0.0
    assert any("Unsupported room type." in note for note in score.notes)


def test_critic_contradictory_constraints_do_not_crash_heuristic() -> None:
    scene = SceneProgram(
        prompt="bedroom",
        room_type="bedroom",
        assets=[AssetSpec(asset_type="bed", count=1)],
        constraints=[
            ConstraintSpec(text="bed must be in the center"),
            ConstraintSpec(text="bed must be against the wall"),
        ],
    )
    score = _build_heuristic_score(scene)
    assert 0.0 <= score.overall <= 1.0

