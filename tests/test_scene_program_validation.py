"""Validation tests for SceneProgram-related Pydantic models.

Covers:
- Required-field validation for `SceneProgram`, `AssetSpec`, and `ConstraintSpec`.
- Numeric constraints (e.g., positive asset counts).
- Default behavior for `RenderSpec`.
- Malformed nested payload rejection.
- Current model behavior where `room_type` is not enum-restricted.

Expected outcomes:
- Invalid payloads raise `ValidationError`.
- Valid defaults are applied where defined.
- Behavior checks marked as "current behavior" guard against accidental regressions.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from compos3d.models import AssetSpec, ConstraintSpec, RenderSpec, SceneProgram


def test_scene_program_requires_prompt_and_room_type() -> None:
    with pytest.raises(ValidationError):
        SceneProgram.model_validate({"prompt": "Only prompt"})
    with pytest.raises(ValidationError):
        SceneProgram.model_validate({"room_type": "bedroom"})


def test_asset_spec_count_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        AssetSpec(asset_type="bed", count=0)


def test_asset_spec_rejects_invalid_count_type() -> None:
    with pytest.raises(ValidationError):
        AssetSpec(asset_type="chair", count="many")


def test_constraint_spec_requires_text_field() -> None:
    with pytest.raises(ValidationError):
        ConstraintSpec.model_validate({})


def test_render_spec_defaults_are_applied() -> None:
    spec = RenderSpec()
    assert spec.mode == "program_only"


def test_scene_program_accepts_malformed_constraint_shape_as_validation_error() -> None:
    payload = {
        "prompt": "A cozy room",
        "room_type": "living_room",
        "assets": [{"asset_type": "sofa", "count": 1}],
        "constraints": [{"bad_key": "must be center"}],
    }
    with pytest.raises(ValidationError):
        SceneProgram.model_validate(payload)


def test_scene_program_current_model_does_not_enforce_room_enum() -> None:
    # Guard current behavior: room_type is a plain str field in the model.
    program = SceneProgram(prompt="test", room_type="not_a_real_room")
    assert program.room_type == "not_a_real_room"

