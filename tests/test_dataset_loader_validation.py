"""Validation-focused tests for training dataset payloads/models.

Malformed examples, required prompts, and schema edge behavior (empty prompt,
non-enum room types).

Edge cases covered:
- Rejection of malformed or incomplete training examples.
- Required prompt field enforcement.
- Current permissive behavior for empty prompts and room-type strings.

Expected outcomes:
- Structurally invalid examples raise `ValidationError`.
- Behavior currently allowed by schema (non-enum room type, empty prompt)
  remains explicitly documented via assertions.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from compos3d.models import TrainingDataset


def test_training_dataset_rejects_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        TrainingDataset.model_validate({"dataset_id": "d1", "examples": [{}]})


def test_training_dataset_rejects_missing_prompt() -> None:
    payload = {
        "dataset_id": "d1",
        "examples": [{"example_id": "e1", "room_type": "bedroom", "required_assets": ["bed"]}],
    }
    with pytest.raises(ValidationError):
        TrainingDataset.model_validate(payload)


def test_training_dataset_empty_prompt_currently_allowed_model_behavior() -> None:
    payload = {
        "dataset_id": "d1",
        "examples": [
            {
                "example_id": "e1",
                "room_type": "bedroom",
                "prompt": "",
                "required_assets": ["bed"],
            }
        ],
    }
    ds = TrainingDataset.model_validate(payload)
    assert ds.examples[0].prompt == ""


def test_training_dataset_room_type_is_not_enum_restricted() -> None:
    payload = {
        "dataset_id": "d1",
        "examples": [
            {
                "example_id": "e1",
                "room_type": "unknown_room",
                "prompt": "x",
                "required_assets": ["bed"],
            }
        ],
    }
    ds = TrainingDataset.model_validate(payload)
    assert ds.examples[0].room_type == "unknown_room"

