"""Unit tests for `compos3d.config` normalization and validation behavior.

Why these tests:
- Config loading is a central dependency for training and inference, so small
  regressions here can silently affect many flows.
- The Bedrock/VLM branches contain important auto-fill and fail-loud behavior
  that is not covered well by the current suite.

Edge cases covered:
- Missing config path -> default experiment config stays usable.
- Bedrock generator without model id -> default text model is inserted.
- VLM critic without explicit model id -> default vision model is inserted.
- VLM critic with non-Bedrock provider -> raises a clear ValueError.
"""

from __future__ import annotations

import pytest

from compos3d.config import (
    DEFAULT_TEXT_MODEL_ID,
    DEFAULT_VISION_MODEL_ID,
    load_experiment_config,
)


class _FakePath:
    def __init__(self, payload: dict) -> None:
        import json

        self.payload = json.dumps(payload)

    def read_text(self) -> str:
        return self.payload


def test_load_experiment_config_none_returns_independent_default_copy() -> None:
    cfg1 = load_experiment_config(None)
    cfg2 = load_experiment_config(None)

    assert cfg1.generator.provider == "mock"
    assert cfg1.critic.mode == "heuristic"

    cfg1.training.top_k = 99
    assert cfg2.training.top_k != 99


def test_load_experiment_config_autofills_bedrock_generator_model_id() -> None:
    cfg = load_experiment_config(_FakePath({"generator": {"provider": "bedrock"}}))
    assert cfg.generator.model_id == DEFAULT_TEXT_MODEL_ID


def test_load_experiment_config_vlm_sets_bedrock_defaults() -> None:
    cfg = load_experiment_config(
        _FakePath(
            {
                "generator": {"provider": "bedrock"},
                "critic": {"mode": "vlm", "provider": "bedrock"},
            }
        )
    )
    assert cfg.critic.provider == "bedrock"
    assert cfg.critic.model_id == DEFAULT_VISION_MODEL_ID


def test_load_experiment_config_vlm_rejects_non_bedrock_provider() -> None:
    with pytest.raises(ValueError, match="VLM critic requires provider='bedrock'"):
        load_experiment_config(
            _FakePath(
                {
                    "generator": {"provider": "mock"},
                    "critic": {"mode": "vlm", "provider": "mock"},
                }
            )
        )
