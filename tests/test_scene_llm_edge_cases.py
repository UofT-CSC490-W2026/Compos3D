"""Edge-case unit tests for `compos3d.llm.scene_llm`.

Provider routing, Bedrock response parsing, structured-output validation, and
determinism knobs for scene/hypothesis generation.

Edge cases covered:
- Provider selection branches and invalid-provider errors.
- Bedrock response parsing failures and graceful exception mapping.
- Structured output validation failures for hypotheses and scene programs.
- Determinism controls by checking `temperature`/`maxTokens` forwarding.

Expected outcomes:
- Unsupported providers raise `ValueError`.
- Invalid/malformed model responses raise `StructuredOutputError`.
- Credential-related transport errors raise `LLMUnavailableError`.
- Inference config uses configured deterministic values.
"""

from __future__ import annotations

import pytest

from compos3d.config import GeneratorConfig
from compos3d.llm import scene_llm
from compos3d.llm.scene_llm import (
    BedrockSceneLLM,
    LLMUnavailableError,
    StructuredOutputError,
    build_scene_llm,
)
from compos3d.models import TrainingExample


def test_build_scene_llm_provider_selection_and_invalid() -> None:
    assert build_scene_llm("mock").provider_name == "mock"
    assert build_scene_llm(GeneratorConfig(provider="mock")).provider_name == "mock"
    with pytest.raises(ValueError, match="Unsupported llm provider"):
        build_scene_llm("unknown")


def test_bedrock_run_json_prompt_invalid_output_raises_structured_error() -> None:
    llm = BedrockSceneLLM(GeneratorConfig(provider="bedrock"))

    class _FakeClient:
        def converse(self, **kwargs):
            return {"bad": "shape"}

    llm.client = _FakeClient()
    with pytest.raises(StructuredOutputError, match="invalid JSON output"):
        llm._run_json_prompt("hello")  # noqa: SLF001


def test_bedrock_run_json_prompt_credential_error_maps_to_unavailable() -> None:
    llm = BedrockSceneLLM(GeneratorConfig(provider="bedrock"))

    class _FakeClient:
        def converse(self, **kwargs):
            raise RuntimeError("session has expired")

    llm.client = _FakeClient()
    with pytest.raises(LLMUnavailableError, match="Reauthenticate with aws login"):
        llm._run_json_prompt("hello")  # noqa: SLF001


def test_generate_hypotheses_bad_payload_raises() -> None:
    llm = BedrockSceneLLM(GeneratorConfig(provider="bedrock"))
    llm._run_json_prompt = lambda _p: {"hypotheses": "not-a-list"}  # noqa: SLF001
    with pytest.raises(StructuredOutputError, match="hypotheses list"):
        llm.generate_hypotheses(
            "bedroom",
            [
                TrainingExample(
                    example_id="e1",
                    room_type="bedroom",
                    prompt="p",
                    required_assets=["bed"],
                )
            ],
            num_hypotheses=1,
        )


def test_generate_scene_program_normalization_failure_raises() -> None:
    llm = BedrockSceneLLM(GeneratorConfig(provider="bedrock"))
    llm._run_json_prompt = lambda _p: {"room_type": "living_room"}  # noqa: SLF001
    with pytest.raises(StructuredOutputError, match="expected 'bedroom'"):
        llm.generate_scene_program(
            prompt="p", room_type="bedroom", selected_hypotheses=["h"]
        )


def test_determinism_controls_forwarded_inference_config() -> None:
    captured = {}
    llm = BedrockSceneLLM(
        GeneratorConfig(provider="bedrock", max_tokens=321, temperature=0.0)
    )

    class _FakeClient:
        def converse(self, **kwargs):
            captured.update(kwargs)
            return {
                "output": {"message": {"content": [{"text": '{"hypotheses":["h1"]}'}]}}
            }

    llm.client = _FakeClient()
    llm.generate_hypotheses(
        "bedroom",
        [
            TrainingExample(
                example_id="e1",
                room_type="bedroom",
                prompt="p",
                required_assets=["bed"],
            )
        ],
        num_hypotheses=1,
    )
    assert captured["inference_config"]["temperature"] == 0.0
    assert captured["inference_config"]["maxTokens"] == 321
