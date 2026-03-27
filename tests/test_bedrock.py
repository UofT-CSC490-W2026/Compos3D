"""Unit tests for Bedrock client integration surfaces.

Config resolution, converse payloads, optional fields, critic JSON parsing,
and error mapping for credential vs generic failures.

Edge cases covered:
- Env/default config resolution and explicit override precedence.
- Request payload construction for Bedrock `converse` calls.
- Optional field omission and low-level client exception propagation.
- VLM critic JSON response parsing, score clamping to [0, 1], and mode tagging.
- Error mapping behavior: credential/session failures vs generic request failures.

Expected outcomes:
- Correct payload keys/values are sent to the mocked runtime client.
- Parsed critic scores are bounded and normalized.
- Credential-related failures raise `CriticUnavailableError` with reauth guidance.
- Non-credential failures also raise `CriticUnavailableError` with a generic message.
"""

from __future__ import annotations

import pytest

from compos3d.llm.bedrock import (
    BedrockChatClient,
    BedrockConfig,
    resolve_bedrock_config,
)
from compos3d.config import CriticConfig
from compos3d.evaluation.critic import BedrockVLMCritic, CriticUnavailableError
from compos3d.models import SceneProgram


class _FakeRuntimeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.response = {"ok": True}

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_resolve_bedrock_config_uses_env_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COMPOS3D_BEDROCK_MODEL_ID", "env-model")
    monkeypatch.setenv("COMPOS3D_BEDROCK_REGION", "us-west-2")

    cfg = resolve_bedrock_config()
    assert cfg.model_id == "env-model"
    assert cfg.region_name == "us-west-2"


def test_resolve_bedrock_config_explicit_values_override_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COMPOS3D_BEDROCK_MODEL_ID", "env-model")
    monkeypatch.setenv("COMPOS3D_BEDROCK_REGION", "us-west-2")

    cfg = resolve_bedrock_config(model_id="explicit-model", region_name="eu-west-1")
    assert cfg.model_id == "explicit-model"
    assert cfg.region_name == "eu-west-1"


def test_bedrock_chat_client_payload_construction() -> None:
    fake = _FakeRuntimeClient()
    client = BedrockChatClient(
        config=BedrockConfig(model_id="m1", region_name="r1"),
        runtime_client=fake,
    )

    response = client.converse(
        messages=[{"role": "user", "content": [{"text": "hello"}]}],
        system=[{"text": "system rules"}],
        inference_config={"maxTokens": 100, "temperature": 0.1},
        additional_model_request_fields={"top_k": 40},
    )

    assert response == {"ok": True}
    assert len(fake.calls) == 1
    req = fake.calls[0]
    assert req["modelId"] == "m1"
    assert req["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]
    assert req["system"] == [{"text": "system rules"}]
    assert req["inferenceConfig"] == {"maxTokens": 100, "temperature": 0.1}
    assert req["additionalModelRequestFields"] == {"top_k": 40}


def test_bedrock_chat_client_omits_optional_fields_when_not_given() -> None:
    fake = _FakeRuntimeClient()
    client = BedrockChatClient(
        config=BedrockConfig(model_id="m2", region_name="r2"),
        runtime_client=fake,
    )

    client.converse(messages=[{"role": "user", "content": [{"text": "ping"}]}])
    req = fake.calls[0]
    assert "system" not in req
    assert "inferenceConfig" not in req
    assert "additionalModelRequestFields" not in req


def test_bedrock_chat_client_propagates_runtime_client_error() -> None:
    class _Boom:
        def converse(self, **kwargs):
            raise RuntimeError("network down")

    client = BedrockChatClient(
        config=BedrockConfig(model_id="m3", region_name="r3"),
        runtime_client=_Boom(),
    )

    with pytest.raises(RuntimeError, match="network down"):
        client.converse(messages=[{"role": "user", "content": [{"text": "x"}]}])


def test_bedrock_vlm_response_parsing_and_clamping(tmp_path) -> None:
    image = tmp_path / "view_001.png"
    image.write_bytes(b"fake-png-bytes")

    critic = BedrockVLMCritic(CriticConfig(mode="vlm", provider="bedrock"))

    class _FakeBedrock:
        def converse(self, **kwargs):
            return {
                "output": {
                    "message": {
                        "content": [
                            {
                                "text": """```json
{"validity": 2, "prompt_adherence": -1, "asset_precision": 0.8, "asset_recall": 0.5, "room_match": 1.0, "overall": 1.2, "notes": "ok"}
```"""
                            }
                        ]
                    }
                }
            }

    critic.client = _FakeBedrock()
    score = critic.evaluate(
        scene_program=SceneProgram(prompt="bedroom", room_type="bedroom"),
        image_paths=[image],
    )
    assert score.validity == 1.0
    assert score.prompt_adherence == 0.0
    assert score.asset_precision == 0.8
    assert score.overall == 1.0
    assert score.critic_mode == "bedrock_vlm_image"


def test_bedrock_vlm_credential_error_maps_to_unavailable(tmp_path) -> None:
    image = tmp_path / "view_001.png"
    image.write_bytes(b"fake-png-bytes")
    critic = BedrockVLMCritic(CriticConfig(mode="vlm", provider="bedrock"))

    class _FakeBedrock:
        def converse(self, **kwargs):
            raise RuntimeError("AWS login required: session has expired")

    critic.client = _FakeBedrock()
    with pytest.raises(CriticUnavailableError, match="Reauthenticate with aws login"):
        critic.evaluate(
            scene_program=SceneProgram(prompt="bedroom", room_type="bedroom"),
            image_paths=[image],
        )


def test_bedrock_vlm_non_retryable_error_maps_to_unavailable_generic(tmp_path) -> None:
    image = tmp_path / "view_001.png"
    image.write_bytes(b"fake-png-bytes")
    critic = BedrockVLMCritic(CriticConfig(mode="vlm", provider="bedrock"))

    class _FakeBedrock:
        def converse(self, **kwargs):
            raise RuntimeError("validation failed")

    critic.client = _FakeBedrock()
    with pytest.raises(CriticUnavailableError, match="Bedrock critic request failed"):
        critic.evaluate(
            scene_program=SceneProgram(prompt="bedroom", room_type="bedroom"),
            image_paths=[image],
        )
