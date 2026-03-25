"""Cross-module tests for graceful error handling on common failure modes.

Representative error paths that should fail cleanly with actionable
exceptions/messages rather than cryptic crashes.

Edge cases covered:
- malformed model/critic JSON payload extraction
- unavailable/expired Bedrock credentials mapping
- missing local files for storage reads
- missing prediction artifacts in evaluation entrypoint
- unsupported provider selection

Expected outcomes:
- Exceptions are explicit (`ValueError`, `FileNotFoundError`, custom runtime errors).
- Error messages contain enough context for quick diagnosis.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from compos3d.config import GeneratorConfig
from compos3d.evaluation import critic as critic_mod

# Repo-local import shim: some branches import compos3d.data.dataset even when
# this package is absent in checkout state.
_stub_data_pkg = types.ModuleType("compos3d.data")
_stub_data_dataset = types.ModuleType("compos3d.data.dataset")
_stub_data_dataset.load_training_dataset = lambda *_args, **_kwargs: None
sys.modules.setdefault("compos3d.data", _stub_data_pkg)
sys.modules.setdefault("compos3d.data.dataset", _stub_data_dataset)

from compos3d.hypothesis import engine
from compos3d.llm.scene_llm import BedrockSceneLLM, LLMUnavailableError, StructuredOutputError, build_scene_llm
from compos3d.storage.local import LocalStore


def test_scene_llm_extract_json_payload_malformed_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Could not locate JSON object"):
        # importing private helper via module to keep behavior-bound coverage
        from compos3d.llm import scene_llm as scene_llm_mod

        scene_llm_mod._extract_json_payload("no-json-here")  # noqa: SLF001


def test_critic_extract_json_payload_malformed_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Could not locate JSON object"):
        critic_mod._extract_json_payload("totally invalid payload")  # noqa: SLF001


def test_bedrock_scene_llm_credential_error_maps_to_unavailable() -> None:
    llm = BedrockSceneLLM(GeneratorConfig(provider="bedrock"))

    class _FakeClient:
        def converse(self, **kwargs):
            raise RuntimeError("session has expired")

    llm.client = _FakeClient()
    with pytest.raises(LLMUnavailableError, match="Reauthenticate with aws login"):
        llm._run_json_prompt("ping")  # noqa: SLF001


def test_bedrock_scene_llm_invalid_json_maps_to_structured_output_error() -> None:
    llm = BedrockSceneLLM(GeneratorConfig(provider="bedrock"))

    class _FakeClient:
        def converse(self, **kwargs):
            return {"output": {"message": {"content": [{"text": "not-json"}]}}}

    llm.client = _FakeClient()
    with pytest.raises(StructuredOutputError, match="invalid JSON output"):
        llm._run_json_prompt("ping")  # noqa: SLF001


def test_local_store_read_missing_file_raises_file_not_found(tmp_path) -> None:
    store = LocalStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.read_json("bronze/missing.json")


def test_evaluate_prediction_dir_missing_artifacts_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Could not find predictions.jsonl"):
        engine.evaluate_prediction_dir(
            predictions_dir=tmp_path / "does_not_exist",
            output_dir=tmp_path / "eval_out",
        )


def test_build_scene_llm_unsupported_provider_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported llm provider"):
        build_scene_llm("totally-unsupported-provider")


def test_evaluate_prediction_dir_fallback_manifest_without_critic_file_raises(tmp_path) -> None:
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    (pred_dir / "inference_manifest.json").write_text(json.dumps({"ok": True}))
    # fallback path requires critic_score.json; verify failure is explicit.
    with pytest.raises(FileNotFoundError):
        engine.evaluate_prediction_dir(predictions_dir=pred_dir, output_dir=tmp_path / "out")

