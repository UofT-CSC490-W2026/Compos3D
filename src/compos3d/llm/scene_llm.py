from __future__ import annotations

import json
import re
from collections import Counter

from compos3d.catalog import (
    DEFAULT_ASSETS_BY_ROOM,
    assets_mentioned_in_prompt,
    infer_room_type,
    normalize_assets,
    supported_assets_for_room,
)
from compos3d.config import GeneratorConfig
from compos3d.llm.bedrock import BedrockChatClient, resolve_bedrock_config
from compos3d.models import AssetSpec, ConstraintSpec, SceneProgram, TrainingExample


class LLMUnavailableError(RuntimeError):
    pass


class StructuredOutputError(RuntimeError):
    pass


def _extract_json_payload(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError(
            f"Could not locate JSON object in model response: {cleaned[:200]}"
        )
    return json.loads(cleaned[start : end + 1])


def _extract_style(prompt: str) -> str | None:
    styles = ("cozy", "bright", "minimal", "modern", "warm", "calm")
    lower = prompt.lower()
    for style in styles:
        if style in lower:
            return style
    return None


def _extract_requested_count(prompt: str, asset_type: str) -> int:
    text = prompt.lower()
    if asset_type == "chair":
        number_match = re.search(r"(\d+)\s+chairs?", text)
        if number_match:
            return max(1, int(number_match.group(1)))
        if "four chairs" in text:
            return 4
    return 1


def _normalize_hypotheses(raw_hypotheses: object) -> list[str]:
    if not isinstance(raw_hypotheses, list):
        raise StructuredOutputError(
            "SceneProgram response must include a hypotheses list."
        )

    normalized = [str(item).strip() for item in raw_hypotheses if str(item).strip()]
    normalized = _dedupe_keep_order(normalized)
    if not normalized:
        raise StructuredOutputError(
            "SceneProgram response must include at least one non-empty hypothesis."
        )
    return normalized


def _normalize_constraints(raw_constraints: object) -> list[dict[str, str]]:
    if not isinstance(raw_constraints, list):
        raise StructuredOutputError(
            "SceneProgram response must include a constraints list."
        )

    normalized: list[dict[str, str]] = []
    for item in raw_constraints:
        if isinstance(item, str) and item.strip():
            normalized.append({"text": item.strip()})
            continue
        if not isinstance(item, dict):
            raise StructuredOutputError(
                "Each SceneProgram constraint must be a string or an object with a text field."
            )
        text = str(item.get("text", "")).strip()
        if not text:
            raise StructuredOutputError(
                "Each SceneProgram constraint must contain a non-empty text field."
            )
        normalized.append({"text": text})

    if not normalized:
        raise StructuredOutputError(
            "SceneProgram response must include at least one valid constraint."
        )
    return normalized


def _normalize_assets(raw_assets: object, room_type: str) -> list[dict[str, object]]:
    if not isinstance(raw_assets, list):
        raise StructuredOutputError(
            "SceneProgram response must include an assets list."
        )

    supported = set(supported_assets_for_room(room_type))
    normalized_assets: list[dict[str, object]] = []
    for item in raw_assets:
        if not isinstance(item, dict):
            raise StructuredOutputError("Each SceneProgram asset must be an object.")

        asset_type = str(item.get("asset_type", "")).strip()
        if asset_type not in supported:
            raise StructuredOutputError(
                f"SceneProgram asset_type '{asset_type}' is not supported for room_type '{room_type}'."
            )

        try:
            count = int(item.get("count", 1) or 1)
        except (TypeError, ValueError) as exc:
            raise StructuredOutputError(
                f"Invalid asset count for '{asset_type}'."
            ) from exc
        if count <= 0:
            raise StructuredOutputError(
                f"Asset '{asset_type}' must have a positive count."
            )

        placement = str(item.get("placement", "")).strip()
        rationale = str(item.get("rationale", "")).strip()
        if not placement:
            raise StructuredOutputError(
                f"Asset '{asset_type}' is missing a placement string."
            )
        if not rationale:
            raise StructuredOutputError(
                f"Asset '{asset_type}' is missing a rationale string."
            )

        normalized_assets.append(
            {
                "asset_type": asset_type,
                "count": count,
                "placement": placement,
                "rationale": rationale,
            }
        )

    if not normalized_assets:
        raise StructuredOutputError(
            "SceneProgram response must include at least one valid asset."
        )
    return normalized_assets


def _normalize_scene_program_payload(
    payload: dict, *, prompt: str, room_type: str
) -> dict:
    if not isinstance(payload, dict):
        raise StructuredOutputError("SceneProgram response must be a JSON object.")

    normalized_payload = dict(payload)
    returned_room_type = str(normalized_payload.get("room_type", "")).strip()
    if returned_room_type and returned_room_type != room_type:
        raise StructuredOutputError(
            f"SceneProgram returned room_type '{returned_room_type}', expected '{room_type}'."
        )

    render_spec = normalized_payload.get("render_spec")
    if render_spec is not None and not isinstance(render_spec, dict):
        raise StructuredOutputError(
            "SceneProgram render_spec must be an object if provided."
        )

    return {
        "prompt": prompt,
        "room_type": room_type,
        "style": str(normalized_payload.get("style", "")).strip() or None,
        "hypotheses": _normalize_hypotheses(normalized_payload.get("hypotheses")),
        "assets": _normalize_assets(normalized_payload.get("assets"), room_type),
        "constraints": _normalize_constraints(normalized_payload.get("constraints")),
        "render_spec": render_spec or {"mode": "program_only"},
    }


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _mock_hypotheses(
    room_type: str, examples: list[TrainingExample], *, num_hypotheses: int, focus: str
) -> list[str]:
    asset_counter: Counter[str] = Counter()
    for example in examples:
        asset_counter.update(example.required_assets)

    common_assets = [asset for asset, _ in asset_counter.most_common(3)]
    anchor_assets = (
        ", ".join(common_assets)
        if common_assets
        else ", ".join(DEFAULT_ASSETS_BY_ROOM.get(room_type, ()))
    )
    if focus == "repair":
        hypotheses = [
            f"In a {room_type}, include the prompt-critical assets such as {anchor_assets} before adding decorative objects.",
            f"For a {room_type}, avoid unsupported assets and keep object choices inside the room-specific asset catalog.",
            f"When a {room_type} layout fails, preserve the room type and simplify placements around the main anchor objects.",
        ]
    else:
        hypotheses = [
            f"In a {room_type}, anchor the composition around {anchor_assets}.",
            f"For a {room_type}, only use assets from the supported room catalog and avoid unrelated objects.",
            f"When the prompt specifies style cues for a {room_type}, preserve them while keeping placements simple and realistic.",
        ]
    return hypotheses[:num_hypotheses]


class MockSceneLLM:
    provider_name = "mock"

    def generate_hypotheses(
        self,
        room_type: str,
        examples: list[TrainingExample],
        *,
        num_hypotheses: int = 3,
        focus: str = "general",
    ) -> list[str]:
        return _mock_hypotheses(
            room_type, examples, num_hypotheses=num_hypotheses, focus=focus
        )

    def generate_scene_program(
        self, *, prompt: str, room_type: str | None, selected_hypotheses: list[str]
    ) -> SceneProgram:
        resolved_room_type = room_type or infer_room_type(prompt)
        prompt_assets = assets_mentioned_in_prompt(prompt, resolved_room_type)
        hypothesis_assets: list[str] = []
        for hypothesis in selected_hypotheses:
            lower_hypothesis = hypothesis.lower()
            for asset in supported_assets_for_room(resolved_room_type):
                if (
                    asset.replace("_", " ") in lower_hypothesis
                    or asset in lower_hypothesis
                ):
                    hypothesis_assets.append(asset)

        asset_types = normalize_assets(
            prompt_assets + hypothesis_assets, resolved_room_type
        )
        assets = [
            AssetSpec(
                asset_type=asset_type,
                count=_extract_requested_count(prompt, asset_type),
                placement=(
                    "center of room"
                    if asset_type in {"dining_table", "sofa"}
                    else "near wall or support surface"
                ),
                rationale=f"Selected from prompt and hypotheses for {resolved_room_type}.",
            )
            for asset_type in asset_types
        ]
        constraints = [
            ConstraintSpec(text=hypothesis) for hypothesis in selected_hypotheses
        ]
        return SceneProgram(
            prompt=prompt,
            room_type=resolved_room_type,
            style=_extract_style(prompt),
            hypotheses=selected_hypotheses,
            assets=assets,
            constraints=constraints,
        )


class BedrockSceneLLM:
    provider_name = "bedrock"

    def __init__(self, model_config: GeneratorConfig | None = None) -> None:
        self.model_config = model_config or GeneratorConfig(provider="bedrock")
        self.client = BedrockChatClient(
            config=resolve_bedrock_config(
                model_id=self.model_config.model_id,
                region_name=self.model_config.region_name,
            )
        )

    def _run_json_prompt(self, user_text: str) -> dict:
        try:
            response = self.client.converse(
                messages=[{"role": "user", "content": [{"text": user_text}]}],
                inference_config={
                    "maxTokens": self.model_config.max_tokens,
                    "temperature": self.model_config.temperature,
                },
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            if (
                "aws login" in message
                or "refresh token" in message
                or "session has expired" in message
            ):
                raise LLMUnavailableError(
                    "Bedrock credentials are unavailable or expired. Reauthenticate with aws login."
                ) from exc
            raise LLMUnavailableError(f"Bedrock request failed: {exc}") from exc

        try:
            content = response["output"]["message"]["content"][0]["text"]
            return _extract_json_payload(content)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise StructuredOutputError(
                "Bedrock returned invalid JSON output."
            ) from exc

    def generate_hypotheses(
        self,
        room_type: str,
        examples: list[TrainingExample],
        *,
        num_hypotheses: int = 3,
        focus: str = "general",
    ) -> list[str]:
        example_lines = [
            f"- {example.prompt} | required_assets={example.required_assets}"
            for example in examples
        ]
        if focus == "repair":
            prompt = (
                "You are generating repair hypotheses for controllable procedural 3D indoor scenes. "
                'Return strictly valid JSON with the shape {"hypotheses": [string, ...]}. '
                f"Generate {num_hypotheses} short repair hypotheses for room_type={room_type}. "
                f"Supported assets: {list(supported_assets_for_room(room_type))}. "
                "These examples exposed failures. Write reusable patch rules that would prevent the same mistakes in future scenes. "
                "Focus on missing required assets, wrong room semantics, unsupported objects, or unrealistic placement patterns. "
                "Each hypothesis must be abstract and reusable, not a scene description. "
                "Prefer language like should, avoid, typically, rarely, group, align, support, and preserve. "
                "Examples:\n" + "\n".join(example_lines)
            )
        else:
            prompt = (
                "You are generating abstract design hypotheses for controllable procedural 3D indoor scenes. "
                'Return strictly valid JSON with the shape {"hypotheses": [string, ...]}. '
                f"Generate {num_hypotheses} short hypotheses for room_type={room_type}. "
                f"Supported assets: {list(supported_assets_for_room(room_type))}. "
                "Each hypothesis must be a reusable rule, not a scene description. "
                "Do not copy or paraphrase the example scenes. Do not include required_assets strings. "
                "Good hypothesis style: 'In a living_room, a sofa should usually anchor the seating area and face the main opening or focal point.' "
                "Good hypothesis style: 'In a dining_room, chairs should be grouped around the dining_table instead of scattered across the room.' "
                "Bad style: 'A bright living room with a sofa near a window and a rug underneath.' "
                "Prefer words like should, avoid, typically, rarely, place, and support. "
                "Examples:\n" + "\n".join(example_lines)
            )
        payload = self._run_json_prompt(prompt)
        raw_hypotheses = payload.get("hypotheses")
        if not isinstance(raw_hypotheses, list):
            raise StructuredOutputError(
                "Bedrock hypothesis response must include a hypotheses list."
            )
        hypotheses = [str(item).strip() for item in raw_hypotheses if str(item).strip()]
        hypotheses = _dedupe_keep_order(hypotheses)
        if not hypotheses:
            raise StructuredOutputError("Bedrock returned no valid hypotheses.")
        return hypotheses[:num_hypotheses]

    def generate_scene_program(
        self, *, prompt: str, room_type: str | None, selected_hypotheses: list[str]
    ) -> SceneProgram:
        resolved_room_type = room_type or infer_room_type(prompt)
        request = (
            "You are generating a structured SceneProgram for a controllable procedural indoor scene. "
            "Return strictly valid JSON with fields prompt, room_type, style, hypotheses, assets, constraints, and render_spec. "
            f"room_type must be one of {list(DEFAULT_ASSETS_BY_ROOM.keys())}. "
            f"Supported assets for {resolved_room_type}: {list(supported_assets_for_room(resolved_room_type))}. "
            "Each asset item must have asset_type, count, placement, and rationale. "
            "Use only supported assets, keep the program minimal, and preserve the selected hypotheses. "
            f"Prompt: {prompt}\n"
            f"Selected hypotheses: {selected_hypotheses}"
        )
        raw_payload = self._run_json_prompt(request)
        normalized_payload = _normalize_scene_program_payload(
            raw_payload,
            prompt=prompt,
            room_type=resolved_room_type,
        )
        return SceneProgram.model_validate(normalized_payload)


def build_scene_llm(provider_or_config: str | GeneratorConfig):
    if isinstance(provider_or_config, GeneratorConfig):
        provider = provider_or_config.provider.strip().lower()
        if provider == "mock":
            return MockSceneLLM()
        if provider == "bedrock":
            return BedrockSceneLLM(provider_or_config)
        raise ValueError(f"Unsupported llm provider: {provider_or_config.provider}")

    normalized = provider_or_config.strip().lower()
    if normalized == "mock":
        return MockSceneLLM()
    if normalized == "bedrock":
        return BedrockSceneLLM(GeneratorConfig(provider="bedrock"))
    raise ValueError(f"Unsupported llm provider: {provider_or_config}")
