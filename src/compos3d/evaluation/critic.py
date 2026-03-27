from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path

from compos3d.catalog import (
    assets_mentioned_in_prompt,
    infer_room_type,
    supported_assets_for_room,
)
from compos3d.config import CriticConfig
from compos3d.llm.bedrock import BedrockChatClient, resolve_bedrock_config
from compos3d.models import (
    CriticScore,
    EvaluationSummary,
    PredictionRecord,
    SceneProgram,
    TrainingExample,
)


class CriticUnavailableError(RuntimeError):
    pass


class _CriticPayload(CriticScore):
    pass


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return numerator / denominator


def _predicted_asset_set(scene_program: SceneProgram) -> set[str]:
    return {asset.asset_type for asset in scene_program.assets}


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
            f"Could not locate JSON object in critic response: {cleaned[:200]}"
        )
    return json.loads(cleaned[start : end + 1])


def _normalize_notes(raw_notes: object) -> list[str]:
    if isinstance(raw_notes, str):
        return [raw_notes.strip()] if raw_notes.strip() else []
    if not isinstance(raw_notes, list):
        return []
    return [str(item).strip() for item in raw_notes if str(item).strip()]


def _clamp_unit(value: object) -> float:
    numeric = float(value)
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


@lru_cache(maxsize=64)
def _supported_assets_set(room_type: str) -> frozenset[str]:
    return frozenset(supported_assets_for_room(room_type))


@lru_cache(maxsize=1024)
def _mentioned_assets_set(prompt: str, room_type: str) -> frozenset[str]:
    return frozenset(assets_mentioned_in_prompt(prompt, room_type))


def _build_heuristic_score(
    scene_program: SceneProgram, reference_example: TrainingExample | None = None
) -> CriticScore:
    notes: list[str] = []
    # supported_assets = set(supported_assets_for_room(scene_program.room_type))
    supported_assets = _supported_assets_set(
        scene_program.room_type
    )  # performance improvement: cache room asset sets across repeated scoring calls
    predicted_assets = _predicted_asset_set(scene_program)

    unsupported_assets = predicted_assets - supported_assets
    if unsupported_assets:
        notes.append(
            f"Unsupported assets for room type: {', '.join(sorted(unsupported_assets))}"
        )

    validity = 1.0
    if scene_program.room_type not in ("dining_room", "living_room", "bedroom"):
        validity = 0.0
        notes.append("Unsupported room type.")
    elif unsupported_assets:
        validity = 0.4
    elif not predicted_assets:
        validity = 0.2
        notes.append("No assets predicted.")

    if reference_example is not None:
        required_assets = set(reference_example.required_assets)
        overlap = predicted_assets & required_assets
        asset_precision = _safe_ratio(len(overlap), len(predicted_assets))
        asset_recall = _safe_ratio(len(overlap), len(required_assets))
        room_match = (
            1.0 if scene_program.room_type == reference_example.room_type else 0.0
        )
    else:
        # mentioned_assets = set(
        #     assets_mentioned_in_prompt(scene_program.prompt, scene_program.room_type)
        # )
        mentioned_assets = _mentioned_assets_set(
            scene_program.prompt, scene_program.room_type
        )  # performance improvement: cache prompt asset sets across repeated scoring calls
        overlap = predicted_assets & mentioned_assets
        asset_precision = _safe_ratio(len(overlap), len(predicted_assets))
        asset_recall = _safe_ratio(len(overlap), len(mentioned_assets))
        inferred_room_type = infer_room_type(scene_program.prompt)
        room_match = 1.0 if scene_program.room_type == inferred_room_type else 0.0

    prompt_adherence = (asset_precision + asset_recall + room_match) / 3
    overall = (
        (0.25 * validity)
        + (0.35 * prompt_adherence)
        + (0.2 * asset_precision)
        + (0.2 * asset_recall)
    )

    return CriticScore(
        validity=round(validity, 4),
        prompt_adherence=round(prompt_adherence, 4),
        asset_precision=round(asset_precision, 4),
        asset_recall=round(asset_recall, 4),
        room_match=round(room_match, 4),
        overall=round(overall, 4),
        notes=notes,
        critic_mode="heuristic",
    )


class HeuristicSceneCritic:
    def evaluate(
        self,
        *,
        scene_program: SceneProgram,
        reference_example: TrainingExample | None = None,
        image_paths: Sequence[Path] | None = None,
    ) -> CriticScore:
        score = _build_heuristic_score(scene_program, reference_example)
        score.used_image_paths = [str(path) for path in image_paths or []]
        return score


class BedrockVLMCritic:
    def __init__(self, config: CriticConfig) -> None:
        self.config = config
        self.client = BedrockChatClient(
            config=resolve_bedrock_config(
                model_id=self.config.model_id,
                region_name=self.config.region_name,
            )
        )

    def _run_json_prompt(
        self, *, prompt_text: str, image_paths: Sequence[Path]
    ) -> dict:
        content: list[dict] = []
        for path in image_paths:
            content.append(
                {
                    "image": {
                        "format": self._image_format(path),
                        "source": {"bytes": path.read_bytes()},
                    }
                }
            )
        content.append({"text": prompt_text})

        try:
            response = self.client.converse(
                messages=[{"role": "user", "content": content}],
                inference_config={
                    "maxTokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                },
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            if (
                "aws login" in message
                or "refresh token" in message
                or "session has expired" in message
            ):
                raise CriticUnavailableError(
                    "Bedrock critic credentials are unavailable or expired. Reauthenticate with aws login."
                ) from exc
            raise CriticUnavailableError(
                f"Bedrock critic request failed: {exc}"
            ) from exc

        text_blocks = [
            item.get("text", "")
            for item in response["output"]["message"]["content"]
            if "text" in item
        ]
        return _extract_json_payload("\n".join(text_blocks))

    def evaluate(
        self,
        *,
        scene_program: SceneProgram,
        reference_example: TrainingExample | None = None,
        image_paths: Sequence[Path] | None = None,
    ) -> CriticScore:
        normalized_image_paths = [
            Path(path) for path in image_paths or [] if Path(path).exists()
        ]
        if not normalized_image_paths:
            raise CriticUnavailableError(
                "VLM critic requires render images, but none were provided."
            )

        prompt_text = self._build_prompt(
            scene_program=scene_program,
            reference_example=reference_example,
        )
        payload = self._run_json_prompt(
            prompt_text=prompt_text, image_paths=normalized_image_paths
        )
        score = _CriticPayload.model_validate(
            {
                "validity": round(_clamp_unit(payload.get("validity", 0.0)), 4),
                "prompt_adherence": round(
                    _clamp_unit(payload.get("prompt_adherence", 0.0)), 4
                ),
                "asset_precision": round(
                    _clamp_unit(payload.get("asset_precision", 0.0)), 4
                ),
                "asset_recall": round(_clamp_unit(payload.get("asset_recall", 0.0)), 4),
                "room_match": round(_clamp_unit(payload.get("room_match", 0.0)), 4),
                "overall": round(_clamp_unit(payload.get("overall", 0.0)), 4),
                "notes": _normalize_notes(payload.get("notes", [])),
            }
        )

        notes = list(score.notes)
        notes.append("critic_mode=bedrock_vlm_image")

        return CriticScore(
            validity=score.validity,
            prompt_adherence=score.prompt_adherence,
            asset_precision=score.asset_precision,
            asset_recall=score.asset_recall,
            room_match=score.room_match,
            overall=score.overall,
            notes=notes,
            critic_mode="bedrock_vlm_image",
            used_image_paths=[str(path) for path in normalized_image_paths],
        )

    @staticmethod
    def _image_format(path: Path) -> str:
        suffix = path.suffix.lower().lstrip(".")
        if suffix in {"jpg", "jpeg"}:
            return "jpeg"
        if suffix in {"png", "webp"}:
            return suffix
        return "png"

    @staticmethod
    def _build_prompt(
        *, scene_program: SceneProgram, reference_example: TrainingExample | None
    ) -> str:
        reference_block = ""
        if reference_example is not None:
            reference_block = (
                f"Reference room_type: {reference_example.room_type}\n"
                f"Reference required_assets: {reference_example.required_assets}\n"
            )

        return (
            "You are an academic critic for controllable 3D scene generation. "
            "You are given rendered scene images and a structured SceneProgram. Judge the visual scene against the prompt and JSON program. "
            "Return strictly valid JSON with numeric fields validity, prompt_adherence, asset_precision, asset_recall, room_match, overall, and notes. "
            "Each score must be in [0, 1]. Keep notes short.\n"
            + f"Prompt: {scene_program.prompt}\n"
            + f"Scene room_type: {scene_program.room_type}\n"
            + reference_block
            + "SceneProgram JSON:\n"
            + json.dumps(scene_program.model_dump(), indent=2)
        )


def build_scene_critic(config: CriticConfig | None = None):
    resolved = config or CriticConfig()
    mode = resolved.mode.strip().lower()
    provider = resolved.provider.strip().lower()

    if mode == "heuristic":
        return HeuristicSceneCritic()
    if mode != "vlm":
        raise ValueError(f"Unsupported critic mode: {resolved.mode}")
    if provider == "mock":
        raise ValueError(
            "Mock VLM critic has been removed. Use heuristic mode or provider='bedrock'."
        )
    if provider == "bedrock":
        return BedrockVLMCritic(resolved)
    raise ValueError(f"Unsupported critic provider: {resolved.provider}")


def evaluate_scene_program(
    scene_program: SceneProgram,
    reference_example: TrainingExample | None = None,
    *,
    image_paths: Sequence[Path] | None = None,
    critic=None,
) -> CriticScore:
    active_critic = critic or HeuristicSceneCritic()
    return active_critic.evaluate(
        scene_program=scene_program,
        reference_example=reference_example,
        image_paths=image_paths,
    )


def aggregate_prediction_scores(
    predictions: Iterable[PredictionRecord],
) -> EvaluationSummary:
    prediction_list = list(predictions)
    if not prediction_list:
        return EvaluationSummary(
            num_predictions=0,
            average_validity=0.0,
            average_prompt_adherence=0.0,
            average_asset_precision=0.0,
            average_asset_recall=0.0,
            average_room_match=0.0,
            average_overall=0.0,
        )

    total = len(prediction_list)
    return EvaluationSummary(
        num_predictions=total,
        average_validity=round(
            sum(item.critic_score.validity for item in prediction_list) / total, 4
        ),
        average_prompt_adherence=round(
            sum(item.critic_score.prompt_adherence for item in prediction_list) / total,
            4,
        ),
        average_asset_precision=round(
            sum(item.critic_score.asset_precision for item in prediction_list) / total,
            4,
        ),
        average_asset_recall=round(
            sum(item.critic_score.asset_recall for item in prediction_list) / total, 4
        ),
        average_room_match=round(
            sum(item.critic_score.room_match for item in prediction_list) / total, 4
        ),
        average_overall=round(
            sum(item.critic_score.overall for item in prediction_list) / total, 4
        ),
    )
