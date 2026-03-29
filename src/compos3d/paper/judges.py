from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from compos3d.config import CriticConfig
from compos3d.evaluation.critic import CriticUnavailableError
from compos3d.llm.bedrock import BedrockChatClient, resolve_bedrock_config


def _extract_json_payload(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError(f"Could not locate JSON object in judge response: {cleaned[:200]}")
    return json.loads(cleaned[start : end + 1])


class BedrockPairwiseJudge:
    def __init__(self, config: CriticConfig) -> None:
        self.config = config
        self.client = BedrockChatClient(
            config=resolve_bedrock_config(
                model_id=config.model_id,
                region_name=config.region_name,
            )
        )

    @staticmethod
    def _image_format(path: Path) -> str:
        suffix = path.suffix.lower().lstrip(".")
        if suffix in {"jpg", "jpeg"}:
            return "jpeg"
        if suffix in {"png", "webp"}:
            return suffix
        return "png"

    def _run_json_prompt(self, *, prompt_text: str, image_paths: list[Path]) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
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
                    "Bedrock judge credentials are unavailable or expired. Reauthenticate with aws login."
                ) from exc
            raise CriticUnavailableError(f"Bedrock judge request failed: {exc}") from exc

        text_blocks = [
            item.get("text", "")
            for item in response["output"]["message"]["content"]
            if "text" in item
        ]
        return _extract_json_payload("\n".join(text_blocks))

    def judge_generation_pair(
        self,
        *,
        prompt: str,
        candidate_a_paths: list[Path],
        candidate_b_paths: list[Path],
        method_a: str,
        method_b: str,
    ) -> dict[str, Any]:
        count_a = len(candidate_a_paths)
        count_b = len(candidate_b_paths)
        prompt_text = (
            "You are an academic evaluator for controllable 3D indoor scene generation. "
            f"The first {count_a} images belong to Candidate A. "
            f"The next {count_b} images belong to Candidate B. "
            "Given the user prompt, compare the two scenes. "
            "Return strictly valid JSON with fields asset_selection, layout_coherence, overall_preference, and notes. "
            "Each preference field must be one of 'A', 'B', or 'tie'. Keep notes short.\n"
            f"Prompt: {prompt}\n"
            f"Candidate A method: {method_a}\n"
            f"Candidate B method: {method_b}\n"
        )
        return self._run_json_prompt(
            prompt_text=prompt_text,
            image_paths=[*candidate_a_paths, *candidate_b_paths],
        )

    def judge_edit_pair(
        self,
        *,
        base_prompt: str,
        edited_prompt: str,
        edit_instruction: str,
        before_paths: list[Path],
        after_paths: list[Path],
    ) -> dict[str, Any]:
        prompt_text = (
            "You are an academic evaluator for text-guided 3D scene editing. "
            "The first four images show the base scene. The next four images show the edited scene. "
            "Judge whether the requested edit happened and whether non-target content was preserved. "
            "Return strictly valid JSON with numeric fields edit_success, preservation, and overall in [0, 1], plus notes.\n"
            f"Base prompt: {base_prompt}\n"
            f"Edited prompt: {edited_prompt}\n"
            f"Edit instruction: {edit_instruction}\n"
        )
        return self._run_json_prompt(
            prompt_text=prompt_text,
            image_paths=[*before_paths, *after_paths],
        )


@lru_cache(maxsize=1)
def _load_open_clip_components():
    try:
        import open_clip  # type: ignore[import-not-found]
        import torch
    except ModuleNotFoundError:
        return None

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, preprocess, tokenizer, torch, device


def clip_directional_similarity(
    *,
    before_image_paths: list[Path],
    after_image_paths: list[Path],
    before_text: str,
    after_text: str,
) -> float | None:
    bundle = _load_open_clip_components()
    if bundle is None or not before_image_paths or not after_image_paths:
        return None

    model, preprocess, tokenizer, torch, device = bundle

    def _encode_images(image_paths: list[Path]):
        tensors = []
        for path in image_paths:
            if not path.exists():
                continue
            with Image.open(path) as image:
                tensors.append(preprocess(image.convert("RGB")))
        if not tensors:
            return None
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.mean(dim=0)

    with torch.no_grad():
        tokenized = tokenizer([before_text, after_text]).to(device)
        text_features = model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    before_image = _encode_images(before_image_paths)
    after_image = _encode_images(after_image_paths)
    if before_image is None or after_image is None:
        return None

    image_direction = after_image - before_image
    text_direction = text_features[1] - text_features[0]
    if float(image_direction.norm().item()) == 0.0 or float(text_direction.norm().item()) == 0.0:
        return None
    image_direction = image_direction / image_direction.norm()
    text_direction = text_direction / text_direction.norm()
    score = float((image_direction * text_direction).sum().item())
    return round(score, 4)
