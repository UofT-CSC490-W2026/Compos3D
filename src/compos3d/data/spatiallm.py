from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOM_PRIORITY: tuple[str, ...] = ("dining_room", "living_room", "bedroom")

SOURCE_ROOM_TYPE_TO_ROOM: dict[str, str] = {
    "bedroom": "bedroom",
    "master bedroom": "bedroom",
    "primary bedroom": "bedroom",
    "secondary bedroom": "bedroom",
    "children's bedroom": "bedroom",
    "child room": "bedroom",
    "guest room": "bedroom",
    "living room": "living_room",
    "dining room": "dining_room",
}

SOURCE_LABEL_TO_ASSET: dict[str, str] = {
    "dining_table": "dining_table",
    "dining_table_combination": "dining_table",
    "chair": "chair",
    "dining_chair": "chair",
    "bar_chair": "chair",
    "stool": "chair",
    "sofa": "sofa",
    "combination_sofa": "sofa",
    "coffee_table": "table_top",
    "side_table": "table_top",
    "nightstand": "table_top",
    "desk": "table_top",
    "dressing_table": "table_top",
    "carpet": "rug",
    "rug": "rug",
    "illumination": "lamp",
    "lighting": "lamp",
    "chandelier": "lamp",
    "ceiling_light": "lamp",
    "wall_lamp": "lamp",
    "desk_lamp": "lamp",
    "floor_standing_lamp": "lamp",
    "floor_standinglamp": "lamp",
    "floor_lamp": "lamp",
    "vase": "vase",
}

ASSET_PRIORITY: dict[str, tuple[str, ...]] = {
    "dining_room": ("dining_table", "chair", "lamp", "rug", "window", "vase"),
    "living_room": ("sofa", "table_top", "lamp", "rug", "window", "vase"),
    "bedroom": ("table_top", "lamp", "rug", "window", "chair"),
}

ANCHOR_ASSETS: dict[str, tuple[str, ...]] = {
    "dining_room": ("dining_table", "chair"),
    "living_room": ("sofa",),
    "bedroom": (),
}

MIN_ASSETS: dict[str, int] = {
    "dining_room": 3,
    "living_room": 3,
    "bedroom": 2,
}

MAX_ASSETS_PER_EXAMPLE = 4

BBOX_LABEL_RE = re.compile(r"Bbox\(([^,]+),")
WINDOW_RE = re.compile(r"\bwindow_\d+=Window\(")


@dataclass(frozen=True)
class SpatialLMCandidate:
    room_id: str
    sample_id: str
    room_type: str
    source_room_type: str
    prompt: str
    required_assets: list[str]
    asset_counts: dict[str, int]


def canonicalize_source_room_type(source_room_type: str) -> str | None:
    return SOURCE_ROOM_TYPE_TO_ROOM.get(source_room_type.strip().lower())


def normalize_source_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def parse_layout_asset_counts(layout_text: str) -> Counter[str]:
    asset_counts: Counter[str] = Counter()
    for raw_label in BBOX_LABEL_RE.findall(layout_text):
        mapped = SOURCE_LABEL_TO_ASSET.get(normalize_source_label(raw_label))
        if mapped:
            asset_counts[mapped] += 1

    if WINDOW_RE.search(layout_text):
        asset_counts["window"] = max(asset_counts["window"], 1)
    return asset_counts


def select_required_assets(room_type: str, asset_counts: Counter[str]) -> list[str] | None:
    required: list[str] = []
    available_assets = {
        asset for asset in ASSET_PRIORITY[room_type] if asset_counts.get(asset, 0) > 0
    }

    anchors = ANCHOR_ASSETS[room_type]
    if any(anchor not in available_assets for anchor in anchors):
        return None

    for asset in anchors:
        required.append(asset)

    for asset in ASSET_PRIORITY[room_type]:
        if asset in available_assets and asset not in required:
            required.append(asset)
        if len(required) >= MAX_ASSETS_PER_EXAMPLE:
            break

    if len(required) < MIN_ASSETS[room_type]:
        return None
    return required


def asset_prompt_noun(room_type: str, asset_type: str) -> str:
    if asset_type == "dining_table":
        return "dining table"
    if asset_type == "table_top":
        return "nightstand" if room_type == "bedroom" else "side table"
    return asset_type.replace("_", " ")


def pluralize(noun: str) -> str:
    if noun.endswith("s"):
        return noun
    if noun.endswith("x"):
        return noun + "es"
    return noun + "s"


def describe_asset(room_type: str, asset_type: str, count: int) -> str:
    noun = asset_prompt_noun(room_type, asset_type)
    if count <= 1:
        article = "an" if noun[0].lower() in {"a", "e", "i", "o", "u"} else "a"
        return f"{article} {noun}"
    if count == 2:
        return f"two {pluralize(noun)}"
    if count == 3:
        return f"three {pluralize(noun)}"
    if count == 4:
        return f"four {pluralize(noun)}"
    return f"several {pluralize(noun)}"


def join_phrases(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def build_prompt(
    room_type: str, required_assets: list[str], asset_counts: Counter[str]
) -> str:
    room_phrase = room_type.replace("_", " ")
    described_assets = [
        describe_asset(room_type, asset_type, asset_counts.get(asset_type, 1))
        for asset_type in required_assets
    ]
    return f"A {room_phrase} with {join_phrases(described_assets)}"


def load_split_metadata(path: Path) -> dict[str, dict[str, str | int]]:
    metadata: dict[str, dict[str, str | int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = str(row["id"])
            metadata[sample_id] = {
                "room_type": str(row["room_type"]),
                "scene_id": str(row["scene_id"]),
                "room_id": int(row["room_id"]),
                "sample": int(row["sample"]),
                "split": str(row["split"]),
            }
    return metadata


def load_local_spatiallm_rows(path: Path):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised in real env
        raise RuntimeError(
            "The `datasets` package is required to convert SpatialLM exports."
        ) from exc

    dataset = load_dataset("json", data_files=str(path), split="train")
    for row in dataset:
        yield row


def room_base_id(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[0]


def extract_sample_id(row: dict) -> str | None:
    point_clouds = row.get("point_clouds") or []
    if not point_clouds:
        return None
    point_cloud = str(point_clouds[0])
    return Path(point_cloud).stem


def candidate_sort_key(candidate: SpatialLMCandidate) -> tuple[int, int, int, str]:
    asset_count = len(candidate.required_assets)
    total_mentions = sum(
        candidate.asset_counts.get(asset_type, 0) for asset_type in candidate.required_assets
    )
    window_bonus = 1 if "window" in candidate.required_assets else 0
    return (-asset_count, -total_mentions, -window_bonus, candidate.room_id)


def collect_spatiallm_candidates(
    *,
    split_metadata: dict[str, dict[str, str | int]],
    raw_json_paths: list[Path],
    allowed_split: str = "train",
) -> dict[str, list[SpatialLMCandidate]]:
    candidates_by_room: dict[str, list[SpatialLMCandidate]] = defaultdict(list)
    seen_room_ids: set[str] = set()

    for json_path in raw_json_paths:
        for row in load_local_spatiallm_rows(json_path):
            sample_id = extract_sample_id(row)
            if sample_id is None:
                continue

            metadata = split_metadata.get(sample_id)
            if metadata is None:
                continue
            if metadata["split"] != allowed_split:
                continue
            if int(metadata["sample"]) != 0:
                continue

            source_room_type = str(metadata["room_type"])
            room_type = canonicalize_source_room_type(source_room_type)
            if room_type is None:
                continue

            base_id = room_base_id(sample_id)
            if base_id in seen_room_ids:
                continue

            conversations = row.get("conversations") or []
            if len(conversations) < 2:
                continue
            layout_text = str(conversations[1].get("value", ""))
            asset_counts = parse_layout_asset_counts(layout_text)
            required_assets = select_required_assets(room_type, asset_counts)
            if required_assets is None:
                continue

            prompt = build_prompt(room_type, required_assets, asset_counts)
            candidates_by_room[room_type].append(
                SpatialLMCandidate(
                    room_id=base_id,
                    sample_id=sample_id,
                    room_type=room_type,
                    source_room_type=source_room_type,
                    prompt=prompt,
                    required_assets=required_assets,
                    asset_counts=dict(asset_counts),
                )
            )
            seen_room_ids.add(base_id)

    for room_type in ROOM_PRIORITY:
        candidates_by_room[room_type].sort(key=candidate_sort_key)
    return candidates_by_room


def build_training_dataset_payload(
    candidates_by_room: dict[str, list[SpatialLMCandidate]],
    *,
    max_per_room: int,
) -> dict[str, object]:
    examples: list[dict[str, object]] = []
    room_prefix = {
        "dining_room": "dr",
        "living_room": "lr",
        "bedroom": "br",
    }

    for room_type in ROOM_PRIORITY:
        for candidate in candidates_by_room.get(room_type, [])[:max_per_room]:
            examples.append(
                {
                    "example_id": f"{room_prefix[room_type]}_{candidate.room_id}",
                    "room_type": room_type,
                    "prompt": candidate.prompt,
                    "required_assets": candidate.required_assets,
                    "style": "spatiallm",
                }
            )

    return {
        "dataset_id": f"spatiallm_balanced_{max_per_room}_per_room",
        "examples": examples,
    }


def write_training_dataset_payload(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def dataset_summary(payload: dict[str, object]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for example in payload.get("examples", []):
        room_type = str(example["room_type"])
        counts[room_type] += 1
    return dict(counts)
