from __future__ import annotations

from collections.abc import Iterable

SUPPORTED_ASSETS_BY_ROOM: dict[str, tuple[str, ...]] = {
    "dining_room": ("dining_table", "chair", "lamp", "rug", "window", "vase"),
    "living_room": ("sofa", "lamp", "rug", "window", "table_top", "vase"),
    "bedroom": ("lamp", "rug", "window", "chair", "table_top"),
}

ROOM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "dining_room": ("dining room", "dining", "dining table", "chairs"),
    "living_room": ("living room", "living", "sofa", "couch"),
    "bedroom": ("bedroom", "bedside", "nightstand", "sleeping"),
}

ASSET_KEYWORDS: dict[str, tuple[str, ...]] = {
    "dining_table": ("dining table", "table"),
    "chair": ("chair", "chairs"),
    "lamp": ("lamp", "light", "lighting"),
    "rug": ("rug", "carpet"),
    "window": ("window", "windows"),
    "vase": ("vase", "flower vase"),
    "sofa": ("sofa", "couch"),
    "table_top": ("table top", "side table", "small table", "nightstand"),
}

DEFAULT_ASSETS_BY_ROOM: dict[str, tuple[str, ...]] = {
    "dining_room": ("dining_table", "chair", "lamp"),
    "living_room": ("sofa", "lamp", "rug"),
    "bedroom": ("lamp", "rug", "window"),
}


def supported_room_types() -> tuple[str, ...]:
    return tuple(SUPPORTED_ASSETS_BY_ROOM.keys())


def supported_assets_for_room(room_type: str) -> tuple[str, ...]:
    return SUPPORTED_ASSETS_BY_ROOM.get(room_type, ())


def all_supported_assets() -> tuple[str, ...]:
    values: set[str] = set()
    for assets in SUPPORTED_ASSETS_BY_ROOM.values():
        values.update(assets)
    return tuple(sorted(values))


def infer_room_type(prompt: str) -> str:
    text = prompt.lower()
    scores: dict[str, int] = {}
    for room_type, keywords in ROOM_KEYWORDS.items():
        scores[room_type] = sum(1 for keyword in keywords if keyword in text)

    best_room_type = max(scores, key=scores.get)
    if scores[best_room_type] > 0:
        return best_room_type
    return "living_room"


def assets_mentioned_in_prompt(prompt: str, room_type: str | None = None) -> list[str]:
    text = prompt.lower()
    supported = set(all_supported_assets())
    if room_type is not None:
        supported &= set(supported_assets_for_room(room_type))

    mentioned: list[str] = []
    for asset_type, keywords in ASSET_KEYWORDS.items():
        if asset_type not in supported:
            continue
        if any(keyword in text for keyword in keywords):
            mentioned.append(asset_type)

    if not mentioned and room_type in DEFAULT_ASSETS_BY_ROOM:
        return list(DEFAULT_ASSETS_BY_ROOM[room_type])
    return mentioned


def normalize_assets(assets: Iterable[str], room_type: str) -> list[str]:
    supported = set(supported_assets_for_room(room_type))
    normalized = [asset for asset in assets if asset in supported]
    if not normalized:
        normalized = list(DEFAULT_ASSETS_BY_ROOM.get(room_type, ()))
    return list(dict.fromkeys(normalized))
