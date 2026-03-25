"""Edge-case unit tests for `compos3d.catalog`.

Why these tests:
- `infer_room_type` drives downstream generation defaults, so we test both a
  strong keyword match and the no-keyword fallback.
- `assets_mentioned_in_prompt` can leak unsupported assets into a room if its
  filtering changes, so we check room-restricted parsing explicitly.
- `normalize_assets` is the final safety net before generation, so we test
  deduplication and the fallback-to-default-assets branch.

Linked edge cases:
- Ambiguous prompt -> graceful default room type instead of raising.
- Prompt mentions unsupported assets -> unsupported assets are ignored.
- All normalized assets invalid -> room defaults returned instead of empty list.
"""

from __future__ import annotations

from compos3d.catalog import (
    assets_mentioned_in_prompt,
    infer_room_type,
    normalize_assets,
)


def test_infer_room_type_prefers_matching_keywords() -> None:
    room_type = infer_room_type("A warm dining room with a dining table and chairs")
    assert room_type == "dining_room"


def test_infer_room_type_defaults_to_living_room_when_prompt_is_ambiguous() -> None:
    room_type = infer_room_type("A stylish interior with balanced composition")
    assert room_type == "living_room"


def test_assets_mentioned_in_prompt_filters_assets_by_room_type() -> None:
    assets = assets_mentioned_in_prompt(
        "A bedroom with a sofa, lamp, and small table",
        room_type="bedroom",
    )
    assert assets == ["lamp", "table_top"]


def test_assets_mentioned_in_prompt_uses_room_defaults_when_nothing_matches() -> None:
    assets = assets_mentioned_in_prompt(
        "A calm bedroom with soft colors and balanced composition",
        room_type="bedroom",
    )
    assert assets == ["lamp", "rug", "window"]


def test_normalize_assets_deduplicates_and_falls_back_for_unsupported_inputs() -> None:
    assert normalize_assets(["chair", "chair", "lamp"], room_type="dining_room") == [
        "chair",
        "lamp",
    ]
    assert normalize_assets(["sofa", "bed"], room_type="dining_room") == [
        "dining_table",
        "chair",
        "lamp",
    ]
