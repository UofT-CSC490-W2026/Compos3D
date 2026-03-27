from __future__ import annotations

from collections import Counter

from compos3d.data.spatiallm import (
    build_prompt,
    canonicalize_source_room_type,
    parse_layout_asset_counts,
    select_required_assets,
)


def test_canonicalize_source_room_type_maps_supported_variants() -> None:
    assert canonicalize_source_room_type("master bedroom") == "bedroom"
    assert canonicalize_source_room_type("living room") == "living_room"
    assert canonicalize_source_room_type("dining room") == "dining_room"
    assert canonicalize_source_room_type("living room with dining area") is None


def test_parse_layout_asset_counts_maps_labels_and_windows() -> None:
    layout_text = """
window_0=Window(wall_1,0.5,1.0,1.0,0.1,1.2)
bbox_0=Bbox(dining_table_combination,1,2,3,0,1,1,1)
bbox_1=Bbox(dining_chair,1,2,3,0,1,1,1)
bbox_2=Bbox(floor-standing_lamp,1,2,3,0,1,1,1)
bbox_3=Bbox(carpet,1,2,3,0,1,1,1)
bbox_4=Bbox(side_table,1,2,3,0,1,1,1)
"""
    asset_counts = parse_layout_asset_counts(layout_text)

    assert asset_counts == Counter(
        {
            "dining_table": 1,
            "chair": 1,
            "lamp": 1,
            "rug": 1,
            "table_top": 1,
            "window": 1,
        }
    )


def test_select_required_assets_prefers_room_priority() -> None:
    asset_counts = Counter(
        {
            "dining_table": 1,
            "chair": 4,
            "lamp": 2,
            "rug": 1,
            "window": 1,
        }
    )

    assert select_required_assets("dining_room", asset_counts) == [
        "dining_table",
        "chair",
        "lamp",
        "rug",
    ]


def test_build_prompt_uses_room_specific_nouns() -> None:
    prompt = build_prompt(
        "bedroom",
        ["table_top", "lamp", "window"],
        Counter({"table_top": 2, "lamp": 1, "window": 1}),
    )

    assert prompt.startswith("A bedroom with ")
    assert "nightstand" in prompt
    assert "lamp" in prompt
    assert "window" in prompt
