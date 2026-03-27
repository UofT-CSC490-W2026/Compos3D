from __future__ import annotations

import sys
import types
from collections import Counter
from pathlib import Path

from compos3d.data import spatiallm
from compos3d.data.spatiallm import (
    build_training_dataset_payload,
    build_prompt,
    candidate_sort_key,
    canonicalize_source_room_type,
    collect_spatiallm_candidates,
    dataset_summary,
    describe_asset,
    extract_sample_id,
    join_phrases,
    load_split_metadata,
    normalize_source_label,
    parse_layout_asset_counts,
    pluralize,
    room_base_id,
    select_required_assets,
    write_training_dataset_payload,
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


def test_spatiallm_helper_functions_cover_edge_cases(tmp_path: Path) -> None:
    assert normalize_source_label("Floor-Standing Lamp") == "floor_standing_lamp"
    assert pluralize("glass") == "glass"
    assert pluralize("box") == "boxes"
    assert describe_asset("dining_room", "lamp", 1) == "a lamp"
    assert describe_asset("dining_room", "chair", 2) == "two chairs"
    assert describe_asset("dining_room", "chair", 3) == "three chairs"
    assert describe_asset("dining_room", "chair", 4) == "four chairs"
    assert describe_asset("dining_room", "chair", 5) == "several chairs"
    assert join_phrases(["a"]) == "a"
    assert join_phrases(["a", "b"]) == "a and b"
    assert room_base_id("123_0") == "123"
    assert extract_sample_id({"point_clouds": []}) is None
    assert extract_sample_id({"point_clouds": ["/tmp/abc_0.ply"]}) == "abc_0"

    split_csv = tmp_path / "split.csv"
    split_csv.write_text(
        "id,room_type,scene_id,room_id,sample,split\n"
        "abc_0,living room,scene,1,0,train\n",
        encoding="utf-8",
    )
    metadata = load_split_metadata(split_csv)
    assert metadata["abc_0"]["room_type"] == "living room"


def test_select_required_assets_and_candidate_sort_key_edge_cases() -> None:
    assert (
        select_required_assets(
            "dining_room", Counter({"chair": 4, "lamp": 1, "rug": 1})
        )
        is None
    )
    assert select_required_assets("bedroom", Counter({"lamp": 1})) is None

    class _Candidate:
        required_assets = ["sofa", "rug"]
        asset_counts = {"sofa": 1, "rug": 1, "window": 1}
        room_id = "r1"

    assert candidate_sort_key(_Candidate()) == (-2, -2, 0, "r1")


def test_collect_spatiallm_candidates_and_payload_generation(
    monkeypatch, tmp_path: Path
) -> None:
    split_metadata = {
        "d_0": {
            "room_type": "dining room",
            "scene_id": "s1",
            "room_id": 1,
            "sample": 0,
            "split": "train",
        },
        "skip_0": {
            "room_type": "living room",
            "scene_id": "s2",
            "room_id": 2,
            "sample": 1,
            "split": "train",
        },
        "bad_0": {
            "room_type": "office",
            "scene_id": "s3",
            "room_id": 3,
            "sample": 0,
            "split": "train",
        },
    }
    rows = [
        {
            "point_clouds": ["/tmp/d_0.ply"],
            "conversations": [
                {},
                {
                    "value": "window_0=Window(w,0,0,0,0,0)\nbbox=Bbox(dining_table,1,2,3,0,1,1,1)\nbbox=Bbox(chair,1,2,3,0,1,1,1)\nbbox=Bbox(floor_lamp,1,2,3,0,1,1,1)"
                },
            ],
        },
        {
            "point_clouds": ["/tmp/d_1.ply"],
            "conversations": [{}, {"value": "bbox=Bbox(dining_table,1,2,3,0,1,1,1)"}],
        },
        {
            "point_clouds": ["/tmp/skip_0.ply"],
            "conversations": [{}, {"value": "bbox=Bbox(sofa,1,2,3,0,1,1,1)"}],
        },
        {
            "point_clouds": ["/tmp/bad_0.ply"],
            "conversations": [{}, {"value": "bbox=Bbox(sofa,1,2,3,0,1,1,1)"}],
        },
        {
            "point_clouds": [],
            "conversations": [{}, {"value": "bbox=Bbox(sofa,1,2,3,0,1,1,1)"}],
        },
        {"point_clouds": ["/tmp/short_0.ply"], "conversations": [{}]},
    ]
    monkeypatch.setattr(
        "compos3d.data.spatiallm.load_local_spatiallm_rows",
        lambda _path: iter(rows),
    )

    candidates = collect_spatiallm_candidates(
        split_metadata=split_metadata,
        raw_json_paths=[tmp_path / "spatiallm.json"],
    )

    assert list(candidates) == ["dining_room", "living_room", "bedroom"]
    assert len(candidates["dining_room"]) == 1
    candidate = candidates["dining_room"][0]
    assert candidate.prompt.startswith("A dining room with ")
    payload = build_training_dataset_payload(candidates, max_per_room=1)
    assert payload["dataset_id"] == "spatiallm_balanced_1_per_room"
    assert dataset_summary(payload) == {"dining_room": 1}

    out_path = tmp_path / "dataset.json"
    write_training_dataset_payload(payload, out_path)
    assert out_path.exists()


def test_spatiallm_loader_branches_and_candidate_filters(
    monkeypatch, tmp_path: Path
) -> None:
    fake_datasets = types.SimpleNamespace(
        load_dataset=lambda *_a, **_k: [{"id": 1}, {"id": 2}]
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    assert list(spatiallm.load_local_spatiallm_rows(tmp_path / "x.json")) == [
        {"id": 1},
        {"id": 2},
    ]

    split_metadata = {
        "a_0": {
            "room_type": "dining room",
            "scene_id": "s1",
            "room_id": 1,
            "sample": 0,
            "split": "train",
        },
        "b_0": {
            "room_type": "dining room",
            "scene_id": "s2",
            "room_id": 2,
            "sample": 0,
            "split": "train",
        },
        "c_0": {
            "room_type": "living room",
            "scene_id": "s3",
            "room_id": 3,
            "sample": 0,
            "split": "val",
        },
    }
    rows = [
        {
            "point_clouds": ["/tmp/a_0.ply"],
            "conversations": [
                {},
                {
                    "value": "bbox=Bbox(dining_table,1,2,3,0,1,1,1)\nbbox=Bbox(chair,1,2,3,0,1,1,1)\nbbox=Bbox(lamp,1,2,3,0,1,1,1)"
                },
            ],
        },
        {
            "point_clouds": ["/tmp/b_0.ply"],
            "conversations": [
                {},
                {
                    "value": "window_0=Window(w,0,0,0,0,0)\nbbox=Bbox(dining_table,1,2,3,0,1,1,1)\nbbox=Bbox(chair,1,2,3,0,1,1,1)\nbbox=Bbox(lamp,1,2,3,0,1,1,1)\nbbox=Bbox(rug,1,2,3,0,1,1,1)"
                },
            ],
        },
        {
            "point_clouds": ["/tmp/c_0.ply"],
            "conversations": [{}, {"value": "bbox=Bbox(sofa,1,2,3,0,1,1,1)"}],
        },
    ]
    monkeypatch.setattr(spatiallm, "load_local_spatiallm_rows", lambda _path: iter(rows))

    candidates = collect_spatiallm_candidates(
        split_metadata=split_metadata,
        raw_json_paths=[tmp_path / "spatiallm.json"],
    )

    assert len(candidates["dining_room"]) == 1
    assert candidates["dining_room"][0].sample_id == "b_0"
