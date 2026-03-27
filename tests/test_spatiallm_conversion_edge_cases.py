from __future__ import annotations

import builtins
import shutil
import sys
import types
from collections import Counter
from pathlib import Path
from uuid import uuid4

import pytest

from compos3d.data import spatiallm as spatiallm_mod
from compos3d.data.spatiallm import (
    SpatialLMCandidate,
    asset_prompt_noun,
    build_training_dataset_payload,
    candidate_sort_key,
    dataset_summary,
    describe_asset,
    extract_sample_id,
    join_phrases,
    load_local_spatiallm_rows,
    load_split_metadata,
    normalize_source_label,
    pluralize,
    room_base_id,
    select_required_assets,
    write_training_dataset_payload,
)


def _candidate(
    room_id: str,
    room_type: str,
    required_assets: list[str],
    asset_counts: dict[str, int],
) -> SpatialLMCandidate:
    return SpatialLMCandidate(
        room_id=room_id,
        sample_id=f"{room_id}_0",
        room_type=room_type,
        source_room_type=room_type.replace("_", " "),
        prompt=f"A {room_type} prompt",
        required_assets=required_assets,
        asset_counts=asset_counts,
    )


def _layout_text(*labels: str, include_window: bool = False) -> str:
    lines = [
        f"bbox_{index}=Bbox({label},1,2,3,0,1,1,1)" for index, label in enumerate(labels)
    ]
    if include_window:
        lines.insert(0, "window_0=Window(wall_1,0.5,1.0,1.0,0.1,1.2)")
    return "\n".join(lines)


def _repo_temp_dir() -> Path:
    path = Path.cwd() / ".test_artifacts" / f"spatiallm_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_spatiallm_helper_branches_cover_selection_and_phrase_shapes() -> None:
    assert normalize_source_label(" Floor Standing Lamp ") == "floor_standing_lamp"

    assert select_required_assets(
        "living_room", Counter({"table_top": 1, "lamp": 1, "rug": 1})
    ) is None
    assert select_required_assets("bedroom", Counter({"lamp": 1})) is None

    assert asset_prompt_noun("dining_room", "dining_table") == "dining table"
    assert asset_prompt_noun("living_room", "table_top") == "side table"
    assert pluralize("glass") == "glass"
    assert pluralize("box") == "boxes"
    assert pluralize("lamp") == "lamps"

    assert describe_asset("living_room", "ottoman", 1) == "an ottoman"
    assert describe_asset("living_room", "lamp", 3) == "three lamps"
    assert describe_asset("living_room", "lamp", 4) == "four lamps"
    assert describe_asset("living_room", "lamp", 5) == "several lamps"

    assert join_phrases(["solo"]) == "solo"
    assert join_phrases(["left", "right"]) == "left and right"


def test_spatiallm_metadata_and_row_loader_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    temp_dir = _repo_temp_dir()
    try:
        csv_path = temp_dir / "split.csv"
        csv_path.write_text(
            "id,room_type,scene_id,room_id,sample,split\n"
            "scene_alpha_0,living room,scene_alpha,7,0,train\n",
            encoding="utf-8",
        )

        metadata = load_split_metadata(csv_path)
        assert metadata["scene_alpha_0"] == {
            "room_type": "living room",
            "scene_id": "scene_alpha",
            "room_id": 7,
            "sample": 0,
            "split": "train",
        }

        fake_datasets = types.ModuleType("datasets")

        def _load_dataset(
            fmt: str, *, data_files: str, split: str
        ) -> list[dict[str, str]]:
            assert fmt == "json"
            assert data_files.endswith("rows.json")
            assert split == "train"
            return [{"row": "one"}, {"row": "two"}]

        fake_datasets.load_dataset = _load_dataset
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        rows = list(load_local_spatiallm_rows(Path("rows.json")))
        assert rows == [{"row": "one"}, {"row": "two"}]

        assert room_base_id("scene_alpha_0") == "scene_alpha"
        assert extract_sample_id({"point_clouds": ["nested/scene_alpha_0.ply"]}) == "scene_alpha_0"
        assert extract_sample_id({"point_clouds": []}) is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_load_local_spatiallm_rows_raises_when_datasets_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def _fake_import(name: str, globals=None, locals=None, fromlist=(), level=0):
        if name == "datasets":
            raise ImportError("missing datasets")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(RuntimeError, match="datasets"):
        list(load_local_spatiallm_rows(Path("rows.json")))


def test_collect_spatiallm_candidates_filters_duplicates_and_sorts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_metadata = {
        "dr_keep_0": {
            "room_type": "dining room",
            "scene_id": "scene_dr",
            "room_id": 1,
            "sample": 0,
            "split": "train",
        },
        "dr_missing_anchor_0": {
            "room_type": "dining room",
            "scene_id": "scene_bad",
            "room_id": 2,
            "sample": 0,
            "split": "train",
        },
        "lr_better_0": {
            "room_type": "living room",
            "scene_id": "scene_lr_a",
            "room_id": 3,
            "sample": 0,
            "split": "train",
        },
        "lr_weaker_0": {
            "room_type": "living room",
            "scene_id": "scene_lr_b",
            "room_id": 4,
            "sample": 0,
            "split": "train",
        },
        "br_keep_0": {
            "room_type": "guest room",
            "scene_id": "scene_br",
            "room_id": 5,
            "sample": 0,
            "split": "train",
        },
        "skip_split_0": {
            "room_type": "living room",
            "scene_id": "scene_skip_split",
            "room_id": 6,
            "sample": 0,
            "split": "test",
        },
        "skip_sample_1": {
            "room_type": "living room",
            "scene_id": "scene_skip_sample",
            "room_id": 7,
            "sample": 1,
            "split": "train",
        },
        "skip_unknown_room_0": {
            "room_type": "kitchen",
            "scene_id": "scene_skip_room",
            "room_id": 8,
            "sample": 0,
            "split": "train",
        },
        "skip_short_conversation_0": {
            "room_type": "living room",
            "scene_id": "scene_skip_conv",
            "room_id": 9,
            "sample": 0,
            "split": "train",
        },
    }

    rows_by_path = {
        "one.json": [
            {"point_clouds": [], "conversations": []},
            {
                "point_clouds": ["/tmp/not_in_metadata.ply"],
                "conversations": [{"value": "u"}, {"value": _layout_text("sofa")}],
            },
            {
                "point_clouds": ["/tmp/skip_split_0.ply"],
                "conversations": [{"value": "u"}, {"value": _layout_text("sofa")}],
            },
            {
                "point_clouds": ["/tmp/skip_sample_1.ply"],
                "conversations": [{"value": "u"}, {"value": _layout_text("sofa")}],
            },
            {
                "point_clouds": ["/tmp/skip_unknown_room_0.ply"],
                "conversations": [{"value": "u"}, {"value": _layout_text("sofa")}],
            },
            {
                "point_clouds": ["/tmp/skip_short_conversation_0.ply"],
                "conversations": [{"value": "u"}],
            },
            {
                "point_clouds": ["/tmp/dr_missing_anchor_0.ply"],
                "conversations": [
                    {"value": "u"},
                    {"value": _layout_text("chair", "lighting", "carpet")},
                ],
            },
            {
                "point_clouds": ["/tmp/dr_keep_0.ply"],
                "conversations": [
                    {"value": "u"},
                    {
                        "value": _layout_text(
                            "dining_table",
                            "dining_chair",
                            "dining_chair",
                            "lighting",
                            "carpet",
                            include_window=True,
                        )
                    },
                ],
            },
            {
                "point_clouds": ["/tmp/lr_weaker_0.ply"],
                "conversations": [
                    {"value": "u"},
                    {"value": _layout_text("sofa", "side_table", "illumination")},
                ],
            },
        ],
        "two.json": [
            {
                "point_clouds": ["/tmp/dr_keep_0.ply"],
                "conversations": [
                    {"value": "u"},
                    {"value": _layout_text("dining_table", "dining_chair", "carpet")},
                ],
            },
            {
                "point_clouds": ["/tmp/lr_better_0.ply"],
                "conversations": [
                    {"value": "u"},
                    {
                        "value": _layout_text(
                            "sofa",
                            "side_table",
                            "illumination",
                            "vase",
                            include_window=True,
                        )
                    },
                ],
            },
            {
                "point_clouds": ["/tmp/br_keep_0.ply"],
                "conversations": [
                    {"value": "u"},
                    {"value": _layout_text("nightstand", "desk_lamp", include_window=True)},
                ],
            },
        ],
    }

    def _fake_rows(path: Path):
        return iter(rows_by_path[path.name])

    monkeypatch.setattr(spatiallm_mod, "load_local_spatiallm_rows", _fake_rows)

    candidates = spatiallm_mod.collect_spatiallm_candidates(
        split_metadata=split_metadata,
        raw_json_paths=[Path("one.json"), Path("two.json")],
    )

    assert [candidate.room_id for candidate in candidates["dining_room"]] == ["dr_keep"]
    assert [candidate.room_id for candidate in candidates["living_room"]] == [
        "lr_better",
        "lr_weaker",
    ]
    assert [candidate.room_id for candidate in candidates["bedroom"]] == ["br_keep"]
    assert "window" in candidates["bedroom"][0].required_assets
    assert candidates["living_room"][0].prompt.startswith("A living room with ")


def test_candidate_sort_payload_write_and_summary() -> None:
    better = _candidate(
        "lr_better",
        "living_room",
        ["sofa", "table_top", "lamp", "window"],
        {"sofa": 1, "table_top": 1, "lamp": 1, "window": 1},
    )
    weaker = _candidate(
        "lr_weaker",
        "living_room",
        ["sofa", "table_top", "lamp"],
        {"sofa": 1, "table_top": 1, "lamp": 1},
    )
    dining = _candidate(
        "dr_keep",
        "dining_room",
        ["dining_table", "chair", "lamp", "rug"],
        {"dining_table": 1, "chair": 4, "lamp": 1, "rug": 1},
    )
    bedroom = _candidate(
        "br_keep",
        "bedroom",
        ["table_top", "lamp"],
        {"table_top": 1, "lamp": 1},
    )

    assert candidate_sort_key(better) < candidate_sort_key(weaker)

    payload = build_training_dataset_payload(
        {
            "living_room": [weaker, better],
            "bedroom": [bedroom],
            "dining_room": [dining],
        },
        max_per_room=1,
    )

    assert payload["dataset_id"] == "spatiallm_balanced_1_per_room"
    assert payload["examples"] == [
        {
            "example_id": "dr_dr_keep",
            "room_type": "dining_room",
            "prompt": dining.prompt,
            "required_assets": dining.required_assets,
            "style": "spatiallm",
        },
        {
            "example_id": "lr_lr_weaker",
            "room_type": "living_room",
            "prompt": weaker.prompt,
            "required_assets": weaker.required_assets,
            "style": "spatiallm",
        },
        {
            "example_id": "br_br_keep",
            "room_type": "bedroom",
            "prompt": bedroom.prompt,
            "required_assets": bedroom.required_assets,
            "style": "spatiallm",
        },
    ]

    temp_dir = _repo_temp_dir()
    try:
        output_path = temp_dir / "nested" / "dataset.json"
        write_training_dataset_payload(payload, output_path)

        assert output_path.read_text(encoding="utf-8").endswith("\n")
        assert dataset_summary(payload) == {
            "dining_room": 1,
            "living_room": 1,
            "bedroom": 1,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
