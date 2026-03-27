from __future__ import annotations

import os

from compos3d.procedural.layout import (
    CHAIR_ORBIT_RADIUS,
    _chair_positions,
    preview_anchor_map,
    resolve_asset_positions,
    resolve_factory_params,
)
from compos3d.procedural.runner import PROJECT_ROOT, _build_env


def test_resolve_asset_positions_respects_dining_relationships() -> None:
    scene_program = {
        "prompt": "A cozy dining room with a round table, chairs, and a rug",
        "room_type": "dining_room",
        "assets": [
            {
                "asset_type": "dining_table",
                "count": 1,
                "placement": "centered in the room",
            },
            {
                "asset_type": "chair",
                "count": 4,
                "placement": "evenly distributed around the dining_table perimeter",
            },
            {"asset_type": "rug", "count": 1, "placement": "beneath the dining_table"},
        ],
    }

    anchors = preview_anchor_map(scene_program)
    chair_positions = resolve_asset_positions(
        scene_program, scene_program["assets"][1], anchors
    )
    rug_positions = resolve_asset_positions(
        scene_program, scene_program["assets"][2], anchors
    )

    assert anchors["dining_table"]["xy"] == (0.0, 0.0)
    assert len(chair_positions) == 4
    assert rug_positions == [{"xy": (0.0, 0.0), "rot_z": 0.0}]

    observed = {
        tuple(round(value, 3) for value in pose["xy"]) for pose in chair_positions
    }
    expected = {
        (0.0, round(-CHAIR_ORBIT_RADIUS, 3)),
        (round(CHAIR_ORBIT_RADIUS, 3), 0.0),
        (0.0, round(CHAIR_ORBIT_RADIUS, 3)),
        (round(-CHAIR_ORBIT_RADIUS, 3), 0.0),
    }
    assert observed == expected


def test_resolve_factory_params_adds_round_table_and_rug_hints() -> None:
    scene_program = {
        "prompt": "A cozy dining room with a round table, chairs, and a rug",
        "room_type": "dining_room",
        "hypotheses": [],
        "constraints": [{"text": "The rug must sit beneath the dining table"}],
    }

    table_params = resolve_factory_params(
        "dining_table",
        scene_program,
        {
            "asset_type": "dining_table",
            "placement": "centered in the room",
            "rationale": "A round dining table anchors the room",
        },
        {
            "a": {
                "dimensions": (1.8, 0.9, 0.75),
                "Top Thickness": 0.04,
                "Leg Diameter": 0.07,
            },
            "b": {
                "dimensions": (1.4, 0.7, 0.75),
                "Top Thickness": 0.035,
                "Leg Diameter": 0.18,
            },
        },
    )
    rug_params = resolve_factory_params(
        "rug",
        scene_program,
        {
            "asset_type": "rug",
            "placement": "beneath the dining_table",
            "rationale": "The rug anchors the dining area under the table",
        },
        {
            "a": {"width": 2.0, "length": 3.0, "rug_shape": "rectangle"},
            "b": {"width": 2.5, "length": 2.5, "rug_shape": "circle"},
            "c": {
                "width": 2.0,
                "length": 3.5,
                "rug_shape": "rounded",
                "rounded_buffer": 0.5,
            },
        },
    )

    assert table_params["Top Profile N-gon"] == 32
    assert table_params["dimensions"][:2] == (1.45, 1.45)
    assert table_params["Leg Style"] == "single_stand"
    assert rug_params["rug_shape"] == "circle"
    assert rug_params["width"] == rug_params["length"] == 3.1


def test_runner_env_includes_src_tree_for_procedural_subprocesses() -> None:
    env = _build_env()
    pythonpath = env["PYTHONPATH"].split(os.pathsep)
    assert str(PROJECT_ROOT / "src") in pythonpath


def test_layout_additional_relationship_branches() -> None:
    scene_program = {
        "prompt": "A cozy living room with a floating sofa, a rug in front of the sofa, and a lamp near the sofa",
        "room_type": "living_room",
        "assets": [
            {
                "asset_type": "sofa",
                "count": 1,
                "placement": "floating in the middle of the room",
            },
            {"asset_type": "rug", "count": 1, "placement": "in front of the sofa"},
            {"asset_type": "lamp", "count": 1, "placement": "adjacent to the sofa"},
            {"asset_type": "vase", "count": 1, "placement": "on the side table"},
        ],
    }

    anchors = preview_anchor_map(scene_program)
    rug_positions = resolve_asset_positions(
        scene_program, scene_program["assets"][1], anchors
    )
    lamp_positions = resolve_asset_positions(
        scene_program, scene_program["assets"][2], anchors
    )
    vase_positions = resolve_asset_positions(
        scene_program, scene_program["assets"][3], anchors
    )
    unknown_positions = resolve_asset_positions(
        {"room_type": "dining_room", "assets": []},
        {"asset_type": "window", "count": 1, "placement": "near wall"},
        {},
    )

    assert anchors["sofa"]["xy"] == (0.0, 0.0)
    assert rug_positions == [{"xy": (0.0, 1.45), "rot_z": 0.0}]
    assert lamp_positions == [{"xy": (1.6, 0.15), "rot_z": 0.0}]
    assert vase_positions == [{"xy": (0.15, -0.3), "rot_z": 0.0, "z_offset": 0.62}]
    assert unknown_positions == [{"xy": (0.0, 0.0), "rot_z": 0.0}]


def test_resolve_factory_params_additional_branches() -> None:
    dining_scene = {
        "prompt": "A dining room with a rectangular dining table and a rug under the dining table",
        "room_type": "dining_room",
        "hypotheses": [],
        "constraints": [],
    }
    living_scene = {
        "prompt": "A cozy living room with a sofa and rug under sofa",
        "room_type": "living_room",
        "hypotheses": ["keep a cozy sofa"],
        "constraints": [{"text": "Rug goes under sofa"}],
    }

    rug_opts = {
        "a": {"width": 2.0, "length": 3.0, "rug_shape": "rectangle"},
        "b": {"width": 2.5, "length": 2.5, "rug_shape": "circle"},
        "c": {
            "width": 2.0,
            "length": 3.5,
            "rug_shape": "rounded",
            "rounded_buffer": 0.5,
        },
    }
    chair_opts = {"a": {"legs": 3}, "b": {"legs": 4}}
    sofa_opts = {"a": {"variant": "default"}, "b": {"variant": "cozy"}}

    dining_rug = resolve_factory_params(
        "rug",
        dining_scene,
        {
            "asset_type": "rug",
            "placement": "under the dining table",
            "rationale": "anchor",
        },
        rug_opts,
    )
    living_rug = resolve_factory_params(
        "rug",
        living_scene,
        {"asset_type": "rug", "placement": "under sofa", "rationale": "anchor"},
        rug_opts,
    )
    dining_chair = resolve_factory_params(
        "chair",
        dining_scene,
        {"asset_type": "chair", "placement": "around table", "rationale": "seat"},
        chair_opts,
    )
    cozy_sofa = resolve_factory_params(
        "sofa",
        living_scene,
        {"asset_type": "sofa", "placement": "center", "rationale": "anchor"},
        sofa_opts,
    )

    assert dining_rug["rug_shape"] == "rounded"
    assert dining_rug["rounded_buffer"] == 0.35
    assert living_rug["rug_shape"] == "rectangle"
    assert dining_chair == {"legs": 4}
    assert cozy_sofa == {"variant": "cozy"}


def test_layout_edge_fallback_helpers() -> None:
    scene_program = {
        "prompt": "A dining room with a lamp near the dining table",
        "room_type": "dining_room",
        "assets": [
            {
                "asset_type": "dining_table",
                "count": 1,
                "placement": "centered in the room",
            },
            {"asset_type": "dining_table", "count": 1, "placement": "duplicate anchor"},
            {
                "asset_type": "lamp",
                "count": 1,
                "placement": "near the dining table",
                "rationale": "light",
            },
        ],
    }

    anchors = preview_anchor_map(scene_program)
    lamp_positions = resolve_asset_positions(
        scene_program, scene_program["assets"][2], anchors
    )
    no_chairs = _chair_positions({"xy": (0.0, 0.0), "rot_z": 0.0}, 0)
    no_room_positions = resolve_asset_positions(
        {"room_type": "garage", "prompt": "", "assets": []},
        {"asset_type": "lamp", "count": 1, "placement": "corner"},
        {},
    )
    passthrough_params = resolve_factory_params(
        "lamp",
        {"prompt": "", "room_type": "dining_room", "hypotheses": [], "constraints": []},
        {"asset_type": "lamp", "placement": "corner", "rationale": "light"},
        {"a": {"kind": "default"}},
    )

    assert list(anchors) == ["dining_table", "lamp"]
    assert lamp_positions == [{"xy": (1.8, 1.0), "rot_z": 0.0}]
    assert no_chairs == []
    assert no_room_positions == [{"xy": (0.0, 0.0), "rot_z": 0.0}]
    assert passthrough_params == {"kind": "default"}
