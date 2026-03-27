from __future__ import annotations

from compos3d.procedural.layout import (
    CHAIR_ORBIT_RADIUS,
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
    pythonpath = env["PYTHONPATH"].split(":")
    assert str(PROJECT_ROOT / "src") in pythonpath
