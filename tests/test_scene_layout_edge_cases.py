from __future__ import annotations

import pytest

from compos3d.procedural import layout as layout_mod


def test_layout_private_helpers_cover_clone_offset_and_anchor_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_pose = {"xy": (1.0, 2.0), "rot_z": 0.3, "z_offset": 0.4}
    cloned = layout_mod._clone_pose(source_pose)  # noqa: SLF001
    assert cloned == source_pose
    assert cloned is not source_pose

    assert (
        layout_mod._normalize_text("Side_Table", None, " Near Sofa ")
        == "side table near sofa"
    )  # noqa: SLF001
    assert layout_mod._contains_any("place near sofa", ("sofa",))  # noqa: SLF001
    assert not layout_mod._contains_any("place near sofa", ("chair",))  # noqa: SLF001

    assert layout_mod._fallback_pose("living_room", "window") == {  # noqa: SLF001
        "xy": (0.0, 0.0),
        "rot_z": 0.0,
    }
    assert layout_mod._fallback_pose("kitchen", "sink") == {  # noqa: SLF001
        "xy": (0.0, 0.0),
        "rot_z": 0.0,
    }

    assert layout_mod._offset_pose(None, dx=1.0, dy=-2.0, rot_z=0.7, z_offset=0.9) == {  # noqa: SLF001
        "xy": (1.0, -2.0),
        "rot_z": 0.7,
        "z_offset": 0.9,
    }
    assert layout_mod._chair_positions(None, 0) == []  # noqa: SLF001
    assert layout_mod._anchor_pose(  # noqa: SLF001
        "living_room",
        {"asset_type": "sofa", "placement": "floating in the middle of the room"},
    ) == {"xy": (0.0, 0.0), "rot_z": 0.0}

    monkeypatch.setattr(layout_mod, "_fallback_pose", lambda *_args, **_kwargs: None)
    assert layout_mod._anchor_pose("living_room", {"asset_type": "sofa"}) is None  # noqa: SLF001
    assert (
        layout_mod.resolve_asset_positions(  # noqa: SLF001
            {"room_type": "living_room"},
            {"asset_type": "rug", "count": 2},
        )
        == []
    )


def test_preview_anchor_map_and_resolve_positions_cover_living_room_relationships() -> (
    None
):
    scene_program = {
        "prompt": "A cozy living room seating area",
        "room_type": "living_room",
        "assets": [
            {"asset_type": "sofa", "placement": "floating in the middle of the room"},
            {"asset_type": "sofa", "placement": "against the wall"},
            {"asset_type": "table_top", "placement": "in front of the sofa"},
        ],
    }

    anchors = layout_mod.preview_anchor_map(scene_program)
    assert anchors == {
        "sofa": {"xy": (0.0, 0.0), "rot_z": 0.0},
        "table_top": {"xy": (0.0, -0.3), "rot_z": 0.0},
    }

    rug_under_sofa = layout_mod.resolve_asset_positions(
        scene_program,
        {"asset_type": "rug", "count": 2, "placement": "under the sofa"},
        anchors,
    )
    rug_in_front = layout_mod.resolve_asset_positions(
        scene_program,
        {"asset_type": "rug", "count": 1, "placement": "directly in front of the sofa"},
        anchors,
    )
    lamp_near_sofa = layout_mod.resolve_asset_positions(
        scene_program,
        {"asset_type": "lamp", "count": 1, "placement": "beside the sofa"},
        anchors,
    )
    vase_on_table = layout_mod.resolve_asset_positions(
        scene_program,
        {"asset_type": "vase", "count": 1, "placement": "on the side table"},
        anchors,
    )
    window_fallback = layout_mod.resolve_asset_positions(
        scene_program,
        {"asset_type": "window", "count": 2},
        anchors,
    )

    assert rug_under_sofa == [
        {"xy": (0.0, 0.75), "rot_z": 0.0},
        {"xy": (0.0, 0.75), "rot_z": 0.0},
    ]
    assert rug_in_front == [{"xy": (0.0, 1.45), "rot_z": 0.0}]
    assert lamp_near_sofa == [{"xy": (1.6, 0.15), "rot_z": 0.0}]
    assert vase_on_table == [{"xy": (0.0, -0.3), "rot_z": 0.0, "z_offset": 0.62}]
    assert window_fallback == [
        {"xy": (0.0, 0.0), "rot_z": 0.0},
        {"xy": (0.0, 0.0), "rot_z": 0.0},
    ]


def test_resolve_positions_and_factory_params_cover_remaining_branches() -> None:
    dining_scene = {
        "prompt": "A dining room with a rectangular table and rug",
        "room_type": "dining_room",
        "hypotheses": ["Keep the composition cozy"],
        "constraints": [{"text": "Place the rug beneath the dining table"}],
    }
    living_scene = {
        "prompt": "A cozy living room",
        "room_type": "living_room",
        "hypotheses": [],
        "constraints": [{"text": "Keep the rug in front of the sofa"}],
    }
    dining_anchors = {"dining_table": {"xy": (0.0, 0.0), "rot_z": 0.0}}

    lamp_near_table = layout_mod.resolve_asset_positions(
        dining_scene,
        {
            "asset_type": "lamp",
            "count": 1,
            "placement": "above or near the dining table",
        },
        dining_anchors,
    )
    assert lamp_near_table == [{"xy": (1.8, 1.0), "rot_z": 0.0}]

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
    chair_opts = {
        "a": {"style": "simple"},
        "b": {"style": "dining"},
    }
    sofa_opts = {
        "a": {"seat_depth": 0.8},
        "b": {"seat_depth": 1.0},
    }
    lamp_opts = {
        "a": {"height": 1.8},
    }

    dining_rug = layout_mod.resolve_factory_params(
        "rug",
        dining_scene,
        {
            "asset_type": "rug",
            "placement": "beneath the dining_table",
            "rationale": "anchor the table",
        },
        rug_opts,
    )
    sofa_rug = layout_mod.resolve_factory_params(
        "rug",
        living_scene,
        {
            "asset_type": "rug",
            "placement": "in front of sofa",
            "rationale": "anchor the sofa",
        },
        rug_opts,
    )
    dining_chair = layout_mod.resolve_factory_params(
        "chair",
        dining_scene,
        {
            "asset_type": "chair",
            "placement": "around the table",
            "rationale": "seat guests",
        },
        chair_opts,
    )
    cozy_sofa = layout_mod.resolve_factory_params(
        "sofa",
        living_scene,
        {"asset_type": "sofa", "placement": "centered", "rationale": "cozy seating"},
        sofa_opts,
    )
    default_lamp = layout_mod.resolve_factory_params(
        "lamp",
        {
            "prompt": "A spare room",
            "room_type": "bedroom",
            "hypotheses": [],
            "constraints": [],
        },
        {"asset_type": "lamp", "placement": "corner", "rationale": "light the room"},
        lamp_opts,
    )

    assert dining_rug == {
        "width": 2.8,
        "length": 3.6,
        "rug_shape": "rounded",
        "rounded_buffer": 0.35,
    }
    assert sofa_rug == {"width": 2.8, "length": 3.8, "rug_shape": "rectangle"}
    assert dining_chair == {"style": "dining"}
    assert cozy_sofa == {"seat_depth": 1.0}
    assert default_lamp == {"height": 1.8}
