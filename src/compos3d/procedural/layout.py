from __future__ import annotations

from copy import deepcopy
from typing import Any

Pose = dict[str, Any]

DEFAULT_LAYOUTS: dict[str, dict[str, Pose | None]] = {
    "dining_room": {
        "dining_table": {"xy": (0.0, 0.0), "rot_z": 0.0},
        "rug": {"xy": (0.0, 0.0), "rot_z": 0.0},
        "lamp": {"xy": (2.1, 2.1), "rot_z": 0.0},
        "window": None,
        "table_top": {"xy": (-2.1, 1.8), "rot_z": 0.785},
        "vase": {"xy": (-2.1, 1.8), "rot_z": 0.0, "z_offset": 0.62},
    },
    "living_room": {
        "sofa": {"xy": (0.0, -1.9), "rot_z": 0.0},
        "rug": {"xy": (0.0, -0.3), "rot_z": 0.0},
        "lamp": {"xy": (2.0, -1.9), "rot_z": 0.0},
        "table_top": {"xy": (0.0, -0.3), "rot_z": 0.0},
        "window": None,
        "vase": {"xy": (0.15, -0.3), "rot_z": 0.0, "z_offset": 0.62},
    },
    "bedroom": {
        "lamp": {"xy": (1.5, 1.5), "rot_z": 0.0},
        "rug": {"xy": (0.0, 0.0), "rot_z": 0.0},
        "chair": {"xy": (-1.5, 1.0), "rot_z": -0.785},
        "table_top": {"xy": (1.5, 0.5), "rot_z": 0.0},
        "window": None,
        "vase": {"xy": (1.5, 0.5), "rot_z": 0.0, "z_offset": 0.62},
    },
}

CHAIR_ORBIT_RADIUS = 1.12


def _clone_pose(pose: Pose | None) -> Pose | None:
    return deepcopy(pose) if pose is not None else None


def _normalize_text(*parts: object) -> str:
    chunks = [str(part or "").replace("_", " ").lower() for part in parts]
    return " ".join(chunk.strip() for chunk in chunks if chunk.strip())


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _fallback_pose(room_type: str, asset_type: str) -> Pose | None:
    room_defaults = DEFAULT_LAYOUTS.get(room_type, {})
    pose = room_defaults.get(asset_type)
    if pose is not None:
        return _clone_pose(pose)
    return {"xy": (0.0, 0.0), "rot_z": 0.0}


def _offset_pose(
    pose: Pose | None,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    rot_z: float | None = None,
    z_offset: float | None = None,
) -> Pose:
    base = _clone_pose(pose) or {"xy": (0.0, 0.0), "rot_z": 0.0}
    x, y = base.get("xy", (0.0, 0.0))
    out = {
        "xy": (float(x) + dx, float(y) + dy),
        "rot_z": float(base.get("rot_z", 0.0) if rot_z is None else rot_z),
    }
    final_z = base.get("z_offset", 0.0) if z_offset is None else z_offset
    if final_z:
        out["z_offset"] = float(final_z)
    return out


def _chair_positions(center_pose: Pose | None, count: int) -> list[Pose]:
    if count <= 0:
        return []
    center = center_pose or {"xy": (0.0, 0.0), "rot_z": 0.0}
    center_x, center_y = center.get("xy", (0.0, 0.0))
    import math

    positions: list[Pose] = []
    for i in range(count):
        angle = 2 * math.pi * i / count
        x = float(center_x) + CHAIR_ORBIT_RADIUS * math.sin(angle)
        y = float(center_y) - CHAIR_ORBIT_RADIUS * math.cos(angle)
        positions.append({"xy": (x, y), "rot_z": angle})
    return positions


def _anchor_pose(room_type: str, spec: dict[str, Any]) -> Pose | None:
    asset_type = str(spec.get("asset_type", ""))
    placement = _normalize_text(spec.get("placement", ""))
    fallback = _fallback_pose(room_type, asset_type)
    if fallback is None:
        return None

    if asset_type == "dining_table":
        if _contains_any(
            placement,
            ("center", "centred", "centered", "middle of the room", "room center"),
        ):
            return {"xy": (0.0, 0.0), "rot_z": 0.0}
    if asset_type == "sofa":
        if _contains_any(placement, ("center", "floating", "middle of the room")):
            return {"xy": (0.0, 0.0), "rot_z": 0.0}
    return fallback


def preview_anchor_map(scene_program: dict[str, Any]) -> dict[str, Pose]:
    room_type = str(scene_program.get("room_type", "dining_room"))
    anchors: dict[str, Pose] = {}
    for spec in scene_program.get("assets", []):
        asset_type = str(spec.get("asset_type", ""))
        if asset_type in anchors:
            continue
        pose = _anchor_pose(room_type, spec)
        if pose is not None:
            anchors[asset_type] = pose
    return anchors


def resolve_asset_positions(
    scene_program: dict[str, Any],
    spec: dict[str, Any],
    anchors: dict[str, Pose] | None = None,
) -> list[Pose]:
    room_type = str(scene_program.get("room_type", "dining_room"))
    asset_type = str(spec.get("asset_type", ""))
    count = max(1, int(spec.get("count", 1)))
    placement_text = _normalize_text(
        scene_program.get("prompt", ""),
        spec.get("placement", ""),
        spec.get("rationale", ""),
    )
    anchors = dict(anchors or {})
    fallback = _fallback_pose(room_type, asset_type)

    if fallback is None:
        return []

    if asset_type == "chair" and _contains_any(
        placement_text,
        (
            "around table",
            "around the dining table",
            "around the dining_table",
            "around dining table",
            "perimeter",
            "distributed around",
            "grouped around",
        ),
    ):
        return _chair_positions(anchors.get("dining_table"), count)

    if asset_type == "rug":
        table_anchor = anchors.get("dining_table")
        sofa_anchor = anchors.get("sofa")
        if table_anchor and _contains_any(
            placement_text,
            (
                "beneath the dining table",
                "beneath dining table",
                "under the dining table",
                "under dining table",
                "beneath the table",
                "under the table",
            ),
        ):
            return [_offset_pose(table_anchor) for _ in range(count)]
        if sofa_anchor and _contains_any(
            placement_text,
            ("under the sofa", "under sofa", "beneath the sofa", "beneath sofa"),
        ):
            return [_offset_pose(sofa_anchor, dy=0.75) for _ in range(count)]
        if sofa_anchor and _contains_any(
            placement_text,
            ("in front of the sofa", "in front of sofa", "directly in front of the sofa"),
        ):
            return [_offset_pose(sofa_anchor, dy=1.45) for _ in range(count)]

    if asset_type == "lamp":
        sofa_anchor = anchors.get("sofa")
        table_anchor = anchors.get("dining_table")
        if sofa_anchor and _contains_any(
            placement_text,
            (
                "near sofa",
                "near the sofa",
                "adjacent to sofa",
                "adjacent to the sofa",
                "beside sofa",
                "beside the sofa",
                "seating area",
            ),
        ):
            return [_offset_pose(sofa_anchor, dx=1.6, dy=0.15) for _ in range(count)]
        if table_anchor and _contains_any(
            placement_text,
            (
                "near dining table",
                "near the dining table",
                "above or near the dining table",
                "above or near dining table",
            ),
        ):
            return [_offset_pose(table_anchor, dx=1.8, dy=1.0) for _ in range(count)]

    if asset_type == "vase":
        anchor = anchors.get("table_top") or anchors.get("dining_table")
        if anchor is not None:
            return [_offset_pose(anchor, z_offset=0.62) for _ in range(count)]

    return [_clone_pose(fallback) or {"xy": (0.0, 0.0), "rot_z": 0.0} for _ in range(count)]


def resolve_factory_params(
    asset_type: str,
    scene_program: dict[str, Any],
    spec: dict[str, Any],
    param_opts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base_key = next(iter(param_opts))
    params = deepcopy(param_opts[base_key])
    text = _normalize_text(
        scene_program.get("prompt", ""),
        spec.get("placement", ""),
        spec.get("rationale", ""),
        *scene_program.get("hypotheses", []),
        *(constraint.get("text", "") for constraint in scene_program.get("constraints", [])),
    )

    if asset_type == "dining_table" and "round" in text:
        params = deepcopy(param_opts.get("b", params))
        _, _, base_height = params.get("dimensions", (1.45, 1.45, 0.75))
        params.update(
            {
                "dimensions": (1.45, 1.45, float(base_height)),
                "Top Profile N-gon": 32,
                "Top Profile Fillet Ratio": 0.0,
                "Leg Style": "single_stand",
                "Leg Number": 2,
                "Leg Placement Top Relative Scale": 0.42,
                "Leg Placement Bottom Relative Scale": 0.9,
                "Leg Diameter": max(float(params.get("Leg Diameter", 0.18)), 0.25),
                "Leg Curve Control Points": [(0.0, 1.0), (0.6, 0.72), (1.0, 1.15)],
            }
        )
        return params

    if asset_type == "rug":
        if _contains_any(
            text,
            (
                "beneath the dining table",
                "under the dining table",
                "beneath dining table",
                "under dining table",
            ),
        ):
            if "round" in text:
                params = deepcopy(param_opts.get("b", params))
                params.update({"width": 3.1, "length": 3.1, "rug_shape": "circle"})
            else:
                params = deepcopy(param_opts.get("c", params))
                params.update(
                    {
                        "width": 2.8,
                        "length": 3.6,
                        "rug_shape": "rounded",
                        "rounded_buffer": 0.35,
                    }
                )
            return params
        if _contains_any(
            text,
            (
                "under the sofa",
                "under sofa",
                "in front of the sofa",
                "in front of sofa",
            ),
        ):
            params = deepcopy(param_opts.get("a", params))
            params.update({"width": 2.8, "length": 3.8, "rug_shape": "rectangle"})
            return params

    if asset_type == "chair" and scene_program.get("room_type") == "dining_room":
        return deepcopy(param_opts.get("b", params))

    if asset_type == "sofa" and "cozy" in text:
        return deepcopy(param_opts.get("b", params))

    return params
