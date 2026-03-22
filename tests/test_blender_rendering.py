"""Blender rendering tests (ported from compos3d_dp).

Tests the scene building and rendering pipeline using ``build_scene``.
Blender (bpy) must be installed in the environment.  These tests are marked
``blender`` and are skipped automatically if bpy is not importable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Skip entire module if bpy is not available.
bpy = pytest.importorskip("bpy", reason="bpy not installed — Blender tests skipped")


from compos3d.procedural.service import BuildSceneRequest, build_scene


# ---------------------------------------------------------------------------
# Minimal SceneProgram fixtures
# ---------------------------------------------------------------------------

SIMPLE_DINING_ROOM = {
    "room_type": "dining_room",
    "style": "modern",
    "hypotheses": [],
    "assets": [
        {"asset_type": "dining_table", "count": 1, "placement": "center"},
        {"asset_type": "chair", "count": 2, "placement": "around table"},
    ],
    "constraints": [],
}

SIMPLE_LIVING_ROOM = {
    "room_type": "living_room",
    "style": "cozy",
    "hypotheses": [],
    "assets": [
        {"asset_type": "sofa", "count": 1, "placement": "against wall"},
        {"asset_type": "lamp", "count": 1, "placement": "corner"},
    ],
    "constraints": [],
}


@pytest.fixture
def dining_room_sp(tmp_path) -> Path:
    p = tmp_path / "scene_program.json"
    p.write_text(json.dumps(SIMPLE_DINING_ROOM, indent=2))
    return p


@pytest.fixture
def living_room_sp(tmp_path) -> Path:
    p = tmp_path / "scene_program.json"
    p.write_text(json.dumps(SIMPLE_LIVING_ROOM, indent=2))
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.blender
def test_build_scene_renders_dining_room(dining_room_sp: Path, tmp_path: Path) -> None:
    """build_scene produces 4 rendered PNG views for a dining room."""
    req = BuildSceneRequest(
        scene_program_path=dining_room_sp,
        output_dir=tmp_path / "renders",
        seed=0,
        resolution="256x256",
        view_samples=4,
        no_video=True,
        save_blend=False,
    )
    result = build_scene(req)

    assert "rendered_views" in result
    rendered = result["rendered_views"]
    assert len(rendered) > 0, "At least one view must be rendered"

    for img_path in rendered:
        p = Path(img_path)
        assert p.exists(), f"Rendered file missing: {p}"
        assert p.stat().st_size > 0, f"Rendered file is empty: {p}"


@pytest.mark.blender
def test_build_scene_renders_living_room(living_room_sp: Path, tmp_path: Path) -> None:
    """build_scene works for a living room SceneProgram."""
    req = BuildSceneRequest(
        scene_program_path=living_room_sp,
        output_dir=tmp_path / "renders",
        seed=1,
        resolution="256x256",
        view_samples=4,
        no_video=True,
        save_blend=False,
    )
    result = build_scene(req)
    assert "rendered_views" in result
    assert len(result["rendered_views"]) > 0


@pytest.mark.blender
def test_build_scene_manifest(dining_room_sp: Path, tmp_path: Path) -> None:
    """build_scene returns a manifest dict with expected keys."""
    req = BuildSceneRequest(
        scene_program_path=dining_room_sp,
        output_dir=tmp_path / "renders",
        seed=0,
        resolution="128x128",
        view_samples=2,
        no_video=True,
        save_blend=False,
    )
    result = build_scene(req)

    for key in ("rendered_views", "elapsed_seconds"):
        assert key in result, f"Missing key in manifest: {key}"

    assert isinstance(result["elapsed_seconds"], (int, float))
    assert result["elapsed_seconds"] > 0


@pytest.mark.blender
def test_build_scene_different_seeds_produce_variation(dining_room_sp: Path, tmp_path: Path) -> None:
    """Running build_scene twice with different seeds completes without error."""
    for seed in (0, 42):
        req = BuildSceneRequest(
            scene_program_path=dining_room_sp,
            output_dir=tmp_path / f"renders_seed{seed}",
            seed=seed,
            resolution="128x128",
            view_samples=2,
            no_video=True,
            save_blend=False,
        )
        result = build_scene(req)
        assert len(result["rendered_views"]) > 0
