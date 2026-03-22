"""Direct render test (ported from compos3d_dp).

Verifies that a SceneProgram can be built end-to-end: generation pipeline
produces a scene_program.json, which build_scene then renders.

Blender (bpy) must be installed.  Skip gracefully if it is not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

bpy = pytest.importorskip("bpy", reason="bpy not installed — skipping direct render test")

from compos3d.hypothesis.engine import run_vertical_inference
from compos3d.procedural.service import BuildSceneRequest, build_scene


@pytest.fixture
def frozen_bank(tmp_path) -> Path:
    bank = [
        {
            "hypothesis_id": "h1",
            "text": "anchor composition around dining_table",
            "room_type": "dining_room",
            "reward": 0.85, "accuracy": 0.90, "mean_score": 0.87,
            "num_visits": 5, "num_successes": 4, "generation_round": 1,
            "source_example_ids": [], "support_example_ids": [],
            "applicability_tags": [], "failure_tags": [],
        }
    ]
    p = tmp_path / "bank.json"
    p.write_text(json.dumps(bank, indent=2))
    return p


@pytest.mark.blender
def test_direct_render_full_pipeline(frozen_bank: Path, tmp_path: Path) -> None:
    """Infer → render: scene_program.json produced by mock LLM is rendered to PNGs."""
    # Step 1: generate SceneProgram via mock LLM.
    inf_result = run_vertical_inference(
        bank_path=frozen_bank,
        prompt="a dining room with a table and two chairs",
        output_dir=tmp_path / "inference",
        llm_provider="mock",
        render_scene=False,
    )

    sp_path = Path(inf_result["scene_program_path"])
    assert sp_path.exists()

    # Step 2: build and render the scene.
    render_dir = tmp_path / "renders"
    req = BuildSceneRequest(
        scene_program_path=sp_path,
        output_dir=render_dir,
        seed=0,
        resolution="256x256",
        view_samples=4,
        no_video=True,
        save_blend=False,
    )
    render_result = build_scene(req)

    rendered = render_result.get("rendered_views", [])
    assert len(rendered) > 0, "At least one view must be rendered"

    for img_path in rendered:
        p = Path(img_path)
        assert p.exists(), f"Rendered image missing: {p}"
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name}: {size_kb:.1f} KB")
        assert size_kb > 0
