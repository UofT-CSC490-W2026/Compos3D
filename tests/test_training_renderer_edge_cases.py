"""Unit tests for `make_training_renderer` in procedural service.

Why these tests:
- This helper bridges the training loop and the rendering backend, so mistakes
  here can break the pipeline even if training and rendering work separately.
- The unit tests focus on fail-safe behavior and argument wiring rather than
  Blender execution.

Edge cases covered:
- Rendering disabled -> returns `None`.
- Rendering enabled -> scene program is serialized and passed to `build_scene`.
- Training renderer always forces `no_video=True`.
- Missing render outputs are filtered out gracefully.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

from compos3d.config import RenderConfig
from compos3d.models import AssetSpec, SceneProgram
from compos3d.procedural import service


def test_make_training_renderer_returns_none_when_disabled() -> None:
    renderer = service.make_training_renderer(RenderConfig(enabled=False))
    assert renderer is None


def test_make_training_renderer_writes_scene_program_and_filters_missing_paths(
    monkeypatch,
) -> None:
    captured = {}
    written = {}
    state = {"temp_root_exists": True, "cleaned": None}

    monkeypatch.setattr(
        tempfile,
        "mkdtemp",
        lambda prefix, dir: ".pytest_tmp\\renderer_test_virtual",
    )
    monkeypatch.setattr(os, "makedirs", lambda path, exist_ok=False: None)

    def _fake_rmtree(path: str, ignore_errors: bool = False) -> None:
        assert ignore_errors is False
        state["cleaned"] = path
        state["temp_root_exists"] = False

    real_exists = os.path.exists

    def _fake_os_path_exists(path) -> bool:
        if str(path) == ".pytest_tmp\\renderer_test_virtual":
            return bool(state["temp_root_exists"])
        return real_exists(path)

    monkeypatch.setattr(shutil, "rmtree", _fake_rmtree)
    monkeypatch.setattr(os.path, "exists", _fake_os_path_exists)

    os.makedirs(".pytest_tmp", exist_ok=True)
    temp_root = tempfile.mkdtemp(prefix="renderer_test_", dir=".pytest_tmp")

    def _fake_build_scene(request):
        captured["request"] = request
        existing = request.output_dir / "views" / "view_front.png"
        return {
            "rendered_views": [
                str(existing),
                str(request.output_dir / "views" / "missing.png"),
            ]
        }

    monkeypatch.setattr(service, "build_scene", _fake_build_scene)
    monkeypatch.setattr(Path, "mkdir", lambda self, parents=False, exist_ok=False: None)

    def _fake_write_text(self: Path, text: str) -> int:
        written[str(self)] = text
        return len(text)

    def _fake_exists(self: Path) -> bool:
        return self.name in {"scene_program.json", "view_front.png"}

    monkeypatch.setattr(Path, "write_text", _fake_write_text)
    monkeypatch.setattr(Path, "exists", _fake_exists)

    try:
        renderer = service.make_training_renderer(
            RenderConfig(
                enabled=True,
                resolution="320x240",
                view_samples=7,
                save_blend=True,
            )
        )
        assert renderer is not None

        scene_program = SceneProgram(
            prompt="a dining room",
            room_type="dining_room",
            assets=[AssetSpec(asset_type="chair")],
        )

        render_root = Path(temp_root) / "render_run"
        rendered = renderer(scene_program, render_root)

        scene_program_path = render_root / "scene_program.json"
        assert scene_program_path.exists()
        assert json.loads(written[str(scene_program_path)])["room_type"] == "dining_room"

        req = captured["request"]
        assert req.scene_program_path == scene_program_path
        assert req.resolution == "320x240"
        assert req.view_samples == 7
        assert req.no_video is True
        assert req.save_blend is True

        assert rendered == [render_root / "views" / "view_front.png"]
        assert all(isinstance(path, Path) for path in rendered)
    finally:
        shutil.rmtree(temp_root, ignore_errors=False)
        assert state["cleaned"] == temp_root
        assert not os.path.exists(temp_root)
