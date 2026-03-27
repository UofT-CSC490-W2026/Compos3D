"""Edge-case unit tests for procedural service and runner modules.

Validation in service entrypoints, subprocess outcomes, and missing artifacts
in procedural build flows.

Edge cases covered:
- Request validation branches in procedural service commands.
- Missing expected output files after subprocess success.
- Runner subprocess failure and capture-output behavior.
- Missing input scene-program error behavior in build-scene flow.

Expected outcomes:
- Invalid inputs raise clear exceptions.
- Subprocess non-zero exits raise `RuntimeError` with context.
- Service functions fail fast when expected files are missing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from compos3d.procedural import runner, service


def test_generate_reference_scene_invalid_room_type_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="room_type must be one of"):
        service.generate_reference_scene(
            service.ReferenceGenerationRequest(room_type="garage", output_dir=tmp_path)
        )


def test_run_backend_smoke_missing_render_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        service, "run_script", lambda *_a, **_k: {"exit_code": 0}, raising=False
    )
    # patch in module namespace that function uses
    import compos3d.procedural.runner as runner_mod

    monkeypatch.setattr(runner_mod, "run_script", lambda *_a, **_k: {"exit_code": 0})
    with pytest.raises(RuntimeError, match="render not found"):
        service.run_backend_smoke(
            service.BackendSmokeRequest(asset_name="chair", output_dir=tmp_path)
        )


def test_build_scene_missing_input_file_surfaces_subprocess_error(
    monkeypatch, tmp_path
) -> None:
    import compos3d.procedural.runner as runner_mod

    def _boom(*_a, **_k):
        raise RuntimeError("Procedural script failed (exit 2): build_scene.py")

    monkeypatch.setattr(runner_mod, "run_script", _boom)
    with pytest.raises(RuntimeError, match="build_scene.py"):
        service.build_scene(
            service.BuildSceneRequest(
                scene_program_path=tmp_path / "missing_scene_program.json",
                output_dir=tmp_path / "out",
            )
        )


def test_runner_run_script_nonzero_exit_raises(monkeypatch, tmp_path) -> None:
    class _Completed:
        returncode = 1
        stdout = "x"
        stderr = "failed hard"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed())
    with pytest.raises(RuntimeError, match="Procedural script failed"):
        runner.run_script(tmp_path / "script.py", ["--x"], capture_output=True)


def test_runner_run_script_capture_output_success(monkeypatch, tmp_path) -> None:
    class _Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed())
    result = runner.run_script(tmp_path / "script.py", ["--x"], capture_output=True)
    assert result["exit_code"] == 0
    assert result["stdout"] == "ok"
