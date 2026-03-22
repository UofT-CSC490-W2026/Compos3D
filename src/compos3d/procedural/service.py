"""
Stage 2 procedural backend service.

backend-smoke:      render a single controllable asset with bpy, write a manifest.
reference-generate: run the infinigen indoor pipeline for one room type, write a manifest.
build-scene:        build a 3D scene from a SceneProgram JSON, render 4 views + video.
feature-extract:    Stage 4 stub (not yet implemented).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from compos3d._stages import stage_pending


@dataclass(frozen=True)
class BackendSmokeRequest:
    asset_name: str
    output_dir: Path
    seed: int = 0
    param_mode: str = "a"
    resolution: str = "512x512"
    samples: int = 32
    save_blend: bool = False


@dataclass(frozen=True)
class ReferenceGenerationRequest:
    room_type: str
    output_dir: Path
    seed: int = 0
    tasks: tuple[str, ...] = ("coarse",)
    minimal: bool = False


@dataclass(frozen=True)
class BuildSceneRequest:
    scene_program_path: Path
    output_dir: Path
    seed: int = 0
    resolution: str = "512x512"
    view_samples: int = 48
    video_samples: int = 16
    video_frames: int = 90
    no_video: bool = False
    save_blend: bool = False
    fps: int = 30


def run_backend_smoke(request: BackendSmokeRequest) -> dict:
    """
    Render one asset from llm_doc/library.py via asset_smoke.py.

    Writes:
      <output_dir>/<asset_name>_seed<seed>.png
      <output_dir>/<asset_name>_seed<seed>_manifest.json
      <output_dir>/smoke_manifest.json  (service-level manifest)
    """
    from compos3d.procedural.runner import SCRIPTS_DIR, run_script

    out_name = f"{request.asset_name}_seed{request.seed}"
    out_dir = request.output_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    script = SCRIPTS_DIR / "asset_smoke.py"
    cli_args: list[str] = [
        "--factory_name",
        request.asset_name,
        "--seed",
        str(request.seed),
        "--out_dir",
        str(out_dir),
        "--out_name",
        out_name,
        "--param_mode",
        request.param_mode,
        "--resolution",
        request.resolution,
        "--samples",
        str(request.samples),
    ]
    if request.save_blend:
        cli_args.append("--save_blend")

    run_result = run_script(script, cli_args)

    render_path = out_dir / f"{out_name}.png"
    if not render_path.exists():
        raise RuntimeError(
            f"Backend smoke completed but render not found at {render_path}.\n"
            "Check the script output above for errors."
        )

    manifest = {
        "stage": "backend_smoke",
        "asset_name": request.asset_name,
        "param_mode": request.param_mode,
        "seed": request.seed,
        "resolution": request.resolution,
        "samples": request.samples,
        "out_dir": str(out_dir),
        "render_path": str(render_path),
        **run_result,
    }
    (out_dir / "smoke_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def generate_reference_scene(request: ReferenceGenerationRequest) -> dict:
    """
    Generate a single-room scene via generate_room.py.

    Default tasks=("coarse",) produces the structural layout quickly.
    Add "populate" + "render" for a full furnished + rendered output.

    Writes:
      <output_dir>/<room_type>_seed<seed>/   (infinigen output tree)
      <output_dir>/<room_type>_seed<seed>/room_manifest.json
      <output_dir>/ref_manifest.json
    """
    from compos3d.procedural.runner import SCRIPTS_DIR, run_script

    valid_rooms = ("dining_room", "living_room", "bedroom")
    if request.room_type not in valid_rooms:
        raise ValueError(
            f"room_type must be one of {valid_rooms}, got '{request.room_type}'"
        )

    scene_dir = request.output_dir / f"{request.room_type}_seed{request.seed}"
    scene_dir.mkdir(parents=True, exist_ok=True)

    script = SCRIPTS_DIR / "generate_room.py"
    cli_args: list[str] = [
        "--room",
        request.room_type,
        "--output_folder",
        str(scene_dir),
        "--seed",
        str(request.seed),
        "--task",
        *list(request.tasks),
    ]
    if request.minimal:
        cli_args.append("--minimal")

    # generate_room.py resolves gin configs relative to a directory that
    # contains infinigen_examples/configs_indoor/.  GIN_CONFIG_ROOT is
    # resolved at import time to the best available candidate.
    from compos3d.procedural.runner import GIN_CONFIG_ROOT

    run_result = run_script(script, cli_args, cwd=GIN_CONFIG_ROOT)

    manifest = {
        "stage": "reference_generate",
        "room_type": request.room_type,
        "seed": request.seed,
        "tasks": list(request.tasks),
        "scene_dir": str(scene_dir),
        **run_result,
    }
    (request.output_dir / "ref_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    return manifest


def build_scene(request: BuildSceneRequest) -> dict:
    """
    Build a 3D scene from a SceneProgram JSON using build_scene.py.

    Writes into request.output_dir:
      views/view_overhead.png, view_front.png, view_left.png, view_right.png
      video_frames/frame_NNNN.png
      video.mp4                 (if ffmpeg available and no_video=False)
      scene.blend               (if save_blend=True)
      build_manifest.json
    """
    from compos3d.procedural.runner import GIN_CONFIG_ROOT, SCRIPTS_DIR, run_script

    request.output_dir.mkdir(parents=True, exist_ok=True)
    script = SCRIPTS_DIR / "build_scene.py"

    # Resolve to absolute so the subprocess can find them regardless of cwd
    scene_program_abs = request.scene_program_path.resolve()
    output_dir_abs = request.output_dir.resolve()

    cli_args: list[str] = [
        "--scene_program",
        str(scene_program_abs),
        "--output_dir",
        str(output_dir_abs),
        "--seed",
        str(request.seed),
        "--resolution",
        request.resolution,
        "--view_samples",
        str(request.view_samples),
        "--video_samples",
        str(request.video_samples),
        "--video_frames",
        str(request.video_frames),
        "--fps",
        str(request.fps),
    ]
    if request.no_video:
        cli_args.append("--no_video")
    if request.save_blend:
        cli_args.append("--save_blend")

    # build_scene.py uses gin configs just like generate_room.py
    run_result = run_script(script, cli_args, cwd=GIN_CONFIG_ROOT)

    manifest_path = request.output_dir / "build_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "stage": "build_scene",
            "output_dir": str(request.output_dir),
            **run_result,
        }
    return manifest


def make_training_renderer(render_config):
    """
    Return a renderer callable suitable for passing to SceneHypothesisLoop.

    If render_config.enabled is False, returns None (no rendering).

    The returned callable has signature:
        renderer(scene_program: SceneProgram, render_dir: Path) -> list[Path]

    It writes a scene_program.json into render_dir, calls build_scene, and
    returns the list of rendered view image Paths.
    """
    if not render_config.enabled:
        return None

    cfg = render_config

    def _renderer(scene_program, render_dir: Path) -> list[Path]:
        import json as _json

        render_dir.mkdir(parents=True, exist_ok=True)
        sp_path = render_dir / "scene_program.json"
        sp_path.write_text(_json.dumps(scene_program.model_dump(), indent=2))
        req = BuildSceneRequest(
            scene_program_path=sp_path,
            output_dir=render_dir,
            resolution=cfg.resolution,
            view_samples=cfg.view_samples,
            no_video=True,  # never render video during training loop
            save_blend=getattr(cfg, "save_blend", False),
        )
        result = build_scene(req)
        rendered = result.get("rendered_views", [])
        return [Path(p) for p in rendered if p and Path(p).exists()]

    return _renderer


def feature_extract(*, input_path: Path, output_dir: Path) -> None:
    _ = (input_path, output_dir)
    stage_pending("Reference feature extraction", stage=4)
