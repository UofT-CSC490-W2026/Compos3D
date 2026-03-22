from __future__ import annotations
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
import bpy
from mathutils import Vector
from infinigen.core import init
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.rendering.render import set_displacement_mode
from infinigen.assets.lighting import sky_lighting
PLACEMENTS: dict[str, dict[str, object]] = {'dining_room': {'dining_table': {'xy': (0.0, 0.0), 'rot_z': 0.0}, 'chair': 'around_table', 'rug': {'xy': (0.0, 0.0), 'rot_z': 0.0}, 'lamp': {'xy': (2.1, 2.1), 'rot_z': 0.0}, 'window': None, 'table_top': {'xy': (-2.1, 1.8), 'rot_z': 0.785}, 'vase': {'xy': (-2.1, 1.8), 'rot_z': 0.0, 'z_offset': 0.62}}, 'living_room': {'sofa': {'xy': (0.0, -1.9), 'rot_z': 0.0}, 'rug': {'xy': (0.0, -0.3), 'rot_z': 0.0}, 'lamp': {'xy': (2.0, -1.9), 'rot_z': 0.0}, 'table_top': {'xy': (0.0, -0.3), 'rot_z': 0.0}, 'window': None, 'vase': {'xy': (0.15, -0.3), 'rot_z': 0.0, 'z_offset': 0.62}}, 'bedroom': {'lamp': {'xy': (1.5, 1.5), 'rot_z': 0.0}, 'rug': {'xy': (0.0, 0.0), 'rot_z': 0.0}, 'chair': {'xy': (-1.5, 1.0), 'rot_z': -0.785}, 'table_top': {'xy': (1.5, 0.5), 'rot_z': 0.0}, 'window': None, 'vase': {'xy': (1.5, 0.5), 'rot_z': 0.0, 'z_offset': 0.62}}}
CHAIR_ORBIT_RADIUS = 0.87

def _chair_positions(count: int) -> list[dict]:
    positions = []
    for i in range(count):
        angle = 2 * math.pi * i / count
        x = CHAIR_ORBIT_RADIUS * math.sin(angle)
        y = -CHAIR_ORBIT_RADIUS * math.cos(angle)
        rot_z = angle
        positions.append({'xy': (x, y), 'rot_z': rot_z})
    return positions

def build_room(room_size: float=8.0) -> None:
    bpy.ops.mesh.primitive_plane_add(size=room_size, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = 'RoomFloor'
    mat = bpy.data.materials.new(name='FloorMat')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.82, 0.8, 0.76, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.8
    floor.data.materials.append(mat)

def spawn_assets(scene_program: dict, args) -> None:
    import llm_doc.library as lib
    room_type = scene_program.get('room_type', 'dining_room')
    placement_table = PLACEMENTS.get(room_type, {})
    assets = scene_program.get('assets', [])
    chair_count = sum((spec.get('count', 1) for spec in assets if spec['asset_type'] == 'chair'))
    asset_instance_idx = 0
    for spec in assets:
        asset_type = spec.get('asset_type', '')
        count = max(1, int(spec.get('count', 1)))
        if asset_type not in lib.LIB_MAP:
            print(f"[build_scene] Warning: '{asset_type}' not in LIB_MAP, skipping.")
            continue
        placement = placement_table.get(asset_type)
        if placement is None:
            print(f'[build_scene] Skipping wall/unsupported asset: {asset_type}')
            continue
        FI = lib.LIB_MAP[asset_type]
        if placement == 'around_table':
            positions = _chair_positions(chair_count)[:count]
        else:
            positions = [placement] * count
        for i, pos_spec in enumerate(positions):
            seed = args.seed + asset_instance_idx & 65535
            asset_instance_idx += 1
            with FixedSeed(seed):
                param_mode = list(FI.PARAM_OPTS.keys())[0]
                params = dict(FI.PARAM_OPTS[param_mode])
                fac = FI.CLS(params)
                asset = fac.spawn_asset(seed)
                fac.finalize_assets(asset)
            bpy.context.view_layer.objects.active = asset
            xy = pos_spec.get('xy', (0.0, 0.0))
            z_off = pos_spec.get('z_offset', 0.0)
            rot_z = pos_spec.get('rot_z', 0.0)
            asset.location = (float(xy[0]), float(xy[1]), float(z_off))
            asset.rotation_euler[2] = float(rot_z)
            print(f'[build_scene] Placed {asset_type}[{i}] at {xy} z_off={z_off:.2f}')
    set_displacement_mode()

def _aim_camera(camera, position: tuple, target: tuple=(0.0, 0.0, 1.0)) -> None:
    camera.location = position
    direction = Vector(target) - Vector(position)
    rot = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot.to_euler()
    bpy.context.view_layer.update()

def setup_lighting(scene_program: dict) -> None:
    bpy.ops.object.camera_add(location=(0, -5, 3))
    tmp_cam = bpy.context.active_object
    bpy.context.scene.camera = tmp_cam
    _aim_camera(tmp_cam, (0, -5, 3), (0, 0, 1))
    with FixedSeed(42):
        sky_lighting.add_lighting(tmp_cam)
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 4.5))
    fill = bpy.context.active_object
    fill.data.energy = 150
    fill.data.size = 4.0
    bpy.data.objects.remove(tmp_cam, do_unlink=True)
_ORBIT_R = 5.0
_ORBIT_H = 3.5
_TARGET = (0.0, 0.0, 0.9)
VIEWS = [{'name': 'overhead', 'position': (0.0, 0.0, 8.0), 'target': (0.0, 0.0, 0.0)}, {'name': 'front', 'position': (0.0, -_ORBIT_R, _ORBIT_H), 'target': _TARGET}, {'name': 'left', 'position': (_ORBIT_R * 0.866, _ORBIT_R * 0.5, _ORBIT_H), 'target': _TARGET}, {'name': 'right', 'position': (-_ORBIT_R * 0.866, _ORBIT_R * 0.5, _ORBIT_H), 'target': _TARGET}]

def render_views(out_dir: Path, resolution: tuple[int, int], samples: int) -> list[Path]:
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.cycles.samples = samples
    scene.render.image_settings.file_format = 'PNG'
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = 'SceneCamera'
    bpy.context.scene.camera = camera
    rendered = []
    for view in VIEWS:
        _aim_camera(camera, view['position'], view['target'])
        path = out_dir / ("view_" + view["name"] + ".png")
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        print(f'[build_scene] Rendered view: {path.name}')
        rendered.append(path)
    return rendered

def render_orbital_video(frames_dir: Path, n_frames: int, resolution: tuple[int, int], samples: int, orbit_radius: float=5.5, orbit_elevation: float=3.2, target_z: float=1.0) -> list[Path]:
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.cycles.samples = samples
    scene.render.image_settings.file_format = 'PNG'
    cam_obj = bpy.data.objects.get('SceneCamera')
    if cam_obj is None:
        bpy.ops.object.camera_add()
        cam_obj = bpy.context.active_object
        cam_obj.name = 'SceneCamera'
    bpy.context.scene.camera = cam_obj
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    for i in range(n_frames):
        angle = 2 * math.pi * i / n_frames
        x = orbit_radius * math.sin(angle)
        y = -orbit_radius * math.cos(angle)
        pos = (x, y, orbit_elevation)
        _aim_camera(cam_obj, pos, (0.0, 0.0, target_z))
        path = frames_dir / f'frame_{i:04d}.png'
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        frame_paths.append(path)
        if i % 10 == 0:
            print(f'[build_scene] Video frame {i + 1}/{n_frames}')
    return frame_paths

def _get_ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    return None

def compile_video(frames_dir: Path, out_path: Path, fps: int=30) -> bool:
    ffmpeg_exe = _get_ffmpeg_exe()
    if ffmpeg_exe is None:
        print('[build_scene] No ffmpeg found — frames saved, compile manually.')
        return False
    try:
        result = subprocess.run([ffmpeg_exe, '-y', '-r', str(fps), '-i', str(frames_dir / 'frame_%04d.png'), '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', str(out_path)], capture_output=True, text=True)
        if result.returncode == 0:
            print(f'[build_scene] Video compiled: {out_path}')
            return True
        else:
            print(f'[build_scene] ffmpeg failed: {result.stderr[-300:]}')
            return False
    except Exception as exc:
        print(f'[build_scene] ffmpeg error: {exc}')
        return False

def main():
    parser = argparse.ArgumentParser(description='Build a 3D scene from a SceneProgram JSON.')
    parser.add_argument('--scene_program', type=Path, required=True, help='Path to scene_program.json produced by run-inference')
    parser.add_argument('--output_dir', type=Path, required=True, help='Directory to write all outputs into')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resolution', type=str, default='512x512')
    parser.add_argument('--view_samples', type=int, default=48, help='CYCLES samples per 4-view image (48 = fast+clean)')
    parser.add_argument('--video_samples', type=int, default=16, help='CYCLES samples per video frame (16 = fast)')
    parser.add_argument('--video_frames', type=int, default=90, help='Number of orbital video frames (90 = 3s at 30fps)')
    parser.add_argument('--no_video', action='store_true', default=False, help='Skip video rendering (only render 4 views)')
    parser.add_argument('--save_blend', action='store_true', default=False)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--configs', type=str, nargs='+', default=[])
    parser.add_argument('--overrides', type=str, nargs='+', default=[])
    try:
        args = init.parse_args_blender(parser)
    except Exception:
        args = parser.parse_args()
    t0 = time.time()
    res = tuple((int(x) for x in args.resolution.split('x')))
    scene_program = json.loads(args.scene_program.read_text())
    room_type = scene_program.get('room_type', 'dining_room')
    asset_types = [a["asset_type"] for a in scene_program.get("assets", [])]
    print(f"[build_scene] room_type={room_type}  assets={asset_types}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    views_dir = args.output_dir / 'views'
    views_dir.mkdir(exist_ok=True)
    frames_dir = args.output_dir / 'video_frames'
    init.apply_gin_configs(['infinigen_examples/configs_nature', 'infinigen_examples/configs_indoor'], configs=args.configs, overrides=args.overrides, skip_unknown=True)
    init.configure_blender()
    init.configure_render_cycles()
    butil.clear_scene()
    build_room()
    setup_lighting(scene_program)
    spawn_assets(scene_program, args)
    if args.save_blend:
        blend_path = args.output_dir / 'scene.blend'
        butil.save_blend(blend_path, autopack=True)
        print(f'[build_scene] Saved: {blend_path}')
    rendered_views = render_views(views_dir, res, args.view_samples)
    video_path = args.output_dir / 'video.mp4'
    video_compiled = False
    frame_paths: list[Path] = []
    if not args.no_video:
        frame_paths = render_orbital_video(frames_dir=frames_dir, n_frames=args.video_frames, resolution=res, samples=args.video_samples, target_z=0.9)
        video_compiled = compile_video(frames_dir, video_path, fps=args.fps)
    elapsed = round(time.time() - t0, 2)
    manifest = {'scene_program_path': str(args.scene_program), 'room_type': room_type, 'seed': args.seed, 'resolution': args.resolution, 'view_samples': args.view_samples, 'video_samples': args.video_samples, 'video_frames': args.video_frames, 'rendered_views': [str(p) for p in rendered_views], 'video_frames_dir': str(frames_dir) if frame_paths else None, 'video_path': str(video_path) if video_compiled else None, 'video_compiled': video_compiled, 'elapsed_seconds': elapsed, 'output_dir': str(args.output_dir)}
    manifest_path = args.output_dir / 'build_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'[build_scene] Done in {elapsed}s  manifest={manifest_path}')
if __name__ == '__main__':
    main()
