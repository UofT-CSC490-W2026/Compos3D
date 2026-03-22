from __future__ import annotations
import argparse
import json
import math
import time
from itertools import product
from pathlib import Path
import bpy
import gin
import numpy as np
import infinigen
from infinigen.core import init
from infinigen.core.util import blender as butil
from infinigen.core.util.camera import points_inview
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.test_utils import import_item, load_txt_list
from infinigen.assets.lighting import sky_lighting
from infinigen.assets.utils.decorate import read_base_co
from infinigen.core.rendering.render import set_displacement_mode
from numpy.random import choice, normal, uniform

def setup_camera(args):
    cam_dist = args.cam_dist if args.cam_dist > 0 else 6.0
    bpy.ops.object.camera_add(location=(0, -cam_dist, 0), rotation=(np.pi / 2, 0, 0))
    camera = bpy.context.active_object
    camera.parent = butil.spawn_empty('Camera parent')
    camera.parent.location = (0, 0, 0)
    camera.parent.rotation_euler = np.deg2rad(np.array(args.cam_angle))
    bpy.context.scene.camera = camera
    scene = bpy.context.scene
    camera.data.sensor_height = camera.data.sensor_width * scene.render.resolution_y / scene.render.resolution_x
    return (camera, camera.parent)

def adjust_cam_target_to_asset(asset, camera_parent, zoff=0.0):
    co = read_base_co(asset)
    center_local = (np.amin(co, 0) + np.amax(co, 0)) / 2
    center_world = (np.array(asset.matrix_world) @ np.array([*center_local, 1.0]))[:-1]
    camera_parent.location = (center_world[0], center_world[1], center_world[2] + zoff)

def adjust_cam_distance_autofit(asset, camera, margin=0.01, percent=0.999):
    co = read_base_co(asset)
    lowest = np.amin(co, 0)
    highest = np.amax(co, 0)
    interp = np.linspace(lowest, highest, 11)
    bbox_pts_local = np.array(list(product(*zip(*interp))))
    bbox_pts_world = (np.array(asset.matrix_world) @ np.c_[bbox_pts_local, np.ones(len(bbox_pts_local))].T).T[:, :3]
    for cam_dist in np.exp(np.linspace(-1.0, 5.5, 500)):
        camera.location[1] = -float(cam_dist)
        bpy.context.view_layer.update()
        inview = points_inview(bbox_pts_world, camera)
        if inview.sum() / inview.size >= percent:
            camera.location[1] *= 1.0 + float(margin)
            bpy.context.view_layer.update()
            return float(-camera.location[1])
    camera.location[1] = -6.0
    bpy.context.view_layer.update()
    return 6.0

def build_object(FI, args):
    factory_cls = FI.CLS
    assert args.param_mode in FI.PARAM_OPTS, f"param_mode '{args.param_mode}' not in PARAM_OPTS {list(FI.PARAM_OPTS.keys())}"
    params = dict(FI.PARAM_OPTS[args.param_mode])
    if args.param_data is not None:
        params.update(args.param_data)
    fac = factory_cls(params)
    asset = fac.spawn_asset(args.seed)
    fac.finalize_assets(asset)
    return asset

def main():
    parser = argparse.ArgumentParser(description='Render a single controllable asset.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-fn', '--factory_name', type=str, required=True)
    parser.add_argument('-on', '--out_name', type=str, required=True)
    parser.add_argument('-pm', '--param_mode', type=str, default='a')
    parser.add_argument('-pd', '--param_data', type=json.loads, default=None)
    parser.add_argument('-od', '--out_dir', type=Path, default=Path('outputs/'))
    parser.add_argument('--save_blend', action='store_true', default=False)
    parser.add_argument('--render', action='store_true', default=True)
    parser.add_argument('--no_render', action='store_true', default=False)
    parser.add_argument('--resolution', type=str, default='512x512')
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--cam_dist', type=float, default=0.0)
    parser.add_argument('--cam_angle', type=float, nargs='+', default=(-30, 0, 45))
    parser.add_argument('--cam_zoff', type=float, default=0.0)
    parser.add_argument('--sun_elevation', type=float, default=60.0)
    parser.add_argument('--margin', type=float, default=0.01)
    parser.add_argument('--film_transparent', action='store_true')
    parser.add_argument('--configs', type=str, nargs='+', default=[])
    parser.add_argument('--overrides', type=str, nargs='+', default=[])
    try:
        args = init.parse_args_blender(parser)
    except Exception:
        args = parser.parse_args()
    if args.no_render:
        args.render = False
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    init.apply_gin_configs(['infinigen_examples/configs_nature', 'infinigen_examples/configs_indoor'], configs=args.configs, overrides=args.overrides, skip_unknown=True)
    init.configure_blender()
    if args.gpu:
        init.configure_render_cycles()
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x, scene.render.resolution_y = map(int, args.resolution.split('x'))
    scene.cycles.samples = args.samples
    scene.render.film_transparent = bool(args.film_transparent)
    butil.clear_scene()
    import llm_doc.library as lib
    if args.factory_name not in lib.LIB_MAP:
        raise ValueError(f"factory_name '{args.factory_name}' not in LIB_MAP. Available: {sorted(lib.LIB_MAP.keys())}")
    FI = lib.LIB_MAP[args.factory_name]
    asset = build_object(FI, args)
    bpy.context.view_layer.objects.active = asset
    set_displacement_mode()
    camera, camera_parent = setup_camera(args)
    try:
        adjust_cam_target_to_asset(asset, camera_parent, zoff=args.cam_zoff)
    except Exception as exc:
        print(f'[asset_smoke] Camera target adjustment failed: {exc}')
    if args.cam_dist <= 0:
        try:
            fitted = adjust_cam_distance_autofit(asset, camera, margin=args.margin)
            print(f'[asset_smoke] Auto-fit camera distance: {fitted:.3f}')
        except Exception as exc:
            print(f'[asset_smoke] Auto-fit failed: {exc}')
    with FixedSeed(args.seed + 999):
        sky_lighting.add_lighting(camera)
        try:
            nodes = bpy.data.worlds['World'].node_tree.nodes
            sky_nodes = [n for n in nodes if n.name.startswith('Sky Texture')]
            if sky_nodes:
                sky_nodes[-1].sun_elevation = np.deg2rad(args.sun_elevation)
        except Exception:
            pass
    render_path: Path | None = None
    if args.save_blend:
        blend_path = args.out_dir / f'{args.out_name}.blend'
        butil.save_blend(blend_path, autopack=True)
        print(f'[asset_smoke] Saved blend: {blend_path}')
    if args.render:
        render_path = args.out_dir / f'{args.out_name}.png'
        scene.render.filepath = str(render_path)
        bpy.ops.render.render(write_still=True)
        print(f'[asset_smoke] Rendered: {render_path}')
    elapsed = round(time.time() - t0, 2)
    manifest = {'factory_name': args.factory_name, 'param_mode': args.param_mode, 'seed': args.seed, 'out_dir': str(args.out_dir), 'out_name': args.out_name, 'render_path': str(render_path) if render_path else None, 'render_exists': render_path.exists() if render_path else False, 'elapsed_seconds': elapsed}
    manifest_path = args.out_dir / f'{args.out_name}_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'[asset_smoke] Manifest: {manifest_path}')
    print(f'[asset_smoke] Done in {elapsed}s')
if __name__ == '__main__':
    main()
