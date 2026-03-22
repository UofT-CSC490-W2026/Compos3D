from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path
_repo = Path(__file__).resolve().parent.parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))
logging.basicConfig(format='[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
ROOM_TAGS = {'bedroom': 'Bedroom', 'dining_room': 'DiningRoom', 'living_room': 'LivingRoom'}

def main():
    parser = argparse.ArgumentParser(description='Generate a single-room indoor scene.')
    parser.add_argument('-r', '--room', required=True, choices=list(ROOM_TAGS.keys()))
    parser.add_argument('-o', '--output_folder', type=Path, required=True)
    parser.add_argument('-s', '--seed', default=None)
    parser.add_argument('-t', '--task', nargs='+', default=['coarse'], choices=['coarse', 'populate', 'fine_terrain', 'ground_truth', 'render', 'mesh_save', 'export'])
    parser.add_argument('--input_folder', type=Path, default=None)
    parser.add_argument('-p', '--overrides', nargs='+', default=[])
    parser.add_argument('-g', '--configs', nargs='+', default=[])
    parser.add_argument('--minimal', action='store_true')
    parser.add_argument('--task_uniqname', type=str, default=None)
    parser.add_argument('-d', '--debug', type=str, nargs='*', default=None)
    try:
        from infinigen.core import init as _init
        args = _init.parse_args_blender(parser)
    except Exception:
        args = parser.parse_args()
    t0 = time.time()
    args.output_folder.mkdir(parents=True, exist_ok=True)
    room_tag = ROOM_TAGS[args.room]
    overrides = list(args.overrides)
    overrides.append('restrict_solving.solve_max_rooms=1')
    overrides.append(f'restrict_solving.restrict_parent_rooms=["{room_tag}"]')
    configs = list(args.configs) if args.configs else ['singleroom', 'overhead']
    if getattr(args, 'minimal', False):
        configs.append('minimal_solve')
    from infinigen.core import init
    from infinigen.core import execute_tasks
    from infinigen_examples import generate_indoors
    scene_seed = init.apply_scene_seed(args.seed)
    init.apply_gin_configs(configs=['base_indoors.gin'] + configs, overrides=overrides, config_folders=['infinigen_examples/configs_indoor', 'infinigen_examples/configs_nature'])
    execute_tasks.main(compose_scene_func=generate_indoors.compose_indoors, populate_scene_func=None, input_folder=args.input_folder, output_folder=args.output_folder, task=args.task, task_uniqname=args.task_uniqname, scene_seed=scene_seed)
    elapsed = round(time.time() - t0, 2)
    manifest = {'room_type': args.room, 'seed': str(args.seed), 'scene_seed': int(scene_seed), 'tasks': list(args.task), 'output_folder': str(args.output_folder), 'elapsed_seconds': elapsed}
    manifest_path = args.output_folder / 'room_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'[generate_room] Manifest: {manifest_path}')
    print(f'[generate_room] Done in {elapsed}s')
if __name__ == '__main__':
    main()
