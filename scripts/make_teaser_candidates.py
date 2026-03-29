#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from compos3d.catalog import infer_room_type
from compos3d.config import DEFAULT_EXPERIMENT_CONFIG, load_experiment_config
from compos3d.hypothesis.engine import (
    _load_bank,
    _run_filter_and_weight_inference,
    _select_hypotheses_for_inference,
    _write_json,
)
from compos3d.llm.scene_llm import build_scene_llm
from compos3d.paper.benchmarks import expected_asset_counts, load_benchmark_examples
from compos3d.procedural.service import BuildSceneRequest, build_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank showcase prompts for the teaser image and optionally render them."
    )
    parser.add_argument(
        "--showcase-path",
        type=Path,
        default=Path("paper/benchmarks/dining_showcase.json"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("paper/benchmarks/teaser_candidates.json"),
    )
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--selected-example-ids",
        type=str,
        default="",
        help="Comma-separated example IDs to render. Defaults to the top-ranked prompts.",
    )
    parser.add_argument(
        "--bank-path",
        type=Path,
        default=Path("artifacts/training/claude_qwen/hypothesis_bank.json"),
    )
    parser.add_argument(
        "--config-path", type=Path, default=Path("train_configs/compos3d.json")
    )
    parser.add_argument(
        "--render-output-dir",
        type=Path,
        default=Path("paper/results/generation/teaser_candidates"),
    )
    parser.add_argument("--render-resolution", type=str, default="1024x1024")
    parser.add_argument("--render-view-samples", type=int, default=1000)
    parser.add_argument(
        "--hero-view",
        type=str,
        default="front",
        choices=("overhead", "front", "left", "right"),
    )
    return parser


def _complexity_score(example: dict) -> float:
    counts = expected_asset_counts(example)
    return (
        len(example.get("required_assets", [])) * 4
        + sum(max(int(count) - 1, 0) for count in counts.values())
        + (2 if counts.get("dining_table", 0) > 1 else 0)
        + (1 if "rug" in example.get("required_assets", []) else 0)
        + (1 if "window" in example.get("required_assets", []) else 0)
    )


def _parse_selected_ids(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _hero_view_path(scene_dir: Path, hero_view: str) -> Path:
    return scene_dir / "views" / f"view_{hero_view}.png"


def _render_teaser_examples(
    *,
    selected_examples: list[dict],
    bank_path: Path,
    config_path: Path,
    output_dir: Path,
    resolution: str,
    view_samples: int,
    hero_view: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    heroes_dir = output_dir / "heroes"
    heroes_dir.mkdir(parents=True, exist_ok=True)
    bank = _load_bank(bank_path)
    experiment_config = (
        load_experiment_config(config_path)
        if config_path is not None
        else DEFAULT_EXPERIMENT_CONFIG.model_copy(deep=True)
    )
    llm = build_scene_llm(experiment_config.generator)

    rows: list[dict] = []
    for index, example in enumerate(selected_examples, start=1):
        example_id = str(example["example_id"])
        example_dir = output_dir / f"{index:02d}_{example_id}"
        inference_dir = example_dir / "inference"
        prompt = str(example["prompt"])
        room_type = infer_room_type(prompt)
        candidates = _select_hypotheses_for_inference(
            bank,
            room_type,
            prompt=prompt,
            top_k=experiment_config.training.top_k,
        )
        selected_records, scene_program = _run_filter_and_weight_inference(
            llm=llm,
            records=candidates,
            prompt=prompt,
            room_type=room_type,
        )
        inference_dir.mkdir(parents=True, exist_ok=True)
        scene_program_path = inference_dir / "scene_program.json"
        _write_json(scene_program_path, scene_program.model_dump())
        _write_json(
            inference_dir / "experiment_config.json", experiment_config.model_dump()
        )
        inference_manifest = {
            "bank_path": str(bank_path),
            "config_path": str(config_path) if config_path is not None else None,
            "prompt": prompt,
            "room_type": room_type,
            "inference_strategy": "filter_and_weight",
            "candidate_hypothesis_ids": [record.hypothesis_id for record in candidates],
            "candidate_hypotheses": [record.text for record in candidates],
            "selected_hypothesis_ids": [
                record.hypothesis_id for record in selected_records
            ],
            "selected_hypotheses": [record.text for record in selected_records],
            "scene_program_path": str(scene_program_path),
        }
        _write_json(inference_dir / "inference_manifest.json", inference_manifest)
        scene_dir = inference_dir / "scene"
        render_manifest = build_scene(
            BuildSceneRequest(
                scene_program_path=scene_program_path,
                output_dir=scene_dir,
                resolution=resolution,
                view_samples=view_samples,
                video_frames=1,
                video_samples=1,
                no_video=True,
                save_blend=False,
            )
        )
        hero_source = _hero_view_path(scene_dir, hero_view)
        hero_target = heroes_dir / f"{index:02d}_{example_id}_{hero_view}.png"
        shutil.copy2(hero_source, hero_target)
        prompt_path = heroes_dir / f"{index:02d}_{example_id}_prompt.txt"
        prompt_path.write_text(prompt.strip() + "\n", encoding="utf-8")
        rows.append(
            {
                "rank": index,
                "example_id": example_id,
                "prompt": prompt,
                "required_assets": example["required_assets"],
                "expected_asset_counts": expected_asset_counts(example),
                "hero_view": hero_view,
                "hero_image_path": str(hero_target),
                "prompt_path": str(prompt_path),
                "inference_dir": str(inference_dir),
                "scene_dir": str(scene_dir),
                "scene_program_path": str(scene_program_path),
                "selected_hypotheses": inference_manifest.get(
                    "selected_hypotheses", []
                ),
                "render_manifest": render_manifest,
            }
        )

    payload = {
        "bank_path": str(bank_path),
        "config_path": str(config_path),
        "resolution": resolution,
        "view_samples": view_samples,
        "hero_view": hero_view,
        "scenes": rows,
    }
    manifest_path = output_dir / "teaser_render_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    args = build_parser().parse_args()
    examples = load_benchmark_examples(args.showcase_path)
    ranked = sorted(
        examples,
        key=lambda example: (_complexity_score(example), str(example["example_id"])),
        reverse=True,
    )[: args.top_n]
    payload = {
        "showcase_path": str(args.showcase_path),
        "top_n": args.top_n,
        "candidates": [
            {
                "rank": index + 1,
                "example_id": example["example_id"],
                "prompt": example["prompt"],
                "required_assets": example["required_assets"],
                "expected_asset_counts": expected_asset_counts(example),
                "complexity_score": _complexity_score(example),
            }
            for index, example in enumerate(ranked)
        ],
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote teaser candidate ranking to {args.output_path}")

    if args.run:
        selected_ids = _parse_selected_ids(args.selected_example_ids)
        selected_examples = (
            [
                example
                for example in ranked
                if str(example["example_id"]) in selected_ids
            ]
            if selected_ids
            else ranked[:2]
        )
        render_payload = _render_teaser_examples(
            selected_examples=selected_examples,
            bank_path=args.bank_path,
            config_path=args.config_path,
            output_dir=args.render_output_dir,
            resolution=args.render_resolution,
            view_samples=args.render_view_samples,
            hero_view=args.hero_view,
        )
        print(json.dumps(render_payload, indent=2))
        print(f"Rendered teaser scenes into {args.render_output_dir}")


if __name__ == "__main__":
    main()
