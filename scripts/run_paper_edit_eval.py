#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from compos3d.config import CriticConfig
from compos3d.paper import export_edit_human_study_sheet, run_edit_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen-bank prompt-edit benchmark for the dining-room paper."
    )
    parser.add_argument(
        "--edit-pairs-path",
        type=Path,
        default=Path("paper/benchmarks/edit_pairs.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-name", type=str, required=True)
    parser.add_argument("--bank-path", type=Path, default=None)
    parser.add_argument("--use-empty-bank", action="store_true")
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--llm-provider", type=str, default="mock")
    parser.add_argument("--top-k-override", type=int, default=None)
    parser.add_argument("--inference-strategy", type=str, default="filter_and_weight")
    parser.add_argument("--render-scene", action="store_true")
    parser.add_argument("--render-resolution", type=str, default="512x512")
    parser.add_argument("--render-view-samples", type=int, default=48)
    parser.add_argument("--render-video-frames", type=int, default=90)
    parser.add_argument("--render-video-samples", type=int, default=16)
    parser.add_argument("--judge-edits", action="store_true")
    parser.add_argument(
        "--judge-model-id",
        type=str,
        default="qwen.qwen3-vl-235b-a22b",
    )
    parser.add_argument("--judge-region", type=str, default="us-east-1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--export-human-csv",
        type=Path,
        default=None,
        help="Optional CSV template for the 30-pair human edit study.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_overrides = (
        {"training": {"top_k": args.top_k_override}}
        if args.top_k_override is not None
        else None
    )
    judge_config = None
    if args.judge_edits:
        judge_config = CriticConfig(
            mode="vlm",
            provider="bedrock",
            model_id=args.judge_model_id,
            region_name=args.judge_region,
            max_tokens=800,
            temperature=0.0,
        )

    result = run_edit_benchmark(
        edit_pairs_path=args.edit_pairs_path,
        output_dir=args.output_dir,
        method_name=args.method_name,
        bank_path=args.bank_path,
        use_empty_bank=args.use_empty_bank,
        config_path=args.config_path,
        config_overrides=config_overrides,
        llm_provider=args.llm_provider,
        inference_strategy=args.inference_strategy,
        render_scene=args.render_scene,
        render_resolution=args.render_resolution,
        render_view_samples=args.render_view_samples,
        render_video_frames=args.render_video_frames,
        render_video_samples=args.render_video_samples,
        judge_config=judge_config,
        limit=args.limit,
        force=args.force,
    )
    print(f"Wrote {len(result['rows'])} edit rows to {result['results_path']}")
    print(result["summary"])
    if args.export_human_csv is not None:
        csv_path = export_edit_human_study_sheet(
            editing_results_path=args.output_dir,
            output_csv_path=args.export_human_csv,
        )
        print(f"Wrote edit human-study sheet to {csv_path}")


if __name__ == "__main__":
    main()
