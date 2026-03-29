#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from compos3d.paper.figures import render_training_paper_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper figures from the finalized local training run."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/training/claude_qwen"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures"),
    )
    parser.add_argument("--room-type", type=str, default="dining_room")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = render_training_paper_figures(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        room_type=args.room_type,
    )
    print(jsonable(manifest))


def jsonable(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2)


if __name__ == "__main__":
    main()
