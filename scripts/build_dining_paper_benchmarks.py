#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from compos3d.paper import build_dining_paper_benchmarks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the deterministic dining-room paper benchmark and edit pairs."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/external/spatiallm"),
        help="Directory containing split.csv and spatiallm_train.json.",
    )
    parser.add_argument(
        "--canonical-training-dataset-path",
        type=Path,
        default=Path("examples/vertical_slice_dataset.json"),
        help="Training dataset used for the final dining-room run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/benchmarks"),
        help="Output directory for benchmark manifests.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_dining_paper_benchmarks(
        raw_dir=args.raw_dir,
        canonical_training_dataset_path=args.canonical_training_dataset_path,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    metadata = result["metadata"]
    print(f"Wrote dining benchmark manifests to {args.output_dir}")
    print(f"Held-out dining examples: {metadata['heldout_dining_examples']}")
    print(f"Edit pairs: {sum(metadata['edit_pair_counts'].values())}")


if __name__ == "__main__":
    main()
