#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from compos3d.paper.training_matrix import build_paper_training_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize ablation and sweep configs plus runnable commands for the dining-room paper."
    )
    parser.add_argument(
        "--base-config-path",
        type=Path,
        default=Path("train_configs/compos3d.json"),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("examples/vertical_slice_dataset.json"),
    )
    parser.add_argument(
        "--benchmark-val-path",
        type=Path,
        default=Path("paper/benchmarks/dining_val.json"),
    )
    parser.add_argument(
        "--benchmark-test-path",
        type=Path,
        default=Path("paper/benchmarks/dining_test.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/results/training_matrix"),
    )
    parser.add_argument(
        "--training-output-dir",
        type=Path,
        default=Path("artifacts/training"),
    )
    parser.add_argument("--experiment-prefix", type=str, default="paper_dining")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    matrix = build_paper_training_matrix(
        base_config_path=args.base_config_path,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        benchmark_val_path=args.benchmark_val_path,
        benchmark_test_path=args.benchmark_test_path,
        training_output_dir=args.training_output_dir,
        experiment_prefix=args.experiment_prefix,
    )
    print(f"Wrote experiment matrix to {args.output_dir / 'experiment_matrix.json'}")
    print(
        f"Train runs: {len(matrix['train_runs'])}, sweeps: {len(matrix['sweeps'])}, inference ablations: {len(matrix['inference_ablations'])}"
    )


if __name__ == "__main__":
    main()
