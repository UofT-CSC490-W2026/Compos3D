#!/usr/bin/env python3
"""Run the complete data pipeline on Anyscale Cloud"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import ray
from compos3d_dp.pipelines.bronze_ingestion_distributed import (
    run_bronze_ingestion_distributed,
)
from compos3d_dp.pipelines.silver_transformation_distributed import (
    run_silver_transformation_distributed,
)
from compos3d_dp.pipelines.gold_aggregation_distributed import (
    run_gold_aggregation_distributed,
)
from compos3d_dp.compute.anyscale_cloud import init_ray_for_pipeline


def run_complete_pipeline(
    env: str = "dev", num_scenes: int = 10, use_anyscale: bool = True
):
    print(f"Running pipeline: env={env}, scenes={num_scenes}, anyscale={use_anyscale}")

    if use_anyscale:
        cluster_type = init_ray_for_pipeline(use_anyscale=True)
        print(f"Running on {cluster_type} cluster")
    else:
        ray.init(ignore_reinit_error=True)
        print("Running on local Ray cluster")

    print("\nStage 1: Bronze Ingestion")

    bronze_result = run_bronze_ingestion_distributed(
        env=env,
        num_scenes=num_scenes,
        ray_address="auto" if use_anyscale else None,
    )

    print(f"Bronze: {bronze_result['successful']} scenes ingested")

    print("\nStage 2: Silver Transformation")

    silver_result = run_silver_transformation_distributed(
        env=env,
        date_filter=None,
        ray_address="auto" if use_anyscale else None,
    )

    print(f"Silver: {silver_result['successful']} scenes transformed")

    print("\nStage 3: Gold Aggregation")

    gold_result = run_gold_aggregation_distributed(
        env=env,
        date_filter=None,
        ray_address="auto" if use_anyscale else None,
    )

    print(f"Gold: {gold_result['total_scenes']} scenes aggregated")

    print("\nPipeline complete:")
    print(f"  Bronze: {bronze_result['successful']} scenes")
    print(f"  Silver: {silver_result['successful']} scenes")
    print(f"  Gold: {gold_result['total_scenes']} scenes")
    print(f"  Dataset: {gold_result['dataset_id']}")
    print(f"  URI: {gold_result['gold_uri']}")

    return {
        "bronze": bronze_result,
        "silver": silver_result,
        "gold": gold_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Compos3D data pipeline on Anyscale"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="dev",
        choices=["dev", "staging", "prod"],
        help="Environment to run in",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=10,
        help="Number of scenes to ingest (Bronze layer)",
    )
    parser.add_argument(
        "--local", action="store_true", help="Run locally instead of on Anyscale cloud"
    )

    args = parser.parse_args()

    if not args.local and "ANYSCALE_API_KEY" not in os.environ:
        print("Error: ANYSCALE_API_KEY not set")
        print("Set with: export ANYSCALE_API_KEY=your_api_key")
        print("Or run locally with --local flag")
        sys.exit(1)

    results = run_complete_pipeline(
        env=args.env,
        num_scenes=args.num_scenes,
        use_anyscale=not args.local,
    )
