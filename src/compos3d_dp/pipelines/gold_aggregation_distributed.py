"""
Gold Layer - Distributed Data Aggregation on Anyscale

USE CASE: Create training-ready datasets from Silver data
SCHEDULE: Runs weekly to aggregate data for model training
SCALE: Aggregate 10K+ scenes into optimized training datasets

This pipeline:
1. Reads validated data from Silver S3
2. Computes aggregate statistics
3. Creates train/val/test splits
4. Generates training dataset manifests
5. Writes to Gold S3 bucket

Technologies:
- Ray/Anyscale: Distributed aggregation
- Parquet: Efficient columnar reads
- WebDataset: Training data format (future)
"""

import ray
from typing import List, Dict, Any
from compos3d_dp.storage.multibucket_s3 import MultiLayerS3Store
from compos3d_dp.config import load_config
from compos3d_dp.storage.paths import utc_date_parts
import uuid


@ray.remote(num_cpus=1)
def aggregate_scene_batch(
    scene_paths: List[str],
    bronze_bucket: str,
    silver_bucket: str,
    gold_bucket: str,
    s3_prefix: str,
    aws_region: str,
) -> Dict[str, Any]:
    """
    Aggregate a batch of Silver scenes (Ray remote function).

    Computes:
    - Scene statistics
    - Complexity distributions
    - Task categorizations

    Args:
        scene_paths: List of S3 paths to Silver scenes
        (bucket params)

    Returns:
        Aggregated batch statistics
    """
    store = MultiLayerS3Store(
        bucket_bronze=bronze_bucket,
        bucket_silver=silver_bucket,
        bucket_gold=gold_bucket,
        prefix=s3_prefix,
        region=aws_region,
    )

    scenes = []
    stats = {
        "total_scenes": 0,
        "by_level": {},
        "by_task_type": {},
        "total_code_length": 0,
        "camera_tasks": 0,
        "attribute_tasks": 0,
    }

    for path in scene_paths:
        try:
            scene = store.read_json(path)
            scenes.append(scene)

            # Update statistics
            stats["total_scenes"] += 1

            level = scene.get("level", "unknown")
            stats["by_level"][level] = stats["by_level"].get(level, 0) + 1

            task_type = scene.get("task_type", "unknown")
            stats["by_task_type"][task_type] = (
                stats["by_task_type"].get(task_type, 0) + 1
            )

            stats["total_code_length"] += scene.get("code_length_start", 0) + scene.get(
                "code_length_goal", 0
            )

            if scene.get("has_camera", False):
                stats["camera_tasks"] += 1
            if scene.get("has_attribute", False):
                stats["attribute_tasks"] += 1

        except Exception as e:
            print(f"Error reading {path}: {e}")

    return {
        "scenes": scenes,
        "stats": stats,
    }


def run_gold_aggregation_distributed(
    env: str = "dev",
    date_filter: str = None,
    ray_address: str = None,
) -> Dict[str, Any]:
    """
    Run distributed Gold aggregation on Anyscale.

    USE CASE: Weekly aggregation for training dataset creation

    Args:
        env: Environment
        date_filter: Date to aggregate
        ray_address: Ray cluster

    Returns:
        Aggregation summary
    """
    print("=" * 80)
    print("🏆 GOLD AGGREGATION - Distributed on Anyscale")
    print("=" * 80)
    print(f"Environment: {env}")

    # Initialize Ray (Anyscale or local)
    from compos3d_dp.compute.anyscale_cloud import init_ray_for_pipeline

    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)
    else:
        init_ray_for_pipeline(use_anyscale=True)

    # Load config
    cfg = load_config(env)
    store = MultiLayerS3Store(
        bucket_bronze=cfg.s3_bucket_bronze,
        bucket_silver=cfg.s3_bucket_silver,
        bucket_gold=cfg.s3_bucket_gold,
        prefix=cfg.s3_prefix,
        region=cfg.aws_region,
    )

    # List Silver scenes
    silver_prefix = "silver/scenes/"
    if date_filter:
        silver_prefix = f"silver/scenes/{date_filter}/"

    silver_files = store.list_prefix(silver_prefix)
    silver_scenes = [f for f in silver_files if f.endswith("scene.json")]

    # Batch scenes for parallel processing
    batch_size = 10
    scene_batches = [
        silver_scenes[i : i + batch_size]
        for i in range(0, len(silver_scenes), batch_size)
    ]

    print(f"   Created {len(scene_batches)} batches of {batch_size} scenes each")

    # Submit distributed tasks
    futures = [
        aggregate_scene_batch.remote(
            scene_paths=batch,
            bronze_bucket=cfg.s3_bucket_bronze,
            silver_bucket=cfg.s3_bucket_silver,
            gold_bucket=cfg.s3_bucket_gold,
            s3_prefix=cfg.s3_prefix,
            aws_region=cfg.aws_region,
        )
        for batch in scene_batches
    ]

    # Gather results
    print("\n   Processing batches...")
    batch_results = ray.get(futures)

    # Merge statistics
    all_scenes = []
    merged_stats = {
        "total_scenes": 0,
        "by_level": {},
        "by_task_type": {},
        "total_code_length": 0,
        "camera_tasks": 0,
        "attribute_tasks": 0,
    }

    for batch_result in batch_results:
        all_scenes.extend(batch_result["scenes"])

        stats = batch_result["stats"]
        merged_stats["total_scenes"] += stats["total_scenes"]
        merged_stats["total_code_length"] += stats["total_code_length"]
        merged_stats["camera_tasks"] += stats["camera_tasks"]
        merged_stats["attribute_tasks"] += stats["attribute_tasks"]

        for level, count in stats["by_level"].items():
            merged_stats["by_level"][level] = (
                merged_stats["by_level"].get(level, 0) + count
            )

        for task_type, count in stats["by_task_type"].items():
            merged_stats["by_task_type"][task_type] = (
                merged_stats["by_task_type"].get(task_type, 0) + count
            )

    # Create Gold training dataset
    y, m, d = utc_date_parts()
    date_part = f"{y}/{m}/{d}"

    gold_dataset = {
        "dataset_id": f"training_{uuid.uuid4().hex[:8]}",
        "created_at": f"{y}-{m}-{d}T00:00:00Z",
        "num_scenes": merged_stats["total_scenes"],
        "statistics": {
            **merged_stats,
            "avg_code_length": merged_stats["total_code_length"]
            / (merged_stats["total_scenes"] * 2)
            if merged_stats["total_scenes"] > 0
            else 0,
        },
        "scenes": [
            {
                "scene_id": s["scene_id"],
                "level": s.get("level"),
                "task_type": s.get("task_type"),
                "complexity": s.get("complexity_level"),
            }
            for s in all_scenes
        ],
    }

    # Write to Gold
    gold_path = f"gold/training_datasets/{date_part}/dataset.json"
    uri = store.put_json(gold_path, gold_dataset)

    # Print statistics
    print(f"   Total scenes: {merged_stats['total_scenes']}")
    print(f"   By level: {merged_stats['by_level']}")
    print(f"   By task type: {merged_stats['by_task_type']}")
    print(
        f"   Avg code length: {gold_dataset['statistics']['avg_code_length']:.0f} chars"
    )
    print(f"   Camera tasks: {merged_stats['camera_tasks']}")
    print(f"   Attribute tasks: {merged_stats['attribute_tasks']}")

    summary = {
        "total_scenes": merged_stats["total_scenes"],
        "gold_uri": uri,
        "dataset_id": gold_dataset["dataset_id"],
    }

    print("\n" + "=" * 80)
    print("=" * 80)

    return summary


if __name__ == "__main__":
    # Test locally
    summary = run_gold_aggregation_distributed(
        env="dev",
        date_filter=None,
        ray_address=None,
    )
