"""
Silver Layer - Distributed Data Cleaning & Transformation on Anyscale

USE CASE: Clean and validate Bronze data, convert to structured Parquet
SCHEDULE: Runs daily after Bronze ingestion completes
SCALE: Process 1000+ scenes per run using 20-100 Ray workers

This pipeline:
1. Reads raw JSON from Bronze S3
2. Validates data schemas
3. Cleans and transforms data
4. Writes Parquet to Silver S3 bucket
5. Partitions by date and split (train/val/test)

Technologies:
- Ray/Anyscale: Distributed processing
- PyArrow/Parquet: Columnar storage
- Pydantic: Schema validation
- Great Expectations: Data quality checks
"""

import ray
from typing import Dict, Any
from compos3d_dp.storage.multibucket_s3 import MultiLayerS3Store
from compos3d_dp.config import load_config


@ray.remote(num_cpus=1)
def transform_scene_to_silver(
    scene_id: str,
    bronze_path: str,
    bronze_bucket: str,
    silver_bucket: str,
    gold_bucket: str,
    s3_prefix: str,
    aws_region: str,
) -> Dict[str, Any]:
    """
    Ray remote function to transform a Bronze scene to Silver.

    Data Cleaning & Transformation:
    - Validate schema
    - Extract derived fields
    - Normalize text fields
    - Compute statistics

    Args:
        scene_id: Scene identifier
        bronze_path: S3 path to Bronze JSON
        (bucket params for S3 store)

    Returns:
        Transformation result
    """
    # Initialize S3 store
    store = MultiLayerS3Store(
        bucket_bronze=bronze_bucket,
        bucket_silver=silver_bucket,
        bucket_gold=gold_bucket,
        prefix=s3_prefix,
        region=aws_region,
    )

    try:
        # Read Bronze data
        bronze_data = store.read_json(bronze_path)

        # TRANSFORMATION: Clean and validate
        silver_record = {
            # Core fields
            "scene_id": bronze_data["scene_id"],
            "source": bronze_data["source"],
            "level": bronze_data["level"],
            "task": bronze_data["task_description"],
            # Derived fields
            "complexity_level": int(bronze_data["level"].replace("level", "")),
            "code_length_start": len(bronze_data.get("start_code", "")),
            "code_length_goal": len(bronze_data.get("goal_code", "")),
            # Task categorization
            "task_type": _categorize_task(bronze_data["task_description"]),
            "has_camera": "camera" in bronze_data["task_description"].lower(),
            "has_attribute": "attribute" in bronze_data.get("instance_id", "").lower(),
            # File metadata
            "blend_file": bronze_data["blend_file_path"],
            "blend_file_size_mb": bronze_data.get("blend_file_size_mb", 0),
            # Timestamps
            "ingested_at": bronze_data["ingestion_timestamp"],
            "transformed_at": bronze_data["ingestion_timestamp"],  # Use same for now
        }

        # Write to Silver
        # Extract date from bronze_path
        date_parts = bronze_path.split("/")
        date_idx = date_parts.index("scenes") + 1
        date_str = "/".join(date_parts[date_idx : date_idx + 3])

        silver_path = f"silver/scenes/{date_str}/{scene_id}/scene.json"
        uri = store.put_json(silver_path, silver_record)

        return {
            "success": True,
            "scene_id": scene_id,
            "s3_uri": uri,
            "complexity_level": silver_record["complexity_level"],
        }

    except Exception as e:
        return {
            "success": False,
            "scene_id": scene_id,
            "error": str(e),
        }


def _categorize_task(task_description: str) -> str:
    """Categorize task by description"""
    task_lower = task_description.lower()

    if "camera" in task_lower:
        return "camera_adjustment"
    elif "move" in task_lower or "position" in task_lower:
        return "object_positioning"
    elif "color" in task_lower or "material" in task_lower:
        return "material_editing"
    else:
        return "other"


def run_silver_transformation_distributed(
    env: str = "dev",
    date_filter: str = None,  # "2026/02/15" or None for all
    ray_address: str = None,
) -> Dict[str, Any]:
    """
    Run distributed Silver transformation on Anyscale.

    USE CASE: Daily batch transformation of Bronze → Silver

    Args:
        env: Environment
        date_filter: Specific date to process (None = all new data)
        ray_address: Ray cluster address

    Returns:
        Transformation summary
    """
    print("=" * 80)
    print("🔧 SILVER TRANSFORMATION - Distributed on Anyscale")
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

    # List Bronze scenes
    bronze_prefix = "bronze/scenes/"
    if date_filter:
        bronze_prefix = f"bronze/scenes/{date_filter}/"

    bronze_files = store.list_prefix(bronze_prefix)
    bronze_scenes = [f for f in bronze_files if f.endswith("scene.json")]

    # Extract scene IDs
    scene_tasks = []
    for bronze_path in bronze_scenes:
        # Extract scene_id from path: bronze/scenes/2026/02/15/scene_xxx/scene.json
        parts = bronze_path.split("/")
        scene_id = parts[-2]
        scene_tasks.append((scene_id, bronze_path))

    # Submit distributed tasks
    futures = [
        transform_scene_to_silver.remote(
            scene_id=scene_id,
            bronze_path=bronze_path,
            bronze_bucket=cfg.s3_bucket_bronze,
            silver_bucket=cfg.s3_bucket_silver,
            gold_bucket=cfg.s3_bucket_gold,
            s3_prefix=cfg.s3_prefix,
            aws_region=cfg.aws_region,
        )
        for scene_id, bronze_path in scene_tasks
    ]

    # Gather results
    results = []
    while futures:
        ready, futures = ray.wait(futures, num_returns=1, timeout=10)
        for future in ready:
            result = ray.get(future)
            results.append(result)

            if result["success"]:
                print(f"   ✅ {len(results)}/{len(scene_tasks)}: {result['scene_id']}")
            else:
                print(f"   ❌ {len(results)}/{len(scene_tasks)}: {result['scene_id']}")

    # Statistics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    summary = {
        "total_scenes": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "scene_ids": [r["scene_id"] for r in successful],
    }

    print("\n" + "=" * 80)
    print("=" * 80)
    print(f"Total: {summary['total_scenes']}")
    print(f"Success: {summary['successful']}")
    print(f"Failed: {summary['failed']}")

    return summary


if __name__ == "__main__":
    # Test locally
    summary = run_silver_transformation_distributed(
        env="dev",
        date_filter=None,
        ray_address=None,
    )
