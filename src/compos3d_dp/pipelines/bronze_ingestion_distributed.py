"""Bronze Layer - Distributed Data Ingestion on Anyscale"""

import ray
from typing import Dict, Any
from compos3d_dp.datasets.blenderbench import BlenderBenchDataset
from compos3d_dp.storage.multibucket_s3 import MultiLayerS3Store
from compos3d_dp.storage.paths import utc_date_parts
from compos3d_dp.config import load_config
import uuid


@ray.remote(num_cpus=1)
def ingest_scene_to_bronze(
    instance_id: str,
    bronze_bucket: str,
    silver_bucket: str,
    gold_bucket: str,
    s3_prefix: str,
    aws_region: str,
) -> Dict[str, Any]:
    dataset = BlenderBenchDataset(cache_dir="/tmp/blenderbench_cache")
    dataset.download()

    instance = dataset.get_instance(instance_id)
    if instance is None:
        return {
            "success": False,
            "instance_id": instance_id,
            "error": "Instance not found",
        }

    try:
        blend_file = dataset.download_blend_file(instance)
    except Exception as e:
        return {"success": False, "instance_id": instance_id, "error": str(e)}

    store = MultiLayerS3Store(
        bucket_bronze=bronze_bucket,
        bucket_silver=silver_bucket,
        bucket_gold=gold_bucket,
        prefix=s3_prefix,
        region=aws_region,
    )

    y, m, d = utc_date_parts()
    date_part = f"{y}/{m}/{d}"
    scene_id = f"scene_{uuid.uuid4().hex[:8]}"

    bronze_record = {
        "scene_id": scene_id,
        "source": "blenderbench",
        "instance_id": instance.instance_id,
        "level": instance.level,
        "task_description": instance.task_description,
        "blend_file_path": str(blend_file),
        "blend_file_size_mb": blend_file.stat().st_size / 1024 / 1024,
        "start_code": instance.start_code,
        "goal_code": instance.goal_code,
        "ingestion_timestamp": f"{y}-{m}-{d}T00:00:00Z",
    }

    bronze_path = f"bronze/scenes/{date_part}/{scene_id}/scene.json"
    uri = store.put_json(bronze_path, bronze_record)

    return {
        "success": True,
        "scene_id": scene_id,
        "instance_id": instance_id,
        "s3_uri": uri,
        "blend_file_size_mb": bronze_record["blend_file_size_mb"],
    }


def run_bronze_ingestion_distributed(
    env: str = "dev",
    num_scenes: int = 100,
    ray_address: str = None,
) -> Dict[str, Any]:
    from compos3d_dp.compute.anyscale_cloud import init_ray_for_pipeline

    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)
    else:
        init_ray_for_pipeline(use_anyscale=True)

    cfg = load_config(env)
    dataset = BlenderBenchDataset(cache_dir="data/blenderbench")
    dataset.download()

    all_instances = dataset.get_all_instances()[:num_scenes]
    instance_ids = [inst.instance_id for inst in all_instances]

    futures = [
        ingest_scene_to_bronze.remote(
            instance_id=inst_id,
            bronze_bucket=cfg.s3_bucket_bronze,
            silver_bucket=cfg.s3_bucket_silver,
            gold_bucket=cfg.s3_bucket_gold,
            s3_prefix=cfg.s3_prefix,
            aws_region=cfg.aws_region,
        )
        for inst_id in instance_ids
    ]

    results = []
    while futures:
        ready, futures = ray.wait(futures, num_returns=1, timeout=10)
        for future in ready:
            result = ray.get(future)
            results.append(result)

            if result["success"]:
                print(
                    f"   ✅ {len(results)}/{len(instance_ids)}: {result['instance_id']}"
                )
            else:
                print(
                    f"   ❌ {len(results)}/{len(instance_ids)}: {result['instance_id']} - {result['error']}"
                )

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    total_size_mb = sum(r.get("blend_file_size_mb", 0) for r in successful)

    summary = {
        "total_scenes": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "total_size_mb": total_size_mb,
        "scene_ids": [r["scene_id"] for r in successful],
    }

    print("\n" + "=" * 80)
    print("=" * 80)
    print(f"Total: {summary['total_scenes']}")
    print(f"Success: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total data: {summary['total_size_mb']:.1f} MB")

    return summary


if __name__ == "__main__":
    # Test with 3 scenes locally
    summary = run_bronze_ingestion_distributed(
        env="dev",
        num_scenes=3,
        ray_address=None,  # Local
    )
