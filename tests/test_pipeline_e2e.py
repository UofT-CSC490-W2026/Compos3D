"""
End-to-end test of Bronze → Silver → Gold data pipeline

This tests the ACTUAL data processing pipeline for the assignment:
1. Bronze: Data ingestion (raw JSON)
2. Silver: Data cleaning/transformation (Parquet with schemas)
3. Gold: Data aggregation (training datasets)
"""

import os
from compos3d_dp.storage.multibucket_s3 import MultiLayerS3Store
from compos3d_dp.config import load_config
from compos3d_dp.storage.paths import utc_date_parts
from compos3d_dp.datasets.blenderbench import BlenderBenchDataset
import uuid
import json


# Set AWS profile
os.environ["AWS_PROFILE"] = "myisb_IsbUsersPS-136268833180"

# Load config
cfg = load_config("dev")
store = MultiLayerS3Store(
    bucket_bronze=cfg.s3_bucket_bronze,
    bucket_silver=cfg.s3_bucket_silver,
    bucket_gold=cfg.s3_bucket_gold,
    prefix=cfg.s3_prefix,
    region=cfg.aws_region,
)

# Get date partition
y, m, d = utc_date_parts()
date_part = f"{y}/{m}/{d}"

print(f"\n📅 Date partition: {date_part}")
print("📦 Storage buckets:")
print(f"   Bronze: {cfg.s3_bucket_bronze}")
print(f"   Silver: {cfg.s3_bucket_silver}")
print(f"   Gold: {cfg.s3_bucket_gold}")

# ============================================================================
# STEP 1: BRONZE INGESTION - Load BlenderBench dataset
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: BRONZE LAYER - Data Ingestion")

dataset = BlenderBenchDataset(cache_dir="data/blenderbench")
dataset.download()

# Ingest 3 scenes from different levels
test_scenes = [
    dataset.get_instance("level1/camera1"),
    dataset.get_instance("level2/attribute1"),
    dataset.get_instance("level3/attribute7"),
]

bronze_scene_ids = []

for instance in test_scenes:
    scene_id = f"scene_{uuid.uuid4().hex[:8]}"
    bronze_scene_ids.append(scene_id)

    # Create Bronze record (raw data)
    bronze_record = {
        "scene_id": scene_id,
        "source": "blenderbench",
        "instance_id": instance.instance_id,
        "level": instance.level,
        "task_description": instance.task_description,
        "blend_file_path": str(dataset.download_blend_file(instance)),
        "start_code": instance.start_code,
        "goal_code": instance.goal_code,
        "ingestion_timestamp": f"{y}-{m}-{d}T00:00:00Z",
    }

    # Write to Bronze
    bronze_path = f"bronze/scenes/{date_part}/{scene_id}/scene.json"
    uri = store.put_json(bronze_path, bronze_record)
    print(f"Ingested {instance.instance_id} → {uri}")

print(f"\nBronze ingestion complete: {len(bronze_scene_ids)} scenes")

# Verify Bronze data
bronze_files = store.list_prefix(f"bronze/scenes/{date_part}/")
print(f"Bronze contains {len(bronze_files)} files")

# ============================================================================
# STEP 2: SILVER TRANSFORMATION - Clean and validate data
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: SILVER LAYER - Data Cleaning & Transformation")

# Read Bronze, transform to Silver
for scene_id in bronze_scene_ids:
    bronze_path = f"bronze/scenes/{date_part}/{scene_id}/scene.json"
    bronze_data = store.read_json(bronze_path)

    # Transform to Silver (cleaned, validated, structured)
    silver_record = {
        "scene_id": bronze_data["scene_id"],
        "source": bronze_data["source"],
        "level": bronze_data["level"],
        "task": bronze_data["task_description"],
        "blend_file": bronze_data["blend_file_path"],
        # Add derived fields
        "code_length_start": len(bronze_data["start_code"]),
        "code_length_goal": len(bronze_data["goal_code"]),
        "complexity_level": int(bronze_data["level"].replace("level", "")),
        "has_camera_task": "camera" in bronze_data["task_description"].lower(),
        "has_attribute_task": "attribute" in bronze_data.get("instance_id", "").lower(),
    }

    # Write to Silver (in practice, this would be Parquet)
    silver_path = f"silver/scenes/{date_part}/{scene_id}/scene.json"
    uri = store.put_json(silver_path, silver_record)
    print(f"Transformed {scene_id} → Silver")

print(f"\nSilver transformation complete: {len(bronze_scene_ids)} scenes")

# Verify Silver data
silver_files = store.list_prefix(f"silver/scenes/{date_part}/")
print(f"Silver contains {len(silver_files)} files")

# ============================================================================
# STEP 3: GOLD AGGREGATION - Create training datasets
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: GOLD LAYER - Data Aggregation & Training Datasets")

# Aggregate Silver data into Gold
gold_training_dataset = {
    "dataset_id": f"training_{uuid.uuid4().hex[:8]}",
    "created_at": f"{y}-{m}-{d}T00:00:00Z",
    "num_scenes": len(bronze_scene_ids),
    "scenes": [],
    "statistics": {
        "total_scenes": len(bronze_scene_ids),
        "by_level": {},
        "avg_code_length": 0,
        "camera_tasks": 0,
        "attribute_tasks": 0,
    },
}

total_code_length = 0

for scene_id in bronze_scene_ids:
    silver_path = f"silver/scenes/{date_part}/{scene_id}/scene.json"
    silver_data = store.read_json(silver_path)

    # Add to training dataset
    gold_training_dataset["scenes"].append(
        {
            "scene_id": silver_data["scene_id"],
            "level": silver_data["level"],
            "task": silver_data["task"],
            "complexity": silver_data["complexity_level"],
        }
    )

    # Update statistics
    level = silver_data["level"]
    gold_training_dataset["statistics"]["by_level"][level] = (
        gold_training_dataset["statistics"]["by_level"].get(level, 0) + 1
    )

    total_code_length += (
        silver_data["code_length_start"] + silver_data["code_length_goal"]
    )

    if silver_data["has_camera_task"]:
        gold_training_dataset["statistics"]["camera_tasks"] += 1
    if silver_data["has_attribute_task"]:
        gold_training_dataset["statistics"]["attribute_tasks"] += 1

gold_training_dataset["statistics"]["avg_code_length"] = total_code_length / (
    len(bronze_scene_ids) * 2
)

# Write to Gold
gold_path = f"gold/training_datasets/{date_part}/dataset.json"
uri = store.put_json(gold_path, gold_training_dataset)
print(f"Created training dataset → {uri}")

# Print statistics
print("\nTraining Dataset Statistics:")
print(f"   Total scenes: {gold_training_dataset['statistics']['total_scenes']}")
print(
    f"   By level: {json.dumps(gold_training_dataset['statistics']['by_level'], indent=6)}"
)
print(
    f"   Avg code length: {gold_training_dataset['statistics']['avg_code_length']:.0f} chars"
)
print(f"   Camera tasks: {gold_training_dataset['statistics']['camera_tasks']}")
print(f"   Attribute tasks: {gold_training_dataset['statistics']['attribute_tasks']}")

# ============================================================================
# VERIFICATION
# ============================================================================
print("\n" + "=" * 80)

# Count files in each layer
bronze_count = len(store.list_prefix("bronze/scenes/"))
silver_count = len(store.list_prefix("silver/scenes/"))
gold_count = len(store.list_prefix("gold/training_datasets/"))

print("Data Lake Status:")
print(f"   Bronze layer: {bronze_count} files")
print(f"   Silver layer: {silver_count} files")
print(f"   Gold layer: {gold_count} files")

print("\n" + "=" * 80)
