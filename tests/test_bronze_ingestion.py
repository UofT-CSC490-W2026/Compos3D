"""Test BlenderBench dataset and Bronze ingestion"""

from compos3d_dp.datasets.blenderbench import BlenderBenchDataset
from compos3d_dp.storage.multibucket_s3 import MultiLayerS3Store
from compos3d_dp.config import load_config
from compos3d_dp.storage.paths import utc_date_parts
import uuid


# Load dataset
print("📥 Loading BlenderBench...")
dataset = BlenderBenchDataset(cache_dir="data/blenderbench")
dataset.download()
print(f"Loaded {len(dataset.instances)} instances")

# Get one instance
instance = dataset.get_instance("level1/camera1")
print(f"\nGot instance: {instance.instance_id}")
print(f"   Task: {instance.task_description}")
print(f"   Level: {instance.level}")

# Download blend file
blend_file = dataset.download_blend_file(instance)
print(f"\nDownloaded blend file: {blend_file}")
print(f"   Exists: {blend_file.exists()}")
print(f"   Size: {blend_file.stat().st_size / 1024 / 1024:.2f} MB")

# Test Bronze ingestion
print("\n📦 Testing Bronze ingestion...")
cfg = load_config("dev")
store = MultiLayerS3Store(
    bucket_bronze=cfg.s3_bucket_bronze,
    bucket_silver=cfg.s3_bucket_silver,
    bucket_gold=cfg.s3_bucket_gold,
    prefix=cfg.s3_prefix,
    region=cfg.aws_region,
)

# Create Bronze record
y, m, d = utc_date_parts()
date_part = f"{y}/{m}/{d}"
scene_id = f"test_scene_{uuid.uuid4().hex[:8]}"

bronze_record = {
    "scene_id": scene_id,
    "prompt": instance.task_description,
    "generator": "blenderbench",
    "level": instance.level,
    "instance_id": instance.instance_id,
    "blend_file_path": str(blend_file),
    "start_code": instance.start_code[:100] + "...",  # Truncate for display
    "goal_code": instance.goal_code[:100] + "...",
}

# Write to Bronze
bronze_path = f"bronze/scenes/{date_part}/{scene_id}/scene.json"
uri = store.put_json(bronze_path, bronze_record)
print(f"\nWrote to Bronze: {uri}")

# Read back
read_record = store.read_json(bronze_path)
assert read_record["scene_id"] == scene_id
print(f"Read from Bronze: {read_record['scene_id']}")

# List Bronze scenes
scenes = store.list_prefix(f"bronze/scenes/{date_part}/")
print(f"\nListed Bronze scenes: {len(scenes)} found")
for s in scenes[:3]:
    print(f"   - {s}")

print("\n" + "=" * 60)
