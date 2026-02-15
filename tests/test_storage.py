"""Test S3 multi-bucket storage"""

from compos3d_dp.storage.multibucket_s3 import MultiLayerS3Store
from compos3d_dp.config import load_config


# Load dev config
cfg = load_config("dev")
print("Config loaded")
print(f"   Bronze: {cfg.s3_bucket_bronze}")
print(f"   Silver: {cfg.s3_bucket_silver}")
print(f"   Gold: {cfg.s3_bucket_gold}")

# Initialize store
store = MultiLayerS3Store(
    bucket_bronze=cfg.s3_bucket_bronze,
    bucket_silver=cfg.s3_bucket_silver,
    bucket_gold=cfg.s3_bucket_gold,
    prefix=cfg.s3_prefix,
    region=cfg.aws_region,
)
print("\nStore initialized")

# Test 1: Write to Bronze
test_data = {
    "test_id": "test_001",
    "timestamp": "2026-02-14T12:00:00Z",
    "data": {"x": 1, "y": 2},
}

bronze_path = "bronze/test/test_001.json"
uri = store.put_json(bronze_path, test_data)
print(f"\nWrote to Bronze: {uri}")

# Test 2: Read from Bronze
read_data = store.read_json(bronze_path)
assert read_data == test_data, "Data mismatch!"
print(f"Read from Bronze: {read_data['test_id']}")

# Test 3: List Bronze prefix
files = store.list_prefix("bronze/test/")
print(f"\nListed Bronze files: {len(files)} found")
for f in files[:3]:
    print(f"   - {f}")

# Test 4: Write to Silver
silver_path = "silver/test/test_001.json"
uri = store.put_json(silver_path, test_data)
print(f"\nWrote to Silver: {uri}")

# Test 5: Write to Gold
gold_path = "gold/test/test_001.json"
uri = store.put_json(gold_path, test_data)
print(f"\nWrote to Gold: {uri}")

print("\n" + "=" * 60)
