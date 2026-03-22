"""Bronze layer ingestion tests.

Verifies that raw scene data (ScenePrograms, training examples) can be written
to and read back from the bronze layer of the local data lake store.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from compos3d.storage.local import LocalStore
from compos3d.storage.paths import utc_date_parts


@pytest.mark.bronze
@pytest.mark.unit
def test_bronze_write_and_read_scene(tmp_store: LocalStore) -> None:
    """Write a raw scene record to bronze and read it back intact."""
    y, m, d = utc_date_parts()
    date_part = f"{y}/{m}/{d}"
    scene_id = f"scene_{uuid.uuid4().hex[:8]}"

    record = {
        "scene_id": scene_id,
        "room_type": "dining_room",
        "prompt": "a cozy dining room with a large table",
        "generator": "mock",
        "assets": [{"asset_type": "dining_table", "count": 1}],
    }

    path = f"bronze/scenes/{date_part}/{scene_id}/scene.json"
    uri = tmp_store.put_json(path, record)
    assert uri  # non-empty path/URI returned

    read_back = tmp_store.read_json(path)
    assert read_back["scene_id"] == scene_id
    assert read_back["room_type"] == "dining_room"


@pytest.mark.bronze
@pytest.mark.unit
def test_bronze_list_prefix(tmp_store: LocalStore) -> None:
    """list_prefix returns only paths under the requested prefix."""
    y, m, d = utc_date_parts()
    date_part = f"{y}/{m}/{d}"

    ids = [uuid.uuid4().hex[:8] for _ in range(3)]
    for sid in ids:
        tmp_store.put_json(f"bronze/scenes/{date_part}/{sid}/scene.json", {"scene_id": sid})

    # Also write to silver to confirm prefix isolation.
    tmp_store.put_json(f"silver/scenes/{date_part}/other.json", {"layer": "silver"})

    bronze_files = tmp_store.list_prefix(f"bronze/scenes/{date_part}")
    assert len(bronze_files) == 3
    assert all("bronze" in f for f in bronze_files)


@pytest.mark.bronze
@pytest.mark.unit
def test_bronze_bytes_round_trip(tmp_store: LocalStore) -> None:
    """put_bytes / read via exists works correctly."""
    data = b"raw binary content"
    tmp_store.put_bytes("bronze/renders/frame_0000.png", data, content_type="image/png")
    assert tmp_store.exists("bronze/renders/frame_0000.png")


@pytest.mark.bronze
@pytest.mark.unit
def test_bronze_ingestion_validates_dataset(dummy_dataset_path: Path) -> None:
    """The dummy dataset can be loaded and matches the expected schema."""
    from compos3d.data.dataset import load_training_dataset

    dataset = load_training_dataset(dummy_dataset_path)
    assert dataset.dataset_id == "dummy_fast"
    assert len(dataset.examples) >= 2
    for ex in dataset.examples:
        assert ex.room_type in ("dining_room", "living_room", "bedroom")
        assert ex.prompt
        assert ex.required_assets
