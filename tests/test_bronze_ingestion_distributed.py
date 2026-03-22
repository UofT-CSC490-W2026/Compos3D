"""Distributed Bronze ingestion unit tests (ported from compos3d_dp).

In the current architecture Bronze ingestion happens as part of the training
pipeline: ``train_vertical_slice`` with a store writes raw ScenePrograms and
bank snapshots to ``bronze/training/<run_id>/``.  These tests verify that the
bronze write path works correctly using a local store.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from compos3d.storage.local import LocalStore
from compos3d.storage.paths import training_bronze_prefix, utc_date_parts


@pytest.mark.bronze
@pytest.mark.unit
def test_bronze_ingestion_single_scene(tmp_store: LocalStore) -> None:
    """A single raw scene program can be ingested into bronze."""
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    pfx = training_bronze_prefix(run_id)

    scene = {
        "room_type": "dining_room",
        "assets": [{"asset_type": "dining_table", "count": 1}],
        "constraints": [],
        "style": "modern",
    }
    uri = tmp_store.put_json(f"{pfx}/programs/dr_001_epoch_1.json", scene)
    assert uri

    files = tmp_store.list_prefix(pfx)
    assert len(files) == 1
    assert files[0].endswith("dr_001_epoch_1.json")


@pytest.mark.bronze
@pytest.mark.unit
def test_bronze_ingestion_multiple_scenes(tmp_store: LocalStore) -> None:
    """Multiple scenes can be ingested and listed under one run."""
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    pfx = training_bronze_prefix(run_id)

    for i in range(3):
        tmp_store.put_json(
            f"{pfx}/programs/scene_{i:03d}.json",
            {"room_type": "dining_room", "index": i},
        )

    files = tmp_store.list_prefix(pfx)
    assert len(files) == 3


@pytest.mark.bronze
@pytest.mark.unit
def test_bronze_ingestion_validates_env(local_app_config) -> None:
    """get_store returns a valid store for local environment."""
    from compos3d.storage import get_store
    store = get_store(local_app_config)
    assert store is not None

    uri = store.put_json("bronze/test/check.json", {"ok": True})
    assert uri


@pytest.mark.bronze
@pytest.mark.unit
def test_bronze_ingestion_run_manifest(tmp_store: LocalStore) -> None:
    """A run manifest can be stored in bronze for provenance."""
    from compos3d.schemas.manifest import create_manifest, finalize_manifest
    run_id = f"run_{uuid.uuid4().hex[:8]}"

    manifest = create_manifest(
        run_id=run_id,
        run_type="training",
        config_snapshot={"mock": True},
        input_paths=["examples/dummy_fast.json"],
    )
    manifest = finalize_manifest(manifest, status="success", output_uris=["bronze/test/x.json"])

    uri = tmp_store.put_json(
        f"bronze/training/{run_id}/run_manifest.json",
        manifest.model_dump(mode="json"),
    )
    assert uri

    loaded = tmp_store.read_json(f"bronze/training/{run_id}/run_manifest.json")
    assert loaded["run_id"] == run_id
    assert loaded["status"] == "success"
    assert loaded["run_type"] == "training"
