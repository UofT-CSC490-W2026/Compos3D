"""Tests for storage abstraction and AppConfig behavior.

Edge cases covered:
- missing required S3 bucket configuration for `storage_backend="s3"`
- prefix listing behavior and path-shape expectations
- layer separation invariants across bronze/silver/gold

Primarily happy-path checks are also included for local read/write and defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compos3d.app_config import AppConfig, load_app_config
from compos3d.storage import AnyStore, get_store
from compos3d.storage.local import LocalStore
from compos3d.storage.paths import (
    inference_bronze_prefix,
    inference_gold_prefix,
    inference_silver_prefix,
    training_bronze_prefix,
    training_gold_prefix,
    training_silver_prefix,
    utc_date_parts,
)


class TestAppConfig:
    def test_default_is_local(self) -> None:
        cfg = AppConfig()
        assert cfg.env == "local"
        assert cfg.storage_backend == "local"

    def test_local_env_yaml(self) -> None:
        cfg = load_app_config("local")
        assert cfg.env == "local"
        assert cfg.storage_backend == "local"

    def test_dev_env_yaml(self) -> None:
        cfg = load_app_config("dev")
        assert cfg.env == "dev"
        assert cfg.storage_backend == "s3"
        assert cfg.s3_bucket_bronze == "compos3d-dev-bronze"
        assert cfg.s3_bucket_silver == "compos3d-dev-silver"
        assert cfg.s3_bucket_gold == "compos3d-dev-gold"

    def test_staging_env_yaml(self) -> None:
        cfg = load_app_config("staging")
        assert cfg.s3_bucket_bronze == "compos3d-staging-bronze"

    def test_prod_env_yaml(self) -> None:
        cfg = load_app_config("prod")
        assert cfg.s3_bucket_bronze == "compos3d-prod-bronze"
        assert cfg.ec2_spot is False  # prod uses on-demand

    def test_get_store_local(self, tmp_path: Path) -> None:
        cfg = AppConfig(
            env="local", storage_backend="local", local_lake_root=str(tmp_path)
        )
        store = get_store(cfg)
        assert isinstance(store, LocalStore)

    def test_get_store_s3_missing_buckets_raises(self) -> None:
        cfg = AppConfig(env="dev", storage_backend="s3")
        with pytest.raises(ValueError, match="s3_bucket_bronze"):
            get_store(cfg)


class TestLocalStore:
    def test_put_and_read_json(self, tmp_path: Path) -> None:
        store = LocalStore(root=tmp_path)
        obj = {"key": "value", "num": 42}
        uri = store.put_json("bronze/test/data.json", obj)
        assert Path(uri).exists()
        assert store.read_json("bronze/test/data.json") == obj

    def test_put_and_read_bytes(self, tmp_path: Path) -> None:
        store = LocalStore(root=tmp_path)
        data = b"hello bytes"
        uri = store.put_bytes("silver/blobs/file.bin", data)
        assert Path(uri).read_bytes() == data

    def test_list_prefix(self, tmp_path: Path) -> None:
        store = LocalStore(root=tmp_path)
        store.put_json("bronze/run1/a.json", {"a": 1})
        store.put_json("bronze/run1/b.json", {"b": 2})
        store.put_json("silver/run1/c.json", {"c": 3})

        bronze_files = store.list_prefix("bronze/run1")
        assert len(bronze_files) == 2
        assert all("bronze/run1" in f for f in bronze_files)

    def test_exists(self, tmp_path: Path) -> None:
        store = LocalStore(root=tmp_path)
        assert not store.exists("bronze/x.json")
        store.put_json("bronze/x.json", {})
        assert store.exists("bronze/x.json")

    def test_bronze_silver_gold_dirs_are_separate(self, tmp_path: Path) -> None:
        store = LocalStore(root=tmp_path)
        store.put_json("bronze/run/data.json", {"layer": "bronze"})
        store.put_json("silver/run/data.json", {"layer": "silver"})
        store.put_json("gold/run/data.json", {"layer": "gold"})

        assert store.read_json("bronze/run/data.json")["layer"] == "bronze"
        assert store.read_json("silver/run/data.json")["layer"] == "silver"
        assert store.read_json("gold/run/data.json")["layer"] == "gold"


class TestLakePaths:
    def test_training_paths(self) -> None:
        run_id = "my_experiment_20250101"
        assert training_bronze_prefix(run_id) == f"bronze/training/{run_id}"
        assert training_silver_prefix(run_id) == f"silver/training/{run_id}"
        assert (
            training_gold_prefix("my_experiment")
            == "gold/hypothesis_banks/my_experiment"
        )

    def test_inference_paths(self) -> None:
        run_id = "inference_20250101"
        assert inference_bronze_prefix(run_id) == f"bronze/inference/{run_id}"
        assert inference_silver_prefix(run_id) == f"silver/inference/{run_id}"
        assert inference_gold_prefix(run_id) == f"gold/inference/{run_id}"

    def test_utc_date_parts_format(self) -> None:
        year, month, day = utc_date_parts()
        assert len(year) == 4
        assert len(month) == 2
        assert len(day) == 2
