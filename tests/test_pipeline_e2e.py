"""End-to-end Bronze → Silver → Gold data pipeline test (ported from compos3d_dp).

Verifies the full data lake flow:
1. Bronze: Raw scene programs are ingested during training.
2. Silver: Validated hypothesis banks and metrics are written.
3. Gold:  Final hypothesis bank (latest.json) is published.

All LLM calls use the mock provider; storage uses a LocalStore backed by a
pytest tmp_path directory, so no AWS credentials are required.

Edge cases covered:
- end-to-end resilience when some layer-specific artifacts may be absent
- data-lake status checks across bronze/silver/gold boundaries
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compos3d.storage.local import LocalStore
from compos3d.storage.paths import (
    training_bronze_prefix,
    training_gold_prefix,
    training_silver_prefix,
    utc_date_parts,
)
from compos3d.hypothesis.engine import train_vertical_slice


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lake(tmp_path) -> LocalStore:
    return LocalStore(root=tmp_path / "_lake")


@pytest.fixture
def pipeline_result(dummy_dataset_path, tmp_path, lake):
    """Run one training pass with lake mirroring and return (result, lake)."""
    result = train_vertical_slice(
        dataset_path=dummy_dataset_path,
        output_dir=tmp_path / "artifacts",
        experiment_name="e2e_test",
        llm_provider="mock",
        num_init_examples_per_room=1,
        init_hypotheses_per_room=2,
        num_epochs=1,
        store=lake,
    )
    return result, lake


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_e2e_date_partition() -> None:
    """utc_date_parts returns correctly formatted year/month/day strings."""
    y, m, d = utc_date_parts()
    assert len(y) == 4 and y.isdigit()
    assert len(m) == 2 and m.isdigit()
    assert len(d) == 2 and d.isdigit()


@pytest.mark.integration
def test_e2e_storage_buckets(dummy_dataset_path: Path, tmp_path: Path) -> None:
    """All three lake layers receive files after a training run."""
    lake = LocalStore(root=tmp_path / "_lake_buckets")
    train_vertical_slice(
        dataset_path=dummy_dataset_path,
        output_dir=tmp_path / "artifacts",
        experiment_name="bucket_test",
        llm_provider="mock",
        num_init_examples_per_room=1,
        init_hypotheses_per_room=2,
        num_epochs=1,
        store=lake,
    )

    bronze_files = lake.list_prefix("bronze/training")
    silver_files = lake.list_prefix("silver/training")
    gold_files = lake.list_prefix("gold/hypothesis_banks")

    assert len(bronze_files) > 0, "Bronze layer should have files"
    assert len(silver_files) > 0, "Silver layer should have files"
    assert len(gold_files) > 0, "Gold layer should have files"


@pytest.mark.integration
def test_e2e_bronze_ingestion(dummy_dataset_path: Path, tmp_path: Path) -> None:
    """Bronze layer contains raw scene programs after training."""
    lake = LocalStore(root=tmp_path / "_lake_bronze")
    result = train_vertical_slice(
        dataset_path=dummy_dataset_path,
        output_dir=tmp_path / "artifacts",
        experiment_name="bronze_e2e",
        llm_provider="mock",
        num_init_examples_per_room=1,
        init_hypotheses_per_room=2,
        num_epochs=1,
        store=lake,
    )
    run_id = result["lake_run_id"]
    pfx = training_bronze_prefix(run_id)

    bronze_files = lake.list_prefix(pfx)
    assert len(bronze_files) > 0

    # Verify run manifest is present.
    assert any("run_manifest.json" in f for f in bronze_files), (
        "run_manifest.json must be in bronze"
    )


@pytest.mark.integration
def test_e2e_silver_transformation(dummy_dataset_path: Path, tmp_path: Path) -> None:
    """Silver layer contains validated hypothesis bank and metrics after training."""
    lake = LocalStore(root=tmp_path / "_lake_silver")
    result = train_vertical_slice(
        dataset_path=dummy_dataset_path,
        output_dir=tmp_path / "artifacts",
        experiment_name="silver_e2e",
        llm_provider="mock",
        num_init_examples_per_room=1,
        init_hypotheses_per_room=2,
        num_epochs=1,
        store=lake,
    )
    run_id = result["lake_run_id"]
    pfx = training_silver_prefix(run_id)

    silver_files = lake.list_prefix(pfx)
    assert any("hypothesis_bank.json" in f for f in silver_files)
    assert any("metrics.json" in f for f in silver_files)

    # Confirm bank is a non-empty list of valid records.
    bank_data = lake.read_json(f"{pfx}/hypothesis_bank.json")
    assert isinstance(bank_data, list)
    assert len(bank_data) > 0
    assert "hypothesis_id" in bank_data[0]


@pytest.mark.integration
def test_e2e_gold_aggregation(dummy_dataset_path: Path, tmp_path: Path) -> None:
    """Gold layer contains latest.json and training_summary.json after training."""
    lake = LocalStore(root=tmp_path / "_lake_gold")
    result = train_vertical_slice(
        dataset_path=dummy_dataset_path,
        output_dir=tmp_path / "artifacts",
        experiment_name="gold_e2e",
        llm_provider="mock",
        num_init_examples_per_room=1,
        init_hypotheses_per_room=2,
        num_epochs=1,
        store=lake,
    )
    gold_pfx = training_gold_prefix("gold_e2e")
    gold_files = lake.list_prefix(gold_pfx)

    assert any("latest.json" in f for f in gold_files)
    assert any("training_summary.json" in f for f in gold_files)

    # latest.json must be a valid bank.
    latest = lake.read_json(f"{gold_pfx}/latest.json")
    assert isinstance(latest, list)
    assert len(latest) > 0

    # training_summary must reference the run.
    summary = lake.read_json(f"{gold_pfx}/training_summary.json")
    assert summary["experiment_name"] == "gold_e2e"
    assert "metrics" in summary


@pytest.mark.integration
def test_e2e_data_lake_status(dummy_dataset_path: Path, tmp_path: Path) -> None:
    """After a full training run the lake has files in all three layers."""
    lake = LocalStore(root=tmp_path / "_lake_status")
    train_vertical_slice(
        dataset_path=dummy_dataset_path,
        output_dir=tmp_path / "artifacts",
        experiment_name="status_check",
        llm_provider="mock",
        num_init_examples_per_room=1,
        init_hypotheses_per_room=2,
        num_epochs=1,
        store=lake,
    )

    bronze_count = len(lake.list_prefix("bronze"))
    silver_count = len(lake.list_prefix("silver"))
    gold_count = len(lake.list_prefix("gold"))

    print(f"\nData Lake Status:")
    print(f"  Bronze: {bronze_count} files")
    print(f"  Silver: {silver_count} files")
    print(f"  Gold:   {gold_count} files")

    assert bronze_count > 0
    assert silver_count > 0
    assert gold_count > 0
