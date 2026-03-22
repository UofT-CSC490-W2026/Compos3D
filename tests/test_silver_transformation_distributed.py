"""Distributed Silver transformation unit tests (ported from compos3d_dp).

Silver represents validated, cleaned data.  In Compos3D this means validated
hypothesis banks, critic scores, and metrics written after training completes.
"""

from __future__ import annotations

import uuid

import pytest

from compos3d.storage.local import LocalStore
from compos3d.storage.paths import training_silver_prefix


@pytest.mark.silver
@pytest.mark.unit
def test_silver_transformation_empty_bronze(tmp_store: LocalStore) -> None:
    """Listing a non-existent silver prefix returns an empty list, not an error."""
    files = tmp_store.list_prefix("silver/scenes/2099/01/01")
    assert files == []


@pytest.mark.silver
@pytest.mark.unit
def test_silver_write_hypothesis_bank(tmp_store: LocalStore) -> None:
    """A validated hypothesis bank can be stored in silver."""
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    pfx = training_silver_prefix(run_id)

    bank = [
        {
            "hypothesis_id": "h1",
            "text": "anchor around dining_table",
            "room_type": "dining_room",
            "reward": 0.8,
            "accuracy": 0.9,
            "num_visits": 5,
        },
        {
            "hypothesis_id": "h2",
            "text": "place rug under furniture",
            "room_type": "living_room",
            "reward": 0.6,
            "accuracy": 0.7,
            "num_visits": 3,
        },
    ]
    uri = tmp_store.put_json(f"{pfx}/hypothesis_bank.json", bank)
    assert uri

    loaded = tmp_store.read_json(f"{pfx}/hypothesis_bank.json")
    assert len(loaded) == 2
    assert loaded[0]["hypothesis_id"] == "h1"


@pytest.mark.silver
@pytest.mark.unit
def test_silver_write_metrics(tmp_store: LocalStore) -> None:
    """Training metrics are written to silver after training."""
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    pfx = training_silver_prefix(run_id)

    metrics = {
        "num_predictions": 4,
        "average_validity": 0.75,
        "average_prompt_adherence": 0.80,
        "average_asset_precision": 0.70,
        "average_asset_recall": 0.65,
        "average_room_match": 1.0,
        "average_overall": 0.78,
    }
    tmp_store.put_json(f"{pfx}/metrics.json", metrics)

    loaded = tmp_store.read_json(f"{pfx}/metrics.json")
    assert loaded["average_overall"] == pytest.approx(0.78)


@pytest.mark.silver
@pytest.mark.unit
def test_silver_write_critic_scores(tmp_store: LocalStore) -> None:
    """Per-scene critic scores can be stored in silver."""
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    pfx = training_silver_prefix(run_id)

    score = {
        "validity": 1.0,
        "prompt_adherence": 0.85,
        "asset_precision": 0.75,
        "asset_recall": 0.80,
        "room_match": 1.0,
        "overall": 0.88,
        "notes": [],
    }
    tmp_store.put_json(f"{pfx}/critic_score.json", score)

    assert tmp_store.exists(f"{pfx}/critic_score.json")


@pytest.mark.silver
@pytest.mark.unit
def test_silver_transformation_data_quality() -> None:
    """Critic scores are clamped between 0 and 1."""
    from compos3d.evaluation.critic import _clamp_unit  # internal but stable

    assert _clamp_unit(0.0) == 0.0
    assert _clamp_unit(1.0) == 1.0
    assert _clamp_unit(-0.5) == 0.0
    assert _clamp_unit(1.5) == 1.0
    assert _clamp_unit(0.75) == pytest.approx(0.75)


@pytest.mark.silver
@pytest.mark.integration
def test_silver_populated_after_training_with_store(
    dummy_dataset_path, tmp_path
) -> None:
    """Running train_vertical_slice with a LocalStore populates silver."""
    from compos3d.storage.local import LocalStore
    from compos3d.hypothesis.engine import train_vertical_slice

    store = LocalStore(root=tmp_path / "_lake")
    result = train_vertical_slice(
        dataset_path=dummy_dataset_path,
        output_dir=tmp_path / "artifacts",
        experiment_name="test_exp",
        llm_provider="mock",
        num_init_examples_per_room=1,
        init_hypotheses_per_room=2,
        num_epochs=1,
        store=store,
    )

    assert "lake_run_id" in result
    run_id = result["lake_run_id"]
    silver_pfx = training_silver_prefix(run_id)

    silver_files = store.list_prefix(silver_pfx)
    assert len(silver_files) > 0, (
        "Silver layer must have at least one file after training"
    )
    assert any("hypothesis_bank" in f for f in silver_files), (
        "Silver must contain hypothesis_bank.json"
    )
