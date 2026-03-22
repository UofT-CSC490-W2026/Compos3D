"""Distributed Gold aggregation unit tests (ported from compos3d_dp).

Gold represents the final, publication-ready artifacts: the trained hypothesis
bank ready for inference, aggregated metrics, and scene features.
"""

from __future__ import annotations

import uuid

import pytest

from compos3d.storage.local import LocalStore
from compos3d.storage.paths import training_gold_prefix


@pytest.mark.gold
@pytest.mark.unit
def test_gold_aggregation_empty_silver(tmp_store: LocalStore) -> None:
    """Listing a non-existent gold prefix returns an empty list, not an error."""
    files = tmp_store.list_prefix("gold/hypothesis_banks/nonexistent_exp")
    assert files == []


@pytest.mark.gold
@pytest.mark.unit
def test_gold_write_latest_bank(tmp_store: LocalStore) -> None:
    """The final hypothesis bank is written to gold as latest.json."""
    exp_name = f"exp_{uuid.uuid4().hex[:6]}"
    pfx = training_gold_prefix(exp_name)

    bank = [
        {"hypothesis_id": "h1", "text": "anchor the room around the dining table",
         "room_type": "dining_room", "reward": 0.88, "accuracy": 0.92, "num_visits": 10},
    ]
    tmp_store.put_json(f"{pfx}/latest.json", bank)

    loaded = tmp_store.read_json(f"{pfx}/latest.json")
    assert len(loaded) == 1
    assert loaded[0]["reward"] == pytest.approx(0.88)


@pytest.mark.gold
@pytest.mark.unit
def test_gold_write_training_summary(tmp_store: LocalStore) -> None:
    """A training summary is written to gold with run metadata."""
    exp_name = f"exp_{uuid.uuid4().hex[:6]}"
    pfx = training_gold_prefix(exp_name)

    summary = {
        "run_id": "my_experiment_20260101_120000",
        "experiment_name": exp_name,
        "silver_bank_path": f"silver/training/my_run/hypothesis_bank.json",
        "metrics": {
            "average_overall": 0.81,
            "num_predictions": 8,
        },
    }
    tmp_store.put_json(f"{pfx}/training_summary.json", summary)

    loaded = tmp_store.read_json(f"{pfx}/training_summary.json")
    assert loaded["experiment_name"] == exp_name
    assert loaded["metrics"]["average_overall"] == pytest.approx(0.81)


@pytest.mark.gold
@pytest.mark.unit
def test_gold_aggregation_creates_statistics() -> None:
    """Aggregated prediction scores produce a well-formed EvaluationSummary."""
    from compos3d.evaluation.critic import aggregate_prediction_scores
    from compos3d.models import PredictionRecord, CriticScore

    scores = [
        CriticScore(validity=1.0, prompt_adherence=0.8, asset_precision=0.7,
                    asset_recall=0.75, room_match=1.0, overall=0.85, notes=[]),
        CriticScore(validity=1.0, prompt_adherence=0.9, asset_precision=0.8,
                    asset_recall=0.80, room_match=1.0, overall=0.90, notes=[]),
    ]
    records = [
        PredictionRecord(
            example_id=f"ex_{i}",
            prompt="a dining room",
            room_type="dining_room",
            selected_hypotheses=[],
            scene_program={
                "prompt": "a dining room",
                "room_type": "dining_room",
                "assets": [],
                "constraints": [],
                "style": "modern",
                "hypotheses": [],
            },
            critic_score=s,
        )
        for i, s in enumerate(scores)
    ]

    summary = aggregate_prediction_scores(records)
    assert summary.num_predictions == 2
    assert 0 < summary.average_overall <= 1.0
    assert summary.average_room_match == pytest.approx(1.0)


@pytest.mark.gold
@pytest.mark.integration
def test_gold_populated_after_training_with_store(dummy_dataset_path, tmp_path) -> None:
    """Running train_vertical_slice with a LocalStore populates gold."""
    from compos3d.storage.local import LocalStore
    from compos3d.hypothesis.engine import train_vertical_slice

    store = LocalStore(root=tmp_path / "_lake")
    result = train_vertical_slice(
        dataset_path=dummy_dataset_path,
        output_dir=tmp_path / "artifacts",
        experiment_name="test_exp_gold",
        llm_provider="mock",
        num_init_examples_per_room=1,
        init_hypotheses_per_room=2,
        num_epochs=1,
        store=store,
    )

    assert "lake_run_id" in result
    gold_pfx = training_gold_prefix("test_exp_gold")
    gold_files = store.list_prefix(gold_pfx)

    assert len(gold_files) > 0, "Gold layer must have at least one file after training"
    assert any("latest.json" in f for f in gold_files), "Gold must contain latest.json"
    assert any("training_summary" in f for f in gold_files), "Gold must contain training_summary.json"
