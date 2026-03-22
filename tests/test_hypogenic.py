"""HypoGeniC UCB bandit tests.

Verifies the hypothesis selection, update, persistence, and statistics logic
that powers the UCB-style hypothesis bank in Compos3D.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from compos3d.models import HypothesisRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(room_type: str = "dining_room", text: str = "anchor around dining_table") -> HypothesisRecord:
    import uuid
    return HypothesisRecord(
        hypothesis_id=uuid.uuid4().hex[:8],
        text=text,
        room_type=room_type,
        reward=0.0,
        accuracy=0.0,
        mean_score=0.0,
        num_visits=0,
        num_successes=0,
        generation_round=0,
        source_example_ids=[],
        support_example_ids=[],
        applicability_tags=[],
        failure_tags=[],
    )


def _ucb_score(record: HypothesisRecord, total_visits: int, alpha: float = 1.0) -> float:
    """Mirror of the UCB formula used in the loop."""
    if record.num_visits == 0:
        return float("inf")
    exploitation = record.accuracy
    exploration = alpha * math.sqrt(math.log(max(total_visits, 1)) / record.num_visits)
    return exploitation + exploration


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_hypothesis_record_creation() -> None:
    """HypothesisRecord can be created and serialised."""
    rec = _make_record()
    assert rec.hypothesis_id
    assert rec.room_type == "dining_room"
    dumped = rec.model_dump()
    assert dumped["reward"] == 0.0


@pytest.mark.unit
def test_ucb_score_unvisited_is_infinite() -> None:
    """An unvisited hypothesis has infinite UCB score — it must be tried first."""
    rec = _make_record()
    assert math.isinf(_ucb_score(rec, total_visits=10))


@pytest.mark.unit
def test_ucb_score_decreases_with_visits() -> None:
    """UCB score exploration bonus shrinks as a hypothesis is visited more."""
    rec = _make_record()
    total = 100

    scores = []
    for v in [1, 5, 20, 50]:
        rec = rec.model_copy(update={"num_visits": v, "num_successes": v // 2, "accuracy": 0.5})
        scores.append(_ucb_score(rec, total_visits=total))

    # More visits → lower exploration bonus → lower UCB score (accuracy constant)
    assert scores[0] > scores[1] > scores[2] > scores[3]


@pytest.mark.unit
def test_hypothesis_selection_prefers_unvisited() -> None:
    """UCB selection should prefer hypotheses with 0 visits."""
    records = [_make_record() for _ in range(5)]
    # Simulate that first 4 have been visited.
    for i, rec in enumerate(records[:4]):
        records[i] = rec.model_copy(update={"num_visits": 10, "accuracy": 0.8})

    # The 5th record has 0 visits → infinite UCB → must be selected first.
    total = sum(r.num_visits for r in records)
    scores = [_ucb_score(r, total) for r in records]
    best_idx = scores.index(max(scores))
    assert best_idx == 4  # the unvisited one


@pytest.mark.unit
def test_hypothesis_update_fields() -> None:
    """Updating a hypothesis record changes num_visits, num_successes, and reward."""
    rec = _make_record()
    assert rec.num_visits == 0

    # Simulate a successful evaluation.
    rec = rec.model_copy(update={
        "num_visits": rec.num_visits + 1,
        "num_successes": rec.num_successes + 1,
        "reward": 0.85,
        "accuracy": 1.0,
    })

    assert rec.num_visits == 1
    assert rec.num_successes == 1
    assert rec.reward == pytest.approx(0.85)


@pytest.mark.unit
def test_hypothesis_bank_save_and_load() -> None:
    """A list of HypothesisRecords can be serialised to JSON and loaded back."""
    bank = [_make_record("dining_room", f"hypothesis {i}") for i in range(5)]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "hypothesis_bank.json"
        path.write_text(json.dumps([r.model_dump() for r in bank], indent=2))

        loaded = [HypothesisRecord.model_validate(item) for item in json.loads(path.read_text())]

    assert len(loaded) == 5
    assert {r.text for r in loaded} == {r.text for r in bank}


@pytest.mark.unit
def test_hypothesis_bank_statistics() -> None:
    """Basic statistics over a set of hypothesis records."""
    records = [
        _make_record("dining_room"),
        _make_record("dining_room"),
        _make_record("living_room"),
    ]
    records[0] = records[0].model_copy(update={"num_visits": 3, "num_successes": 2, "accuracy": 0.67})
    records[1] = records[1].model_copy(update={"num_visits": 5, "num_successes": 4, "accuracy": 0.80})

    total_visits = sum(r.num_visits for r in records)
    by_room = {}
    for r in records:
        by_room.setdefault(r.room_type, []).append(r)

    assert total_visits == 8
    assert len(by_room["dining_room"]) == 2
    assert len(by_room["living_room"]) == 1


@pytest.mark.unit
def test_top_k_selection_by_ucb() -> None:
    """Selecting top-k by UCB score returns highest-scoring hypotheses."""
    import math

    records = [_make_record() for _ in range(6)]
    # Give different visit/accuracy profiles.
    profiles = [
        {"num_visits": 10, "accuracy": 0.9},
        {"num_visits": 1,  "accuracy": 0.1},
        {"num_visits": 5,  "accuracy": 0.7},
        {"num_visits": 0,  "accuracy": 0.0},   # unvisited → inf
        {"num_visits": 3,  "accuracy": 0.5},
        {"num_visits": 0,  "accuracy": 0.0},   # unvisited → inf
    ]
    records = [r.model_copy(update=p) for r, p in zip(records, profiles)]
    total = sum(r.num_visits for r in records)

    scored = sorted(records, key=lambda r: _ucb_score(r, total), reverse=True)
    top2 = scored[:2]

    # Both unvisited hypotheses (indices 3 and 5) have inf score → top 2
    assert all(r.num_visits == 0 for r in top2)
