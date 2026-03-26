"""Focused tests for hypothesis update/reward mechanics in the loop.

Running averages, failure tagging, reward math with sparse visits, and
deterministic tie-breaking when sorting records.

Edge cases covered:
- Running-average score updates and visit/success bookkeeping.
- Failure-tag updates when scores miss success threshold.
- Reward math edge behavior (zero visits, early samples, exploration term).
- Deterministic tie handling in record sorting.

Expected outcomes:
- Counts and averages are updated exactly as specified by formulas.
- Reward computation follows UCB-style `accuracy + alpha * exploration`.
- Sorting is deterministic under ties by the configured tuple ordering.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from compos3d.hypothesis.loop import HypothesisLoopConfig, SceneHypothesisLoop
from compos3d.models import CriticScore, HypothesisRecord, TrainingDataset, TrainingExample


def _make_loop() -> SceneHypothesisLoop:
    dataset = TrainingDataset(
        dataset_id="d",
        examples=[
            TrainingExample(
                example_id="e1", room_type="bedroom", prompt="p", required_assets=["bed"]
            )
        ],
    )
    return SceneHypothesisLoop(
        dataset=dataset,
        llm=object(),
        critic=object(),
        run_dir=Path("."),
        llm_provider="mock",
        config=HypothesisLoopConfig(success_threshold=0.65, alpha=0.5),
    )


def _record() -> HypothesisRecord:
    return HypothesisRecord(
        hypothesis_id="h1",
        text="use bed near wall",
        room_type="bedroom",
        mean_score=0.6,
        num_visits=2,
        num_successes=1,
        accuracy=0.5,
        reward=0.0,
    )


def test_update_record_updates_running_mean_accuracy_and_counts() -> None:
    loop = _make_loop()
    record = _record()
    example = TrainingExample(
        example_id="e2", room_type="bedroom", prompt="p2", required_assets=["bed"]
    )
    score = CriticScore(
        validity=1, prompt_adherence=1, asset_precision=1, asset_recall=1, room_match=1, overall=0.8
    )
    loop._update_record(record, example, score, current_sample=3)  # noqa: SLF001

    assert record.num_visits == 3
    assert record.num_successes == 2
    assert record.mean_score == round((0.6 * 2 + 0.8) / 3, 4)
    assert record.accuracy == round(2 / 3, 4)
    assert "e2" in record.support_example_ids


def test_update_record_adds_failure_tag_when_below_threshold() -> None:
    loop = _make_loop()
    record = _record()
    example = TrainingExample(
        example_id="e3", room_type="bedroom", prompt="p3", required_assets=["bed"]
    )
    score = CriticScore(
        validity=1, prompt_adherence=1, asset_precision=1, asset_recall=1, room_match=1, overall=0.1
    )
    loop._update_record(record, example, score, current_sample=3)  # noqa: SLF001
    assert record.num_successes == 1
    assert "e3" in record.failure_tags


def test_compute_reward_bounds_and_zero_visits_behavior() -> None:
    loop = _make_loop()
    assert loop._compute_reward(accuracy=0.8, num_visits=0, current_sample=10) == 0.0  # noqa: SLF001
    assert loop._compute_reward(accuracy=0.8, num_visits=4, current_sample=1) == 0.8  # noqa: SLF001

    reward = loop._compute_reward(accuracy=0.5, num_visits=4, current_sample=10)  # noqa: SLF001
    assert reward == pytest.approx(0.5 + 0.5 * math.sqrt(math.log(10) / 4))


def test_sort_records_tie_breaking_is_deterministic() -> None:
    loop = _make_loop()
    r1 = HypothesisRecord(
        hypothesis_id="h1",
        text="a",
        room_type="bedroom",
        reward=0.5,
        accuracy=0.7,
        mean_score=0.6,
        num_visits=3,
    )
    r2 = HypothesisRecord(
        hypothesis_id="h2",
        text="b",
        room_type="bedroom",
        reward=0.5,
        accuracy=0.7,
        mean_score=0.6,
        num_visits=2,
    )
    ranked = loop._sort_records([r1, r2])  # noqa: SLF001
    assert ranked[0].hypothesis_id == "h1"
    assert ranked[1].hypothesis_id == "h2"

