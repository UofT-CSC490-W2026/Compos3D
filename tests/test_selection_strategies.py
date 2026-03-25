"""Isolated selection-strategy tests for hypothesis bank sampling.

UCB, greedy, and random strategies with empty banks and room-scoped filters.

Edge cases covered:
- Empty-bank handling.
- UCB strategy ordering (reward-first via existing sort rules).
- Greedy strategy ordering (accuracy, mean score, visits).
- Random strategy constraints (room filtering and `k` limit).

Expected outcomes:
- Each strategy returns the intended subset/order for controlled fixtures.
- No cross-room hypotheses leak into selected results.
- Empty room banks produce an empty selection without errors.
"""

from __future__ import annotations

from pathlib import Path

from compos3d.hypothesis.loop import HypothesisLoopConfig, SceneHypothesisLoop
from compos3d.models import HypothesisRecord, TrainingDataset, TrainingExample


def _loop(strategy: str = "ucb", k: int = 2) -> SceneHypothesisLoop:
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
        config=HypothesisLoopConfig(selection_strategy=strategy, k=k),
    )


def _records() -> list[HypothesisRecord]:
    return [
        HypothesisRecord(hypothesis_id="h1", text="one", room_type="bedroom", reward=0.9, accuracy=0.7, mean_score=0.7, num_visits=3),
        HypothesisRecord(hypothesis_id="h2", text="two", room_type="bedroom", reward=0.6, accuracy=0.8, mean_score=0.6, num_visits=1),
        HypothesisRecord(hypothesis_id="h3", text="three", room_type="bedroom", reward=0.7, accuracy=0.6, mean_score=0.9, num_visits=5),
    ]


def test_selection_empty_bank_returns_empty() -> None:
    loop = _loop()
    assert loop._select_for_training("bedroom") == []  # noqa: SLF001


def test_selection_ucb_uses_reward_sorting() -> None:
    loop = _loop(strategy="ucb", k=2)
    loop.bank = _records()
    selected = loop._select_for_training("bedroom")  # noqa: SLF001
    assert [r.hypothesis_id for r in selected] == ["h1", "h3"]


def test_selection_greedy_uses_accuracy_then_mean_score_then_visits() -> None:
    loop = _loop(strategy="greedy", k=2)
    loop.bank = _records()
    selected = loop._select_for_training("bedroom")  # noqa: SLF001
    assert [r.hypothesis_id for r in selected] == ["h2", "h1"]


def test_selection_random_respects_k_and_room_filter() -> None:
    loop = _loop(strategy="random", k=2)
    loop.bank = _records() + [
        HypothesisRecord(hypothesis_id="x", text="other", room_type="living_room")
    ]
    selected = loop._select_for_training("bedroom")  # noqa: SLF001
    assert len(selected) == 2
    assert all(item.room_type == "bedroom" for item in selected)

