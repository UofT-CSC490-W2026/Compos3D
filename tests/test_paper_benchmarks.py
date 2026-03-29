from __future__ import annotations

import random
from collections import Counter

import pytest

from compos3d.paper.benchmarks import (
    build_edit_pairs_from_splits,
    build_program_metric_row,
    compute_edit_program_metrics,
    stratified_split_examples,
)


def _example(
    example_id: str,
    required_assets: list[str],
    *,
    counts: dict[str, int] | None = None,
) -> dict:
    return {
        "example_id": example_id,
        "room_type": "dining_room",
        "prompt": f"prompt for {example_id}",
        "required_assets": required_assets,
        "expected_asset_counts": counts or {asset: 1 for asset in required_assets},
        "asset_tuple": list(required_assets),
    }


@pytest.mark.unit
def test_stratified_split_examples_keeps_rare_tuples_out_of_showcase() -> None:
    examples = [
        _example(f"rare_{index}", ["dining_table", "chair", "rug"])
        for index in range(3)
    ] + [
        _example(f"common_{index}", ["dining_table", "chair", "lamp"])
        for index in range(7)
    ]

    splits = stratified_split_examples(
        examples,
        split_sizes={"val": 2, "test": 5, "showcase": 3},
        seed=42,
        rare_asset_tuple_threshold=3,
    )

    showcase_ids = {row["example_id"] for row in splits["showcase"]}
    assert not any(example_id.startswith("rare_") for example_id in showcase_ids)
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 5
    assert len(splits["showcase"]) == 3


@pytest.mark.unit
def test_build_edit_pairs_from_splits_covers_all_edit_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(random.Random, "shuffle", lambda self, seq: None)
    val_examples = [
        _example("a_add", ["dining_table", "chair"], counts={"dining_table": 1, "chair": 2}),
        _example("b_remove", ["dining_table", "chair", "rug"], counts={"dining_table": 1, "chair": 4, "rug": 1}),
    ]
    test_examples = [
        _example("c_count", ["dining_table", "chair"], counts={"dining_table": 1, "chair": 1}),
        _example("d_swap", ["dining_table", "chair", "window"], counts={"dining_table": 1, "chair": 4, "window": 1}),
    ]

    pairs = build_edit_pairs_from_splits(
        val_examples=val_examples,
        test_examples=test_examples,
        seed=42,
        count_per_type=1,
    )

    assert Counter(pair["edit_type"] for pair in pairs) == {
        "add_asset": 1,
        "remove_asset": 1,
        "count_change": 1,
        "swap_asset": 1,
    }
    assert {pair["base_example_id"] for pair in pairs} == {
        "a_add",
        "b_remove",
        "c_count",
        "d_swap",
    }


@pytest.mark.unit
def test_program_and_edit_metrics_capture_counts_and_deltas() -> None:
    example = _example(
        "metric_example",
        ["dining_table", "chair", "lamp"],
        counts={"dining_table": 1, "chair": 4, "lamp": 1},
    )
    program_metrics = build_program_metric_row(
        example=example,
        scene_program={
            "room_type": "dining_room",
            "assets": [
                {"asset_type": "dining_table", "count": 1},
                {"asset_type": "chair", "count": 4},
                {"asset_type": "lamp", "count": 2},
                {"asset_type": "rug", "count": 1},
            ],
        },
    )

    assert program_metrics["exact_room_type_accuracy"] == 1.0
    assert program_metrics["requested_asset_recall"] == 1.0
    assert program_metrics["hallucinated_asset_rate"] == 0.25
    assert program_metrics["requested_count_error_total"] == 1.0

    pair = {
        "base_expected_asset_counts": {"dining_table": 1, "chair": 4, "lamp": 1},
        "edited_expected_asset_counts": {"dining_table": 1, "chair": 4, "window": 1},
        "changed_assets": ["lamp", "window"],
        "unchanged_assets": ["dining_table", "chair"],
    }
    edit_metrics = compute_edit_program_metrics(
        pair=pair,
        base_scene_program={
            "assets": [
                {"asset_type": "dining_table", "count": 1},
                {"asset_type": "chair", "count": 4},
                {"asset_type": "lamp", "count": 1},
            ]
        },
        edited_scene_program={
            "assets": [
                {"asset_type": "dining_table", "count": 1},
                {"asset_type": "chair", "count": 4},
                {"asset_type": "window", "count": 1},
            ]
        },
    )

    assert edit_metrics["delta_asset_f1"] == 1.0
    assert edit_metrics["unchanged_asset_retention_rate"] == 1.0
    assert edit_metrics["delta_count_l1"] == 0.0
