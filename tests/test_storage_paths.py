"""Unit tests for canonical bronze/silver/gold path builders.

Stable key strings for training/inference layers and UTC date partition parts.

Edge cases covered:
- Stable key-format generation for training/inference prefixes.
- UTC date partition helper output formatting.

Expected outcomes:
- Path builders return deterministic, exact string keys.
- Date parts are zero-padded and UTC-aligned.
"""

from __future__ import annotations

from datetime import datetime, timezone

from compos3d.storage.paths import (
    inference_bronze_prefix,
    inference_gold_prefix,
    inference_silver_prefix,
    training_bronze_prefix,
    training_gold_prefix,
    training_silver_prefix,
    utc_date_parts,
)


def test_training_path_builders_are_stable() -> None:
    assert training_bronze_prefix("run1") == "bronze/training/run1"
    assert training_silver_prefix("run1") == "silver/training/run1"
    assert training_gold_prefix("exp1") == "gold/hypothesis_banks/exp1"


def test_inference_path_builders_are_stable() -> None:
    assert inference_bronze_prefix("r2") == "bronze/inference/r2"
    assert inference_silver_prefix("r2") == "silver/inference/r2"
    assert inference_gold_prefix("r2") == "gold/inference/r2"


def test_utc_date_parts_zero_padded_and_utc_based() -> None:
    ts = datetime(2026, 3, 4, 1, 2, 3, tzinfo=timezone.utc)
    assert utc_date_parts(ts) == ("2026", "03", "04")
