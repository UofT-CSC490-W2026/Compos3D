"""Canonical lake path helpers for bronze/silver/gold layout."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class LakePaths:
    """Logical prefix roots for each lake layer (without leading slash)."""

    bronze_prefix: str = "bronze"
    silver_prefix: str = "silver"
    gold_prefix: str = "gold"


def utc_date_parts(ts: datetime | None = None) -> tuple[str, str, str]:
    """Return (year, month, day) strings in UTC for date-partitioned paths."""
    if ts is None:
        ts = datetime.now(timezone.utc)
    return ts.strftime("%Y"), ts.strftime("%m"), ts.strftime("%d")


def training_bronze_prefix(run_id: str) -> str:
    return f"bronze/training/{run_id}"


def training_silver_prefix(run_id: str) -> str:
    return f"silver/training/{run_id}"


def training_gold_prefix(experiment_name: str) -> str:
    return f"gold/hypothesis_banks/{experiment_name}"


def training_checkpoint_prefix(experiment_name: str) -> str:
    return f"bronze/checkpoints/training/{experiment_name}/latest"


def inference_bronze_prefix(run_id: str) -> str:
    return f"bronze/inference/{run_id}"


def inference_silver_prefix(run_id: str) -> str:
    return f"silver/inference/{run_id}"


def inference_gold_prefix(run_id: str) -> str:
    return f"gold/inference/{run_id}"
