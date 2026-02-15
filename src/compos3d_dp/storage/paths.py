from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class LakePaths:
    bronze_prefix: str
    silver_prefix: str
    gold_prefix: str


def utc_date_parts(ts: datetime | None = None) -> tuple[str, str, str]:
    if ts is None:
        ts = datetime.now(timezone.utc)
    return (ts.strftime("%Y"), ts.strftime("%m"), ts.strftime("%d"))
