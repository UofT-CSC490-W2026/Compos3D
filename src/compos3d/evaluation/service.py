from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from compos3d.hypothesis.engine import evaluate_prediction_dir


@dataclass(frozen=True)
class EvaluateRequest:
    predictions_dir: Path
    output_dir: Path


def evaluate_run(request: EvaluateRequest) -> dict:
    return evaluate_prediction_dir(
        predictions_dir=request.predictions_dir, output_dir=request.output_dir
    )
