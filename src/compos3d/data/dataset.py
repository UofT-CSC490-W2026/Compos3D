from __future__ import annotations

import json
from pathlib import Path

from compos3d.models import TrainingDataset


def load_training_dataset(path: Path) -> TrainingDataset:
    data = json.loads(path.read_text())
    return TrainingDataset.model_validate(data)
