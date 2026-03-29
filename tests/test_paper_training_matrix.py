from __future__ import annotations

import json
from pathlib import Path

import pytest

from compos3d.paper.training_matrix import (
    build_paper_training_matrix,
    write_experiment_config,
)


@pytest.mark.unit
def test_write_experiment_config_deep_merges_nested_training_fields(
    tmp_path: Path,
) -> None:
    base = {
        "generator": {"provider": "bedrock", "model_id": "base-model"},
        "critic": {"mode": "vlm", "provider": "bedrock", "model_id": "critic"},
        "training": {"top_k": 2, "alpha": 0.5, "use_repair": True},
    }
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base) + "\n", encoding="utf-8")
    out_path = tmp_path / "updated.json"

    write_experiment_config(
        base_config_path=base_path,
        output_path=out_path,
        overrides={"training": {"alpha": 1.0, "use_repair": False}},
    )

    payload = json.loads(out_path.read_text())
    assert payload["training"]["top_k"] == 2
    assert payload["training"]["alpha"] == 1.0
    assert payload["training"]["use_repair"] is False


@pytest.mark.unit
def test_build_paper_training_matrix_writes_expected_runs(tmp_path: Path) -> None:
    base_config_path = tmp_path / "base.json"
    base_config_path.write_text(
        json.dumps(
            {
                "generator": {"provider": "bedrock", "model_id": "g"},
                "critic": {"mode": "vlm", "provider": "bedrock", "model_id": "c"},
                "training": {
                    "top_k": 2,
                    "alpha": 0.5,
                    "num_wrong_scale": 0.8,
                    "success_threshold": 0.7,
                    "use_repair": True,
                    "selection_strategy": "ucb",
                    "baseline_mode": None,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    matrix = build_paper_training_matrix(
        base_config_path=base_config_path,
        output_dir=tmp_path / "matrix",
        dataset_path=Path("examples/vertical_slice_dataset.json"),
        benchmark_val_path=Path("paper/benchmarks/dining_val.json"),
        benchmark_test_path=Path("paper/benchmarks/dining_test.json"),
    )

    assert len(matrix["train_runs"]) == 6
    assert len(matrix["sweeps"]) == 10
    assert len(matrix["inference_ablations"]) == 8
    assert (tmp_path / "matrix" / "experiment_matrix.json").exists()
    assert (tmp_path / "matrix" / "run_commands.sh").exists()
