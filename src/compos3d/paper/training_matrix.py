from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from compos3d.paper.benchmarks import write_json


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in overrides.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
            and merged.get(key) is not None
        ):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def write_experiment_config(
    *,
    base_config_path: Path,
    output_path: Path,
    overrides: dict[str, Any],
) -> Path:
    base_config = json.loads(base_config_path.read_text())
    write_json(output_path, _deep_merge(base_config, overrides))
    return output_path


def build_paper_training_matrix(
    *,
    base_config_path: Path,
    output_dir: Path,
    dataset_path: Path,
    benchmark_val_path: Path,
    benchmark_test_path: Path,
    training_output_dir: Path = Path("artifacts/training"),
    experiment_prefix: str = "paper_dining",
    final_bank_path: Path = Path("artifacts/training/claude_qwen/hypothesis_bank.json"),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    ablations = [
        ("full_system", {}),
        ("no_hypotheses", {"training": {"baseline_mode": "no_hypotheses"}}),
        ("fixed_hypotheses", {"training": {"baseline_mode": "fixed_hypotheses"}}),
        ("no_repair", {"training": {"use_repair": False}}),
        ("greedy", {"training": {"selection_strategy": "greedy"}}),
        ("random", {"training": {"selection_strategy": "random"}}),
    ]
    sweep_specs = [
        ("alpha_0_0", {"training": {"alpha": 0.0}}),
        ("alpha_0_25", {"training": {"alpha": 0.25}}),
        ("alpha_0_5", {"training": {"alpha": 0.5}}),
        ("alpha_1_0", {"training": {"alpha": 1.0}}),
        ("num_wrong_scale_0_4", {"training": {"num_wrong_scale": 0.4}}),
        ("num_wrong_scale_0_8", {"training": {"num_wrong_scale": 0.8}}),
        ("num_wrong_scale_1_2", {"training": {"num_wrong_scale": 1.2}}),
        ("success_threshold_0_65", {"training": {"success_threshold": 0.65}}),
        ("success_threshold_0_70", {"training": {"success_threshold": 0.7}}),
        ("success_threshold_0_75", {"training": {"success_threshold": 0.75}}),
    ]
    inference_ablations = [
        {
            "name": f"filter_and_weight_topk_{top_k}",
            "inference_strategy": "filter_and_weight",
            "config_overrides": {"training": {"top_k": top_k}},
        }
        for top_k in (1, 2, 3, 4)
    ] + [
        {
            "name": f"joint_top_k_topk_{top_k}",
            "inference_strategy": "joint_top_k",
            "config_overrides": {"training": {"top_k": top_k}},
        }
        for top_k in (1, 2, 3, 4)
    ]

    matrix: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "benchmark_val_path": str(benchmark_val_path),
        "benchmark_test_path": str(benchmark_test_path),
        "training_output_dir": str(training_output_dir),
        "train_runs": [],
        "sweeps": [],
        "inference_ablations": [],
    }

    for name, overrides in ablations:
        config_path = config_dir / f"{name}.json"
        write_experiment_config(
            base_config_path=base_config_path,
            output_path=config_path,
            overrides=overrides,
        )
        experiment_name = f"{experiment_prefix}_{name}"
        matrix["train_runs"].append(
            {
                "name": name,
                "experiment_name": experiment_name,
                "config_path": str(config_path),
                "train_command": _train_command(
                    dataset_path=dataset_path,
                    training_output_dir=training_output_dir,
                    experiment_name=experiment_name,
                    config_path=config_path,
                ),
                "val_eval_command": _generation_eval_command(
                    benchmark_path=benchmark_val_path,
                    output_dir=Path("paper/results/generation")
                    / f"{experiment_name}_val",
                    method_name=name,
                    bank_path=training_output_dir
                    / experiment_name
                    / "hypothesis_bank.json",
                    config_path=config_path,
                    inference_strategy="filter_and_weight",
                    use_empty_bank=name == "no_hypotheses",
                ),
            }
        )

    for name, overrides in sweep_specs:
        config_path = config_dir / f"{name}.json"
        write_experiment_config(
            base_config_path=base_config_path,
            output_path=config_path,
            overrides=overrides,
        )
        experiment_name = f"{experiment_prefix}_{name}"
        matrix["sweeps"].append(
            {
                "name": name,
                "experiment_name": experiment_name,
                "config_path": str(config_path),
                "train_command": _train_command(
                    dataset_path=dataset_path,
                    training_output_dir=training_output_dir,
                    experiment_name=experiment_name,
                    config_path=config_path,
                ),
                "val_eval_command": _generation_eval_command(
                    benchmark_path=benchmark_val_path,
                    output_dir=Path("paper/results/generation")
                    / f"{experiment_name}_val",
                    method_name=name,
                    bank_path=training_output_dir
                    / experiment_name
                    / "hypothesis_bank.json",
                    config_path=config_path,
                    inference_strategy="filter_and_weight",
                ),
            }
        )

    for spec in inference_ablations:
        config_path = config_dir / f"{spec['name']}.json"
        write_experiment_config(
            base_config_path=base_config_path,
            output_path=config_path,
            overrides=spec["config_overrides"],
        )
        matrix["inference_ablations"].append(
            {
                "name": spec["name"],
                "config_path": str(config_path),
                "test_eval_command": _generation_eval_command(
                    benchmark_path=benchmark_test_path,
                    output_dir=Path("paper/results/generation") / spec["name"],
                    method_name=spec["name"],
                    bank_path=final_bank_path,
                    config_path=config_path,
                    inference_strategy=spec["inference_strategy"],
                ),
            }
        )

    write_json(output_dir / "experiment_matrix.json", matrix)
    run_script = output_dir / "run_commands.sh"
    commands: list[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "source api_key",
    ]
    for section in ("train_runs", "sweeps", "inference_ablations"):
        for item in matrix[section]:
            if "train_command" in item:
                commands.append(item["train_command"])
                commands.append(item["val_eval_command"])
            else:
                commands.append(item["test_eval_command"])
    run_script.write_text("\n\n".join(commands) + "\n", encoding="utf-8")
    return matrix


def _train_command(
    *,
    dataset_path: Path,
    training_output_dir: Path,
    experiment_name: str,
    config_path: Path,
) -> str:
    return (
        "./.venv/bin/compos3d train-hypotheses "
        f"--dataset-path {dataset_path} "
        f"--output-dir {training_output_dir} "
        f"--experiment-name {experiment_name} "
        f"--config-path {config_path}"
    )


def _generation_eval_command(
    *,
    benchmark_path: Path,
    output_dir: Path,
    method_name: str,
    bank_path: Path,
    config_path: Path,
    inference_strategy: str,
    use_empty_bank: bool = False,
) -> str:
    command = (
        "./.venv/bin/python scripts/run_paper_generation_eval.py "
        f"--benchmark-path {benchmark_path} "
        f"--output-dir {output_dir} "
        f"--method-name {method_name} "
        f"--config-path {config_path} "
        f"--inference-strategy {inference_strategy} "
        "--render-scene"
    )
    if use_empty_bank:
        return f"{command} --use-empty-bank"
    return f"{command} --bank-path {bank_path}"
