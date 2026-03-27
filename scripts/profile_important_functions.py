from __future__ import annotations

import argparse
import io
import json
import profile
import pstats
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compos3d.config import DEFAULT_EXPERIMENT_CONFIG
from compos3d.data.dataset import load_training_dataset
from compos3d.evaluation.critic import _build_heuristic_score, build_scene_critic
from compos3d.hypothesis.engine import (
    _loop_config_from_experiment,
    run_vertical_inference,
    train_vertical_slice,
)
from compos3d.hypothesis.loop import SceneHypothesisLoop
from compos3d.llm.scene_llm import MockSceneLLM
from compos3d.models import HypothesisRecord, TrainingDataset, TrainingExample


@dataclass(frozen=True)
class ProfileTarget:
    label: str
    qualified_name: str
    filename_suffix: str
    func_name: str
    runner: Callable[[], None]


def _clone_dataset(dataset: TrainingDataset, repeats: int) -> TrainingDataset:
    examples: list[TrainingExample] = []
    for repeat_idx in range(repeats):
        for example in dataset.examples:
            clone = example.model_copy(
                update={
                    "example_id": f"{example.example_id}_r{repeat_idx:03d}",
                }
            )
            examples.append(clone)
    return TrainingDataset(dataset_id=f"{dataset.dataset_id}_x{repeats}", examples=examples)


def _write_dataset(dataset: TrainingDataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset.model_dump(), indent=2))


def _find_stats_entry(
    stats: pstats.Stats, *, filename_suffix: str, func_name: str
) -> tuple[tuple[str, int, str], tuple[int, int, float, float, dict]]:
    normalized_suffix = filename_suffix.replace("\\", "/")
    for key, value in stats.stats.items():
        filename, _line, candidate_name = key
        normalized_filename = str(filename).replace("\\", "/")
        if candidate_name == func_name and normalized_filename.endswith(normalized_suffix):
            return key, value
    raise KeyError(f"Could not locate stats entry for {filename_suffix}:{func_name}")


def _write_profile_text(profile_obj: profile.Profile, path: Path) -> None:
    stream = io.StringIO()
    stats = pstats.Stats(profile_obj, stream=stream).strip_dirs()
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)
    stats.sort_stats(pstats.SortKey.TIME).print_stats(20)
    path.write_text(stream.getvalue())


def _profile_target(
    target: ProfileTarget,
    *,
    output_dir: Path,
) -> dict[str, object]:
    profiler = profile.Profile(timer=time.perf_counter)
    profiler.runcall(target.runner)
    profiler.create_stats()

    profile_path = output_dir / f"{target.label}.prof"
    text_path = output_dir / f"{target.label}.txt"
    profiler.dump_stats(str(profile_path))
    _write_profile_text(profiler, text_path)

    stats = pstats.Stats(profiler)
    key, value = _find_stats_entry(
        stats, filename_suffix=target.filename_suffix, func_name=target.func_name
    )
    filename, line_no, func_name = key
    total_calls, primitive_calls, total_time, cumulative_time, _callers = value
    return {
        "label": target.label,
        "function": target.qualified_name,
        "file": str(Path(filename).resolve()),
        "line": line_no,
        "function_name": func_name,
        "calls": total_calls,
        "primitive_calls": primitive_calls,
        "total_time_seconds": round(total_time, 6),
        "cumulative_time_seconds": round(cumulative_time, 6),
        "profile_dump": str(profile_path),
        "text_report": str(text_path),
    }


def _build_loop(
    *,
    dataset: TrainingDataset,
    run_dir: Path,
) -> SceneHypothesisLoop:
    experiment_config = DEFAULT_EXPERIMENT_CONFIG.model_copy(deep=True)
    return SceneHypothesisLoop(
        dataset=dataset,
        llm=MockSceneLLM(),
        critic=build_scene_critic(experiment_config.critic),
        run_dir=run_dir,
        llm_provider="mock",
        config=_loop_config_from_experiment(experiment_config),
        experiment_config=experiment_config.model_dump(),
        renderer=None,
    )


def _seed_bank_records(loop: SceneHypothesisLoop) -> list[HypothesisRecord]:
    loop._initialize_bank()
    return [record.model_copy(deep=True) for record in loop.bank]


def _write_summary(results: list[dict[str, object]], output_path: Path) -> None:
    ordered = sorted(
        results, key=lambda item: item["cumulative_time_seconds"], reverse=True
    )
    payload = {
        "method": {
            "module": "profile",
            "api": ["profile.Profile.runcall", "pstats.Stats.sort_stats"],
            "notes": [
                "These numbers include the overhead of the pure-Python profile module.",
                "The target functions were run on mock/heuristic local workloads to avoid AWS and Blender dependencies.",
            ],
        },
        "results": ordered,
    }
    output_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile 5 important Compos3D functions with the stdlib profile module."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "examples" / "dummy_fast.json",
        help="Base dataset used to synthesize profiling workloads.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "profiling" / "important_functions",
        help="Directory that will receive .prof dumps, text reports, and summary JSON.",
    )
    parser.add_argument(
        "--dataset-repeats",
        type=int,
        default=20,
        help="How many times to repeat the dummy dataset for training workloads.",
    )
    parser.add_argument(
        "--scene-program-iterations",
        type=int,
        default=2500,
        help="How many scene programs to generate when profiling MockSceneLLM.generate_scene_program.",
    )
    parser.add_argument(
        "--score-iterations",
        type=int,
        default=5000,
        help="How many heuristic score calls to make when profiling _build_heuristic_score.",
    )
    parser.add_argument(
        "--inference-iterations",
        type=int,
        default=150,
        help="How many inference runs to execute when profiling run_vertical_inference.",
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = load_training_dataset(args.dataset)
    expanded_dataset = _clone_dataset(base_dataset, args.dataset_repeats)
    expanded_dataset_path = args.output_dir / "workload_dataset.json"
    _write_dataset(expanded_dataset, expanded_dataset_path)

    tmpdir = args.output_dir / "_tmp"
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    train_output_root = tmpdir / "train_profile"
    inference_training_root = tmpdir / "inference_seed"
    loop_train_root = tmpdir / "loop_train"
    make_records_root = tmpdir / "make_records"

    seed_result = train_vertical_slice(
        dataset_path=expanded_dataset_path,
        output_dir=inference_training_root,
        experiment_name="seed_bank",
        llm_provider="mock",
        num_epochs=2,
        save_every_n_examples=200,
    )
    bank_path = Path(seed_result["run_dir"]) / "hypothesis_bank.json"

    inference_prompts = [
        "a cozy dining room with a wooden table, four chairs, and warm lighting",
        "a bright living room with a sofa, rug, and lamp",
        "a modern dining room with a round table and chairs",
    ]

    loop_seed = _build_loop(dataset=expanded_dataset, run_dir=make_records_root / "seed")
    seed_records = _seed_bank_records(loop_seed)
    evaluation_examples = expanded_dataset.examples[:8]
    hypotheses = [record.text for record in seed_records[:3]]

    llm = MockSceneLLM()
    scored_programs = [
        llm.generate_scene_program(
            prompt=example.prompt,
            room_type=example.room_type,
            selected_hypotheses=hypotheses[:2],
        )
        for example in evaluation_examples
    ]

    def run_train_vertical_slice() -> None:
        train_vertical_slice(
            dataset_path=expanded_dataset_path,
            output_dir=train_output_root,
            experiment_name="profile_train",
            llm_provider="mock",
            num_epochs=2,
            save_every_n_examples=200,
        )

    def run_vertical_inference_many() -> None:
        for idx in range(args.inference_iterations):
            run_vertical_inference(
                bank_path=bank_path,
                prompt=inference_prompts[idx % len(inference_prompts)],
                output_dir=tmpdir / "inference_runs" / f"run_{idx:03d}",
                llm_provider="mock",
                top_k=2,
                render_scene=False,
            )

    def run_loop_train() -> None:
        loop = _build_loop(dataset=expanded_dataset, run_dir=loop_train_root)
        loop.train()

    def run_generate_scene_program_many() -> None:
        prompts = [example.prompt for example in expanded_dataset.examples[:6]]
        room_types = [example.room_type for example in expanded_dataset.examples[:6]]
        selected = hypotheses[:2]
        for idx in range(args.scene_program_iterations):
            llm.generate_scene_program(
                prompt=prompts[idx % len(prompts)],
                room_type=room_types[idx % len(room_types)],
                selected_hypotheses=selected,
            )

    def run_build_heuristic_score_many() -> None:
        for idx in range(args.score_iterations):
            example = evaluation_examples[idx % len(evaluation_examples)]
            scene_program = scored_programs[idx % len(scored_programs)]
            _build_heuristic_score(scene_program, example)

    targets = [
        ProfileTarget(
            label="train_vertical_slice",
            qualified_name="compos3d.hypothesis.engine.train_vertical_slice",
            filename_suffix="src/compos3d/hypothesis/engine.py",
            func_name="train_vertical_slice",
            runner=run_train_vertical_slice,
        ),
        ProfileTarget(
            label="run_vertical_inference",
            qualified_name="compos3d.hypothesis.engine.run_vertical_inference",
            filename_suffix="src/compos3d/hypothesis/engine.py",
            func_name="run_vertical_inference",
            runner=run_vertical_inference_many,
        ),
        ProfileTarget(
            label="scene_hypothesis_loop_train",
            qualified_name="compos3d.hypothesis.loop.SceneHypothesisLoop.train",
            filename_suffix="src/compos3d/hypothesis/loop.py",
            func_name="train",
            runner=run_loop_train,
        ),
        ProfileTarget(
            label="mock_scene_llm_generate_scene_program",
            qualified_name="compos3d.llm.scene_llm.MockSceneLLM.generate_scene_program",
            filename_suffix="src/compos3d/llm/scene_llm.py",
            func_name="generate_scene_program",
            runner=run_generate_scene_program_many,
        ),
        ProfileTarget(
            label="build_heuristic_score",
            qualified_name="compos3d.evaluation.critic._build_heuristic_score",
            filename_suffix="src/compos3d/evaluation/critic.py",
            func_name="_build_heuristic_score",
            runner=run_build_heuristic_score_many,
        ),
    ]

    results = [_profile_target(target, output_dir=args.output_dir) for target in targets]
    _write_summary(results, args.output_dir / "summary.json")

    ordered = sorted(
        results, key=lambda item: item["cumulative_time_seconds"], reverse=True
    )
    print(json.dumps(ordered, indent=2))


if __name__ == "__main__":
    main()
