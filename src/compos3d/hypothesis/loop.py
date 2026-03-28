from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from compos3d.config import LoggingConfig
from compos3d.evaluation.critic import (
    CriticUnavailableError,
    aggregate_prediction_scores,
    evaluate_scene_program,
)
from compos3d.hypothesis.training_logging import TrainingLogger
from compos3d.models import (
    CriticScore,
    FailureRecord,
    HypothesisRecord,
    HypothesisEvaluationRecord,
    PredictionRecord,
    SceneProgram,
    TrainingDataset,
    TrainingExample,
    TrainingTraceRecord,
)


@dataclass(frozen=True)
class HypothesisLoopConfig:
    num_init_examples_per_room: int = 1
    init_hypotheses_per_room: int = 3
    k: int = 2
    alpha: float = 0.5
    max_num_hypotheses_per_room: int = 8
    num_wrong_scale: float = 0.8
    update_batch_size: int = 1
    num_hypotheses_to_update: int = 1
    update_hypotheses_per_batch: int = 2
    only_best_hypothesis: bool = False
    num_epochs: int = 2
    success_threshold: float = 0.65
    save_every_n_examples: int = 5
    # ablation flags
    selection_strategy: Literal["ucb", "greedy", "random"] = "ucb"
    use_repair: bool = True
    baseline_mode: str | None = None
    seed: int = 42


@dataclass(frozen=True)
class RenderArtifacts:
    render_dir: Path | None
    image_paths: list[Path]
    build_manifest_path: Path | None = None
    video_path: Path | None = None


@dataclass(frozen=True)
class SceneEvaluation:
    scene_program: SceneProgram
    critic_score: CriticScore
    render: RenderArtifacts


_CREATED_JSON_DIRS: set[Path] = set()


def _write_json(path: Path, payload: dict | list) -> None:
    parent = path.parent
    if parent not in _CREATED_JSON_DIRS or not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
        _CREATED_JSON_DIRS.add(parent)
    path.write_text(json.dumps(payload, separators=(",", ":")))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    parent = path.parent
    if parent not in _CREATED_JSON_DIRS or not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
        _CREATED_JSON_DIRS.add(parent)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class SceneHypothesisLoop:
    def __init__(
        self,
        *,
        dataset: TrainingDataset,
        llm,
        critic,
        run_dir: Path,
        experiment_name: str = "training_run",
        llm_provider: str,
        config: HypothesisLoopConfig,
        logging_config: LoggingConfig | None = None,
        experiment_config: dict[str, Any] | None = None,
        config_path: str | None = None,
        renderer: Callable[[SceneProgram, Path], list[Path]] | None = None,
        resume: bool = False,
    ) -> None:
        self.dataset = dataset
        self.llm = llm
        self.critic = critic
        self.run_dir = run_dir
        self.experiment_name = experiment_name
        self.llm_provider = llm_provider
        self.config = config
        self.logging_config = logging_config or LoggingConfig()
        self.experiment_config = experiment_config or {}
        self.config_path = config_path
        self.renderer = renderer
        self.resume = resume

        self.by_room: dict[str, list[TrainingExample]] = {}
        for example in dataset.examples:
            self.by_room.setdefault(example.room_type, []).append(example)
        self.example_by_id = {example.example_id: example for example in dataset.examples}

        self.bank: list[HypothesisRecord] = []
        self.predictions: list[PredictionRecord] = []
        self.hypothesis_evaluations: list[HypothesisEvaluationRecord] = []
        self.training_trace: list[TrainingTraceRecord] = []
        self.failed_scene_bank: list[FailureRecord] = []
        self.pending_failure_examples: dict[str, list[TrainingExample]] = {
            room_type: [] for room_type in self.by_room
        }
        self.seed_example_ids_by_room: dict[str, set[str]] = {}

        self._next_hypothesis_index = 0
        self._render_counter = 0
        self._current_epoch = 0
        self.regeneration_events = 0
        self.processed_examples = 0
        self._since_snapshot = 0
        self._resume_next_epoch = 0
        self._resume_next_position = 1
        self._restored_from_state = False
        self._logger = TrainingLogger(
            config=self.logging_config,
            run_dir=self.run_dir,
            experiment_name=self.experiment_name,
            config_payload=self.experiment_config,
            resume=resume,
        )

    def train(self) -> dict:
        random.seed(self.config.seed)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.experiment_config:
            _write_json(self.run_dir / "experiment_config.json", self.experiment_config)
        if self.resume and self._state_file().exists():
            self._load_or_initialize_state()
            self._logger.start()
        else:
            self._logger.start()
            self._load_or_initialize_state()
        self._persist_progress(
            next_epoch=self._resume_next_epoch,
            next_position=self._resume_next_position,
        )

        for epoch in range(self._resume_next_epoch, self.config.num_epochs):
            self._current_epoch = epoch
            for position_in_epoch, example in enumerate(self.dataset.examples, start=1):
                if (
                    epoch == self._resume_next_epoch
                    and position_in_epoch < self._resume_next_position
                ):
                    continue
                if (
                    epoch == 0
                    and example.example_id
                    in self.seed_example_ids_by_room.get(example.room_type, set())
                ):
                    continue

                self.processed_examples += 1
                global_step = self.processed_examples
                current_sample = position_in_epoch
                selected = self._select_for_training(example.room_type)
                selected_ids = [record.hypothesis_id for record in selected]
                selected_texts = [record.text for record in selected]
                individual_scores: dict[str, float] = {}
                num_wrong_hypotheses = 0

                for record in selected:
                    before = {
                        "reward": record.reward,
                        "accuracy": record.accuracy,
                        "mean_score": record.mean_score,
                        "num_visits": record.num_visits,
                        "num_successes": record.num_successes,
                    }
                    eval_result = self._evaluate_hypothesis(record.text, example)
                    critic_score = eval_result.critic_score
                    individual_scores[record.hypothesis_id] = critic_score.overall
                    was_successful = (
                        critic_score.overall >= self.config.success_threshold
                    )
                    if not was_successful:
                        num_wrong_hypotheses += 1
                    self._update_record(record, example, critic_score, current_sample)
                    eval_record = HypothesisEvaluationRecord(
                        global_step=global_step,
                        epoch=epoch,
                        current_sample=current_sample,
                        example_id=example.example_id,
                        room_type=example.room_type,
                        hypothesis_id=record.hypothesis_id,
                        hypothesis_text=record.text,
                        score=critic_score.overall,
                        validity=critic_score.validity,
                        prompt_adherence=critic_score.prompt_adherence,
                        asset_precision=critic_score.asset_precision,
                        asset_recall=critic_score.asset_recall,
                        room_match=critic_score.room_match,
                        was_successful=was_successful,
                        reward_before=before["reward"],
                        reward_after=record.reward,
                        accuracy_before=before["accuracy"],
                        accuracy_after=record.accuracy,
                        mean_score_before=before["mean_score"],
                        mean_score_after=record.mean_score,
                        num_visits_before=before["num_visits"],
                        num_visits_after=record.num_visits,
                        num_successes_before=before["num_successes"],
                        num_successes_after=record.num_successes,
                        image_paths=[
                            str(path) for path in eval_result.render.image_paths
                        ],
                        render_dir=(
                            str(eval_result.render.render_dir)
                            if eval_result.render.render_dir is not None
                            else None
                        ),
                        build_manifest_path=(
                            str(eval_result.render.build_manifest_path)
                            if eval_result.render.build_manifest_path is not None
                            else None
                        ),
                        video_path=(
                            str(eval_result.render.video_path)
                            if eval_result.render.video_path is not None
                            else None
                        ),
                    )
                    self.hypothesis_evaluations.append(eval_record)
                    if (
                        global_step % self.logging_config.log_every_n_examples == 0
                    ):
                        self._logger.log_hypothesis_eval(
                            eval_record,
                            log_media=(
                                global_step
                                % self.logging_config.log_hypothesis_eval_media_every_n_examples
                                == 0
                            ),
                        )

                combined_prediction = self._build_prediction(example, selected_texts)
                self.predictions.append(combined_prediction)
                _write_json(
                    self.run_dir
                    / "programs"
                    / f"{example.example_id}_epoch_{epoch}.json",
                    combined_prediction.model_dump(),
                )

                wrong_threshold = self._wrong_threshold(len(selected), current_sample)
                triggered_regeneration = False
                failure_record: FailureRecord | None = None
                if len(selected) == 0 or num_wrong_hypotheses >= wrong_threshold:
                    failure_record = self._record_failure(
                        example=example,
                        global_step=global_step,
                        epoch=epoch,
                        current_sample=current_sample,
                        selected_ids=selected_ids,
                        selected_texts=selected_texts,
                        num_wrong_hypotheses=num_wrong_hypotheses,
                        wrong_threshold=wrong_threshold,
                        individual_scores=individual_scores,
                        combined_score=combined_prediction.critic_score.overall,
                        combined_critic_score=combined_prediction.critic_score,
                        critic_notes=combined_prediction.critic_score.notes,
                        combined_image_paths=combined_prediction.image_paths,
                        combined_render_dir=combined_prediction.render_dir,
                        combined_build_manifest_path=combined_prediction.build_manifest_path,
                        combined_video_path=combined_prediction.video_path,
                    )
                    if global_step % self.logging_config.log_every_n_examples == 0:
                        self._logger.log_failure(failure_record)
                    triggered_regeneration = self._maybe_regenerate(
                        example.room_type, current_sample, epoch, global_step
                    )

                trace = TrainingTraceRecord(
                    global_step=global_step,
                    epoch=epoch,
                    current_sample=current_sample,
                    example_id=example.example_id,
                    room_type=example.room_type,
                    selected_hypothesis_ids=selected_ids,
                    selected_hypotheses=selected_texts,
                    num_selected_hypotheses=len(selected),
                    individual_scores=individual_scores,
                    combined_score=combined_prediction.critic_score.overall,
                    combined_validity=combined_prediction.critic_score.validity,
                    combined_prompt_adherence=combined_prediction.critic_score.prompt_adherence,
                    combined_asset_precision=combined_prediction.critic_score.asset_precision,
                    combined_asset_recall=combined_prediction.critic_score.asset_recall,
                    combined_room_match=combined_prediction.critic_score.room_match,
                    wrong_threshold=math.ceil(wrong_threshold)
                    if wrong_threshold > 0
                    else 0,
                    num_wrong_hypotheses=num_wrong_hypotheses,
                    triggered_regeneration=triggered_regeneration,
                    failure_buffer_size=len(
                        self.pending_failure_examples.get(example.room_type, [])
                    ),
                    selected_hypothesis_rewards={
                        record.hypothesis_id: record.reward for record in selected
                    },
                    selected_hypothesis_accuracies={
                        record.hypothesis_id: record.accuracy for record in selected
                    },
                    selected_hypothesis_mean_scores={
                        record.hypothesis_id: record.mean_score for record in selected
                    },
                    bank_size=len(self.bank),
                    bank_size_room=len(self._room_bank(example.room_type)),
                    room_pending_failures=len(
                        self.pending_failure_examples.get(example.room_type, [])
                    ),
                    total_pending_failures=sum(
                        len(items) for items in self.pending_failure_examples.values()
                    ),
                    regeneration_events=self.regeneration_events,
                    combined_image_paths=combined_prediction.image_paths,
                    combined_render_dir=combined_prediction.render_dir,
                    combined_build_manifest_path=combined_prediction.build_manifest_path,
                    combined_video_path=combined_prediction.video_path,
                )
                self.training_trace.append(trace)

                self._since_snapshot += 1
                if self._since_snapshot >= self.config.save_every_n_examples:
                    self._save_bank_snapshot(f"epoch_{epoch}_sample_{current_sample}")
                    self._since_snapshot = 0

                next_epoch = epoch
                next_position = position_in_epoch + 1
                if next_position > len(self.dataset.examples):
                    next_epoch = epoch + 1
                    next_position = 1
                self._persist_progress(
                    next_epoch=next_epoch,
                    next_position=next_position,
                )
                if global_step % self.logging_config.log_every_n_examples == 0:
                    self._logger.log_training_step(
                        trace=trace,
                        running_summary=aggregate_prediction_scores(self.predictions),
                        bank=self._sort_records(self.bank),
                        prediction=combined_prediction,
                        log_bank_table=(
                            global_step
                            % self.logging_config.log_bank_table_every_n_examples
                            == 0
                        ),
                        log_prediction_media=(
                            global_step
                            % self.logging_config.log_prediction_media_every_n_examples
                            == 0
                        ),
                    )

        summary = aggregate_prediction_scores(self.predictions)
        self._persist_progress(
            next_epoch=self.config.num_epochs,
            next_position=1,
        )
        self._save_bank_snapshot("final")
        self._logger.finish(
            summary=summary,
            run_dir=self.run_dir,
            manifest=self._manifest_payload(summary.model_dump()),
        )

        return {
            "run_dir": str(self.run_dir),
            "metrics": summary.model_dump(),
            "num_hypotheses": len(self.bank),
            "num_predictions": len(self.predictions),
            "num_regeneration_events": self.regeneration_events,
        }

    def _load_or_initialize_state(self) -> None:
        if self.resume and self._state_file().exists():
            self._restored_from_state = True
            self._restore_from_state(json.loads(self._state_file().read_text()))
            return

        self._initialize_bank()
        self._resume_next_epoch = 0
        self._resume_next_position = 1
        self._persist_progress(next_epoch=0, next_position=1)

    def _persist_progress(self, *, next_epoch: int, next_position: int) -> None:
        summary = aggregate_prediction_scores(self.predictions)
        _write_json(self.run_dir / "metrics.json", summary.model_dump())
        _write_json(self.run_dir / "hypothesis_bank.json", self._bank_payload())
        _write_jsonl(
            self.run_dir / "predictions.jsonl",
            [prediction.model_dump() for prediction in self.predictions],
        )
        _write_jsonl(
            self.run_dir / "training_trace.jsonl",
            [trace.model_dump() for trace in self.training_trace],
        )
        _write_jsonl(
            self.run_dir / "hypothesis_evaluations.jsonl",
            [record.model_dump() for record in self.hypothesis_evaluations],
        )
        _write_jsonl(
            self.run_dir / "failed_scene_bank.jsonl",
            [failure.model_dump() for failure in self.failed_scene_bank],
        )
        manifest = self._manifest_payload(summary.model_dump())
        _write_json(self.run_dir / "manifest.json", manifest)
        _write_json(
            self._state_file(),
            {
                "dataset_id": self.dataset.dataset_id,
                "next_epoch": next_epoch,
                "next_position": next_position,
                "since_snapshot": self._since_snapshot,
                "processed_examples": self.processed_examples,
                "current_epoch": self._current_epoch,
                "next_hypothesis_index": self._next_hypothesis_index,
                "render_counter": self._render_counter,
                "regeneration_events": self.regeneration_events,
                "seed_example_ids_by_room": {
                    room_type: sorted(example_ids)
                    for room_type, example_ids in self.seed_example_ids_by_room.items()
                },
                "pending_failure_example_ids": {
                    room_type: [example.example_id for example in examples]
                    for room_type, examples in self.pending_failure_examples.items()
                },
                "bank": [record.model_dump() for record in self.bank],
                "predictions": [record.model_dump() for record in self.predictions],
                "hypothesis_evaluations": [
                    record.model_dump() for record in self.hypothesis_evaluations
                ],
                "training_trace": [record.model_dump() for record in self.training_trace],
                "failed_scene_bank": [
                    record.model_dump() for record in self.failed_scene_bank
                ],
                "wandb_run_id": self._logger.run_id,
                "manifest": manifest,
            },
        )

    def _restore_from_state(self, payload: dict[str, Any]) -> None:
        if payload.get("dataset_id") != self.dataset.dataset_id:
            raise ValueError(
                "Resume state dataset_id does not match the requested dataset."
            )

        self._resume_next_epoch = int(payload.get("next_epoch", 0))
        self._resume_next_position = int(payload.get("next_position", 1))
        self._since_snapshot = int(payload.get("since_snapshot", 0))
        self.processed_examples = int(payload.get("processed_examples", 0))
        self._current_epoch = int(payload.get("current_epoch", 0))
        self._next_hypothesis_index = int(payload.get("next_hypothesis_index", 0))
        self._render_counter = int(payload.get("render_counter", 0))
        self.regeneration_events = int(payload.get("regeneration_events", 0))
        self.seed_example_ids_by_room = {
            room_type: set(example_ids)
            for room_type, example_ids in payload.get(
                "seed_example_ids_by_room", {}
            ).items()
        }
        self.bank = [
            HypothesisRecord.model_validate(item) for item in payload.get("bank", [])
        ]
        self.predictions = [
            PredictionRecord.model_validate(item)
            for item in payload.get("predictions", [])
        ]
        self.hypothesis_evaluations = [
            HypothesisEvaluationRecord.model_validate(item)
            for item in payload.get("hypothesis_evaluations", [])
        ]
        self.training_trace = [
            TrainingTraceRecord.model_validate(item)
            for item in payload.get("training_trace", [])
        ]
        self.failed_scene_bank = [
            FailureRecord.model_validate(item)
            for item in payload.get("failed_scene_bank", [])
        ]
        pending_ids = payload.get("pending_failure_example_ids", {})
        self.pending_failure_examples = {
            room_type: [
                self.example_by_id[example_id]
                for example_id in example_ids
                if example_id in self.example_by_id
            ]
            for room_type, example_ids in pending_ids.items()
        }
        for room_type in self.by_room:
            self.pending_failure_examples.setdefault(room_type, [])
        self._logger.run_id = payload.get("wandb_run_id")

    def _state_file(self) -> Path:
        return self.run_dir / "resume_state.json"

    def _manifest_payload(self, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset.dataset_id,
            "llm_provider": self.llm_provider,
            "config_path": self.config_path,
            "hypothesis_loop_config": asdict(self.config),
            "experiment_config": self.experiment_config,
            "bank_path": str(self.run_dir / "hypothesis_bank.json"),
            "predictions_path": str(self.run_dir / "predictions.jsonl"),
            "training_trace_path": str(self.run_dir / "training_trace.jsonl"),
            "hypothesis_evaluations_path": str(
                self.run_dir / "hypothesis_evaluations.jsonl"
            ),
            "failed_scene_bank_path": str(self.run_dir / "failed_scene_bank.jsonl"),
            "resume_state_path": str(self._state_file()),
            "runtime_context_path": str(self.run_dir / "runtime_context.json"),
            "num_hypotheses": len(self.bank),
            "num_predictions": len(self.predictions),
            "num_regeneration_events": self.regeneration_events,
            "processed_examples": self.processed_examples,
            "wandb_run_id": self._logger.run_id,
            "metrics": metrics,
        }

    def _initialize_bank(self) -> None:
        for room_type, examples in self.by_room.items():
            seed_examples = examples[
                : max(1, min(self.config.num_init_examples_per_room, len(examples)))
            ]
            self.seed_example_ids_by_room[room_type] = {
                example.example_id for example in seed_examples
            }
            generated = self.llm.generate_hypotheses(
                room_type,
                seed_examples,
                num_hypotheses=self.config.init_hypotheses_per_room,
                focus="general",
            )
            current_sample = max(len(seed_examples), 2)
            new_records: list[HypothesisRecord] = []
            existing_texts = {
                self._normalize_text(record.text) for record in self._room_bank(room_type)
            }
            seen_new: set[str] = set()
            for hypothesis in generated:
                record = self._make_record(
                    room_type=room_type,
                    hypothesis=hypothesis,
                    evaluation_examples=seed_examples,
                    generation_round=0,
                    current_sample=current_sample,
                    existing_texts=existing_texts,
                    seen_new=seen_new,
                )
                if record is None:
                    continue
                new_records.append(record)
                self._replace_room_bank(room_type, new_records)
                self._persist_progress(next_epoch=0, next_position=1)
                self._logger.log_initial_bank(self.bank)
        self._save_bank_snapshot("initial")

    def _make_records(
        self,
        *,
        room_type: str,
        hypotheses: list[str],
        evaluation_examples: list[TrainingExample],
        generation_round: int,
        current_sample: int,
    ) -> list[HypothesisRecord]:
        records: list[HypothesisRecord] = []
        existing_texts = {
            self._normalize_text(record.text) for record in self._room_bank(room_type)
        }
        seen_new: set[str] = set()

        for hypothesis in hypotheses:
            record = self._make_record(
                room_type=room_type,
                hypothesis=hypothesis,
                evaluation_examples=evaluation_examples,
                generation_round=generation_round,
                current_sample=current_sample,
                existing_texts=existing_texts,
                seen_new=seen_new,
            )
            if record is not None:
                records.append(record)
        return records

    def _make_record(
        self,
        *,
        room_type: str,
        hypothesis: str,
        evaluation_examples: list[TrainingExample],
        generation_round: int,
        current_sample: int,
        existing_texts: set[str],
        seen_new: set[str],
    ) -> HypothesisRecord | None:
        normalized = self._normalize_text(hypothesis)
        if not normalized or normalized in existing_texts or normalized in seen_new:
            return None
        seen_new.add(normalized)

        scores: list[CriticScore] = []
        support_example_ids: list[str] = []
        failure_tags: list[str] = []
        num_successes = 0
        for example in evaluation_examples:
            eval_result = self._evaluate_hypothesis(hypothesis, example)
            critic_score = eval_result.critic_score
            scores.append(critic_score)
            self._logger.log_initial_scene(
                room_type=room_type,
                example_id=example.example_id,
                prompt=example.prompt,
                hypothesis_text=hypothesis,
                score=critic_score.overall,
                image_paths=[str(path) for path in eval_result.render.image_paths],
                video_path=(
                    str(eval_result.render.video_path)
                    if eval_result.render.video_path is not None
                    else None
                ),
            )
            if critic_score.overall >= self.config.success_threshold:
                num_successes += 1
                support_example_ids.append(example.example_id)
            else:
                failure_tags.append(example.example_id)

        num_visits = len(evaluation_examples)
        accuracy = num_successes / num_visits if num_visits else 0.0
        mean_score = (
            sum(score.overall for score in scores) / num_visits if num_visits else 0.0
        )
        reward = self._compute_reward(
            accuracy=accuracy, num_visits=num_visits, current_sample=current_sample
        )

        return HypothesisRecord(
            hypothesis_id=self._next_hypothesis_id(),
            text=hypothesis,
            room_type=room_type,
            applicability_tags=[room_type],
            reward=round(reward, 4),
            accuracy=round(accuracy, 4),
            mean_score=round(mean_score, 4),
            num_visits=num_visits,
            num_successes=num_successes,
            generation_round=generation_round,
            source_example_ids=[example.example_id for example in evaluation_examples],
            support_example_ids=support_example_ids,
            failure_tags=failure_tags,
        )

    def _render(self, scene_program: SceneProgram, tag: str) -> RenderArtifacts:
        """Call the optional renderer; return render metadata or an empty result."""
        render_dir: Path | None = None
        if self.renderer is None:
            return RenderArtifacts(render_dir=None, image_paths=[])
        self._render_counter += 1
        render_dir = self.run_dir / "renders" / f"{self._render_counter:04d}_{tag}"
        try:
            paths = self.renderer(scene_program, render_dir)
            print(f"[loop] Rendered {len(paths)} views → {render_dir.name}")
            build_manifest_path = render_dir / "build_manifest.json"
            video_path = render_dir / "turntable.mp4"
            return RenderArtifacts(
                render_dir=render_dir,
                image_paths=paths,
                build_manifest_path=(
                    build_manifest_path if build_manifest_path.exists() else None
                ),
                video_path=video_path if video_path.exists() else None,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[loop] Render failed ({tag}): {exc}")
            return RenderArtifacts(render_dir=render_dir, image_paths=[])

    def _score(
        self,
        scene_program: SceneProgram,
        example: TrainingExample,
        render: RenderArtifacts,
    ) -> CriticScore:
        """Evaluate scene_program with critic; return zero score on VLM unavailability."""
        try:
            return evaluate_scene_program(
                scene_program,
                example,
                critic=self.critic,
                image_paths=render.image_paths or None,
            )
        except CriticUnavailableError as exc:
            return CriticScore(
                validity=0.0,
                prompt_adherence=0.0,
                asset_precision=0.0,
                asset_recall=0.0,
                room_match=0.0,
                overall=0.0,
                notes=[f"CriticUnavailable: {exc}"],
                critic_mode="unavailable",
                used_image_paths=[str(path) for path in render.image_paths],
            )

    def _evaluate_hypothesis(
        self, hypothesis: str, example: TrainingExample
    ) -> SceneEvaluation:
        hypotheses = (
            [] if self.config.baseline_mode == "no_hypotheses" else [hypothesis]
        )
        return self._generate_scored_scene(
            example=example,
            selected_hypotheses=hypotheses,
            tag=f"ep{self._current_epoch}_{example.example_id}",
        )

    def _build_prediction(
        self, example: TrainingExample, selected_hypotheses: list[str]
    ) -> PredictionRecord:
        active_hypotheses = (
            [] if self.config.baseline_mode == "no_hypotheses" else selected_hypotheses
        )
        result = self._generate_scored_scene(
            example=example,
            selected_hypotheses=active_hypotheses,
            tag=f"pred_ep{self._current_epoch}_{example.example_id}",
        )
        return PredictionRecord(
            example_id=example.example_id,
            prompt=example.prompt,
            room_type=example.room_type,
            selected_hypotheses=active_hypotheses,
            scene_program=result.scene_program,
            critic_score=result.critic_score,
            image_paths=[str(path) for path in result.render.image_paths],
            render_dir=(
                str(result.render.render_dir)
                if result.render.render_dir is not None
                else None
            ),
            build_manifest_path=(
                str(result.render.build_manifest_path)
                if result.render.build_manifest_path is not None
                else None
            ),
            video_path=(
                str(result.render.video_path)
                if result.render.video_path is not None
                else None
            ),
        )

    def _generate_scored_scene(
        self,
        *,
        example: TrainingExample,
        selected_hypotheses: list[str],
        tag: str,
    ) -> SceneEvaluation:
        scene_program = self.llm.generate_scene_program(
            prompt=example.prompt,
            room_type=example.room_type,
            selected_hypotheses=selected_hypotheses,
        )
        render = self._render(scene_program, tag)
        critic_score = self._score(scene_program, example, render)
        return SceneEvaluation(
            scene_program=scene_program,
            critic_score=critic_score,
            render=render,
        )

    def _update_record(
        self,
        record: HypothesisRecord,
        example: TrainingExample,
        critic_score: CriticScore,
        current_sample: int,
    ) -> None:
        old_visits = record.num_visits
        record.num_visits += 1
        record.mean_score = round(
            ((record.mean_score * old_visits) + critic_score.overall)
            / record.num_visits,
            4,
        )

        if critic_score.overall >= self.config.success_threshold:
            record.num_successes += 1
            if example.example_id not in record.support_example_ids:
                record.support_example_ids.append(example.example_id)
        else:
            if example.example_id not in record.failure_tags:
                record.failure_tags.append(example.example_id)

        record.accuracy = round(record.num_successes / record.num_visits, 4)
        record.reward = round(
            self._compute_reward(
                accuracy=record.accuracy,
                num_visits=record.num_visits,
                current_sample=current_sample,
            ),
            4,
        )

    def _record_failure(
        self,
        *,
        example: TrainingExample,
        global_step: int,
        epoch: int,
        current_sample: int,
        selected_ids: list[str],
        selected_texts: list[str],
        num_wrong_hypotheses: int,
        wrong_threshold: float,
        individual_scores: dict[str, float],
        combined_score: float,
        combined_critic_score: CriticScore,
        critic_notes: list[str],
        combined_image_paths: list[str],
        combined_render_dir: str | None,
        combined_build_manifest_path: str | None,
        combined_video_path: str | None,
    ) -> FailureRecord:
        failure_record = FailureRecord(
            example_id=example.example_id,
            room_type=example.room_type,
            prompt=example.prompt,
            global_step=global_step,
            epoch=epoch,
            current_sample=current_sample,
            selected_hypothesis_ids=selected_ids,
            selected_hypotheses=selected_texts,
            num_wrong_hypotheses=num_wrong_hypotheses,
            wrong_threshold=math.ceil(wrong_threshold) if wrong_threshold > 0 else 0,
            combined_score=combined_score,
            combined_validity=combined_critic_score.validity,
            combined_prompt_adherence=combined_critic_score.prompt_adherence,
            combined_asset_precision=combined_critic_score.asset_precision,
            combined_asset_recall=combined_critic_score.asset_recall,
            combined_room_match=combined_critic_score.room_match,
            individual_scores=individual_scores,
            critic_notes=critic_notes,
            combined_image_paths=combined_image_paths,
            combined_render_dir=combined_render_dir,
            combined_build_manifest_path=combined_build_manifest_path,
            combined_video_path=combined_video_path,
        )
        self.failed_scene_bank.append(failure_record)
        self.pending_failure_examples.setdefault(example.room_type, []).append(example)
        return failure_record

    def _maybe_regenerate(
        self,
        room_type: str,
        current_sample: int,
        epoch: int,
        global_step: int = 0,
    ) -> bool:
        if (
            not self.config.use_repair
            or self.config.baseline_mode == "fixed_hypotheses"
        ):
            return False
        pending_examples = self.pending_failure_examples.get(room_type, [])
        target_size = (
            self.config.update_batch_size * self.config.num_hypotheses_to_update
        )
        if target_size <= 0 or len(pending_examples) < target_size:
            return False

        generation_round = self.regeneration_events + 1
        new_records: list[HypothesisRecord] = []
        for _ in range(self.config.num_hypotheses_to_update):
            generated = self.llm.generate_hypotheses(
                room_type,
                pending_examples,
                num_hypotheses=self.config.update_hypotheses_per_batch,
                focus="repair",
            )
            candidate_records = self._make_records(
                room_type=room_type,
                hypotheses=generated,
                evaluation_examples=pending_examples,
                generation_round=generation_round,
                current_sample=max(current_sample, 2),
            )
            if self.config.only_best_hypothesis and candidate_records:
                new_records.append(self._sort_records(candidate_records)[0])
            else:
                new_records.extend(candidate_records)

        self._replace_room_bank(room_type, new_records)
        self.pending_failure_examples[room_type] = []
        self.regeneration_events += 1
        self._save_bank_snapshot(
            f"regen_{self.regeneration_events}_epoch_{epoch}_sample_{current_sample}"
        )
        self._logger.log_regeneration(
            global_step=global_step,
            room_type=room_type,
            new_records=len(new_records),
        )
        return True

    def _replace_room_bank(
        self, room_type: str, new_records: list[HypothesisRecord]
    ) -> None:
        other_rooms = [record for record in self.bank if record.room_type != room_type]
        room_records = self._room_bank(room_type)
        existing_texts = {self._normalize_text(record.text) for record in room_records}
        deduped_new: list[HypothesisRecord] = []
        seen_new: set[str] = set()
        for record in self._sort_records(new_records):
            normalized = self._normalize_text(record.text)
            if normalized in existing_texts or normalized in seen_new:
                continue
            seen_new.add(normalized)
            deduped_new.append(record)

        merged = self._sort_records(room_records + deduped_new)[
            : self.config.max_num_hypotheses_per_room
        ]
        self.bank = other_rooms + merged

    def _select_for_training(self, room_type: str) -> list[HypothesisRecord]:
        room_records = self._room_bank(room_type)
        if not room_records:
            return []
        strategy = self.config.selection_strategy
        if strategy == "random":
            k = min(self.config.k, len(room_records))
            return random.sample(room_records, k)
        if strategy == "greedy":
            ranked = sorted(
                room_records,
                key=lambda r: (r.accuracy, r.mean_score, r.num_visits),
                reverse=True,
            )
            return ranked[: self.config.k]
        # default: "ucb"
        return self._sort_records(room_records)[: self.config.k]

    def _room_bank(self, room_type: str) -> list[HypothesisRecord]:
        return [record for record in self.bank if record.room_type == room_type]

    def _sort_records(self, records: list[HypothesisRecord]) -> list[HypothesisRecord]:
        return sorted(
            records,
            key=lambda record: (
                record.reward,
                record.accuracy,
                record.mean_score,
                record.num_visits,
            ),
            reverse=True,
        )

    def _wrong_threshold(self, num_selected: int, current_sample: int) -> float:
        if num_selected == 0:
            return 0.0
        if self.config.num_wrong_scale <= 0:
            return float(num_selected)
        return (
            num_selected * current_sample / max(len(self.dataset.examples), 1)
        ) * self.config.num_wrong_scale

    def _compute_reward(
        self, *, accuracy: float, num_visits: int, current_sample: int
    ) -> float:
        if num_visits <= 0:
            return 0.0
        if current_sample <= 1:
            return accuracy
        return accuracy + self.config.alpha * math.sqrt(
            math.log(current_sample) / num_visits
        )

    def _save_bank_snapshot(self, suffix: str) -> None:
        _write_json(
            self.run_dir / "bank_snapshots" / f"hypothesis_bank_{suffix}.json",
            self._bank_payload(),
        )

    def _bank_payload(self) -> list[dict]:
        return [record.model_dump() for record in self._sort_records(self.bank)]

    def _next_hypothesis_id(self) -> str:
        hypothesis_id = f"hyp_{self._next_hypothesis_index:03d}"
        self._next_hypothesis_index += 1
        return hypothesis_id

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.lower().split())
