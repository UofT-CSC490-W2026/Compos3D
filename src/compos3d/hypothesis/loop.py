from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from compos3d.evaluation.critic import (
    CriticUnavailableError,
    aggregate_prediction_scores,
    evaluate_scene_program,
)
from compos3d.models import (
    CriticScore,
    FailureRecord,
    HypothesisRecord,
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


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class SceneHypothesisLoop:
    def __init__(
        self,
        *,
        dataset: TrainingDataset,
        llm,
        critic,
        run_dir: Path,
        llm_provider: str,
        config: HypothesisLoopConfig,
        experiment_config: dict[str, Any] | None = None,
        config_path: str | None = None,
        renderer: Callable[[SceneProgram, Path], list[Path]] | None = None,
    ) -> None:
        self.dataset = dataset
        self.llm = llm
        self.critic = critic
        self.run_dir = run_dir
        self.llm_provider = llm_provider
        self.config = config
        self.experiment_config = experiment_config or {}
        self.config_path = config_path
        self.renderer = renderer

        self.by_room: dict[str, list[TrainingExample]] = {}
        for example in dataset.examples:
            self.by_room.setdefault(example.room_type, []).append(example)

        self.bank: list[HypothesisRecord] = []
        self.predictions: list[PredictionRecord] = []
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

    def train(self) -> dict:
        random.seed(self.config.seed)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.experiment_config:
            _write_json(self.run_dir / "experiment_config.json", self.experiment_config)
        self._initialize_bank()

        since_snapshot = 0
        for epoch in range(self.config.num_epochs):
            self._current_epoch = epoch
            for position_in_epoch, example in enumerate(self.dataset.examples, start=1):
                if (
                    epoch == 0
                    and example.example_id
                    in self.seed_example_ids_by_room.get(example.room_type, set())
                ):
                    continue

                self.processed_examples += 1
                current_sample = position_in_epoch
                selected = self._select_for_training(example.room_type)
                selected_ids = [record.hypothesis_id for record in selected]
                selected_texts = [record.text for record in selected]
                individual_scores: dict[str, float] = {}
                num_wrong_hypotheses = 0

                for record in selected:
                    critic_score = self._evaluate_hypothesis(record.text, example)
                    individual_scores[record.hypothesis_id] = critic_score.overall
                    was_successful = (
                        critic_score.overall >= self.config.success_threshold
                    )
                    if not was_successful:
                        num_wrong_hypotheses += 1
                    self._update_record(record, example, critic_score, current_sample)

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
                if len(selected) == 0 or num_wrong_hypotheses >= wrong_threshold:
                    self._record_failure(
                        example=example,
                        current_sample=current_sample,
                        selected_ids=selected_ids,
                        selected_texts=selected_texts,
                        num_wrong_hypotheses=num_wrong_hypotheses,
                        wrong_threshold=wrong_threshold,
                        individual_scores=individual_scores,
                        combined_score=combined_prediction.critic_score.overall,
                        critic_notes=combined_prediction.critic_score.notes,
                    )
                    triggered_regeneration = self._maybe_regenerate(
                        example.room_type, current_sample, epoch
                    )

                self.training_trace.append(
                    TrainingTraceRecord(
                        epoch=epoch,
                        current_sample=current_sample,
                        example_id=example.example_id,
                        room_type=example.room_type,
                        selected_hypothesis_ids=selected_ids,
                        selected_hypotheses=selected_texts,
                        individual_scores=individual_scores,
                        combined_score=combined_prediction.critic_score.overall,
                        triggered_regeneration=triggered_regeneration,
                        failure_buffer_size=len(
                            self.pending_failure_examples.get(example.room_type, [])
                        ),
                    )
                )

                since_snapshot += 1
                if since_snapshot >= self.config.save_every_n_examples:
                    self._save_bank_snapshot(f"epoch_{epoch}_sample_{current_sample}")
                    since_snapshot = 0

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
            self.run_dir / "failed_scene_bank.jsonl",
            [failure.model_dump() for failure in self.failed_scene_bank],
        )
        _write_json(
            self.run_dir / "manifest.json",
            {
                "dataset_id": self.dataset.dataset_id,
                "llm_provider": self.llm_provider,
                "config_path": self.config_path,
                "hypothesis_loop_config": asdict(self.config),
                "experiment_config": self.experiment_config,
                "bank_path": str(self.run_dir / "hypothesis_bank.json"),
                "predictions_path": str(self.run_dir / "predictions.jsonl"),
                "training_trace_path": str(self.run_dir / "training_trace.jsonl"),
                "failed_scene_bank_path": str(self.run_dir / "failed_scene_bank.jsonl"),
                "num_hypotheses": len(self.bank),
                "num_predictions": len(self.predictions),
                "num_regeneration_events": self.regeneration_events,
            },
        )
        self._save_bank_snapshot("final")

        return {
            "run_dir": str(self.run_dir),
            "metrics": summary.model_dump(),
            "num_hypotheses": len(self.bank),
            "num_predictions": len(self.predictions),
            "num_regeneration_events": self.regeneration_events,
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
            new_records = self._make_records(
                room_type=room_type,
                hypotheses=generated,
                evaluation_examples=seed_examples,
                generation_round=0,
                current_sample=max(len(seed_examples), 2),
            )
            self._replace_room_bank(room_type, new_records)
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
            normalized = self._normalize_text(hypothesis)
            if not normalized or normalized in existing_texts or normalized in seen_new:
                continue
            seen_new.add(normalized)

            scores: list[CriticScore] = []
            support_example_ids: list[str] = []
            failure_tags: list[str] = []
            num_successes = 0
            for example in evaluation_examples:
                critic_score = self._evaluate_hypothesis(hypothesis, example)
                scores.append(critic_score)
                if critic_score.overall >= self.config.success_threshold:
                    num_successes += 1
                    support_example_ids.append(example.example_id)
                else:
                    failure_tags.append(example.example_id)

            num_visits = len(evaluation_examples)
            accuracy = num_successes / num_visits if num_visits else 0.0
            mean_score = (
                sum(score.overall for score in scores) / num_visits
                if num_visits
                else 0.0
            )
            reward = self._compute_reward(
                accuracy=accuracy, num_visits=num_visits, current_sample=current_sample
            )

            records.append(
                HypothesisRecord(
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
                    source_example_ids=[
                        example.example_id for example in evaluation_examples
                    ],
                    support_example_ids=support_example_ids,
                    failure_tags=failure_tags,
                )
            )
        return records

    def _render(self, scene_program: SceneProgram, tag: str) -> list[Path]:
        """Call the optional renderer; return image paths or [] on failure."""
        if self.renderer is None:
            return []
        self._render_counter += 1
        render_dir = self.run_dir / "renders" / f"{self._render_counter:04d}_{tag}"
        try:
            paths = self.renderer(scene_program, render_dir)
            print(f"[loop] Rendered {len(paths)} views → {render_dir.name}")
            return paths
        except Exception as exc:  # noqa: BLE001
            print(f"[loop] Render failed ({tag}): {exc}")
            return []

    def _score(
        self,
        scene_program: SceneProgram,
        example: TrainingExample,
        image_paths: list[Path],
    ) -> CriticScore:
        """Evaluate scene_program with critic; return zero score on VLM unavailability."""
        try:
            return evaluate_scene_program(
                scene_program,
                example,
                critic=self.critic,
                image_paths=image_paths or None,
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
            )

    def _evaluate_hypothesis(
        self, hypothesis: str, example: TrainingExample
    ) -> CriticScore:
        hypotheses = (
            [] if self.config.baseline_mode == "no_hypotheses" else [hypothesis]
        )
        scene_program = self.llm.generate_scene_program(
            prompt=example.prompt,
            room_type=example.room_type,
            selected_hypotheses=hypotheses,
        )
        tag = f"ep{self._current_epoch}_{example.example_id}"
        image_paths = self._render(scene_program, tag)
        return self._score(scene_program, example, image_paths)

    def _build_prediction(
        self, example: TrainingExample, selected_hypotheses: list[str]
    ) -> PredictionRecord:
        active_hypotheses = (
            [] if self.config.baseline_mode == "no_hypotheses" else selected_hypotheses
        )
        scene_program = self.llm.generate_scene_program(
            prompt=example.prompt,
            room_type=example.room_type,
            selected_hypotheses=active_hypotheses,
        )
        tag = f"pred_ep{self._current_epoch}_{example.example_id}"
        image_paths = self._render(scene_program, tag)
        critic_score = self._score(scene_program, example, image_paths)
        return PredictionRecord(
            example_id=example.example_id,
            prompt=example.prompt,
            room_type=example.room_type,
            selected_hypotheses=active_hypotheses,
            scene_program=scene_program,
            critic_score=critic_score,
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
        current_sample: int,
        selected_ids: list[str],
        selected_texts: list[str],
        num_wrong_hypotheses: int,
        wrong_threshold: float,
        individual_scores: dict[str, float],
        combined_score: float,
        critic_notes: list[str],
    ) -> None:
        failure_record = FailureRecord(
            example_id=example.example_id,
            room_type=example.room_type,
            prompt=example.prompt,
            current_sample=current_sample,
            selected_hypothesis_ids=selected_ids,
            selected_hypotheses=selected_texts,
            num_wrong_hypotheses=num_wrong_hypotheses,
            wrong_threshold=math.ceil(wrong_threshold) if wrong_threshold > 0 else 0,
            combined_score=combined_score,
            individual_scores=individual_scores,
            critic_notes=critic_notes,
        )
        self.failed_scene_bank.append(failure_record)
        self.pending_failure_examples.setdefault(example.room_type, []).append(example)

    def _maybe_regenerate(
        self, room_type: str, current_sample: int, epoch: int
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
