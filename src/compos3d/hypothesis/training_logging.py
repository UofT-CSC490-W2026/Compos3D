from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from compos3d.config import LoggingConfig
from compos3d.models import (
    EvaluationSummary,
    FailureRecord,
    HypothesisEvaluationRecord,
    HypothesisRecord,
    PredictionRecord,
    TrainingTraceRecord,
)


@dataclass
class TrainingLogger:
    config: LoggingConfig
    run_dir: Path
    experiment_name: str
    config_payload: dict[str, Any]
    resume: bool = False
    run_id: str | None = None
    _wandb: Any | None = field(init=False, default=None)
    _run: Any | None = field(init=False, default=None)
    _media_dir: Path = field(init=False)
    _media_manifest_path: Path = field(init=False)
    _logged_media_slugs: set[str] = field(init=False, default_factory=set)
    _init_table: Any | None = field(init=False, default=None)
    _hyp_eval_table: Any | None = field(init=False, default=None)
    _prediction_table: Any | None = field(init=False, default=None)
    _failure_table: Any | None = field(init=False, default=None)

    def start(self) -> str | None:
        self._media_dir = self.run_dir / "wandb_media"
        self._media_dir.mkdir(parents=True, exist_ok=True)
        self._media_manifest_path = self._media_dir / "media_manifest.jsonl"

        if not self.config.enable_wandb or self.config.wandb_mode == "disabled":
            return self.run_id

        try:
            import wandb  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "W&B logging is enabled, but the 'wandb' package is not installed. "
                "Install it in the Compos3D environment or disable logging.enable_wandb."
            ) from exc

        self._wandb = wandb
        self._run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            mode=self.config.wandb_mode,
            name=self.config.wandb_run_name or self.experiment_name,
            id=self.run_id,
            resume="allow" if self.resume else None,
            config=self.config_payload,
            tags=self.config.wandb_tags,
            dir=str(self.run_dir),
        )
        if self._run is not None:
            self.run_id = getattr(self._run, "id", self.run_id)
            self._define_metrics()
            self._initialize_tables()
        return self.run_id

    def log_initial_bank(self, bank: list[HypothesisRecord]) -> None:
        self._log_bank_table(bank=bank, global_step=0, prefix="bank/initial")
        self.log_scalars(
            {
                "bank/total_size": len(bank),
                "bank/mean_reward": _mean(record.reward for record in bank),
                "bank/mean_accuracy": _mean(record.accuracy for record in bank),
                "bank/mean_score": _mean(record.mean_score for record in bank),
            },
            step=0,
        )

    def log_initial_scene(
        self,
        *,
        room_type: str,
        example_id: str,
        prompt: str,
        hypothesis_text: str,
        score: float,
        image_paths: Sequence[str],
        video_path: str | None = None,
    ) -> None:
        if not self.config.log_initial_media:
            return
        payload = {
            "init/room_type": room_type,
            "init/example_id": example_id,
            "init/score": round(score, 4),
            "init/prompt": prompt,
            "init/hypothesis": hypothesis_text,
            "init/step": 0,
        }
        payload.update(
            self._build_scene_media_payload(
                prefix="init",
                slug=f"init_{room_type}_{example_id}",
                image_paths=image_paths,
                video_path=video_path,
                caption=f"{room_type} {example_id} score={score:.3f}",
                step=0,
                metadata={
                    "phase": "init",
                    "room_type": room_type,
                    "example_id": example_id,
                    "prompt": prompt,
                    "hypothesis_text": hypothesis_text,
                    "score": round(score, 4),
                },
            )
        )
        self.log_scalars(payload, step=0)
        self._log_scene_row(
            table=self._init_table,
            key="tables/init_samples",
            step=0,
            row=[
                0,
                room_type,
                example_id,
                round(score, 4),
                prompt,
                hypothesis_text,
                self._table_image(
                    self._preview_path(_slugify(f"init_{room_type}_{example_id}"))
                ),
                self._table_video(
                    self._media_video_path(
                        slug=_slugify(f"init_{room_type}_{example_id}"),
                        raw_video_path=video_path,
                    )
                ),
            ],
        )

    def log_hypothesis_eval(
        self, record: HypothesisEvaluationRecord, *, log_media: bool = False
    ) -> None:
        payload = {
            "hyp_eval/step": record.global_step,
            "hyp_eval/score": record.score,
            "hyp_eval/validity": record.validity,
            "hyp_eval/prompt_adherence": record.prompt_adherence,
            "hyp_eval/asset_precision": record.asset_precision,
            "hyp_eval/asset_recall": record.asset_recall,
            "hyp_eval/room_match": record.room_match,
            "hyp_eval/success": float(record.was_successful),
            "hyp_eval/reward_before": record.reward_before,
            "hyp_eval/reward_after": record.reward_after,
            "hyp_eval/accuracy_before": record.accuracy_before,
            "hyp_eval/accuracy_after": record.accuracy_after,
            "hyp_eval/num_visits_after": record.num_visits_after,
            "hyp_eval/example_id": record.example_id,
            "hyp_eval/room_type": record.room_type,
            "hyp_eval/hypothesis_id": record.hypothesis_id,
        }
        if log_media:
            payload.update(
                self._build_scene_media_payload(
                    prefix="hyp_eval",
                    slug=f"hyp_eval_{record.global_step}_{record.hypothesis_id}",
                    image_paths=record.image_paths,
                    video_path=record.video_path,
                    caption=(
                        f"{record.hypothesis_id} on {record.example_id} "
                        f"score={record.score:.3f}"
                    ),
                    step=record.global_step,
                    metadata={
                        "phase": "hypothesis_eval",
                        "global_step": record.global_step,
                        "epoch": record.epoch,
                        "example_id": record.example_id,
                        "room_type": record.room_type,
                        "hypothesis_id": record.hypothesis_id,
                        "score": record.score,
                        "was_successful": record.was_successful,
                        "render_dir": record.render_dir,
                    },
                )
            )
        self.log_scalars(payload, step=record.global_step)
        self._log_scene_row(
            table=self._hyp_eval_table,
            key="tables/hypothesis_evals",
            step=record.global_step,
            row=[
                record.global_step,
                record.epoch,
                record.room_type,
                record.example_id,
                record.hypothesis_id,
                round(record.score, 4),
                bool(record.was_successful),
                self._table_image(
                    self._preview_path(
                        _slugify(
                            f"hyp_eval_{record.global_step}_{record.hypothesis_id}"
                        )
                    )
                ),
                self._table_video(
                    self._media_video_path(
                        slug=_slugify(
                            f"hyp_eval_{record.global_step}_{record.hypothesis_id}"
                        ),
                        raw_video_path=record.video_path,
                    )
                ),
                record.render_dir,
            ],
        )

    def log_training_step(
        self,
        *,
        trace: TrainingTraceRecord,
        running_summary: EvaluationSummary,
        bank: list[HypothesisRecord],
        prediction: PredictionRecord | None = None,
        log_bank_table: bool = False,
        log_prediction_media: bool = False,
    ) -> None:
        payload = {
            "train/step": trace.global_step,
            "train/epoch": trace.epoch,
            "train/current_sample": trace.current_sample,
            "train/global_step": trace.global_step,
            "train/example_id": trace.example_id,
            "train/room_type": trace.room_type,
            "train/combined_overall": trace.combined_score,
            "train/combined_validity": trace.combined_validity,
            "train/combined_prompt_adherence": trace.combined_prompt_adherence,
            "train/combined_asset_precision": trace.combined_asset_precision,
            "train/combined_asset_recall": trace.combined_asset_recall,
            "train/combined_room_match": trace.combined_room_match,
            "train/num_selected_hypotheses": trace.num_selected_hypotheses,
            "train/num_wrong_hypotheses": trace.num_wrong_hypotheses,
            "train/wrong_threshold": trace.wrong_threshold,
            "train/triggered_regeneration": float(trace.triggered_regeneration),
            "train/failure_buffer_size": trace.failure_buffer_size,
            "train/bank_size": trace.bank_size,
            "train/bank_size_room": trace.bank_size_room,
            "train/room_pending_failures": trace.room_pending_failures,
            "train/total_pending_failures": trace.total_pending_failures,
            "train/regeneration_events": trace.regeneration_events,
            "summary/num_predictions": running_summary.num_predictions,
            "summary/average_validity": running_summary.average_validity,
            "summary/average_prompt_adherence": running_summary.average_prompt_adherence,
            "summary/average_asset_precision": running_summary.average_asset_precision,
            "summary/average_asset_recall": running_summary.average_asset_recall,
            "summary/average_room_match": running_summary.average_room_match,
            "summary/average_overall": running_summary.average_overall,
            "bank/mean_reward": _mean(record.reward for record in bank),
            "bank/mean_accuracy": _mean(record.accuracy for record in bank),
            "bank/mean_score": _mean(record.mean_score for record in bank),
            "bank/max_reward": max((record.reward for record in bank), default=0.0),
            "bank/max_accuracy": max((record.accuracy for record in bank), default=0.0),
        }
        if prediction is not None:
            payload.update(
                {
                    "pred/prompt": prediction.prompt,
                    "pred/selected_hypotheses_json": json.dumps(
                        prediction.selected_hypotheses
                    ),
                    "pred/scene_assets_json": json.dumps(
                        [
                            {
                                "asset_type": asset.asset_type,
                                "count": asset.count,
                                "placement": asset.placement,
                            }
                            for asset in prediction.scene_program.assets
                        ]
                    ),
                    "pred/scene_constraints_json": json.dumps(
                        [
                            constraint.text
                            for constraint in prediction.scene_program.constraints
                        ]
                    ),
                    "pred/score": prediction.critic_score.overall,
                    "pred/room_type": prediction.room_type,
                    "pred/example_id": prediction.example_id or "",
                    "pred/step": trace.global_step,
                }
            )
            if log_prediction_media:
                payload.update(
                    self._build_scene_media_payload(
                        prefix="pred",
                        slug=f"pred_{trace.global_step}_{trace.example_id}",
                        image_paths=prediction.image_paths,
                        video_path=prediction.video_path,
                        caption=(
                            f"{prediction.room_type} {trace.example_id} "
                            f"overall={prediction.critic_score.overall:.3f}"
                        ),
                        step=trace.global_step,
                        metadata={
                            "phase": "prediction",
                            "global_step": trace.global_step,
                            "epoch": trace.epoch,
                            "example_id": trace.example_id,
                            "room_type": trace.room_type,
                            "prompt": prediction.prompt,
                            "selected_hypotheses": prediction.selected_hypotheses,
                            "score": prediction.critic_score.overall,
                            "render_dir": prediction.render_dir,
                        },
                    )
                )

        self.log_scalars(payload, step=trace.global_step)
        if prediction is not None:
            self._log_scene_row(
                table=self._prediction_table,
                key="tables/predictions",
                step=trace.global_step,
                row=[
                    trace.global_step,
                    trace.epoch,
                    trace.room_type,
                    trace.example_id,
                    round(prediction.critic_score.overall, 4),
                    prediction.prompt,
                    json.dumps(prediction.selected_hypotheses),
                    self._table_image(
                        self._preview_path(
                            _slugify(f"pred_{trace.global_step}_{trace.example_id}")
                        )
                    ),
                    self._table_video(
                        self._media_video_path(
                            slug=_slugify(
                                f"pred_{trace.global_step}_{trace.example_id}"
                            ),
                            raw_video_path=prediction.video_path,
                        )
                    ),
                    prediction.render_dir,
                ],
            )
        if log_bank_table:
            self._log_bank_table(
                bank=bank,
                global_step=trace.global_step,
                prefix="bank/latest",
            )
        self._log_bank_histograms(bank=bank, global_step=trace.global_step)

    def log_failure(self, record: FailureRecord) -> None:
        payload = {
            "failure/step": record.global_step,
            "failure/event": 1.0,
            "failure/example_id": record.example_id,
            "failure/room_type": record.room_type,
            "failure/combined_score": record.combined_score,
            "failure/num_wrong_hypotheses": record.num_wrong_hypotheses,
            "failure/wrong_threshold": record.wrong_threshold,
        }
        if self.config.log_failure_media:
            payload.update(
                self._build_scene_media_payload(
                    prefix="failure",
                    slug=f"failure_{record.global_step}_{record.example_id}",
                    image_paths=record.combined_image_paths,
                    video_path=record.combined_video_path,
                    caption=(
                        f"{record.room_type} {record.example_id} "
                        f"failure overall={record.combined_score:.3f}"
                    ),
                    step=record.global_step,
                    metadata={
                        "phase": "failure",
                        "global_step": record.global_step,
                        "epoch": record.epoch,
                        "example_id": record.example_id,
                        "room_type": record.room_type,
                        "prompt": record.prompt,
                        "combined_score": record.combined_score,
                        "render_dir": record.combined_render_dir,
                    },
                )
            )
        self.log_scalars(payload, step=record.global_step)
        self._log_scene_row(
            table=self._failure_table,
            key="tables/failures",
            step=record.global_step,
            row=[
                record.global_step,
                record.epoch,
                record.room_type,
                record.example_id,
                round(record.combined_score, 4),
                record.prompt,
                json.dumps(record.selected_hypotheses),
                self._table_image(
                    self._preview_path(
                        _slugify(f"failure_{record.global_step}_{record.example_id}")
                    )
                ),
                self._table_video(
                    self._media_video_path(
                        slug=_slugify(
                            f"failure_{record.global_step}_{record.example_id}"
                        ),
                        raw_video_path=record.combined_video_path,
                    )
                ),
                record.combined_render_dir,
            ],
        )

    def log_regeneration(
        self, *, global_step: int, room_type: str, new_records: int
    ) -> None:
        self.log_scalars(
            {
                "regeneration/event": 1.0,
                "regeneration/new_records": float(new_records),
                f"regeneration/by_room/{room_type}": 1.0,
            },
            step=global_step,
        )

    def finish(
        self, *, summary: EvaluationSummary, run_dir: Path, manifest: dict[str, Any]
    ) -> None:
        self.log_scalars(
            {
                "final/num_predictions": summary.num_predictions,
                "final/average_validity": summary.average_validity,
                "final/average_prompt_adherence": summary.average_prompt_adherence,
                "final/average_asset_precision": summary.average_asset_precision,
                "final/average_asset_recall": summary.average_asset_recall,
                "final/average_room_match": summary.average_room_match,
                "final/average_overall": summary.average_overall,
            },
            step=summary.num_predictions,
        )

        if (
            self._run is not None
            and self._wandb is not None
            and self.config.upload_artifact_at_end
        ):
            artifact = self._wandb.Artifact(
                name=f"{self.experiment_name}-training",
                type="training-run",
                metadata={"manifest": manifest},
            )
            for rel_name in (
                "metrics.json",
                "hypothesis_bank.json",
                "predictions.jsonl",
                "training_trace.jsonl",
                "hypothesis_evaluations.jsonl",
                "failed_scene_bank.jsonl",
                "manifest.json",
                "resume_state.json",
            ):
                path = run_dir / rel_name
                if path.exists():
                    artifact.add_file(str(path), name=rel_name)
            snapshots_dir = run_dir / "bank_snapshots"
            if snapshots_dir.exists():
                for path in sorted(snapshots_dir.glob("*.json")):
                    artifact.add_file(str(path), name=f"bank_snapshots/{path.name}")
            if self.config.upload_media_artifact_at_end and self._media_dir.exists():
                for path in sorted(self._media_dir.glob("**/*")):
                    if path.is_file():
                        artifact.add_file(
                            str(path),
                            name=str(path.relative_to(self.run_dir)),
                        )
            self._run.log_artifact(artifact)

        if self._run is not None:
            self._run.finish()

    def log_scalars(self, payload: dict[str, Any], *, step: int) -> None:
        if self._run is None:
            return
        self._run.log(payload, step=step)

    def _build_scene_media_payload(
        self,
        *,
        prefix: str,
        slug: str,
        image_paths: Sequence[str] | Sequence[Path],
        video_path: str | None,
        caption: str,
        step: int,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if self._run is None or self._wandb is None:
            return {}

        normalized_image_paths = [
            Path(path) for path in image_paths if path and Path(path).exists()
        ][: self.config.max_media_images]
        normalized_video_path = (
            Path(video_path) if video_path and Path(video_path).exists() else None
        )
        if not normalized_image_paths and normalized_video_path is None:
            return {}

        safe_slug = _slugify(slug)
        preview_path = self._preview_path(safe_slug)
        animation_path = self._animation_path(safe_slug)
        if normalized_image_paths:
            self._ensure_preview(preview_path, normalized_image_paths)
        if normalized_video_path is None and self.config.create_animation_gif:
            self._ensure_animation(animation_path, normalized_image_paths)

        payload: dict[str, Any] = {
            f"{prefix}/caption": caption,
            f"{prefix}/image_count": len(normalized_image_paths),
        }
        render_dir = self._infer_render_dir(
            image_paths=normalized_image_paths, video_path=normalized_video_path
        )
        if render_dir is not None:
            payload[f"{prefix}/render_dir"] = str(render_dir)
        if preview_path.exists():
            payload[f"{prefix}/preview"] = self._wandb.Image(
                str(preview_path),
                caption=caption,
            )
            payload[f"{prefix}/examples"] = [
                self._wandb.Image(str(path), caption=f"{caption} :: {path.name}")
                for path in normalized_image_paths
            ]
        media_video_path = self._media_video_path(
            slug=safe_slug,
            raw_video_path=str(normalized_video_path)
            if normalized_video_path
            else None,
        )
        if media_video_path is not None:
            payload[f"{prefix}/animation"] = self._wandb.Video(
                str(media_video_path),
                format=media_video_path.suffix.lower().lstrip(".") or None,
            )

        self._append_media_manifest(
            slug=safe_slug,
            step=step,
            prefix=prefix,
            metadata=metadata,
            preview_path=preview_path if preview_path.exists() else None,
            animation_path=media_video_path,
            image_paths=normalized_image_paths,
            video_path=normalized_video_path,
        )
        return payload

    def _define_metrics(self) -> None:
        if self._run is None or not hasattr(self._run, "define_metric"):
            return
        self._run.define_metric("train/step")
        self._run.define_metric("train/*", step_metric="train/step")
        self._run.define_metric("summary/*", step_metric="train/step")
        self._run.define_metric("pred/step")
        self._run.define_metric("pred/*", step_metric="pred/step")
        self._run.define_metric("hyp_eval/step")
        self._run.define_metric("hyp_eval/*", step_metric="hyp_eval/step")
        self._run.define_metric("failure/step")
        self._run.define_metric("failure/*", step_metric="failure/step")
        self._run.define_metric("regeneration/*", step_metric="train/step")

    def _initialize_tables(self) -> None:
        self._init_table = self._make_table(
            columns=[
                "step",
                "room_type",
                "example_id",
                "score",
                "prompt",
                "hypothesis_text",
                "preview",
                "animation",
            ]
        )
        self._hyp_eval_table = self._make_table(
            columns=[
                "step",
                "epoch",
                "room_type",
                "example_id",
                "hypothesis_id",
                "score",
                "was_successful",
                "preview",
                "animation",
                "render_dir",
            ]
        )
        self._prediction_table = self._make_table(
            columns=[
                "step",
                "epoch",
                "room_type",
                "example_id",
                "score",
                "prompt",
                "selected_hypotheses_json",
                "preview",
                "animation",
                "render_dir",
            ]
        )
        self._failure_table = self._make_table(
            columns=[
                "step",
                "epoch",
                "room_type",
                "example_id",
                "score",
                "prompt",
                "selected_hypotheses_json",
                "preview",
                "animation",
                "render_dir",
            ]
        )

    def _make_table(self, *, columns: list[str]) -> Any | None:
        if self._wandb is None:
            return None
        try:
            return self._wandb.Table(columns=columns, log_mode="MUTABLE")
        except TypeError:
            return self._wandb.Table(columns=columns)

    def _log_scene_row(
        self,
        *,
        table: Any | None,
        key: str,
        step: int,
        row: list[Any],
    ) -> None:
        if self._run is None or table is None:
            return
        table.add_data(*row)
        self._run.log({key: table}, step=step)

    def _log_bank_histograms(
        self,
        *,
        bank: list[HypothesisRecord],
        global_step: int,
    ) -> None:
        if self._run is None or self._wandb is None or not bank:
            return
        try:
            self._run.log(
                {
                    "bank/reward_hist": self._wandb.Histogram(
                        [record.reward for record in bank]
                    ),
                    "bank/accuracy_hist": self._wandb.Histogram(
                        [record.accuracy for record in bank]
                    ),
                    "bank/score_hist": self._wandb.Histogram(
                        [record.mean_score for record in bank]
                    ),
                },
                step=global_step,
            )
        except Exception:
            return

    def _log_bank_table(
        self, *, bank: list[HypothesisRecord], global_step: int, prefix: str
    ) -> None:
        if self._run is None or self._wandb is None:
            return
        table = self._wandb.Table(
            columns=[
                "hypothesis_id",
                "room_type",
                "text",
                "reward",
                "accuracy",
                "mean_score",
                "num_visits",
                "num_successes",
                "generation_round",
                "support_example_ids",
                "failure_tags",
            ]
        )
        for record in bank:
            table.add_data(
                record.hypothesis_id,
                record.room_type,
                record.text,
                record.reward,
                record.accuracy,
                record.mean_score,
                record.num_visits,
                record.num_successes,
                record.generation_round,
                json.dumps(record.support_example_ids),
                json.dumps(record.failure_tags),
            )
        self._run.log({prefix: table}, step=global_step)

    def _preview_path(self, slug: str) -> Path:
        return self._media_dir / f"{slug}_preview.png"

    def _animation_path(self, slug: str) -> Path:
        return self._media_dir / f"{slug}_animation.gif"

    def _media_video_path(
        self, *, slug: str, raw_video_path: str | None
    ) -> Path | None:
        if raw_video_path and Path(raw_video_path).exists():
            return Path(raw_video_path)
        animation_path = self._animation_path(slug)
        if animation_path.exists():
            return animation_path
        return None

    def _ensure_preview(self, preview_path: Path, image_paths: Sequence[Path]) -> None:
        if preview_path.exists() or not image_paths:
            return
        frames = self._load_frames(image_paths)
        if not frames:
            return
        cols = 2 if len(frames) > 1 else 1
        rows = math.ceil(len(frames) / cols)
        width = max(frame.width for frame in frames)
        height = max(frame.height for frame in frames)
        preview = Image.new("RGB", (cols * width, rows * height), color=(248, 248, 248))
        for index, frame in enumerate(frames):
            resized = frame.resize((width, height))
            row = index // cols
            col = index % cols
            preview.paste(resized, (col * width, row * height))
        preview.save(preview_path)

    def _ensure_animation(
        self, animation_path: Path, image_paths: Sequence[Path]
    ) -> None:
        if animation_path.exists() or len(image_paths) < 2:
            return
        frames = self._load_frames(image_paths)
        if len(frames) < 2:
            return
        ping_pong = frames + list(reversed(frames[1:-1]))
        ping_pong[0].save(
            animation_path,
            save_all=True,
            append_images=ping_pong[1:],
            duration=450,
            loop=0,
        )

    @staticmethod
    def _load_frames(image_paths: Sequence[Path]) -> list[Image.Image]:
        frames: list[Image.Image] = []
        for path in image_paths:
            try:
                with Image.open(path) as source:
                    frames.append(source.convert("RGB").copy())
            except Exception:
                continue
        return frames

    @staticmethod
    def _infer_render_dir(
        *, image_paths: Sequence[Path], video_path: Path | None
    ) -> Path | None:
        if video_path is not None:
            return video_path.parent
        if not image_paths:
            return None
        first = image_paths[0]
        if first.parent.name == "views":
            return first.parent.parent
        return first.parent

    def _append_media_manifest(
        self,
        *,
        slug: str,
        step: int,
        prefix: str,
        metadata: dict[str, Any],
        preview_path: Path | None,
        animation_path: Path | None,
        image_paths: Sequence[Path],
        video_path: Path | None,
    ) -> None:
        if slug in self._logged_media_slugs:
            return
        self._logged_media_slugs.add(slug)
        payload = {
            "slug": slug,
            "step": step,
            "prefix": prefix,
            "preview_path": str(preview_path) if preview_path is not None else None,
            "animation_path": str(animation_path)
            if animation_path is not None
            else None,
            "image_paths": [str(path) for path in image_paths],
            "video_path": str(video_path) if video_path is not None else None,
            **metadata,
        }
        with self._media_manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _table_image(self, path: Path) -> Any | None:
        if self._wandb is None or not path.exists():
            return None
        return self._wandb.Image(str(path))

    def _table_video(self, path: Path | None) -> Any | None:
        if self._wandb is None or path is None or not path.exists():
            return None
        return self._wandb.Video(
            str(path), format=path.suffix.lower().lstrip(".") or None
        )


def _mean(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(sum(items) / len(items), 4)


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return normalized.strip("._") or "scene"
