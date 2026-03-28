from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from compos3d.config import LoggingConfig
from compos3d.evaluation.critic import HeuristicSceneCritic
from compos3d.hypothesis.loop import HypothesisLoopConfig, SceneHypothesisLoop
from compos3d.models import SceneProgram, TrainingDataset, TrainingExample


class _DeterministicLLM:
    def generate_hypotheses(
        self,
        room_type: str,
        examples: list[TrainingExample],
        *,
        num_hypotheses: int,
        focus: str,
    ) -> list[str]:
        return [f"{focus} rule {i} for {room_type}" for i in range(num_hypotheses)]

    def generate_scene_program(
        self, *, prompt: str, room_type: str, selected_hypotheses: list[str]
    ) -> SceneProgram:
        assets = []
        lowered = prompt.lower()
        for asset in ("bed", "lamp", "rug", "dining_table", "chair", "sofa"):
            if asset in lowered:
                assets.append({"asset_type": asset, "count": 1})
        return SceneProgram(
            prompt=prompt,
            room_type=room_type,
            hypotheses=selected_hypotheses,
            assets=assets,
        )


def _dataset() -> TrainingDataset:
    return TrainingDataset(
        dataset_id="resume_ds",
        examples=[
            TrainingExample(
                example_id="e1",
                room_type="bedroom",
                prompt="bedroom with a bed",
                required_assets=["bed"],
            ),
            TrainingExample(
                example_id="e2",
                room_type="bedroom",
                prompt="bedroom with a bed and lamp",
                required_assets=["bed", "lamp"],
            ),
            TrainingExample(
                example_id="e3",
                room_type="bedroom",
                prompt="bedroom with a bed, lamp, and rug",
                required_assets=["bed", "lamp", "rug"],
            ),
        ],
    )


def _make_loop(
    *,
    run_dir: Path,
    resume: bool = False,
    logging_config: LoggingConfig | None = None,
    renderer=None,
) -> SceneHypothesisLoop:
    return SceneHypothesisLoop(
        dataset=_dataset(),
        llm=_DeterministicLLM(),
        critic=HeuristicSceneCritic(),
        run_dir=run_dir,
        experiment_name="resume_test",
        llm_provider="mock",
        config=HypothesisLoopConfig(
            num_init_examples_per_room=1,
            init_hypotheses_per_room=2,
            k=1,
            num_epochs=2,
            save_every_n_examples=1,
        ),
        logging_config=logging_config,
        experiment_config={"generator": {"provider": "mock"}},
        renderer=renderer,
        resume=resume,
    )


def test_training_resume_continues_from_saved_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interrupted_run_dir = tmp_path / "interrupted"
    loop = _make_loop(run_dir=interrupted_run_dir)
    original_persist = loop._persist_progress
    state = {"raised": False}

    def _interrupting_persist(*, next_epoch: int, next_position: int) -> None:
        original_persist(next_epoch=next_epoch, next_position=next_position)
        if loop.training_trace and not state["raised"]:
            state["raised"] = True
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(loop, "_persist_progress", _interrupting_persist)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        loop.train()

    assert (interrupted_run_dir / "resume_state.json").exists()
    resumed = _make_loop(run_dir=interrupted_run_dir, resume=True).train()
    fresh = _make_loop(run_dir=tmp_path / "fresh").train()

    assert resumed["num_predictions"] == fresh["num_predictions"]
    assert resumed["metrics"] == fresh["metrics"]
    resumed_bank = json.loads(
        (interrupted_run_dir / "hypothesis_bank.json").read_text()
    )
    fresh_bank = json.loads((tmp_path / "fresh" / "hypothesis_bank.json").read_text())
    assert resumed_bank == fresh_bank


def test_training_logs_to_wandb_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logs: list[tuple[dict, int | None]] = []
    artifacts: list[object] = []

    def _renderer(_scene_program: SceneProgram, render_dir: Path) -> list[Path]:
        views_dir = render_dir / "views"
        views_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for name, color in (
            ("view_overhead.png", (255, 0, 0)),
            ("view_front.png", (0, 255, 0)),
            ("view_left.png", (0, 0, 255)),
            ("view_right.png", (255, 255, 0)),
        ):
            out = views_dir / name
            Image.new("RGB", (12, 12), color=color).save(out)
            paths.append(out)
        (render_dir / "build_manifest.json").write_text(
            json.dumps({"rendered_views": [str(path) for path in paths]})
        )
        return paths

    class _FakeTable:
        def __init__(self, columns):
            self.columns = columns
            self.rows = []

        def add_data(self, *row):
            self.rows.append(row)

    class _FakeArtifact:
        def __init__(self, name, type, metadata=None):
            self.name = name
            self.type = type
            self.metadata = metadata
            self.files = []

        def add_file(self, path, name=None):
            self.files.append((path, name))

    class _FakeImage:
        def __init__(self, path, **kwargs):
            self.path = path
            self.kwargs = kwargs

    class _FakeVideo:
        def __init__(self, path, **kwargs):
            self.path = path
            self.kwargs = kwargs

    class _FakeRun:
        def __init__(self):
            self.id = "wandb-test-id"
            self.finished = False
            self.defined_metrics = []

        def log(self, payload, step=None):
            logs.append((payload, step))

        def log_artifact(self, artifact):
            artifacts.append(artifact)

        def define_metric(self, *args, **kwargs):
            self.defined_metrics.append((args, kwargs))

        def finish(self):
            self.finished = True

    class _FakeWandb:
        __version__ = "0.0-test"
        last_run = None

        @staticmethod
        def init(**kwargs):
            _FakeWandb.last_run = _FakeRun()
            return _FakeWandb.last_run

        Table = _FakeTable
        Artifact = _FakeArtifact
        Image = _FakeImage
        Video = _FakeVideo

    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb)

    logging_config = LoggingConfig(
        enable_wandb=True,
        wandb_project="compos3d-test",
        wandb_mode="offline",
        log_every_n_examples=1,
        log_bank_table_every_n_examples=1,
        log_prediction_media_every_n_examples=1,
        log_hypothesis_eval_media_every_n_examples=1,
    )
    result = _make_loop(
        run_dir=tmp_path / "wandb_run",
        logging_config=logging_config,
        renderer=_renderer,
    ).train()

    assert result["num_predictions"] > 0
    assert logs
    assert any("train/combined_overall" in payload for payload, _ in logs)
    assert any("bank/latest" in payload for payload, _ in logs)
    assert any("init/preview" in payload for payload, _ in logs)
    assert any("pred/preview" in payload for payload, _ in logs)
    assert any("pred/animation" in payload for payload, _ in logs)
    assert any("tables/predictions" in payload for payload, _ in logs)
    assert any("tables/hypothesis_evals" in payload for payload, _ in logs)
    assert artifacts
    assert any(name == "resume_state.json" for _, name in artifacts[0].files)
    assert any(
        name == "wandb_media/media_manifest.jsonl" for _, name in artifacts[0].files
    )
    assert _FakeWandb.last_run is not None and _FakeWandb.last_run.finished
    assert _FakeWandb.last_run.defined_metrics
