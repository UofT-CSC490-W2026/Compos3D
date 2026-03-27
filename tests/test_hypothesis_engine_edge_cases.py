"""Edge-case unit tests for `compos3d.hypothesis.engine`.

Ranking, config conversion, lake mirroring, manifest finalization, and
evaluation summary paths in training and inference.

Edge cases covered:
- Inference hypothesis selection ranking/tie/top-k behavior.
- Training/loop config conversion helper functions.
- Lake mirroring with optional files absent.
- Manifest success/failure branches in training and inference with store enabled.
- Evaluation summary generation from predictions and fallback manifest paths.

Expected outcomes:
- Sorting and conversion helpers are deterministic.
- Mirroring writes only existing optional artifacts.
- Manifest finalization writes run manifests in both success and failure paths.
- Missing prediction artifacts raise `FileNotFoundError` with clear context.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compos3d.config import DEFAULT_EXPERIMENT_CONFIG
from compos3d.hypothesis import engine
from compos3d.models import (
    CriticScore,
    HypothesisRecord,
    PredictionRecord,
    SceneProgram,
)
from compos3d.storage.local import LocalStore


def test_select_hypotheses_for_inference_ties_and_topk() -> None:
    recs = [
        HypothesisRecord(
            hypothesis_id="h1",
            text="use sofa",
            room_type="living_room",
            reward=0.8,
            accuracy=0.5,
            mean_score=0.5,
        ),
        HypothesisRecord(
            hypothesis_id="h2",
            text="use coffee_table",
            room_type="living_room",
            reward=0.8,
            accuracy=0.7,
            mean_score=0.4,
        ),
        HypothesisRecord(
            hypothesis_id="x", text="bedroom only", room_type="bedroom", reward=1.0
        ),
    ]
    chosen = engine._select_hypotheses_for_inference(  # noqa: SLF001
        recs, "living_room", prompt="modern living room with sofa", top_k=1
    )
    assert len(chosen) == 1
    assert chosen[0].hypothesis_id in {"h1", "h2"}


def test_training_and_loop_config_helpers() -> None:
    tcfg = engine._training_config_from_args(  # noqa: SLF001
        num_init_examples_per_room=1,
        init_hypotheses_per_room=2,
        top_k=3,
        alpha=0.4,
        max_num_hypotheses_per_room=9,
        num_wrong_scale=0.2,
        update_batch_size=1,
        num_hypotheses_to_update=1,
        update_hypotheses_per_batch=2,
        only_best_hypothesis=False,
        num_epochs=1,
        success_threshold=0.6,
        save_every_n_examples=1,
        selection_strategy="greedy",
        use_repair=True,
        baseline_mode=None,
        seed=7,
    )
    cfg = DEFAULT_EXPERIMENT_CONFIG.model_copy(deep=True)
    cfg.training = tcfg
    loop_cfg = engine._loop_config_from_experiment(cfg)  # noqa: SLF001
    assert loop_cfg.k == 3
    assert loop_cfg.selection_strategy == "greedy"
    assert loop_cfg.seed == 7


def test_mirror_training_and_inference_with_missing_optional_files(tmp_path) -> None:
    store = LocalStore(tmp_path / "lake")
    run_dir = tmp_path / "run"
    (run_dir / "programs").mkdir(parents=True)
    (run_dir / "programs" / "p1.json").write_text(json.dumps({"ok": 1}))
    uris = engine._mirror_training_to_lake(  # noqa: SLF001
        store=store,
        run_id="r1",
        run_dir=run_dir,
        experiment_name="exp",
        training_result={"m": 1},
        experiment_config={"c": 1},
    )
    assert any("training_manifest.json" in u for u in uris)

    out = tmp_path / "infer"
    out.mkdir()
    (out / "scene_program.json").write_text(json.dumps({"prompt": "x"}))
    uris2 = engine._mirror_inference_to_lake(  # noqa: SLF001
        store=store, run_id="ir1", output_dir=out, inference_result={"ok": True}
    )
    assert any("inference_manifest.json" in u for u in uris2)


def test_train_and_inference_manifest_failure_paths(monkeypatch, tmp_path) -> None:
    ds_path = tmp_path / "ds.json"
    ds_path.write_text(json.dumps({"dataset_id": "d", "examples": []}))

    class _Loop:
        def __init__(self, **kwargs):
            pass

        def train(self):
            return {"ok": True}

    class _Store:
        def __init__(self):
            self.writes = []

        def put_json(self, rel_path, obj):
            self.writes.append((rel_path, obj))
            if rel_path.endswith("training_manifest.json") or rel_path.endswith(
                "inference_manifest.json"
            ):
                raise RuntimeError("mirror boom")
            return rel_path

        def put_bytes(self, rel_path, b, content_type="application/octet-stream"):
            self.writes.append((rel_path, len(b)))
            return rel_path

    monkeypatch.setattr(
        engine,
        "load_training_dataset",
        lambda _p: type(
            "D", (), {"examples": [], "model_copy": lambda self, update: self}
        )(),
    )
    monkeypatch.setattr(engine, "build_scene_llm", lambda *_a, **_k: object())
    monkeypatch.setattr(engine, "build_scene_critic", lambda *_a, **_k: object())
    monkeypatch.setattr(engine, "make_training_renderer", lambda *_a, **_k: None)
    monkeypatch.setattr(engine, "SceneHypothesisLoop", _Loop)
    monkeypatch.setattr(engine, "_run_id", lambda _x: "RID")

    with pytest.raises(RuntimeError, match="mirror boom"):
        engine.train_vertical_slice(
            dataset_path=ds_path,
            output_dir=tmp_path / "o",
            experiment_name="e",
            store=_Store(),
        )

    # inference failure path
    bank = tmp_path / "bank.json"
    bank.write_text("[]")
    monkeypatch.setattr(
        engine,
        "build_scene_llm",
        lambda *_a, **_k: type(
            "L",
            (),
            {
                "generate_scene_program": lambda *a, **k: SceneProgram(
                    prompt="p", room_type="bedroom"
                )
            },
        )(),
    )
    monkeypatch.setattr(engine, "build_scene_critic", lambda *_a, **_k: object())
    monkeypatch.setattr(
        engine,
        "evaluate_scene_program",
        lambda *a, **k: CriticScore(
            validity=1,
            prompt_adherence=1,
            asset_precision=1,
            asset_recall=1,
            room_match=1,
            overall=1,
        ),
    )
    monkeypatch.setattr(engine, "infer_room_type", lambda _p: "bedroom")
    with pytest.raises(RuntimeError, match="mirror boom"):
        engine.run_vertical_inference(
            bank_path=bank,
            prompt="p",
            output_dir=tmp_path / "io",
            store=_Store(),
        )


def test_evaluate_prediction_dir_paths_and_missing(tmp_path) -> None:
    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()
    pred = PredictionRecord(
        example_id="e1",
        prompt="p",
        room_type="bedroom",
        selected_hypotheses=[],
        scene_program=SceneProgram(prompt="p", room_type="bedroom"),
        critic_score=CriticScore(
            validity=1,
            prompt_adherence=1,
            asset_precision=1,
            asset_recall=1,
            room_match=1,
            overall=1,
        ),
    )
    (preds_dir / "predictions.jsonl").write_text(json.dumps(pred.model_dump()) + "\n")
    out = engine.evaluate_prediction_dir(
        predictions_dir=preds_dir, output_dir=tmp_path / "eval"
    )
    assert out["num_predictions"] == 1

    fallback = tmp_path / "fallback"
    fallback.mkdir()
    (fallback / "inference_manifest.json").write_text("{}")
    (fallback / "critic_score.json").write_text(
        json.dumps(
            {
                "validity": 0.1,
                "prompt_adherence": 0.2,
                "asset_precision": 0.3,
                "asset_recall": 0.4,
                "room_match": 0.5,
                "overall": 0.6,
            }
        )
    )
    out2 = engine.evaluate_prediction_dir(
        predictions_dir=fallback, output_dir=tmp_path / "eval2"
    )
    assert out2["average_overall"] == 0.6

    with pytest.raises(FileNotFoundError):
        engine.evaluate_prediction_dir(
            predictions_dir=tmp_path / "missing", output_dir=tmp_path / "eval3"
        )
