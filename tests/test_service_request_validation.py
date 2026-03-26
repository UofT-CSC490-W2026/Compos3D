"""Service-layer request tests for defaults and call forwarding.

Default request fields and exact forwarding from service wrappers to engine
entrypoints (including evaluate).

Edge cases covered:
- `TrainingRequest`/`InferenceRequest` default values.
- Correct argument forwarding from service wrappers to engine functions.
- `EvaluateRequest` forwarding to evaluation engine entrypoint.

Expected outcomes:
- Wrapper functions pass through all expected fields unchanged.
- Default knobs (e.g., top_k/render flags) are preserved when unspecified.
- Monkeypatched engine functions receive expected values exactly once.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_stub_data_pkg = types.ModuleType("compos3d.data")
_stub_data_dataset = types.ModuleType("compos3d.data.dataset")
_stub_data_dataset.load_training_dataset = lambda *_args, **_kwargs: None
sys.modules.setdefault("compos3d.data", _stub_data_pkg)
sys.modules.setdefault("compos3d.data.dataset", _stub_data_dataset)

import compos3d.evaluation.service as eval_service
import compos3d.hypothesis.service as hyp_service


def test_training_request_defaults_and_forwarding(monkeypatch, tmp_path) -> None:
    seen = {}

    def _fake_train_vertical_slice(**kwargs):
        seen.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(hyp_service, "train_vertical_slice", _fake_train_vertical_slice)

    req = hyp_service.TrainingRequest(
        dataset_path=tmp_path / "dataset.json",
        output_dir=tmp_path / "out",
        experiment_name="exp-a",
    )
    out = hyp_service.train_hypotheses(req)
    assert out == {"ok": True}
    assert seen["dataset_path"] == req.dataset_path
    assert seen["experiment_name"] == "exp-a"
    assert seen["top_k"] == 2


def test_inference_request_defaults_and_forwarding(monkeypatch, tmp_path) -> None:
    seen = {}

    def _fake_run_vertical_inference(**kwargs):
        seen.update(kwargs)
        return {"done": True}

    monkeypatch.setattr(
        hyp_service, "run_vertical_inference", _fake_run_vertical_inference
    )

    req = hyp_service.InferenceRequest(
        bank_path=tmp_path / "bank.json",
        prompt="hello",
        output_dir=tmp_path / "out",
    )
    out = hyp_service.run_frozen_inference(req)
    assert out == {"done": True}
    assert seen["prompt"] == "hello"
    assert seen["top_k"] == 2
    assert seen["render_scene"] is False


def test_evaluate_request_forwarding(monkeypatch, tmp_path) -> None:
    seen = {}

    def _fake_evaluate_prediction_dir(*, predictions_dir: Path, output_dir: Path):
        seen["predictions_dir"] = predictions_dir
        seen["output_dir"] = output_dir
        return {"metric": 1.0}

    monkeypatch.setattr(
        eval_service, "evaluate_prediction_dir", _fake_evaluate_prediction_dir
    )

    req = eval_service.EvaluateRequest(
        predictions_dir=tmp_path / "preds",
        output_dir=tmp_path / "eval",
    )
    out = eval_service.evaluate_run(req)
    assert out == {"metric": 1.0}
    assert seen["predictions_dir"] == req.predictions_dir
    assert seen["output_dir"] == req.output_dir

