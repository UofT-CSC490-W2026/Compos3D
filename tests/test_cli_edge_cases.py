"""CLI behavior tests for argument handling and graceful failures.

Exit codes, user-visible errors, and smoke paths for representative commands
with dependencies monkeypatched.

Edge cases covered:
- `_run_or_exit` returns non-zero on expected failures.
- Helpful error text is emitted for pending/exception cases.
- Smoke invocation of representative CLI commands with monkeypatched services.

Expected outcomes:
- Failing command handlers exit with code 1 and readable error messages.
- Successful commands complete with exit code 0.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

from typer.testing import CliRunner

# Test-local import shim for repo state where `compos3d.data.dataset` is absent.
_stub_data_pkg = types.ModuleType("compos3d.data")
_stub_data_dataset = types.ModuleType("compos3d.data.dataset")
_stub_data_dataset.load_training_dataset = lambda *_args, **_kwargs: None
sys.modules.setdefault("compos3d.data", _stub_data_pkg)
sys.modules.setdefault("compos3d.data.dataset", _stub_data_dataset)

import compos3d.cli as cli
from compos3d._stages import StagePendingError


def test_run_or_exit_stage_pending_returns_exit1(monkeypatch) -> None:
    r = CliRunner()

    def _pending(*_a, **_k):
        raise StagePendingError("not implemented")

    monkeypatch.setattr(cli, "run_backend_smoke", _pending)
    result = r.invoke(cli.app, ["backend-smoke"])
    assert result.exit_code == 1
    assert "not implemented" in result.output


def test_run_or_exit_generic_exception_returns_exit1(monkeypatch) -> None:
    r = CliRunner()
    monkeypatch.setattr(cli, "evaluate_run", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("bad input")))
    result = r.invoke(
        cli.app,
        ["evaluate", "--predictions-dir", "x", "--output-dir", "y"],
    )
    assert result.exit_code == 1
    assert "bad input" in result.output


def test_cli_smoke_train_and_inference_commands(monkeypatch, tmp_path) -> None:
    r = CliRunner()
    monkeypatch.setattr(cli, "train_hypotheses", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(cli, "run_frozen_inference", lambda *_a, **_k: {"ok": True})

    train = r.invoke(
        cli.app,
        [
            "train-hypotheses",
            "--dataset-path",
            str(tmp_path / "d.json"),
            "--output-dir",
            str(tmp_path / "out"),
            "--experiment-name",
            "exp",
        ],
    )
    infer = r.invoke(
        cli.app,
        [
            "run-inference",
            "--bank-path",
            str(tmp_path / "bank.json"),
            "--prompt",
            "bedroom prompt",
            "--output-dir",
            str(tmp_path / "inf"),
        ],
    )
    assert train.exit_code == 0
    assert infer.exit_code == 0

