"""CLI wiring tests for environment-aware storage and AWS launch behavior.

Why these tests:
- Existing CLI tests focus on error handling, but not the branches that wire in
  environment-backed storage or the EC2 launcher.
- These flows are important user-facing use cases and should fail loudly if the
  request wiring changes.

Edge cases covered:
- `--env dev` on train/inference -> loads app config and injects a store.
- `--env local`/no env -> keeps local-only behavior.
- `launch-aws --wait` -> launches and then blocks on runner.wait().
"""

from __future__ import annotations

from typer.testing import CliRunner

import compos3d.app_config as app_config_module
import compos3d.cli as cli
import compos3d.compute.ec2_runner as ec2_runner_module
import compos3d.storage as storage_module


def test_train_and_inference_env_dev_load_store(monkeypatch) -> None:
    runner = CliRunner()
    seen = {"train": None, "infer": None}
    fake_cfg = app_config_module.AppConfig(
        env="dev",
        storage_backend="local",
        local_lake_root="virtual_lake",
    )
    fake_store = object()

    monkeypatch.setattr(app_config_module, "load_app_config", lambda env: fake_cfg)
    monkeypatch.setattr(storage_module, "get_store", lambda cfg: fake_store)

    def _fake_train(request):
        seen["train"] = request
        return {"ok": True}

    def _fake_infer(request):
        seen["infer"] = request
        return {"ok": True}

    monkeypatch.setattr(cli, "train_hypotheses", _fake_train)
    monkeypatch.setattr(cli, "run_frozen_inference", _fake_infer)

    train_result = runner.invoke(
        cli.app,
        [
            "train-hypotheses",
            "--dataset-path",
            "dataset.json",
            "--output-dir",
            "out",
            "--experiment-name",
            "exp",
            "--env",
            "dev",
        ],
    )
    infer_result = runner.invoke(
        cli.app,
        [
            "run-inference",
            "--bank-path",
            "bank.json",
            "--prompt",
            "a living room with a sofa",
            "--output-dir",
            "inf",
            "--env",
            "dev",
        ],
    )

    assert train_result.exit_code == 0
    assert infer_result.exit_code == 0
    assert seen["train"].store is fake_store
    assert seen["infer"].store is fake_store


def test_train_without_env_keeps_store_none(monkeypatch) -> None:
    runner = CliRunner()
    seen = {}

    def _fake_train(request):
        seen["request"] = request
        return {"ok": True}

    monkeypatch.setattr(cli, "train_hypotheses", _fake_train)

    result = runner.invoke(
        cli.app,
        [
            "train-hypotheses",
            "--dataset-path",
            "dataset.json",
            "--output-dir",
            "out",
            "--experiment-name",
            "exp",
        ],
    )

    assert result.exit_code == 0
    assert seen["request"].store is None


def test_launch_aws_wait_invokes_runner_launch_and_wait(monkeypatch) -> None:
    runner = CliRunner()
    fake_cfg = app_config_module.AppConfig(env="dev", storage_backend="local")
    seen = {}

    class _FakeRunner:
        def __init__(self, app_config):
            seen["app_config"] = app_config

        def launch(self, spec):
            seen["spec"] = spec
            return "i-123", {"instance_id": "i-123"}

        def wait(self, instance_id):
            seen["waited_for"] = instance_id
            return "terminated"

    monkeypatch.setattr(app_config_module, "load_app_config", lambda env: fake_cfg)
    monkeypatch.setattr(ec2_runner_module, "EC2JobRunner", _FakeRunner)

    result = runner.invoke(
        cli.app,
        [
            "launch-aws",
            "train-hypotheses",
            "--cli-args",
            "--dataset-path ds.json --output-dir out --env dev",
            "--env",
            "dev",
            "--wait",
        ],
    )

    assert result.exit_code == 0
    assert seen["app_config"] is fake_cfg
    assert seen["spec"].command == "train-hypotheses"
    assert seen["spec"].cli_args == [
        "--dataset-path",
        "ds.json",
        "--output-dir",
        "out",
        "--env",
        "dev",
    ]
    assert seen["waited_for"] == "i-123"
