from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from compos3d.app_config import AppConfig
from compos3d.aws_runtime import (
    CheckpointSyncManager,
    _hydrate_remote_inputs,
    _hydrate_secrets,
    _resolve_runtime_paths,
    main,
)


class _FakePaginator:
    def __init__(self, s3: "_FakeS3"):
        self.s3 = s3

    def paginate(self, **kwargs):
        bucket = kwargs["Bucket"]
        prefix = kwargs["Prefix"]
        return [
            {
                "Contents": [
                    {"Key": key}
                    for file_bucket, key in sorted(self.s3.files)
                    if file_bucket == bucket and key.startswith(prefix)
                ]
            }
        ]


class _FakeS3:
    def __init__(self) -> None:
        self.files: dict[tuple[str, str], bytes] = {}
        self.uploads: list[tuple[str, str, str]] = []

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        Path(filename).write_bytes(self.files[(bucket, key)])

    def upload_file(self, filename: str, bucket: str, key: str, ExtraArgs=None) -> None:  # noqa: N803
        self.uploads.append((filename, bucket, key))
        self.files[(bucket, key)] = Path(filename).read_bytes()

    def get_paginator(self, _name: str):
        return _FakePaginator(self)


class _FakeSecrets:
    def get_secret_value(self, SecretId: str):  # noqa: N803
        return {"SecretString": f"value-for-{SecretId}", "VersionId": "v1"}


class _FakeSecretsMissingOptional:
    def get_secret_value(self, SecretId: str):  # noqa: N803
        if SecretId == "compos3d-dev-openai-key":
            raise ClientError(
                {
                    "Error": {
                        "Code": "ResourceNotFoundException",
                        "Message": "missing AWSCURRENT",
                    }
                },
                "GetSecretValue",
            )
        return {"SecretString": f"value-for-{SecretId}", "VersionId": "v1"}


def test_hydrate_remote_inputs_downloads_supported_s3_paths(tmp_path: Path) -> None:
    s3 = _FakeS3()
    s3.files[("bucket", "datasets/train.json")] = b'{"dataset_id": "d", "examples": []}'

    cli_args, resolved = _hydrate_remote_inputs(
        ["--dataset-path", "s3://bucket/datasets/train.json"],
        target_dir=tmp_path,
        s3_client=s3,
    )

    assert cli_args[1].endswith("dataset_path.json")
    assert Path(cli_args[1]).exists()
    assert resolved["--dataset-path"] == cli_args[1]


def test_hydrate_secrets_sets_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    bindings = _hydrate_secrets(
        AppConfig(
            env="dev",
            storage_backend="s3",
            s3_bucket_bronze="bronze",
            s3_bucket_silver="silver",
            s3_bucket_gold="gold",
            aws_secret_env_map={"WANDB_API_KEY": "compos3d-dev-wandb-key"},
        ),
        secrets_client=_FakeSecrets(),
    )

    assert bindings[0].env_var == "WANDB_API_KEY"
    assert bindings[0].secret_id == "compos3d-dev-wandb-key"
    assert bindings[0].version_id == "v1"


def test_hydrate_secrets_skips_missing_optional_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    bindings = _hydrate_secrets(
        AppConfig(
            env="dev",
            storage_backend="s3",
            s3_bucket_bronze="bronze",
            s3_bucket_silver="silver",
            s3_bucket_gold="gold",
            aws_secret_env_map={
                "WANDB_API_KEY": "compos3d-dev-wandb-key",
                "OPENAI_API_KEY": "compos3d-dev-openai-key",
            },
        ),
        secrets_client=_FakeSecretsMissingOptional(),
    )

    assert [binding.env_var for binding in bindings] == ["WANDB_API_KEY"]
    assert os.environ["WANDB_API_KEY"] == "value-for-compos3d-dev-wandb-key"
    assert "OPENAI_API_KEY" not in os.environ


def test_resolve_runtime_paths_training_uses_checkpoint_prefix() -> None:
    cfg = AppConfig(
        env="dev",
        storage_backend="s3",
        s3_bucket_bronze="compos3d-dev-bronze",
        s3_bucket_silver="compos3d-dev-silver",
        s3_bucket_gold="compos3d-dev-gold",
        s3_prefix="compos3d",
    )

    paths = _resolve_runtime_paths(
        "train-hypotheses",
        ["--output-dir", "artifacts/training", "--experiment-name", "aws_smoke"],
        cfg,
    )

    assert paths.run_dir is not None
    assert paths.run_dir.as_posix().endswith("artifacts/training/aws_smoke")
    assert (
        paths.checkpoint_uri
        == "s3://compos3d-dev-bronze/compos3d/bronze/checkpoints/training/aws_smoke/latest"
    )


def test_checkpoint_sync_manager_restores_and_uploads(tmp_path: Path) -> None:
    s3 = _FakeS3()
    s3.files[("bucket", "prefix/latest/resume_state.json")] = b'{"ok": true}'
    run_dir = tmp_path / "run"

    sync = CheckpointSyncManager(
        s3_client=s3,
        local_dir=run_dir,
        remote_uri="s3://bucket/prefix/latest",
        interval_seconds=15,
    )

    restored = sync.restore()
    assert restored == 1
    assert json.loads((run_dir / "resume_state.json").read_text()) == {"ok": True}

    (run_dir / "metrics.json").write_text('{"score": 1}')
    uploaded = sync.sync_once()
    assert uploaded >= 2
    assert any(key.endswith("metrics.json") for _fn, _bucket, key in s3.uploads)


def test_runtime_main_hydrates_s3_inputs_and_writes_runtime_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "artifacts").mkdir()

    cfg = AppConfig(
        env="dev",
        storage_backend="s3",
        s3_bucket_bronze="bronze-b",
        s3_bucket_silver="silver-b",
        s3_bucket_gold="gold-b",
        s3_prefix="compos3d",
        aws_secret_env_map={"WANDB_API_KEY": "compos3d-dev-wandb-key"},
    )
    s3 = _FakeS3()
    s3.files[("bronze-b", "dataset.json")] = b'{"dataset_id": "d", "examples": []}'
    secrets = _FakeSecrets()
    seen: dict[str, object] = {}

    monkeypatch.setattr("compos3d.aws_runtime._repo_root", lambda: repo_root)
    monkeypatch.setattr("compos3d.aws_runtime.load_app_config", lambda env: cfg)

    def _fake_boto_client(name: str, region_name: str | None = None):  # noqa: ARG001
        if name == "s3":
            return s3
        if name == "secretsmanager":
            return secrets
        raise AssertionError(name)

    monkeypatch.setattr("compos3d.aws_runtime.boto3.client", _fake_boto_client)

    def _fake_run_inner_cli(
        *, command: str, cli_args: list[str], instance_type: str | None
    ):
        seen["command"] = command
        seen["cli_args"] = cli_args
        seen["instance_type"] = instance_type
        run_dir = repo_root / "artifacts" / "training" / "aws_smoke"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "resume_state.json").write_text("{}")
        return 0

    monkeypatch.setattr("compos3d.aws_runtime._run_inner_cli", _fake_run_inner_cli)

    exit_code = main(
        [
            "--runtime-env",
            "dev",
            "--work-dir",
            str(tmp_path / "work"),
            "train-hypotheses",
            "--dataset-path",
            "s3://bronze-b/dataset.json",
            "--output-dir",
            "artifacts/training",
            "--experiment-name",
            "aws_smoke",
        ]
    )

    assert exit_code == 0
    assert seen["command"] == "train-hypotheses"
    cli_args = seen["cli_args"]
    assert isinstance(cli_args, list)
    assert any(str(arg).endswith("dataset_path.json") for arg in cli_args)
    runtime_context = json.loads(
        (
            repo_root / "artifacts" / "training" / "aws_smoke" / "runtime_context.json"
        ).read_text()
    )
    assert runtime_context["runtime_env"] == "dev"
    assert runtime_context["command"] == "train-hypotheses"
    assert runtime_context["secret_bindings"][0]["env_var"] == "WANDB_API_KEY"
