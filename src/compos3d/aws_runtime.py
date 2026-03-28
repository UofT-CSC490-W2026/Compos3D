from __future__ import annotations

import argparse
import json
import mimetypes
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import boto3
from botocore.exceptions import ClientError

from compos3d.app_config import AppConfig, load_app_config
from compos3d.storage.paths import training_checkpoint_prefix

_SUPPORTED_REMOTE_INPUT_FLAGS = (
    "--dataset-path",
    "--bank-path",
    "--config-path",
)
_WORK_DIR = Path("/work")
_SYNC_INTERVAL_SECONDS = 60
_IMDS_TOKEN_TTL_SECONDS = "21600"


@dataclass(frozen=True)
class SecretBinding:
    env_var: str
    secret_id: str
    version_id: str | None


@dataclass(frozen=True)
class RuntimePaths:
    output_dir: Path | None
    run_dir: Path | None
    checkpoint_uri: str | None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Container-side Compos3D runtime wrapper for AWS jobs."
    )
    parser.add_argument(
        "--runtime-env",
        dest="runtime_env",
        default=os.environ.get("COMPOS3D_ENV", "dev"),
        help="Application environment name used for app config and secret lookup.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=_WORK_DIR,
        help="Writable work directory inside the container.",
    )
    parser.add_argument(
        "--instance-type",
        default=os.environ.get("COMPOS3D_INSTANCE_TYPE"),
        help="Optional EC2 instance type for provenance and manifest wiring.",
    )
    parser.add_argument(
        "--sync-interval-seconds",
        type=int,
        default=_SYNC_INTERVAL_SECONDS,
        help="Checkpoint sync interval for long-running training jobs.",
    )
    parser.add_argument(
        "command",
        help="Inner compos3d command to execute.",
    )
    parser.add_argument(
        "cli_args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass through to the inner compos3d command.",
    )
    return parser.parse_args(argv)


def _normalize_cli_args(cli_args: list[str]) -> list[str]:
    if cli_args and cli_args[0] == "--":
        return cli_args[1:]
    return cli_args


def _option_value(args: list[str], flag: str) -> str | None:
    for idx, item in enumerate(args):
        if item == flag:
            if idx + 1 < len(args):
                return args[idx + 1]
            return None
        if item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
    return None


def _flag_present(args: list[str], flag: str) -> bool:
    return any(item == flag or item.startswith(f"{flag}=") for item in args)


def _replace_option(args: list[str], flag: str, value: str) -> list[str]:
    updated = list(args)
    for idx, item in enumerate(updated):
        if item == flag:
            if idx + 1 < len(updated):
                updated[idx + 1] = value
            else:
                updated.append(value)
            return updated
        if item.startswith(f"{flag}="):
            updated[idx] = f"{flag}={value}"
            return updated
    updated.extend([flag, value])
    return updated


def _guess_content_type(path: Path) -> str:
    guessed, _encoding = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got '{uri}'")
    remainder = uri[5:]
    bucket, separator, key = remainder.partition("/")
    if not bucket or not separator or not key:
        raise ValueError(f"Malformed s3:// URI: '{uri}'")
    return bucket, key


def _s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _resolve_runtime_paths(command: str, cli_args: list[str], app_cfg: AppConfig) -> RuntimePaths:
    output_dir_value = _option_value(cli_args, "--output-dir")
    if output_dir_value:
        output_dir = (
            (_repo_root() / output_dir_value).resolve()
            if not Path(output_dir_value).is_absolute()
            else Path(output_dir_value).resolve()
        )
    elif command == "run-inference":
        output_dir = (_repo_root() / "artifacts" / "inference").resolve()
    else:
        output_dir = None
    if command != "train-hypotheses":
        return RuntimePaths(output_dir=output_dir, run_dir=output_dir, checkpoint_uri=None)

    if output_dir is None:
        output_dir = (_repo_root() / "artifacts" / "training").resolve()
    experiment_name = _option_value(cli_args, "--experiment-name") or "vertical_slice"
    run_dir = output_dir / experiment_name
    if not app_cfg.s3_bucket_bronze:
        return RuntimePaths(output_dir=output_dir, run_dir=run_dir, checkpoint_uri=None)
    prefix = app_cfg.s3_prefix.strip("/")
    checkpoint_key = training_checkpoint_prefix(experiment_name)
    if prefix:
        checkpoint_key = f"{prefix}/{checkpoint_key}"
    checkpoint_uri = _s3_uri(app_cfg.s3_bucket_bronze, checkpoint_key)
    return RuntimePaths(output_dir=output_dir, run_dir=run_dir, checkpoint_uri=checkpoint_uri)


def _download_if_s3(uri: str, *, target_dir: Path, s3_client: Any, label: str) -> Path:
    bucket, key = _parse_s3_uri(uri)
    suffix = Path(key).suffix
    target_path = target_dir / f"{label}{suffix or '.json'}"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, key, str(target_path))
    return target_path


def _hydrate_remote_inputs(cli_args: list[str], *, target_dir: Path, s3_client: Any) -> tuple[list[str], dict[str, str]]:
    updated = list(cli_args)
    resolved_paths: dict[str, str] = {}
    for flag in _SUPPORTED_REMOTE_INPUT_FLAGS:
        current = _option_value(updated, flag)
        if not current or not current.startswith("s3://"):
            continue
        local_path = _download_if_s3(
            current,
            target_dir=target_dir,
            s3_client=s3_client,
            label=flag.lstrip("-").replace("-", "_"),
        )
        updated = _replace_option(updated, flag, str(local_path))
        resolved_paths[flag] = str(local_path)
    return updated, resolved_paths


def _list_s3_objects(s3_client: Any, *, bucket: str, prefix: str) -> list[dict[str, Any]]:
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    out: list[dict[str, Any]] = []
    for page in pages:
        out.extend(page.get("Contents", []))
    return out


def _download_s3_prefix(s3_client: Any, *, source_uri: str, destination_dir: Path) -> int:
    bucket, prefix = _parse_s3_uri(source_uri)
    objects = _list_s3_objects(s3_client, bucket=bucket, prefix=prefix.rstrip("/") + "/")
    if not objects:
        return 0
    downloaded = 0
    for item in objects:
        key = item["Key"]
        rel = key[len(prefix.rstrip("/") + "/") :]
        if not rel:
            continue
        local_path = destination_dir / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, key, str(local_path))
        downloaded += 1
    return downloaded


def _upload_directory(s3_client: Any, *, source_dir: Path, destination_uri: str) -> int:
    if not source_dir.exists():
        return 0
    bucket, prefix = _parse_s3_uri(destination_uri)
    prefix = prefix.rstrip("/")
    uploaded = 0
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source_dir).as_posix()
        key = f"{prefix}/{rel}" if prefix else rel
        s3_client.upload_file(
            str(path),
            bucket,
            key,
            ExtraArgs={"ContentType": _guess_content_type(path)},
        )
        uploaded += 1
    return uploaded


class _IMDSv2:
    def __init__(self) -> None:
        self._token: str | None = None
        self._token_deadline = 0.0

    def _refresh_token(self) -> str:
        req = urllib_request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": _IMDS_TOKEN_TTL_SECONDS},
        )
        with urllib_request.urlopen(req, timeout=2) as response:  # noqa: S310
            token = response.read().decode()
        self._token = token
        self._token_deadline = time.time() + int(_IMDS_TOKEN_TTL_SECONDS) - 30
        return token

    def get(self, path: str) -> str | None:
        if self._token is None or time.time() >= self._token_deadline:
            try:
                self._refresh_token()
            except Exception:  # noqa: BLE001
                return None
        req = urllib_request.Request(
            f"http://169.254.169.254/latest/{path.lstrip('/')}",
            headers={"X-aws-ec2-metadata-token": self._token or ""},
        )
        try:
            with urllib_request.urlopen(req, timeout=2) as response:  # noqa: S310
                return response.read().decode()
        except urllib_error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        except Exception:  # noqa: BLE001
            return None


class CheckpointSyncManager:
    def __init__(
        self,
        *,
        s3_client: Any,
        local_dir: Path | None,
        remote_uri: str | None,
        interval_seconds: int,
    ) -> None:
        self.s3_client = s3_client
        self.local_dir = local_dir
        self.remote_uri = remote_uri
        self.interval_seconds = max(15, interval_seconds)
        self._metadata = _IMDSv2()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.local_dir is None or self.remote_uri is None:
            return
        self._thread = threading.Thread(target=self._run, name="checkpoint-sync", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def sync_once(self) -> int:
        if self.local_dir is None or self.remote_uri is None:
            return 0
        with self._lock:
            return _upload_directory(
                self.s3_client,
                source_dir=self.local_dir,
                destination_uri=self.remote_uri,
            )

    def restore(self) -> int:
        if self.local_dir is None or self.remote_uri is None:
            return 0
        self.local_dir.mkdir(parents=True, exist_ok=True)
        return _download_s3_prefix(
            self.s3_client,
            source_uri=self.remote_uri,
            destination_dir=self.local_dir,
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.sync_once()
            if self._spot_interruption_pending():
                self.sync_once()
                self._stop.set()

    def _spot_interruption_pending(self) -> bool:
        try:
            payload = self._metadata.get("meta-data/spot/instance-action")
        except Exception:  # noqa: BLE001
            return False
        return bool(payload)


def _hydrate_secrets(app_cfg: AppConfig, *, secrets_client: Any) -> list[SecretBinding]:
    bindings: list[SecretBinding] = []
    for env_var, secret_id in sorted(app_cfg.aws_secret_env_map.items()):
        try:
            response = secrets_client.get_secret_value(SecretId=secret_id)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code == "ResourceNotFoundException":
                print(
                    f"[aws_runtime] Secret '{secret_id}' has no current value; skipping {env_var}"
                )
                continue
            raise
        secret_value = response.get("SecretString")
        if secret_value is None and response.get("SecretBinary") is not None:
            secret_value = response["SecretBinary"].decode()
        if secret_value is None:
            continue
        os.environ[env_var] = secret_value
        bindings.append(
            SecretBinding(
                env_var=env_var,
                secret_id=secret_id,
                version_id=response.get("VersionId"),
            )
        )
    return bindings


def _write_runtime_context(
    *,
    target_dir: Path | None,
    command: str,
    cli_args: list[str],
    resolved_inputs: dict[str, str],
    bindings: list[SecretBinding],
    checkpoint_uri: str | None,
    instance_type: str | None,
    runtime_env: str,
) -> Path | None:
    if target_dir is None:
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "runtime_env": runtime_env,
        "command": command,
        "cli_args": cli_args,
        "resolved_inputs": resolved_inputs,
        "checkpoint_uri": checkpoint_uri,
        "instance_type": instance_type,
        "secret_bindings": [
            {
                "env_var": binding.env_var,
                "secret_id": binding.secret_id,
                "version_id": binding.version_id,
            }
            for binding in bindings
        ],
        "generated_at_unix": time.time(),
    }
    out = target_dir / "runtime_context.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out


def _install_shutdown_handlers(sync_manager: CheckpointSyncManager) -> None:
    def _handler(signum: int, _frame: Any) -> None:
        sync_manager.sync_once()
        sync_manager.stop()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def _run_inner_cli(
    *,
    command: str,
    cli_args: list[str],
    instance_type: str | None,
) -> int:
    passthrough = list(cli_args)
    if instance_type:
        passthrough = _replace_option(passthrough, "--instance-type", instance_type)
    passthrough = _replace_option(passthrough, "--compute-platform", "aws_ec2")
    cmd = [sys.executable, "-m", "compos3d.cli", command, *passthrough]
    completed = subprocess.run(cmd, cwd=str(_repo_root()), check=False)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cli_args = _normalize_cli_args(list(args.cli_args))
    app_cfg = load_app_config(args.runtime_env)
    os.environ["COMPOS3D_ENV"] = args.runtime_env
    os.environ.setdefault("COMPOS3D_BEDROCK_REGION", app_cfg.aws_region)
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    s3_client = boto3.client("s3", region_name=app_cfg.aws_region)
    secrets_client = boto3.client("secretsmanager", region_name=app_cfg.aws_region)
    runtime_paths = _resolve_runtime_paths(args.command, cli_args, app_cfg)
    inputs_dir = work_dir / "inputs"
    hydrated_args, resolved_inputs = _hydrate_remote_inputs(
        cli_args,
        target_dir=inputs_dir,
        s3_client=s3_client,
    )

    sync_manager = CheckpointSyncManager(
        s3_client=s3_client,
        local_dir=runtime_paths.run_dir,
        remote_uri=runtime_paths.checkpoint_uri,
        interval_seconds=args.sync_interval_seconds,
    )
    if args.command == "train-hypotheses" and _flag_present(hydrated_args, "--resume"):
        restored = sync_manager.restore()
        print(f"[aws_runtime] Restored {restored} checkpoint artifacts")

    secret_bindings = _hydrate_secrets(app_cfg, secrets_client=secrets_client)
    _write_runtime_context(
        target_dir=runtime_paths.run_dir or runtime_paths.output_dir,
        command=args.command,
        cli_args=hydrated_args,
        resolved_inputs=resolved_inputs,
        bindings=secret_bindings,
        checkpoint_uri=runtime_paths.checkpoint_uri,
        instance_type=args.instance_type,
        runtime_env=args.runtime_env,
    )

    _install_shutdown_handlers(sync_manager)
    sync_manager.start()
    try:
        exit_code = _run_inner_cli(
            command=args.command,
            cli_args=hydrated_args,
            instance_type=args.instance_type,
        )
    finally:
        sync_manager.sync_once()
        sync_manager.stop()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
