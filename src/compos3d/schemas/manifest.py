"""Run manifest schema for reproducibility and provenance tracking."""

from __future__ import annotations

import subprocess
import sys
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class GitInfo(BaseModel):
    commit_sha: str
    branch: str
    is_dirty: bool = False
    remote_url: Optional[str] = None


class RunManifest(BaseModel):
    schema_version: Literal["v1"] = "v1"
    run_id: str
    run_type: Literal["training", "inference", "build_scene", "evaluate"]

    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    git_info: Optional[GitInfo] = None
    python_version: str
    package_versions: Dict[str, str] = Field(default_factory=dict)

    # LLM / model provenance
    generator_model: Optional[str] = None
    critic_model: Optional[str] = None

    # Inputs / outputs
    input_paths: List[str] = Field(default_factory=list)
    output_uris: List[str] = Field(default_factory=list)

    # Execution environment
    compute_platform: Literal["local", "aws_ec2", "other"] = "local"
    instance_type: Optional[str] = None

    config_snapshot: Dict[str, Any] = Field(default_factory=dict)

    status: Literal["running", "success", "failed", "cancelled"] = "running"
    error_message: Optional[str] = None


def _get_git_info() -> Optional[GitInfo]:
    try:
        def _run(cmd: list[str]) -> str:
            return subprocess.run(
                ["git"] + cmd, capture_output=True, text=True, check=True, timeout=5
            ).stdout.strip()

        commit_sha = _run(["rev-parse", "HEAD"])
        branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
        is_dirty = bool(_run(["status", "--porcelain"]))
        try:
            remote_url = _run(["config", "--get", "remote.origin.url"])
        except subprocess.CalledProcessError:
            remote_url = None

        return GitInfo(commit_sha=commit_sha, branch=branch, is_dirty=is_dirty, remote_url=remote_url)
    except Exception:
        return None


def _key_package_versions() -> Dict[str, str]:
    import importlib.metadata
    packages = ["pydantic", "boto3", "pillow", "imageio"]
    out: Dict[str, str] = {}
    for pkg in packages:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def create_manifest(
    run_id: str,
    run_type: Literal["training", "inference", "build_scene", "evaluate"],
    config_snapshot: Dict[str, Any] | None = None,
    compute_platform: Literal["local", "aws_ec2", "other"] = "local",
    instance_type: Optional[str] = None,
    generator_model: Optional[str] = None,
    critic_model: Optional[str] = None,
    input_paths: List[str] | None = None,
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        run_type=run_type,
        started_at=datetime.now(timezone.utc),
        git_info=_get_git_info(),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        package_versions=_key_package_versions(),
        generator_model=generator_model,
        critic_model=critic_model,
        config_snapshot=config_snapshot or {},
        compute_platform=compute_platform,
        instance_type=instance_type,
        input_paths=input_paths or [],
        status="running",
    )


def finalize_manifest(
    manifest: RunManifest,
    status: Literal["success", "failed", "cancelled"] = "success",
    output_uris: List[str] | None = None,
    error_message: Optional[str] = None,
) -> RunManifest:
    completed_at = datetime.now(timezone.utc)
    duration = (completed_at - manifest.started_at).total_seconds()
    return manifest.model_copy(
        update={
            "completed_at": completed_at,
            "duration_seconds": duration,
            "status": status,
            "output_uris": output_uris or manifest.output_uris,
            "error_message": error_message,
        }
    )
