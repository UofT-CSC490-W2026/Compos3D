"""Utilities for generating run manifests with full provenance tracking"""

from __future__ import annotations
import subprocess
import sys
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from compos3d_dp.schemas.manifest import RunManifest, GitInfo


def get_git_info(repo_path: Optional[str] = None) -> Optional[GitInfo]:
    """
    Extract git provenance information from the current repository.
    Returns None if not in a git repo or git is not available.
    """
    try:
        if repo_path is None:
            repo_path = os.getcwd()

        def run_git(cmd: list[str]) -> str:
            result = subprocess.run(
                ["git", "-C", repo_path] + cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return result.stdout.strip()

        commit_sha = run_git(["rev-parse", "HEAD"])
        branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])

        # Check for uncommitted changes
        status = run_git(["status", "--porcelain"])
        is_dirty = len(status) > 0

        # Try to get remote URL (may fail if no remote configured)
        try:
            remote_url = run_git(["config", "--get", "remote.origin.url"])
        except subprocess.CalledProcessError:
            remote_url = None

        return GitInfo(
            commit_sha=commit_sha,
            branch=branch,
            is_dirty=is_dirty,
            remote_url=remote_url,
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        # Git not available or not a git repo
        return None


def get_package_versions() -> Dict[str, str]:
    """Get versions of key packages for reproducibility"""
    import importlib.metadata

    key_packages = [
        "pydantic",
        "pandas",
        "pyarrow",
        "great-expectations",
        "boto3",
        "s3fs",
    ]

    versions = {}
    for pkg in key_packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "unknown"

    return versions


def create_run_manifest(
    run_type: str,
    config: Dict[str, Any],
    run_id: Optional[str] = None,
    docker_image: Optional[str] = None,
) -> RunManifest:
    """
    Create a new run manifest with automatic provenance capture.

    Args:
        run_type: Type of run (e.g., "silver_transform", "bronze_ingest")
        config: Configuration snapshot (dict representation)
        run_id: Optional run ID (will generate if not provided)
        docker_image: Optional docker image tag

    Returns:
        RunManifest with captured provenance information
    """
    if run_id is None:
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    return RunManifest(
        run_id=run_id,
        run_type=run_type,  # type: ignore
        started_at=datetime.now(timezone.utc),
        git_info=get_git_info(),
        docker_image=docker_image,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        package_versions=get_package_versions(),
        config_snapshot=config,
        compute_platform="local",  # Will be overridden in AWS/Modal runs
        status="running",
    )


def finalize_manifest(
    manifest: RunManifest,
    status: str = "success",
    output_uris: Optional[list[str]] = None,
    output_record_counts: Optional[Dict[str, int]] = None,
    error_message: Optional[str] = None,
    quality_checks_passed: Optional[int] = None,
    quality_checks_failed: Optional[int] = None,
    data_quality_report_uri: Optional[str] = None,
) -> RunManifest:
    """
    Finalize a run manifest with completion information.

    Returns a new RunManifest instance (Pydantic models are immutable by default).
    """
    completed_at = datetime.now(timezone.utc)
    duration = (completed_at - manifest.started_at).total_seconds()

    return manifest.model_copy(
        update={
            "completed_at": completed_at,
            "duration_seconds": duration,
            "status": status,
            "output_uris": output_uris or manifest.output_uris,
            "output_record_counts": output_record_counts
            or manifest.output_record_counts,
            "error_message": error_message,
            "quality_checks_passed": quality_checks_passed,
            "quality_checks_failed": quality_checks_failed,
            "data_quality_report_uri": data_quality_report_uri,
        }
    )
