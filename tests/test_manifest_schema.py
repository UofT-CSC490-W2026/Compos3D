"""Unit tests for run manifest creation/finalization lifecycle.

Covers:
- Required metadata at manifest creation time.
- Status transition from `running` to terminal states.
- Completion timestamps and duration computation.
- Error message persistence for failed runs.

Expected outcomes:
- `create_manifest` returns a running manifest with core provenance fields.
- `finalize_manifest` fills `completed_at`/`duration_seconds` and updates status.
- Duration is non-negative and time ordering remains consistent.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from compos3d.schemas.manifest import create_manifest, finalize_manifest


def test_create_manifest_has_required_metadata() -> None:
    manifest = create_manifest(
        run_id="run_123",
        run_type="training",
        config_snapshot={"alpha": 0.5},
        input_paths=["dataset.jsonl"],
    )
    assert manifest.run_id == "run_123"
    assert manifest.run_type == "training"
    assert manifest.status == "running"
    assert manifest.python_version
    assert isinstance(manifest.package_versions, dict)
    assert manifest.started_at.tzinfo is not None


def test_finalize_manifest_success_sets_completion_fields() -> None:
    manifest = create_manifest(run_id="r1", run_type="inference")
    finalized = finalize_manifest(
        manifest,
        status="success",
        output_uris=["silver/inference/r1/critic_score.json"],
    )
    assert finalized.status == "success"
    assert finalized.completed_at is not None
    assert finalized.duration_seconds is not None
    assert finalized.duration_seconds >= 0.0
    assert finalized.output_uris == ["silver/inference/r1/critic_score.json"]


def test_finalize_manifest_failed_status_keeps_error_message() -> None:
    manifest = create_manifest(run_id="r2", run_type="evaluate")
    finalized = finalize_manifest(
        manifest, status="failed", error_message="critical failure"
    )
    assert finalized.status == "failed"
    assert finalized.error_message == "critical failure"


def test_manifest_timestamp_consistency_after_finalize() -> None:
    started_at = datetime.now(timezone.utc) - timedelta(seconds=7)
    manifest = create_manifest(run_id="r3", run_type="build_scene").model_copy(
        update={"started_at": started_at}
    )
    finalized = finalize_manifest(manifest, status="cancelled")
    assert finalized.completed_at is not None
    assert finalized.completed_at >= started_at
    assert finalized.duration_seconds is not None
    assert finalized.duration_seconds >= 7.0

