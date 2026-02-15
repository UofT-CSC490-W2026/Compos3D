"""Unit tests for distributed Bronze ingestion"""

import pytest


@pytest.mark.bronze
@pytest.mark.unit
def test_bronze_ingestion_single_scene(test_env):
    """Test Bronze ingestion of a single scene"""
    from compos3d_dp.pipelines.bronze_ingestion_distributed import (
        run_bronze_ingestion_distributed,
    )

    result = run_bronze_ingestion_distributed(
        env=test_env,
        num_scenes=1,
        ray_address=None,
    )

    assert result["total_scenes"] == 1
    assert result["successful"] >= 0
    assert result["failed"] >= 0
    assert result["successful"] + result["failed"] == result["total_scenes"]
    assert "scene_ids" in result


@pytest.mark.bronze
@pytest.mark.unit
def test_bronze_ingestion_multiple_scenes(test_env):
    """Test Bronze ingestion of multiple scenes in parallel"""
    from compos3d_dp.pipelines.bronze_ingestion_distributed import (
        run_bronze_ingestion_distributed,
    )

    result = run_bronze_ingestion_distributed(
        env=test_env,
        num_scenes=2,
        ray_address=None,
    )

    assert result["total_scenes"] == 2
    assert result["successful"] > 0
    assert len(result["scene_ids"]) == result["successful"]
    assert result["total_size_mb"] > 0


@pytest.mark.bronze
@pytest.mark.unit
def test_bronze_ingestion_validates_env(test_env):
    """Test that Bronze ingestion validates environment"""
    from compos3d_dp.pipelines.bronze_ingestion_distributed import (
        run_bronze_ingestion_distributed,
    )

    # Should not crash with valid env
    result = run_bronze_ingestion_distributed(
        env=test_env,
        num_scenes=1,
        ray_address=None,
    )

    assert "total_scenes" in result
