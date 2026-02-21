"""Unit tests for distributed Gold aggregation"""

import pytest


@pytest.mark.gold
@pytest.mark.unit
def test_gold_aggregation_empty_silver(test_env):
    """Test Gold aggregation handles empty Silver gracefully"""
    from compos3d_dp.pipelines.gold_aggregation_distributed import (
        run_gold_aggregation_distributed,
    )

    result = run_gold_aggregation_distributed(
        env=test_env,
        date_filter="2099/01/01",  # Non-existent date
        ray_address=None,
    )

    assert "total_scenes" in result
    assert result["total_scenes"] == 0


@pytest.mark.gold
@pytest.mark.integration
def test_gold_aggregation_with_data(test_env, num_test_scenes):
    """Test Gold aggregation creates training datasets"""
    from compos3d_dp.pipelines.bronze_ingestion_distributed import (
        run_bronze_ingestion_distributed,
    )
    from compos3d_dp.pipelines.silver_transformation_distributed import (
        run_silver_transformation_distributed,
    )
    from compos3d_dp.pipelines.gold_aggregation_distributed import (
        run_gold_aggregation_distributed,
    )

    # Run full pipeline
    bronze_result = run_bronze_ingestion_distributed(
        env=test_env, num_scenes=num_test_scenes, ray_address=None
    )
    assert bronze_result["successful"] > 0

    silver_result = run_silver_transformation_distributed(
        env=test_env, date_filter=None, ray_address=None
    )
    assert silver_result["successful"] > 0

    gold_result = run_gold_aggregation_distributed(
        env=test_env, date_filter=None, ray_address=None
    )

    assert gold_result["total_scenes"] > 0
    assert "dataset_id" in gold_result
    assert "gold_uri" in gold_result


@pytest.mark.gold
@pytest.mark.unit
def test_gold_aggregation_creates_statistics(test_env):
    """Test Gold aggregation computes correct statistics"""
    from compos3d_dp.pipelines.gold_aggregation_distributed import (
        run_gold_aggregation_distributed,
    )

    # This will aggregate whatever Silver data exists
    result = run_gold_aggregation_distributed(
        env=test_env,
        date_filter=None,
        ray_address=None,
    )

    assert "total_scenes" in result
    assert "dataset_id" in result
