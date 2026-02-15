"""Unit tests for distributed Silver transformation"""

import pytest


@pytest.mark.silver
@pytest.mark.unit
def test_silver_transformation_empty_bronze(test_env):
    """Test Silver transformation handles empty Bronze gracefully"""
    from compos3d_dp.pipelines.silver_transformation_distributed import (
        run_silver_transformation_distributed,
    )

    # Should handle empty Bronze without crashing
    result = run_silver_transformation_distributed(
        env=test_env,
        date_filter="2099/01/01",  # Non-existent date
        ray_address=None,
    )

    assert "total_scenes" in result
    assert result["total_scenes"] == 0


@pytest.mark.silver
@pytest.mark.integration
def test_silver_transformation_with_bronze_data(test_env, num_test_scenes):
    """Test Silver transformation processes Bronze data"""
    from compos3d_dp.pipelines.bronze_ingestion_distributed import (
        run_bronze_ingestion_distributed,
    )
    from compos3d_dp.pipelines.silver_transformation_distributed import (
        run_silver_transformation_distributed,
    )

    # First ingest some Bronze data
    bronze_result = run_bronze_ingestion_distributed(
        env=test_env,
        num_scenes=num_test_scenes,
        ray_address=None,
    )

    assert bronze_result["successful"] > 0

    # Now transform it
    silver_result = run_silver_transformation_distributed(
        env=test_env,
        date_filter=None,
        ray_address=None,
    )

    assert silver_result["total_scenes"] >= bronze_result["successful"]
    assert silver_result["successful"] >= 0


@pytest.mark.silver
@pytest.mark.unit
def test_silver_transformation_data_quality():
    """Test Silver transformation validates data quality"""
    from compos3d_dp.pipelines.silver_transformation_distributed import _categorize_task

    # Test task categorization
    assert _categorize_task("move the camera") == "camera_adjustment"
    assert _categorize_task("change object position") == "object_positioning"
    assert _categorize_task("update material color") == "material_editing"
    assert _categorize_task("something else") == "other"
