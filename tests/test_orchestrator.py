"""Unit tests for pipeline orchestrator"""

import pytest


@pytest.mark.orchestrator
@pytest.mark.unit
def test_orchestrator_initialization(test_env):
    """Test Orchestrator can be initialized"""
    from compos3d_dp.orchestration.pipeline_orchestrator import Compos3DOrchestrator

    orchestrator = Compos3DOrchestrator(env=test_env)

    assert orchestrator.env == test_env


@pytest.mark.orchestrator
@pytest.mark.integration
def test_orchestrator_data_pipeline(test_env):
    """Test Orchestrator can run data pipeline"""
    from compos3d_dp.orchestration.pipeline_orchestrator import Compos3DOrchestrator

    orchestrator = Compos3DOrchestrator(env=test_env)

    result = orchestrator.run_data_pipeline(num_scenes=1)

    assert "bronze" in result
    assert "silver" in result
    assert "gold" in result

    assert result["bronze"]["total_scenes"] == 1
    assert result["silver"]["total_scenes"] >= 0
    assert result["gold"]["total_scenes"] >= 0


@pytest.mark.orchestrator
@pytest.mark.unit
def test_orchestrator_training_pipeline(test_env):
    """Test Orchestrator can run training pipeline"""
    from compos3d_dp.orchestration.pipeline_orchestrator import Compos3DOrchestrator

    orchestrator = Compos3DOrchestrator(env=test_env)

    result = orchestrator.run_training_pipeline(num_epochs=1)

    assert "total_epochs" in result
    assert result["total_epochs"] == 1


@pytest.mark.orchestrator
@pytest.mark.unit
def test_orchestrator_generation_pipeline(test_env):
    """Test Orchestrator can run generation pipeline"""
    from compos3d_dp.orchestration.pipeline_orchestrator import Compos3DOrchestrator

    orchestrator = Compos3DOrchestrator(env=test_env)

    result = orchestrator.run_generation_pipeline(prompt="a cube")

    assert "success" in result
    assert "generation_id" in result


@pytest.mark.orchestrator
@pytest.mark.unit
def test_orchestrator_api(test_env):
    """Test Orchestrator exposes correct API"""
    from compos3d_dp.orchestration.pipeline_orchestrator import Compos3DOrchestrator

    orchestrator = Compos3DOrchestrator(env=test_env)

    # Check methods exist
    assert hasattr(orchestrator, "run_data_pipeline")
    assert hasattr(orchestrator, "run_training_pipeline")
    assert hasattr(orchestrator, "run_generation_pipeline")
    assert hasattr(orchestrator, "run_full_system")

    # Check methods are callable
    assert callable(orchestrator.run_data_pipeline)
    assert callable(orchestrator.run_training_pipeline)
    assert callable(orchestrator.run_generation_pipeline)
    assert callable(orchestrator.run_full_system)


@pytest.mark.orchestrator
@pytest.mark.unit
def test_pipeline_types_enum():
    """Test pipeline types are properly defined"""
    from compos3d_dp.orchestration.pipeline_orchestrator import (
        PipelineType,
        PipelineStatus,
    )

    # Check enum values
    assert PipelineType.DATA_PROCESSING
    assert PipelineType.MODEL_TRAINING
    assert PipelineType.SCENE_GENERATION

    assert PipelineStatus.PENDING
    assert PipelineStatus.RUNNING
    assert PipelineStatus.SUCCESS
    assert PipelineStatus.FAILED
