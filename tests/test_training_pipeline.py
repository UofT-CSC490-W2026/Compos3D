"""Unit tests for training pipeline"""

import pytest


@pytest.mark.training
@pytest.mark.unit
def test_trainer_initialization(test_env):
    """Test Trainer can be initialized"""
    from compos3d_dp.training.trainer import Compos3DTrainer

    trainer = Compos3DTrainer(env=test_env, checkpoint_dir="test_checkpoints")

    assert trainer.env == test_env
    assert trainer.cfg is not None
    assert trainer.store is not None


@pytest.mark.training
@pytest.mark.integration
def test_trainer_loads_training_data(test_env):
    """Test Trainer can load Gold training data"""
    from compos3d_dp.training.trainer import Compos3DTrainer
    from compos3d_dp.pipelines.gold_aggregation_distributed import (
        run_gold_aggregation_distributed,
    )

    # Ensure we have Gold data
    gold_result = run_gold_aggregation_distributed(
        env=test_env, date_filter=None, ray_address=None
    )

    if gold_result["total_scenes"] > 0:
        trainer = Compos3DTrainer(env=test_env)
        dataset = trainer.load_training_data()

        assert dataset is not None
        assert "dataset_id" in dataset
        assert "num_scenes" in dataset
        assert dataset["num_scenes"] > 0


@pytest.mark.training
@pytest.mark.unit
def test_training_loop_structure(test_env):
    """Test training loop has correct structure"""
    from compos3d_dp.training.trainer import train_compos3d

    # Run for 1 epoch to test structure
    result = train_compos3d(env=test_env, num_epochs=1, ray_address=None)

    assert "total_epochs" in result
    assert result["total_epochs"] == 1
    assert "final_metrics" in result
    assert "checkpoint_dir" in result


@pytest.mark.training
@pytest.mark.unit
def test_trainer_api(test_env):
    """Test Trainer exposes correct API for future implementation"""
    from compos3d_dp.training.trainer import Compos3DTrainer

    trainer = Compos3DTrainer(env=test_env)

    # Check methods exist
    assert hasattr(trainer, "load_training_data")
    assert hasattr(trainer, "initialize_hypothesis_bank")
    assert hasattr(trainer, "train_epoch")
    assert hasattr(trainer, "save_checkpoint")
    assert hasattr(trainer, "train")

    # Check methods are callable
    assert callable(trainer.load_training_data)
    assert callable(trainer.initialize_hypothesis_bank)
    assert callable(trainer.train_epoch)
    assert callable(trainer.save_checkpoint)
    assert callable(trainer.train)
