"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path

import pytest

from src.compos3d.app_config import AppConfig
from src.compos3d.storage.local import LocalStore


@pytest.fixture(scope="session")
def test_env():
    """Infrastructure environment name used for testing."""
    return "local"


@pytest.fixture(scope="session")
def num_test_scenes():
    """Number of scenes to generate in integration tests."""
    return 2


@pytest.fixture
def tmp_store(tmp_path):
    """A fresh LocalStore backed by a pytest tmp_path."""
    return LocalStore(root=tmp_path)


@pytest.fixture
def local_app_config(tmp_path):
    """AppConfig wired to a local temp lake root."""
    return AppConfig(
        env="local",
        storage_backend="local",
        local_lake_root=str(tmp_path / "_lake"),
    )


@pytest.fixture
def dummy_dataset_path():
    """Path to the bundled dummy_fast.json dataset."""
    p = Path(__file__).parent / "examples" / "dummy_fast.json"
    assert p.exists(), f"dummy_fast.json not found at {p}"
    return p
