"""Pytest configuration and fixtures"""

import os
import pytest

# Set AWS profile for all tests
os.environ["AWS_PROFILE"] = "myisb_IsbUsersPS-136268833180"


@pytest.fixture(scope="session")
def test_env():
    """Test environment"""
    return "dev"


@pytest.fixture(scope="session")
def num_test_scenes():
    """Number of scenes for testing"""
    return 2
