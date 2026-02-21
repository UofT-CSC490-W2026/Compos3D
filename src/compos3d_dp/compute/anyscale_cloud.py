"""
Anyscale Cloud Execution - Actually run on Anyscale cloud, not locally

This module submits jobs to Anyscale cloud clusters.
"""

import os
import ray
from typing import Optional


def connect_to_anyscale_cloud(
    api_key: Optional[str] = None,
    cluster_env: str = "compos3d-cluster",
) -> str:
    """
    Connect to Anyscale cloud cluster.

    Args:
        api_key: Anyscale API key (or uses ANYSCALE_API_KEY env var)
        cluster_env: Cluster environment name

    Returns:
        Ray cluster address
    """
    # Set API key
    if api_key:
        os.environ["ANYSCALE_API_KEY"] = api_key

    # Try to connect to Anyscale cluster
    # If ANYSCALE_RAY_ADDRESS is set, we're running on Anyscale
    ray_address = os.environ.get("ANYSCALE_RAY_ADDRESS", "auto")

    if ray.is_initialized():
        ray.shutdown()

    print(f"🌥️  Connecting to Ray cluster: {ray_address}")

    try:
        ray.init(address=ray_address, ignore_reinit_error=True)
        resources = ray.cluster_resources()

        # Check if we're on a real cluster (not local)
        num_nodes = resources.get("node:10.70.1.166", 0)  # Local IP

        if num_nodes > 0:
            print("⚠️  Running on LOCAL Ray cluster")
        else:
            print("Connected to Anyscale cluster")

        print(f"   Resources: {resources}")
        return ray_address

    except Exception as e:
        print(f"Failed to connect to Anyscale: {e}")
        print("   Falling back to local Ray")
        ray.init(ignore_reinit_error=True)
        return "local"


def init_ray_for_pipeline(use_anyscale: bool = True) -> str:
    """
    Initialize Ray for pipeline execution.

    Args:
        use_anyscale: Whether to try connecting to Anyscale cloud

    Returns:
        Ray cluster type ("anyscale" or "local")
    """
    if not ray.is_initialized():
        if use_anyscale and "ANYSCALE_API_KEY" in os.environ:
            # Try Anyscale first
            connect_to_anyscale_cloud()

            # Check if we're actually on cloud
            resources = ray.cluster_resources()
            if "node:10.70.1.166" not in str(resources):
                return "anyscale"

        # Fall back to local
        print("🖥️  Initializing LOCAL Ray cluster")
        ray.init(ignore_reinit_error=True)
        return "local"

    return "existing"
