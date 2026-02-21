"""
Anyscale / Ray integration for distributed compute.

This module provides utilities to run compute-intensive workloads
on Anyscale's managed Ray clusters.
"""

from __future__ import annotations
import os
from typing import Any, Callable, Optional
import ray
from compos3d_dp.utils.secrets import get_anyscale_key


def init_anyscale_runtime(
    env: str = "dev",
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
    cluster_name: Optional[str] = None,
):
    """
    Initialize Ray runtime on Anyscale.

    Args:
        env: Environment name (dev, staging, prod)
        num_cpus: Number of CPUs to request
        num_gpus: Number of GPUs to request
        cluster_name: Optional Anyscale cluster name
    """
    # Get Anyscale API key from secrets
    anyscale_key = get_anyscale_key(env)

    # Set environment variable for Anyscale CLI
    os.environ["ANYSCALE_API_KEY"] = anyscale_key

    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()

    # Check if running on Anyscale cluster
    if "ANYSCALE_CLUSTER_ID" in os.environ or cluster_name:
        # Already on Anyscale cluster, just connect
        ray.init(address="auto")
    else:
        # Local development mode
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus)


def run_distributed(
    func: Callable,
    items: list[Any],
    num_cpus_per_task: int = 1,
    num_gpus_per_task: float = 0,
) -> list[Any]:
    """
    Run a function in parallel across many items using Ray.

    Args:
        func: Function to run (must be serializable)
        items: List of items to process
        num_cpus_per_task: CPUs per task
        num_gpus_per_task: GPUs per task

    Returns:
        List of results in same order as items
    """
    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized. Call init_anyscale_runtime() first.")

    # Create remote function
    remote_func = ray.remote(num_cpus=num_cpus_per_task, num_gpus=num_gpus_per_task)(
        func
    )

    # Submit all tasks
    futures = [remote_func.remote(item) for item in items]

    # Gather results
    results = ray.get(futures)

    return results


def shutdown():
    """Shutdown Ray runtime"""
    if ray.is_initialized():
        ray.shutdown()
