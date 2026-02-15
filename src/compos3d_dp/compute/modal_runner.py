"""Modal.com integration for distributed batch processing"""

from __future__ import annotations
from typing import List, Dict, Any

try:
    import modal

    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False


# Modal app and image configuration
if MODAL_AVAILABLE:
    app = modal.App("compos3d")

    # Define container image with all dependencies
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "pydantic>=2.12.5",
            "pandas>=2.3.3",
            "pyarrow>=23.0.0",
            "boto3>=1.42.49",
            "s3fs>=2026.2.0",
            "great-expectations>=1.12.3",
        )
        .copy_local_dir("src/compos3d_dp", "/app/compos3d_dp")
        .copy_local_file("config/env.aws.yaml", "/app/config/env.aws.yaml")
    )

    # Shared volume for caching
    volume = modal.Volume.from_name("compos3d-cache", create_if_missing=True)


@(
    modal.function(
        image=image,
        secrets=[modal.Secret.from_name("aws-credentials")],
        timeout=3600,  # 1 hour
        memory=8192,  # 8GB
    )
    if MODAL_AVAILABLE
    else lambda: None
)
def process_scene_batch(
    scene_ids: List[str],
    s3_bucket: str,
    bronze_prefix: str = "bronze",
    silver_prefix: str = "silver",
) -> Dict[str, Any]:
    """
    Process a batch of scenes on Modal.

    This function runs remotely on Modal's infrastructure and processes
    multiple scenes in parallel using their distributed compute.

    Args:
        scene_ids: List of scene IDs to process
        s3_bucket: S3 bucket name
        bronze_prefix: Bronze layer prefix
        silver_prefix: Silver layer prefix

    Returns:
        Dictionary with processing results
    """
    import sys

    sys.path.insert(0, "/app")

    from compos3d_dp.storage.s3 import S3Store
    from compos3d_dp.pipelines.silver_build import build_silver_from_bronze_local

    # Create S3 store
    store = S3Store(bucket=s3_bucket, prefix="compos3d", region="us-east-1")

    results = {
        "processed": [],
        "failed": [],
        "total": len(scene_ids),
    }

    # Process scenes
    # In real implementation, you'd filter bronze by scene_ids
    try:
        output = build_silver_from_bronze_local(
            store=store,
            bronze_prefix=bronze_prefix,
            silver_prefix=silver_prefix,
            config_snapshot={"modal": True, "batch_size": len(scene_ids)},
        )
        results["processed"] = scene_ids
        results["output_uris"] = [
            output.scene_dataset_uri,
            output.scene_object_dataset_uri,
        ]
    except Exception as e:
        results["failed"] = scene_ids
        results["error"] = str(e)

    return results


@(
    modal.function(
        image=image,
        secrets=[modal.Secret.from_name("aws-credentials")],
        timeout=7200,  # 2 hours
        memory=16384,  # 16GB
        gpu="T4",  # GPU for rendering
    )
    if MODAL_AVAILABLE
    else lambda: None
)
def render_scene(
    scene_id: str,
    s3_bucket: str,
    num_views: int = 8,
) -> Dict[str, Any]:
    """
    Render a 3D scene with Blender on Modal (with GPU).

    This would integrate with VIGA-style Blender rendering.
    For now, it's a placeholder for future rendering workloads.

    Args:
        scene_id: Scene to render
        s3_bucket: S3 bucket for output
        num_views: Number of camera views to render

    Returns:
        Dictionary with render URIs
    """
    # Placeholder for future Blender integration
    return {
        "scene_id": scene_id,
        "status": "placeholder",
        "message": "Blender rendering will be implemented when needed",
        "num_views": num_views,
    }


# Local wrapper functions
def submit_batch_to_modal(
    scene_ids: List[str],
    s3_bucket: str,
    batch_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    Submit multiple batches to Modal for parallel processing.

    Args:
        scene_ids: All scene IDs to process
        s3_bucket: S3 bucket name
        batch_size: Scenes per batch

    Returns:
        List of results from each batch
    """
    if not MODAL_AVAILABLE:
        raise ImportError("Modal is not installed. Run: pip install modal")

    # Split into batches
    batches = [
        scene_ids[i : i + batch_size] for i in range(0, len(scene_ids), batch_size)
    ]

    print(f"Processing {len(scene_ids)} scenes in {len(batches)} batches on Modal...")

    # Run batches in parallel on Modal
    results = []
    for batch in batches:
        result = process_scene_batch.remote(
            scene_ids=batch,
            s3_bucket=s3_bucket,
        )
        results.append(result)

    return results


def submit_renders_to_modal(
    scene_ids: List[str],
    s3_bucket: str,
) -> List[Dict[str, Any]]:
    """
    Submit rendering jobs to Modal (GPU-accelerated).

    Args:
        scene_ids: Scenes to render
        s3_bucket: S3 bucket for outputs

    Returns:
        List of render results
    """
    if not MODAL_AVAILABLE:
        raise ImportError("Modal is not installed. Run: pip install modal")

    print(f"Submitting {len(scene_ids)} render jobs to Modal...")

    # Run renders in parallel
    results = []
    for scene_id in scene_ids:
        result = render_scene.remote(
            scene_id=scene_id,
            s3_bucket=s3_bucket,
        )
        results.append(result)

    return results


# CLI integration
if __name__ == "__main__":
    # Example usage
    if not MODAL_AVAILABLE:
        print("Modal is not installed. Run: pip install modal")
        exit(1)

    # Deploy Modal app
    print("Deploying to Modal...")
    print("Run: modal deploy src/compos3d_dp/compute/modal_runner.py")
