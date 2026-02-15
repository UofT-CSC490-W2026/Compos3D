"""Compos3D Data Platform CLI"""

from __future__ import annotations
import typer
from rich import print

app = typer.Typer(
    help="Compos3D - Production 3D Generation System", no_args_is_help=True
)


# ============================================================================
# FULL SYSTEM COMMANDS
# ============================================================================


@app.command("run-full-system")
def run_full_system(
    env: str = typer.Option("dev", help="Environment (dev/staging/prod)"),
    num_scenes: int = typer.Option(10, help="Number of scenes to ingest"),
    num_epochs: int = typer.Option(3, help="Training epochs"),
    test_prompt: str = typer.Option(
        "a red cube on a table", help="Test generation prompt"
    ),
):
    """Run the complete Compos3D system end-to-end"""
    from compos3d_dp.orchestration.pipeline_orchestrator import Compos3DOrchestrator

    orchestrator = Compos3DOrchestrator(env=env)
    orchestrator.run_full_system(
        num_scenes=num_scenes,
        num_epochs=num_epochs,
        test_prompt=test_prompt,
    )


@app.command("data-pipeline")
def data_pipeline(
    env: str = typer.Option("dev"),
    num_scenes: int = typer.Option(100, help="Number of scenes to process"),
    skip_bronze: bool = typer.Option(False, help="Skip Bronze ingestion"),
    skip_silver: bool = typer.Option(False, help="Skip Silver transformation"),
    skip_gold: bool = typer.Option(False, help="Skip Gold aggregation"),
):
    """Run complete data processing pipeline (Bronze → Silver → Gold)"""
    from compos3d_dp.orchestration.pipeline_orchestrator import Compos3DOrchestrator

    orchestrator = Compos3DOrchestrator(env=env)
    orchestrator.run_data_pipeline(
        num_scenes=num_scenes,
        skip_bronze=skip_bronze,
        skip_silver=skip_silver,
        skip_gold=skip_gold,
    )


@app.command("train")
def train(
    env: str = typer.Option("dev"),
    num_epochs: int = typer.Option(10, help="Number of training epochs"),
    checkpoint: str = typer.Option(None, help="Resume from checkpoint"),
):
    """Train the Compos3D generative model"""
    from compos3d_dp.orchestration.pipeline_orchestrator import Compos3DOrchestrator

    orchestrator = Compos3DOrchestrator(env=env)
    orchestrator.run_training_pipeline(
        num_epochs=num_epochs,
        checkpoint_path=checkpoint,
    )


@app.command("generate")
def generate(
    prompt: str = typer.Argument(..., help="Text description of scene to generate"),
    env: str = typer.Option("prod"),
    checkpoint: str = typer.Option(None, help="Model checkpoint to use"),
):
    """Generate a 3D scene from text prompt"""
    from compos3d_dp.orchestration.pipeline_orchestrator import Compos3DOrchestrator

    orchestrator = Compos3DOrchestrator(env=env)
    result = orchestrator.run_generation_pipeline(
        prompt=prompt,
        checkpoint_path=checkpoint,
    )

    if not result["success"]:
        print(f"Generation failed: {result.get('error')}")


# ============================================================================
# INDIVIDUAL PIPELINE COMMANDS
# ============================================================================


@app.command("bronze-ingest")
def bronze_ingest(
    env: str = typer.Option("dev"),
    num_scenes: int = typer.Option(100, help="Number of scenes to ingest"),
):
    """Bronze Layer: Ingest raw scenes from BlenderBench"""
    from compos3d_dp.pipelines.bronze_ingestion_distributed import (
        run_bronze_ingestion_distributed,
    )

    run_bronze_ingestion_distributed(env=env, num_scenes=num_scenes)


@app.command("silver-transform")
def silver_transform(
    env: str = typer.Option("dev"),
    date_filter: str = typer.Option(None, help="Filter by date (YYYY/MM/DD)"),
):
    """Silver Layer: Clean, validate, and transform Bronze data"""
    from compos3d_dp.pipelines.silver_transformation_distributed import (
        run_silver_transformation_distributed,
    )

    run_silver_transformation_distributed(env=env, date_filter=date_filter)


@app.command("gold-aggregate")
def gold_aggregate(
    env: str = typer.Option("dev"),
    date_filter: str = typer.Option(None, help="Filter by date (YYYY/MM/DD)"),
):
    """Gold Layer: Aggregate Silver data into training datasets"""
    from compos3d_dp.pipelines.gold_aggregation_distributed import (
        run_gold_aggregation_distributed,
    )

    run_gold_aggregation_distributed(env=env, date_filter=date_filter)


# ============================================================================
# UTILITY COMMANDS
# ============================================================================


@app.command("show-config")
def show_config(
    env: str = typer.Option("dev"),
):
    """Display configuration for environment"""
    from compos3d_dp.config import load_config

    cfg = load_config(env)
    print(f"[bold]Configuration for {env}:[/bold]")
    print(f"  Storage: {cfg.storage_backend}")
    print(f"  Bronze: s3://{cfg.s3_bucket_bronze}")
    print(f"  Silver: s3://{cfg.s3_bucket_silver}")
    print(f"  Gold: s3://{cfg.s3_bucket_gold}")
    print(f"  Region: {cfg.aws_region}")


@app.command("download-blenderbench")
def download_blenderbench(
    cache_dir: str = typer.Option("data/blenderbench", help="Cache directory"),
):
    """Download BlenderBench dataset from HuggingFace"""
    from compos3d_dp.datasets.blenderbench import BlenderBenchDataset

    dataset = BlenderBenchDataset(cache_dir=cache_dir)
    dataset.download()


if __name__ == "__main__":
    app()
