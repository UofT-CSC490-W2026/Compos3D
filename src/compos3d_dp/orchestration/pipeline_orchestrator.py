"""Compos3D Pipeline Orchestrator"""

from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class PipelineType(Enum):
    """Types of pipelines in Compos3D"""

    DATA_PROCESSING = "data_processing"
    MODEL_TRAINING = "model_training"
    SCENE_GENERATION = "scene_generation"


class PipelineStatus(Enum):
    """Status of a pipeline run"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class PipelineRun:
    """Metadata for a pipeline run"""

    run_id: str
    pipeline_type: PipelineType
    status: PipelineStatus
    started_at: datetime
    completed_at: Optional[datetime]
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    error: Optional[str]


class Compos3DOrchestrator:
    def __init__(self, env: str = "dev"):
        self.env = env

    def run_data_pipeline(
        self,
        num_scenes: int = 100,
        skip_bronze: bool = False,
        skip_silver: bool = False,
        skip_gold: bool = False,
    ) -> Dict[str, Any]:
        print(f"Running data pipeline: env={self.env}, scenes={num_scenes}")

        results = {}

        if not skip_bronze:
            from compos3d_dp.pipelines.bronze_ingestion_distributed import (
                run_bronze_ingestion_distributed,
            )

            print("Bronze: Ingesting scenes...")
            results["bronze"] = run_bronze_ingestion_distributed(
                env=self.env,
                num_scenes=num_scenes,
            )

        if not skip_silver:
            from compos3d_dp.pipelines.silver_transformation_distributed import (
                run_silver_transformation_distributed,
            )

            print("Silver: Transforming...")
            results["silver"] = run_silver_transformation_distributed(
                env=self.env,
            )

        if not skip_gold:
            from compos3d_dp.pipelines.gold_aggregation_distributed import (
                run_gold_aggregation_distributed,
            )

            print("Gold: Aggregating...")
            results["gold"] = run_gold_aggregation_distributed(
                env=self.env,
            )

        return results

    def run_training_pipeline(
        self,
        num_epochs: int = 10,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        print(f"Running training: epochs={num_epochs}")

        from compos3d_dp.training.trainer import train_compos3d

        result = train_compos3d(
            env=self.env,
            num_epochs=num_epochs,
        )

        return result

    def run_generation_pipeline(
        self,
        prompt: str,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        print(f"Running generation: prompt='{prompt}'")

        from compos3d_dp.inference.generator import generate_scene

        result = generate_scene(
            prompt=prompt,
            env=self.env,
            checkpoint_path=checkpoint_path,
        )

        return result

    def run_full_system(
        self,
        num_scenes: int = 100,
        num_epochs: int = 10,
        test_prompt: str = "a red cube on a table",
    ) -> Dict[str, Any]:
        print(
            f"Running full system: env={self.env}, scenes={num_scenes}, epochs={num_epochs}"
        )

        results = {}

        print("\nPhase 1: Data Processing")
        results["data"] = self.run_data_pipeline(num_scenes=num_scenes)

        print("\nPhase 2: Training")
        results["training"] = self.run_training_pipeline(num_epochs=num_epochs)

        print("\nPhase 3: Generation")
        results["generation"] = self.run_generation_pipeline(prompt=test_prompt)

        print(
            f"\nComplete: {results['data'].get('gold', {}).get('total_scenes', 0)} scenes, "
            f"{results['training'].get('total_epochs', 0)} epochs"
        )

        return results


if __name__ == "__main__":
    # Test orchestrator
    orchestrator = Compos3DOrchestrator(env="dev")

    # Run data pipeline only
    results = orchestrator.run_data_pipeline(num_scenes=3)
