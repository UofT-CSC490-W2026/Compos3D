"""Compos3D Training Loop"""

from typing import Dict, Any, Optional
import ray
from pathlib import Path
from compos3d_dp.config import load_config
from compos3d_dp.storage.multibucket_s3 import MultiLayerS3Store


class Compos3DTrainer:
    def __init__(
        self,
        env: str = "dev",
        checkpoint_dir: str = "checkpoints",
        ray_address: Optional[str] = None,
    ):
        self.env = env
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if ray_address:
            ray.init(address=ray_address)
        elif not ray.is_initialized():
            ray.init()

        self.cfg = load_config(env)
        self.store = MultiLayerS3Store(
            bucket_bronze=self.cfg.s3_bucket_bronze,
            bucket_silver=self.cfg.s3_bucket_silver,
            bucket_gold=self.cfg.s3_bucket_gold,
            prefix=self.cfg.s3_prefix,
            region=self.cfg.aws_region,
        )

    def load_training_data(self) -> Dict[str, Any]:
        gold_files = self.store.list_prefix("gold/training_datasets/")
        dataset_files = [f for f in gold_files if f.endswith("dataset.json")]

        if not dataset_files:
            raise ValueError("No training datasets found in Gold layer")

        latest_dataset_path = sorted(dataset_files)[-1]
        dataset = self.store.read_json(latest_dataset_path)

        return dataset

    def initialize_hypothesis_bank(self):
        pass

    def train_epoch(self, epoch: int, dataset: Dict[str, Any]) -> Dict[str, float]:
        metrics = {
            "epoch": epoch,
            "loss": 0.0,
            "critic_score": 0.0,
            "successful_generations": 0,
        }

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        pass

    def train(
        self,
        num_epochs: int = 10,
        save_every: int = 1,
    ) -> Dict[str, Any]:
        dataset = self.load_training_data()
        self.initialize_hypothesis_bank()

        all_metrics = []
        for epoch in range(1, num_epochs + 1):
            metrics = self.train_epoch(epoch, dataset)
            all_metrics.append(metrics)

            if epoch % save_every == 0:
                self.save_checkpoint(epoch, metrics)

        summary = {
            "total_epochs": num_epochs,
            "final_metrics": all_metrics[-1] if all_metrics else {},
            "dataset_id": dataset["dataset_id"],
            "checkpoint_dir": str(self.checkpoint_dir),
        }

        return summary


def train_compos3d(
    env: str = "dev",
    num_epochs: int = 10,
    ray_address: Optional[str] = None,
) -> Dict[str, Any]:
    trainer = Compos3DTrainer(
        env=env,
        checkpoint_dir=f"checkpoints/{env}",
        ray_address=ray_address,
    )

    return trainer.train(num_epochs=num_epochs)


if __name__ == "__main__":
    # Test training
    summary = train_compos3d(
        env="dev",
        num_epochs=3,
        ray_address=None,
    )
