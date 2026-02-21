"""
BlenderBench Dataset Loader

Downloads and manages the BlenderBench dataset from HuggingFace.
https://huggingface.co/datasets/DietCoke4671/BlenderBench

The dataset contains:
- 27 instances across 3 difficulty levels
- .blend files with 3D scenes
- Start/goal code pairs
- Rendered images (512x512)
- Task descriptions
"""

from __future__ import annotations
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import json

try:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print(
        "⚠️  HuggingFace datasets not installed. Run: pip install datasets huggingface-hub"
    )


@dataclass
class BlenderBenchInstance:
    """A single instance from BlenderBench"""

    instance_id: str
    level: str
    task_description: str
    start_code: str
    goal_code: str
    blend_file_path: str
    local_blend_file: Optional[Path] = None
    start_render: Optional[Any] = None  # PIL Image
    goal_render: Optional[Any] = None  # PIL Image
    blend_file_size_mb: float = 0.0


class BlenderBenchDataset:
    """
    Manages the BlenderBench dataset.

    Usage:
        dataset = BlenderBenchDataset(cache_dir="data/blenderbench")
        dataset.download()

        # Get all instances
        instances = dataset.get_all_instances()

        # Get by level
        level1 = dataset.get_instances_by_level("level1")

        # Get specific instance
        instance = dataset.get_instance("level1/camera1")
    """

    def __init__(self, cache_dir: str = "data/blenderbench"):
        if not HF_AVAILABLE:
            raise ImportError(
                "datasets and huggingface_hub required. Run: pip install datasets huggingface-hub"
            )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_name = "DietCoke4671/BlenderBench"
        self.dataset = None
        self.instances: Dict[str, BlenderBenchInstance] = {}

    def download(self, force_reload: bool = False):
        """Download the BlenderBench dataset from HuggingFace"""

        # Load dataset
        self.dataset = load_dataset(
            self.dataset_name,
            cache_dir=str(self.cache_dir / "hf_cache"),
        )

        # Parse all instances
        for idx, example in enumerate(self.dataset["train"]):
            instance = self._parse_instance(example)
            self.instances[instance.instance_id] = instance

            if idx < 3:  # Show first few
                print(f"   - {instance.instance_id}: {instance.level}")

        print(f"   Total instances: {len(self.instances)}")
        for level in ["level1", "level2", "level3"]:
            count = len([i for i in self.instances.values() if i.level == level])
            print(f"   {level}: {count} instances")

    def _parse_instance(self, example: Dict[str, Any]) -> BlenderBenchInstance:
        """Parse a HuggingFace dataset example into our format"""
        return BlenderBenchInstance(
            instance_id=example["instance_id"],
            level=example["instance_id"].split("/")[0],  # "level1/camera1" -> "level1"
            task_description=example["task_description"],
            start_code=example["start_code"],
            goal_code=example["goal_code"],
            blend_file_path=example["blend_file_path"],
            start_render=example.get("start_render"),
            goal_render=example.get("goal_render"),
            blend_file_size_mb=example.get("blend_file_size_mb", 0.0),
        )

    def get_all_instances(self) -> List[BlenderBenchInstance]:
        """Get all instances"""
        return list(self.instances.values())

    def get_instances_by_level(self, level: str) -> List[BlenderBenchInstance]:
        """Get instances by difficulty level (level1, level2, level3)"""
        return [i for i in self.instances.values() if i.level == level]

    def get_instance(self, instance_id: str) -> Optional[BlenderBenchInstance]:
        """Get a specific instance by ID (e.g., 'level1/camera1')"""
        return self.instances.get(instance_id)

    def download_blend_file(self, instance: BlenderBenchInstance) -> Path:
        """
        Download the .blend file for a specific instance.

        Returns: Path to the local .blend file
        """
        # Check if already downloaded
        local_path = self.cache_dir / "blend_files" / instance.blend_file_path

        if local_path.exists():
            return local_path

        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Download from HuggingFace Hub
            downloaded_path = hf_hub_download(
                repo_id=self.dataset_name,
                filename=instance.blend_file_path,
                repo_type="dataset",
                cache_dir=str(self.cache_dir / "hf_cache"),
            )

            # Copy to our organized location
            shutil.copy(downloaded_path, local_path)

            instance.local_blend_file = local_path
            return local_path

        except Exception:
            raise

    def save_renders(self, instance: BlenderBenchInstance, output_dir: Path):
        """Save the start and goal renders as PNG files"""
        output_dir.mkdir(parents=True, exist_ok=True)

        if instance.start_render:
            start_path = (
                output_dir / f"{instance.instance_id.replace('/', '_')}_start.png"
            )
            instance.start_render.save(start_path)
            print(f"   Saved start render: {start_path}")

        if instance.goal_render:
            goal_path = (
                output_dir / f"{instance.instance_id.replace('/', '_')}_goal.png"
            )
            instance.goal_render.save(goal_path)
            print(f"   Saved goal render: {goal_path}")

    def export_metadata(self, output_file: Path):
        """Export all metadata to JSON"""
        metadata = []
        for instance in self.instances.values():
            metadata.append(
                {
                    "instance_id": instance.instance_id,
                    "level": instance.level,
                    "task_description": instance.task_description,
                    "blend_file_path": instance.blend_file_path,
                    "blend_file_size_mb": instance.blend_file_size_mb,
                    "start_code_length": len(instance.start_code),
                    "goal_code_length": len(instance.goal_code),
                }
            )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    # Test the dataset loader
    print("🧪 Testing BlenderBench dataset loader...")

    dataset = BlenderBenchDataset(cache_dir="data/blenderbench")
    dataset.download()

    # Get level 1 instances
    level1_instances = dataset.get_instances_by_level("level1")
    print(f"\n📋 Level 1 instances ({len(level1_instances)}):")
    for inst in level1_instances[:3]:
        print(f"   - {inst.instance_id}")
        print(f"     Task: {inst.task_description}")
        print(f"     Blend file: {inst.blend_file_path}")

    # Download a blend file
    if level1_instances:
        first_instance = level1_instances[0]
        blend_file = dataset.download_blend_file(first_instance)
