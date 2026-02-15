"""Compos3D Generation Application"""

from typing import Dict, Any, Optional
from pathlib import Path
from compos3d_dp.config import load_config
from compos3d_dp.storage.multibucket_s3 import MultiLayerS3Store
from compos3d_dp.generation.blender_executor import BlenderExecutor
from compos3d_dp.generation.critic import SceneCritic
import uuid


class Compos3DGenerator:
    def __init__(
        self,
        env: str = "prod",
        checkpoint_path: Optional[str] = None,
    ):
        self.env = env
        self.checkpoint_path = checkpoint_path

        self.cfg = load_config(env)
        self.store = MultiLayerS3Store(
            bucket_bronze=self.cfg.s3_bucket_bronze,
            bucket_silver=self.cfg.s3_bucket_silver,
            bucket_gold=self.cfg.s3_bucket_gold,
            prefix=self.cfg.s3_prefix,
            region=self.cfg.aws_region,
        )

        self.blender = BlenderExecutor()
        self.critic = SceneCritic()

    def load_model(self):
        pass

    def generate_blender_code(
        self, prompt: str, hypothesis_id: Optional[str] = None
    ) -> str:
        code = """
import bpy

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Create cube
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
cube = bpy.context.active_object
cube.name = "GeneratedCube"

# Create camera
bpy.ops.object.camera_add(location=(5, -5, 5))
camera = bpy.context.active_object
camera.rotation_euler = (1.1, 0, 0.785)
bpy.context.scene.camera = camera

# Create light
bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
"""

        return code

    def evaluate_scene(self, render_path: Path, prompt: str) -> Dict[str, float]:
        scores = self.critic.evaluate_render(render_path, prompt)
        return scores

    def generate(
        self,
        prompt: str,
        max_iterations: int = 5,
        quality_threshold: float = 0.7,
        output_dir: str = "output/generated",
    ) -> Dict[str, Any]:
        generation_id = f"gen_{uuid.uuid4().hex[:8]}"
        output_path = Path(output_dir) / generation_id
        output_path.mkdir(parents=True, exist_ok=True)

        self.load_model()

        best_result = None
        best_score = 0.0

        for iteration in range(1, max_iterations + 1):
            code = self.generate_blender_code(prompt)
            exec_result = self.blender.execute(
                code=code,
                blend_file=None,  # Start with empty scene
                render=True,  # Render automatically
                resolution=(512, 512),
                save_blend=True,
            )

            if not exec_result.success:
                continue

            if not exec_result.rendered_images or len(exec_result.rendered_images) == 0:
                continue

            render_path = exec_result.rendered_images[0]
            scores = self.evaluate_scene(
                render_path=render_path,
                prompt=prompt,
            )

            avg_score = sum(scores.values()) / len(scores)

            if avg_score > best_score:
                best_score = avg_score
                best_result = {
                    "iteration": iteration,
                    "blend_file": str(exec_result.blend_file),
                    "render": str(render_path),
                    "scores": scores,
                    "code": code,
                }

            if avg_score >= quality_threshold:
                break

        if best_result is None:
            return {
                "success": False,
                "error": "No successful generation",
                "generation_id": generation_id,
            }

        with open(best_result["blend_file"], "rb") as f:
            blend_data = f.read()

        with open(best_result["render"], "rb") as f:
            render_data = f.read()

        blend_uri = self.store.put_bytes(
            f"gold/generated_scenes/{generation_id}/scene.blend",
            blend_data,
            content_type="application/x-blender",
        )

        render_uri = self.store.put_bytes(
            f"gold/generated_scenes/{generation_id}/render.png",
            render_data,
            content_type="image/png",
        )

        result = {
            "success": True,
            "generation_id": generation_id,
            "prompt": prompt,
            "iterations": best_result["iteration"],
            "scores": best_result["scores"],
            "blend_file": best_result["blend_file"],
            "render": best_result["render"],
            "s3_blend": blend_uri,
            "s3_render": render_uri,
        }

        return result


def generate_scene(
    prompt: str,
    env: str = "prod",
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    generator = Compos3DGenerator(
        env=env,
        checkpoint_path=checkpoint_path,
    )

    return generator.generate(prompt=prompt)


if __name__ == "__main__":
    # Test generation
    result = generate_scene(
        prompt="a red cube on a wooden table",
        env="dev",
    )
