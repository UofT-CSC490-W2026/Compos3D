"""
Scene Critic using Vision-Language Models

Evaluates 3D scenes based on visual quality, physical plausibility,
and prompt adherence using CLIP and other VLMs.

Similar to VIGA's "Verifier" agent but focused on scoring/metrics.
"""

from __future__ import annotations
import torch
import open_clip
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SceneCritiqueScore:
    """Scores for different aspects of a scene"""

    overall: float  # 0-1
    visual_quality: float  # Clarity, rendering quality
    physical_plausibility: float  # Physics, object relationships
    prompt_adherence: float  # How well it matches description
    composition: float  # Camera angle, framing
    details: Dict[str, float]  # Additional metrics


class SceneCritic:
    """
    Evaluates 3D scenes using vision-language models.

    Uses CLIP for:
    - Prompt-image similarity
    - Visual quality assessment
    - Compositional analysis

    Usage:
        critic = SceneCritic()
        score = critic.evaluate(
            image_path="render.png",
            prompt="A red cube on a table"
        )
        print(f"Score: {score.overall:.2f}")
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
    ):
        """
        Initialize the critic.

        Args:
            model_name: CLIP model architecture
            pretrained: Pretrained weights
            device: Device to run on (auto-detect if None)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.eval()

        # Quality assessment prompts
        self.quality_prompts = {
            "high_quality": "a high quality, photorealistic 3D render",
            "low_quality": "a low quality, blurry, noisy image",
            "well_lit": "a well-lit scene with good lighting",
            "poorly_lit": "a dark, poorly lit scene",
            "good_composition": "a well-composed image with good framing",
            "poor_composition": "a poorly framed image with bad composition",
        }

        # Physics/plausibility prompts
        self.physics_prompts = {
            "realistic": "a physically realistic scene",
            "unrealistic": "an impossible, physically unrealistic scene",
            "stable": "objects in stable positions",
            "unstable": "floating or unstable objects",
        }

    def evaluate(
        self,
        image_path: str | Path,
        prompt: str,
        reference_images: Optional[List[Path]] = None,
    ) -> SceneCritiqueScore:
        """
        Evaluate a rendered scene.

        Args:
            image_path: Path to rendered image
            prompt: Text description of desired scene
            reference_images: Optional ground truth images

        Returns:
            SceneCritiqueScore with detailed metrics
        """
        image = Image.open(image_path).convert("RGB")

        # 1. Prompt adherence
        prompt_score = self._compute_prompt_similarity(image, prompt)

        # 2. Visual quality
        quality_score = self._assess_visual_quality(image)

        # 3. Physical plausibility
        physics_score = self._assess_physics(image)

        # 4. Composition
        composition_score = self._assess_composition(image)

        # 5. Reference similarity (if provided)
        reference_score = 1.0
        if reference_images:
            reference_score = self._compute_reference_similarity(
                image, reference_images
            )

        # Compute overall score (weighted average)
        overall = (
            0.35 * prompt_score
            + 0.25 * quality_score
            + 0.20 * physics_score
            + 0.15 * composition_score
            + 0.05 * reference_score
        )

        return SceneCritiqueScore(
            overall=float(overall),
            visual_quality=float(quality_score),
            physical_plausibility=float(physics_score),
            prompt_adherence=float(prompt_score),
            composition=float(composition_score),
            details={
                "reference_similarity": float(reference_score),
            },
        )

    def _compute_prompt_similarity(self, image: Image.Image, prompt: str) -> float:
        """Compute CLIP similarity between image and prompt"""
        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            text_tensor = self.tokenizer([prompt]).to(self.device)

            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tensor)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).item()

            # Convert from [-1, 1] to [0, 1]
            return (similarity + 1) / 2

    def _assess_visual_quality(self, image: Image.Image) -> float:
        """Assess visual quality using contrastive prompts"""
        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            prompts = [
                self.quality_prompts["high_quality"],
                self.quality_prompts["low_quality"],
            ]
            text_tensor = self.tokenizer(prompts).to(self.device)

            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tensor)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.T)[0]

            # Softmax to get probability
            probs = torch.softmax(similarities * 100, dim=0)

            return float(probs[0])  # Probability of "high quality"

    def _assess_physics(self, image: Image.Image) -> float:
        """Assess physical plausibility"""
        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            prompts = [
                self.physics_prompts["realistic"],
                self.physics_prompts["unrealistic"],
            ]
            text_tensor = self.tokenizer(prompts).to(self.device)

            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tensor)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.T)[0]
            probs = torch.softmax(similarities * 100, dim=0)

            return float(probs[0])

    def _assess_composition(self, image: Image.Image) -> float:
        """Assess compositional quality"""
        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            prompts = [
                self.quality_prompts["good_composition"],
                self.quality_prompts["poor_composition"],
            ]
            text_tensor = self.tokenizer(prompts).to(self.device)

            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tensor)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.T)[0]
            probs = torch.softmax(similarities * 100, dim=0)

            return float(probs[0])

    def _compute_reference_similarity(
        self, image: Image.Image, reference_images: List[Path]
    ) -> float:
        """Compute similarity to reference images"""
        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarities = []
            for ref_path in reference_images:
                ref_image = Image.open(ref_path).convert("RGB")
                ref_tensor = self.preprocess(ref_image).unsqueeze(0).to(self.device)
                ref_features = self.model.encode_image(ref_tensor)
                ref_features /= ref_features.norm(dim=-1, keepdim=True)

                sim = (image_features @ ref_features.T).item()
                similarities.append((sim + 1) / 2)

            return float(np.mean(similarities))

    def batch_evaluate(
        self,
        images: List[Path],
        prompts: List[str],
    ) -> List[SceneCritiqueScore]:
        """Evaluate multiple images in batch"""
        return [self.evaluate(img, prompt) for img, prompt in zip(images, prompts)]


if __name__ == "__main__":
    # Test the critic
    print("🧪 Testing Scene Critic...")

    critic = SceneCritic()

    # Create a test image
    import numpy as np
    from PIL import Image

    test_image = Image.fromarray((np.random.rand(512, 512, 3) * 255).astype(np.uint8))
    test_image.save("/tmp/test_scene.png")

    score = critic.evaluate(
        image_path="/tmp/test_scene.png",
        prompt="A simple 3D scene with geometric shapes",
    )

    print(f"   Overall: {score.overall:.3f}")
    print(f"   Visual Quality: {score.visual_quality:.3f}")
    print(f"   Physics: {score.physical_plausibility:.3f}")
    print(f"   Prompt Adherence: {score.prompt_adherence:.3f}")
    print(f"   Composition: {score.composition:.3f}")
