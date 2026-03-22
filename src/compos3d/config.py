from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_TEXT_MODEL_ID = "qwen.qwen3-next-80b-a3b"
DEFAULT_VISION_MODEL_ID = "qwen.qwen3-vl-235b-a22b"
DEFAULT_REGION = "us-east-1"

SelectionStrategy = Literal["ucb", "greedy", "random"]
BaselineMode = Literal["no_hypotheses", "fixed_hypotheses"] | None


class GeneratorConfig(BaseModel):
    provider: str = "mock"
    model_id: str | None = None
    region_name: str = DEFAULT_REGION
    max_tokens: int = Field(default=1200, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)


class CriticConfig(BaseModel):
    mode: str = "heuristic"
    provider: str = "mock"
    model_id: str | None = None
    region_name: str = DEFAULT_REGION
    max_tokens: int = Field(default=800, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class TrainingConfig(BaseModel):
    # --- Core loop hyperparameters ---
    num_init_examples_per_room: int = Field(default=1, ge=1)
    init_hypotheses_per_room: int = Field(default=3, ge=1)
    top_k: int = Field(default=2, ge=1)
    alpha: float = Field(default=0.5, ge=0.0)
    max_num_hypotheses_per_room: int = Field(default=8, ge=1)
    num_wrong_scale: float = Field(default=0.8, ge=0.0)
    update_batch_size: int = Field(default=1, ge=1)
    num_hypotheses_to_update: int = Field(default=1, ge=1)
    update_hypotheses_per_batch: int = Field(default=2, ge=1)
    only_best_hypothesis: bool = False
    num_epochs: int = Field(default=2, ge=1)
    success_threshold: float = Field(default=0.65, ge=0.0, le=1.01)
    save_every_n_examples: int = Field(default=5, ge=1)
    # --- Ablation flags ---
    # Controls the hypothesis selection policy from the bank.
    # "ucb": UCB1-style reward (accuracy + exploration bonus) — default, full system.
    # "greedy": sort by accuracy alone, no exploration.
    # "random": uniformly sample top_k hypotheses from the bank.
    selection_strategy: SelectionStrategy = "ucb"
    # Whether failed hypotheses trigger a repair regeneration round.
    # Setting to false ablates the repair / update loop entirely.
    use_repair: bool = True
    # Baseline conditioning modes:
    # null: use hypothesis-conditioned generation — the full system.
    # "no_hypotheses": generate without any hypothesis conditioning (LLM-only baseline).
    # "fixed_hypotheses": initialize bank normally but never update it (frozen bank baseline).
    baseline_mode: BaselineMode = None
    # --- Reproducibility ---
    seed: int = 42


class RenderConfig(BaseModel):
    """
    Controls whether ScenePrograms are rendered with bpy during training and
    inference.  Rendering provides real image evidence for the VLM critic and
    saves visual artifacts alongside every prediction.

    resolution:   WxH string, e.g. "256x256" (fast training) or "512x512" (paper quality).
    view_samples: CYCLES samples per image (8-16 for training, 48+ for paper figures).
    no_video:     Always True during training (skip orbital video, saves time).
    save_blend:   Save the Blender file alongside renders (useful for debugging).
    """

    enabled: bool = False
    resolution: str = "256x256"
    view_samples: int = Field(default=16, ge=1)
    no_video: bool = True
    save_blend: bool = False


class ExperimentConfig(BaseModel):
    generator: GeneratorConfig = Field(
        default_factory=lambda: GeneratorConfig(provider="mock")
    )
    critic: CriticConfig = Field(default_factory=CriticConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    room_types: list[str] | None = None


DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig(
    generator=GeneratorConfig(
        provider="mock", model_id=DEFAULT_TEXT_MODEL_ID, region_name=DEFAULT_REGION
    ),
    critic=CriticConfig(
        mode="heuristic",
        provider="mock",
        model_id=DEFAULT_VISION_MODEL_ID,
        region_name=DEFAULT_REGION,
    ),
    training=TrainingConfig(),
    render=RenderConfig(),
)


def load_experiment_config(path: Path | None = None) -> ExperimentConfig:
    if path is None:
        config = DEFAULT_EXPERIMENT_CONFIG.model_copy(deep=True)
    else:
        payload = json.loads(path.read_text())
        config = ExperimentConfig.model_validate(payload)

    if config.generator.provider == "bedrock" and not config.generator.model_id:
        config.generator.model_id = DEFAULT_TEXT_MODEL_ID

    if config.critic.mode == "vlm":
        if not config.critic.provider:
            config.critic.provider = config.generator.provider
        if config.critic.provider != "bedrock":
            raise ValueError(
                "VLM critic requires provider='bedrock'. Use heuristic mode for non-VLM evaluation."
            )
        if not config.critic.model_id:
            config.critic.model_id = DEFAULT_VISION_MODEL_ID

    return config
