from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from compos3d.hypothesis.engine import run_vertical_inference, train_vertical_slice

if TYPE_CHECKING:
    from compos3d.storage import AnyStore


@dataclass(frozen=True)
class TrainingRequest:
    dataset_path: Path
    output_dir: Path
    experiment_name: str
    llm_provider: str = "mock"
    config_path: Path | None = None
    num_init_examples_per_room: int = 1
    init_hypotheses_per_room: int = 3
    top_k: int = 2
    alpha: float = 0.5
    max_num_hypotheses_per_room: int = 8
    num_wrong_scale: float = 0.8
    update_batch_size: int = 1
    num_hypotheses_to_update: int = 1
    update_hypotheses_per_batch: int = 2
    only_best_hypothesis: bool = False
    num_epochs: int = 2
    success_threshold: float = 0.65
    save_every_n_examples: int = 5
    resume: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_mode: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: tuple[str, ...] = ()
    # Lake storage (optional — if None, outputs go to local output_dir only)
    store: "AnyStore | None" = None
    compute_platform: str = "local"
    instance_type: str | None = None


@dataclass(frozen=True)
class InferenceRequest:
    bank_path: Path
    prompt: str
    output_dir: Path
    llm_provider: str = "mock"
    config_path: Path | None = None
    top_k: int = 2
    inference_strategy: str = "joint_top_k"
    # --- Render options ---
    render_scene: bool = False
    render_resolution: str = "512x512"
    render_view_samples: int = 48
    render_video_frames: int = 90
    render_video_samples: int = 16
    # Lake storage (optional)
    store: "AnyStore | None" = None
    compute_platform: str = "local"
    instance_type: str | None = None


def train_hypotheses(request: TrainingRequest) -> dict:
    return train_vertical_slice(
        dataset_path=request.dataset_path,
        output_dir=request.output_dir,
        experiment_name=request.experiment_name,
        llm_provider=request.llm_provider,
        config_path=request.config_path,
        num_init_examples_per_room=request.num_init_examples_per_room,
        init_hypotheses_per_room=request.init_hypotheses_per_room,
        top_k=request.top_k,
        alpha=request.alpha,
        max_num_hypotheses_per_room=request.max_num_hypotheses_per_room,
        num_wrong_scale=request.num_wrong_scale,
        update_batch_size=request.update_batch_size,
        num_hypotheses_to_update=request.num_hypotheses_to_update,
        update_hypotheses_per_batch=request.update_hypotheses_per_batch,
        only_best_hypothesis=request.only_best_hypothesis,
        num_epochs=request.num_epochs,
        success_threshold=request.success_threshold,
        save_every_n_examples=request.save_every_n_examples,
        resume=request.resume,
        wandb_project=request.wandb_project,
        wandb_entity=request.wandb_entity,
        wandb_mode=request.wandb_mode,
        wandb_run_name=request.wandb_run_name,
        wandb_tags=list(request.wandb_tags),
        store=request.store,
        compute_platform=request.compute_platform,
        instance_type=request.instance_type,
    )


def run_frozen_inference(request: InferenceRequest) -> dict:
    return run_vertical_inference(
        bank_path=request.bank_path,
        prompt=request.prompt,
        output_dir=request.output_dir,
        llm_provider=request.llm_provider,
        config_path=request.config_path,
        top_k=request.top_k,
        inference_strategy=request.inference_strategy,
        render_scene=request.render_scene,
        render_resolution=request.render_resolution,
        render_view_samples=request.render_view_samples,
        render_video_frames=request.render_video_frames,
        render_video_samples=request.render_video_samples,
        store=request.store,
        compute_platform=request.compute_platform,
        instance_type=request.instance_type,
    )
