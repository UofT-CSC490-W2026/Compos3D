from pathlib import Path
import shlex
from typing import Optional

import typer
from rich import print

from compos3d._stages import StagePendingError
from compos3d.evaluation.service import EvaluateRequest, evaluate_run
from compos3d.hypothesis.service import (
    InferenceRequest,
    TrainingRequest,
    run_frozen_inference,
    train_hypotheses,
)
from compos3d.procedural.service import (
    BackendSmokeRequest,
    BuildSceneRequest,
    ReferenceGenerationRequest,
    build_scene,
    feature_extract,
    generate_reference_scene,
    run_backend_smoke,
)

app = typer.Typer(
    help="Compos3D staged research pipeline.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


def _run_or_exit(func):
    try:
        result = func()
        if result is not None:
            print(result)
    except StagePendingError as exc:
        print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command("backend-smoke")
def backend_smoke(
    asset_name: str = typer.Option(
        "dining_table", help="Controllable asset to smoke-test"
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/backend_smoke"), help="Local output directory"
    ),
    seed: int = typer.Option(0, help="Random seed"),
):
    _run_or_exit(
        lambda: run_backend_smoke(
            BackendSmokeRequest(asset_name=asset_name, output_dir=output_dir, seed=seed)
        )
    )


@app.command("reference-generate")
def reference_generate(
    room_type: str = typer.Option(
        "dining_room", help="Target room type: dining_room, living_room, bedroom"
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/reference_runs"), help="Local output directory"
    ),
    seed: int = typer.Option(0, help="Random seed"),
    tasks: str = typer.Option(
        "coarse",
        help='Space-separated infinigen task list. "coarse" generates layout only (~20-45 min). '
        'Add "populate render" for a fully furnished rendered scene (~60-90 min). '
        "This is an offline generation job — expect long runtimes.",
    ),
    minimal: bool = typer.Option(
        False,
        help="Use minimal_solve gin config (fewer annealing steps, faster but sparser layout). "
        "Reduces coarse time to ~5-15 min.",
    ),
):
    task_list = tuple(tasks.split())
    _run_or_exit(
        lambda: generate_reference_scene(
            ReferenceGenerationRequest(
                room_type=room_type,
                output_dir=output_dir,
                seed=seed,
                tasks=task_list,
                minimal=minimal,
            )
        )
    )


@app.command("build-scene")
def build_scene_cmd(
    scene_program: Path = typer.Option(
        ..., exists=False, help="Path to scene_program.json from run-inference"
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/scenes"),
        help="Output directory for views, video, and blend file",
    ),
    seed: int = typer.Option(0, help="Asset variation seed"),
    resolution: str = typer.Option(
        "512x512", help="Render resolution (WxH, e.g. 512x512 or 1024x1024)"
    ),
    view_samples: int = typer.Option(
        48, help="CYCLES samples per 4-view image (48 = fast+clean)"
    ),
    video_samples: int = typer.Option(
        16, help="CYCLES samples per orbital video frame (16 = fast)"
    ),
    video_frames: int = typer.Option(
        90, help="Number of orbital video frames (90 = 3s @ 30fps)"
    ),
    no_video: bool = typer.Option(
        False,
        "--no-video",
        help="Skip orbital video rendering; only produce the 4 views",
    ),
    save_blend: bool = typer.Option(
        False, "--save-blend", help="Also export the Blender scene file"
    ),
):
    """
    Build a 3D scene from a SceneProgram JSON, render 4 canonical views,
    and produce an orbital turntable video.
    """
    _run_or_exit(
        lambda: build_scene(
            BuildSceneRequest(
                scene_program_path=scene_program,
                output_dir=output_dir,
                seed=seed,
                resolution=resolution,
                view_samples=view_samples,
                video_samples=video_samples,
                video_frames=video_frames,
                no_video=no_video,
                save_blend=save_blend,
            )
        )
    )


@app.command("feature-extract")
def feature_extract_cmd(
    input_path: Path = typer.Option(..., exists=False, help="Input artifact root"),
    output_dir: Path = typer.Option(
        Path("artifacts/features"), help="Local output directory"
    ),
):
    _run_or_exit(lambda: feature_extract(input_path=input_path, output_dir=output_dir))


@app.command("train-hypotheses")
def train_hypotheses_cmd(
    dataset_path: Path = typer.Option(
        ..., exists=False, help="Gold dataset or split manifest"
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/training"), help="Local output directory"
    ),
    experiment_name: str = typer.Option("vertical_slice", help="Experiment label"),
    llm_provider: str = typer.Option(
        "mock", help="Generator provider to use when config_path is not supplied"
    ),
    config_path: Path | None = typer.Option(
        None,
        exists=False,
        help="Experiment JSON config for generator, critic, and loop hyperparameters",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume an interrupted training run from <output_dir>/<experiment_name>/resume_state.json",
    ),
    wandb_project: str | None = typer.Option(
        None,
        help="Enable W&B logging and override the project name",
    ),
    wandb_entity: str | None = typer.Option(
        None,
        help="Optional W&B entity/team",
    ),
    wandb_mode: str | None = typer.Option(
        None,
        help="Optional W&B mode override: online, offline, or disabled",
    ),
    wandb_run_name: str | None = typer.Option(
        None,
        help="Optional W&B run name override",
    ),
    wandb_tags: str = typer.Option(
        "",
        help="Comma-separated W&B tags",
    ),
    env: Optional[str] = typer.Option(
        None,
        "--env",
        help="Infrastructure environment: local (default), dev, staging, or prod. "
        "dev/staging/prod route outputs to S3 bronze/silver/gold buckets.",
    ),
    compute_platform: str = typer.Option(
        "local",
        "--compute-platform",
        hidden=True,
    ),
    instance_type: str | None = typer.Option(
        None,
        "--instance-type",
        hidden=True,
    ),
):
    """Train a hypothesis bank on a labelled scene dataset."""
    store = None
    if env and env != "local":
        from compos3d.app_config import load_app_config
        from compos3d.storage import get_store

        app_cfg = load_app_config(env)  # type: ignore[arg-type]
        store = get_store(app_cfg)
        print(
            f"[bold green]Lake storage:[/bold green] {env} → {app_cfg.storage_backend}"
        )

    _run_or_exit(
        lambda: train_hypotheses(
            TrainingRequest(
                dataset_path=dataset_path,
                output_dir=output_dir,
                experiment_name=experiment_name,
                llm_provider=llm_provider,
                config_path=config_path,
                resume=resume,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                wandb_mode=wandb_mode,
                wandb_run_name=wandb_run_name,
                wandb_tags=tuple(
                    item.strip() for item in wandb_tags.split(",") if item.strip()
                ),
                store=store,
                compute_platform=compute_platform,
                instance_type=instance_type,
            )
        )
    )


@app.command("run-inference")
def run_inference_cmd(
    bank_path: Path = typer.Option(..., exists=False, help="Frozen hypothesis bank"),
    prompt: str = typer.Option(..., help="Prompt to generate from"),
    output_dir: Path = typer.Option(
        Path("artifacts/inference"), help="Local output directory"
    ),
    llm_provider: str = typer.Option(
        "mock", help="Generator provider to use when config_path is not supplied"
    ),
    config_path: Path | None = typer.Option(
        None, exists=False, help="Experiment JSON config for generator and critic"
    ),
    inference_strategy: str = typer.Option(
        "joint_top_k",
        help="Inference strategy: joint_top_k or filter_and_weight",
    ),
    render_scene: bool = typer.Option(
        False,
        "--render-scene",
        help="After generating the SceneProgram, build the 3D scene, render 4 canonical views, "
        "and produce an orbital video.  Results go into <output_dir>/scene/.",
    ),
    render_resolution: str = typer.Option(
        "512x512", help="Render resolution (WxH) when --render-scene is set"
    ),
    render_view_samples: int = typer.Option(
        48, help="CYCLES samples per view (48 = fast+clean)"
    ),
    render_video_frames: int = typer.Option(
        90, help="Orbital video frames (90 = 3s @ 30fps)"
    ),
    render_video_samples: int = typer.Option(16, help="CYCLES samples per video frame"),
    env: Optional[str] = typer.Option(
        None,
        "--env",
        help="Infrastructure environment: local (default), dev, staging, or prod.",
    ),
    compute_platform: str = typer.Option(
        "local",
        "--compute-platform",
        hidden=True,
    ),
    instance_type: str | None = typer.Option(
        None,
        "--instance-type",
        hidden=True,
    ),
):
    """
    Generate a SceneProgram from a frozen hypothesis bank.

    With --render-scene the pipeline continues automatically:
    SceneProgram → 3D Blender scene → 4 canonical views → orbital video → scene_features.json
    """
    store = None
    if env and env != "local":
        from compos3d.app_config import load_app_config
        from compos3d.storage import get_store

        app_cfg = load_app_config(env)  # type: ignore[arg-type]
        store = get_store(app_cfg)
        print(
            f"[bold green]Lake storage:[/bold green] {env} → {app_cfg.storage_backend}"
        )

    _run_or_exit(
        lambda: run_frozen_inference(
            InferenceRequest(
                bank_path=bank_path,
                prompt=prompt,
                output_dir=output_dir,
                llm_provider=llm_provider,
                config_path=config_path,
                inference_strategy=inference_strategy,
                render_scene=render_scene,
                render_resolution=render_resolution,
                render_view_samples=render_view_samples,
                render_video_frames=render_video_frames,
                render_video_samples=render_video_samples,
                store=store,
                compute_platform=compute_platform,
                instance_type=instance_type,
            )
        )
    )


@app.command("evaluate")
def evaluate_cmd(
    predictions_dir: Path = typer.Option(
        ..., exists=False, help="Inference artifact root"
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/evaluation"), help="Local output directory"
    ),
):
    _run_or_exit(
        lambda: evaluate_run(
            EvaluateRequest(predictions_dir=predictions_dir, output_dir=output_dir)
        )
    )


@app.command("launch-aws")
def launch_aws_cmd(
    command: str = typer.Argument(
        ...,
        help="compos3d sub-command to run on EC2 (e.g. train-hypotheses, run-inference)",
    ),
    cli_args: Optional[str] = typer.Option(
        None,
        "--cli-args",
        help="Additional CLI arguments as a single quoted string, "
        "e.g. '--dataset-path s3://... --config-path train_configs/compos3d.json --env dev'",
    ),
    env: str = typer.Option(
        "dev", "--env", help="Target environment: dev, staging, prod"
    ),
    instance_type: Optional[str] = typer.Option(
        None, help="Override EC2 instance type from env config"
    ),
    repo_url: str = typer.Option(
        "https://github.com/yourorg/Compos3D.git",
        help="Git remote metadata to record with the submitted job",
    ),
    git_ref: str = typer.Option("main", help="Git ref metadata to record with the job"),
    image_tag: str | None = typer.Option(
        None,
        "--image-tag",
        help="Override the container image tag from env config",
    ),
    wait: bool = typer.Option(
        False, "--wait", help="Block until the EC2 job completes"
    ),
    log_group: str | None = typer.Option(
        None,
        help="Override the CloudWatch Logs group from env config",
    ),
):
    """
    Launch a compos3d job on AWS EC2 (spot by default).

    The instance bootstraps Docker, pulls the Compos3D runtime image from ECR,
    runs the requested compos3d command inside the container, writes outputs to
    S3, then self-terminates.

    Example — submit a training run to the dev environment:

    \\b
        compos3d launch-aws train-hypotheses \\
            --cli-args '--dataset-path s3://compos3d-dev-bronze/datasets/vs.json --config-path train_configs/compos3d.json --env dev' \\
            --env dev --wait
    """
    from compos3d.app_config import load_app_config
    from compos3d.compute.ec2_runner import EC2JobRunner, EC2JobSpec

    app_cfg = load_app_config(env)  # type: ignore[arg-type]
    if instance_type:
        app_cfg = app_cfg.model_copy(update={"ec2_instance_type": instance_type})

    args_list = shlex.split(cli_args) if cli_args else []
    if "--env" not in args_list:
        args_list.extend(["--env", env])

    spec = EC2JobSpec(
        command=command,
        cli_args=args_list,
        repo_url=repo_url,
        git_ref=git_ref,
        log_group=log_group or app_cfg.ec2_log_group,
        image_tag=image_tag or app_cfg.container_image_tag,
    )

    def _launch():
        runner = EC2JobRunner(app_config=app_cfg)
        instance_id, job_info = runner.launch(spec)
        if wait:
            runner.wait(instance_id)
        return job_info

    _run_or_exit(_launch)


if __name__ == "__main__":
    app()
