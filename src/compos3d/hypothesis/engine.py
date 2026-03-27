from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from heapq import nlargest
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from compos3d.catalog import assets_mentioned_in_prompt, infer_room_type
from compos3d.config import (
    DEFAULT_EXPERIMENT_CONFIG,
    ExperimentConfig,
    TrainingConfig,
    load_experiment_config,
)
from compos3d.data.dataset import load_training_dataset
from compos3d.evaluation.critic import (
    aggregate_prediction_scores,
    build_scene_critic,
    evaluate_scene_program,
)
from compos3d.hypothesis.loop import HypothesisLoopConfig, SceneHypothesisLoop
from compos3d.llm.scene_llm import build_scene_llm
from compos3d.models import (
    AssetSpec,
    ConstraintSpec,
    HypothesisRecord,
    PredictionRecord,
    SceneProgram,
)
from compos3d.procedural.service import (
    BuildSceneRequest,
    build_scene,
    make_training_renderer,
)
from compos3d.schemas.manifest import create_manifest, finalize_manifest
from compos3d.storage.paths import (
    inference_bronze_prefix,
    inference_gold_prefix,
    inference_silver_prefix,
    training_bronze_prefix,
    training_gold_prefix,
    training_silver_prefix,
)

if TYPE_CHECKING:
    from compos3d.storage import AnyStore

InferenceStrategy = Literal["joint_top_k", "filter_and_weight"]


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _run_id(experiment_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{experiment_name}_{ts}"


def _mirror_training_to_lake(
    store: "AnyStore",
    run_id: str,
    run_dir: Path,
    experiment_name: str,
    training_result: dict,
    experiment_config: dict,
) -> list[str]:
    """Push training outputs to bronze/silver/gold layers.  Returns list of written URIs."""
    written: list[str] = []

    bronze_pfx = training_bronze_prefix(run_id)
    silver_pfx = training_silver_prefix(run_id)
    gold_pfx = training_gold_prefix(experiment_name)

    # Bronze: raw per-example scene programs and bank snapshots.
    for prog_file in sorted((run_dir / "programs").glob("*.json")):
        uri = store.put_json(
            f"{bronze_pfx}/programs/{prog_file.name}",
            json.loads(prog_file.read_text()),
        )
        written.append(uri)

    snapshots_dir = run_dir / "bank_snapshots"
    if snapshots_dir.exists():
        for snap in sorted(snapshots_dir.glob("*.json")):
            uri = store.put_json(
                f"{bronze_pfx}/bank_snapshots/{snap.name}",
                json.loads(snap.read_text()),
            )
            written.append(uri)

    # Bronze: training trace + failed scenes for full provenance.
    for fname in (
        "predictions.jsonl",
        "training_trace.jsonl",
        "failed_scene_bank.jsonl",
    ):
        p = run_dir / fname
        if p.exists():
            uri = store.put_bytes(
                f"{bronze_pfx}/{fname}",
                p.read_bytes(),
                content_type="application/x-ndjson",
            )
            written.append(uri)

    # Bronze: raw experiment config.
    uri = store.put_json(f"{bronze_pfx}/experiment_config.json", experiment_config)
    written.append(uri)

    # Silver: validated metrics + final bank.
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        uri = store.put_json(
            f"{silver_pfx}/metrics.json", json.loads(metrics_file.read_text())
        )
        written.append(uri)

    bank_file = run_dir / "hypothesis_bank.json"
    if bank_file.exists():
        uri = store.put_json(
            f"{silver_pfx}/hypothesis_bank.json", json.loads(bank_file.read_text())
        )
        written.append(uri)

    # Silver: full training manifest.
    uri = store.put_json(f"{silver_pfx}/training_manifest.json", training_result)
    written.append(uri)

    # Gold: publish the final bank as the latest artifact for this experiment.
    if bank_file.exists():
        uri = store.put_json(
            f"{gold_pfx}/latest.json", json.loads(bank_file.read_text())
        )
        written.append(uri)

    uri = store.put_json(
        f"{gold_pfx}/training_summary.json",
        {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "silver_bank_path": f"{silver_pfx}/hypothesis_bank.json",
            "metrics": json.loads(metrics_file.read_text())
            if metrics_file.exists()
            else {},
        },
    )
    written.append(uri)

    print(f"[engine] Mirrored {len(written)} artifacts to lake (run_id={run_id})")
    return written


def _mirror_inference_to_lake(
    store: "AnyStore",
    run_id: str,
    output_dir: Path,
    inference_result: dict,
) -> list[str]:
    """Push inference outputs to bronze/silver/gold layers.  Returns list of written URIs."""
    written: list[str] = []

    bronze_pfx = inference_bronze_prefix(run_id)
    silver_pfx = inference_silver_prefix(run_id)
    gold_pfx = inference_gold_prefix(run_id)

    # Bronze: raw scene program.
    sp_file = output_dir / "scene_program.json"
    if sp_file.exists():
        uri = store.put_json(
            f"{bronze_pfx}/scene_program.json", json.loads(sp_file.read_text())
        )
        written.append(uri)

    # Bronze: full inference manifest.
    uri = store.put_json(f"{bronze_pfx}/inference_manifest.json", inference_result)
    written.append(uri)

    # Silver: critic score + scene features.
    for fname in ("critic_score.json", "scene_features.json", "experiment_config.json"):
        p = output_dir / fname
        if p.exists():
            uri = store.put_json(f"{silver_pfx}/{fname}", json.loads(p.read_text()))
            written.append(uri)

    # Silver: rendered views as bytes, if present.
    for img_path in output_dir.rglob("view_*.png"):
        uri = store.put_bytes(
            f"{silver_pfx}/renders/{img_path.name}",
            img_path.read_bytes(),
            content_type="image/png",
        )
        written.append(uri)

    video_path = output_dir / "scene" / "turntable.mp4"
    if video_path.exists():
        uri = store.put_bytes(
            f"{silver_pfx}/renders/turntable.mp4",
            video_path.read_bytes(),
            content_type="video/mp4",
        )
        written.append(uri)

    # Gold: scene features for downstream analysis.
    sf_file = output_dir / "scene_features.json"
    if sf_file.exists():
        uri = store.put_json(
            f"{gold_pfx}/scene_features.json", json.loads(sf_file.read_text())
        )
        written.append(uri)

    print(
        f"[engine] Mirrored {len(written)} inference artifacts to lake (run_id={run_id})"
    )
    return written


def _load_bank(bank_path: Path) -> list[HypothesisRecord]:
    data = json.loads(bank_path.read_text())
    return [HypothesisRecord.model_validate(item) for item in data]


def _text_overlap_score(record: HypothesisRecord, prompt_assets: list[str]) -> int:
    lower_text = record.text.lower()
    return sum(
        1
        for asset in prompt_assets
        if asset in lower_text or asset.replace("_", " ") in lower_text
    )


def _select_hypotheses_for_inference(
    records: list[HypothesisRecord],
    room_type: str,
    *,
    prompt: str,
    top_k: int,
) -> list[HypothesisRecord]:
    prompt_assets = assets_mentioned_in_prompt(prompt, room_type)
    candidates = (record for record in records if record.room_type == room_type)
    return nlargest(
        top_k,
        candidates,
        key=lambda record: (
            record.reward,
            _text_overlap_score(record, prompt_assets),
            record.accuracy,
            len(record.support_example_ids),
            record.mean_score,
        ),
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _normalize_inference_strategy(strategy: str) -> InferenceStrategy:
    normalized = strategy.strip().lower()
    if normalized not in {"joint_top_k", "filter_and_weight"}:
        raise ValueError(
            "inference_strategy must be one of 'joint_top_k' or 'filter_and_weight'"
        )
    return normalized  # type: ignore[return-value]


def _best_accuracy_hypothesis(
    records: list[HypothesisRecord],
) -> HypothesisRecord | None:
    if not records:
        return None
    return max(
        records,
        key=lambda record: (
            record.accuracy,
            record.reward,
            record.mean_score,
            len(record.support_example_ids),
        ),
    )


def _heuristic_filter_hypotheses_for_prompt(
    records: list[HypothesisRecord], *, prompt: str, room_type: str
) -> list[HypothesisRecord]:
    prompt_assets = assets_mentioned_in_prompt(prompt, room_type)
    filtered: list[HypothesisRecord] = []
    for record in records:
        lower_text = record.text.lower()
        overlap = any(
            asset in lower_text or asset.replace("_", " ") in lower_text
            for asset in prompt_assets
        )
        if overlap or room_type.replace("_", " ") in lower_text:
            filtered.append(record)
    return filtered


def _filter_hypotheses_for_inference(
    llm,
    records: list[HypothesisRecord],
    *,
    prompt: str,
    room_type: str,
) -> list[HypothesisRecord]:
    if not records:
        return []

    candidate_texts = [record.text for record in records]
    relevant_texts: list[str] | None = None
    if hasattr(llm, "filter_relevant_hypotheses"):
        try:
            relevant_texts = llm.filter_relevant_hypotheses(
                prompt=prompt,
                room_type=room_type,
                candidate_hypotheses=candidate_texts,
            )
        except Exception:  # noqa: BLE001
            relevant_texts = None

    if relevant_texts is None:
        return _heuristic_filter_hypotheses_for_prompt(
            records,
            prompt=prompt,
            room_type=room_type,
        )

    record_by_text = {_normalize_text(record.text): record for record in records}
    filtered: list[HypothesisRecord] = []
    for text in relevant_texts:
        record = record_by_text.get(_normalize_text(text))
        if record is not None:
            filtered.append(record)
    return filtered


def _hypothesis_vote_weight(record: HypothesisRecord) -> float:
    if record.accuracy > 0:
        return record.accuracy
    if record.mean_score > 0:
        return record.mean_score
    if record.reward > 0:
        return record.reward
    return 1.0


def _fallback_asset_spec(asset_type: str) -> AssetSpec:
    placement = (
        "center of room"
        if asset_type in {"dining_table", "sofa"}
        else "near wall or support surface"
    )
    return AssetSpec(
        asset_type=asset_type,
        count=1,
        placement=placement,
        rationale="Added from weighted filter-and-weight inference.",
    )


def _weighted_vote_scene_program(
    *,
    prompt: str,
    room_type: str,
    records: list[HypothesisRecord],
    candidate_programs: list[SceneProgram],
) -> SceneProgram:
    style_votes: dict[str, float] = defaultdict(float)
    asset_votes: dict[str, float] = defaultdict(float)
    asset_weighted_counts: dict[str, float] = defaultdict(float)
    asset_weight_totals: dict[str, float] = defaultdict(float)
    best_asset_spec: dict[str, tuple[float, AssetSpec]] = {}

    weights = [_hypothesis_vote_weight(record) for record in records]
    total_weight = sum(weights) or float(len(weights) or 1)

    for record, program in zip(records, candidate_programs):
        weight = _hypothesis_vote_weight(record)
        if program.style:
            style_votes[program.style] += weight
        for asset in program.assets:
            asset_votes[asset.asset_type] += weight
            asset_weighted_counts[asset.asset_type] += weight * asset.count
            asset_weight_totals[asset.asset_type] += weight
            best = best_asset_spec.get(asset.asset_type)
            if best is None or weight > best[0]:
                best_asset_spec[asset.asset_type] = (weight, asset)

    prompt_assets = list(dict.fromkeys(assets_mentioned_in_prompt(prompt, room_type)))
    selected_asset_types: list[str] = []
    for asset_type in prompt_assets:
        if asset_type not in selected_asset_types:
            selected_asset_types.append(asset_type)

    ranked_assets = sorted(
        asset_votes,
        key=lambda asset_type: (
            asset_type in prompt_assets,
            asset_votes[asset_type],
            asset_weighted_counts[asset_type],
            asset_type,
        ),
        reverse=True,
    )
    for asset_type in ranked_assets:
        if asset_type in selected_asset_types:
            continue
        if asset_votes[asset_type] >= total_weight / 2:
            selected_asset_types.append(asset_type)

    target_assets = min(4, max(len(prompt_assets), 3))
    for asset_type in ranked_assets:
        if len(selected_asset_types) >= target_assets:
            break
        if asset_type not in selected_asset_types:
            selected_asset_types.append(asset_type)

    if not selected_asset_types and ranked_assets:
        selected_asset_types.append(ranked_assets[0])

    assets: list[AssetSpec] = []
    for asset_type in selected_asset_types[:6]:
        if asset_type in best_asset_spec:
            _, reference_spec = best_asset_spec[asset_type]
            count = max(
                1,
                int(
                    round(
                        asset_weighted_counts[asset_type]
                        / max(asset_weight_totals[asset_type], 1e-6)
                    )
                ),
            )
            assets.append(
                AssetSpec(
                    asset_type=asset_type,
                    count=count,
                    placement=reference_spec.placement,
                    rationale=reference_spec.rationale,
                )
            )
        else:
            assets.append(_fallback_asset_spec(asset_type))

    style = max(style_votes, key=style_votes.get) if style_votes else None
    selected_hypotheses = [record.text for record in records]
    constraints = [ConstraintSpec(text=record.text) for record in records]
    return SceneProgram(
        prompt=prompt,
        room_type=room_type,
        style=style,
        hypotheses=selected_hypotheses,
        assets=assets,
        constraints=constraints,
    )


def _run_filter_and_weight_inference(
    *,
    llm,
    records: list[HypothesisRecord],
    prompt: str,
    room_type: str,
) -> tuple[list[HypothesisRecord], SceneProgram]:
    filtered_records = _filter_hypotheses_for_inference(
        llm,
        records,
        prompt=prompt,
        room_type=room_type,
    )
    if not filtered_records:
        fallback = _best_accuracy_hypothesis(records)
        filtered_records = [fallback] if fallback is not None else []

    candidate_programs: list[SceneProgram] = []
    active_records: list[HypothesisRecord] = []
    for record in filtered_records:
        try:
            candidate_programs.append(
                llm.generate_scene_program(
                    prompt=prompt,
                    room_type=room_type,
                    selected_hypotheses=[record.text],
                )
            )
            active_records.append(record)
        except Exception:  # noqa: BLE001
            continue

    if candidate_programs:
        return active_records, _weighted_vote_scene_program(
            prompt=prompt,
            room_type=room_type,
            records=active_records,
            candidate_programs=candidate_programs,
        )

    selected_hypotheses = [record.text for record in filtered_records]
    scene_program = llm.generate_scene_program(
        prompt=prompt,
        room_type=room_type,
        selected_hypotheses=selected_hypotheses,
    )
    return filtered_records, scene_program


def _training_config_from_args(
    *,
    num_init_examples_per_room: int,
    init_hypotheses_per_room: int,
    top_k: int,
    alpha: float,
    max_num_hypotheses_per_room: int,
    num_wrong_scale: float,
    update_batch_size: int,
    num_hypotheses_to_update: int,
    update_hypotheses_per_batch: int,
    only_best_hypothesis: bool,
    num_epochs: int,
    success_threshold: float,
    save_every_n_examples: int,
    selection_strategy: str,
    use_repair: bool,
    baseline_mode: str | None,
    seed: int,
) -> TrainingConfig:
    return TrainingConfig(
        num_init_examples_per_room=num_init_examples_per_room,
        init_hypotheses_per_room=init_hypotheses_per_room,
        top_k=top_k,
        alpha=alpha,
        max_num_hypotheses_per_room=max_num_hypotheses_per_room,
        num_wrong_scale=num_wrong_scale,
        update_batch_size=update_batch_size,
        num_hypotheses_to_update=num_hypotheses_to_update,
        update_hypotheses_per_batch=update_hypotheses_per_batch,
        only_best_hypothesis=only_best_hypothesis,
        num_epochs=num_epochs,
        success_threshold=success_threshold,
        save_every_n_examples=save_every_n_examples,
        selection_strategy=selection_strategy,
        use_repair=use_repair,
        baseline_mode=baseline_mode,
        seed=seed,
    )


def _experiment_config_from_args(
    *,
    config_path: Path | None,
    llm_provider: str,
    num_init_examples_per_room: int,
    init_hypotheses_per_room: int,
    top_k: int,
    alpha: float,
    max_num_hypotheses_per_room: int,
    num_wrong_scale: float,
    update_batch_size: int,
    num_hypotheses_to_update: int,
    update_hypotheses_per_batch: int,
    only_best_hypothesis: bool,
    num_epochs: int,
    success_threshold: float,
    save_every_n_examples: int,
    selection_strategy: str,
    use_repair: bool,
    baseline_mode: str | None,
    seed: int,
) -> ExperimentConfig:
    if config_path is not None:
        return load_experiment_config(config_path)

    config = DEFAULT_EXPERIMENT_CONFIG.model_copy(deep=True)
    config.generator.provider = llm_provider
    config.training = _training_config_from_args(
        num_init_examples_per_room=num_init_examples_per_room,
        init_hypotheses_per_room=init_hypotheses_per_room,
        top_k=top_k,
        alpha=alpha,
        max_num_hypotheses_per_room=max_num_hypotheses_per_room,
        num_wrong_scale=num_wrong_scale,
        update_batch_size=update_batch_size,
        num_hypotheses_to_update=num_hypotheses_to_update,
        update_hypotheses_per_batch=update_hypotheses_per_batch,
        only_best_hypothesis=only_best_hypothesis,
        num_epochs=num_epochs,
        success_threshold=success_threshold,
        save_every_n_examples=save_every_n_examples,
        selection_strategy=selection_strategy,
        use_repair=use_repair,
        baseline_mode=baseline_mode,
        seed=seed,
    )
    return config


def _loop_config_from_experiment(
    experiment_config: ExperimentConfig,
) -> HypothesisLoopConfig:
    t = experiment_config.training
    return HypothesisLoopConfig(
        num_init_examples_per_room=t.num_init_examples_per_room,
        init_hypotheses_per_room=t.init_hypotheses_per_room,
        k=t.top_k,
        alpha=t.alpha,
        max_num_hypotheses_per_room=t.max_num_hypotheses_per_room,
        num_wrong_scale=t.num_wrong_scale,
        update_batch_size=t.update_batch_size,
        num_hypotheses_to_update=t.num_hypotheses_to_update,
        update_hypotheses_per_batch=t.update_hypotheses_per_batch,
        only_best_hypothesis=t.only_best_hypothesis,
        num_epochs=t.num_epochs,
        success_threshold=t.success_threshold,
        save_every_n_examples=t.save_every_n_examples,
        selection_strategy=t.selection_strategy,
        use_repair=t.use_repair,
        baseline_mode=t.baseline_mode,
        seed=t.seed,
    )


def train_vertical_slice(
    *,
    dataset_path: Path,
    output_dir: Path,
    experiment_name: str,
    llm_provider: str = "mock",
    config_path: Path | None = None,
    num_init_examples_per_room: int = 1,
    init_hypotheses_per_room: int = 3,
    top_k: int = 2,
    alpha: float = 0.5,
    max_num_hypotheses_per_room: int = 8,
    num_wrong_scale: float = 0.8,
    update_batch_size: int = 1,
    num_hypotheses_to_update: int = 1,
    update_hypotheses_per_batch: int = 2,
    only_best_hypothesis: bool = False,
    num_epochs: int = 2,
    success_threshold: float = 0.65,
    save_every_n_examples: int = 5,
    selection_strategy: str = "ucb",
    use_repair: bool = True,
    baseline_mode: str | None = None,
    seed: int = 42,
    store: "AnyStore | None" = None,
    compute_platform: str = "local",
    instance_type: str | None = None,
) -> dict:
    dataset = load_training_dataset(dataset_path)
    experiment_config = _experiment_config_from_args(
        config_path=config_path,
        llm_provider=llm_provider,
        num_init_examples_per_room=num_init_examples_per_room,
        init_hypotheses_per_room=init_hypotheses_per_room,
        top_k=top_k,
        alpha=alpha,
        max_num_hypotheses_per_room=max_num_hypotheses_per_room,
        num_wrong_scale=num_wrong_scale,
        update_batch_size=update_batch_size,
        num_hypotheses_to_update=num_hypotheses_to_update,
        update_hypotheses_per_batch=update_hypotheses_per_batch,
        only_best_hypothesis=only_best_hypothesis,
        num_epochs=num_epochs,
        success_threshold=success_threshold,
        save_every_n_examples=save_every_n_examples,
        selection_strategy=selection_strategy,
        use_repair=use_repair,
        baseline_mode=baseline_mode,
        seed=seed,
    )

    # Apply room_types filter if set in config
    if experiment_config.room_types:
        allowed = set(experiment_config.room_types)
        dataset = dataset.model_copy(
            update={
                "examples": [ex for ex in dataset.examples if ex.room_type in allowed]
            }
        )

    llm = build_scene_llm(experiment_config.generator)
    critic = build_scene_critic(experiment_config.critic)
    renderer = make_training_renderer(experiment_config.render)
    run_dir = output_dir / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if renderer is not None:
        print(
            f"[engine] Render enabled: resolution={experiment_config.render.resolution}, "
            f"view_samples={experiment_config.render.view_samples}"
        )

    config = _loop_config_from_experiment(experiment_config)
    loop = SceneHypothesisLoop(
        dataset=dataset,
        llm=llm,
        critic=critic,
        run_dir=run_dir,
        llm_provider=experiment_config.generator.provider,
        config=config,
        experiment_config=experiment_config.model_dump(),
        config_path=str(config_path) if config_path is not None else None,
        renderer=renderer,
    )
    result = loop.train()

    if store is not None:
        run_id = _run_id(experiment_name)
        manifest = create_manifest(
            run_id=run_id,
            run_type="training",
            config_snapshot=experiment_config.model_dump(),
            compute_platform=compute_platform,  # type: ignore[arg-type]
            instance_type=instance_type,
            generator_model=experiment_config.generator.model_id,
            critic_model=experiment_config.critic.model_id,
            input_paths=[str(dataset_path)],
        )
        try:
            uris = _mirror_training_to_lake(
                store=store,
                run_id=run_id,
                run_dir=run_dir,
                experiment_name=experiment_name,
                training_result=result,
                experiment_config=experiment_config.model_dump(),
            )
            manifest = finalize_manifest(manifest, status="success", output_uris=uris)
        except Exception as exc:
            manifest = finalize_manifest(
                manifest, status="failed", error_message=str(exc)
            )
            raise
        finally:
            store.put_json(
                f"bronze/training/{run_id}/run_manifest.json",
                manifest.model_dump(mode="json"),
            )

        result["lake_run_id"] = run_id
        result["lake_output_uris"] = uris

    return result


def run_vertical_inference(
    *,
    bank_path: Path,
    prompt: str,
    output_dir: Path,
    llm_provider: str = "mock",
    config_path: Path | None = None,
    top_k: int = 2,
    inference_strategy: str = "joint_top_k",
    render_scene: bool = False,
    render_resolution: str = "512x512",
    render_view_samples: int = 48,
    render_video_frames: int = 90,
    render_video_samples: int = 16,
    store: "AnyStore | None" = None,
    compute_platform: str = "local",
    instance_type: str | None = None,
) -> dict:
    bank = _load_bank(bank_path)
    resolved_inference_strategy = _normalize_inference_strategy(inference_strategy)
    experiment_config = (
        load_experiment_config(config_path)
        if config_path is not None
        else DEFAULT_EXPERIMENT_CONFIG.model_copy(deep=True)
    )
    if config_path is None:
        experiment_config.generator.provider = llm_provider
        experiment_config.training.top_k = top_k
    llm = build_scene_llm(experiment_config.generator)
    critic = build_scene_critic(experiment_config.critic)
    room_type = infer_room_type(prompt)
    candidates = _select_hypotheses_for_inference(
        bank, room_type, prompt=prompt, top_k=experiment_config.training.top_k
    )
    if resolved_inference_strategy == "filter_and_weight":
        selected, scene_program = _run_filter_and_weight_inference(
            llm=llm,
            records=candidates,
            prompt=prompt,
            room_type=room_type,
        )
    else:
        selected = candidates
        selected_text = [item.text for item in selected]
        scene_program = llm.generate_scene_program(
            prompt=prompt, room_type=room_type, selected_hypotheses=selected_text
        )
    selected_text = [item.text for item in selected]

    output_dir.mkdir(parents=True, exist_ok=True)
    sp_path = output_dir / "scene_program.json"
    _write_json(sp_path, scene_program.model_dump())

    # --- Optional: render the scene and score with VLM images ---
    render_manifest: dict = {}
    image_paths: list[Path] = []
    if render_scene:
        print(
            f"[engine] Building 3D scene for inference output → {output_dir / 'scene'}"
        )
        render_out = output_dir / "scene"
        req = BuildSceneRequest(
            scene_program_path=sp_path,
            output_dir=render_out,
            resolution=render_resolution,
            view_samples=render_view_samples,
            video_frames=render_video_frames,
            video_samples=render_video_samples,
            no_video=False,
            save_blend=False,
        )
        render_manifest = build_scene(req)
        image_paths = [
            Path(p)
            for p in render_manifest.get("rendered_views", [])
            if p and Path(p).exists()
        ]

    critic_score = evaluate_scene_program(
        scene_program, critic=critic, image_paths=image_paths or None
    )
    _write_json(output_dir / "critic_score.json", critic_score.model_dump())
    _write_json(output_dir / "experiment_config.json", experiment_config.model_dump())

    # --- Scene features: lightweight structured metadata from the SceneProgram ---
    scene_features = {
        "room_type": room_type,
        "asset_counts": {
            asset.asset_type: asset.count for asset in scene_program.assets
        },
        "total_assets": sum(asset.count for asset in scene_program.assets),
        "rendered_views": render_manifest.get("rendered_views"),
        "video_path": render_manifest.get("video_path"),
        "render_elapsed_seconds": render_manifest.get("elapsed_seconds"),
    }
    _write_json(output_dir / "scene_features.json", scene_features)

    manifest = {
        "bank_path": str(bank_path),
        "prompt": prompt,
        "room_type": room_type,
        "llm_provider": experiment_config.generator.provider,
        "inference_strategy": resolved_inference_strategy,
        "config_path": str(config_path) if config_path is not None else None,
        "experiment_config": experiment_config.model_dump(),
        "candidate_hypothesis_ids": [item.hypothesis_id for item in candidates],
        "candidate_hypotheses": [item.text for item in candidates],
        "selected_hypothesis_ids": [item.hypothesis_id for item in selected],
        "selected_hypotheses": selected_text,
        "scene_program_path": str(sp_path),
        "critic_score_path": str(output_dir / "critic_score.json"),
        "scene_features_path": str(output_dir / "scene_features.json"),
        "render_scene": render_scene,
        "render_manifest": render_manifest if render_scene else None,
    }
    _write_json(output_dir / "inference_manifest.json", manifest)

    if store is not None:
        inf_run_id = _run_id("inference")
        run_manifest = create_manifest(
            run_id=inf_run_id,
            run_type="inference",
            config_snapshot=experiment_config.model_dump(),
            compute_platform=compute_platform,  # type: ignore[arg-type]
            instance_type=instance_type,
            generator_model=experiment_config.generator.model_id,
            critic_model=experiment_config.critic.model_id,
            input_paths=[str(bank_path)],
        )
        try:
            uris = _mirror_inference_to_lake(
                store=store,
                run_id=inf_run_id,
                output_dir=output_dir,
                inference_result=manifest,
            )
            run_manifest = finalize_manifest(
                run_manifest, status="success", output_uris=uris
            )
        except Exception as exc:
            run_manifest = finalize_manifest(
                run_manifest, status="failed", error_message=str(exc)
            )
            raise
        finally:
            store.put_json(
                f"bronze/inference/{inf_run_id}/run_manifest.json",
                run_manifest.model_dump(mode="json"),
            )

        manifest["lake_run_id"] = inf_run_id
        manifest["lake_output_uris"] = uris

    return manifest


def evaluate_prediction_dir(*, predictions_dir: Path, output_dir: Path) -> dict:
    predictions_file = predictions_dir / "predictions.jsonl"
    manifest_file = predictions_dir / "inference_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if predictions_file.exists():
        rows = [
            json.loads(line)
            for line in predictions_file.read_text().splitlines()
            if line.strip()
        ]
        predictions = [PredictionRecord.model_validate(row) for row in rows]
        payload = aggregate_prediction_scores(predictions).model_dump()
    elif manifest_file.exists():
        critic_score = json.loads((predictions_dir / "critic_score.json").read_text())
        payload = {
            "num_predictions": 1,
            "average_validity": critic_score["validity"],
            "average_prompt_adherence": critic_score["prompt_adherence"],
            "average_asset_precision": critic_score["asset_precision"],
            "average_asset_recall": critic_score["asset_recall"],
            "average_room_match": critic_score["room_match"],
            "average_overall": critic_score["overall"],
        }
    else:
        raise FileNotFoundError(
            f"Could not find predictions.jsonl or inference_manifest.json in {predictions_dir}"
        )

    _write_json(output_dir / "evaluation_summary.json", payload)
    return payload
