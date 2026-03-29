from __future__ import annotations

import csv
import json
import random
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from compos3d.config import CriticConfig
from compos3d.evaluation.critic import CriticUnavailableError
from compos3d.hypothesis.engine import run_vertical_inference
from compos3d.paper.benchmarks import (
    build_program_metric_row,
    compute_edit_program_metrics,
    load_benchmark_examples,
    load_edit_pairs,
    summarize_generation_rows,
    write_json,
    write_jsonl,
)
from compos3d.paper.judges import BedrockPairwiseJudge, clip_directional_similarity


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_results_file(path: Path, filename: str) -> Path:
    if path.is_dir():
        return path / filename
    return path


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in overrides.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
            and merged.get(key) is not None
        ):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def write_experiment_config(
    *,
    base_config_path: Path,
    output_path: Path,
    overrides: dict[str, Any],
) -> Path:
    base_config = json.loads(base_config_path.read_text())
    write_json(output_path, _deep_merge(base_config, overrides))
    return output_path


def _maybe_materialize_config(
    *,
    config_path: Path | None,
    config_overrides: dict[str, Any] | None,
) -> tuple[Path | None, tempfile.TemporaryDirectory[str] | None]:
    if config_path is None or not config_overrides:
        return config_path, None
    tmp_dir = tempfile.TemporaryDirectory(prefix="compos3d_paper_cfg_")
    resolved = Path(tmp_dir.name) / "experiment_config.json"
    write_experiment_config(
        base_config_path=config_path,
        output_path=resolved,
        overrides=config_overrides,
    )
    return resolved, tmp_dir


def _empty_bank_path() -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    tmp_dir = tempfile.TemporaryDirectory(prefix="compos3d_empty_bank_")
    bank_path = Path(tmp_dir.name) / "hypothesis_bank.json"
    bank_path.write_text("[]\n", encoding="utf-8")
    return bank_path, tmp_dir


def _normalize_image_paths(payload: dict[str, Any]) -> list[str]:
    image_paths = payload.get("image_paths")
    if isinstance(image_paths, list):
        return [str(path) for path in image_paths]
    return []


def run_generation_benchmark(
    *,
    benchmark_path: Path,
    output_dir: Path,
    method_name: str,
    bank_path: Path | None = None,
    use_empty_bank: bool = False,
    config_path: Path | None = None,
    config_overrides: dict[str, Any] | None = None,
    llm_provider: str = "mock",
    top_k: int = 2,
    inference_strategy: str = "joint_top_k",
    render_scene: bool = False,
    render_resolution: str = "512x512",
    render_view_samples: int = 48,
    render_video_frames: int = 90,
    render_video_samples: int = 16,
    example_ids: set[str] | None = None,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    examples = load_benchmark_examples(benchmark_path)
    if example_ids is not None:
        examples = [
            example for example in examples if str(example["example_id"]) in example_ids
        ]
    if limit is not None:
        examples = examples[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_bank: tempfile.TemporaryDirectory[str] | None = None
    temp_config: tempfile.TemporaryDirectory[str] | None = None
    resolved_bank_path = bank_path
    if use_empty_bank:
        resolved_bank_path, temp_bank = _empty_bank_path()
    if resolved_bank_path is None:
        raise ValueError("bank_path is required unless use_empty_bank=True.")

    resolved_config_path, temp_config = _maybe_materialize_config(
        config_path=config_path,
        config_overrides=config_overrides,
    )
    rows: list[dict[str, Any]] = []
    try:
        for example in examples:
            example_id = str(example["example_id"])
            run_dir = output_dir / "runs" / example_id
            manifest_path = run_dir / "inference_manifest.json"
            if force or not manifest_path.exists():
                run_vertical_inference(
                    bank_path=resolved_bank_path,
                    prompt=str(example["prompt"]),
                    output_dir=run_dir,
                    llm_provider=llm_provider,
                    config_path=resolved_config_path,
                    top_k=top_k,
                    inference_strategy=inference_strategy,
                    render_scene=render_scene,
                    render_resolution=render_resolution,
                    render_view_samples=render_view_samples,
                    render_video_frames=render_video_frames,
                    render_video_samples=render_video_samples,
                )

            scene_program = _read_json(run_dir / "scene_program.json")
            critic_score = _read_json(run_dir / "critic_score.json")
            manifest = _read_json(manifest_path)
            row = {
                "method_name": method_name,
                "example_id": example_id,
                "prompt": str(example["prompt"]),
                "room_type": str(example["room_type"]),
                "required_assets": list(example.get("required_assets", [])),
                "asset_tuple": list(
                    example.get("asset_tuple", example.get("required_assets", []))
                ),
                "output_dir": str(run_dir),
                "scene_program_path": str(run_dir / "scene_program.json"),
                "critic_score_path": str(run_dir / "critic_score.json"),
                "selected_hypothesis_ids": list(
                    manifest.get("selected_hypothesis_ids", [])
                ),
                "selected_hypotheses": list(manifest.get("selected_hypotheses", [])),
                "candidate_hypothesis_ids": list(
                    manifest.get("candidate_hypothesis_ids", [])
                ),
                "candidate_hypotheses": list(manifest.get("candidate_hypotheses", [])),
                "inference_strategy": str(manifest.get("inference_strategy")),
                "render_scene": bool(manifest.get("render_scene")),
                "image_paths": _normalize_image_paths(scene_program)
                or [
                    str(path)
                    for path in critic_score.get("used_image_paths", [])
                    if path
                ],
                "validity": float(critic_score["validity"]),
                "prompt_adherence": float(critic_score["prompt_adherence"]),
                "asset_precision": float(critic_score["asset_precision"]),
                "asset_recall": float(critic_score["asset_recall"]),
                "room_match": float(critic_score["room_match"]),
                "overall": float(critic_score["overall"]),
                "critic_notes": list(critic_score.get("notes", [])),
            }
            row.update(
                build_program_metric_row(example=example, scene_program=scene_program)
            )
            rows.append(row)
    finally:
        if temp_bank is not None:
            temp_bank.cleanup()
        if temp_config is not None:
            temp_config.cleanup()

    results_path = output_dir / "generation_results.jsonl"
    summary_path = output_dir / "summary.json"
    summary = summarize_generation_rows(rows)
    write_jsonl(results_path, rows)
    write_json(
        summary_path,
        {
            "method_name": method_name,
            "benchmark_path": str(benchmark_path),
            "bank_path": str(resolved_bank_path),
            "results_path": str(results_path),
            "summary": summary,
        },
    )
    return {"rows": rows, "summary": summary, "results_path": results_path}


def _pairwise_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        winner = str(row.get(key, "unknown"))
        counts[winner] += 1
    return dict(counts)


def _pairwise_metric_summary(
    rows: list[dict[str, Any]], key: str, *, preferred_label: str = "A"
) -> dict[str, Any]:
    counts = _pairwise_counts(rows, key)
    total = len(rows)
    preferred_wins = counts.get(preferred_label, 0)
    tie_count = counts.get("tie", 0)
    other_wins = total - preferred_wins - tie_count

    def _rate(value: int) -> float:
        return round(value / total, 4) if total else 0.0

    non_tie_total = preferred_wins + other_wins
    return {
        "counts": counts,
        "preferred_win_rate": _rate(preferred_wins),
        "other_win_rate": _rate(other_wins),
        "tie_rate": _rate(tie_count),
        "preferred_score": round((preferred_wins + (0.5 * tie_count)) / total, 4)
        if total
        else 0.0,
        "preferred_non_tie_win_rate": round(preferred_wins / non_tie_total, 4)
        if non_tie_total
        else None,
    }


def judge_generation_pairwise(
    *,
    method_a_results_path: Path,
    method_b_results_path: Path,
    output_dir: Path,
    judge_config: CriticConfig,
    sample_size: int = 50,
    seed: int = 42,
    force: bool = False,
) -> dict[str, Any]:
    results_a = load_jsonl_rows(
        _resolve_results_file(method_a_results_path, "generation_results.jsonl")
    )
    results_b = load_jsonl_rows(
        _resolve_results_file(method_b_results_path, "generation_results.jsonl")
    )
    by_id_a = {str(row["example_id"]): row for row in results_a}
    by_id_b = {str(row["example_id"]): row for row in results_b}
    shared_ids = sorted(
        example_id
        for example_id in set(by_id_a) & set(by_id_b)
        if by_id_a[example_id].get("image_paths")
        and by_id_b[example_id].get("image_paths")
    )
    if not shared_ids:
        raise ValueError(
            "No shared generation examples with rendered images were found."
        )

    rng = random.Random(seed)
    chosen_ids = sorted(rng.sample(shared_ids, min(sample_size, len(shared_ids))))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "pairwise_generation.jsonl"
    existing_rows = [] if force else load_jsonl_rows(results_path)
    existing_by_id = {
        str(row["example_id"]): row
        for row in existing_rows
        if str(row.get("example_id", "")) in chosen_ids
    }
    if len(existing_by_id) == len(chosen_ids):
        ordered_rows = [existing_by_id[example_id] for example_id in chosen_ids]
        summary = _summarize_generation_pairwise(ordered_rows)
        write_json(output_dir / "pairwise_generation_summary.json", summary)
        return {
            "rows": ordered_rows,
            "summary": summary,
            "results_path": results_path,
        }

    judge = BedrockPairwiseJudge(judge_config)
    rows_by_id = dict(existing_by_id)
    views_per_candidate = 2
    max_attempts = 8
    for example_id in chosen_ids:
        if example_id in rows_by_id:
            continue
        row_a = by_id_a[example_id]
        row_b = by_id_b[example_id]
        candidate_a_paths = [
            Path(path) for path in row_a["image_paths"][:views_per_candidate]
        ]
        candidate_b_paths = [
            Path(path) for path in row_b["image_paths"][:views_per_candidate]
        ]
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                judgment = judge.judge_generation_pair(
                    prompt=str(row_a["prompt"]),
                    candidate_a_paths=candidate_a_paths,
                    candidate_b_paths=candidate_b_paths,
                    method_a=str(row_a["method_name"]),
                    method_b=str(row_b["method_name"]),
                )
                rows_by_id[example_id] = {
                    "example_id": example_id,
                    "prompt": row_a["prompt"],
                    "method_a": row_a["method_name"],
                    "method_b": row_b["method_name"],
                    "candidate_a_paths": [str(path) for path in candidate_a_paths],
                    "candidate_b_paths": [str(path) for path in candidate_b_paths],
                    **judgment,
                }
                write_jsonl(
                    results_path,
                    [
                        rows_by_id[item_id]
                        for item_id in chosen_ids
                        if item_id in rows_by_id
                    ],
                )
                break
            except CriticUnavailableError as exc:
                last_error = exc
                if attempt == max_attempts - 1:
                    raise
                time.sleep(min(8, 2**attempt))
        if last_error is not None and example_id not in rows_by_id:
            raise last_error

    rows = [rows_by_id[example_id] for example_id in chosen_ids]
    write_jsonl(results_path, rows)
    summary = _summarize_generation_pairwise(rows)
    write_json(output_dir / "pairwise_generation_summary.json", summary)
    return {"rows": rows, "summary": summary, "results_path": results_path}


def _summarize_generation_pairwise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    method_a = str(rows[0].get("method_a")) if rows else None
    method_b = str(rows[0].get("method_b")) if rows else None
    asset_selection = _pairwise_metric_summary(rows, "asset_selection")
    layout_coherence = _pairwise_metric_summary(rows, "layout_coherence")
    overall_preference = _pairwise_metric_summary(rows, "overall_preference")
    return {
        "num_examples": len(rows),
        "method_a": method_a,
        "method_b": method_b,
        "headline_metrics": {
            "generation_quality_score": overall_preference["preferred_score"],
            "generation_quality_win_rate": overall_preference["preferred_win_rate"],
            "generation_quality_non_tie_win_rate": overall_preference[
                "preferred_non_tie_win_rate"
            ],
            "asset_selection_score": asset_selection["preferred_score"],
            "layout_coherence_score": layout_coherence["preferred_score"],
        },
        "asset_selection": asset_selection,
        "layout_coherence": layout_coherence,
        "overall_preference": overall_preference,
    }


def run_edit_benchmark(
    *,
    edit_pairs_path: Path,
    output_dir: Path,
    method_name: str,
    bank_path: Path | None = None,
    use_empty_bank: bool = False,
    config_path: Path | None = None,
    config_overrides: dict[str, Any] | None = None,
    llm_provider: str = "mock",
    top_k: int = 2,
    inference_strategy: str = "filter_and_weight",
    render_scene: bool = False,
    render_resolution: str = "512x512",
    render_view_samples: int = 48,
    render_video_frames: int = 90,
    render_video_samples: int = 16,
    judge_config: CriticConfig | None = None,
    pair_ids: set[str] | None = None,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    pairs = load_edit_pairs(edit_pairs_path)
    if pair_ids is not None:
        pairs = [pair for pair in pairs if str(pair["pair_id"]) in pair_ids]
    if limit is not None:
        pairs = pairs[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_bank: tempfile.TemporaryDirectory[str] | None = None
    temp_config: tempfile.TemporaryDirectory[str] | None = None
    resolved_bank_path = bank_path
    if use_empty_bank:
        resolved_bank_path, temp_bank = _empty_bank_path()
    if resolved_bank_path is None:
        raise ValueError("bank_path is required unless use_empty_bank=True.")

    resolved_config_path, temp_config = _maybe_materialize_config(
        config_path=config_path,
        config_overrides=config_overrides,
    )
    judge = BedrockPairwiseJudge(judge_config) if judge_config is not None else None
    rows: list[dict[str, Any]] = []
    try:
        for pair in pairs:
            pair_id = str(pair["pair_id"])
            base_run_dir = output_dir / "runs" / pair_id / "base"
            edited_run_dir = output_dir / "runs" / pair_id / "edited"
            if force or not (base_run_dir / "inference_manifest.json").exists():
                run_vertical_inference(
                    bank_path=resolved_bank_path,
                    prompt=str(pair["base_prompt"]),
                    output_dir=base_run_dir,
                    llm_provider=llm_provider,
                    config_path=resolved_config_path,
                    top_k=top_k,
                    inference_strategy=inference_strategy,
                    render_scene=render_scene,
                    render_resolution=render_resolution,
                    render_view_samples=render_view_samples,
                    render_video_frames=render_video_frames,
                    render_video_samples=render_video_samples,
                )
            if force or not (edited_run_dir / "inference_manifest.json").exists():
                run_vertical_inference(
                    bank_path=resolved_bank_path,
                    prompt=str(pair["edited_prompt"]),
                    output_dir=edited_run_dir,
                    llm_provider=llm_provider,
                    config_path=resolved_config_path,
                    top_k=top_k,
                    inference_strategy=inference_strategy,
                    render_scene=render_scene,
                    render_resolution=render_resolution,
                    render_view_samples=render_view_samples,
                    render_video_frames=render_video_frames,
                    render_video_samples=render_video_samples,
                )

            base_program = _read_json(base_run_dir / "scene_program.json")
            edited_program = _read_json(edited_run_dir / "scene_program.json")
            base_critic = _read_json(base_run_dir / "critic_score.json")
            edited_critic = _read_json(edited_run_dir / "critic_score.json")
            base_images = [
                str(path) for path in base_critic.get("used_image_paths", []) if path
            ]
            edited_images = [
                str(path) for path in edited_critic.get("used_image_paths", []) if path
            ]

            row = {
                "method_name": method_name,
                "pair_id": pair_id,
                "edit_type": str(pair["edit_type"]),
                "base_example_id": str(pair["base_example_id"]),
                "edited_example_id": str(pair["edited_example_id"]),
                "base_prompt": str(pair["base_prompt"]),
                "edited_prompt": str(pair["edited_prompt"]),
                "edit_instruction": str(pair["edit_instruction"]),
                "changed_assets": list(pair.get("changed_assets", [])),
                "unchanged_assets": list(pair.get("unchanged_assets", [])),
                "base_image_paths": base_images,
                "edited_image_paths": edited_images,
                "base_overall": float(base_critic["overall"]),
                "edited_overall": float(edited_critic["overall"]),
                "base_validity": float(base_critic["validity"]),
                "edited_validity": float(edited_critic["validity"]),
            }
            base_metric_row = build_program_metric_row(
                example={
                    "prompt": pair["base_prompt"],
                    "room_type": "dining_room",
                    "required_assets": pair["base_required_assets"],
                    "expected_asset_counts": pair["base_expected_asset_counts"],
                },
                scene_program=base_program,
            )
            edited_metric_row = build_program_metric_row(
                example={
                    "prompt": pair["edited_prompt"],
                    "room_type": "dining_room",
                    "required_assets": pair["edited_required_assets"],
                    "expected_asset_counts": pair["edited_expected_asset_counts"],
                },
                scene_program=edited_program,
            )
            row.update({f"base_{key}": value for key, value in base_metric_row.items()})
            row.update(
                {f"edited_{key}": value for key, value in edited_metric_row.items()}
            )
            row.update(
                compute_edit_program_metrics(
                    pair=pair,
                    base_scene_program=base_program,
                    edited_scene_program=edited_program,
                )
            )

            if judge is not None and base_images and edited_images:
                judgment = judge.judge_edit_pair(
                    base_prompt=str(pair["base_prompt"]),
                    edited_prompt=str(pair["edited_prompt"]),
                    edit_instruction=str(pair["edit_instruction"]),
                    before_paths=[Path(path) for path in base_images[:4]],
                    after_paths=[Path(path) for path in edited_images[:4]],
                )
                row["vlm_edit_success"] = float(judgment["edit_success"])
                row["vlm_preservation"] = float(judgment["preservation"])
                row["vlm_overall"] = float(judgment["overall"])
                row["vlm_notes"] = str(judgment.get("notes", ""))
            else:
                row["vlm_edit_success"] = None
                row["vlm_preservation"] = None
                row["vlm_overall"] = None
                row["vlm_notes"] = None

            row["clip_directional_similarity"] = clip_directional_similarity(
                before_image_paths=[Path(path) for path in base_images[:4]],
                after_image_paths=[Path(path) for path in edited_images[:4]],
                before_text=str(pair["base_prompt"]),
                after_text=str(pair["edited_prompt"]),
            )
            rows.append(row)
    finally:
        if temp_bank is not None:
            temp_bank.cleanup()
        if temp_config is not None:
            temp_config.cleanup()

    results_path = output_dir / "editing_results.jsonl"
    summary = summarize_edit_rows(rows)
    write_jsonl(results_path, rows)
    write_json(
        output_dir / "summary.json",
        {
            "method_name": method_name,
            "edit_pairs_path": str(edit_pairs_path),
            "bank_path": str(resolved_bank_path),
            "results_path": str(results_path),
            "summary": summary,
        },
    )
    return {"rows": rows, "summary": summary, "results_path": results_path}


def summarize_edit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "delta_asset_precision",
        "delta_asset_recall",
        "delta_asset_f1",
        "delta_count_l1",
        "unchanged_asset_retention_rate",
        "base_overall",
        "edited_overall",
        "vlm_edit_success",
        "vlm_preservation",
        "vlm_overall",
        "clip_directional_similarity",
    ]
    summary = {
        "num_examples": len(rows),
        "metrics": _mean_metrics(rows, numeric_keys),
        "by_edit_type": {},
    }
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("edit_type", "unknown"))].append(row)
    for edit_type, bucket in sorted(by_type.items()):
        summary["by_edit_type"][edit_type] = {
            "num_examples": len(bucket),
            "metrics": _mean_metrics(bucket, numeric_keys),
        }
    return summary


def _mean_metrics(
    rows: list[dict[str, Any]], numeric_keys: list[str]
) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        summary[key] = round(sum(values) / len(values), 4) if values else None
    return summary


def export_generation_human_study_sheet(
    *,
    method_a_results_path: Path,
    method_b_results_path: Path,
    output_csv_path: Path,
    sample_size: int = 30,
    seed: int = 42,
) -> Path:
    rows_a = load_jsonl_rows(
        _resolve_results_file(method_a_results_path, "generation_results.jsonl")
    )
    rows_b = load_jsonl_rows(
        _resolve_results_file(method_b_results_path, "generation_results.jsonl")
    )
    by_id_a = {str(row["example_id"]): row for row in rows_a}
    by_id_b = {str(row["example_id"]): row for row in rows_b}
    shared_ids = sorted(
        example_id
        for example_id in set(by_id_a) & set(by_id_b)
        if by_id_a[example_id].get("image_paths")
        and by_id_b[example_id].get("image_paths")
    )
    rng = random.Random(seed)
    chosen_ids = sorted(rng.sample(shared_ids, min(sample_size, len(shared_ids))))
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "example_id",
                "prompt",
                "left_method",
                "left_image_path",
                "right_method",
                "right_image_path",
                "asset_selection_winner",
                "layout_coherence_winner",
                "overall_preference",
                "rater_id",
                "notes",
            ],
        )
        writer.writeheader()
        for example_id in chosen_ids:
            row_a = by_id_a[example_id]
            row_b = by_id_b[example_id]
            left_row, right_row = (
                (row_a, row_b) if rng.random() < 0.5 else (row_b, row_a)
            )
            writer.writerow(
                {
                    "example_id": example_id,
                    "prompt": row_a["prompt"],
                    "left_method": left_row["method_name"],
                    "left_image_path": left_row["image_paths"][0],
                    "right_method": right_row["method_name"],
                    "right_image_path": right_row["image_paths"][0],
                    "asset_selection_winner": "",
                    "layout_coherence_winner": "",
                    "overall_preference": "",
                    "rater_id": "",
                    "notes": "",
                }
            )
    return output_csv_path


def export_edit_human_study_sheet(
    *,
    editing_results_path: Path,
    output_csv_path: Path,
    sample_size: int = 30,
    seed: int = 42,
) -> Path:
    rows = load_jsonl_rows(
        _resolve_results_file(editing_results_path, "editing_results.jsonl")
    )
    eligible = [
        row
        for row in rows
        if row.get("base_image_paths") and row.get("edited_image_paths")
    ]
    rng = random.Random(seed)
    chosen = rng.sample(eligible, min(sample_size, len(eligible)))
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "edit_type",
                "base_prompt",
                "edited_prompt",
                "edit_instruction",
                "base_image_path",
                "edited_image_path",
                "edit_success",
                "preservation",
                "rater_id",
                "notes",
            ],
        )
        writer.writeheader()
        for row in chosen:
            writer.writerow(
                {
                    "pair_id": row["pair_id"],
                    "edit_type": row["edit_type"],
                    "base_prompt": row["base_prompt"],
                    "edited_prompt": row["edited_prompt"],
                    "edit_instruction": row["edit_instruction"],
                    "base_image_path": row["base_image_paths"][0],
                    "edited_image_path": row["edited_image_paths"][0],
                    "edit_success": "",
                    "preservation": "",
                    "rater_id": "",
                    "notes": "",
                }
            )
    return output_csv_path


def build_paper_training_matrix(
    *,
    base_config_path: Path,
    output_dir: Path,
    dataset_path: Path,
    benchmark_val_path: Path,
    benchmark_test_path: Path,
    training_output_dir: Path = Path("artifacts/training"),
    experiment_prefix: str = "paper_dining",
    final_bank_path: Path = Path("artifacts/training/claude_qwen/hypothesis_bank.json"),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    ablations = [
        ("full_system", {}),
        ("no_hypotheses", {"training": {"baseline_mode": "no_hypotheses"}}),
        ("fixed_hypotheses", {"training": {"baseline_mode": "fixed_hypotheses"}}),
        ("no_repair", {"training": {"use_repair": False}}),
        ("greedy", {"training": {"selection_strategy": "greedy"}}),
        ("random", {"training": {"selection_strategy": "random"}}),
    ]
    sweep_specs = [
        ("alpha_0_0", {"training": {"alpha": 0.0}}),
        ("alpha_0_25", {"training": {"alpha": 0.25}}),
        ("alpha_0_5", {"training": {"alpha": 0.5}}),
        ("alpha_1_0", {"training": {"alpha": 1.0}}),
        ("num_wrong_scale_0_4", {"training": {"num_wrong_scale": 0.4}}),
        ("num_wrong_scale_0_8", {"training": {"num_wrong_scale": 0.8}}),
        ("num_wrong_scale_1_2", {"training": {"num_wrong_scale": 1.2}}),
        ("success_threshold_0_65", {"training": {"success_threshold": 0.65}}),
        ("success_threshold_0_70", {"training": {"success_threshold": 0.7}}),
        ("success_threshold_0_75", {"training": {"success_threshold": 0.75}}),
    ]
    inference_ablations = [
        {
            "name": f"filter_and_weight_topk_{top_k}",
            "inference_strategy": "filter_and_weight",
            "config_overrides": {"training": {"top_k": top_k}},
        }
        for top_k in (1, 2, 3, 4)
    ] + [
        {
            "name": f"joint_top_k_topk_{top_k}",
            "inference_strategy": "joint_top_k",
            "config_overrides": {"training": {"top_k": top_k}},
        }
        for top_k in (1, 2, 3, 4)
    ]

    matrix: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "benchmark_val_path": str(benchmark_val_path),
        "benchmark_test_path": str(benchmark_test_path),
        "training_output_dir": str(training_output_dir),
        "train_runs": [],
        "sweeps": [],
        "inference_ablations": [],
    }

    for name, overrides in ablations:
        config_path = config_dir / f"{name}.json"
        write_experiment_config(
            base_config_path=base_config_path,
            output_path=config_path,
            overrides=overrides,
        )
        experiment_name = f"{experiment_prefix}_{name}"
        matrix["train_runs"].append(
            {
                "name": name,
                "experiment_name": experiment_name,
                "config_path": str(config_path),
                "train_command": _train_command(
                    dataset_path=dataset_path,
                    training_output_dir=training_output_dir,
                    experiment_name=experiment_name,
                    config_path=config_path,
                ),
                "val_eval_command": _generation_eval_command(
                    benchmark_path=benchmark_val_path,
                    output_dir=Path("paper/results/generation")
                    / f"{experiment_name}_val",
                    method_name=name,
                    bank_path=training_output_dir
                    / experiment_name
                    / "hypothesis_bank.json",
                    config_path=config_path,
                    inference_strategy="filter_and_weight",
                    use_empty_bank=name == "no_hypotheses",
                ),
            }
        )

    for name, overrides in sweep_specs:
        config_path = config_dir / f"{name}.json"
        write_experiment_config(
            base_config_path=base_config_path,
            output_path=config_path,
            overrides=overrides,
        )
        experiment_name = f"{experiment_prefix}_{name}"
        matrix["sweeps"].append(
            {
                "name": name,
                "experiment_name": experiment_name,
                "config_path": str(config_path),
                "train_command": _train_command(
                    dataset_path=dataset_path,
                    training_output_dir=training_output_dir,
                    experiment_name=experiment_name,
                    config_path=config_path,
                ),
                "val_eval_command": _generation_eval_command(
                    benchmark_path=benchmark_val_path,
                    output_dir=Path("paper/results/generation")
                    / f"{experiment_name}_val",
                    method_name=name,
                    bank_path=training_output_dir
                    / experiment_name
                    / "hypothesis_bank.json",
                    config_path=config_path,
                    inference_strategy="filter_and_weight",
                ),
            }
        )

    for spec in inference_ablations:
        config_path = config_dir / f"{spec['name']}.json"
        write_experiment_config(
            base_config_path=base_config_path,
            output_path=config_path,
            overrides=spec["config_overrides"],
        )
        matrix["inference_ablations"].append(
            {
                "name": spec["name"],
                "config_path": str(config_path),
                "test_eval_command": _generation_eval_command(
                    benchmark_path=benchmark_test_path,
                    output_dir=Path("paper/results/generation") / spec["name"],
                    method_name=spec["name"],
                    bank_path=final_bank_path,
                    config_path=config_path,
                    inference_strategy=spec["inference_strategy"],
                ),
            }
        )

    write_json(output_dir / "experiment_matrix.json", matrix)
    run_script = output_dir / "run_commands.sh"
    commands: list[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "source api_key",
    ]
    for section in ("train_runs", "sweeps", "inference_ablations"):
        for item in matrix[section]:
            if "train_command" in item:
                commands.append(item["train_command"])
                commands.append(item["val_eval_command"])
            else:
                commands.append(item["test_eval_command"])
    run_script.write_text("\n\n".join(commands) + "\n", encoding="utf-8")
    return matrix


def _train_command(
    *,
    dataset_path: Path,
    training_output_dir: Path,
    experiment_name: str,
    config_path: Path,
) -> str:
    return (
        "./.venv/bin/compos3d train-hypotheses "
        f"--dataset-path {dataset_path} "
        f"--output-dir {training_output_dir} "
        f"--experiment-name {experiment_name} "
        f"--config-path {config_path}"
    )


def _generation_eval_command(
    *,
    benchmark_path: Path,
    output_dir: Path,
    method_name: str,
    bank_path: Path,
    config_path: Path,
    inference_strategy: str,
    use_empty_bank: bool = False,
) -> str:
    command = (
        "./.venv/bin/python scripts/run_paper_generation_eval.py "
        f"--benchmark-path {benchmark_path} "
        f"--output-dir {output_dir} "
        f"--method-name {method_name} "
        f"--config-path {config_path} "
        f"--inference-strategy {inference_strategy} "
        "--render-scene"
    )
    if use_empty_bank:
        return f"{command} --use-empty-bank"
    return f"{command} --bank-path {bank_path}"
