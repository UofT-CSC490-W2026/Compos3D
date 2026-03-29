from __future__ import annotations

import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from compos3d.catalog import supported_assets_for_room
from compos3d.data.spatiallm import (
    build_prompt,
    collect_spatiallm_candidates,
    load_split_metadata,
)
from compos3d.models import SceneProgram

DINING_ROOM = "dining_room"
DINING_ROOM_ASSET_ORDER: tuple[str, ...] = (
    "dining_table",
    "chair",
    "lamp",
    "rug",
    "window",
    "vase",
)
DINING_ROOM_ANCHORS: frozenset[str] = frozenset({"dining_table", "chair"})
RARE_ASSET_TUPLE_THRESHOLD = 3
DEFAULT_SPLIT_SIZES = {"val": 24, "test": 100, "showcase": 24}
NUMERIC_WORDS: dict[int, str] = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
}
PROMPT_COUNT_WORDS: dict[str, int] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "several": 4,
}
ASSET_NOUNS: dict[str, tuple[str, ...]] = {
    "dining_table": ("dining table", "dining tables"),
    "chair": ("chair", "chairs"),
    "lamp": ("lamp", "lamps"),
    "rug": ("rug", "rugs"),
    "window": ("window", "windows"),
    "vase": ("vase", "vases"),
}


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def load_benchmark_examples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"Benchmark payload at {path} is missing an examples list.")
    return [dict(example) for example in examples]


def load_edit_pairs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_dataset_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset payload at {path} must be a JSON object.")
    return payload


def _candidate_to_example(candidate) -> dict[str, Any]:
    required_assets = [str(asset) for asset in candidate.required_assets]
    filtered_counts = {
        str(asset): int(candidate.asset_counts.get(asset, 1))
        for asset in required_assets
    }
    return {
        "example_id": f"dr_{candidate.room_id}",
        "room_type": DINING_ROOM,
        "prompt": candidate.prompt,
        "required_assets": required_assets,
        "style": "spatiallm",
        "source_room_id": candidate.room_id,
        "source_sample_id": candidate.sample_id,
        "expected_asset_counts": filtered_counts,
        "asset_tuple": list(required_assets),
    }


def _training_room_ids(dataset_payload: dict[str, Any]) -> set[str]:
    room_ids: set[str] = set()
    for raw_example in dataset_payload.get("examples", []):
        if not isinstance(raw_example, dict):
            continue
        if raw_example.get("room_type") != DINING_ROOM:
            continue
        example_id = str(raw_example.get("example_id", ""))
        if example_id.startswith("dr_"):
            room_ids.add(example_id[3:])
    return room_ids


def _set_hf_cache_dir(hf_cache_dir: Path | None) -> tuple[str | None, Path]:
    resolved = hf_cache_dir or (
        Path(os.environ.get("HF_DATASETS_CACHE", ""))
        if os.environ.get("HF_DATASETS_CACHE")
        else Path(tempfile.gettempdir()) / "compos3d_hf_datasets_cache"
    )
    resolved.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("HF_DATASETS_CACHE")
    os.environ["HF_DATASETS_CACHE"] = str(resolved)
    return previous, resolved


def _restore_hf_cache_dir(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("HF_DATASETS_CACHE", None)
    else:
        os.environ["HF_DATASETS_CACHE"] = previous


def build_dining_paper_benchmarks(
    *,
    raw_dir: Path,
    canonical_training_dataset_path: Path,
    output_dir: Path,
    seed: int = 42,
    allowed_split: str = "train",
    split_sizes: dict[str, int] | None = None,
    rare_asset_tuple_threshold: int = RARE_ASSET_TUPLE_THRESHOLD,
    hf_cache_dir: Path | None = None,
) -> dict[str, Any]:
    split_sizes = split_sizes or dict(DEFAULT_SPLIT_SIZES)
    required_total = sum(split_sizes.values())
    if set(split_sizes) != {"val", "test", "showcase"}:
        raise ValueError("split_sizes must define val, test, and showcase.")

    dataset_payload = _load_dataset_payload(canonical_training_dataset_path)
    used_room_ids = _training_room_ids(dataset_payload)
    if not used_room_ids:
        raise ValueError(
            f"No dining-room training examples found in {canonical_training_dataset_path}."
        )

    previous_hf_cache, resolved_hf_cache = _set_hf_cache_dir(hf_cache_dir)
    try:
        split_metadata = load_split_metadata(raw_dir / "split.csv")
        candidates_by_room = collect_spatiallm_candidates(
            split_metadata=split_metadata,
            raw_json_paths=[raw_dir / "spatiallm_train.json"],
            allowed_split=allowed_split,
        )
    finally:
        _restore_hf_cache_dir(previous_hf_cache)

    all_examples = [
        _candidate_to_example(candidate)
        for candidate in candidates_by_room.get(DINING_ROOM, [])
    ]
    unseen_examples = [
        example
        for example in all_examples
        if str(example["source_room_id"]) not in used_room_ids
    ]
    if len(unseen_examples) != required_total:
        raise ValueError(
            f"Expected {required_total} unseen dining-room examples, found {len(unseen_examples)}."
        )

    split_examples = stratified_split_examples(
        unseen_examples,
        split_sizes=split_sizes,
        seed=seed,
        rare_asset_tuple_threshold=rare_asset_tuple_threshold,
    )
    edit_pairs = build_edit_pairs_from_splits(
        val_examples=split_examples["val"],
        test_examples=split_examples["test"],
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    bench_metadata = {
        "paper_scope": "dining_room",
        "canonical_training_dataset_path": str(canonical_training_dataset_path),
        "raw_dir": str(raw_dir),
        "allowed_split": allowed_split,
        "seed": seed,
        "rare_asset_tuple_threshold": rare_asset_tuple_threshold,
        "hf_cache_dir": str(resolved_hf_cache),
        "canonical_training_dining_examples": len(used_room_ids),
        "all_convertible_dining_examples": len(all_examples),
        "heldout_dining_examples": len(unseen_examples),
        "split_sizes": split_sizes,
    }
    for split_name, examples in split_examples.items():
        write_json(
            output_dir / f"dining_{split_name}.json",
            {
                "dataset_id": f"dining_room_paper_{split_name}",
                "metadata": {
                    **bench_metadata,
                    "split_name": split_name,
                    "num_examples": len(examples),
                    "asset_tuple_counts": _asset_tuple_counts(examples),
                },
                "examples": examples,
            },
        )

    write_jsonl(output_dir / "edit_pairs.jsonl", edit_pairs)
    metadata_payload = {
        **bench_metadata,
        "asset_tuple_counts": _asset_tuple_counts(unseen_examples),
        "edit_pair_counts": dict(Counter(pair["edit_type"] for pair in edit_pairs)),
    }
    write_json(output_dir / "metadata.json", metadata_payload)
    return {
        "splits": split_examples,
        "edit_pairs": edit_pairs,
        "metadata": metadata_payload,
    }


def stratified_split_examples(
    examples: Sequence[dict[str, Any]],
    *,
    split_sizes: dict[str, int],
    seed: int,
    rare_asset_tuple_threshold: int,
) -> dict[str, list[dict[str, Any]]]:
    if len(examples) != sum(split_sizes.values()):
        raise ValueError("Split sizes must sum to the total number of examples.")

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        groups[_asset_tuple(example)].append(dict(example))

    rng = random.Random(seed)
    for bucket in groups.values():
        bucket.sort(key=lambda item: str(item["example_id"]))
        rng.shuffle(bucket)

    ordered_keys = sorted(
        groups,
        key=lambda key: (
            len(groups[key]) > rare_asset_tuple_threshold,
            len(groups[key]),
            list(key),
        ),
    )

    remaining = dict(split_sizes)
    split_examples: dict[str, list[dict[str, Any]]] = {key: [] for key in split_sizes}
    for bucket_index, key in enumerate(ordered_keys):
        bucket = groups[key]
        allow_showcase = len(bucket) > rare_asset_tuple_threshold
        if bucket_index == len(ordered_keys) - 1:
            counts = {
                split_name: remaining[split_name]
                if allow_showcase or split_name != "showcase"
                else 0
                for split_name in remaining
            }
            if sum(counts.values()) != len(bucket):
                raise ValueError(
                    "Final bucket could not satisfy remaining split quotas."
                )
        else:
            counts = _allocate_bucket_counts(
                bucket_size=len(bucket),
                remaining=remaining,
                allow_showcase=allow_showcase,
            )

        cursor = 0
        for split_name in ("val", "test", "showcase"):
            count = counts.get(split_name, 0)
            split_examples[split_name].extend(bucket[cursor : cursor + count])
            remaining[split_name] -= count
            cursor += count

    if any(value != 0 for value in remaining.values()):
        raise ValueError(f"Split quotas were not exhausted: {remaining}")  # pragma: no cover

    for split_name, bucket in split_examples.items():
        bucket.sort(key=lambda item: (_asset_tuple(item), str(item["example_id"])))
    return split_examples


def build_edit_pairs_from_splits(
    *,
    val_examples: Sequence[dict[str, Any]],
    test_examples: Sequence[dict[str, Any]],
    seed: int,
    count_per_type: int = 25,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    source_pool = [dict(example, source_split="val") for example in val_examples] + [
        dict(example, source_split="test") for example in test_examples
    ]
    source_pool.sort(key=lambda item: str(item["example_id"]))
    rng.shuffle(source_pool)

    used_example_ids: set[str] = set()
    pairs: list[dict[str, Any]] = []

    pending = [
        ("add_asset", _eligible_add_asset, _build_add_asset_edit),
        ("remove_asset", _eligible_remove_asset, _build_remove_asset_edit),
        ("count_change", _eligible_count_change, _build_count_change_edit),
        ("swap_asset", _eligible_swap_asset, _build_swap_asset_edit),
    ]
    while pending:
        edit_type, predicate, builder = min(
            pending,
            key=lambda item: len(
                [
                    example
                    for example in source_pool
                    if example["example_id"] not in used_example_ids
                    and item[1](example)
                ]
            ),
        )
        chosen = [
            example
            for example in source_pool
            if example["example_id"] not in used_example_ids and predicate(example)
        ][:count_per_type]
        if len(chosen) != count_per_type:
            raise ValueError(
                f"Could not build {count_per_type} unique {edit_type} edit pairs; found {len(chosen)}."
            )
        for example in chosen:
            used_example_ids.add(str(example["example_id"]))
            pairs.append(builder(example))
        pending = [item for item in pending if item[0] != edit_type]

    pairs.sort(key=lambda item: item["pair_id"])
    return pairs


def _allocate_bucket_counts(
    *, bucket_size: int, remaining: dict[str, int], allow_showcase: bool
) -> dict[str, int]:
    allowed = ["val", "test"] if not allow_showcase else ["val", "test", "showcase"]
    total_allowed = sum(remaining[split_name] for split_name in allowed)
    if total_allowed < bucket_size:
        raise ValueError("Not enough split quota remains for this bucket.")

    raw_targets = {
        split_name: bucket_size * remaining[split_name] / total_allowed
        for split_name in allowed
    }
    counts = {
        split_name: min(int(math.floor(raw_targets[split_name])), remaining[split_name])
        for split_name in allowed
    }
    need = bucket_size - sum(counts.values())
    while need > 0:
        ranked = sorted(
            allowed,
            key=lambda split_name: (
                raw_targets[split_name] - counts[split_name],
                remaining[split_name] - counts[split_name],
                split_name,
            ),
            reverse=True,
        )
        placed = False
        for split_name in ranked:
            if counts[split_name] < remaining[split_name]:
                counts[split_name] += 1
                need -= 1
                placed = True
                break
        if not placed:
            raise ValueError(
                "Could not distribute bucket counts within remaining quotas."
            )  # pragma: no cover

    return {split_name: counts.get(split_name, 0) for split_name in remaining}


def _asset_tuple(example: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(asset) for asset in example.get("required_assets", []))


def _asset_tuple_counts(examples: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(_asset_tuple(example) for example in examples)
    return {json.dumps(list(key)): value for key, value in counts.items()}


def _sorted_required_assets(assets: Iterable[str]) -> list[str]:
    priority = {asset: index for index, asset in enumerate(DINING_ROOM_ASSET_ORDER)}
    return sorted(dict.fromkeys(assets), key=lambda asset: priority.get(asset, 999))


def _optional_assets(example: dict[str, Any]) -> list[str]:
    return [
        asset
        for asset in example.get("required_assets", [])
        if asset not in DINING_ROOM_ANCHORS
    ]


def _eligible_add_asset(example: dict[str, Any]) -> bool:
    missing = [
        asset
        for asset in ("rug", "window", "lamp")
        if asset not in example.get("required_assets", [])
    ]
    return bool(missing)


def _eligible_remove_asset(example: dict[str, Any]) -> bool:
    return bool(_optional_assets(example))


def _eligible_count_change(example: dict[str, Any]) -> bool:
    try:
        _count_change_target(example)
    except ValueError:
        return False
    return True


def _eligible_swap_asset(example: dict[str, Any]) -> bool:
    assets = set(example.get("required_assets", []))
    return ("rug" in assets) ^ ("window" in assets)


def _build_add_asset_edit(example: dict[str, Any]) -> dict[str, Any]:
    base_counts = Counter(expected_asset_counts(example))
    missing = [
        asset
        for asset in ("rug", "window", "lamp")
        if asset not in example.get("required_assets", [])
    ]
    target_asset = missing[0]
    base_counts[target_asset] = 1
    edited_assets = _sorted_required_assets(
        list(example["required_assets"]) + [target_asset]
    )
    return _edit_pair_payload(
        example=example,
        edit_type="add_asset",
        edited_required_assets=edited_assets,
        edited_asset_counts=base_counts,
        changed_assets=[target_asset],
        edit_instruction=f"Add {_asset_phrase(target_asset, 1)} to the dining room.",
    )


def _build_remove_asset_edit(example: dict[str, Any]) -> dict[str, Any]:
    base_counts = Counter(expected_asset_counts(example))
    removable = _optional_assets(example)
    target_asset = sorted(removable, key=DINING_ROOM_ASSET_ORDER.index)[0]
    base_counts.pop(target_asset, None)
    edited_assets = [
        asset for asset in example["required_assets"] if asset != target_asset
    ]
    return _edit_pair_payload(
        example=example,
        edit_type="remove_asset",
        edited_required_assets=edited_assets,
        edited_asset_counts=base_counts,
        changed_assets=[target_asset],
        edit_instruction=f"Remove the {target_asset.replace('_', ' ')} from the dining room.",
    )


def _build_count_change_edit(example: dict[str, Any]) -> dict[str, Any]:
    base_counts = Counter(expected_asset_counts(example))
    target_asset, new_count = _count_change_target(example)
    old_count = int(base_counts[target_asset])
    base_counts[target_asset] = new_count
    return _edit_pair_payload(
        example=example,
        edit_type="count_change",
        edited_required_assets=list(example["required_assets"]),
        edited_asset_counts=base_counts,
        changed_assets=[target_asset],
        edit_instruction=(
            f"Change the number of {ASSET_NOUNS[target_asset][1]} "
            f"from {old_count} to {new_count}."
        ),
    )


def _build_swap_asset_edit(example: dict[str, Any]) -> dict[str, Any]:
    base_counts = Counter(expected_asset_counts(example))
    if "rug" in example["required_assets"]:
        removed_asset, added_asset = "rug", "window"
    else:
        removed_asset, added_asset = "window", "rug"
    base_counts.pop(removed_asset, None)
    base_counts[added_asset] = 1
    edited_assets = _sorted_required_assets(
        [asset for asset in example["required_assets"] if asset != removed_asset]
        + [added_asset]
    )
    return _edit_pair_payload(
        example=example,
        edit_type="swap_asset",
        edited_required_assets=edited_assets,
        edited_asset_counts=base_counts,
        changed_assets=[removed_asset, added_asset],
        edit_instruction=(
            f"Replace the {removed_asset.replace('_', ' ')} with {_asset_phrase(added_asset, 1)}."
        ),
    )


def _count_change_target(example: dict[str, Any]) -> tuple[str, int]:
    counts = expected_asset_counts(example)
    for asset in ("dining_table", "chair", "lamp", "rug"):
        if asset not in example.get("required_assets", []):
            continue
        current = int(counts.get(asset, 1))
        new_count = _next_count(current)
        if new_count != current:
            return asset, new_count
    raise ValueError(f"No count-change target found for {example['example_id']}.")


def _next_count(current: int) -> int:
    if current <= 1:
        return 2
    if current == 2:
        return 3
    if current == 3:
        return 4
    if current == 4:
        return 2
    return 4


def _edit_pair_payload(
    *,
    example: dict[str, Any],
    edit_type: str,
    edited_required_assets: list[str],
    edited_asset_counts: Counter[str],
    changed_assets: list[str],
    edit_instruction: str,
) -> dict[str, Any]:
    base_counts = dict(expected_asset_counts(example))
    edited_counts = {
        asset: int(edited_asset_counts.get(asset, 1))
        for asset in edited_required_assets
    }
    edited_prompt = build_prompt(
        DINING_ROOM, edited_required_assets, Counter(edited_counts)
    )
    unchanged_assets = [
        asset
        for asset in example["required_assets"]
        if asset not in set(changed_assets)
    ]
    changed_suffix = "_".join(changed_assets)
    return {
        "pair_id": f"{example['example_id']}__{edit_type}__{changed_suffix}",
        "edit_type": edit_type,
        "source_split": example.get("source_split", "unknown"),
        "base_example_id": example["example_id"],
        "base_prompt": example["prompt"],
        "base_required_assets": list(example["required_assets"]),
        "base_expected_asset_counts": base_counts,
        "base_asset_tuple": list(example["required_assets"]),
        "edited_example_id": f"{example['example_id']}__{edit_type}",
        "edited_prompt": edited_prompt,
        "edited_required_assets": edited_required_assets,
        "edited_expected_asset_counts": edited_counts,
        "edited_asset_tuple": edited_required_assets,
        "changed_assets": changed_assets,
        "unchanged_assets": unchanged_assets,
        "edit_instruction": edit_instruction,
    }


def _asset_phrase(asset: str, count: int) -> str:
    noun_singular, noun_plural = ASSET_NOUNS[asset]
    if count == 1:
        article = "an" if noun_singular[0].lower() in {"a", "e", "i", "o", "u"} else "a"  # pragma: no cover
        return f"{article} {noun_singular}"  # pragma: no cover
    count_text = NUMERIC_WORDS.get(count, str(count))
    return f"{count_text} {noun_plural}"


def expected_asset_counts(example: dict[str, Any]) -> dict[str, int]:
    raw_counts = example.get("expected_asset_counts")
    if isinstance(raw_counts, dict) and raw_counts:
        return {str(key): int(value) for key, value in raw_counts.items()}
    return parse_prompt_asset_counts(
        prompt=str(example["prompt"]),
        room_type=str(example.get("room_type", DINING_ROOM)),
        required_assets=[str(asset) for asset in example.get("required_assets", [])],
    )


def parse_prompt_asset_counts(
    *, prompt: str, room_type: str, required_assets: Sequence[str]
) -> dict[str, int]:
    prompt_lower = prompt.lower()
    counts: dict[str, int] = {}
    for asset in required_assets:
        singular, plural = ASSET_NOUNS.get(
            asset, (asset.replace("_", " "), f"{asset}s")
        )
        phrases = [singular, plural]
        matched_value: int | None = None
        for phrase in phrases:
            for word, value in PROMPT_COUNT_WORDS.items():
                token = f"{word} {phrase}"
                if token in prompt_lower:
                    matched_value = value
                    break
            if matched_value is not None:
                break
        if matched_value is None:
            matched_value = 1
        counts[asset] = matched_value
    return counts


def scene_program_asset_counts(
    scene_program: SceneProgram | dict[str, Any],
) -> dict[str, int]:
    if isinstance(scene_program, SceneProgram):
        assets = scene_program.assets
    else:
        assets = scene_program.get("assets", [])
    counts: dict[str, int] = {}
    for asset in assets:
        if isinstance(asset, dict):
            asset_type = str(asset.get("asset_type", ""))
            count = int(asset.get("count", 1) or 1)
        else:
            asset_type = str(asset.asset_type)
            count = int(asset.count)
        counts[asset_type] = count
    return counts


def build_program_metric_row(
    *,
    example: dict[str, Any],
    scene_program: SceneProgram | dict[str, Any],
) -> dict[str, Any]:
    room_type = str(example.get("room_type", DINING_ROOM))
    expected_counts = expected_asset_counts(example)
    predicted_counts = scene_program_asset_counts(scene_program)
    requested_assets = set(str(asset) for asset in example.get("required_assets", []))
    predicted_assets = set(predicted_counts)
    overlap = requested_assets & predicted_assets

    exact_room_type = float(
        (
            scene_program.room_type
            if isinstance(scene_program, SceneProgram)
            else scene_program.get("room_type")
        )
        == room_type
    )
    precision = len(overlap) / len(predicted_assets) if predicted_assets else 1.0
    recall = len(overlap) / len(requested_assets) if requested_assets else 1.0
    f1 = (
        0.0
        if (precision + recall) == 0
        else 2 * precision * recall / (precision + recall)
    )
    count_error_total = float(
        sum(
            abs(predicted_counts.get(asset, 0) - expected_counts.get(asset, 0))
            for asset in requested_assets
        )
    )
    count_error_mean = count_error_total / max(len(requested_assets), 1)
    hallucinated_assets = predicted_assets - requested_assets
    asset_tuple = list(example.get("required_assets", []))
    return {
        "exact_room_type_accuracy": round(exact_room_type, 4),
        "requested_asset_precision": round(precision, 4),
        "requested_asset_recall": round(recall, 4),
        "requested_asset_f1": round(f1, 4),
        "requested_count_error_total": round(count_error_total, 4),
        "requested_count_error_mean": round(count_error_mean, 4),
        "exact_asset_set_match": float(predicted_assets == requested_assets),
        "hallucinated_asset_rate": round(
            len(hallucinated_assets) / max(len(predicted_assets), 1), 4
        ),
        "predicted_assets": sorted(predicted_assets),
        "requested_assets": sorted(requested_assets),
        "expected_asset_counts": expected_counts,
        "predicted_asset_counts": predicted_counts,
        "asset_tuple": asset_tuple,
        "has_window": float("window" in requested_assets),
        "has_rug": float("rug" in requested_assets),
        "count_heavy": float(any(count > 1 for count in expected_counts.values())),
        "multi_table": float(expected_counts.get("dining_table", 0) > 1),
    }


def summarize_generation_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "exact_room_type_accuracy",
        "requested_asset_precision",
        "requested_asset_recall",
        "requested_asset_f1",
        "requested_count_error_total",
        "requested_count_error_mean",
        "exact_asset_set_match",
        "hallucinated_asset_rate",
        "validity",
        "prompt_adherence",
        "asset_precision",
        "asset_recall",
        "room_match",
        "overall",
    ]
    summary = {
        "num_examples": len(rows),
        "metrics": _mean_metrics(rows, numeric_keys),
        "subgroups": {},
    }
    subgroup_specs = {
        "has_window": lambda row: bool(row.get("has_window")),
        "has_rug": lambda row: bool(row.get("has_rug")),
        "count_heavy": lambda row: bool(row.get("count_heavy")),
        "multi_table": lambda row: bool(row.get("multi_table")),
    }
    for subgroup_name, predicate in subgroup_specs.items():
        bucket = [row for row in rows if predicate(row)]
        summary["subgroups"][subgroup_name] = {
            "num_examples": len(bucket),
            "metrics": _mean_metrics(bucket, numeric_keys),
        }

    asset_tuple_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        asset_tuple_groups[json.dumps(row.get("asset_tuple", []))].append(row)
    summary["subgroups"]["asset_tuple"] = {
        key: {
            "num_examples": len(bucket),
            "metrics": _mean_metrics(bucket, numeric_keys),
        }
        for key, bucket in sorted(asset_tuple_groups.items())
    }
    return summary


def compute_edit_program_metrics(
    *,
    pair: dict[str, Any],
    base_scene_program: SceneProgram | dict[str, Any],
    edited_scene_program: SceneProgram | dict[str, Any],
) -> dict[str, Any]:
    base_counts = scene_program_asset_counts(base_scene_program)
    edited_counts = scene_program_asset_counts(edited_scene_program)
    expected_base = {
        str(key): int(value)
        for key, value in pair["base_expected_asset_counts"].items()
    }
    expected_edited = {
        str(key): int(value)
        for key, value in pair["edited_expected_asset_counts"].items()
    }
    changed_assets = set(str(asset) for asset in pair.get("changed_assets", []))
    unchanged_assets = set(str(asset) for asset in pair.get("unchanged_assets", []))

    expected_delta = {
        asset: expected_edited.get(asset, 0) - expected_base.get(asset, 0)
        for asset in set(expected_base) | set(expected_edited)
    }
    predicted_delta = {
        asset: edited_counts.get(asset, 0) - base_counts.get(asset, 0)
        for asset in set(base_counts) | set(edited_counts) | set(expected_delta)
    }
    expected_changed = {asset for asset, delta in expected_delta.items() if delta != 0}
    predicted_changed = {
        asset for asset, delta in predicted_delta.items() if delta != 0
    }
    overlap = expected_changed & predicted_changed
    precision = len(overlap) / len(predicted_changed) if predicted_changed else 1.0
    recall = len(overlap) / len(expected_changed) if expected_changed else 1.0
    f1 = (
        0.0
        if (precision + recall) == 0
        else 2 * precision * recall / (precision + recall)
    )
    unchanged_retention = sum(
        1
        for asset in unchanged_assets
        if edited_counts.get(asset, 0) == base_counts.get(asset, 0)
    ) / max(len(unchanged_assets), 1)
    delta_count_l1 = sum(
        abs(predicted_delta.get(asset, 0) - expected_delta.get(asset, 0))
        for asset in expected_changed
    )
    return {
        "delta_asset_precision": round(precision, 4),
        "delta_asset_recall": round(recall, 4),
        "delta_asset_f1": round(f1, 4),
        "delta_count_l1": round(float(delta_count_l1), 4),
        "unchanged_asset_retention_rate": round(float(unchanged_retention), 4),
        "changed_assets": sorted(changed_assets),
        "unchanged_assets": sorted(unchanged_assets),
        "expected_delta": expected_delta,
        "predicted_delta": predicted_delta,
    }


def _mean_metrics(
    rows: Sequence[dict[str, Any]], numeric_keys: Sequence[str]
) -> dict[str, float | None]:
    if not rows:
        return {key: None for key in numeric_keys}
    summary: dict[str, float | None] = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        summary[key] = round(sum(values) / len(values), 4) if values else None
    return summary
