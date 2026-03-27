#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from compos3d.data.spatiallm import (
    build_training_dataset_payload,
    collect_spatiallm_candidates,
    dataset_summary,
    load_split_metadata,
    write_training_dataset_payload,
)


def download_spatiallm_files(raw_dir: Path) -> list[Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised in real env
        raise RuntimeError(
            "The `huggingface_hub` package is required to download SpatialLM."
        ) from exc

    raw_dir.mkdir(parents=True, exist_ok=True)
    filenames = ("split.csv", "spatiallm_train.json")
    local_paths: list[Path] = []
    for filename in filenames:
        local_path = hf_hub_download(
            repo_id="manycore-research/SpatialLM-Dataset",
            repo_type="dataset",
            filename=filename,
            local_dir=str(raw_dir),
        )
        local_paths.append(Path(local_path))
    return local_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and convert the SpatialLM dataset into a Compos3D training dataset."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/external/spatiallm"),
        help="Directory for the downloaded SpatialLM source files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("examples/vertical_slice_dataset.json"),
        help="Path for the converted Compos3D dataset JSON.",
    )
    parser.add_argument(
        "--max-per-room",
        type=int,
        default=60,
        help="Maximum number of converted examples to keep per Compos3D room type.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use files already present in --raw-dir instead of downloading them.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.skip_download:
        download_spatiallm_files(args.raw_dir)

    split_csv_path = args.raw_dir / "split.csv"
    train_json_path = args.raw_dir / "spatiallm_train.json"
    if not split_csv_path.exists() or not train_json_path.exists():
        raise SystemExit(
            "Expected both split.csv and spatiallm_train.json in the raw directory."
        )

    split_metadata = load_split_metadata(split_csv_path)
    candidates_by_room = collect_spatiallm_candidates(
        split_metadata=split_metadata,
        raw_json_paths=[train_json_path],
        allowed_split="train",
    )
    payload = build_training_dataset_payload(
        candidates_by_room,
        max_per_room=args.max_per_room,
    )
    write_training_dataset_payload(payload, args.output_path)

    counts = dataset_summary(payload)
    print(f"Wrote {len(payload['examples'])} examples to {args.output_path}")
    print("Room counts:")
    for room_type, count in counts.items():
        print(f"  {room_type}: {count}")


if __name__ == "__main__":
    main()
