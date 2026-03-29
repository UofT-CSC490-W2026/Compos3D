from __future__ import annotations

import json
import os
import re
import statistics
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from compos3d.paper.benchmarks import write_json

os.environ.setdefault("MPLCONFIGDIR", "/tmp/compos3d_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_training_paper_figures(
    *,
    run_dir: Path,
    output_dir: Path,
    room_type: str = "dining_room",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = _load_rows(run_dir / "predictions.jsonl", room_type=room_type)
    trace_rows = _load_rows(run_dir / "training_trace.jsonl", room_type=room_type)
    failed_rows = _load_rows(run_dir / "failed_scene_bank.jsonl", room_type=room_type)
    final_bank = _load_json(run_dir / "hypothesis_bank.json")
    initial_bank = _load_json(
        run_dir / "bank_snapshots" / "hypothesis_bank_initial.json"
    )
    media_rows = _load_rows(
        run_dir / "wandb_media" / "media_manifest.jsonl", room_type=room_type
    )
    wandb_summary_path = next(
        (
            path
            for path in (run_dir / "wandb").rglob("wandb-summary.json")
            if path.is_file()
        ),
        None,
    )
    wandb_summary = _load_json(wandb_summary_path) if wandb_summary_path else {}

    figure_paths = {
        "bank_evolution": output_dir / "bank_evolution.png",
        "bank_metrics_timeline": output_dir / "bank_metrics_timeline.png",
        "failure_taxonomy": output_dir / "failure_taxonomy.png",
        "best_worst_cases": output_dir / "best_worst_cases.png",
    }

    _plot_bank_evolution(
        run_dir=run_dir,
        initial_bank=initial_bank,
        final_bank=final_bank,
        output_path=figure_paths["bank_evolution"],
    )
    _plot_bank_metrics_timeline(
        run_dir=run_dir,
        output_path=figure_paths["bank_metrics_timeline"],
    )
    _plot_failure_taxonomy(
        prediction_rows=predictions,
        failed_rows=failed_rows,
        output_path=figure_paths["failure_taxonomy"],
    )
    _build_best_worst_cases(
        prediction_rows=predictions,
        media_rows=media_rows,
        output_path=figure_paths["best_worst_cases"],
    )

    manifest = {
        "run_dir": str(run_dir),
        "room_type": room_type,
        "num_predictions": len(predictions),
        "num_training_trace_rows": len(trace_rows),
        "num_failures": len(failed_rows),
        "num_media_rows": len(media_rows),
        "runtime_seconds": wandb_summary.get("_runtime"),
        "runtime_minutes": round(float(wandb_summary["_runtime"]) / 60.0, 2)
        if wandb_summary.get("_runtime") is not None
        else None,
        "figure_paths": {key: str(path) for key, path in figure_paths.items()},
    }
    write_json(output_dir / "figure_manifest.json", manifest)
    return manifest


def _load_rows(path: Path, *, room_type: str) -> list[dict[str, Any]]:
    rows = _load_jsonl_rows(path)
    return [row for row in rows if str(row.get("room_type")) == room_type]


def _load_json(path: Path | None) -> Any:
    if path is None:
        return {}
    return json.loads(path.read_text())


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _snapshot_stats(run_dir: Path) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for path in sorted(
        (run_dir / "bank_snapshots").glob("*.json"), key=_snapshot_sort_key
    ):
        records = _load_json(path)
        rewards = [float(record["reward"]) for record in records] if records else [0.0]
        accuracies = (
            [float(record["accuracy"]) for record in records] if records else [0.0]
        )
        scores = (
            [float(record["mean_score"]) for record in records] if records else [0.0]
        )
        stats.append(
            {
                "label": path.stem,
                "step": _snapshot_step(path),
                "bank_size": len(records),
                "mean_reward": statistics.mean(rewards),
                "mean_accuracy": statistics.mean(accuracies),
                "mean_score": statistics.mean(scores),
                "rewards": rewards,
                "accuracies": accuracies,
            }
        )
    return stats


def _snapshot_step(path: Path) -> int:
    if "initial" in path.stem:
        return 0
    match = re.search(r"sample_(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def _snapshot_sort_key(path: Path) -> tuple[int, str]:
    return (_snapshot_step(path), path.name)


def _plot_bank_evolution(
    *,
    run_dir: Path,
    initial_bank: list[dict[str, Any]],
    final_bank: list[dict[str, Any]],
    output_path: Path,
) -> None:
    snapshots = _snapshot_stats(run_dir)
    if not snapshots:
        return

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
    steps = [item["step"] for item in snapshots]

    axes[0].plot(
        steps, [item["bank_size"] for item in snapshots], color="#0f766e", linewidth=2.4
    )
    axes[0].set_title("Hypothesis Count")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Hypotheses in Bank")
    axes[0].grid(alpha=0.25)

    axes[1].bar(
        ["Initial", "Final"],
        [len(initial_bank), len(final_bank)],
        color=["#94a3b8", "#0f766e"],
        width=0.6,
    )
    axes[1].set_title("Initial vs Final Bank Size")
    axes[1].set_ylabel("Hypotheses")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].hist(
        [float(record["reward"]) for record in initial_bank],
        bins=8,
        alpha=0.65,
        color="#94a3b8",
        label="initial",
    )
    axes[2].hist(
        [float(record["reward"]) for record in final_bank],
        bins=8,
        alpha=0.65,
        color="#0f766e",
        label="final",
    )
    axes[2].set_title("UCB Reward Distribution")
    axes[2].set_xlabel("Reward")
    axes[2].set_ylabel("Hypotheses")
    axes[2].legend()

    axes[3].hist(
        [float(record["accuracy"]) for record in initial_bank],
        bins=8,
        alpha=0.65,
        color="#cbd5e1",
        label="initial",
    )
    axes[3].hist(
        [float(record["accuracy"]) for record in final_bank],
        bins=8,
        alpha=0.65,
        color="#2563eb",
        label="final",
    )
    axes[3].set_title("Training Accuracy Distribution")
    axes[3].set_xlabel("Accuracy")
    axes[3].set_ylabel("Hypotheses")
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_bank_metrics_timeline(
    *,
    run_dir: Path,
    output_path: Path,
) -> None:
    snapshots = _snapshot_stats(run_dir)
    if not snapshots:
        return

    steps = [item["step"] for item in snapshots]
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.plot(
        steps,
        [item["mean_score"] for item in snapshots],
        color="#15803d",
        linewidth=2.4,
        label="Mean Hypothesis Score",
    )
    ax.plot(
        steps,
        [item["mean_reward"] for item in snapshots],
        color="#b91c1c",
        linewidth=2.4,
        label="Mean UCB Reward",
    )
    ax.plot(
        steps,
        [item["mean_accuracy"] for item in snapshots],
        color="#1d4ed8",
        linewidth=2.4,
        label="Mean Training Accuracy",
    )
    ax.set_title("Hypothesis Bank Metrics Over Training")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Value")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_failure_taxonomy(
    *,
    prediction_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    notes: list[str] = []
    for row in prediction_rows:
        notes.extend(str(note) for note in row.get("critic_score", {}).get("notes", []))
        notes.extend(str(note) for note in row.get("critic_notes", []))
    for row in failed_rows:
        notes.extend(str(note) for note in row.get("critic_notes", []))

    taxonomy = {
        "lamp placement": r"lamp.*(placement|above|floor|overhead|corner)",
        "rug issues": r"rug",
        "chair arrangement": r"chair",
        "missing window": r"window",
        "table count": r"table",
    }
    counts: dict[str, int] = {}
    for label, pattern in taxonomy.items():
        counts[label] = sum(
            1 for note in notes if re.search(pattern, note.lower()) is not None
        )

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = list(counts)
    values = [counts[label] for label in labels]
    ax.bar(
        labels, values, color=["#b91c1c", "#1d4ed8", "#0f766e", "#a16207", "#7c3aed"]
    )
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_best_worst_cases(
    *,
    prediction_rows: list[dict[str, Any]],
    media_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    def _overall(row: dict[str, Any]) -> float:
        if row.get("overall") is not None:
            return float(row["overall"])
        return float(row.get("critic_score", {}).get("overall", 0.0))

    preview_by_id = {
        str(row.get("example_id")): row
        for row in media_rows
        if str(row.get("phase")) == "prediction" and row.get("preview_path")
    }
    ranked = sorted(
        prediction_rows,
        key=lambda row: (_overall(row), str(row.get("example_id"))),
    )
    chosen = ranked[:4] + ranked[-4:]
    rendered_rows = []
    for row in chosen:
        media = preview_by_id.get(str(row["example_id"]))
        if media is None:
            continue
        rendered_rows.append(
            {
                "preview_path": media["preview_path"],
                "example_id": row["example_id"],
                "overall": _overall(row),
                "prompt": row["prompt"],
            }
        )
    if rendered_rows:
        _image_grid(
            rows=rendered_rows,
            output_path=output_path,
            title="Best and worst dining-room predictions",
            caption_fn=lambda row: (
                f"{row['example_id']} | overall {float(row['overall']):.2f}"
            ),
            columns=4,
        )


def _image_grid(
    *,
    rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
    caption_fn,
    columns: int = 3,
) -> None:
    cell_width = 420
    cell_height = 320
    title_height = 56
    caption_height = 50
    padding = 18
    grid_rows = (len(rows) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (
            padding + columns * (cell_width + padding),
            title_height
            + padding
            + grid_rows * (cell_height + caption_height + padding),
        ),
        color=(247, 248, 250),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, 16), title, fill=(24, 24, 27))

    for index, row in enumerate(rows):
        row_index = index // columns
        col_index = index % columns
        x0 = padding + col_index * (cell_width + padding)
        y0 = title_height + row_index * (cell_height + caption_height + padding)
        preview_path = Path(str(row["preview_path"]))
        if not preview_path.is_absolute():
            preview_path = Path.cwd() / preview_path
        with Image.open(preview_path) as image:
            image = ImageOps.contain(image.convert("RGB"), (cell_width, cell_height))
            x_image = x0 + (cell_width - image.width) // 2
            y_image = y0 + (cell_height - image.height) // 2
            canvas.paste(image, (x_image, y_image))
        caption = caption_fn(row)
        draw.text((x0, y0 + cell_height + 8), caption, fill=(39, 39, 42))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
