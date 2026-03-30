import json
from pathlib import Path

import pytest
from PIL import Image

from compos3d.paper.figures import (
    _build_best_worst_cases,
    _failure_taxonomy_counts,
    _image_grid,
    _load_json,
    _load_jsonl_rows,
    _load_rows,
    _plot_bank_evolution,
    _plot_bank_metrics_timeline,
    _plot_failure_taxonomy,
    _plot_mean_hypothesis_score,
    _plot_training_diagnostics_overview,
    _plot_validation_sensitivity_axis,
    _save_matplotlib_figure,
    _snapshot_sort_key,
    _snapshot_stats,
    _snapshot_step,
    render_training_paper_figures,
)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    base = tmp_path / "run"
    base.mkdir()
    (base / "bank_snapshots").mkdir()
    (base / "wandb_media").mkdir()
    (base / "wandb").mkdir()

    # jsonl files
    def write_jsonl(name, rows):
        (base / name).write_text("\n".join(json.dumps(r) for r in rows))

    write_jsonl(
        "predictions.jsonl",
        [
            {
                "room_type": "dining_room",
                "example_id": "1",
                "overall": 0.9,
                "prompt": "a",
                "critic_score": {"notes": ["lamp placement issue"]},
            },
            {
                "room_type": "dining_room",
                "example_id": "2",
                "overall": 0.2,
                "prompt": "b",
            },
            {
                "room_type": "dining_room",
                "example_id": "3",
                "critic_score": {"overall": 0.5},
                "prompt": "c",
                "critic_notes": ["rug missing"],
            },
            {
                "room_type": "dining_room",
                "example_id": "4",
                "overall": 0.8,
                "prompt": "d",
            },
            {
                "room_type": "dining_room",
                "example_id": "5",
                "overall": 0.1,
                "prompt": "e",
            },
            {"room_type": "living_room", "example_id": "6"},
        ],
    )
    write_jsonl(
        "training_trace.jsonl",
        [{"room_type": "dining_room"}],
    )
    write_jsonl(
        "failed_scene_bank.jsonl",
        [{"room_type": "dining_room", "critic_notes": ["chair layout"]}],
    )
    write_jsonl(
        "wandb_media/media_manifest.jsonl",
        [
            {
                "room_type": "dining_room",
                "example_id": str(i),
                "phase": "prediction",
                "preview_path": str(base / f"img{i}.png"),
            }
            for i in range(1, 6)
        ],
    )

    # json files
    (base / "hypothesis_bank.json").write_text(
        json.dumps([{"reward": 1.0, "accuracy": 1.0, "mean_score": 1.0}])
    )
    (base / "bank_snapshots/hypothesis_bank_initial.json").write_text(
        json.dumps([{"reward": 0.5, "accuracy": 0.5, "mean_score": 0.5}])
    )
    (base / "bank_snapshots/hypothesis_bank_sample_10.json").write_text(
        json.dumps([{"reward": 0.8, "accuracy": 0.8, "mean_score": 0.8}])
    )
    (base / "wandb/wandb-summary.json").write_text(json.dumps({"_runtime": 120.0}))

    # create fake images
    for i in range(1, 6):
        Image.new("RGB", (10, 10)).save(base / f"img{i}.png")

    return base


@pytest.mark.unit
def test_load_utils(tmp_path):
    assert _load_json(None) == {}
    p = tmp_path / "test.json"
    p.write_text('{"a": 1}')
    assert _load_json(p) == {"a": 1}

    assert _load_jsonl_rows(tmp_path / "nonexistent.jsonl") == []
    jl = tmp_path / "test.jsonl"
    jl.write_text('{"a": 1}\n\n{"a": 2}')
    assert _load_jsonl_rows(jl) == [{"a": 1}, {"a": 2}]

    jl2 = tmp_path / "test2.jsonl"
    jl2.write_text('{"room_type": "a"}\n{"room_type": "b"}')
    assert _load_rows(jl2, room_type="a") == [{"room_type": "a"}]


@pytest.mark.unit
def test_snapshot_utils(tmp_path):
    assert _snapshot_step(Path("hypothesis_bank_initial.json")) == 0
    assert _snapshot_step(Path("hypothesis_bank_sample_42.json")) == 42
    assert _snapshot_step(Path("other.json")) == 0

    assert _snapshot_sort_key(Path("hypothesis_bank_sample_42.json")) == (
        42,
        "hypothesis_bank_sample_42.json",
    )

    (tmp_path / "bank_snapshots").mkdir()
    (tmp_path / "bank_snapshots" / "hypothesis_bank_sample_1.json").write_text("[]")
    stats = _snapshot_stats(tmp_path)
    assert stats[0]["bank_size"] == 0
    assert stats[0]["mean_reward"] == 0.0


@pytest.mark.unit
def test_render_training_paper_figures(run_dir, tmp_path):
    out_dir = tmp_path / "out"
    manifest = render_training_paper_figures(run_dir=run_dir, output_dir=out_dir)
    assert manifest["runtime_seconds"] == 120.0
    assert manifest["runtime_minutes"] == 2.0
    assert (out_dir / "bank_evolution.pdf").exists()
    assert (out_dir / "bank_metrics_timeline.pdf").exists()
    assert (out_dir / "failure_taxonomy.pdf").exists()
    assert (out_dir / "best_worst_cases.png").exists()
    assert (out_dir / "training_diagnostics_overview.pdf").exists()
    assert (out_dir / "mean_hypothesis_score.pdf").exists()


@pytest.mark.unit
def test_render_training_paper_figures_empty(tmp_path, monkeypatch):
    import compos3d.paper.figures as mod

    monkeypatch.setattr(mod, "_snapshot_stats", lambda r: [])

    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    (run_dir / "wandb_media").mkdir()
    (run_dir / "bank_snapshots").mkdir()
    (run_dir / "wandb").mkdir()

    (run_dir / "hypothesis_bank.json").write_text("[]")
    (run_dir / "bank_snapshots/hypothesis_bank_initial.json").write_text("[]")

    out_dir = tmp_path / "empty_out"
    # This will trigger `if not snapshots: return` in all plot functions during render_training_paper_figures
    manifest = render_training_paper_figures(run_dir=run_dir, output_dir=out_dir)
    assert manifest["runtime_minutes"] is None


@pytest.mark.unit
def test_save_matplotlib_figure(tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])

    # Test pdf
    pdf_path = tmp_path / "plot.pdf"
    _save_matplotlib_figure(fig, pdf_path)
    assert pdf_path.exists()
    assert pdf_path.with_suffix(".png").exists()

    # Test png
    png_path = tmp_path / "plot_only.png"
    _save_matplotlib_figure(fig, png_path)
    assert png_path.exists()
    plt.close(fig)


@pytest.mark.unit
def test_failure_taxonomy_counts():
    counts = _failure_taxonomy_counts(
        prediction_rows=[{"critic_score": {"notes": ["lamp placement is bad"]}}],
        failed_rows=[{"critic_notes": ["rug is missing", "table"]}],
    )
    assert counts["lamp placement"] == 1
    assert counts["rug issues"] == 1
    assert counts["table count"] == 1
    assert counts["missing window"] == 0


@pytest.mark.unit
def test_build_best_worst_cases(tmp_path):
    img = tmp_path / "img.png"
    Image.new("RGB", (10, 10)).save(img)

    preds = [
        {"example_id": "1", "overall": 0.9, "prompt": "p1"},
        {"example_id": "2", "critic_score": {"overall": 0.8}, "prompt": "p2"},
        {"example_id": "3", "overall": 0.1, "prompt": "p3"},
        {"example_id": "4", "overall": 0.2, "prompt": "p4"},
    ]
    media = [
        {"example_id": "1", "phase": "prediction", "preview_path": str(img)},
        {
            "example_id": "2",
            "phase": "prediction",
            "preview_path": "relative.png",
        },  # missing file but will test relative path logic later
    ]

    # Just running to verify it doesn't crash, missing image might crash so let's make it exist
    (tmp_path / "relative.png").write_bytes(img.read_bytes())

    # To test the cwd fallback
    import os

    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        _build_best_worst_cases(
            prediction_rows=preds, media_rows=media, output_path=tmp_path / "out.png"
        )
    finally:
        os.chdir(orig_cwd)


@pytest.mark.unit
def test_image_grid(tmp_path):
    img1 = tmp_path / "1.png"
    Image.new("RGB", (200, 200)).save(img1)

    _image_grid(
        rows=[{"preview_path": str(img1), "id": 1}],
        output_path=tmp_path / "grid.png",
        title="Test Grid",
        caption_fn=lambda r: f"id {r['id']}",
        columns=2,
    )
    assert (tmp_path / "grid.png").exists()
