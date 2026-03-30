import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from compos3d.config import CriticConfig
from compos3d.paper.experiments import (
    _deep_merge,
    _empty_bank_path,
    _maybe_materialize_config,
    _normalize_image_paths,
    _pairwise_counts,
    _pairwise_metric_summary,
    _read_json,
    _resolve_results_file,
    _summarize_generation_pairwise,
    build_paper_training_matrix,
    export_edit_human_study_sheet,
    export_generation_human_study_sheet,
    judge_generation_pairwise,
    load_jsonl_rows,
    run_edit_benchmark,
    run_generation_benchmark,
    summarize_edit_rows,
    write_experiment_config,
)


@pytest.fixture
def tmp_run_env(tmp_path):
    d = tmp_path / "env"
    d.mkdir()
    return d


@pytest.mark.unit
def test_utils(tmp_path):
    assert load_jsonl_rows(tmp_path / "nonexist.jsonl") == []

    jl = tmp_path / "data.jsonl"
    jl.write_text('{"a": 1}\n{"b": 2}')
    assert load_jsonl_rows(jl) == [{"a": 1}, {"b": 2}]

    j = tmp_path / "data.json"
    j.write_text('{"c": 3}')
    assert _read_json(j) == {"c": 3}

    assert _resolve_results_file(tmp_path, "res.json") == tmp_path / "res.json"
    f = tmp_path / "file.json"
    f.touch()
    assert _resolve_results_file(f, "res.json") == f

    base = {"a": {"b": 1}, "c": 2}
    over = {"a": {"d": 3}, "c": 4}
    assert _deep_merge(base, over) == {"a": {"b": 1, "d": 3}, "c": 4}

    bconf = tmp_path / "base_config.json"
    bconf.write_text('{"a": {"b": 1}}')
    outconf = tmp_path / "out_config.json"
    res = write_experiment_config(
        base_config_path=bconf, output_path=outconf, overrides={"a": {"c": 2}}
    )
    assert res == outconf
    assert _read_json(outconf) == {"a": {"b": 1, "c": 2}}

    # _maybe_materialize_config
    assert _maybe_materialize_config(config_path=None, config_overrides=None) == (
        None,
        None,
    )
    assert _maybe_materialize_config(config_path=bconf, config_overrides=None) == (
        bconf,
        None,
    )
    p, d = _maybe_materialize_config(config_path=bconf, config_overrides={"x": 1})
    assert p is not None
    assert d is not None
    assert p.name == "experiment_config.json"
    assert _read_json(p) == {"a": {"b": 1}, "x": 1}
    d.cleanup()

    bp, bd = _empty_bank_path()
    assert bp.read_text().strip() == "[]"
    bd.cleanup()

    assert _normalize_image_paths({}) == []
    assert _normalize_image_paths({"image_paths": ["a", "b"]}) == ["a", "b"]


@pytest.mark.unit
def test_run_generation_benchmark(monkeypatch, tmp_path):
    import compos3d.paper.experiments as mod

    benchmark = tmp_path / "bench.json"
    benchmark.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "example_id": "1",
                        "prompt": "a",
                        "room_type": "dining_room",
                        "required_assets": ["table"],
                    },
                    {"example_id": "2", "prompt": "b", "room_type": "dining_room"},
                ]
            }
        )
    )

    out_dir = tmp_path / "out"

    def fake_run_inference(output_dir, **kw):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "inference_manifest.json").write_text(
            '{"selected_hypothesis_ids": ["h1"], "render_scene": true}'
        )
        (output_dir / "scene_program.json").write_text(
            '{"room_type": "dining_room", "assets": [], "image_paths": ["img1.png"]}'
        )
        (output_dir / "critic_score.json").write_text(
            '{"validity": 1.0, "prompt_adherence": 1.0, "asset_precision": 1.0, "asset_recall": 1.0, "room_match": 1.0, "overall": 1.0}'
        )

    monkeypatch.setattr(mod, "run_vertical_inference", fake_run_inference)
    monkeypatch.setattr(
        mod, "build_program_metric_row", lambda **kw: {"exact_room_type_accuracy": 1.0}
    )

    with pytest.raises(ValueError):
        run_generation_benchmark(
            benchmark_path=benchmark, output_dir=out_dir, method_name="m"
        )

    base_conf = tmp_path / "base.json"
    base_conf.write_text('{"a": 0}')
    res = run_generation_benchmark(
        benchmark_path=benchmark,
        output_dir=out_dir,
        method_name="m",
        use_empty_bank=True,
        config_path=base_conf,
        config_overrides={"a": 1},
        example_ids={"1"},
        limit=1,
        force=True,
    )
    assert len(res["rows"]) == 1
    assert res["rows"][0]["example_id"] == "1"
    assert res["results_path"].exists()


@pytest.mark.unit
def test_pairwise_metrics():
    rows = [
        {"asset_selection": "A"},
        {"asset_selection": "A"},
        {"asset_selection": "B"},
        {"asset_selection": "tie"},
    ]
    counts = _pairwise_counts(rows, "asset_selection")
    assert counts == {"A": 2, "B": 1, "tie": 1}

    summary = _pairwise_metric_summary(rows, "asset_selection")
    assert summary["preferred_win_rate"] == 0.5
    assert summary["tie_rate"] == 0.25
    assert summary["other_win_rate"] == 0.25
    assert summary["preferred_score"] == 0.625
    assert summary["preferred_non_tie_win_rate"] == round(2 / 3, 4)

    assert _pairwise_metric_summary([], "asset_selection")["preferred_score"] == 0.0


@pytest.mark.unit
def test_judge_generation_pairwise(monkeypatch, tmp_path):
    import compos3d.paper.experiments as mod

    a_res = tmp_path / "a.jsonl"
    a_res.write_text(
        '{"example_id": "1", "image_paths": ["a1.png"], "prompt": "a", "method_name": "A"}\n'
        '{"example_id": "2", "image_paths": ["a2.png"], "prompt": "b", "method_name": "A"}'
    )

    b_res = tmp_path / "b.jsonl"
    b_res.write_text(
        '{"example_id": "1", "image_paths": ["b1.png"], "prompt": "a", "method_name": "B"}\n'
        '{"example_id": "2", "image_paths": ["b2.png"], "prompt": "b", "method_name": "B"}'
    )

    out_dir = tmp_path / "out"

    class FakeJudge:
        def __init__(self, config):
            pass

        def judge_generation_pair(self, **kw):
            return {
                "asset_selection": "A",
                "layout_coherence": "B",
                "overall_preference": "tie",
            }

    monkeypatch.setattr(mod, "BedrockPairwiseJudge", FakeJudge)

    res = judge_generation_pairwise(
        method_a_results_path=a_res,
        method_b_results_path=b_res,
        output_dir=out_dir,
        judge_config=CriticConfig(),
        sample_size=1,
    )
    assert len(res["rows"]) == 1
    assert res["summary"]["headline_metrics"]["generation_quality_score"] == 0.5

    # Test retry on CriticUnavailableError
    class FlakyJudge:
        def __init__(self, config):
            self.attempts = 0

        def judge_generation_pair(self, **kw):
            self.attempts += 1
            if self.attempts < 2:
                from compos3d.evaluation.critic import CriticUnavailableError

                raise CriticUnavailableError("fail")
            return {
                "asset_selection": "A",
                "layout_coherence": "B",
                "overall_preference": "tie",
            }

    monkeypatch.setattr(mod, "BedrockPairwiseJudge", FlakyJudge)
    monkeypatch.setattr(mod.time, "sleep", lambda x: None)

    res = judge_generation_pairwise(
        method_a_results_path=a_res,
        method_b_results_path=b_res,
        output_dir=tmp_path / "out2",
        judge_config=CriticConfig(),
        force=True,
    )
    assert len(res["rows"]) == 2

    # Test existing complete results
    existing_out = tmp_path / "out3"
    existing_out.mkdir()
    (existing_out / "pairwise_generation.jsonl").write_text(
        '{"example_id": "1", "asset_selection": "A", "method_a": "A"}\n{"example_id": "2", "asset_selection": "B", "method_a": "A"}'
    )
    res = judge_generation_pairwise(
        method_a_results_path=a_res,
        method_b_results_path=b_res,
        output_dir=existing_out,
        judge_config=CriticConfig(),
        force=False,
    )
    assert len(res["rows"]) == 2

    # Test existing partial results
    existing_out4 = tmp_path / "out4"
    existing_out4.mkdir()
    (existing_out4 / "pairwise_generation.jsonl").write_text(
        '{"example_id": "1", "asset_selection": "A", "method_a": "A"}'
    )
    monkeypatch.setattr(mod, "BedrockPairwiseJudge", FakeJudge)
    res = judge_generation_pairwise(
        method_a_results_path=a_res,
        method_b_results_path=b_res,
        output_dir=existing_out4,
        judge_config=CriticConfig(),
        force=False,
    )
    assert len(res["rows"]) == 2

    # Test max retries
    class AlwaysFailJudge:
        def __init__(self, config):
            pass

        def judge_generation_pair(self, **kw):
            from compos3d.evaluation.critic import CriticUnavailableError

            raise CriticUnavailableError("fail always")

    monkeypatch.setattr(mod, "BedrockPairwiseJudge", AlwaysFailJudge)
    with pytest.raises(mod.CriticUnavailableError):
        judge_generation_pairwise(
            method_a_results_path=a_res,
            method_b_results_path=b_res,
            output_dir=tmp_path / "out5",
            judge_config=CriticConfig(),
            force=True,
        )

    # No shared IDs
    b_res.write_text('{"example_id": "3", "image_paths": ["b3.png"]}')
    with pytest.raises(ValueError):
        judge_generation_pairwise(
            method_a_results_path=a_res,
            method_b_results_path=b_res,
            output_dir=out_dir,
            judge_config=CriticConfig(),
        )


@pytest.mark.unit
def test_run_edit_benchmark(monkeypatch, tmp_path):
    import compos3d.paper.experiments as mod

    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text(
        json.dumps(
            {
                "pair_id": "p1",
                "base_prompt": "a",
                "edited_prompt": "b",
                "base_example_id": "b1",
                "edited_example_id": "e1",
                "edit_instruction": "do it",
                "edit_type": "add",
                "base_required_assets": [],
                "base_expected_asset_counts": {},
                "edited_required_assets": [],
                "edited_expected_asset_counts": {},
            }
        )
        + "\n"
    )

    def fake_run_inference(output_dir, **kw):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "inference_manifest.json").write_text("{}")
        (output_dir / "scene_program.json").write_text(
            '{"room_type": "dining_room", "assets": []}'
        )
        (output_dir / "critic_score.json").write_text(
            '{"validity": 1.0, "overall": 1.0, "used_image_paths": ["i.png"]}'
        )

    monkeypatch.setattr(mod, "run_vertical_inference", fake_run_inference)
    monkeypatch.setattr(
        mod, "build_program_metric_row", lambda **kw: {"exact_room_type_accuracy": 1.0}
    )
    monkeypatch.setattr(
        mod, "compute_edit_program_metrics", lambda **kw: {"delta_asset_precision": 1.0}
    )
    monkeypatch.setattr(mod, "clip_directional_similarity", lambda **kw: 0.5)

    class FakeJudge:
        def __init__(self, config):
            pass

        def judge_edit_pair(self, **kw):
            return {"edit_success": 1.0, "preservation": 1.0, "overall": 1.0}

    monkeypatch.setattr(mod, "BedrockPairwiseJudge", FakeJudge)

    out_dir = tmp_path / "out"
    base_conf = tmp_path / "base.json"
    base_conf.write_text('{"a": 0}')
    res = run_edit_benchmark(
        edit_pairs_path=pairs_path,
        output_dir=out_dir,
        method_name="m",
        use_empty_bank=True,
        judge_config=CriticConfig(),
        pair_ids={"p1"},
        limit=1,
        config_path=base_conf,
        config_overrides={"b": 2},
    )
    assert len(res["rows"]) == 1
    assert res["summary"]["by_edit_type"]["add"]["metrics"]["vlm_edit_success"] == 1.0

    # Test missing bank
    with pytest.raises(ValueError, match="bank_path is required"):
        run_edit_benchmark(
            edit_pairs_path=pairs_path,
            output_dir=out_dir,
            method_name="m",
        )

    # Test judge is None
    res2 = run_edit_benchmark(
        edit_pairs_path=pairs_path,
        output_dir=tmp_path / "out_no_judge",
        method_name="m",
        use_empty_bank=True,
        judge_config=None,
    )
    assert res2["rows"][0]["vlm_edit_success"] is None

    # Test summarize_edit_rows with empty
    assert summarize_edit_rows([])["metrics"] == {
        "delta_asset_precision": None,
        "base_overall": None,
        "clip_directional_similarity": None,
        "vlm_edit_success": None,
        "delta_asset_recall": None,
        "delta_asset_f1": None,
        "delta_count_l1": None,
        "unchanged_asset_retention_rate": None,
        "edited_overall": None,
        "vlm_preservation": None,
        "vlm_overall": None,
    }


@pytest.mark.unit
def test_export_human_study_sheets(tmp_path):
    a_res = tmp_path / "a.jsonl"
    a_res.write_text(
        '{"example_id": "1", "image_paths": ["a1.png"], "prompt": "a", "method_name": "A"}'
    )
    b_res = tmp_path / "b.jsonl"
    b_res.write_text(
        '{"example_id": "1", "image_paths": ["b1.png"], "prompt": "a", "method_name": "B"}'
    )

    out_csv = tmp_path / "gen.csv"
    export_generation_human_study_sheet(
        method_a_results_path=a_res,
        method_b_results_path=b_res,
        output_csv_path=out_csv,
    )
    assert out_csv.exists()

    edit_res = tmp_path / "e.jsonl"
    edit_res.write_text(
        json.dumps(
            {
                "pair_id": "p1",
                "edit_type": "add",
                "base_prompt": "a",
                "edited_prompt": "b",
                "edit_instruction": "c",
                "base_image_paths": ["b.png"],
                "edited_image_paths": ["e.png"],
            }
        )
    )

    out_edit = tmp_path / "edit.csv"
    export_edit_human_study_sheet(
        editing_results_path=edit_res, output_csv_path=out_edit
    )
    assert out_edit.exists()


@pytest.mark.unit
def test_build_paper_training_matrix(tmp_path):
    base_config = tmp_path / "base.json"
    base_config.write_text('{"training": {"alpha": 0.5}}')

    out_dir = tmp_path / "out"

    mat = build_paper_training_matrix(
        base_config_path=base_config,
        output_dir=out_dir,
        dataset_path=Path("dataset.json"),
        benchmark_val_path=Path("val.json"),
        benchmark_test_path=Path("test.json"),
    )

    assert len(mat["train_runs"]) == 6
    assert len(mat["sweeps"]) == 10
    assert len(mat["inference_ablations"]) == 8

    assert (out_dir / "experiment_matrix.json").exists()
    assert (out_dir / "run_commands.sh").exists()

    script = (out_dir / "run_commands.sh").read_text()
    assert "compos3d train-hypotheses" in script
    assert "run_paper_generation_eval.py" in script
