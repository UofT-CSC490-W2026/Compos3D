from __future__ import annotations

import json
from pathlib import Path

from compos3d.config import GeneratorConfig
from compos3d.hypothesis import engine
from compos3d.llm.scene_llm import BedrockSceneLLM, MockSceneLLM, StructuredOutputError
from compos3d.models import AssetSpec, CriticScore, HypothesisRecord, SceneProgram


def _make_record(
    hypothesis_id: str,
    text: str,
    *,
    room_type: str = "dining_room",
    accuracy: float = 0.5,
    reward: float = 0.5,
    mean_score: float = 0.5,
) -> HypothesisRecord:
    return HypothesisRecord(
        hypothesis_id=hypothesis_id,
        text=text,
        room_type=room_type,
        accuracy=accuracy,
        reward=reward,
        mean_score=mean_score,
        num_visits=4,
        num_successes=2,
    )


def test_mock_llm_filter_relevant_hypotheses_prefers_prompt_overlap() -> None:
    llm = MockSceneLLM()
    relevant = llm.filter_relevant_hypotheses(
        prompt="a dining room with a dining table and four chairs",
        room_type="dining_room",
        candidate_hypotheses=[
            "chairs should surround the dining_table",
            "a lamp can soften the corner",
        ],
    )

    assert relevant == ["chairs should surround the dining_table"]


def test_bedrock_filter_relevant_hypotheses_bad_payload_raises() -> None:
    llm = BedrockSceneLLM(GeneratorConfig(provider="bedrock"))
    llm._run_json_prompt = lambda _p: {"relevant_hypotheses": "bad"}  # noqa: SLF001

    try:
        llm.filter_relevant_hypotheses(
            prompt="p",
            room_type="bedroom",
            candidate_hypotheses=["h1"],
        )
    except StructuredOutputError as exc:
        assert "relevant_hypotheses list" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected StructuredOutputError")


def test_weighted_vote_scene_program_aggregates_assets_and_style() -> None:
    records = [
        _make_record("h1", "chairs around the dining_table", accuracy=0.9),
        _make_record("h2", "include a rug under the dining_table", accuracy=0.4),
    ]
    candidate_programs = [
        SceneProgram(
            prompt="p",
            room_type="dining_room",
            style="modern",
            hypotheses=["h1"],
            assets=[
                AssetSpec(
                    asset_type="dining_table",
                    count=1,
                    placement="center of room",
                    rationale="anchor",
                ),
                AssetSpec(
                    asset_type="chair",
                    count=4,
                    placement="around table",
                    rationale="seat guests",
                ),
            ],
        ),
        SceneProgram(
            prompt="p",
            room_type="dining_room",
            style="cozy",
            hypotheses=["h2"],
            assets=[
                AssetSpec(
                    asset_type="dining_table",
                    count=1,
                    placement="center of room",
                    rationale="anchor",
                ),
                AssetSpec(
                    asset_type="rug",
                    count=1,
                    placement="under table",
                    rationale="ground the layout",
                ),
            ],
        ),
    ]

    program = engine._weighted_vote_scene_program(  # noqa: SLF001
        prompt="a modern dining room with a dining table and four chairs",
        room_type="dining_room",
        records=records,
        candidate_programs=candidate_programs,
    )

    assert program.style == "modern"
    assert [asset.asset_type for asset in program.assets[:2]] == [
        "dining_table",
        "chair",
    ]
    assert any(asset.asset_type == "rug" for asset in program.assets)
    chair = next(asset for asset in program.assets if asset.asset_type == "chair")
    assert chair.count == 4


def test_filter_and_weight_falls_back_to_best_accuracy_when_filter_empty() -> None:
    class _LLM:
        def filter_relevant_hypotheses(self, **_kwargs):
            return []

        def generate_scene_program(self, *, prompt, room_type, selected_hypotheses):
            asset_type = "chair" if "chair" in selected_hypotheses[0] else "lamp"
            return SceneProgram(
                prompt=prompt,
                room_type=room_type,
                hypotheses=selected_hypotheses,
                assets=[
                    AssetSpec(
                        asset_type="dining_table",
                        count=1,
                        placement="center of room",
                        rationale="anchor",
                    ),
                    AssetSpec(
                        asset_type=asset_type,
                        count=1,
                        placement="near table",
                        rationale="fallback",
                    ),
                ],
            )

    chosen, program = engine._run_filter_and_weight_inference(  # noqa: SLF001
        llm=_LLM(),
        records=[
            _make_record("h1", "chair rule", accuracy=0.9),
            _make_record("h2", "lamp rule", accuracy=0.2),
        ],
        prompt="a dining room with a table and chair",
        room_type="dining_room",
    )

    assert [record.hypothesis_id for record in chosen] == ["h1"]
    assert any(asset.asset_type == "chair" for asset in program.assets)


def test_run_vertical_inference_filter_and_weight_end_to_end(
    monkeypatch, tmp_path: Path
) -> None:
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(
        json.dumps(
            [
                _make_record(
                    "h1",
                    "chairs should surround the dining_table",
                    accuracy=0.9,
                    reward=0.9,
                ).model_dump(),
                _make_record(
                    "h2",
                    "a lamp can sit near the dining_table",
                    accuracy=0.4,
                    reward=0.8,
                ).model_dump(),
            ]
        )
    )

    class _LLM:
        def filter_relevant_hypotheses(self, **_kwargs):
            return ["chairs should surround the dining_table"]

        def generate_scene_program(self, *, prompt, room_type, selected_hypotheses):
            asset_type = "chair" if "chair" in selected_hypotheses[0] else "lamp"
            return SceneProgram(
                prompt=prompt,
                room_type=room_type,
                style="modern",
                hypotheses=selected_hypotheses,
                assets=[
                    AssetSpec(
                        asset_type="dining_table",
                        count=1,
                        placement="center of room",
                        rationale="anchor",
                    ),
                    AssetSpec(
                        asset_type=asset_type,
                        count=4 if asset_type == "chair" else 1,
                        placement="around table" if asset_type == "chair" else "corner",
                        rationale="selected",
                    ),
                ],
            )

    monkeypatch.setattr(engine, "build_scene_llm", lambda _cfg: _LLM())
    monkeypatch.setattr(engine, "build_scene_critic", lambda _cfg: object())
    monkeypatch.setattr(
        engine,
        "evaluate_scene_program",
        lambda *a, **k: CriticScore(
            validity=1.0,
            prompt_adherence=1.0,
            asset_precision=1.0,
            asset_recall=1.0,
            room_match=1.0,
            overall=1.0,
        ),
    )

    result = engine.run_vertical_inference(
        bank_path=bank_path,
        prompt="a modern dining room with a dining table and four chairs",
        output_dir=tmp_path / "inference",
        llm_provider="mock",
        inference_strategy="filter_and_weight",
        render_scene=False,
    )

    assert result["inference_strategy"] == "filter_and_weight"
    assert result["selected_hypothesis_ids"] == ["h1"]
    scene_program = json.loads(Path(result["scene_program_path"]).read_text())
    assert scene_program["hypotheses"] == ["chairs should surround the dining_table"]
    assert any(asset["asset_type"] == "chair" for asset in scene_program["assets"])
