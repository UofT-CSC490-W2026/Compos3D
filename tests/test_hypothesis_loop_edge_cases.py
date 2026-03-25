"""Edge-case unit tests for `compos3d.hypothesis.loop`.

Guards early returns, threshold logic, bank replacement, renderer and critic
failures, and baseline-mode branches in the hypothesis loop.

Edge cases covered:
- Early-return branches in `_maybe_regenerate`.
- Threshold behavior in `_wrong_threshold`.
- Deduplication and truncation behavior in `_replace_room_bank`.
- Renderer failure handling in `_render`.
- `CriticUnavailableError` fallback payload from `_score`.
- Baseline-mode branches for `_evaluate_hypothesis` and `_build_prediction`.

Expected outcomes:
- Guard branches return quickly without mutating state unexpectedly.
- Threshold math and selection rules are deterministic.
- Failures are handled gracefully and converted into safe fallback scores.
"""

from __future__ import annotations

from pathlib import Path

from compos3d.evaluation.critic import CriticUnavailableError
from compos3d.hypothesis import loop as loop_mod
from compos3d.hypothesis.loop import HypothesisLoopConfig, SceneHypothesisLoop
from compos3d.models import CriticScore, HypothesisRecord, SceneProgram, TrainingDataset, TrainingExample


def _make_loop(**cfg_overrides) -> SceneHypothesisLoop:
    dataset = TrainingDataset(
        dataset_id="d",
        examples=[
            TrainingExample(
                example_id="e1",
                room_type="bedroom",
                prompt="cozy bedroom",
                required_assets=["bed"],
            )
        ],
    )
    cfg = HypothesisLoopConfig(**cfg_overrides)
    return SceneHypothesisLoop(
        dataset=dataset,
        llm=type("L", (), {"generate_hypotheses": lambda *a, **k: []})(),
        critic=object(),
        run_dir=Path("."),
        llm_provider="mock",
        config=cfg,
    )


def test_maybe_regenerate_early_returns() -> None:
    loop = _make_loop(use_repair=False)
    assert loop._maybe_regenerate("bedroom", current_sample=2, epoch=0) is False  # noqa: SLF001

    loop = _make_loop(use_repair=True, baseline_mode="fixed_hypotheses")
    assert loop._maybe_regenerate("bedroom", current_sample=2, epoch=0) is False  # noqa: SLF001

    loop = _make_loop(use_repair=True, update_batch_size=0)
    loop.pending_failure_examples["bedroom"] = [loop.dataset.examples[0]]
    assert loop._maybe_regenerate("bedroom", current_sample=2, epoch=0) is False  # noqa: SLF001

    loop = _make_loop(use_repair=True, update_batch_size=2, num_hypotheses_to_update=2)
    loop.pending_failure_examples["bedroom"] = [loop.dataset.examples[0]]
    assert loop._maybe_regenerate("bedroom", current_sample=2, epoch=0) is False  # noqa: SLF001


def test_wrong_threshold_edge_cases() -> None:
    loop = _make_loop(num_wrong_scale=0.0)
    assert loop._wrong_threshold(0, current_sample=4) == 0.0  # noqa: SLF001
    assert loop._wrong_threshold(3, current_sample=4) == 3.0  # noqa: SLF001


def test_replace_room_bank_dedup_and_cap() -> None:
    loop = _make_loop(max_num_hypotheses_per_room=2)
    loop.bank = [
        HypothesisRecord(
            hypothesis_id="h1",
            text="Place bed near wall",
            room_type="bedroom",
            reward=0.5,
        )
    ]
    new_records = [
        HypothesisRecord(hypothesis_id="h2", text="place BED near wall", room_type="bedroom", reward=0.9),
        HypothesisRecord(hypothesis_id="h3", text="Add lamp by bed", room_type="bedroom", reward=0.8),
        HypothesisRecord(hypothesis_id="h4", text="Keep walkway clear", room_type="bedroom", reward=0.7),
    ]
    loop._replace_room_bank("bedroom", new_records)  # noqa: SLF001
    assert len(loop.bank) == 2
    assert {r.hypothesis_id for r in loop.bank} == {"h3", "h4"}


def test_render_exception_path_returns_empty() -> None:
    loop = _make_loop()
    loop.renderer = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))
    out = loop._render(SceneProgram(prompt="p", room_type="bedroom"), "tag")  # noqa: SLF001
    assert out == []


def test_score_handles_critic_unavailable(monkeypatch) -> None:
    loop = _make_loop()

    def _raise(*_a, **_k):
        raise CriticUnavailableError("missing creds")

    monkeypatch.setattr(loop_mod, "evaluate_scene_program", _raise)
    score = loop._score(SceneProgram(prompt="p", room_type="bedroom"), loop.dataset.examples[0], [])  # noqa: SLF001
    assert score.overall == 0.0
    assert score.critic_mode == "unavailable"
    assert "CriticUnavailable" in score.notes[0]


def test_baseline_mode_no_hypotheses_branch() -> None:
    captured = {}

    class _LLM:
        def generate_scene_program(self, *, prompt, room_type, selected_hypotheses):
            captured["selected_hypotheses"] = selected_hypotheses
            return SceneProgram(prompt=prompt, room_type=room_type)

    loop = _make_loop(baseline_mode="no_hypotheses")
    loop.llm = _LLM()
    loop._score = lambda *_a, **_k: CriticScore(  # noqa: SLF001
        validity=1.0,
        prompt_adherence=1.0,
        asset_precision=1.0,
        asset_recall=1.0,
        room_match=1.0,
        overall=1.0,
    )
    loop._render = lambda *_a, **_k: []  # noqa: SLF001
    loop._evaluate_hypothesis("use bed", loop.dataset.examples[0])  # noqa: SLF001
    pred = loop._build_prediction(loop.dataset.examples[0], ["h1", "h2"])  # noqa: SLF001
    assert captured["selected_hypotheses"] == []
    assert pred.selected_hypotheses == []

