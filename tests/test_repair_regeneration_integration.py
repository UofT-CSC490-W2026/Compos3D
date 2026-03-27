"""Integration-style tests for repair/regeneration flow in hypothesis training.

Forced low critic scores drive repair regeneration and failure-bank recording
while the loop still completes.

Edge cases covered:
- Critic always below success threshold (persistent failure until repair/regen).
- Failure bank (`failed_scene_bank.jsonl`) and trace artifacts populated.
- Repair focus (`focus="repair"`) and pending-failure buffer clearing.
- Training completes and writes normal outputs despite the stress scenario.

Expected outcomes:
- Training returns successfully with at least one regeneration event.
- Failure-bank and trace artifacts are emitted and non-empty.
- Repair branch is exercised without crashing the full loop.
"""

from __future__ import annotations

import json
from pathlib import Path

from compos3d.hypothesis.loop import HypothesisLoopConfig, SceneHypothesisLoop
from compos3d.models import CriticScore, SceneProgram, TrainingDataset, TrainingExample


class _AlwaysFailCritic:
    def evaluate(self, *, scene_program, reference_example=None, image_paths=None):
        return CriticScore(
            validity=0.2,
            prompt_adherence=0.1,
            asset_precision=0.0,
            asset_recall=0.0,
            room_match=1.0,
            overall=0.0,  # Always below threshold to force failure/repair path.
            notes=["forced-fail"],
        )


class _RepairAwareLLM:
    def __init__(self) -> None:
        self.focus_calls: list[str] = []

    def generate_hypotheses(
        self,
        room_type: str,
        examples: list[TrainingExample],
        *,
        num_hypotheses: int,
        focus: str,
    ) -> list[str]:
        self.focus_calls.append(focus)
        if focus == "repair":
            return [f"repair rule {i} for {room_type}" for i in range(num_hypotheses)]
        return [f"initial rule {i} for {room_type}" for i in range(num_hypotheses)]

    def generate_scene_program(
        self, *, prompt: str, room_type: str, selected_hypotheses: list[str]
    ) -> SceneProgram:
        return SceneProgram(
            prompt=prompt,
            room_type=room_type,
            hypotheses=selected_hypotheses,
        )


def test_repair_regeneration_path_writes_failure_artifacts_and_completes(
    tmp_path: Path,
) -> None:
    dataset = TrainingDataset(
        dataset_id="regen_ds",
        examples=[
            TrainingExample(
                example_id="e1",
                room_type="bedroom",
                prompt="bedroom with a bed",
                required_assets=["bed"],
            ),
            TrainingExample(
                example_id="e2",
                room_type="bedroom",
                prompt="bedroom with a lamp",
                required_assets=["lamp"],
            ),
        ],
    )

    llm = _RepairAwareLLM()
    loop = SceneHypothesisLoop(
        dataset=dataset,
        llm=llm,
        critic=_AlwaysFailCritic(),
        run_dir=tmp_path / "run",
        llm_provider="mock",
        config=HypothesisLoopConfig(
            num_init_examples_per_room=1,
            init_hypotheses_per_room=2,
            k=1,
            num_epochs=2,
            success_threshold=0.65,
            update_batch_size=1,
            num_hypotheses_to_update=1,
            update_hypotheses_per_batch=1,
            save_every_n_examples=1,
            num_wrong_scale=0.0,  # deterministic failure threshold = num_selected
        ),
    )

    result = loop.train()

    assert result["num_predictions"] >= 1
    assert result["num_regeneration_events"] >= 1
    assert "repair" in llm.focus_calls
    assert loop.pending_failure_examples["bedroom"] == []

    failed_path = tmp_path / "run" / "failed_scene_bank.jsonl"
    assert failed_path.exists()
    rows = [
        json.loads(line)
        for line in failed_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) >= 1
    assert rows[0]["room_type"] == "bedroom"

    # Core outputs should still be produced despite repeated failures.
    assert (tmp_path / "run" / "predictions.jsonl").exists()
    assert (tmp_path / "run" / "training_trace.jsonl").exists()
    assert (tmp_path / "run" / "manifest.json").exists()
