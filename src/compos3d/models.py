from __future__ import annotations

from pydantic import BaseModel, Field


class AssetSpec(BaseModel):
    asset_type: str
    count: int = Field(default=1, ge=1)
    placement: str = ""
    rationale: str = ""


class ConstraintSpec(BaseModel):
    text: str


class RenderSpec(BaseModel):
    mode: str = "program_only"


class SceneProgram(BaseModel):
    prompt: str
    room_type: str
    style: str | None = None
    hypotheses: list[str] = Field(default_factory=list)
    assets: list[AssetSpec] = Field(default_factory=list)
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    render_spec: RenderSpec = Field(default_factory=RenderSpec)


class TrainingExample(BaseModel):
    example_id: str
    room_type: str
    prompt: str
    required_assets: list[str]
    style: str | None = None


class TrainingDataset(BaseModel):
    dataset_id: str
    examples: list[TrainingExample]


class HypothesisRecord(BaseModel):
    hypothesis_id: str
    text: str
    room_type: str
    applicability_tags: list[str] = Field(default_factory=list)
    reward: float = 0.0
    accuracy: float = 0.0
    mean_score: float = 0.0
    num_visits: int = 0
    num_successes: int = 0
    generation_round: int = 0
    source_example_ids: list[str] = Field(default_factory=list)
    support_example_ids: list[str] = Field(default_factory=list)
    failure_tags: list[str] = Field(default_factory=list)


class CriticScore(BaseModel):
    validity: float
    prompt_adherence: float
    asset_precision: float
    asset_recall: float
    room_match: float
    overall: float
    notes: list[str] = Field(default_factory=list)
    critic_mode: str = "heuristic"
    used_image_paths: list[str] = Field(default_factory=list)


class PredictionRecord(BaseModel):
    example_id: str | None = None
    prompt: str
    room_type: str
    selected_hypotheses: list[str]
    scene_program: SceneProgram
    critic_score: CriticScore
    image_paths: list[str] = Field(default_factory=list)


class FailureRecord(BaseModel):
    example_id: str
    room_type: str
    prompt: str
    current_sample: int
    selected_hypothesis_ids: list[str] = Field(default_factory=list)
    selected_hypotheses: list[str] = Field(default_factory=list)
    num_wrong_hypotheses: int = 0
    wrong_threshold: int = 0
    combined_score: float = 0.0
    individual_scores: dict[str, float] = Field(default_factory=dict)
    critic_notes: list[str] = Field(default_factory=list)


class TrainingTraceRecord(BaseModel):
    epoch: int
    current_sample: int
    example_id: str
    room_type: str
    selected_hypothesis_ids: list[str] = Field(default_factory=list)
    selected_hypotheses: list[str] = Field(default_factory=list)
    individual_scores: dict[str, float] = Field(default_factory=dict)
    combined_score: float = 0.0
    triggered_regeneration: bool = False
    failure_buffer_size: int = 0


class EvaluationSummary(BaseModel):
    num_predictions: int
    average_validity: float
    average_prompt_adherence: float
    average_asset_precision: float
    average_asset_recall: float
    average_room_match: float
    average_overall: float
