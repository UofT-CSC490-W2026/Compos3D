"""Gold layer schemas - aggregates and training datasets"""

from __future__ import annotations
from pydantic import BaseModel
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime


class GoldTrainingDataset(BaseModel):
    """Metadata for training datasets"""

    dataset_id: str
    dataset_name: str
    version: str  # v1, v2, etc

    # Purpose
    task_type: Literal["scene_generation", "quality_prediction", "hypothesis_selection"]

    # Statistics
    num_scenes: int
    num_train: int
    num_val: int
    num_test: int

    # Data location
    s3_uri: str
    index_file_uri: str  # Parquet index for fast lookup

    # Metadata
    created_at: datetime
    source_tables: List[str]  # Which Silver tables were used
    filters_applied: Dict[str, Any]  # SQL filters used

    # Quality
    data_quality_score: float
    completeness: float  # % of records with all fields


class GoldHypothesisBank(BaseModel):
    """HypoGeniC hypothesis bank entry"""

    hypothesis_id: str
    hypothesis_text: str
    hypothesis_category: Literal["spatial", "semantic", "aesthetic", "physics"]

    # Bandit algorithm stats
    times_selected: int
    times_applicable: int

    # UCB scores
    mean_reward: float
    reward_variance: float
    ucb_score: float
    confidence_interval: List[float]  # [lower, upper]

    # Performance tracking
    success_rate: float
    avg_scene_quality: float

    # Metadata
    created_at: datetime
    last_updated: datetime
    version: int
    is_active: bool

    # Example scenes
    best_scene_ids: List[str]  # Top 5 scenes using this hypothesis
    worst_scene_ids: List[str]  # Bottom 5 for debugging


class GoldModelLeaderboard(BaseModel):
    """Model performance tracking"""

    model_id: str
    model_name: str
    model_type: Literal["scene_generator", "quality_critic", "hypothesis_selector"]

    # Version info
    version: str
    checkpoint_uri: str
    training_dataset_id: str

    # Performance metrics
    val_loss: float
    test_loss: float

    # Task-specific metrics
    metrics: Dict[str, float]  # e.g., {"fid": 45.2, "lpips": 0.15}

    # Resource usage
    training_time_hours: float
    num_gpus_used: int
    total_cost_usd: Optional[float] = None

    # Metadata
    trained_at: datetime
    trained_by: str  # User or system
    hyperparams: Dict[str, Any]

    # Deployment
    is_deployed: bool
    deployment_env: Optional[Literal["dev", "staging", "prod"]] = None


class GoldSceneStatistics(BaseModel):
    """Aggregated scene statistics (daily rollup)"""

    stat_date: str  # YYYY-MM-DD

    # Volume
    total_scenes: int
    scenes_by_source: Dict[str, int]  # {"infinigen": 1000, "viga": 500}
    scenes_by_split: Dict[str, int]

    # Quality
    avg_quality_score: float
    quality_score_p50: float
    quality_score_p95: float

    # Object statistics
    total_objects: int
    objects_by_category: Dict[str, int]  # {"chair": 1500, "table": 800}
    avg_objects_per_scene: float

    # Relations
    total_relations: int
    relations_by_type: Dict[str, int]

    # Renders
    total_renders: int
    avg_renders_per_scene: float

    # Issues
    scenes_with_collisions: int
    scenes_with_floating_objects: int
    scenes_unstable: int


class GoldGenerationCosts(BaseModel):
    """Track generation costs (daily rollup)"""

    cost_date: str  # YYYY-MM-DD

    # LLM costs
    total_prompt_tokens: int
    total_completion_tokens: int
    llm_cost_usd: float

    # Compute costs
    gpu_hours: float
    cpu_hours: float
    compute_cost_usd: float

    # Storage costs
    storage_gb: float
    storage_cost_usd: float

    # Total
    total_cost_usd: float

    # Efficiency metrics
    cost_per_scene: float
    cost_per_successful_scene: float

    # Breakdown by source
    costs_by_source: Dict[str, float]  # {"infinigen": 10.5, "viga": 25.3}
