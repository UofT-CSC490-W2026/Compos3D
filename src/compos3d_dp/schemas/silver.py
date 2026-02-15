"""Silver layer schemas - clean, validated tables"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from datetime import datetime


class SilverScene(BaseModel):
    """Silver scene table - one row per scene"""

    scene_id: str
    source: Literal["infinigen", "viga", "manual"]
    dataset: str  # "infinigen_indoors_v1", etc
    seed: int
    split: Literal["train", "val", "test"]

    # Counts
    object_count: int
    relation_count: int
    render_count: int

    # Quality flags
    has_collisions: bool
    has_floating_objects: bool
    physics_stable: bool

    # Timestamps
    created_at: datetime
    ingested_at: datetime

    # Partitions
    date: str  # YYYY-MM-DD


class SilverSceneObject(BaseModel):
    """Silver object table - one row per object"""

    scene_id: str
    object_id: str
    category: str
    asset_id: Optional[str] = None

    # Transforms
    px: float
    py: float
    pz: float
    rx: float
    ry: float
    rz: float
    sx: float
    sy: float
    sz: float

    # Physics
    has_collision: bool
    is_static: bool
    mass_kg: Optional[float] = None

    # Partitions
    split: Literal["train", "val", "test"]
    date: str


class SilverSpatialRelation(BaseModel):
    """Silver relations table - one row per relation"""

    scene_id: str
    object_id_a: str
    object_id_b: str
    relation_type: Literal["on", "in", "near", "touching", "supporting"]
    confidence: float = Field(ge=0.0, le=1.0)
    distance_meters: Optional[float] = None

    # Partitions
    split: Literal["train", "val", "test"]
    date: str


class SilverRender(BaseModel):
    """Silver render table - one row per rendered view"""

    render_id: str
    scene_id: str
    camera_id: str
    frame_idx: int

    # Image paths
    rgb_uri: str
    depth_uri: Optional[str] = None
    normal_uri: Optional[str] = None
    segmentation_uri: Optional[str] = None

    # Camera parameters
    camera_position: List[float]  # [x, y, z]
    camera_rotation: List[float]  # [rx, ry, rz]
    fov: float
    resolution: List[int]  # [width, height]

    # Embeddings (for retrieval)
    clip_embedding_uri: Optional[str] = None

    # Partitions
    split: Literal["train", "val", "test"]
    date: str


class SilverQualityScore(BaseModel):
    """Silver quality table - one row per scene evaluation"""

    eval_id: str
    scene_id: str

    # Automatic checks
    collision_count: int
    floating_object_count: int
    physics_stable: bool
    objects_in_bounds: bool

    # Learned metrics (0-1)
    aesthetic_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    realism_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    layout_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Overall
    total_reward: float

    # Metadata
    evaluated_at: datetime
    evaluator_version: str

    # Partitions
    split: Literal["train", "val", "test"]
    date: str


class SilverHypothesisUsage(BaseModel):
    """Silver hypothesis usage - track which hypotheses were used"""

    usage_id: str
    scene_id: str
    hypothesis_id: str

    # Application
    was_applicable: bool
    was_selected: bool  # By bandit algorithm

    # Reward
    scene_reward: float
    hypothesis_contribution: Optional[float] = None

    # Metadata
    timestamp: datetime

    # Partitions
    split: Literal["train", "val", "test"]
    date: str
