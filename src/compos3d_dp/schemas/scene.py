from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
from datetime import datetime


class SceneObject(BaseModel):
    """Single object instance in a 3D scene"""

    object_id: str
    category: str
    asset_id: Optional[str] = None
    position_xyz: List[float] = Field(..., min_length=3, max_length=3)
    rotation_xyz: List[float] = Field(..., min_length=3, max_length=3)
    scale_xyz: List[float] = Field(..., min_length=3, max_length=3)

    # Optional physics/contact info
    has_collision: bool = True
    is_static: bool = False
    mass_kg: Optional[float] = None


class SpatialRelation(BaseModel):
    """Spatial/topological relationship between two objects"""

    object_id_a: str
    object_id_b: str
    relation_type: Literal["on", "in", "near", "touching", "supporting", "occluding"]
    confidence: float = 1.0  # For learned/inferred relations
    distance_meters: Optional[float] = None


class SceneRecord(BaseModel):
    """Complete scene representation for Bronze layer (immutable raw event)"""

    schema_version: Literal["v1"] = "v1"
    scene_id: str
    dataset: str = "infinigen_indoors"
    seed: int
    split: Literal["train", "val", "test"]
    objects: List[SceneObject]

    # Optional: spatial relations graph
    relations: Optional[List[SpatialRelation]] = None

    # Metadata for VIGA-style generation
    generated_by: Optional[Literal["infinigen", "viga_agent", "manual"]] = None
    generation_prompt: Optional[str] = None
    hypotheses_used: Optional[List[str]] = None  # Hypothesis IDs from HypoGeniC bank
    blender_code: Optional[str] = None  # Generated Blender Python code

    # Timestamps and provenance
    created_at: Optional[datetime] = None
    run_id: Optional[str] = None
