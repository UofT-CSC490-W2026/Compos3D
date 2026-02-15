"""Bronze layer schemas - raw ingestion events"""

from __future__ import annotations
from pydantic import BaseModel
from typing import Literal, Optional
from datetime import datetime


class InfinigenIngestEvent(BaseModel):
    """Event for Infinigen scene generation"""

    event_id: str
    event_type: Literal["infinigen_generate"] = "infinigen_generate"
    timestamp: datetime

    # Infinigen config
    seed: int
    scene_type: str  # "indoors", "nature", "urban"
    num_objects: int

    # Output paths
    scene_json_path: str
    render_dir_path: str
    blender_file_path: Optional[str] = None

    # Metadata
    infinigen_version: str
    generation_time_seconds: float
    status: Literal["success", "failed", "timeout"]
    error_message: Optional[str] = None


class VIGAIterationEvent(BaseModel):
    """Event for VIGA agent iteration"""

    event_id: str
    event_type: Literal["viga_iteration"] = "viga_iteration"
    timestamp: datetime

    # VIGA context
    run_id: str
    iteration_num: int
    task_prompt: str

    # LLM generation
    model_name: str
    generated_code: str
    prompt_tokens: int
    completion_tokens: int

    # Execution
    execution_status: Literal["success", "syntax_error", "runtime_error", "timeout"]
    execution_log: Optional[str] = None

    # Validation
    verifier_feedback: Optional[str] = None
    quality_score: Optional[float] = None

    # Output
    output_scene_path: Optional[str] = None
    output_renders_path: Optional[str] = None
