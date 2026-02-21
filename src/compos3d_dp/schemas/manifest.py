"""Run manifest schemas for reproducibility and lineage tracking"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime


class GitInfo(BaseModel):
    """Git provenance information"""

    commit_sha: str
    branch: str
    is_dirty: bool = False  # Uncommitted changes
    remote_url: Optional[str] = None


class ModelInfo(BaseModel):
    """Model/checkpoint information"""

    model_name: str  # e.g., "gpt-4", "claude-3-opus"
    version: Optional[str] = None
    checkpoint_uri: Optional[str] = None  # For custom models
    temperature: Optional[float] = None
    other_params: Dict[str, Any] = Field(default_factory=dict)


class DatasetInfo(BaseModel):
    """Input dataset reference"""

    dataset_name: str  # e.g., "infinigen_indoors_v1.2"
    version: str
    split: Literal["train", "val", "test"]
    uri: Optional[str] = None  # S3/GCS path to source data
    num_examples: Optional[int] = None


class RunManifest(BaseModel):
    """Complete provenance manifest for a data pipeline run"""

    schema_version: Literal["v1"] = "v1"
    run_id: str
    run_type: Literal[
        "bronze_ingest", "silver_transform", "gold_aggregate", "training", "inference"
    ]

    # Timestamps
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Code provenance
    git_info: Optional[GitInfo] = None
    docker_image: Optional[str] = None  # e.g., "compos3d:v1.2.3-gpu"
    python_version: str
    package_versions: Dict[str, str] = Field(default_factory=dict)  # Key packages

    # Model/LLM provenance (for generation runs)
    models: List[ModelInfo] = Field(default_factory=list)

    # Input data
    input_datasets: List[DatasetInfo] = Field(default_factory=list)
    input_uris: List[str] = Field(default_factory=list)

    # Output data
    output_uris: List[str] = Field(default_factory=list)
    output_record_counts: Dict[str, int] = Field(
        default_factory=dict
    )  # table_name -> count

    # Execution environment
    compute_platform: Literal["local", "aws_batch", "modal", "anyscale", "other"] = (
        "local"
    )
    instance_type: Optional[str] = None  # e.g., "g5.xlarge"
    num_workers: int = 1

    # Configuration snapshot (YAML/JSON dump)
    config_snapshot: Dict[str, Any] = Field(default_factory=dict)

    # Status
    status: Literal["running", "success", "failed", "cancelled"] = "running"
    error_message: Optional[str] = None

    # Data quality summary
    quality_checks_passed: Optional[int] = None
    quality_checks_failed: Optional[int] = None
    data_quality_report_uri: Optional[str] = None  # Link to GE HTML report
