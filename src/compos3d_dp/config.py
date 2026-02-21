from __future__ import annotations
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional
import pathlib
import yaml

EnvName = Literal["dev", "staging", "prod"]
StorageBackend = Literal["local", "s3"]


class LakeLayout(BaseModel):
    # canonical Bronze/Silver/Gold prefixes
    bronze_prefix: str = "bronze"
    silver_prefix: str = "silver"
    gold_prefix: str = "gold"


class AppConfig(BaseSettings):
    """
    12-factor config:
    - defaults are safe for local dev
    - can be overridden by env vars or YAML per environment
    """

    model_config = SettingsConfigDict(env_prefix="COMPOS3D_", extra="ignore")

    env: EnvName = "dev"
    storage_backend: StorageBackend = "local"

    # local lake root (when storage_backend=local)
    local_lake_root: str = "_lake"

    # s3 configuration (when storage_backend=s3)
    s3_bucket: Optional[str] = None  # Legacy single bucket
    s3_bucket_bronze: Optional[str] = None
    s3_bucket_silver: Optional[str] = None
    s3_bucket_gold: Optional[str] = None
    s3_prefix: str = "compos3d"

    aws_region: str = "us-east-1"

    layout: LakeLayout = Field(default_factory=LakeLayout)


def load_config_from_yaml(path: str, env: EnvName) -> AppConfig:
    p = pathlib.Path(path)
    data = yaml.safe_load(p.read_text())
    # allow either a top-level dict or {env: {...}}
    env_data = data.get(env, data)
    return AppConfig(env=env, **env_data)


def load_config(env: EnvName = "dev") -> AppConfig:
    """Load config for a given environment using standard path"""
    config_path = f"config/env.{env}.yaml"
    return load_config_from_yaml(config_path, env)
