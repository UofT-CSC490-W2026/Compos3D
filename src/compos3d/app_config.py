"""Infrastructure / environment configuration (separate from training hyperparameters).

Follows a 12-factor design: safe defaults for local dev, overridable by
environment variables (``COMPOS3D_*``) or by loading a YAML config file.

Usage
-----
Load the right config for a given environment:

    from compos3d.app_config import load_app_config
    cfg = load_app_config("dev")      # reads config/env.dev.yaml
    cfg = load_app_config("prod")     # reads config/env.prod.yaml
    cfg = load_app_config()           # respects COMPOS3D_ENV, defaults to "local"
"""

from __future__ import annotations

import os
import pathlib
from typing import Literal, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EnvName = Literal["local", "dev", "staging", "prod"]
StorageBackend = Literal["local", "s3"]

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent


class AppConfig(BaseSettings):
    """Top-level application / infrastructure config."""

    model_config = SettingsConfigDict(env_prefix="COMPOS3D_", extra="ignore")

    env: EnvName = "local"
    storage_backend: StorageBackend = "local"

    # Local lake root (used when storage_backend=local).
    local_lake_root: str = "_lake"

    # S3 configuration (used when storage_backend=s3).
    s3_bucket_bronze: Optional[str] = None
    s3_bucket_silver: Optional[str] = None
    s3_bucket_gold: Optional[str] = None
    s3_prefix: str = "compos3d"
    aws_region: str = "us-east-1"

    # EC2 compute options (used by launch-aws command).
    ec2_instance_type: str = "g5.xlarge"
    ec2_ami_id: Optional[str] = (
        None  # defaults to latest Deep Learning AMI at launch time
    )
    ec2_key_name: Optional[str] = None  # EC2 key pair name for SSH fallback
    ec2_subnet_id: Optional[str] = None  # VPC subnet (uses default VPC if None)
    ec2_security_group_id: Optional[str] = None
    ec2_iam_instance_profile: str = (
        "compos3d-ec2-job"  # instance profile with S3 + CW access
    )
    ec2_spot: bool = True  # use spot instances by default
    ec2_spot_max_price: Optional[str] = None  # None → on-demand price cap


def load_app_config(env: EnvName | None = None) -> AppConfig:
    """Load ``AppConfig`` for the given environment.

    Resolution order (highest to lowest priority):

    1. ``config/env.<env>.yaml`` file (if it exists)
    2. ``COMPOS3D_*`` environment variables
    3. Pydantic field defaults (local dev safe)

    Parameters
    ----------
    env:
        One of ``"local"``/``"dev"``/``"staging"``/``"prod"``.  If *None*,
        reads from ``COMPOS3D_ENV`` env var; falls back to ``"local"``.
    """
    if env is None:
        env = os.environ.get("COMPOS3D_ENV", "local")  # type: ignore[assignment]

    config_file = _REPO_ROOT / "config" / f"env.{env}.yaml"
    if config_file.exists():
        raw = yaml.safe_load(config_file.read_text()) or {}
        raw.pop("env", None)  # avoid duplicate keyword argument
        return AppConfig(env=env, **raw)

    # No YAML file — rely on env vars / defaults.
    return AppConfig(env=env)
