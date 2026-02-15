"""
AWS Session Management for Anyscale Jobs

IMPORTANT: Do NOT set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
directly in Anyscale jobs - it breaks Anyscale's artifact storage.

Instead, use custom env var names (APP_AWS_*) and create explicit boto3 sessions.
"""

import os
import boto3
from typing import Optional


def get_session_from_env(
    region: Optional[str] = None,
    access_key_env: str = "APP_AWS_ACCESS_KEY_ID",
    secret_key_env: str = "APP_AWS_SECRET_ACCESS_KEY",
    session_token_env: str = "APP_AWS_SESSION_TOKEN",
    region_env: str = "APP_AWS_REGION",
) -> boto3.Session:
    """
    Create boto3 session from custom env vars (not AWS_* which break Anyscale).

    This is the PROPER pattern for Anyscale hosted cloud:
    1. Pass creds as APP_AWS_* env vars (not AWS_*)
    2. Create explicit boto3.Session with those creds
    3. Anyscale's default AWS_* vars remain intact for artifact storage

    Args:
        region: AWS region (if None, reads from env)
        access_key_env: Env var name for access key ID
        secret_key_env: Env var name for secret access key
        session_token_env: Env var name for session token (optional)
        region_env: Env var name for region

    Returns:
        boto3.Session configured with credentials from custom env vars

    Raises:
        ValueError: If required env vars are not set
    """
    access_key = os.environ.get(access_key_env)
    secret_key = os.environ.get(secret_key_env)
    session_token = os.environ.get(session_token_env)

    if not access_key or not secret_key:
        raise ValueError(
            f"Required env vars not set: {access_key_env}, {secret_key_env}. "
            f"Pass AWS credentials via --env {access_key_env}=... when submitting job."
        )

    if region is None:
        region = os.environ.get(region_env, "us-east-1")

    return boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token,  # Required for temporary creds (SSO)
        region_name=region,
    )


def get_s3_client(region: str = "us-east-1"):
    """Get S3 client using credentials from custom env vars."""
    session = get_session_from_env(region=region)
    return session.client("s3")


def get_s3_resource(region: str = "us-east-1"):
    """Get S3 resource using credentials from custom env vars."""
    session = get_session_from_env(region=region)
    return session.resource("s3")


def get_secrets_client(region: str = "us-east-1"):
    """Get Secrets Manager client using credentials from custom env vars."""
    session = get_session_from_env(region=region)
    return session.client("secretsmanager")


def get_glue_client(region: str = "us-east-1"):
    """Get Glue client using credentials from custom env vars."""
    session = get_session_from_env(region=region)
    return session.client("glue")


# Legacy compatibility
def get_s3_client_for_our_account(
    role_arn: str = None, region: str = "us-east-1", external_id: str = None
):
    """Legacy function - now uses custom env vars instead of role assumption."""
    return get_s3_client(region=region)
