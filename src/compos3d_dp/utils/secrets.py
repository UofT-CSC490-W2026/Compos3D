"""Secure secrets management using AWS Secrets Manager"""

from __future__ import annotations
import os
import boto3
import json
from functools import lru_cache


@lru_cache(maxsize=10)
def get_secret(secret_name: str, region: str = "us-east-1") -> str:
    """
    Retrieve secret from AWS Secrets Manager with caching.

    Falls back to environment variables for local development.

    Args:
        secret_name: Name of the secret (e.g., "compos3d-dev-openai-key")
        region: AWS region

    Returns:
        Secret value as string
    """
    # Check environment variable first (for local dev)
    env_var = secret_name.replace("-", "_").upper()
    if os.getenv(env_var):
        return os.getenv(env_var)

    # Fetch from AWS Secrets Manager
    try:
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)

        # Handle both string and JSON secrets
        if "SecretString" in response:
            secret = response["SecretString"]
            try:
                # Try to parse as JSON
                secret_dict = json.loads(secret)
                # If JSON with single key, return the value
                if len(secret_dict) == 1:
                    return list(secret_dict.values())[0]
                return secret
            except json.JSONDecodeError:
                return secret
        else:
            raise ValueError(f"Secret {secret_name} is binary, not supported")

    except Exception as e:
        raise RuntimeError(f"Failed to retrieve secret {secret_name}: {e}")


def get_openai_key(env: str = "dev") -> str:
    """Get OpenAI API key"""
    return get_secret(f"compos3d-{env}-openai-key")


def get_anthropic_key(env: str = "dev") -> str:
    """Get Anthropic API key"""
    return get_secret(f"compos3d-{env}-anthropic-key")


def get_anyscale_key(env: str = "dev") -> str:
    """Get Anyscale API key"""
    return get_secret(f"compos3d-{env}-anyscale-key")


def get_wandb_key(env: str = "dev") -> str:
    """Get Weights & Biases API key"""
    return get_secret(f"compos3d-{env}-wandb-key")


def set_secret(secret_name: str, secret_value: str, region: str = "us-east-1") -> None:
    """
    Set or update a secret in AWS Secrets Manager.

    Args:
        secret_name: Name of the secret
        secret_value: Value to store
        region: AWS region
    """
    client = boto3.client("secretsmanager", region_name=region)

    try:
        client.put_secret_value(SecretId=secret_name, SecretString=secret_value)
    except client.exceptions.ResourceNotFoundException:
        # Secret doesn't exist, create it
        client.create_secret(Name=secret_name, SecretString=secret_value)
    except Exception as e:
        raise RuntimeError(f"Failed to set secret {secret_name}: {e}")
