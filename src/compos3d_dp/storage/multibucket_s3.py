from __future__ import annotations
import json
import os
from typing import Any
import boto3


class MultiLayerS3Store:
    """S3 Store that routes to different buckets based on layer (bronze/silver/gold)"""

    def __init__(
        self,
        bucket_bronze: str,
        bucket_silver: str,
        bucket_gold: str,
        prefix: str,
        region: str,
    ):
        self.bucket_bronze = bucket_bronze
        self.bucket_silver = bucket_silver
        self.bucket_gold = bucket_gold
        self.prefix = prefix.strip("/")

        # Use APP_AWS_* credentials if available (for Anyscale), otherwise use default chain
        aws_access_key = os.environ.get("APP_AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("APP_AWS_SECRET_ACCESS_KEY")
        aws_session_token = os.environ.get("APP_AWS_SESSION_TOKEN")

        if aws_access_key and aws_secret_key:
            self.s3 = boto3.client(
                "s3",
                region_name=region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                aws_session_token=aws_session_token,
            )
        else:
            self.s3 = boto3.client("s3", region_name=region)

    def _resolve_bucket(self, rel_path: str) -> str:
        """Determine which bucket to use based on path"""
        rel_path_lower = rel_path.lower()
        if rel_path_lower.startswith("bronze/") or "/bronze/" in rel_path_lower:
            return self.bucket_bronze
        elif rel_path_lower.startswith("silver/") or "/silver/" in rel_path_lower:
            return self.bucket_silver
        elif rel_path_lower.startswith("gold/") or "/gold/" in rel_path_lower:
            return self.bucket_gold
        else:
            # Default to bronze for ambiguous paths
            return self.bucket_bronze

    def _key(self, rel_path: str) -> str:
        rel_path = rel_path.lstrip("/")
        return f"{self.prefix}/{rel_path}" if self.prefix else rel_path

    def put_json(self, rel_path: str, obj: Any) -> str:
        bucket = self._resolve_bucket(rel_path)
        key = self._key(rel_path)
        body = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
        self.s3.put_object(
            Bucket=bucket, Key=key, Body=body, ContentType="application/json"
        )
        return f"s3://{bucket}/{key}"

    def put_bytes(
        self, rel_path: str, b: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        bucket = self._resolve_bucket(rel_path)
        key = self._key(rel_path)
        self.s3.put_object(Bucket=bucket, Key=key, Body=b, ContentType=content_type)
        return f"s3://{bucket}/{key}"

    def read_json(self, rel_path: str) -> Any:
        bucket = self._resolve_bucket(rel_path)
        key = self._key(rel_path)
        obj = self.s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

    def list_prefix(self, rel_prefix: str) -> list[str]:
        """Lists objects under prefix and returns rel paths (not s3:// URIs)"""
        bucket = self._resolve_bucket(rel_prefix)
        prefix_key = self._key(rel_prefix).rstrip("/") + "/"
        out: list[str] = []
        token = None
        while True:
            kwargs = {"Bucket": bucket, "Prefix": prefix_key}
            if token:
                kwargs["ContinuationToken"] = token
            resp = self.s3.list_objects_v2(**kwargs)
            for it in resp.get("Contents", []):
                key = it["Key"]
                # convert key back to rel_path under self.prefix
                if self.prefix:
                    rel = key[len(self.prefix) + 1 :]
                else:
                    rel = key
                out.append(rel)
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return sorted(out)
