"""S3 store that routes to separate bronze/silver/gold buckets."""

from __future__ import annotations

import json
import os
from typing import Any

import boto3


class MultiLayerS3Store:
    """Routes writes to different S3 buckets based on the data lake layer.

    Paths beginning with ``bronze/``, ``silver/``, or ``gold/`` are routed to
    the corresponding bucket.  Ambiguous paths default to the bronze bucket.
    """

    def __init__(
        self,
        bucket_bronze: str,
        bucket_silver: str,
        bucket_gold: str,
        prefix: str,
        region: str = "us-east-1",
    ) -> None:
        self.bucket_bronze = bucket_bronze
        self.bucket_silver = bucket_silver
        self.bucket_gold = bucket_gold
        self.prefix = prefix.strip("/")

        self.s3 = boto3.client("s3", region_name=region)

    def _resolve_bucket(self, rel_path: str) -> str:
        lower = rel_path.lower()
        if lower.startswith("bronze/") or "/bronze/" in lower:
            return self.bucket_bronze
        if lower.startswith("silver/") or "/silver/" in lower:
            return self.bucket_silver
        if lower.startswith("gold/") or "/gold/" in lower:
            return self.bucket_gold
        return self.bucket_bronze

    def _key(self, rel_path: str) -> str:
        rel_path = rel_path.lstrip("/")
        return f"{self.prefix}/{rel_path}" if self.prefix else rel_path

    def put_json(self, rel_path: str, obj: Any) -> str:
        bucket = self._resolve_bucket(rel_path)
        key = self._key(rel_path)
        body = json.dumps(obj, indent=2, sort_keys=True).encode()
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
        return json.loads(obj["Body"].read().decode())

    def list_prefix(self, rel_prefix: str) -> list[str]:
        bucket = self._resolve_bucket(rel_prefix)
        prefix_key = self._key(rel_prefix).rstrip("/") + "/"
        out: list[str] = []
        token = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix_key}
            if token:
                kwargs["ContinuationToken"] = token
            resp = self.s3.list_objects_v2(**kwargs)
            for item in resp.get("Contents", []):
                raw_key: str = item["Key"]
                rel = raw_key[len(self.prefix) + 1 :] if self.prefix else raw_key
                out.append(rel)
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return sorted(out)

    def exists(self, rel_path: str) -> bool:
        bucket = self._resolve_bucket(rel_path)
        key = self._key(rel_path)
        try:
            self.s3.head_object(Bucket=bucket, Key=key)
            return True
        except self.s3.exceptions.ClientError:
            return False
