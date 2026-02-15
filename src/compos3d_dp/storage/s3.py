from __future__ import annotations
import json
from typing import Any
import boto3


class S3Store:
    def __init__(self, bucket: str, prefix: str, region: str):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.s3 = boto3.client("s3", region_name=region)

    def _key(self, rel_path: str) -> str:
        rel_path = rel_path.lstrip("/")
        return f"{self.prefix}/{rel_path}" if self.prefix else rel_path

    def put_json(self, rel_path: str, obj: Any) -> str:
        key = self._key(rel_path)
        body = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
        self.s3.put_object(
            Bucket=self.bucket, Key=key, Body=body, ContentType="application/json"
        )
        return f"s3://{self.bucket}/{key}"

    def put_bytes(
        self, rel_path: str, b: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        key = self._key(rel_path)
        self.s3.put_object(
            Bucket=self.bucket, Key=key, Body=b, ContentType=content_type
        )
        return f"s3://{self.bucket}/{key}"

    def read_json(self, rel_path: str) -> Any:
        key = self._key(rel_path)
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

    def list_prefix(self, rel_prefix: str) -> list[str]:
        # lists objects under prefix and returns rel paths (not s3:// URIs)
        prefix_key = self._key(rel_prefix).rstrip("/") + "/"
        out: list[str] = []
        token = None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": prefix_key}
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
