"""Unit tests for `compos3d.storage.multibucket_s3`.

Why these tests:
- Local storage is already covered, but the S3 implementation has separate
  routing logic, pagination behavior, and error handling.
- These branches are pure unit-test targets because they can be verified with
  a fake boto3 client and no real AWS access.

Edge cases covered:
- Layered paths route to the matching bucket, case-insensitively.
- Ambiguous paths default to bronze.
- Paginated `list_prefix` responses are fully collected.
- Head-object client errors are handled gracefully by `exists()`.
"""

from __future__ import annotations

from compos3d.storage.multibucket_s3 import MultiLayerS3Store


class _FakeClientError(Exception):
    pass


class _FakeS3Client:
    class exceptions:
        ClientError = _FakeClientError

    def __init__(self) -> None:
        self.put_calls: list[dict] = []
        self.get_calls: list[dict] = []
        self.list_calls: list[dict] = []
        self.head_calls: list[dict] = []

    def put_object(self, **kwargs):
        self.put_calls.append(kwargs)
        return {"ok": True}

    def get_object(self, **kwargs):
        self.get_calls.append(kwargs)
        return {"Body": _Body('{"value": 3}')}

    def list_objects_v2(self, **kwargs):
        self.list_calls.append(kwargs)
        if len(self.list_calls) == 1:
            return {
                "Contents": [{"Key": "prefix/silver/run/a.json"}],
                "IsTruncated": True,
                "NextContinuationToken": "token-2",
            }
        return {
            "Contents": [{"Key": "prefix/silver/run/b.json"}],
            "IsTruncated": False,
        }

    def head_object(self, **kwargs):
        self.head_calls.append(kwargs)
        raise self.exceptions.ClientError("missing")


class _Body:
    def __init__(self, text: str) -> None:
        self.text = text

    def read(self) -> bytes:
        return self.text.encode()


def test_multilayer_s3_store_routes_writes_to_expected_buckets(monkeypatch) -> None:
    fake = _FakeS3Client()
    monkeypatch.setattr("compos3d.storage.multibucket_s3.boto3.client", lambda *_a, **_k: fake)

    store = MultiLayerS3Store("bronze-b", "silver-b", "gold-b", prefix="prefix")
    bronze_uri = store.put_json("bronze/run/data.json", {"a": 1})
    silver_uri = store.put_bytes("SILVER/run/blob.bin", b"x")
    gold_uri = store.put_json("x/gold/report.json", {"b": 2})

    assert bronze_uri == "s3://bronze-b/prefix/bronze/run/data.json"
    assert silver_uri == "s3://silver-b/prefix/SILVER/run/blob.bin"
    assert gold_uri == "s3://gold-b/prefix/x/gold/report.json"

    assert fake.put_calls[0]["Bucket"] == "bronze-b"
    assert fake.put_calls[1]["Bucket"] == "silver-b"
    assert fake.put_calls[2]["Bucket"] == "gold-b"


def test_multilayer_s3_store_defaults_ambiguous_paths_to_bronze(monkeypatch) -> None:
    fake = _FakeS3Client()
    monkeypatch.setattr("compos3d.storage.multibucket_s3.boto3.client", lambda *_a, **_k: fake)

    store = MultiLayerS3Store("bronze-b", "silver-b", "gold-b", prefix="")
    store.put_json("run/data.json", {"a": 1})

    assert fake.put_calls[0]["Bucket"] == "bronze-b"


def test_multilayer_s3_store_list_prefix_collects_paginated_results(monkeypatch) -> None:
    fake = _FakeS3Client()
    monkeypatch.setattr("compos3d.storage.multibucket_s3.boto3.client", lambda *_a, **_k: fake)

    store = MultiLayerS3Store("bronze-b", "silver-b", "gold-b", prefix="prefix")
    listed = store.list_prefix("silver/run")

    assert listed == ["silver/run/a.json", "silver/run/b.json"]
    assert fake.list_calls[0]["Bucket"] == "silver-b"
    assert "ContinuationToken" not in fake.list_calls[0]
    assert fake.list_calls[1]["ContinuationToken"] == "token-2"


def test_multilayer_s3_store_exists_returns_false_on_client_error(monkeypatch) -> None:
    fake = _FakeS3Client()
    monkeypatch.setattr("compos3d.storage.multibucket_s3.boto3.client", lambda *_a, **_k: fake)

    store = MultiLayerS3Store("bronze-b", "silver-b", "gold-b", prefix="prefix")
    assert store.exists("gold/run/file.json") is False
    assert fake.head_calls[0]["Bucket"] == "gold-b"
