"""Failure-path and edge-case tests for LocalStore behavior.

Read/write failures, JSON decode errors, overwrite semantics, and prefix listing.

Edge cases covered:
- Missing file reads.
- Invalid JSON decode failures.
- Overwrite semantics for repeated writes to same key.
- Nested prefix listing and missing-prefix behavior.

Expected outcomes:
- Appropriate exceptions are raised for invalid reads.
- Later writes replace earlier values at same relative path.
- Prefix listing is stable, recursive, and returns sorted relative keys.
"""

from __future__ import annotations

import json

import pytest

from compos3d.storage.local import LocalStore


def test_read_json_missing_file_raises(tmp_path) -> None:
    store = LocalStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.read_json("missing/path.json")


def test_read_json_invalid_json_raises(tmp_path) -> None:
    store = LocalStore(tmp_path)
    p = tmp_path / "bronze" / "bad.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{bad-json")
    with pytest.raises(json.JSONDecodeError):
        store.read_json("bronze/bad.json")


def test_put_json_overwrite_semantics(tmp_path) -> None:
    store = LocalStore(tmp_path)
    rel = "silver/item.json"
    store.put_json(rel, {"v": 1})
    store.put_json(rel, {"v": 2})
    assert store.read_json(rel) == {"v": 2}


def test_list_prefix_nested_paths_and_missing_prefix(tmp_path) -> None:
    store = LocalStore(tmp_path)
    store.put_json("gold/a/x.json", {"a": 1})
    store.put_json("gold/a/nested/y.json", {"b": 2})
    listed = [item.replace("\\", "/") for item in store.list_prefix("gold/a")]
    assert listed == ["gold/a/nested/y.json", "gold/a/x.json"]
    assert store.list_prefix("gold/does_not_exist") == []
