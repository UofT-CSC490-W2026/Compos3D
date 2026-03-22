"""Local filesystem store mirroring the bronze/silver/gold lake layout."""

from __future__ import annotations

import json
import pathlib
from typing import Any


class LocalStore:
    """Stores artifacts in a local directory tree under a root path.

    The ``bronze/``, ``silver/``, and ``gold/`` subdirectories are created
    automatically under *root*, matching the S3 multi-bucket layout so that
    local and remote runs are structurally identical.
    """

    def __init__(self, root: str | pathlib.Path) -> None:
        self.root = pathlib.Path(root)

    def put_json(self, rel_path: str, obj: Any) -> str:
        p = self.root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, sort_keys=True))
        return str(p)

    def put_bytes(self, rel_path: str, b: bytes, content_type: str = "application/octet-stream") -> str:  # noqa: ARG002
        p = self.root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
        return str(p)

    def read_json(self, rel_path: str) -> Any:
        p = self.root / rel_path
        return json.loads(p.read_text())

    def list_prefix(self, rel_prefix: str) -> list[str]:
        out: list[str] = []
        prefix_path = self.root / rel_prefix
        if prefix_path.is_dir():
            for p in prefix_path.rglob("*"):
                if p.is_file():
                    out.append(str(p.relative_to(self.root)))
        return sorted(out)

    def exists(self, rel_path: str) -> bool:
        return (self.root / rel_path).exists()
