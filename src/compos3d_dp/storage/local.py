from __future__ import annotations
import json
import pathlib
from typing import Any


class LocalStore:
    def __init__(self, root: str):
        self.root = pathlib.Path(root)

    def put_json(self, rel_path: str, obj: Any) -> str:
        p = self.root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, sort_keys=True))
        return str(p)

    def put_bytes(self, rel_path: str, b: bytes) -> str:
        p = self.root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
        return str(p)

    def read_json(self, rel_path: str) -> Any:
        p = self.root / rel_path
        return json.loads(p.read_text())

    def list_glob(self, rel_glob: str) -> list[str]:
        # returns rel paths under store root
        out: list[str] = []
        for p in (self.root).glob(rel_glob):
            if p.is_file():
                out.append(str(p.relative_to(self.root)))
        return sorted(out)
