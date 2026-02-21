from __future__ import annotations
from pydantic import BaseModel
from typing import Literal


class RenderRecord(BaseModel):
    schema_version: Literal["v1"] = "v1"
    scene_id: str
    camera_id: str
    frame_idx: int
    pass_type: Literal["rgb", "depth", "seg", "normal"]
    uri: str
    width: int
    height: int
    checksum_sha256: str
