from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import uuid

from compos3d_dp.schemas.scene import SceneRecord, SceneObject
from compos3d_dp.storage.paths import utc_date_parts


@dataclass
class BronzeOutputs:
    scene_json_uri: str


def run_bronze_demo(store, bronze_prefix: str) -> BronzeOutputs:
    # deterministic-ish example payload (replace with real Infinigen/VIGA artifacts later)
    seed = 123
    scene_id = hashlib.sha256(f"infinigen:{seed}".encode()).hexdigest()[:16]
    split = "train"

    scene = SceneRecord(
        scene_id=scene_id,
        seed=seed,
        split=split,
        objects=[
            SceneObject(
                object_id="obj_0",
                category="chair",
                asset_id="asset_dummy_chair",
                position_xyz=[0.0, 0.0, 0.0],
                rotation_xyz=[0.0, 0.0, 0.0],
                scale_xyz=[1.0, 1.0, 1.0],
            )
        ],
    )

    y, m, d = utc_date_parts(datetime.now(timezone.utc))
    run_id = uuid.uuid4().hex[:10]
    rel = (
        f"{bronze_prefix}/scenes/date={y}-{m}-{d}/run_id={run_id}/scene_{scene_id}.json"
    )
    uri = store.put_json(rel, scene.model_dump())
    return BronzeOutputs(scene_json_uri=uri)
