"""Storage abstraction for the compos3d data lake (local or S3)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from compos3d.storage.local import LocalStore
from compos3d.storage.multibucket_s3 import MultiLayerS3Store

if TYPE_CHECKING:
    from compos3d.app_config import AppConfig

AnyStore = Union[LocalStore, MultiLayerS3Store]


def get_store(app_config: "AppConfig") -> AnyStore:
    """Return the appropriate store for the given app config.

    - ``storage_backend=local``  → :class:`LocalStore` under ``local_lake_root``
    - ``storage_backend=s3``     → :class:`MultiLayerS3Store` with separate bronze/silver/gold buckets
    """
    if app_config.storage_backend == "local":
        return LocalStore(root=app_config.local_lake_root)

    # S3 path: all three bucket names must be configured.
    missing = [
        name
        for name, val in [
            ("s3_bucket_bronze", app_config.s3_bucket_bronze),
            ("s3_bucket_silver", app_config.s3_bucket_silver),
            ("s3_bucket_gold", app_config.s3_bucket_gold),
        ]
        if not val
    ]
    if missing:
        raise ValueError(
            f"storage_backend=s3 requires these config fields to be set: {missing}"
        )

    return MultiLayerS3Store(
        bucket_bronze=app_config.s3_bucket_bronze,  # type: ignore[arg-type]
        bucket_silver=app_config.s3_bucket_silver,  # type: ignore[arg-type]
        bucket_gold=app_config.s3_bucket_gold,  # type: ignore[arg-type]
        prefix=app_config.s3_prefix,
        region=app_config.aws_region,
    )


__all__ = ["LocalStore", "MultiLayerS3Store", "AnyStore", "get_store"]
