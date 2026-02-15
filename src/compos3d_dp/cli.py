from __future__ import annotations
import typer
from rich import print

from compos3d_dp.config import load_config_from_yaml
from compos3d_dp.storage.local import LocalStore
from compos3d_dp.storage.s3 import S3Store
from compos3d_dp.pipelines.bronze_demo_ingest import run_bronze_demo
from compos3d_dp.pipelines.silver_build import build_silver_from_bronze_local

app = typer.Typer(help="Compos3D Data Platform CLI", no_args_is_help=True)


def _make_store(cfg):
    if cfg.storage_backend == "local":
        return LocalStore(cfg.local_lake_root)

    assert cfg.s3_bucket is not None, "Set s3_bucket in config or COMPOS3D_S3_BUCKET"
    return S3Store(cfg.s3_bucket, cfg.s3_prefix, cfg.aws_region)


@app.command("show-config")
def show_config(
    env: str = typer.Option("dev"),
    config_path: str = typer.Option("config/env.dev.yaml"),
):
    cfg = load_config_from_yaml(config_path, env=env)  # type: ignore
    print(cfg.model_dump())


@app.command("bronze-demo")
def bronze_demo(
    env: str = typer.Option("dev"),
    config_path: str = typer.Option("config/env.dev.yaml"),
):
    cfg = load_config_from_yaml(config_path, env=env)  # type: ignore
    store = _make_store(cfg)
    out = run_bronze_demo(store, cfg.layout.bronze_prefix)
    print("[green]Bronze demo wrote:[/green]", out.scene_json_uri)


@app.command("silver-build")
def silver_build(
    env: str = typer.Option("dev"),
    config_path: str = typer.Option("config/env.dev.yaml"),
    date: str = typer.Option(None, help="YYYY-MM-DD (default: today UTC)"),
):
    cfg = load_config_from_yaml(config_path, env=env)  # type: ignore
    store = _make_store(cfg)

    out = build_silver_from_bronze_local(
        store=store,
        bronze_prefix=cfg.layout.bronze_prefix,
        silver_prefix=cfg.layout.silver_prefix,
        date_yyyy_mm_dd=date,
        config_snapshot=cfg.model_dump(),
    )
    print("[green]Silver built:[/green]")
    print(" scene:", out.scene_dataset_uri)
    print(" scene_object:", out.scene_object_dataset_uri)
    if out.relations_dataset_uri:
        print(" relations:", out.relations_dataset_uri)
    if out.manifest_uri:
        print(" manifest:", out.manifest_uri)
    if out.validation_report_uri:
        print(" validation report:", out.validation_report_uri)


if __name__ == "__main__":
    app()
