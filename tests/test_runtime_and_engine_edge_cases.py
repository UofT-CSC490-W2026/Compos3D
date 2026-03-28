from __future__ import annotations

import importlib
import json
import runpy
import subprocess
import sys
import types
from pathlib import Path
from urllib import error as urllib_error

import pytest
from botocore.exceptions import ClientError

from compos3d.app_config import AppConfig
from compos3d.aws_runtime import (
    CheckpointSyncManager,
    RuntimePaths,
    _IMDSv2,
    _download_s3_prefix,
    _hydrate_secrets,
    _normalize_cli_args,
    _option_value,
    _parse_s3_uri,
    _replace_option,
    _resolve_runtime_paths,
    _upload_directory,
    _write_runtime_context,
    _run_inner_cli,
    _install_shutdown_handlers,
    main as aws_runtime_main,
)
from compos3d.compute.ec2_runner import EC2JobRunner
from compos3d.config import LoggingConfig
from compos3d.hypothesis import engine
from compos3d.hypothesis.loop import HypothesisLoopConfig, SceneHypothesisLoop
from compos3d.hypothesis.training_logging import TrainingLogger, _mean
from compos3d.hypothesis.engine import _filter_hypotheses_for_inference
from compos3d.models import (
    AssetSpec,
    HypothesisRecord,
    SceneProgram,
    TrainingDataset,
    TrainingExample,
)


class _FakeS3Lite:
    def __init__(self) -> None:
        self.files: dict[tuple[str, str], bytes] = {}
        self.uploads: list[tuple[str, str]] = []

    def get_paginator(self, _name: str):
        s3 = self

        class _Paginator:
            def paginate(self, **kwargs):
                bucket = kwargs["Bucket"]
                prefix = kwargs["Prefix"]
                rows = []
                for (bkt, key), _content in sorted(s3.files.items()):
                    if bkt == bucket and key.startswith(prefix):
                        rows.append({"Key": key})
                return [{"Contents": rows}]

        return _Paginator()

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(self.files[(bucket, key)])

    def upload_file(self, filename: str, bucket: str, key: str, ExtraArgs=None) -> None:  # noqa: N803
        self.uploads.append((bucket, key))
        self.files[(bucket, key)] = Path(filename).read_bytes()


def test_aws_runtime_cli_parsing_and_path_helpers(tmp_path: Path) -> None:
    assert _normalize_cli_args(["--", "x"]) == ["x"]
    assert _option_value(["--a"], "--a") is None
    assert _option_value(["--a=1"], "--a") == "1"

    assert _replace_option(["--a"], "--a", "v") == ["--a", "v"]
    assert _replace_option(["--a=1"], "--a", "2") == ["--a=2"]
    assert _replace_option(["--b", "1"], "--a", "2") == ["--b", "1", "--a", "2"]

    with pytest.raises(ValueError):
        _parse_s3_uri("x://bucket/key")
    with pytest.raises(ValueError):
        _parse_s3_uri("s3://bucket")

    cfg = AppConfig(
        env="dev",
        storage_backend="s3",
        s3_bucket_bronze="",
        s3_bucket_silver="silver",
        s3_bucket_gold="gold",
    )
    inf_paths = _resolve_runtime_paths("run-inference", [], cfg)
    assert inf_paths.output_dir is not None
    other_paths = _resolve_runtime_paths("evaluate-predictions", [], cfg)
    assert other_paths.output_dir is None
    train_paths = _resolve_runtime_paths("train-hypotheses", [], cfg)
    assert train_paths.checkpoint_uri is None

    assert (
        _write_runtime_context(
            target_dir=None,
            command="x",
            cli_args=[],
            resolved_inputs={},
            bindings=[],
            checkpoint_uri=None,
            instance_type=None,
            runtime_env="dev",
        )
        is None
    )


def test_aws_runtime_s3_sync_edges_and_imds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    s3 = _FakeS3Lite()
    assert (
        _download_s3_prefix(
            s3_client=s3, source_uri="s3://bucket/prefix", destination_dir=tmp_path
        )
        == 0
    )

    s3.files[("bucket", "prefix/")] = b""
    s3.files[("bucket", "prefix/a.json")] = b'{"x":1}'
    downloaded = _download_s3_prefix(
        s3_client=s3, source_uri="s3://bucket/prefix", destination_dir=tmp_path / "dl"
    )
    assert downloaded == 1
    assert (tmp_path / "dl" / "a.json").exists()

    assert (
        _upload_directory(
            s3_client=s3,
            source_dir=tmp_path / "missing",
            destination_uri="s3://bucket/u",
        )
        == 0
    )
    src = tmp_path / "src"
    (src / "dir").mkdir(parents=True)
    (src / "dir" / "x.txt").write_text("x")
    assert (
        _upload_directory(s3_client=s3, source_dir=src, destination_uri="s3://bucket/u")
        == 1
    )

    class _Resp:
        def __init__(self, text: str):
            self._text = text

        def __enter__(self):
            return self

        def __exit__(self, *args):  # noqa: ANN002, ANN003
            return False

        def read(self):
            return self._text.encode()

    def _ok_urlopen(req, timeout=2):  # noqa: ARG001
        if req.full_url.endswith("/api/token"):
            return _Resp("tok")
        return _Resp("meta")

    monkeypatch.setattr("compos3d.aws_runtime.urllib_request.urlopen", _ok_urlopen)
    imds = _IMDSv2()
    assert imds.get("meta-data/instance-id") == "meta"

    monkeypatch.setattr(
        _IMDSv2,
        "_refresh_token",
        lambda self: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert _IMDSv2().get("meta-data/instance-id") is None

    class _404:
        def __call__(self, req, timeout=2):  # noqa: ARG002
            raise urllib_error.HTTPError(req.full_url, 404, "x", None, None)

    imds_404 = _IMDSv2()
    imds_404._token = "tok"
    imds_404._token_deadline = 10**12
    monkeypatch.setattr("compos3d.aws_runtime.urllib_request.urlopen", _404())
    assert imds_404.get("meta-data/spot/instance-action") is None

    class _boom:
        def __call__(self, req, timeout=2):  # noqa: ARG002
            raise RuntimeError("net")

    monkeypatch.setattr("compos3d.aws_runtime.urllib_request.urlopen", _boom())
    assert imds_404.get("meta-data/instance-id") is None

    class _500:
        def __call__(self, req, timeout=2):  # noqa: ARG002
            raise urllib_error.HTTPError(req.full_url, 500, "x", None, None)

    monkeypatch.setattr("compos3d.aws_runtime.urllib_request.urlopen", _500())
    with pytest.raises(urllib_error.HTTPError):
        imds_404.get("meta-data/instance-id")


def test_checkpoint_manager_handlers_and_main_resume_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cm = CheckpointSyncManager(
        s3_client=_FakeS3Lite(),
        local_dir=None,
        remote_uri=None,
        interval_seconds=1,
    )
    cm.start()
    assert cm.sync_once() == 0
    assert cm.restore() == 0

    calls: list[str] = []

    class _Stop:
        def __init__(self) -> None:
            self.set_called = False
            self.wait_calls = 0

        def wait(self, _interval: int) -> bool:
            self.wait_calls += 1
            return self.set_called

        def set(self) -> None:
            self.set_called = True

    cm2 = CheckpointSyncManager(
        s3_client=_FakeS3Lite(),
        local_dir=tmp_path,
        remote_uri="s3://bucket/pfx",
        interval_seconds=15,
    )
    stop = _Stop()
    cm2._stop = stop  # type: ignore[assignment]
    monkeypatch.setattr(cm2, "sync_once", lambda: calls.append("sync") or 1)
    monkeypatch.setattr(cm2, "_spot_interruption_pending", lambda: True)
    cm2._run()  # noqa: SLF001
    assert calls == ["sync", "sync"]

    cm3 = CheckpointSyncManager(
        s3_client=_FakeS3Lite(),
        local_dir=tmp_path,
        remote_uri="s3://bucket/pfx",
        interval_seconds=15,
    )
    cm3._metadata = types.SimpleNamespace(  # type: ignore[assignment]
        get=lambda _path: (_ for _ in ()).throw(RuntimeError("x"))
    )
    assert cm3._spot_interruption_pending() is False  # noqa: SLF001
    cm3._metadata = types.SimpleNamespace(get=lambda _path: "terminate")  # type: ignore[assignment]
    assert cm3._spot_interruption_pending() is True  # noqa: SLF001

    handlers: dict[int, object] = {}
    monkeypatch.setattr(
        "compos3d.aws_runtime.signal.signal",
        lambda sig, fn: handlers.setdefault(int(sig), fn),
    )
    tracker = types.SimpleNamespace(sync=0, stop=0)
    mgr = types.SimpleNamespace(
        sync_once=lambda: setattr(tracker, "sync", tracker.sync + 1),
        stop=lambda: setattr(tracker, "stop", tracker.stop + 1),
    )
    _install_shutdown_handlers(mgr)  # type: ignore[arg-type]
    with pytest.raises(SystemExit) as exc:
        handlers[int(sys.modules["signal"].SIGTERM)](15, None)  # type: ignore[index]
    assert exc.value.code == 143
    assert tracker.sync == 1 and tracker.stop == 1

    fake_cfg = AppConfig(
        env="dev",
        storage_backend="s3",
        s3_bucket_bronze="bronze",
        s3_bucket_silver="silver",
        s3_bucket_gold="gold",
    )
    monkeypatch.setattr("compos3d.aws_runtime.load_app_config", lambda _env: fake_cfg)
    monkeypatch.setattr(
        "compos3d.aws_runtime.boto3.client",
        lambda name, region_name=None: (
            _FakeS3Lite() if name == "s3" else types.SimpleNamespace()
        ),
    )
    monkeypatch.setattr(
        "compos3d.aws_runtime._resolve_runtime_paths",
        lambda *_a, **_k: RuntimePaths(
            output_dir=tmp_path, run_dir=tmp_path, checkpoint_uri="s3://bucket/pfx"
        ),
    )
    monkeypatch.setattr(
        "compos3d.aws_runtime._hydrate_remote_inputs",
        lambda args, **_k: (args, {}),
    )

    class _Sync:
        restored = 0

        def __init__(self, **kwargs):  # noqa: ARG002
            pass

        def restore(self):
            self.restored += 1
            return 3

        def start(self):
            return None

        def sync_once(self):
            return 0

        def stop(self):
            return None

    monkeypatch.setattr("compos3d.aws_runtime.CheckpointSyncManager", _Sync)
    monkeypatch.setattr("compos3d.aws_runtime._hydrate_secrets", lambda *_a, **_k: [])
    monkeypatch.setattr(
        "compos3d.aws_runtime._write_runtime_context", lambda **_k: None
    )
    monkeypatch.setattr(
        "compos3d.aws_runtime._install_shutdown_handlers", lambda _sm: None
    )
    monkeypatch.setattr("compos3d.aws_runtime._run_inner_cli", lambda **_k: 0)

    assert (
        aws_runtime_main(
            [
                "--runtime-env",
                "dev",
                "--work-dir",
                str(tmp_path / "work"),
                "train-hypotheses",
                "--resume",
            ]
        )
        == 0
    )
    assert "Restored 3 checkpoint artifacts" in capsys.readouterr().out


def test_aws_runtime_secret_hydration_and_inner_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_cfg = AppConfig(
        env="dev",
        storage_backend="s3",
        s3_bucket_bronze="b",
        s3_bucket_silver="s",
        s3_bucket_gold="g",
        aws_secret_env_map={"X": "secret-x"},
    )

    class _BadSecrets:
        def get_secret_value(self, SecretId: str):  # noqa: N803
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetSecretValue")

    with pytest.raises(ClientError):
        _hydrate_secrets(bad_cfg, secrets_client=_BadSecrets())

    cfg = bad_cfg.model_copy(
        update={"aws_secret_env_map": {"A": "secret-a", "B": "secret-b"}}
    )

    class _MixedSecrets:
        def get_secret_value(self, SecretId: str):  # noqa: N803
            if SecretId == "secret-a":
                return {"SecretBinary": b"abc", "VersionId": "v1"}
            return {"VersionId": "v2"}

    binds = _hydrate_secrets(cfg, secrets_client=_MixedSecrets())
    assert [b.env_var for b in binds] == ["A"]

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        "compos3d.aws_runtime.subprocess.run",
        lambda cmd, cwd, check=False: (
            seen.update(cmd=cmd, cwd=cwd) or types.SimpleNamespace(returncode=7)
        ),
    )
    code = _run_inner_cli(command="train-hypotheses", cli_args=[], instance_type="g5")
    assert code == 7
    assert "--compute-platform" in seen["cmd"] and "aws_ec2" in seen["cmd"]
    assert "--instance-type" in seen["cmd"] and "g5" in seen["cmd"]

    old_argv = sys.argv[:]
    try:
        sys.argv = ["compos3d.aws_runtime", "--help"]
        with pytest.raises(SystemExit):
            runpy.run_module("compos3d.aws_runtime", run_name="__main__")
    finally:
        sys.argv = old_argv


def test_ec2_runner_remaining_simple_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    class _EC2:
        def describe_images(self, **kwargs):  # noqa: ARG002
            return {
                "Images": [
                    {
                        "ImageId": "ami-123",
                        "CreationDate": "2025-01-01T00:00:00.000Z",
                        "Name": "dlami",
                    }
                ]
            }

        def describe_subnets(self, **kwargs):  # noqa: ARG002
            return {"Subnets": []}

        def describe_security_groups(self, **kwargs):  # noqa: ARG002
            return {"SecurityGroups": []}

    class _SSM:
        def get_parameter(self, **kwargs):  # noqa: ARG002
            raise RuntimeError("no")

    class _ECR:
        def describe_repositories(self, **kwargs):  # noqa: ARG002
            return {"repositories": []}

    monkeypatch.setattr(
        "compos3d.compute.ec2_runner.boto3.client",
        lambda name, region_name=None: (
            _EC2() if name == "ec2" else _SSM() if name == "ssm" else _ECR()
        ),
    )
    cfg = AppConfig(
        env="dev",
        storage_backend="local",
        local_lake_root=".",
        aws_region="us-east-1",
        ec2_instance_type="",
        ec2_iam_instance_profile="p",
        ecr_repository_url="",
    )
    runner = EC2JobRunner(cfg)
    assert runner._resolve_ami() == "ami-123"  # noqa: SLF001
    assert runner._discover_subnet_id() is None  # noqa: SLF001
    assert runner._discover_security_group_id() is None  # noqa: SLF001
    assert runner._requires_gpu(None) is False  # noqa: SLF001
    with pytest.raises(RuntimeError):
        runner._resolve_ecr_repository_url()  # noqa: SLF001
    runner.cfg = runner.cfg.model_copy(
        update={"ecr_repository_url": "123.dkr.ecr.us-east-1.amazonaws.com/repo"}
    )
    assert "repo" in runner._resolve_ecr_repository_url()  # noqa: SLF001


def test_engine_and_loop_remaining_branches(tmp_path: Path) -> None:
    class _Store:
        def __init__(self):
            self.json: list[tuple[str, dict]] = []

        def put_json(self, key, payload):
            self.json.append((key, payload))
            return key

        def put_bytes(self, key, payload, content_type="application/octet-stream"):  # noqa: ARG002
            return key

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "runtime_context.json").write_text(json.dumps({"ok": True}))
    store = _Store()
    engine._mirror_training_to_lake(  # noqa: SLF001
        store=store,
        run_id="r1",
        run_dir=run_dir,
        experiment_name="exp",
        training_result={},
        experiment_config={},
    )
    assert any(key.endswith("/runtime_context.json") for key, _ in store.json)

    inf_dir = tmp_path / "inf"
    inf_dir.mkdir()
    (inf_dir / "runtime_context.json").write_text(json.dumps({"ok": True}))
    engine._mirror_inference_to_lake(  # noqa: SLF001
        store=store,
        run_id="r2",
        output_dir=inf_dir,
        inference_result={},
    )
    assert len([k for k, _ in store.json if k.endswith("/runtime_context.json")]) >= 2

    assert (
        _filter_hypotheses_for_inference(  # noqa: SLF001
            llm=object(), records=[], prompt="p", room_type="dining_room"
        )
        == []
    )

    records = [
        HypothesisRecord(hypothesis_id="h1", text="chair", room_type="dining_room"),
    ]
    candidate_programs = [
        SceneProgram(
            prompt="p",
            room_type="dining_room",
            assets=[
                AssetSpec(asset_type="chair", count=1, placement="p", rationale="r")
            ],
        )
    ]
    program_break = engine._weighted_vote_scene_program(  # noqa: SLF001
        prompt="chair table lamp room",
        room_type="dining_room",
        records=records,
        candidate_programs=candidate_programs,
    )
    assert program_break.assets

    cfg = engine._experiment_config_from_args(  # noqa: SLF001
        config_path=None,
        llm_provider="mock",
        num_init_examples_per_room=1,
        init_hypotheses_per_room=1,
        top_k=1,
        alpha=0.1,
        max_num_hypotheses_per_room=2,
        num_wrong_scale=0.5,
        update_batch_size=1,
        num_hypotheses_to_update=1,
        update_hypotheses_per_batch=1,
        only_best_hypothesis=False,
        num_epochs=1,
        success_threshold=0.5,
        save_every_n_examples=1,
        selection_strategy="ucb",
        use_repair=True,
        baseline_mode=None,
        seed=1,
        wandb_project="p",
        wandb_entity="e",
        wandb_mode="offline",
        wandb_run_name="r",
        wandb_tags=["t"],
    )
    assert cfg.logging.enable_wandb is True

    ds = TrainingDataset(
        dataset_id="d",
        examples=[
            TrainingExample(
                example_id="e1",
                room_type="dining_room",
                prompt="p",
                required_assets=["chair"],
            )
        ],
    )

    class _LLM:
        def generate_hypotheses(self, *a, **k):  # noqa: ARG002
            return ["x"]

    loop = SceneHypothesisLoop(
        dataset=ds,
        llm=_LLM(),
        critic=types.SimpleNamespace(),
        run_dir=tmp_path / "loop",
        llm_provider="mock",
        config=HypothesisLoopConfig(num_epochs=1),
        experiment_config={},
    )
    with pytest.raises(ValueError):
        loop._restore_from_state({"dataset_id": "other"})  # noqa: SLF001

    loop._make_record = lambda **_k: None  # type: ignore[method-assign] # noqa: SLF001
    loop._initialize_bank()  # noqa: SLF001
    assert loop.bank == []


def test_training_logging_remaining_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    logger = TrainingLogger(
        config=LoggingConfig(enable_wandb=True),
        run_dir=tmp_path,
        experiment_name="exp",
        config_payload={},
    )
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    import builtins

    real_import = builtins.__import__

    def _import(name, *args, **kwargs):  # noqa: ANN001
        if name == "wandb":
            raise ModuleNotFoundError("wandb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    with pytest.raises(RuntimeError, match="not installed"):
        logger.start()

    logger2 = TrainingLogger(
        config=LoggingConfig(enable_wandb=False, log_initial_media=False),
        run_dir=tmp_path / "r2",
        experiment_name="exp",
        config_payload={},
    )
    logger2.start()
    logger2.log_initial_scene(
        room_type="dining_room",
        example_id="e1",
        prompt="p",
        hypothesis_text="h",
        score=0.5,
        image_paths=[],
    )
    logger2._run = types.SimpleNamespace()  # noqa: SLF001
    logger2._wandb = types.SimpleNamespace()  # noqa: SLF001
    assert (
        logger2._build_scene_media_payload(  # noqa: SLF001
            prefix="p",
            slug="s",
            image_paths=[],
            video_path=None,
            caption="c",
            step=1,
            metadata={},
        )
        == {}
    )
    logger2._run = types.SimpleNamespace(log=lambda *_a, **_k: None)  # noqa: SLF001
    logger2._define_metrics()  # noqa: SLF001

    logger2._wandb = None  # noqa: SLF001
    assert logger2._make_table(columns=["a"]) is None  # noqa: SLF001

    image = tmp_path / "one.png"
    image.write_bytes(b"not-an-image")
    logger2._media_dir = tmp_path / "r2" / "wandb_media"  # noqa: SLF001
    logger2._media_dir.mkdir(parents=True, exist_ok=True)  # noqa: SLF001
    preview = logger2._preview_path("slug")  # noqa: SLF001
    logger2._ensure_preview(preview, [image])  # noqa: SLF001
    assert not preview.exists()

    gif = logger2._animation_path("slug2")  # noqa: SLF001
    logger2._ensure_animation(gif, [image, image])  # noqa: SLF001
    assert not gif.exists()

    assert logger2._load_frames([image]) == []  # noqa: SLF001

    vid = tmp_path / "scene" / "turntable.mp4"
    vid.parent.mkdir(parents=True, exist_ok=True)
    vid.write_bytes(b"v")
    assert logger2._media_video_path(slug="x", raw_video_path=str(vid)) == vid  # noqa: SLF001
    assert logger2._infer_render_dir(image_paths=[], video_path=vid) == vid.parent  # noqa: SLF001
    assert logger2._infer_render_dir(image_paths=[], video_path=None) is None  # noqa: SLF001
    a = tmp_path / "a.png"
    a.write_bytes(b"a")
    assert logger2._infer_render_dir(image_paths=[a], video_path=None) == a.parent  # noqa: SLF001
    assert _mean([]) == 0.0


def test_procedural_runner_import_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import compos3d.procedural.runner as runner

    real_exists = Path.exists
    real_iterdir = Path.iterdir

    def _fake_exists(path: Path) -> bool:
        if "singleroom.gin" in str(path):
            return False
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", _fake_exists)
    monkeypatch.setattr(Path, "iterdir", lambda self: real_iterdir(self))
    reloaded = importlib.reload(runner)
    assert reloaded.GIN_CONFIG_ROOT == reloaded.PROJECT_ROOT / "infinigen"
    importlib.reload(runner)
