"""Targeted tests to close remaining coverage gaps.

These tests intentionally exercise small, otherwise-hard-to-hit branches such as:
- OS-specific path normalization.
- CLI plumbing branches that are hard to hit in E2E.
- AWS runner logic using faked boto3 clients (no real AWS calls).
- Scene/critic normalization helpers for malformed model payloads.
"""

from __future__ import annotations

import json
import subprocess
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

import compos3d.catalog as catalog
import compos3d.cli as cli
import compos3d.config as config_mod
import compos3d.procedural.runner as runner_mod
import compos3d.procedural.service as proc_service
import compos3d.schemas.manifest as manifest_mod
import compos3d.storage as storage_mod
from compos3d._stages import StagePendingError, stage_pending
from compos3d.app_config import AppConfig, load_app_config
from compos3d.compute.ec2_runner import EC2JobRunner, EC2JobSpec
from compos3d.evaluation.critic import (
    BedrockVLMCritic,
    CriticConfig,
    CriticUnavailableError,
    HeuristicSceneCritic,
    aggregate_prediction_scores,
    build_scene_critic,
)
from compos3d.hypothesis import engine as hyp_engine
from compos3d.hypothesis.loop import HypothesisLoopConfig, SceneHypothesisLoop
from compos3d.models import (
    AssetSpec,
    ConstraintSpec,
    HypothesisRecord,
    SceneProgram,
    TrainingDataset,
    TrainingExample,
)
from compos3d.llm import scene_llm as scene_llm_mod


def test_stage_pending_raises_stage_pending_error() -> None:
    with pytest.raises(StagePendingError, match="Stage 4"):
        stage_pending("Reference feature extraction", stage=4)


def test_load_app_config_env_none_uses_env_var(monkeypatch) -> None:
    monkeypatch.setenv("COMPOS3D_ENV", "local")
    cfg = load_app_config(None)
    assert cfg.env == "local"


def test_supported_room_types_returns_keys_tuple() -> None:
    rooms = catalog.supported_room_types()
    assert set(rooms) == set(catalog.SUPPORTED_ASSETS_BY_ROOM.keys())


def test_load_experiment_config_vlm_defaults_provider_to_generator(tmp_path: Path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text(
        json.dumps(
            {
                "generator": {"provider": "bedrock", "model_id": "m"},
                "critic": {"mode": "vlm", "provider": "", "model_id": "v"},
                "training": {},
                "render": {},
            }
        )
    )
    cfg = config_mod.load_experiment_config(p)
    assert cfg.critic.provider == "bedrock"


def test_cli_reference_generate_splits_tasks(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def _fake_generate_reference_scene(req):
        captured["tasks"] = req.tasks
        captured["minimal"] = req.minimal
        return {"ok": True}

    monkeypatch.setattr(cli, "generate_reference_scene", _fake_generate_reference_scene)
    r = CliRunner()
    res = r.invoke(
        cli.app,
        [
            "reference-generate",
            "--room-type",
            "dining_room",
            "--output-dir",
            str(tmp_path),
            "--tasks",
            "coarse populate",
            "--minimal",
        ],
    )
    assert res.exit_code == 0
    assert captured["tasks"] == ("coarse", "populate")
    assert captured["minimal"] is True


def test_cli_build_scene_and_feature_extract_routes(monkeypatch, tmp_path: Path) -> None:
    seen = {"build": None, "extract": None}

    def _fake_build_scene(req):
        seen["build"] = req
        return {"rendered_views": []}

    def _fake_feature_extract(*, input_path: Path, output_dir: Path) -> None:
        seen["extract"] = (input_path, output_dir)

    monkeypatch.setattr(cli, "build_scene", _fake_build_scene)
    monkeypatch.setattr(cli, "feature_extract", _fake_feature_extract)
    runner = CliRunner()

    sp = tmp_path / "scene_program.json"
    sp.write_text("{}")
    res1 = runner.invoke(
        cli.app,
        [
            "build-scene",
            "--scene-program",
            str(sp),
            "--output-dir",
            str(tmp_path / "out"),
            "--no-video",
            "--save-blend",
        ],
    )
    assert res1.exit_code == 0
    assert seen["build"] is not None
    assert seen["build"].no_video is True
    assert seen["build"].save_blend is True

    res2 = runner.invoke(
        cli.app,
        ["feature-extract", "--input-path", str(tmp_path), "--output-dir", str(tmp_path / "feat")],
    )
    assert res2.exit_code == 0
    assert seen["extract"] == (tmp_path, tmp_path / "feat")


def test_cli_launch_aws_instance_type_override_and_main_guard(monkeypatch) -> None:
    # Cover cli.py: instance_type override branch (line 344) and __main__ guard (line 367).
    fake_cfg = AppConfig(
        env="dev",
        storage_backend="local",
        local_lake_root=".",
        aws_region="us-east-1",
        ec2_instance_type="t3.small",
        ec2_iam_instance_profile="profile",
        ec2_spot=False,
    )

    import compos3d.app_config as app_config_mod
    import compos3d.compute.ec2_runner as ec2_runner_mod

    monkeypatch.setattr(app_config_mod, "load_app_config", lambda _env=None: fake_cfg)

    class _FakeRunner:
        def __init__(self, app_config):  # noqa: ARG002
            pass

        def launch(self, _spec):  # noqa: ARG002
            return ("i-1", {"ok": True})

        def wait(self, _instance_id):  # noqa: ARG002
            return "terminated"

    monkeypatch.setattr(ec2_runner_mod, "EC2JobRunner", _FakeRunner)

    r = CliRunner()
    res = r.invoke(
        cli.app,
        [
            "launch-aws",
            "train-hypotheses",
            "--env",
            "dev",
            "--instance-type",
            "m5.large",
        ],
    )
    assert res.exit_code == 0

    import runpy
    import sys

    old_argv = sys.argv[:]
    try:
        sys.argv = ["compos3d.cli", "--help"]
        with pytest.raises(SystemExit):
            runpy.run_module("compos3d.cli", run_name="__main__")
    finally:
        sys.argv = old_argv


def test_storage_get_store_s3_success() -> None:
    cfg = AppConfig(
        env="dev",
        storage_backend="s3",
        local_lake_root=".",
        s3_bucket_bronze="b1",
        s3_bucket_silver="b2",
        s3_bucket_gold="b3",
        s3_prefix="pfx",
        aws_region="us-east-1",
    )
    store = storage_mod.get_store(cfg)
    assert isinstance(store, storage_mod.MultiLayerS3Store)


def test_multilayer_s3_store_exists_false_on_client_error(monkeypatch) -> None:
    from compos3d.storage.multibucket_s3 import MultiLayerS3Store

    store = MultiLayerS3Store("b1", "b2", "b3", prefix="", region="us-east-1")

    class _Exc(Exception):
        pass

    class _FakeS3:
        class exceptions:
            ClientError = _Exc

        def head_object(self, **_kwargs):
            raise _Exc("nope")

    store.s3 = _FakeS3()
    assert store.exists("bronze/x.json") is False


def test_manifest_git_info_remote_url_missing_and_packages_unknown(monkeypatch) -> None:
    calls = {"i": 0}

    def _fake_run(cmd, capture_output, text, check, timeout):  # noqa: ARG001
        # commit, branch, dirty, remote_url
        outputs = ["sha", "main", "", ""]
        out = outputs[calls["i"]]
        calls["i"] += 1
        if calls["i"] == 4:
            raise subprocess.CalledProcessError(1, ["git", "config"])
        return types.SimpleNamespace(stdout=out)

    monkeypatch.setattr(manifest_mod.subprocess, "run", _fake_run)

    import importlib.metadata as importlib_metadata

    def _fake_version(_pkg: str) -> str:
        raise importlib_metadata.PackageNotFoundError()

    monkeypatch.setattr(importlib_metadata, "version", _fake_version)

    m = manifest_mod.create_manifest(run_id="r1", run_type="training", config_snapshot={})
    assert m.git_info is not None
    assert m.git_info.remote_url is None
    assert all(v == "unknown" for v in m.package_versions.values())


def test_manifest_git_info_total_failure_returns_none(monkeypatch) -> None:
    monkeypatch.setattr(
        manifest_mod.subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    m = manifest_mod.create_manifest(run_id="r2", run_type="evaluate")
    assert m.git_info is None


def test_procedural_service_backend_smoke_generate_reference_build_scene_and_feature_extract(
    monkeypatch, tmp_path: Path
) -> None:
    # backend smoke: cover save_blend flag (line 86) and manifest write path.
    def _fake_run_script(script, args, **kwargs):  # noqa: ARG001
        return {"exit_code": 0, "elapsed_seconds": 0.01, "cmd": "ok"}

    monkeypatch.setattr(runner_mod, "run_script", _fake_run_script)
    # Create the expected output PNG so run_backend_smoke doesn't raise.
    req = proc_service.BackendSmokeRequest(
        asset_name="chair",
        output_dir=tmp_path,
        seed=1,
        save_blend=True,
    )
    out_dir = tmp_path / "chair_seed1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chair_seed1.png").write_bytes(b"x")
    manifest = proc_service.run_backend_smoke(req)
    assert (out_dir / "smoke_manifest.json").exists()
    assert manifest["render_path"].endswith(".png")

    # reference generate: cover minimal branch and manifest file write.
    ref_req = proc_service.ReferenceGenerationRequest(
        room_type="dining_room",
        output_dir=tmp_path,
        seed=2,
        tasks=("coarse", "populate"),
        minimal=True,
    )
    ref = proc_service.generate_reference_scene(ref_req)
    assert (tmp_path / "ref_manifest.json").exists()
    assert ref["tasks"] == ["coarse", "populate"]

    # build_scene: cover no_video/save_blend flags + manifest_path exists branch.
    sp = tmp_path / "scene_program.json"
    sp.write_text("{}")
    scene_out = tmp_path / "scene_out"
    scene_out.mkdir()
    (scene_out / "build_manifest.json").write_text(json.dumps({"rendered_views": []}))
    build_req = proc_service.BuildSceneRequest(
        scene_program_path=sp,
        output_dir=scene_out,
        no_video=True,
        save_blend=True,
    )
    got = proc_service.build_scene(build_req)
    assert got["rendered_views"] == []

    # build_scene: cover manifest_path missing branch in service (src/compos3d/procedural/service.py:220)
    fresh_out = tmp_path / "scene_out2"
    fresh_out.mkdir()
    build_req2 = proc_service.BuildSceneRequest(
        scene_program_path=sp,
        output_dir=fresh_out,
        no_video=False,
        save_blend=False,
    )
    got2 = proc_service.build_scene(build_req2)
    assert got2["stage"] == "build_scene"

    # feature_extract is stage-pending (lines 267-268)
    with pytest.raises(StagePendingError):
        proc_service.feature_extract(input_path=tmp_path, output_dir=tmp_path / "o")


def test_runner_find_gin_config_root_from_sibling(tmp_path: Path, monkeypatch) -> None:
    # Cover procedural/runner.py: sibling detection + return path (lines 60, 66, 87).
    root = tmp_path / "proj"
    root.mkdir()
    sibling = tmp_path / "sibling_infinigen"
    (sibling / "infinigen_examples" / "configs_indoor").mkdir(parents=True)
    (sibling / "infinigen_examples" / "configs_indoor" / "singleroom.gin").write_text("x")

    monkeypatch.setattr(runner_mod, "PROJECT_ROOT", root)
    # Make parent.iterdir() yield the sibling.
    monkeypatch.setattr(
        Path,
        "iterdir",
        lambda self: [sibling] if self == tmp_path else [],  # noqa: ARG005
    )
    found = runner_mod._find_gin_config_root()  # noqa: SLF001
    assert found == sibling

    # cover runner._build_env extra_pythonpath append (src/compos3d/procedural/runner.py:87)
    monkeypatch.setenv("PYTHONPATH", "already")
    env = runner_mod._build_env([tmp_path])  # noqa: SLF001
    assert "already" in env["PYTHONPATH"]
    assert str(tmp_path) in env["PYTHONPATH"]


def test_critic_remaining_branches_and_empty_aggregate(tmp_path: Path) -> None:
    assert aggregate_prediction_scores([]).num_predictions == 0
    assert scene_llm_mod.StructuredOutputError  # sanity: imported and used elsewhere

    # _normalize_notes branches
    from compos3d.evaluation import critic as critic_mod

    assert critic_mod._normalize_notes("  hi ") == ["hi"]  # noqa: SLF001
    assert critic_mod._normalize_notes(123) == []  # noqa: SLF001
    assert critic_mod._normalize_notes(["a", " ", 5]) == ["a", "5"]  # noqa: SLF001

    # _image_format branches
    assert BedrockVLMCritic._image_format(Path("x.jpg")) == "jpeg"  # noqa: SLF001
    assert BedrockVLMCritic._image_format(Path("x.webp")) == "webp"  # noqa: SLF001
    assert BedrockVLMCritic._image_format(Path("x.tiff")) == "png"  # noqa: SLF001

    # _build_prompt reference block branch
    sp = SceneProgram(
        prompt="p",
        room_type="dining_room",
        assets=[AssetSpec(asset_type="chair", count=1, placement="x", rationale="y")],
        constraints=[ConstraintSpec(text="c")],
    )
    ex = TrainingExample(
        example_id="e",
        room_type="dining_room",
        prompt="p",
        required_assets=["chair"],
    )
    prompt = BedrockVLMCritic._build_prompt(scene_program=sp, reference_example=ex)  # noqa: SLF001
    assert "Reference room_type" in prompt

    # build_scene_critic error branches + vlm missing images branch (line 212)
    with pytest.raises(ValueError, match="Unsupported critic mode"):
        build_scene_critic(CriticConfig(mode="weird"))
    with pytest.raises(ValueError, match="Mock VLM critic"):
        build_scene_critic(CriticConfig(mode="vlm", provider="mock"))
    with pytest.raises(ValueError, match="Unsupported critic provider"):
        build_scene_critic(CriticConfig(mode="vlm", provider="nope"))
    assert isinstance(
        build_scene_critic(CriticConfig(mode="vlm", provider="bedrock", model_id="m")),
        BedrockVLMCritic,
    )
    vlm = BedrockVLMCritic(CriticConfig(mode="vlm", provider="bedrock", model_id="m"))
    with pytest.raises(CriticUnavailableError, match="requires render images"):
        vlm.evaluate(scene_program=sp, image_paths=[])


def test_hypothesis_loop_render_success_and_only_best_hypothesis(tmp_path: Path) -> None:
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
            return ["h"]

        def generate_scene_program(self, *a, **k):  # noqa: ARG002
            return SceneProgram(prompt="p", room_type="dining_room")

    class _Critic:
        def evaluate(self, **_k):
            return types.SimpleNamespace(
                validity=1.0,
                prompt_adherence=1.0,
                asset_precision=1.0,
                asset_recall=1.0,
                room_match=1.0,
                overall=1.0,
                notes=[],
                critic_mode="heuristic",
                used_image_paths=[],
                model_dump=lambda *a, **k: {},  # noqa: ARG002
            )

    img = tmp_path / "view_front.png"
    img.write_bytes(b"x")

    def _renderer(_sp, _dir: Path):
        return [img]

    cfg = HypothesisLoopConfig(
        only_best_hypothesis=True,
        update_batch_size=1,
        num_hypotheses_to_update=1,
        update_hypotheses_per_batch=2,
    )
    loop = SceneHypothesisLoop(
        dataset=ds,
        llm=_LLM(),
        critic=_Critic(),
        run_dir=tmp_path,
        llm_provider="mock",
        config=cfg,
        experiment_config={},
        config_path=None,
        renderer=_renderer,
    )

    # cover _render success print lines 328-329
    assert loop._render(SceneProgram(prompt="p", room_type="dining_room"), "t") == [img]  # noqa: SLF001

    # cover only_best_hypothesis branch line 492 by monkeypatching _make_records
    monkey_records = [
        HypothesisRecord(hypothesis_id="h1", text="a", room_type="dining_room", reward=0.9),
        HypothesisRecord(hypothesis_id="h2", text="b", room_type="dining_room", reward=0.1),
    ]
    loop._make_records = lambda **_k: monkey_records  # type: ignore[method-assign] # noqa: SLF001
    loop._sort_records = lambda recs: recs  # type: ignore[method-assign] # noqa: SLF001
    loop.pending_failure_examples["dining_room"] = ds.examples[:]
    assert loop._maybe_regenerate("dining_room", 0, 0) is True  # noqa: SLF001
    assert any(r.hypothesis_id == "h1" for r in loop.bank)


def test_hypothesis_engine_mirror_inference_and_store_fields(tmp_path: Path, monkeypatch) -> None:
    # Cover _mirror_inference_to_lake write branches (177-204) and store output fields (609, 623-624)
    out = tmp_path / "out"
    out.mkdir()
    (out / "scene_program.json").write_text("{}")
    (out / "critic_score.json").write_text(json.dumps({"overall": 0.5}))
    (out / "scene_features.json").write_text(json.dumps({"x": 1}))
    (out / "experiment_config.json").write_text("{}")
    scene_dir = out / "scene"
    (scene_dir / "views").mkdir(parents=True)
    (scene_dir / "views" / "view_front.png").write_bytes(b"x")
    (scene_dir / "turntable.mp4").write_bytes(b"v")

    class _Store:
        def __init__(self):
            self.json = []
            self.bytes = []

        def put_json(self, key, payload):
            self.json.append((key, payload))
            return f"s3://x/{key}"

        def put_bytes(self, key, b, content_type="application/octet-stream"):
            self.bytes.append((key, content_type, len(b)))
            return f"s3://x/{key}"

    store = _Store()
    uris = hyp_engine._mirror_inference_to_lake(  # noqa: SLF001
        store=store, run_id="rid", output_dir=out, inference_result={"ok": True}
    )
    assert any("silver/inference/rid/renders" in k for (k, _ct, _n) in store.bytes)
    assert any("gold/inference/rid/scene_features.json" in k for (k, _p) in store.json)
    assert uris

    # Store branch in run_vertical_inference: monkeypatch to avoid heavy work.
    monkeypatch.setattr(hyp_engine, "_mirror_inference_to_lake", lambda **_k: ["u1"])  # noqa: SLF001
    monkeypatch.setattr(hyp_engine, "create_manifest", lambda **_k: types.SimpleNamespace(model_dump=lambda **_x: {}))  # noqa: SLF001
    monkeypatch.setattr(hyp_engine, "finalize_manifest", lambda m, **_k: m)  # noqa: SLF001
    monkeypatch.setattr(hyp_engine, "_load_bank", lambda _p: [])  # noqa: SLF001
    monkeypatch.setattr(hyp_engine, "build_scene_llm", lambda _c: types.SimpleNamespace(generate_scene_program=lambda **_k: SceneProgram(prompt="p", room_type="living_room")))  # noqa: SLF001
    monkeypatch.setattr(hyp_engine, "build_scene_critic", lambda _c: HeuristicSceneCritic())  # noqa: SLF001
    monkeypatch.setattr(hyp_engine, "_select_hypotheses_for_inference", lambda *a, **k: [])  # noqa: SLF001,ARG005
    manifest = hyp_engine.run_vertical_inference(
        bank_path=tmp_path / "bank.json",
        prompt="p",
        output_dir=tmp_path / "inf",
        render_scene=False,
        store=store,
    )
    assert "lake_run_id" in manifest
    assert "lake_output_uris" in manifest


def test_scene_llm_bedrock_paths_and_build_scene_llm_branches(monkeypatch) -> None:
    # generic bedrock failure -> LLMUnavailableError (line 325)
    llm = scene_llm_mod.BedrockSceneLLM(config_mod.GeneratorConfig(provider="bedrock", model_id="m"))  # type: ignore[arg-type]

    class _Boom:
        def converse(self, **_k):
            raise RuntimeError("some other error")

    llm.client = _Boom()
    with pytest.raises(scene_llm_mod.LLMUnavailableError, match="Bedrock request failed"):
        llm._run_json_prompt("x")  # noqa: SLF001

    # generate_hypotheses repair prompt branch (line 348) and empty hypotheses branch (382)
    monkeypatch.setattr(scene_llm_mod.BedrockSceneLLM, "_run_json_prompt", lambda *_a, **_k: {"hypotheses": []})  # noqa: SLF001
    llm2 = scene_llm_mod.BedrockSceneLLM(config_mod.GeneratorConfig(provider="bedrock", model_id="m"))  # type: ignore[arg-type]
    ex = TrainingExample(example_id="e", room_type="dining_room", prompt="p", required_assets=["chair"])
    with pytest.raises(scene_llm_mod.StructuredOutputError, match="no valid hypotheses"):
        llm2.generate_hypotheses("dining_room", [ex], focus="repair")

    # generate_scene_program success path (covers return at line 405)
    monkeypatch.setattr(
        scene_llm_mod.BedrockSceneLLM,
        "_run_json_prompt",
        lambda *_a, **_k: {
            "style": "cozy",
            "hypotheses": ["h"],
            "assets": [
                {
                    "asset_type": "dining_table",
                    "count": 1,
                    "placement": "center",
                    "rationale": "anchor",
                }
            ],
            "constraints": ["c"],
            "render_spec": {"mode": "program_only"},
        },
    )  # noqa: SLF001
    llm3 = scene_llm_mod.BedrockSceneLLM(config_mod.GeneratorConfig(provider="bedrock", model_id="m"))  # type: ignore[arg-type]
    sp = llm3.generate_scene_program(prompt="p", room_type="dining_room", selected_hypotheses=["h"])
    assert sp.room_type == "dining_room"

    # build_scene_llm branches (413-415, 421)
    assert scene_llm_mod.build_scene_llm(config_mod.GeneratorConfig(provider="bedrock", model_id="m")).provider_name == "bedrock"  # type: ignore[arg-type]
    assert scene_llm_mod.build_scene_llm("bedrock").provider_name == "bedrock"
    with pytest.raises(ValueError, match="Unsupported llm provider"):
        scene_llm_mod.build_scene_llm(config_mod.GeneratorConfig(provider="nope"))  # type: ignore[arg-type]


def test_ec2_runner_remaining_branches(monkeypatch) -> None:
    # cover subnet/security group/key, spot options, ec2_ami_id fast path, and wait timeout
    class _EC2:
        def run_instances(self, **_kwargs):
            return {"Instances": [{"InstanceId": "i-x"}]}

        def describe_instances(self, **_kwargs):
            return {"Reservations": [{"Instances": [{"State": {"Name": "running"}}]}]}

        def terminate_instances(self, **_kwargs):
            return {}

        def describe_images(self, **_kwargs):
            return {"Images": []}

    class _SSM:
        pass

    ec2 = _EC2()

    def _fake_client(name, region_name=None):  # noqa: ARG001
        return ec2 if name == "ec2" else _SSM()

    monkeypatch.setattr(__import__("boto3"), "client", _fake_client)

    cfg = AppConfig(
        env="dev",
        storage_backend="local",
        local_lake_root=".",
        aws_region="us-east-1",
        ec2_instance_type="t3.small",
        ec2_iam_instance_profile="profile",
        ec2_subnet_id="subnet-1",
        ec2_security_group_id="sg-1",
        ec2_key_name="key",
        ec2_spot=True,
        ec2_spot_max_price=None,
        ec2_ami_id="ami-fixed",
    )
    runner = EC2JobRunner(app_config=cfg)
    spec = EC2JobSpec(command="run-inference", cli_args=["--env", "dev"])
    iid, _info = runner.launch(spec)
    assert iid == "i-x"

    with pytest.raises(TimeoutError):
        runner.wait("i-x", poll_interval_seconds=0, timeout_minutes=0)

    # cover _resolve_ami no images error by clearing ec2_ami_id
    runner.cfg = runner.cfg.model_copy(update={"ec2_ami_id": None})
    with pytest.raises(RuntimeError, match="Could not find"):
        runner._resolve_ami()  # noqa: SLF001


def test_multilayer_s3_store_read_json_and_exists_true() -> None:
    from compos3d.storage.multibucket_s3 import MultiLayerS3Store

    store = MultiLayerS3Store("b1", "b2", "b3", prefix="p", region="us-east-1")

    class _Body:
        def read(self):
            return b"{\"a\": 1}"

    class _FakeS3:
        class exceptions:
            ClientError = Exception

        def get_object(self, **_kwargs):
            return {"Body": _Body()}

        def head_object(self, **_kwargs):
            return {}

    store.s3 = _FakeS3()
    assert store.read_json("bronze/x.json") == {"a": 1}
    assert store.exists("bronze/x.json") is True


def test_hypothesis_engine_train_config_path_room_filter_and_inference_render_scene(
    tmp_path: Path, monkeypatch
) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "generator": {"provider": "mock"},
                "critic": {"mode": "heuristic", "provider": "mock"},
                "training": {"top_k": 1},
                "render": {"enabled": True, "resolution": "256x256", "view_samples": 1},
                "room_types": ["dining_room"],
            }
        )
    )

    ds_path = tmp_path / "ds.json"
    ds_path.write_text(
        json.dumps(
            {
                "dataset_id": "d",
                "examples": [
                    {
                        "example_id": "e1",
                        "room_type": "dining_room",
                        "prompt": "p",
                        "required_assets": ["chair"],
                    },
                    {
                        "example_id": "e2",
                        "room_type": "living_room",
                        "prompt": "p",
                        "required_assets": ["sofa"],
                    },
                ],
            }
        )
    )

    class _FakeLoop:
        def __init__(self, **kwargs):
            assert len(kwargs["dataset"].examples) == 1

        def train(self):
            return {"ok": True}

    monkeypatch.setattr(hyp_engine, "SceneHypothesisLoop", _FakeLoop)
    monkeypatch.setattr(
        hyp_engine, "make_training_renderer", lambda _cfg: (lambda *_a, **_k: [])
    )
    monkeypatch.setattr(hyp_engine, "build_scene_llm", lambda _cfg: types.SimpleNamespace())
    monkeypatch.setattr(
        hyp_engine, "build_scene_critic", lambda _cfg: types.SimpleNamespace()
    )

    out = hyp_engine.train_vertical_slice(
        dataset_path=ds_path,
        output_dir=tmp_path / "out",
        experiment_name="x",
        config_path=cfg_path,
    )
    assert out["ok"] is True

    bank_path = tmp_path / "bank.json"
    bank_path.write_text("[]")
    view = tmp_path / "render.png"
    view.write_bytes(b"x")
    monkeypatch.setattr(hyp_engine, "build_scene", lambda _req: {"rendered_views": [str(view)]})
    monkeypatch.setattr(
        hyp_engine,
        "build_scene_llm",
        lambda _cfg: types.SimpleNamespace(
            generate_scene_program=lambda **_k: SceneProgram(prompt="p", room_type="dining_room")
        ),
    )
    monkeypatch.setattr(hyp_engine, "build_scene_critic", lambda _cfg: HeuristicSceneCritic())
    monkeypatch.setattr(hyp_engine, "_select_hypotheses_for_inference", lambda *a, **k: [])  # noqa: ARG005

    m = hyp_engine.run_vertical_inference(
        bank_path=bank_path,
        prompt="p",
        output_dir=tmp_path / "inf_out",
        config_path=cfg_path,
        render_scene=True,
    )
    assert m["render_scene"] is True


def test_runner_run_script_non_capture_success_and_failure(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "x.py"
    script.write_text("print('hi')\n")

    # Success path (capture_output=False, cwd provided)
    def _ok_run(cmd, **kwargs):  # noqa: ARG001
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner_mod.subprocess, "run", _ok_run)
    out = runner_mod.run_script(script, ["--a", "1"], capture_output=False, cwd=tmp_path)
    assert out["exit_code"] == 0
    assert "stdout" not in out

    # Failure path with stderr snippet (capture_output=True)
    def _bad_run(cmd, **kwargs):  # noqa: ARG001
        return types.SimpleNamespace(returncode=7, stdout="o", stderr="e" * 600)

    monkeypatch.setattr(runner_mod.subprocess, "run", _bad_run)
    with pytest.raises(RuntimeError, match="exit 7"):
        runner_mod.run_script(script, [], capture_output=True)


def test_scene_llm_normalizers_cover_error_branches() -> None:
    # fenced code path
    payload = scene_llm_mod._extract_json_payload("```json\n{\"a\": 1}\n```")  # noqa: SLF001
    assert payload["a"] == 1

    assert scene_llm_mod._extract_style("A calm modern room") in {"calm", "modern"}  # noqa: SLF001
    assert scene_llm_mod._extract_requested_count("four chairs please", "chair") == 4  # noqa: SLF001
    assert scene_llm_mod._extract_requested_count("2 chairs please", "chair") == 2  # noqa: SLF001

    with pytest.raises(scene_llm_mod.StructuredOutputError):  # noqa: SLF001
        scene_llm_mod._normalize_hypotheses("nope")  # noqa: SLF001
    with pytest.raises(scene_llm_mod.StructuredOutputError):  # noqa: SLF001
        scene_llm_mod._normalize_hypotheses([])  # noqa: SLF001

    assert scene_llm_mod._normalize_constraints([" hi "]) == [{"text": "hi"}]  # noqa: SLF001
    assert scene_llm_mod._normalize_constraints([{"text": "x"}]) == [{"text": "x"}]  # noqa: SLF001
    with pytest.raises(scene_llm_mod.StructuredOutputError):  # noqa: SLF001
        scene_llm_mod._normalize_constraints([{"text": "   "}])  # noqa: SLF001
    with pytest.raises(scene_llm_mod.StructuredOutputError):  # noqa: SLF001
        scene_llm_mod._normalize_constraints([5])  # noqa: SLF001
    with pytest.raises(scene_llm_mod.StructuredOutputError):  # noqa: SLF001
        scene_llm_mod._normalize_constraints([])  # noqa: SLF001
    with pytest.raises(scene_llm_mod.StructuredOutputError):  # noqa: SLF001
        scene_llm_mod._normalize_constraints({})  # noqa: SLF001

    with pytest.raises(scene_llm_mod.StructuredOutputError):  # noqa: SLF001
        scene_llm_mod._normalize_assets([{"asset_type": "chair"}], "dining_room")  # noqa: SLF001
    with pytest.raises(scene_llm_mod.StructuredOutputError):  # noqa: SLF001
        scene_llm_mod._normalize_assets({}, "dining_room")  # noqa: SLF001
    with pytest.raises(scene_llm_mod.StructuredOutputError):  # noqa: SLF001
        scene_llm_mod._normalize_assets([5], "dining_room")  # noqa: SLF001
    with pytest.raises(scene_llm_mod.StructuredOutputError):  # noqa: SLF001
        scene_llm_mod._normalize_assets(  # noqa: SLF001
            [
                {
                    "asset_type": "not_supported",
                    "count": 1,
                    "placement": "p",
                    "rationale": "r",
                }
            ],
            "dining_room",
        )
    with pytest.raises(scene_llm_mod.StructuredOutputError):  # noqa: SLF001
        scene_llm_mod._normalize_assets(  # noqa: SLF001
            [{"asset_type": "chair", "count": "nope", "placement": "p", "rationale": "r"}],
            "dining_room",
        )
    with pytest.raises(scene_llm_mod.StructuredOutputError):  # noqa: SLF001
        scene_llm_mod._normalize_assets(  # noqa: SLF001
            [{"asset_type": "chair", "count": -1, "placement": "p", "rationale": "r"}],
            "dining_room",
        )
    with pytest.raises(scene_llm_mod.StructuredOutputError):  # noqa: SLF001
        scene_llm_mod._normalize_assets(  # noqa: SLF001
            [{"asset_type": "chair", "count": 1, "placement": "", "rationale": "r"}],
            "dining_room",
        )
    with pytest.raises(scene_llm_mod.StructuredOutputError):  # noqa: SLF001
        scene_llm_mod._normalize_assets(  # noqa: SLF001
            [{"asset_type": "chair", "count": 1, "placement": "p", "rationale": ""}],
            "dining_room",
        )
    with pytest.raises(scene_llm_mod.StructuredOutputError):  # noqa: SLF001
        scene_llm_mod._normalize_assets([], "dining_room")  # noqa: SLF001
    with pytest.raises(scene_llm_mod.StructuredOutputError):  # noqa: SLF001
        scene_llm_mod._normalize_scene_program_payload(  # noqa: SLF001
            {"room_type": "bedroom"}, prompt="p", room_type="dining_room"
        )
    with pytest.raises(scene_llm_mod.StructuredOutputError):  # noqa: SLF001
        scene_llm_mod._normalize_scene_program_payload(  # noqa: SLF001
            "not-a-dict", prompt="p", room_type="dining_room"
        )
    with pytest.raises(scene_llm_mod.StructuredOutputError):  # noqa: SLF001
        scene_llm_mod._normalize_scene_program_payload(  # noqa: SLF001
            {
                "render_spec": "not-a-dict",
                "hypotheses": ["h"],
                "assets": [
                    {
                        "asset_type": "chair",
                        "count": 1,
                        "placement": "p",
                        "rationale": "r",
                    }
                ],
                "constraints": ["c"],
            },
            prompt="p",
            room_type="dining_room",
        )

    # cover _dedupe_keep_order branches (skip empty + skip duplicate)
    assert scene_llm_mod._dedupe_keep_order(["", "A", "a", "B"]) == ["A", "B"]  # noqa: SLF001

    # cover mock repair-hypotheses branch (focus == "repair")
    examples = [
        TrainingExample(
            example_id="e",
            room_type="dining_room",
            prompt="p",
            required_assets=["chair"],
        )
    ]
    out = scene_llm_mod._mock_hypotheses("dining_room", examples, num_hypotheses=3, focus="repair")  # noqa: SLF001,E501
    assert len(out) == 3


def test_ec2_runner_resolve_ami_and_wait_paths(monkeypatch) -> None:
    # Fake boto3 clients
    class _FakeEC2:
        def __init__(self):
            self.calls = 0

        def describe_images(self, **_kwargs):
            return {
                "Images": [
                    {"CreationDate": "2024-01-01T00:00:00.000Z", "ImageId": "ami_old", "Name": "old"},
                    {"CreationDate": "2025-01-01T00:00:00.000Z", "ImageId": "ami_new", "Name": "new"},
                ]
            }

        def run_instances(self, **_kwargs):
            return {"Instances": [{"InstanceId": "i-123"}]}

        def describe_instances(self, **_kwargs):
            self.calls += 1
            state = "terminated" if self.calls >= 2 else "running"
            return {"Reservations": [{"Instances": [{"State": {"Name": state}}]}]}

        def terminate_instances(self, **_kwargs):
            return {}

    class _FakeSSM:
        pass

    def _fake_client(name, region_name=None):  # noqa: ARG001
        return _FakeEC2() if name == "ec2" else _FakeSSM()

    monkeypatch.setattr(
        __import__("boto3"),
        "client",
        _fake_client,
    )

    cfg = AppConfig(
        env="dev",
        storage_backend="local",
        local_lake_root=".",
        aws_region="us-east-1",
        ec2_instance_type="t3.small",
        ec2_iam_instance_profile="profile",
        ec2_spot=False,
    )
    runner = EC2JobRunner(app_config=cfg)
    spec = EC2JobSpec(command="train-hypotheses", cli_args=["--env", "dev"])
    instance_id, info = runner.launch(spec)
    assert instance_id == "i-123"
    assert info["ami_id"] == "ami_new"

    state = runner.wait(instance_id, poll_interval_seconds=0, timeout_minutes=1)
    assert state == "terminated"
    runner.terminate(instance_id)

