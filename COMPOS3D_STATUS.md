# Compos3D Research Handoff

## Overall Intention
- Build `Compos3D` as a hypothesis-guided controllable procedural 3D scene generation system.
- Use LLMs to induce explicit design hypotheses from scene examples, then apply selected hypotheses to generate a structured `SceneProgram` JSON rather than raw Blender code.
- Keep the system academically defensible: explicit hypotheses, saved bank snapshots, saved failure traces, reproducible configs, and fail-loud behavior instead of hidden fallbacks.
- Target the first paper-quality version at indoor single-room scenes, with eventual support for `bedroom`, `dining_room`, and `living_room`.

## Mandatory Environment Setup
- Always run `source .venv/bin/activate` first.
- For real Bedrock calls, also run `source api_key`.
- The public CLI entrypoint is `compos3d`.
- Current focused regression test command is `python -m pytest tests/ -v`.

## What To Find And Where To Find It
- Public CLI surface: `src/compos3d/cli.py`
- Core schemas and saved-record contracts: `src/compos3d/models.py`
- Catalog and room/asset normalization logic: `src/compos3d/catalog.py`
- Experiment config schema and defaults: `src/compos3d/config.py`
- Configs directory: `configs/` (fast_test, fast_render_test, default, bedrock_render)
- Dataset loading: `src/compos3d/data/dataset.py`
- Hypothesis training loop: `src/compos3d/hypothesis/loop.py`
- Training / inference engine: `src/compos3d/hypothesis/engine.py`
- Service-layer requests for training and inference: `src/compos3d/hypothesis/service.py`
- Bedrock client wrapper: `src/compos3d/llm/bedrock.py`
- Scene generator interfaces and structured-output checks: `src/compos3d/llm/scene_llm.py`
- Critic implementations and score aggregation: `src/compos3d/evaluation/critic.py`
- Evaluation service: `src/compos3d/evaluation/service.py`
- 3D scene builder script: `procedural/scripts/build_scene.py`
- Asset documentation (Infinigen factory params): `procedural/llm_doc/` (catalog assets only)
- Subprocess runner for procedural scripts: `src/compos3d/procedural/runner.py`
- Procedural service wiring: `src/compos3d/procedural/service.py`
- Example datasets: `examples/dummy_fast.json`, `examples/vertical_slice_dataset.json`
- Infinigen git submodule: `infinigen/`

## Current Overall State
- The public package is `src/compos3d`.
- **The full procedural-render-hypothesis pipeline is implemented and end-to-end tested.**
- **The codebase has been cleaned up**: `src/compos3d_dp/`, `InfEvolve/`, `hypothesis-generation/`, `terraform/`, `a1/`–`a4/`, and all legacy test files have been removed.

### What works now (all verified):
1. **`compos3d train-hypotheses`** — hypothesis training loop with UCB selection, repair, bank snapshots.
2. **Rendering inside the training loop** — when `render.enabled=true` in the config, every SceneProgram evaluation calls `build_scene` (bpy primitives + sky lighting), writes 4 views to `run_dir/renders/NNNN_<tag>/`, and passes image paths to the critic.
3. **`compos3d run-inference`** — generates a SceneProgram with trained hypotheses; with `--render-scene` it automatically calls `build_scene`, renders 4 views + orbital video, saves `scene_features.json` alongside the SceneProgram.
4. **`compos3d build-scene`** — standalone command: SceneProgram JSON → 4 canonical views + 90-frame MP4 orbital video in ~5 seconds.
5. **`compos3d backend-smoke`** / **`compos3d reference-generate`** — working procedural sub-commands.

### Asset rendering approach:
- `build_scene.py` spawns dimensionally-accurate bpy mesh primitives per asset type (boxes for tables/chairs/sofas, cylinders for lamps/vases).
- No dependency on InfEvolve or any external kenny factory — pure bpy + infinigen core utilities (init, butil, sky_lighting).
- The infinigen submodule (`infinigen/`) is added to `PYTHONPATH` by the subprocess runner; no pip install required.

### Config surface:
- `configs/compos3d.fast_test.json` — mock LLM, heuristic critic, no render (< 1s)
- `configs/compos3d.fast_render_test.json` — mock LLM, heuristic critic, render enabled at 256×256/8 samples (~30s per training run for verification)
- `configs/compos3d.bedrock_render.json` — Bedrock LLM + VLM critic + render enabled (full academic pipeline, requires `source api_key`)
- `configs/compos3d.default.json` — standard heuristic config without rendering

## Exact Things To Inspect Right Now
- Read `src/compos3d/hypothesis/loop.py` to understand the current multi-round bank update logic.
- Read `src/compos3d/llm/scene_llm.py` to understand the structured `SceneProgram` generation path.
- Read `src/compos3d/evaluation/critic.py` to understand the difference between heuristic and VLM evaluation.
- Read `src/compos3d/models.py` to see the contract for `SceneProgram`, `HypothesisRecord`, `CriticScore`, `FailureRecord`, and `TrainingTraceRecord`.
- Read `procedural/scripts/build_scene.py` to see the 3D scene builder and renderer.
- Inspect `tests/test_vertical_slice.py` to see what behavior is currently considered stable.

## Reproducible Commands

### Full pipeline — 3 integrated stages:
```bash
source .venv/bin/activate

# Stage 1: Training (no renders, < 1s):
compos3d train-hypotheses \
  --dataset-path examples/dummy_fast.json \
  --output-dir artifacts/e2e_test/training \
  --experiment-name e2e_test \
  --config-path configs/compos3d.fast_test.json

# Stage 2: Inference from trained bank:
compos3d run-inference \
  --bank-path artifacts/e2e_test/training/e2e_test/hypothesis_bank.json \
  --prompt "a cozy dining room with a wooden table, four chairs, and warm lighting" \
  --output-dir artifacts/e2e_test/inference \
  --config-path configs/compos3d.fast_test.json

# Stage 3: Render SceneProgram to 4 views + orbital video (~5s):
compos3d build-scene \
  --scene-program artifacts/e2e_test/inference/scene_program.json \
  --output-dir artifacts/e2e_test/render \
  --resolution 512x512 --view-samples 32

# Inference + auto-render in one command:
compos3d run-inference \
  --bank-path artifacts/e2e_test/training/e2e_test/hypothesis_bank.json \
  --prompt "a cozy dining room with a wooden table, four chairs, and warm lighting" \
  --output-dir artifacts/e2e_test/inference_render \
  --config-path configs/compos3d.fast_test.json \
  --render-scene --render-resolution 512x512 --render-view-samples 32

# Full academic pipeline (requires source api_key):
compos3d train-hypotheses \
  --dataset-path examples/vertical_slice_dataset.json \
  --output-dir artifacts/bedrock_render \
  --experiment-name dining_vlm \
  --config-path configs/compos3d.bedrock_render.json
```

### Other useful commands:
- Run all tests: `python -m pytest tests/ -v`
- CLI help: `compos3d --help`
- Asset smoke test: `compos3d backend-smoke --asset-name dining_table`
- Reference room generation (~5–15 min with --minimal): `compos3d reference-generate --room-type dining_room --minimal`

## Remaining Next Steps
- **Run with real Bedrock LLM + VLM**: `source api_key` then use `configs/compos3d.bedrock_render.json`.
- **Improve asset primitives**: replace flat-color boxes with more visually distinct geometry or textured materials so a VLM critic can better differentiate furniture types.
- **Add feature extraction (Stage 4)**: extract object bounding boxes and support flags from bpy inside `build_scene.py`; save richer `SceneFeatureRecord`.
- **Strengthen critic prompt with spatial features**: pass `scene_features.json` contents into the VLM critic prompt to help it score layout quality.
- **Stage 7 evaluation harness**: baselines (no_hypotheses, fixed_hypotheses, random), ID/OOD/edit splits, metric tables, qualitative figures, runtime summaries.
- **Improve placement**: replace the xy lookup table with a constraint-based placement engine (no-collision, support-surface detection).

## Gating Rule For Future Work
- After each stage, stop and manually inspect the produced artifacts before moving to the next stage.
- Do not add new fallback logic just to keep a run alive.
- Prefer fail-loud behavior, saved manifests, and small reproducible runs.
- Keep the codebase minimal and remove obsolete prototype paths from the public path once real replacements exist.
