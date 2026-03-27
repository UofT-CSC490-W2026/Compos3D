# Compos3D: Hypothesis-Guided Controllable Procedural 3D Scene Generation

This repository provides the implementation of **Compos3D**, a system that uses LLMs to induce explicit design hypotheses from scene examples and applies them to generate structured 3D scenes via procedural rendering. Rather than generating raw code or direct geometry, Compos3D produces a `SceneProgram`, a structured JSON representation of furniture layout and constraints, which is then rendered into a full 3D scene with multiple camera views and an orbital video using Blender and Infinigen asset factories.

The hypothesis loop is a UCB-style multi-armed bandit that maintains a bank of scene design hypotheses, evaluates them by generating and rendering scenes, scores them with either a heuristic or VLM critic, and iteratively repairs low-performing hypotheses. This gives the system an explicit, inspectable inductive bias that is updated from experience.



## Contents

- [🔧 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [📖 Codebase Overview](#-codebase-overview)
- [📚 Dataset Format](#-dataset-format)
- [💻 Training the Hypothesis Bank](#-training-the-hypothesis-bank)
- [🎬 Inference and Scene Generation](#-inference-and-scene-generation)
- [🖼️ Rendering a Scene](#-rendering-a-scene)
- [☁️ Running on AWS EC2](#-running-on-aws-ec2)
- [⚙️ Configuration](#-configuration)
- [🧪 Running Tests](#-running-tests)
- [🤗 Credits](#-credits)



## 🔧 Installation

Requires Python 3.11 and a working Blender-compatible environment. The project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone --recursive https://github.com/your-org/Compos3D.git
cd Compos3D

# Create and activate the virtual environment
uv venv .venv --python 3.11
source .venv/bin/activate

# Install the project and its dependencies
uv pip install -e .
```

The `--recursive` flag is required to also clone the `infinigen` submodule, which provides the procedural asset factories (dining tables, chairs, sofas, lamps, etc.) used at render time.

For real LLM and VLM calls (training with Bedrock), load your AWS credentials:

```bash
source api_key
```

## 🚀 Quick Start

The full pipeline has three steps: train a hypothesis bank from scene examples, run inference to generate a `SceneProgram`, then render it to images and video.

First load your AWS credentials, then run the three steps:

```bash
source api_key
```

**Train a hypothesis bank:**

```bash
compos3d train-hypotheses \
  --dataset-path examples/vertical_slice_dataset.json \
  --output-dir artifacts/run \
  --experiment-name my_experiment \
  --config-path configs/compos3d.json
```

**Generate a scene** from the trained bank:

```bash
compos3d run-inference \
  --bank-path artifacts/run/my_experiment/hypothesis_bank.json \
  --prompt "a cozy dining room with a wooden table, four chairs, and warm lighting" \
  --output-dir artifacts/run/inference \
  --config-path configs/compos3d.json
```

**Render the scene** to 4 canonical views and an orbital video:

```bash
compos3d build-scene \
  --scene-program artifacts/run/inference/scene_program.json \
  --output-dir artifacts/run/scene \
  --resolution 512x512 \
  --view-samples 48
```

Outputs will be at `artifacts/run/scene/views/view_{overhead,front,left,right}.png` and `artifacts/run/scene/video.mp4`.

You can also combine inference and rendering in one command:

```bash
compos3d run-inference \
  --bank-path artifacts/run/my_experiment/hypothesis_bank.json \
  --prompt "a cozy dining room with a wooden table, four chairs, and warm lighting" \
  --output-dir artifacts/run/inference_with_render \
  --config-path configs/compos3d.json \
  --render-scene \
  --render-resolution 512x512 \
  --render-view-samples 48 \
  --render-video-frames 90
```



## 📖 Codebase Overview

```
src/compos3d/
  cli.py                 , CLI entrypoints (train-hypotheses, run-inference, build-scene, ...)
  config.py              , Pydantic config models (GeneratorConfig, CriticConfig, TrainingConfig, RenderConfig)
  catalog.py             , Supported room types and asset catalogs
  models.py              , Core data contracts (SceneProgram, HypothesisRecord, CriticScore, ...)
  hypothesis/
    loop.py              , UCB hypothesis training loop
    engine.py            , Orchestration: wires LLM, critic, renderer, and loop
    service.py           , Request/response types for training and inference
  llm/
    scene_llm.py         , SceneProgram generator interface and mock implementation
    bedrock.py           , AWS Bedrock LLM client
  evaluation/
    critic.py            , Heuristic and VLM critic implementations
    service.py           , Evaluation service
  procedural/
    runner.py            , Subprocess runner for Blender scripts (sets PYTHONPATH for infinigen + furniture)
    service.py           , Procedural backend service (backend-smoke, reference-generate)

procedural/
  scripts/
    build_scene.py       , Blender script: SceneProgram JSON → 3D scene → renders → video
    asset_smoke.py       , Smoke test a single asset factory
    generate_room.py     , Full Infinigen room generation (uses gin configs from submodule)
  furniture/             , Furniture factory
  llm_doc/               , Asset documentation (factory params, descriptions) used by the LLM

configs/
  compos3d.json          , Production config (LLM + VLM critic + rendering)

examples/
  dummy_fast.json        , Small dataset for smoke testing without API calls
  vertical_slice_dataset.json, Full vertical slice dataset for dining/living/bedroom

infinigen/               , Princeton Infinigen submodule (asset factories, gin configs, bpy utilities)
```



## 📚 Dataset Format

Training examples are JSON files with the following structure:

```json
{
  "dataset_id": "my_dataset",
  "examples": [
    {
      "example_id": "dr_001",
      "room_type": "dining_room",
      "prompt": "A dining room with a wooden table, four chairs, and a floor lamp",
      "required_assets": ["dining_table", "chair", "lamp"],
      "style": "modern"
    }
  ]
}
```

Supported room types are `dining_room`, `living_room`, and `bedroom`. The `required_assets` field specifies which assets must appear in a valid scene for this example. Valid asset types per room:

| Room | Assets |
|||
| `dining_room` | `dining_table`, `chair`, `lamp`, `rug`, `window`, `vase` |
| `living_room` | `sofa`, `lamp`, `rug`, `window`, `table_top`, `vase` |
| `bedroom` | `lamp`, `rug`, `window`, `chair`, `table_top` |



## 💻 Training the Hypothesis Bank

Training runs the UCB hypothesis loop over your dataset. Each epoch selects hypotheses using UCB1, generates a `SceneProgram` conditioned on the selected hypothesis, scores it with the critic, and updates the bank. Failed examples accumulate into a repair pool that triggers hypothesis regeneration.

```bash
source api_key

compos3d train-hypotheses \
  --dataset-path examples/vertical_slice_dataset.json \
  --output-dir artifacts/training \
  --experiment-name dining_vlm \
  --config-path configs/compos3d.json
```

Training artifacts are written to `artifacts/training/<experiment-name>/`:

| File | Description |
|||
| `hypothesis_bank.json` | Final trained hypothesis bank |
| `bank_snapshots/` | Bank state saved every N examples |
| `predictions.jsonl` | All ScenePrograms generated during training |
| `training_trace.jsonl` | Per-example scores and hypothesis selections |
| `failed_scene_bank.jsonl` | Failed ScenePrograms that triggered repair |
| `experiment_config.json` | Saved config for reproducibility |
| `metrics.json` | Aggregate validity, adherence, and recall scores |

Key training hyperparameters (set via `configs/compos3d.json`):

| Parameter | Description |
|||
| `training.num_epochs` | Number of passes over the dataset |
| `training.selection_strategy` | `ucb` (default), `greedy`, or `random` |
| `training.use_repair` | Whether failed hypotheses trigger repair regeneration |
| `training.baseline_mode` | `null` (full system), `no_hypotheses`, or `fixed_hypotheses` |
| `render.enabled` | Render each SceneProgram during training for VLM scoring |
| `render.resolution` | Render resolution, e.g. `256x256` for training, `512x512` for evaluation |



## 🎬 Inference and Scene Generation

Given a trained bank and a text prompt, inference selects the top-k hypotheses by UCB score, conditions the LLM on them, and generates a `SceneProgram`:

```bash
compos3d run-inference \
  --bank-path artifacts/training/dining_vlm/hypothesis_bank.json \
  --prompt "a minimalist dining room with a glass table and two chairs" \
  --output-dir artifacts/inference \
  --config-path configs/compos3d.json
```

The `SceneProgram` JSON is written to `artifacts/inference/scene_program.json` and looks like:

```json
{
  "room_type": "dining_room",
  "style": "minimalist",
  "hypotheses": ["In a dining_room, anchor the composition around dining_table, chair."],
  "assets": [
    {"asset_type": "dining_table", "count": 1, "placement": "center of room"},
    {"asset_type": "chair", "count": 2, "placement": "near table"}
  ],
  "constraints": [{"text": "chairs should face the table"}]
}
```



## 🖼️ Rendering a Scene

`build-scene` takes a `SceneProgram` JSON and produces renders using Blender (via `bpy`) and the Infinigen furniture factories. It outputs 4 canonical views and a 90-frame orbital turntable video.

```bash
compos3d build-scene \
  --scene-program artifacts/inference/scene_program.json \
  --output-dir artifacts/scene \
  --resolution 512x512 \
  --view-samples 48 \
  --video-frames 90
```

Expected outputs:

```
artifacts/scene/
  views/
    view_overhead.png
    view_front.png
    view_left.png
    view_right.png
  video_frames/       , individual PNG frames
  video.mp4           , compiled orbital video
  build_manifest.json , render metadata and paths
```

Typical render time is **5–10 seconds** for 4 views at 512×512 with 48 samples, plus another ~2 seconds for the 90-frame video.

To skip the video and only render the 4 views (faster, useful during training):

```bash
compos3d build-scene \
  --scene-program artifacts/inference/scene_program.json \
  --output-dir artifacts/scene \
  --resolution 256x256 \
  --view-samples 16 \
  --no-video
```



## ☁️ Running on AWS EC2

Every command that runs locally can also be submitted to an EC2 spot instance with a single `launch-aws` call.  All outputs are written directly to S3 (bronze/silver/gold buckets) via the standard `--env` flag, and job logs stream to CloudWatch.

### Data lake layout

When you pass `--env dev` (or `staging` / `prod`) to any command, outputs are mirrored to S3 in three layers:

| Layer | Bucket | Contents |
|---|---|---|
| Bronze | `compos3d-<env>-bronze` | Raw ScenePrograms, bank snapshots, training traces |
| Silver | `compos3d-<env>-silver` | Validated hypothesis banks, metrics, critic scores, rendered images |
| Gold | `compos3d-<env>-gold` | Final hypothesis bank (`latest.json`), training summaries, scene features |

Bucket names and other infra settings live in `config/env.<env>.yaml`.

### Writing to S3 from a local run

Add `--env dev` (or `staging`/`prod`) to any command to also mirror outputs to S3:

```bash
compos3d train-hypotheses \
  --dataset-path examples/vertical_slice_dataset.json \
  --output-dir artifacts/training \
  --experiment-name my_experiment \
  --config-path configs/compos3d.json \
  --env dev
```

```bash
compos3d run-inference \
  --bank-path artifacts/training/my_experiment/hypothesis_bank.json \
  --prompt "a bright dining room with four chairs" \
  --output-dir artifacts/inference \
  --config-path configs/compos3d.json \
  --env dev
```

### Provisioning AWS infrastructure

The `terraform/` directory manages S3 buckets, IAM roles, and the CloudWatch log group.  To also provision the EC2 instance profile and security group, uncomment the `ec2_compute` module in `terraform/main.tf`:

```hcl
module "ec2_compute" {
  source = "./modules/ec2_compute"

  project_name  = var.project_name
  environment   = var.environment
  bronze_bucket = local.bronze_bucket
  silver_bucket = local.silver_bucket
  gold_bucket   = local.gold_bucket
}
```

Then apply:

```bash
cd terraform
terraform init
terraform apply -var-file=environments/dev.tfvars
```

Copy the `ec2_instance_profile_name` and `ec2_security_group_id` outputs into `config/env.dev.yaml`.

### Submitting a job to EC2

Use `launch-aws` to spin up a spot instance that bootstraps itself and runs a `compos3d` command:

```bash
compos3d launch-aws train-hypotheses \
  --cli-args '--dataset-path s3://compos3d-dev-bronze/datasets/vs.json --config-path configs/compos3d.json --env dev' \
  --env dev \
  --git-ref main \
  --wait
```

The instance:
1. Clones the repo and checks out `--git-ref`
2. Installs dependencies (`pip install -e .`)
3. Runs the `compos3d` command
4. Writes outputs to S3 via `--env dev`
5. Self-terminates on completion

Follow logs live:

```bash
aws logs tail /compos3d/jobs --follow
```

Override the instance type (defaults to `g5.xlarge` for dev):

```bash
compos3d launch-aws train-hypotheses \
  --cli-args '...' \
  --env prod \
  --instance-type g5.2xlarge
```

The `ec2_spot` flag in `config/env.prod.yaml` is `false` by default for production (on-demand).



## ⚙️ Configuration

The single production config is `configs/compos3d.json`. It controls the generator, critic, rendering, and training loop in one place. To run ablations, edit `configs/compos3d.json` directly:

| Field | Values | Effect |
||||
| `training.selection_strategy` | `ucb`, `greedy`, `random` | Hypothesis selection policy |
| `training.use_repair` | `true`, `false` | Enable/disable repair on failures |
| `training.baseline_mode` | `null`, `no_hypotheses`, `fixed_hypotheses` | Baseline conditioning modes |
| `training.seed` | integer | Random seed for reproducibility |
| `render.enabled` | `true`, `false` | Render scenes during training (required for VLM critic) |
| `critic.mode` | `heuristic`, `vlm` | Scoring method (`vlm` requires Bedrock credentials) |



## 🧪 Running Tests

```bash
source .venv/bin/activate

pytest -q --cov=compos3d --cov-report=term-missing --cov-report=xml --basetemp .pytest_tmp_fresh -p no:cacheprovider
```

All tests use mock providers and run without any API credentials or Blender.



## 🤗 Credits

---

![tests](https://github.com/UofT-CSC490-W2026/Compos3D/actions/workflows/tests.yml/badge.svg)

This codebase builds on:

- [Infinigen](https://github.com/princeton-vl/infinigen)
- [HypoGenic](https://github.com/ChicagoHAI/hypothesis_generation)
