# Compos3D: Your Reasoning LLM is Secretly a 3D Scene generator

[![codecov](https://codecov.io/github/UofT-CSC490-W2026/Compos3D/branch/main/graph/badge.svg)](https://app.codecov.io/github/UofT-CSC490-W2026/Compos3D)

This repository provides the implementation of Compos3D, a system that uses LLMs to induce explicit design hypotheses from scene examples and applies them to generate structured 3D scenes via procedural rendering. Rather than generating raw code or direct geometry, Compos3D produces a SceneProgram, a structured JSON representation of furniture layout and constraints, which is then rendered into a full 3D scene with multiple camera views and an orbital video using Blender and Infinigen asset factories.

## Contents

- [✨ What the Project Does](#-what-the-project-does)
- [🗂️ Repository Layout](#-repository-layout)
- [🔧 Installation](#-installation)
- [📚 Dataset](#-dataset)
- [🏋️ Training](#-training)
- [🎬 Inference](#-inference)
- [🖼️ Rendering](#-rendering)
- [⚙️ Configuration](#-configuration)
- [📄 Paper Experiments](#-paper-experiments)
- [☁️ AWS](#-aws)
- [🧪 Testing](#-testing)
- [🤝 Credits](#-credits)

## ✨ What the Project Does

Compos3D turns a text prompt like:

`a cozy dining room with a round table, four chairs, a rug, and a lamp`

into:

1. A `SceneProgram` JSON with room type, style, assets, placements, and constraints.
2. A rendered 3D scene with four canonical views.
3. Optionally, an orbital video.
4. A critic score from either a heuristic scorer or a Bedrock VLM critic.

The main CLI commands are:

- `train-hypotheses`
- `run-inference`
- `build-scene`
- `evaluate`
- `backend-smoke`
- `reference-generate`
- `launch-aws`

## 🗂️ Repository Layout

```text
src/compos3d/
  cli.py                     CLI entrypoints
  config.py                  Generator / critic / render / training config models
  models.py                  SceneProgram, CriticScore, HypothesisRecord, dataset models
  catalog.py                 Supported room types and asset catalog
  data/
    dataset.py               Training dataset loader
    spatiallm.py             SpatialLM -> Compos3D dataset conversion
  hypothesis/
    loop.py                  Hypothesis training loop
    engine.py                Training / inference orchestration
    service.py               Request objects for CLI services
  llm/
    scene_llm.py             Mock + Bedrock scene generation and hypothesis filtering
    bedrock.py               Bedrock client helpers
  evaluation/
    critic.py                Heuristic and VLM critics
    service.py               Evaluation service
  procedural/
    runner.py                Blender subprocess runner
    service.py               Procedural backend entrypoints

procedural/
  scripts/build_scene.py     SceneProgram -> Blender scene -> renders/video
  scripts/asset_smoke.py     Asset smoke tests
  scripts/generate_room.py   Infinigen room generation
  furniture/                 Furniture factories and placement helpers

scripts/
  build_spatiallm_dataset.py Download and convert SpatialLM into training data
  build_dining_paper_benchmarks.py
                            Build the held-out dining-room paper benchmark
  run_paper_generation_eval.py
                            Run frozen-bank generation evaluation on a benchmark
  run_paper_edit_eval.py    Run prompt-edit evaluation on the edit-pair benchmark
  build_paper_training_matrix.py
                            Materialize ablation + sweep configs and commands
  make_paper_figures.py     Generate paper plots from a completed local run
  make_teaser_candidates.py Rank showcase prompts and optionally render them
  build_and_push_runtime_image.sh
                            Build and push the AWS runtime image to ECR
  put_secret_value.sh       Populate Secrets Manager values outside Terraform state

train_configs/
  compos3d.json              Full recommended training config
  compos3d_small_claude_wen.json
                             Small 1-epoch Bedrock smoke config

examples/
  dummy_fast.json            Tiny 4-example smoke dataset
  vertical_slice_dataset.json
                             Converted dataset derived from SpatialLM

data/external/spatiallm/
  split.csv
  spatiallm_train.json

paper/
  benchmarks/               Held-out dining-room val/test/showcase manifests
  results/                  Generation/editing/human-study outputs
  figures/                  Paper plots derived from the finalized run

infinigen/
  Princeton Infinigen submodule

terraform/
  main.tf                    Main Compos3D infra stack
  backends/*.hcl             Remote-state backend configs per environment
  bootstrap/                 One-time Terraform state bucket + lock table stack
```

## 🔧 Installation

Requirements:

- Python `3.11`
- Blender-compatible environment
- The `infinigen` submodule checked out
- AWS credentials

Clone and install:

```bash
git clone --recursive <your-repo-url>
cd Compos3D

uv venv .venv --python 3.11
source .venv/bin/activate 

# If on Windows:
# ./.venv/Scripts/activate.ps1
uv pip install -e .
```

If you will build the public dataset from Hugging Face, also install:

```bash
uv pip install huggingface_hub
```

For Bedrock runs, load credentials before the command:

```bash
source api_key
```

You can confirm the CLI is available with:

```bash
./.venv/bin/compos3d --help
```

## 📚 Dataset

Compos3D trains on JSON datasets of the form:

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

Supported room types:

- `dining_room`
- `living_room`
- `bedroom`

Supported assets by room:

| Room | Assets |
|---|---|
| `dining_room` | `dining_table`, `chair`, `lamp`, `rug`, `window`, `vase` |
| `living_room` | `sofa`, `lamp`, `rug`, `window`, `table_top`, `vase` |
| `bedroom` | `lamp`, `rug`, `window`, `chair`, `table_top` |

The dataset in this repo is derived from the public [SpatialLM dataset](https://huggingface.co/datasets/manycore-research/SpatialLM-Dataset). The converter remaps SpatialLM object labels into the Compos3D asset catalog and writes a balanced training set.

Build or rebuild it with:

```bash
./.venv/bin/python scripts/build_spatiallm_dataset.py --max-per-room 60
```

This produces:

- raw source files in `data/external/spatiallm/`
- converted dataset at `examples/vertical_slice_dataset.json`

If you already have the raw files locally and do not want to redownload:

```bash
./.venv/bin/python scripts/build_spatiallm_dataset.py \
  --skip-download \
  --raw-dir data/external/spatiallm \
  --output-path examples/vertical_slice_dataset.json \
  --max-per-room 60
```

## 🏋️ Training

Training runs the hypothesis loop over the dataset:

1. Seed hypotheses from initial examples.
2. Select top hypotheses with UCB-style reward.
3. Generate a `SceneProgram`.
4. Render and score it.
5. Update rewards.
6. Collect failures and repair / regenerate hypotheses when needed.

The recommended training run uses `train_configs/compos3d.json`:

```bash
source api_key
./.venv/bin/compos3d train-hypotheses \
  --dataset-path examples/vertical_slice_dataset.json \
  --output-dir artifacts/training \
  --experiment-name claude_qwen \
  --config-path train_configs/compos3d.json
```

The current full config uses:

- Generator: `us.anthropic.claude-sonnet-4-5-20250929-v1:0`
- Critic: `qwen.qwen3-vl-235b-a22b`
- Room types: `dining_room`
- Training renders: `256x256`
- Training samples: `100`
- `num_epochs=3`
- `selection_strategy="ucb"`
- `use_repair=true`

That section reflects the config currently checked into
`train_configs/compos3d.json`. If you want to train across all three room
types, update `room_types` in that file before launching the run.

Training artifacts go under:

`artifacts/training/<experiment-name>/`

Important files:

- `hypothesis_bank.json`: final hypothesis bank
- `metrics.json`: aggregate training metrics
- `predictions.jsonl`: training-time generated scene programs
- `training_trace.jsonl`: per-example hypothesis selection and scores
- `failed_scene_bank.jsonl`: failed examples captured for repair
- `bank_snapshots/`: periodic bank checkpoints
- `experiment_config.json`: frozen config for reproducibility
- `renders/`: rendered views for training predictions when render is enabled

| Field | Meaning |
|---|---|
| `training.num_epochs` | Number of passes over the dataset |
| `training.top_k` | How many hypotheses are used per example during training |
| `training.selection_strategy` | `ucb`, `greedy`, or `random` |
| `training.use_repair` | Whether failure-driven repair/regeneration is enabled |
| `training.success_threshold` | Score threshold for success |
| `training.max_num_hypotheses_per_room` | Cap on bank size per room type |
| `render.enabled` | Whether to render during training |
| `render.resolution` | Training render resolution |
| `render.view_samples` | Cycles samples per training render |
| `critic.mode` | `heuristic` or `vlm` |

## 🎬 Inference

Run frozen inference from a trained bank:

```bash
source api_key
./.venv/bin/compos3d run-inference \
  --bank-path artifacts/training/claude_qwen/hypothesis_bank.json \
  --prompt "a cozy dining room with a round table, four chairs, a rug, and a lamp" \
  --output-dir artifacts/inference/dining_room \
  --config-path train_configs/compos3d.json \
  --inference-strategy filter_and_weight \
  --render-scene
```

Available inference strategies:

- `joint_top_k`
  The top hypotheses are passed together to the generator.
- `filter_and_weight`
  Top hypotheses are filtered for relevance, candidate scene programs are generated per hypothesis, and the final scene is assembled by a weighted vote using hypothesis quality.

If you do not want rendering/video during frozen inference, omit `--render-scene`.

`artifacts/inference/<run-name>/`

- `scene_program.json`
- `critic_score.json`
- `scene_features.json`
- `experiment_config.json`
- `inference_manifest.json`
- `scene/` if `--render-scene` was used

## 🖼️ Rendering

If you already have a `SceneProgram`, render it directly:

```bash
./.venv/bin/compos3d build-scene \
  --scene-program artifacts/inference/dining_room/scene_program.json \
  --output-dir artifacts/scenes/dining_room \
  --resolution 512x512 \
  --view-samples 48 \
  --video-frames 90
```

To skip the video:

```bash
./.venv/bin/compos3d build-scene \
  --scene-program artifacts/inference/dining_room/scene_program.json \
  --output-dir artifacts/scenes/dining_room_fast \
  --resolution 256x256 \
  --view-samples 16 \
  --no-video
```

Typical outputs:

```text
artifacts/scenes/<name>/
  views/
    view_overhead.png
    view_front.png
    view_left.png
    view_right.png
  video_frames/
  video.mp4
  build_manifest.json
```

## ⚙️ Configuration

The two most important config files are:

- `train_configs/compos3d.json`
  Full recommended Bedrock training config.
- `train_configs/compos3d_small_claude_wen.json`
  Small 1-epoch smoke config.

Example `train_configs/compos3d.json` choices:

- generator provider/model
- critic provider/mode/model
- room type filtering
- render resolution and sample count
- training hyperparameters

If you change the config, keep these constraints in mind:

- `critic.mode="vlm"` requires `critic.provider="bedrock"`
- `render.enabled=true` is strongly recommended for VLM-backed training
- `256x256` and `8-16` samples are a good training-time speed/quality tradeoff
- `512x512` and `48+` samples are better for polished renders than for training

## 📄 Paper Experiments

We perform the following steps:

1. Build the held-out benchmark.
2. Run the final model on the `100`-prompt test split.
3. Run the no-hypotheses baseline on the same split.
4. Run the pairwise VLM quality judge against that baseline.

### 1. Build the held-out dining-room benchmark

This reads the raw SpatialLM files already in `data/external/spatiallm/`,
excludes the `60` dining-room examples used by
`examples/vertical_slice_dataset.json`, and writes the deterministic
`24/100/24` val/test/showcase split plus `100` prompt-edit pairs.

```bash
./.venv/bin/python scripts/build_dining_paper_benchmarks.py \
  --raw-dir data/external/spatiallm \
  --canonical-training-dataset-path examples/vertical_slice_dataset.json \
  --output-dir paper/benchmarks
```

This writes:

- `paper/benchmarks/dining_val.json`
- `paper/benchmarks/dining_test.json`
- `paper/benchmarks/dining_showcase.json`
- `paper/benchmarks/edit_pairs.jsonl`

### 2. Run the primary generation comparison

Final model on the held-out `100`-prompt test set:

```bash
source api_key
./.venv/bin/python scripts/run_paper_generation_eval.py \
  --benchmark-path paper/benchmarks/dining_test.json \
  --output-dir paper/results/generation/final_filter_and_weight_test \
  --method-name final_filter_and_weight \
  --bank-path artifacts/training/claude_qwen/hypothesis_bank.json \
  --config-path train_configs/compos3d.json \
  --inference-strategy filter_and_weight \
  --render-scene
```

No-hypotheses baseline on the same split:

```bash
source api_key
./.venv/bin/python scripts/run_paper_generation_eval.py \
  --benchmark-path paper/benchmarks/dining_test.json \
  --output-dir paper/results/generation/no_hypotheses_test \
  --method-name no_hypotheses \
  --use-empty-bank \
  --config-path train_configs/compos3d.json \
  --inference-strategy joint_top_k \
  --render-scene
```

Important outputs:

- `paper/results/generation/final_filter_and_weight_test/generation_results.jsonl`
- `paper/results/generation/final_filter_and_weight_test/summary.json`
- `paper/results/generation/no_hypotheses_test/generation_results.jsonl`
- `paper/results/generation/no_hypotheses_test/summary.json`

### 3. Run the main generation-quality judge

This adds the pairwise VLM image-comparison score for the final model against
the no-hypotheses baseline.

```bash
source api_key
./.venv/bin/python scripts/run_paper_generation_eval.py \
  --benchmark-path paper/benchmarks/dining_test.json \
  --output-dir paper/results/generation/final_filter_and_weight_test \
  --method-name final_filter_and_weight \
  --bank-path artifacts/training/claude_qwen/hypothesis_bank.json \
  --config-path train_configs/compos3d.json \
  --pairwise-baseline-results paper/results/generation/no_hypotheses_test
```

Important outputs:

- `paper/results/generation/final_filter_and_weight_test/pairwise/pairwise_generation.jsonl`
- `paper/results/generation/final_filter_and_weight_test/pairwise/pairwise_generation_summary.json`
- `paper/results/generation/quantitative_tables.md`

### 4. Generate the ablation and sweep matrix

This makes the training configs and command sheet.

```bash
./.venv/bin/python scripts/build_paper_training_matrix.py \
  --base-config-path train_configs/compos3d.json \
  --dataset-path examples/vertical_slice_dataset.json \
  --benchmark-val-path paper/benchmarks/dining_val.json \
  --benchmark-test-path paper/benchmarks/dining_test.json \
  --output-dir paper/results/training_matrix
```

Important outputs:

- `paper/results/training_matrix/experiment_matrix.json`
- `paper/results/training_matrix/run_commands.sh`
- `paper/results/training_matrix/configs/*.json`

### 5. Run prompt-edit evaluation

```bash
source api_key
./.venv/bin/python scripts/run_paper_edit_eval.py \
  --edit-pairs-path paper/benchmarks/edit_pairs.jsonl \
  --output-dir paper/results/editing/final_filter_and_weight \
  --method-name final_filter_and_weight \
  --bank-path artifacts/training/claude_qwen/hypothesis_bank.json \
  --config-path train_configs/compos3d.json \
  --inference-strategy filter_and_weight \
  --render-scene \
  --judge-edits
```

The main result file is:

- `paper/results/editing/<method>/editing_results.jsonl`

## ☁️ AWS

Compos3D runs on AWS:

- local commands still run directly from your checkout
- `launch-aws` runs the same inner `compos3d` command inside an ECR-backed container on an ephemeral EC2 instance
- secrets come from Secrets Manager at runtime
- CloudWatch captures container logs
- training checkpoints sync to a mutable Bronze checkpoint prefix for resume
- final artifacts still mirror into Bronze, Silver, and Gold through the existing Python engine

If you use AWS SSO or a named profile, set it once before running the AWS
commands below:

```bash
aws sso login --profile <your-profile>
export AWS_PROFILE=<your-profile>
export AWS_REGION=us-east-1
```

### 1. Bootstrap Terraform remote state

Run this once:

```bash
terraform -chdir=terraform/bootstrap init
terraform -chdir=terraform/bootstrap apply
```

Then initialize the main stack for `dev`:

```bash
terraform -chdir=terraform init -reconfigure -backend-config=backends/dev.hcl
```

For `staging` and `prod`, swap in `backends/staging.hcl` or `backends/prod.hcl`.

Keep Terraform in the `default` workspace. In this repo, environment isolation
comes from `terraform/backends/*.hcl` and `terraform/environments/*.tfvars`,
not from Terraform workspaces.

### 2. Apply the dev infrastructure

```bash
terraform -chdir=terraform plan -var-file=environments/dev.tfvars
terraform -chdir=terraform apply -var-file=environments/dev.tfvars
```

You can inspect the live output names with:

```bash
terraform -chdir=terraform output
```

Important outputs:

- `bronze_bucket`, `silver_bucket`, `gold_bucket`
- `ecr_repository_url`
- `ec2_instance_profile_name`
- `ec2_security_group_id`
- `ec2_log_group_name`
- `openai_secret_name`
- `anthropic_secret_name`
- `anyscale_secret_name`
- `wandb_secret_name`

### 3. Populate secret values outside Terraform state

Example for W&B:

```bash
WANDB_SECRET_NAME=$(terraform -chdir=terraform output -raw wandb_secret_name)
./scripts/put_secret_value.sh "$WANDB_SECRET_NAME" "$WANDB_API_KEY" us-east-1
```

Repeat for any other secret names you need, such as:

- `compos3d-dev-openai-key`
- `compos3d-dev-anthropic-key`
- `compos3d-dev-anyscale-key`

For the checked-in Bedrock configs in `train_configs/`, only `WANDB_API_KEY`
is required. The OpenAI, Anthropic, and Anyscale secrets are optional and are
skipped automatically if they do not have a current value.

### 4. Build and push the runtime image

```bash
ECR_REPOSITORY_URL=$(terraform -chdir=terraform output -raw ecr_repository_url)
./scripts/build_and_push_runtime_image.sh "$ECR_REPOSITORY_URL" latest us-east-1
```

This script requires local Docker daemon access.

### 5. Launch a dev smoke run

This uses the same CLI command shape as local training, but wraps it in the AWS runtime:

```bash
./.venv/bin/compos3d launch-aws train-hypotheses \
  --cli-args '--dataset-path examples/dummy_fast.json --output-dir artifacts/training --experiment-name aws_smoke --config-path train_configs/compos3d_small_claude_wen.json' \
  --env dev \
  --image-tag latest \
  --wait
```

Do not include `--env` inside `--cli-args`; `launch-aws` injects the outer
environment automatically.

### 6. Resume an interrupted training run

The AWS runtime syncs training state to:

`s3://<bronze-bucket>/<s3_prefix>/bronze/checkpoints/training/<experiment-name>/latest/`

Resume with the same command plus `--resume` inside `--cli-args`:

```bash
./.venv/bin/compos3d launch-aws train-hypotheses \
  --cli-args '--dataset-path examples/dummy_fast.json --output-dir artifacts/training --experiment-name aws_smoke --config-path train_configs/compos3d_small_claude_wen.json --resume' \
  --env dev \
  --image-tag latest \
  --wait
```

After the smoke run succeeds, the full AWS training command is:

```bash
./.venv/bin/compos3d launch-aws train-hypotheses \
  --cli-args '--dataset-path examples/vertical_slice_dataset.json --output-dir artifacts/training --experiment-name claude_qwen --config-path train_configs/compos3d.json' \
  --env dev \
  --image-tag latest \
  --wait
```

### Environment Policy

- `dev`: apply and smoke-test here first
- `staging`: keep configuration ready, but do not run jobs until dev is clean
- `prod`: same as staging, with a separate backend key, secrets, buckets, and log group

## 🧪 Testing

Run the test suite with:

```bash
./.venv/bin/pytest -q \
  --cov=compos3d \
  --cov-report=term-missing \
  --cov-report=xml \
  --basetemp .pytest_tmp_fresh \
  -p no:cacheprovider
```

The automated tests use mocks; they do not require Bedrock credentials.

## 🤝 Credits

This project builds on:

- [Infinigen](https://github.com/princeton-vl/infinigen)
- [HypoGeniC / hypothesis_generation](https://github.com/ChicagoHAI/hypothesis_generation)
- [SpatialLM dataset](https://huggingface.co/datasets/manycore-research/SpatialLM-Dataset)
