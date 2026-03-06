"""
a2_mtp: Meta-style Multi-Token Prediction (MTP) with Weight Tying
=================================================================

Three training experiments, all at d16, seq=2048, full Chinchilla budget:
  1. Baseline — standard single-token prediction (mtp_k=0)
  2. MTP-2    — predict 2 future tokens in parallel per step (mtp_k=2)
  3. MTP-4    — predict 4 future tokens in parallel per step (mtp_k=4)

All runs start from scratch, same hyperparameters, same token budget.
Only CORE benchmark evaluation is performed (no custom long-context evals).

Evaluation: scripts.base_eval --eval=core on all 3 checkpoints.

Usage
-----
Individual stages:
    modal run nanochat_modal.py::quick_test_d12
    modal run nanochat_modal.py::stage_train_baseline
    modal run nanochat_modal.py::stage_train_mtp2
    modal run nanochat_modal.py::stage_train_mtp4
    modal run nanochat_modal.py::stage_eval_and_report

Full pipeline (smoke test → all 3 trains in parallel → eval):
    modal run nanochat_modal.py

Cost reference (8×H100 ~$31/hr)
---------------------------------
    d12 quick test    : ~15 min   (~$8)
    d16 baseline      : ~60 min   (~$31)
    d16 mtp2          : ~65 min   (~$34)  (parallel with baseline & mtp4)
    d16 mtp4          : ~70 min   (~$36)  (parallel with baseline & mtp2)
    eval (3 ckpts)    : ~30 min   (~$8, 2×H100)
"""

import os
import json
import subprocess
import modal
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# CONFIGURATION
# =============================================================================

DEPTH = 16  # picochat = d16 (~268M params)
GPU_TRAIN = "H100:8"
GPU_EVAL = "H100:4"

# Device batch size: d16 at seq=2048 fits 32 per H100 comfortably
DEVICE_BATCH = 32

# Fixed total batch size (tokens per optimizer step)
TOTAL_BATCH_SIZE = 524288

# Chinchilla-optimal token budget for d16:
#   model_dim = depth * aspect_ratio = 16 * 64 = 1024
#   Each transformer layer: attn ≈ 4*1024² + mlp ≈ 8*1024² ≈ 12.6M params/layer
#   16 layers ≈ 201M transformer_matrices + lm_head ≈ 33.5M → scaling_params ≈ 234M
#   target_param_data_ratio default = 10.5 → target_tokens ≈ 2.46B
#   At total_batch_size=524288 → total_steps ≈ 4693
CHINCHILLA_TOKENS = 2_460_000_000  # ≈ 10.5 × 234M scaling params
N_TOTAL_STEPS = CHINCHILLA_TOKENS // TOTAL_BATCH_SIZE  # ≈ 4693

# d12 quick-test step counts (just enough to exercise all code paths)
N_D12_STEPS = 200

# Model tags (subdirectories under base_checkpoints/)
TAG_BASELINE   = "a2mtp/d16_baseline"
TAG_MTP2       = "a2mtp/d16_mtp2"
TAG_MTP4       = "a2mtp/d16_mtp4"
TAG_MTP2_YARN  = "a2mtp/d16_mtp2_yarn"

TAG_D12_BASELINE  = "a2mtp/d12_baseline"
TAG_D12_MTP2      = "a2mtp/d12_mtp2"
TAG_D12_MTP4      = "a2mtp/d12_mtp4"
TAG_D12_MTP2_YARN = "a2mtp/d12_mtp2_yarn"

YARN_SCALE = 8.0  # YaRN context extension factor

# =============================================================================
# HYPERPARAMETER SWEEP — 9 short d16 runs, H100:2, 300 steps each
# =============================================================================

SWEEP_LRS      = [0.01, 0.02, 0.04]        # matrix LR values to test
SWEEP_BATCHES  = [131072, 262144, 524288]   # total batch sizes to test
SWEEP_STEPS    = 300
GPU_SWEEP      = "H100:2"
DEVICE_BATCH_SWEEP = 16                    # per-GPU device batch; grad_accum derived automatically
TIMEOUT_SWEEP  = 60 * 60 * 14             # 14 h for 4 configs × 9 runs = 36 sequential runs
WANDB_PROJECT_SWEEP = "part2_sweep"

# All 4 configs to sweep — each gets its own 9-run LR×batch grid
SWEEP_CONFIGS = [
    dict(name="baseline",  mtp_k=0, rope_type="rope",  yarn_scale=YARN_SCALE),
    dict(name="mtp2",      mtp_k=2, rope_type="rope",  yarn_scale=YARN_SCALE),
    dict(name="mtp4",      mtp_k=4, rope_type="rope",  yarn_scale=YARN_SCALE),
    dict(name="mtp2_yarn", mtp_k=2, rope_type="yarn",  yarn_scale=YARN_SCALE),
]

WANDB_PROJECT         = "part2_mtp"
WANDB_RUN_BASELINE    = "d16_baseline"
WANDB_RUN_MTP2        = "d16_mtp2"
WANDB_RUN_MTP4        = "d16_mtp4"
WANDB_RUN_MTP2_YARN   = "d16_mtp2_yarn"

# Timeouts
TIMEOUT_TRAIN = 60 * 60 * 2    # 2 h per training run
TIMEOUT_EVAL = 60 * 60 * 2     # 2 h for eval on 3 checkpoints
TIMEOUT_QUICKTEST = 60 * 60 * 1  # 1 h

# Volume / cache paths
VOLUME_MOUNT = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"
BASE_DIR = "/data/.cache/nanochat"

# =============================================================================
# MODAL PRIMITIVES
# =============================================================================

app = modal.App("nanochat-part2-mtp")
volume = Volume.from_name("nanochat-vol", create_if_missing=True)
secret = Secret.from_name("nanochat-secrets")

# Resolve the nanochat submodule path relative to this file (lives at ../a3/nanochat/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_NANOCHAT_DIR = os.path.join(_THIS_DIR, "..", "nanochat")
_PATCHES_DIR = os.path.join(_THIS_DIR, "patches")

image = (
    ModalImage.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git", "build-essential", "curl", "wget", "unzip")
    # nanochat repo — shared with a3 (same submodule, no duplication)
    .add_local_dir(
        local_path=_NANOCHAT_DIR,
        remote_path="/root/nanochat",
        copy=True,
    )
    # Apply patches: overwrite nanochat scripts with our MTP-patched versions
    # so the submodule stays clean.
    .add_local_file(
        local_path=os.path.join(_PATCHES_DIR, "gpt.py"),
        remote_path="/root/nanochat/nanochat/gpt.py",
        copy=True,
    )
    .add_local_file(
        local_path=os.path.join(_PATCHES_DIR, "base_train.py"),
        remote_path="/root/nanochat/scripts/base_train.py",
        copy=True,
    )
    .add_local_file(
        local_path=os.path.join(_PATCHES_DIR, "base_eval.py"),
        remote_path="/root/nanochat/scripts/base_eval.py",
        copy=True,
    )
    .workdir("/root/nanochat")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "echo 'source $HOME/.cargo/env' >> $HOME/.bashrc",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "echo 'export PATH=\"$HOME/.cargo/bin:$PATH\"' >> $HOME/.bashrc",
        "bash -c 'source $HOME/.cargo/env'",
    )
    .pip_install("uv")
    .env(
        {
            "OMP_NUM_THREADS": "1",
            "NANOCHAT_BASE_DIR": BASE_DIR,
            "HF_HOME": "/data/.cache/huggingface",
        }
    )
    .run_commands(
        "cd /root/nanochat && uv sync --extra gpu --no-install-project",
    )
)

# Lightweight CPU-only image used solely for figure generation.
# No CUDA / Rust / uv needed — just matplotlib + wandb + numpy.
figures_image = (
    ModalImage.debian_slim(python_version="3.11")
    .pip_install("wandb>=0.18", "matplotlib>=3.9", "numpy>=1.26")
)


# =============================================================================
# HELPERS
# =============================================================================


def _run(cmd: str) -> None:
    """Shell out to bash, stream stdout/stderr, and raise on failure."""
    print(f"\n>>>  {cmd}\n")
    result = subprocess.run(["bash", "-c", cmd], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command exited with code {result.returncode}:\n  {cmd}")


def _python(module: str, args: list | None = None, *, cwd: str = "/root/nanochat") -> None:
    """Run `uv run python -m {module} [args]` — for non-distributed scripts."""
    args = args or []
    _run(
        f"cd {cwd} && PYTHONPATH=/root/nanochat:$PYTHONPATH "
        f"uv run python -m {module} {' '.join(args)}"
    )


def _torchrun(module: str, args: list | None = None, *, nproc: int) -> None:
    """Run a training script under torchrun for multi-GPU distributed execution."""
    args = args or []
    args_str = (" -- " + " ".join(args)) if args else ""
    cmd = (
        f"cd /root/nanochat && "
        f"PYTHONPATH=/root/nanochat:$PYTHONPATH "
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m {module}{args_str}"
    )
    _run(cmd)


def _setup_cache() -> None:
    """Create cache directories and symlink BASE_DIR → NANOCHAT_CACHE on the volume."""
    os.makedirs(NANOCHAT_CACHE, exist_ok=True)
    if not os.path.lexists(BASE_DIR):
        os.makedirs("/data/.cache/", exist_ok=True)
        os.symlink(NANOCHAT_CACHE, BASE_DIR)
        print(f"Symlinked {BASE_DIR} -> {NANOCHAT_CACHE}")
    else:
        print(f"Cache symlink already exists: {BASE_DIR}")


def _find_last_step(model_tag: str) -> int:
    """Return the highest checkpoint step saved under base_checkpoints/{model_tag}."""
    import glob
    ckpt_dir = os.path.join(BASE_DIR, "base_checkpoints", model_tag)
    files = glob.glob(os.path.join(ckpt_dir, "model_*.pt"))
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return max(int(os.path.basename(f).split("_")[1].split(".")[0]) for f in files)


def _collect_core_csv(tag: str) -> dict:
    """
    After running scripts.base_eval for a given model tag, read the CSV it wrote and
    return:
      { "core_metric": float | None, "tasks": { task_name: score, ... } }
    base_eval (patched) writes to: {NANOCHAT_CACHE}/base_eval/{tag_slug}_{step:06d}.csv
    """
    import csv as _csv

    step = _find_last_step(tag)
    tag_slug = tag.replace("/", "-").replace("\\", "-")
    csv_path = os.path.join(NANOCHAT_CACHE, "base_eval", f"{tag_slug}_{step:06d}.csv")
    if not os.path.exists(csv_path):
        print(f"[warn] CORE CSV not found: {csv_path}")
        return {}
    tasks = {}
    core_metric = None
    with open(csv_path, newline="") as f:
        for row in _csv.reader(f):
            if len(row) < 2:
                continue
            name = row[0].strip()
            if name == "Task":
                continue
            centered = row[2].strip() if len(row) >= 3 else ""
            try:
                val = float(centered)
            except ValueError:
                val = None
            if name == "CORE":
                core_metric = val
            else:
                if val is not None:
                    tasks[name] = val
    print(f"  → {tag}: CORE={core_metric}, {len(tasks)} tasks read from {csv_path}")
    return {"core_metric": core_metric, "tasks": tasks}


def _build_report_markdown(
    results: dict,
    *,
    tag_baseline: str,
    tag_mtp2: str,
    tag_mtp4: str,
    tag_mtp2_yarn: str,
    depth: int,
    n_steps: int,
    chinchilla_tokens: int,
    total_batch_size: int,
    device_batch: int,
    n_gpus: int,
    wandb_run_baseline: str,
    wandb_run_mtp2: str,
    wandb_run_mtp4: str,
    wandb_run_mtp2_yarn: str,
    title_suffix: str = "",
) -> str:
    """
    Build the a2_mtp markdown report as a string.
    Contains ONLY objective tables — no prose commentary.
    Now covers 4 runs: Baseline, MTP-2, MTP-4, MTP-2+YaRN.
    """
    lines = []

    def section(title: str) -> None:
        lines.append(f"\n## {title}\n")

    def table(headers: list, rows: list) -> None:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        lines.append("")

    # Derived architecture values (nanochat defaults: aspect_ratio=64, head_dim=128)
    n_embd = depth * 64
    n_head = n_embd // 128
    n_kv_head = n_head
    approx_params_m = round(12 * depth * n_embd**2 / 1e6)
    grad_accum = total_batch_size // (device_batch * 2048 * n_gpus)

    title_part = f" — {title_suffix}" if title_suffix else ""
    lines.append(f"# a2_mtp: Meta-Style Multi-Token Prediction{title_part}\n")

    # ── 1. Model Architecture ──────────────────────────────────────────────────
    section("1. Model Architecture")
    table(
        ["Parameter", "Value"],
        [
            ["depth (n_layer)", str(depth)],
            ["n_embd", str(n_embd)],
            ["n_head", str(n_head)],
            ["n_kv_head", str(n_kv_head)],
            ["head_dim", "128"],
            ["aspect_ratio", "64"],
            ["vocab_size", "32768"],
            ["window_pattern", "SSSL  (S=half-ctx sliding, L=full-ctx)"],
            ["dtype", "bfloat16"],
            ["approx params", f"~{approx_params_m}M"],
            ["position encoding (baseline/mtp2/mtp4)", "RoPE"],
            ["position encoding (mtp2_yarn)", "YaRN NTK-by-Parts (scale=8.0)"],
            ["MTP heads (baseline)", "0  (standard single-token prediction)"],
            ["MTP heads (MTP-2)", "2  (predict tokens +1, +2; weight-tied)"],
            ["MTP heads (MTP-4)", "4  (predict tokens +1..+4; weight-tied)"],
            ["MTP heads (MTP-2+YaRN)", "2  (predict tokens +1, +2; weight-tied)"],
            ["optimizer (matrix)", "Muon"],
            ["optimizer (embed/unembed)", "Adam"],
            ["matrix_lr", "0.02"],
            ["embedding_lr", "0.3"],
            ["unembedding_lr", "0.004"],
            ["weight_decay", "0.2"],
            ["adam_beta1", "0.8"],
            ["adam_beta2", "0.95"],
            ["scalar_lr", "0.5"],
            ["warmup_ratio", "0.0"],
            ["warmdown_ratio", "0.5"],
            ["final_lr_frac", "0.0"],
        ],
    )

    # ── 2. Training Configuration ──────────────────────────────────────────────
    section("2. Training Configuration")
    table(
        ["Parameter", "Baseline", "MTP-2", "MTP-4", "MTP-2+YaRN"],
        [
            ["mtp_k",            "0",      "2",      "4",      "2"],
            ["rope_type",        "rope",   "rope",   "rope",   "yarn"],
            ["yarn_scale",       "—",      "—",      "—",      "8.0"],
            ["seq_len",          "2048",   "2048",   "2048",   "2048"],
            ["warm-start from",  "—",      "—",      "—",      "—"],
            ["steps",            str(n_steps)] * 4,
            ["tokens",           f"{chinchilla_tokens / 1e9:.3f}B"] * 4,
            ["total_batch_size", str(total_batch_size)] * 4,
            ["device_batch",     str(device_batch)] * 4,
            ["grad_accum",       str(grad_accum)] * 4,
            ["GPUs",             f"{n_gpus}×H100"] * 4,
            ["WandB run",        wandb_run_baseline, wandb_run_mtp2, wandb_run_mtp4, wandb_run_mtp2_yarn],
        ],
    )

    # ── 3. Training Metrics ────────────────────────────────────────────────────
    section("3. Training Metrics")
    all_tags_labels = [
        ("Baseline",    tag_baseline),
        ("MTP-2",       tag_mtp2),
        ("MTP-4",       tag_mtp4),
        ("MTP-2+YaRN",  tag_mtp2_yarn),
    ]
    training_rows = []
    for label, tag in all_tags_labels:
        r = results.get("training", {}).get(tag, {})
        training_rows.append([
            label,
            f"{r['val_bpb']:.6f}" if isinstance(r.get("val_bpb"), float) else "N/A",
            f"{r['core_metric']:.4f}" if isinstance(r.get("core_metric"), float) else "N/A",
            f"{r.get('training_time_min', 'N/A')}",
        ])
    table(["Run", "Final val BPB ↓", "CORE score ↑", "Train time (min)"], training_rows)

    # ── 4. CORE Per-Task Breakdown ─────────────────────────────────────────────
    section("4. CORE Per-Task Breakdown")
    core_tasks = results.get("core_tasks", {})
    all_tasks = sorted({t for tag_results in core_tasks.values() for t in tag_results})
    ordered_tags = [tag_baseline, tag_mtp2, tag_mtp4, tag_mtp2_yarn]
    if all_tasks:
        core_headers = ["Task", "Baseline", "MTP-2", "MTP-4", "MTP-2+YaRN"]
        core_rows = []
        for task in all_tasks:
            row = [task]
            for tag in ordered_tags:
                val = core_tasks.get(tag, {}).get(task, "N/A")
                row.append(f"{val:.4f}" if isinstance(val, float) else "N/A")
            core_rows.append(row)
        # CORE aggregate row at the bottom
        core_rows.append(
            ["**CORE aggregate**"]
            + [
                f"**{results['training'][tag]['core_metric']:.4f}**"
                if isinstance(results.get("training", {}).get(tag, {}).get("core_metric"), float)
                else "N/A"
                for tag in ordered_tags
            ]
        )
        table(core_headers, core_rows)
    else:
        lines.append("_No CORE per-task data available._\n")

    return "\n".join(lines)


_N_TRAIN_GPUS = int(GPU_TRAIN.split(":")[1]) if ":" in GPU_TRAIN else 1
_N_EVAL_GPUS = int(GPU_EVAL.split(":")[1]) if ":" in GPU_EVAL else 1


# =============================================================================
# STAGE: HYPERPARAMETER SWEEP — 9 sequential runs on 2×H100
# =============================================================================

_N_SWEEP_GPUS = int(GPU_SWEEP.split(":")[1]) if ":" in GPU_SWEEP else 1


def _run_sweep_for_config(cfg: dict, depth: int) -> None:
    """
    Runs the 9-run LR×batch grid for a single config dict.
    Called by each of the four per-config sweep stages.
    """
    _setup_cache()
    cfg_name   = cfg["name"]
    mtp_k      = cfg["mtp_k"]
    rope_type  = cfg["rope_type"]
    yarn_scale = cfg["yarn_scale"]
    lrs        = SWEEP_LRS
    batches    = SWEEP_BATCHES
    n_steps    = SWEEP_STEPS
    bs         = DEVICE_BATCH_SWEEP
    nproc      = _N_SWEEP_GPUS
    total      = len(lrs) * len(batches)

    print(f"\n{'#' * 64}\nSweep config: {cfg_name}  mtp_k={mtp_k}  rope_type={rope_type}\n{'#' * 64}")

    idx = 0
    for lr in lrs:
        for batch in batches:
            idx += 1
            run_name = f"sweep_{cfg_name}_lr{lr:g}_bs{batch // 1000}k"
            tag      = f"a2mtp/sweep/{run_name}"
            print(f"\n{'=' * 64}\n[{idx}/{total}] {run_name}\n{'=' * 64}")
            _torchrun(
                "scripts.base_train",
                [
                    f"--depth={depth}",
                    "--max-seq-len=2048",
                    f"--model-tag={tag}",
                    f"--mtp-k={mtp_k}",
                    f"--rope-type={rope_type}",
                    f"--yarn-scale={yarn_scale}",
                    f"--matrix-lr={lr}",
                    f"--device-batch-size={bs}",
                    f"--total-batch-size={batch}",
                    f"--num-iterations={n_steps}",
                    "--save-every=9999",        # no mid-run checkpoints
                    "--core-metric-every=9999", # skip CORE eval
                    "--sample-every=-1",
                    f"--wandb-project={WANDB_PROJECT_SWEEP}",
                    f"--run={run_name}",
                ],
                nproc=nproc,
            )
            volume.commit()
            print(f"  Done: {run_name}")

    print(
        f"\n{'=' * 64}\n"
        f"Config '{cfg_name}' sweep done. {total} runs in '{WANDB_PROJECT_SWEEP}'.\n"
        f"Filter by '{cfg_name}' in W&B and sort by step-{n_steps} train/loss.\n"
        f"{'=' * 64}"
    )


@app.function(image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
              gpu=GPU_SWEEP, timeout=TIMEOUT_SWEEP)
def stage_sweep_baseline(depth: int = DEPTH) -> None:
    """Sweep: baseline (mtp_k=0, rope=rope) — 9 runs, 300 steps each, 2×H100."""
    _run_sweep_for_config(SWEEP_CONFIGS[0], depth)


@app.function(image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
              gpu=GPU_SWEEP, timeout=TIMEOUT_SWEEP)
def stage_sweep_mtp2(depth: int = DEPTH) -> None:
    """Sweep: MTP-2 (mtp_k=2, rope=rope) — 9 runs, 300 steps each, 2×H100."""
    _run_sweep_for_config(SWEEP_CONFIGS[1], depth)


@app.function(image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
              gpu=GPU_SWEEP, timeout=TIMEOUT_SWEEP)
def stage_sweep_mtp4(depth: int = DEPTH) -> None:
    """Sweep: MTP-4 (mtp_k=4, rope=rope) — 9 runs, 300 steps each, 2×H100."""
    _run_sweep_for_config(SWEEP_CONFIGS[2], depth)


@app.function(image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
              gpu=GPU_SWEEP, timeout=TIMEOUT_SWEEP)
def stage_sweep_mtp2_yarn(depth: int = DEPTH) -> None:
    """Sweep: MTP-2+YaRN (mtp_k=2, rope=yarn) — 9 runs, 300 steps each, 2×H100."""
    _run_sweep_for_config(SWEEP_CONFIGS[3], depth)


# =============================================================================
# TRAINING STAGES — all d16, seq=2048, from scratch
# =============================================================================

def _train_run(
    *,
    depth: int,
    mtp_k: int,
    tag: str,
    wandb_run: str,
    n_steps: int,
    device_batch: int = DEVICE_BATCH,
    nproc: int = _N_TRAIN_GPUS,
    core_metric_every: int = 1000,
    rope_type: str = "rope",
    yarn_scale: float = YARN_SCALE,
) -> None:
    """Shared training logic for all MTP/YaRN experiments."""
    _setup_cache()
    print(f"Training: depth={depth} seq=2048 mtp_k={mtp_k} rope_type={rope_type} steps={n_steps} tag={tag}")
    _torchrun(
        "scripts.base_train",
        [
            f"--depth={depth}",
            "--max-seq-len=2048",
            f"--model-tag={tag}",
            f"--mtp-k={mtp_k}",
            f"--rope-type={rope_type}",
            f"--yarn-scale={yarn_scale}",
            f"--device-batch-size={device_batch}",
            f"--total-batch-size={TOTAL_BATCH_SIZE}",
            f"--num-iterations={n_steps}",
            "--save-every=500",
            f"--wandb-project={WANDB_PROJECT}",
            f"--run={wandb_run}",
            f"--core-metric-every={core_metric_every}",
        ],
        nproc=nproc,
    )
    volume.commit()
    print(f"Done. Checkpoint saved: base_checkpoints/{tag}/")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_TRAIN,
)
def stage_train_baseline(depth: int = DEPTH, n_steps: int = N_TOTAL_STEPS) -> None:
    """Baseline: d16, seq=2048, mtp_k=0, full Chinchilla budget from scratch."""
    _train_run(
        depth=depth, mtp_k=0, tag=TAG_BASELINE,
        wandb_run=WANDB_RUN_BASELINE, n_steps=n_steps,
    )


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_TRAIN,
)
def stage_train_mtp2(depth: int = DEPTH, n_steps: int = N_TOTAL_STEPS) -> None:
    """MTP-2: d16, seq=2048, mtp_k=2 (predict 2 tokens ahead, weight-tied)."""
    _train_run(
        depth=depth, mtp_k=2, tag=TAG_MTP2,
        wandb_run=WANDB_RUN_MTP2, n_steps=n_steps,
    )


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_TRAIN,
)
def stage_train_mtp4(depth: int = DEPTH, n_steps: int = N_TOTAL_STEPS) -> None:
    """MTP-4: d16, seq=2048, mtp_k=4 (predict 4 tokens ahead, weight-tied)."""
    _train_run(
        depth=depth, mtp_k=4, tag=TAG_MTP4,
        wandb_run=WANDB_RUN_MTP4, n_steps=n_steps,
    )


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_TRAIN,
)
def stage_train_mtp2_yarn(depth: int = DEPTH, n_steps: int = N_TOTAL_STEPS) -> None:
    """MTP-2 + YaRN: d16, seq=2048, mtp_k=2, rope_type=yarn.
    Combined ablation: both Multi-Token Prediction (k=2, weight-tied) and
    YaRN NTK-by-Parts RoPE (8× scale), everything else identical to baseline.
    """
    _train_run(
        depth=depth, mtp_k=2, rope_type="yarn", yarn_scale=YARN_SCALE,
        tag=TAG_MTP2_YARN,
        wandb_run=WANDB_RUN_MTP2_YARN, n_steps=n_steps,
    )


# =============================================================================
# STAGE: EVAL + REPORT — CORE benchmark on all 3 checkpoints
# =============================================================================


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_EVAL,
    timeout=TIMEOUT_EVAL,
)
def stage_eval_and_report() -> None:
    """
    CORE evaluation on all 4 checkpoints → results JSON → markdown report.

      A. Download eval bundle (if not cached)
      B. CORE eval (scripts.base_eval --eval=core) on baseline, mtp2, mtp4, mtp2_yarn
      C. Merge CORE CSV results → JSON
      D. Build and write markdown report

    Writes to:
      nanochat_cache/a2mtp_eval_results.json
      nanochat_cache/report/a2mtp_report.md
    """
    volume.reload()
    _setup_cache()

    eval_bundle_dir = os.path.join(NANOCHAT_CACHE, "eval_bundle")
    if not os.path.isdir(eval_bundle_dir):
        print("Downloading eval bundle (~1GB)...")
        zip_path = "/tmp/eval_bundle.zip"
        eval_bundle_url = (
            "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
        )
        _run(f"curl -L -o {zip_path} {eval_bundle_url}")
        _run(f"unzip -q {zip_path} -d {NANOCHAT_CACHE} && rm {zip_path}")
        volume.commit()

    tags = [TAG_BASELINE, TAG_MTP2, TAG_MTP4, TAG_MTP2_YARN]
    core_data: dict = {}

    # ── A. CORE eval on all four checkpoints ──────────────────────────────────
    for tag in tags:
        print(f"\n{'=' * 60}\nCORE eval: {tag}\n{'=' * 60}")
        _torchrun(
            "scripts.base_eval",
            [
                f"--model-tag={tag}",
                "--eval=core,bpb",
                "--max-per-task=500",
                "--device-batch-size=16",
            ],
            nproc=_N_EVAL_GPUS,
        )
        core_data[tag] = _collect_core_csv(tag)

    # ── B. Build results JSON ──────────────────────────────────────────────────
    results: dict = {"training": {}, "core_tasks": {}}
    for tag in tags:
        d = core_data.get(tag, {})
        results["training"][tag] = {
            "core_metric": d.get("core_metric"),
            "val_bpb": None,  # not tracked separately; bpb reported via CORE eval
        }
        if d.get("tasks"):
            results["core_tasks"][tag] = d["tasks"]

    results_path = os.path.join(NANOCHAT_CACHE, "a2mtp_eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()

    # ── C. Build and write markdown report ────────────────────────────────────
    print(f"\n{'=' * 60}\nGenerating report\n{'=' * 60}")
    report_md = _build_report_markdown(
        results,
        tag_baseline=TAG_BASELINE,
        tag_mtp2=TAG_MTP2,
        tag_mtp4=TAG_MTP4,
        tag_mtp2_yarn=TAG_MTP2_YARN,
        depth=DEPTH,
        n_steps=N_TOTAL_STEPS,
        chinchilla_tokens=CHINCHILLA_TOKENS,
        total_batch_size=TOTAL_BATCH_SIZE,
        device_batch=DEVICE_BATCH,
        n_gpus=_N_TRAIN_GPUS,
        wandb_run_baseline=WANDB_RUN_BASELINE,
        wandb_run_mtp2=WANDB_RUN_MTP2,
        wandb_run_mtp4=WANDB_RUN_MTP4,
        wandb_run_mtp2_yarn=WANDB_RUN_MTP2_YARN,
    )

    report_dir = os.path.join(NANOCHAT_CACHE, "report")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "a2mtp_report.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    volume.commit()

    print(f"\nEval JSON : {results_path}")
    print(f"Report    : {report_path}")
    print("\n" + "=" * 60)
    print("FULL REPORT:")
    print("=" * 60)
    print(report_md)


# =============================================================================
# D12 QUICK TEST — validates full pipeline in ~15 minutes
# =============================================================================


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu="H100:4",
    timeout=TIMEOUT_QUICKTEST,
)
def quick_test_d12() -> None:
    """
    Smoke test for the full a2_mtp pipeline at d12 scale:
      - Baseline: d12, seq=2048, mtp_k=0, 200 steps
      - MTP-2:    d12, seq=2048, mtp_k=2, 200 steps
      - MTP-4:    d12, seq=2048, mtp_k=4, 200 steps
      - Eval:     CORE on all 3 d12 checkpoints
      - Report:   markdown report generation

    This validates the MTP loss, patching, CORE eval, and report end-to-end
    before spending money on d16.
    """
    _setup_cache()
    nproc = 4
    bs = 8  # small batch for quick test
    total_bs = 65536  # much smaller total batch for quick test
    n_steps = N_D12_STEPS

    # --- Baseline (d12, mtp_k=0) ---
    print("=== Quick test: Baseline (d12, mtp_k=0, 200 steps) ===")
    _torchrun(
        "scripts.base_train",
        [
            "--depth=12",
            "--max-seq-len=2048",
            f"--model-tag={TAG_D12_BASELINE}",
            "--mtp-k=0",
            f"--device-batch-size={bs}",
            f"--total-batch-size={total_bs}",
            f"--num-iterations={n_steps}",
            "--save-every=100",
            "--core-metric-every=999999",
            "--sample-every=-1",
            f"--wandb-project={WANDB_PROJECT}",
            "--run=d12_baseline",
        ],
        nproc=nproc,
    )
    volume.commit()

    # --- MTP-2 (d12, mtp_k=2) ---
    print("=== Quick test: MTP-2 (d12, mtp_k=2, 200 steps) ===")
    _torchrun(
        "scripts.base_train",
        [
            "--depth=12",
            "--max-seq-len=2048",
            f"--model-tag={TAG_D12_MTP2}",
            "--mtp-k=2",
            f"--device-batch-size={bs}",
            f"--total-batch-size={total_bs}",
            f"--num-iterations={n_steps}",
            "--save-every=100",
            "--core-metric-every=999999",
            "--sample-every=-1",
            f"--wandb-project={WANDB_PROJECT}",
            "--run=d12_mtp2",
        ],
        nproc=nproc,
    )
    volume.commit()

    # --- MTP-4 (d12, mtp_k=4) ---
    print("=== Quick test: MTP-4 (d12, mtp_k=4, 200 steps) ===")
    _torchrun(
        "scripts.base_train",
        [
            "--depth=12",
            "--max-seq-len=2048",
            f"--model-tag={TAG_D12_MTP4}",
            "--mtp-k=4",
            f"--device-batch-size={bs}",
            f"--total-batch-size={total_bs}",
            f"--num-iterations={n_steps}",
            "--save-every=100",
            "--core-metric-every=999999",
            "--sample-every=-1",
            f"--wandb-project={WANDB_PROJECT}",
            "--run=d12_mtp4",
        ],
        nproc=nproc,
    )
    volume.commit()

    # --- MTP-2 + YaRN (d12, mtp_k=2, yarn) ---
    print("=== Quick test: MTP-2+YaRN (d12, mtp_k=2, rope_type=yarn, 200 steps) ===")
    _torchrun(
        "scripts.base_train",
        [
            "--depth=12",
            "--max-seq-len=2048",
            f"--model-tag={TAG_D12_MTP2_YARN}",
            "--mtp-k=2",
            "--rope-type=yarn",
            f"--yarn-scale={YARN_SCALE}",
            f"--device-batch-size={bs}",
            f"--total-batch-size={total_bs}",
            f"--num-iterations={n_steps}",
            "--save-every=100",
            "--core-metric-every=999999",
            "--sample-every=-1",
            f"--wandb-project={WANDB_PROJECT}",
            "--run=d12_mtp2_yarn",
        ],
        nproc=nproc,
    )
    volume.commit()

    # --- Download eval bundle ---
    eval_bundle_dir = os.path.join(NANOCHAT_CACHE, "eval_bundle")
    if not os.path.isdir(eval_bundle_dir):
        print("Downloading eval bundle (~1GB)...")
        zip_path = "/tmp/eval_bundle.zip"
        _run(f"curl -L -o {zip_path} https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip")
        _run(f"unzip -q {zip_path} -d {NANOCHAT_CACHE} && rm {zip_path}")
        volume.commit()

    # --- CORE eval on all 4 d12 checkpoints ---
    d12_tags = [TAG_D12_BASELINE, TAG_D12_MTP2, TAG_D12_MTP4, TAG_D12_MTP2_YARN]
    core_data: dict = {}
    for tag in d12_tags:
        print(f"\n=== Quick test CORE eval: {tag} ===")
        _torchrun(
            "scripts.base_eval",
            [
                f"--model-tag={tag}",
                "--eval=core",
                "--max-per-task=50",  # very small for speed
                "--device-batch-size=4",
            ],
            nproc=nproc,
        )
        core_data[tag] = _collect_core_csv(tag)

    # --- Build results dict and report ---
    results: dict = {"training": {}, "core_tasks": {}}
    for tag in d12_tags:
        d = core_data.get(tag, {})
        results["training"][tag] = {"core_metric": d.get("core_metric"), "val_bpb": None}
        if d.get("tasks"):
            results["core_tasks"][tag] = d["tasks"]

    out_path = os.path.join(NANOCHAT_CACHE, "a2mtp_d12_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    report_md = _build_report_markdown(
        results,
        tag_baseline=TAG_D12_BASELINE,
        tag_mtp2=TAG_D12_MTP2,
        tag_mtp4=TAG_D12_MTP4,
        tag_mtp2_yarn=TAG_D12_MTP2_YARN,
        depth=12,
        n_steps=n_steps,
        chinchilla_tokens=CHINCHILLA_TOKENS,
        total_batch_size=total_bs,
        device_batch=bs,
        n_gpus=nproc,
        wandb_run_baseline="d12_baseline",
        wandb_run_mtp2="d12_mtp2",
        wandb_run_mtp4="d12_mtp4",
        wandb_run_mtp2_yarn="d12_mtp2_yarn",
        title_suffix="d12 smoke test",
    )

    report_dir = os.path.join(NANOCHAT_CACHE, "report")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "a2mtp_d12_report.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    volume.commit()

    print("\nQuick test passed! d12 pipeline end-to-end verified.")
    print(f"  Checkpoints : nanochat_cache/base_checkpoints/a2mtp/  (3 dirs)")
    print(f"  Eval JSON   : {out_path}")
    print(f"  Report      : {report_path}")
    print("Ready to run the full d16 MTP experiments.")


# =============================================================================
# MAIN ENTRYPOINT — full a2_mtp pipeline
# =============================================================================


@app.function(
    image=figures_image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    cpu=4,
    timeout=60 * 30,  # 30 min should be plenty for API calls + plotting
)
def stage_make_sweep_figures() -> None:
    """CPU-only job: pull all 36 part2 sweep runs from W&B, produce:

    Figure 1 — 4-panel training-loss curves (one panel per config, 9 lines each,
                same 9 colours across all panels = LR × batch combo).
    Figure 2 — big 2-row horizontal bar chart of final-step train loss across
                all 36 runs; row 0 = baseline + mtp2 (18 bars),
                row 1 = mtp4 + mtp2_yarn (18 bars); colours repeat to mean the
                same LR/batch combo across quadrants.

    Both PNGs are saved to the shared Volume and logged as W&B images in the
    part2_sweep project.
    """
    import re
    import os
    import wandb
    import matplotlib
    matplotlib.use("Agg")                          # headless
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # ── resolve output directory ───────────────────────────────────────────────
    report_dir = os.path.join(NANOCHAT_CACHE, "report")
    os.makedirs(report_dir, exist_ok=True)

    # ── fetch runs from W&B ────────────────────────────────────────────────────
    api = wandb.Api(timeout=120)
    entity = ""
    try:
        entity = api.viewer()["entity"]
    except Exception:
        pass
    if not entity:
        entity = os.environ.get("WANDB_ENTITY", "")
    if not entity:
        # Last resort: parse from WANDB_API_KEY env context — use default entity
        try:
            entity = api.default_entity
        except Exception:
            pass
    project_path = f"{entity}/{WANDB_PROJECT_SWEEP}" if entity else WANDB_PROJECT_SWEEP
    print(f"Fetching runs from: {project_path}  (entity={entity!r})")
    all_runs = api.runs(project_path)

    # ── layout constants ───────────────────────────────────────────────────────
    CONFIGS    = ["baseline", "mtp2", "mtp4", "mtp2_yarn"]
    CONFIG_LABELS = {
        "baseline":  "Baseline (k=0, RoPE)",
        "mtp2":      "MTP-2 (k=2, RoPE)",
        "mtp4":      "MTP-4 (k=4, RoPE)",
        "mtp2_yarn": "MTP-2 + YaRN",
    }
    LRS  = [0.01, 0.02, 0.04]
    BSS  = ["131k", "262k", "524k"]
    COMBO_LABELS = [f"lr={lr}, bs={bs}" for lr in LRS for bs in BSS]

    # 9 visually distinct colours — same mapping across ALL figures
    PALETTE = plt.cm.tab10(np.arange(9) / 10.0)
    COMBO_COLOR = {c: PALETTE[i] for i, c in enumerate(COMBO_LABELS)}

    # ── parse runs into structured data ───────────────────────────────────────
    # run name pattern: sweep_{config}_lr{lr}_bs{bs}k
    # "mtp2_yarn" contains an underscore so we anchor on "_lr" before a float
    _RE = re.compile(r"^sweep_(.+)_lr([\d.]+)_bs(\d+)k$")

    # histories[config][combo_label] = {"steps": [...], "loss": [...]}
    histories: dict[str, dict[str, dict]] = {c: {} for c in CONFIGS}
    # final_loss[config][combo_label] = float (loss at last logged step)
    final_loss: dict[str, dict[str, float]] = {c: {} for c in CONFIGS}
    # core_metric[config][combo_label] = float (last core_metric logged, if any)
    core_metric: dict[str, dict[str, float]] = {c: {} for c in CONFIGS}

    for run in all_runs:
        m = _RE.match(run.name)
        if not m:
            continue
        cfg, lr_str, bs_str = m.group(1), m.group(2), m.group(3)
        if cfg not in CONFIGS:
            continue
        combo = f"lr={float(lr_str)}, bs={bs_str}k"
        if combo not in COMBO_LABELS:
            continue

        try:
            # Fetch all columns — avoids key-name mismatches (e.g. "loss" vs "train/loss")
            rows = list(run.scan_history())
        except Exception as e:
            print(f"  WARNING — could not fetch history for {run.name}: {e}")
            continue

        # Detect which key holds the training loss
        _LOSS_KEYS = ["train/loss", "loss", "train_loss"]
        loss_key: str | None = None
        for row in rows[:10]:       # probe first few rows
            for k in _LOSS_KEYS:
                if row.get(k) is not None:
                    loss_key = k
                    break
            if loss_key:
                break

        steps, losses = [], []
        best_core: float | None = None
        for row in rows:
            if loss_key and row.get(loss_key) is not None:
                steps.append(row.get("_step", len(steps)))
                losses.append(float(row[loss_key]))
            cm = row.get("core_metric")
            if cm is not None:
                best_core = float(cm)

        if not losses:
            print(f"  WARNING — empty history for {run.name}, skipping")
            continue

        histories[cfg][combo] = {"steps": steps, "loss": losses}
        final_loss[cfg][combo] = losses[-1]
        if best_core is not None:
            core_metric[cfg][combo] = best_core
        print(f"  loaded {run.name}: {len(steps)} steps, "
              f"final_loss={losses[-1]:.4f}, core_metric={best_core}")

    # ── Figure 1: 4-panel loss curves ─────────────────────────────────────────
    fig1, axes = plt.subplots(1, 4, figsize=(24, 5), constrained_layout=True)
    fig1.suptitle("Part 2 Hyperparameter Sweep — Training Loss Curves (d16, 300 steps)",
                  fontsize=13, y=1.03)

    for ax, cfg in zip(axes, CONFIGS):
        for combo in COMBO_LABELS:
            if combo not in histories[cfg]:
                continue
            d = histories[cfg][combo]
            ax.plot(d["steps"], d["loss"],
                    color=COMBO_COLOR[combo], label=combo,
                    linewidth=1.4, alpha=0.85)
        ax.set_title(CONFIG_LABELS[cfg], fontsize=9, pad=4)
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("Train Loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    # Shared legend below all panels
    handles = [mpatches.Patch(color=COMBO_COLOR[c], label=c) for c in COMBO_LABELS]
    fig1.legend(handles=handles, loc="lower center", ncol=5,
                bbox_to_anchor=(0.5, -0.18), fontsize=8,
                title="Hyperparameter combo (LR, batch size)", title_fontsize=8)

    fig1_path = os.path.join(report_dir, "p2_sweep_loss_curves.png")
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {fig1_path}")

    # ── Figure 2: big 2-row bar chart — CORE metric across all sweep runs ─────
    # Y-axis = core_metric (aggregate CORE score, higher is better).
    # Falls back to final train loss (inverted label) if core_metric was not
    # logged for a run (e.g. sweep ran without periodic CORE eval).
    # Layout (2 rows × 18 bars):
    #   Row 0 → [baseline × 9 combos]  |  [mtp2 × 9 combos]
    #   Row 1 → [mtp4    × 9 combos]  |  [mtp2_yarn × 9 combos]
    # Same colour = same LR/batch combo across both rows/quadrants.

    use_core = any(bool(core_metric[c]) for c in CONFIGS)
    if use_core:
        bar_data   = core_metric
        y_label    = "CORE Metric ↑ (higher is better)"
        fig2_title = "Part 2 Sweep — Aggregate CORE Metric by Hyperparameter Config"
    else:
        bar_data   = final_loss
        y_label    = "Final Train Loss ↓ (lower is better)"
        fig2_title = "Part 2 Sweep — Final Training Loss at Step 300"
        print("NOTE: no core_metric logged for sweep runs — plotting final train loss instead")

    ROW_PAIRS = [["baseline", "mtp2"], ["mtp4", "mtp2_yarn"]]
    BAR_W     = 0.7
    GROUP_GAP = 2.0

    fig2, row_axes = plt.subplots(2, 1, figsize=(26, 9), constrained_layout=True)
    fig2.suptitle(fig2_title, fontsize=13, y=1.02)

    for ax, pair in zip(row_axes, ROW_PAIRS):
        xtick_pos, xtick_lbl, group_center, group_cfgs = [], [], [], []
        x = 0.0
        for cfg in pair:
            group_start = x
            for combo in COMBO_LABELS:
                val   = bar_data[cfg].get(combo)
                color = COMBO_COLOR[combo]
                if val is not None:
                    ax.bar(x, val, width=BAR_W, color=color,
                           edgecolor="white", linewidth=0.4, zorder=3)
                else:
                    ax.bar(x, 0, width=BAR_W, color=color, alpha=0.15,
                           edgecolor="grey", linewidth=0.4, zorder=3)
                lr_part = combo.split(",")[0]   # "lr=0.02"
                bs_part = combo.split(",")[1].strip()  # "bs=524k"
                xtick_pos.append(x)
                xtick_lbl.append(f"{lr_part}\n{bs_part}")
                x += 1.0
            group_center.append((group_start + x - 1) / 2)
            group_cfgs.append(cfg)
            x += GROUP_GAP

        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_lbl, fontsize=7)
        ax.set_ylabel(y_label, fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_xlim(-0.8, x - GROUP_GAP + 0.8)

        # Quadrant header labels (placed near the top of each group)
        ax.figure.canvas.draw()   # force axis limits to update
        ylo, yhi = ax.get_ylim()
        label_y  = ylo + (yhi - ylo) * 0.93
        for gc, cfg in zip(group_center, group_cfgs):
            ax.text(gc, label_y, CONFIG_LABELS[cfg],
                    ha="center", va="top", fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="lightyellow",
                              edgecolor="grey", alpha=0.85))

    handles2 = [mpatches.Patch(color=COMBO_COLOR[c], label=c) for c in COMBO_LABELS]
    fig2.legend(handles=handles2, loc="lower center", ncol=5,
                bbox_to_anchor=(0.5, -0.10), fontsize=8,
                title="Hyperparameter combo  (same colour = same combo across quadrants)",
                title_fontsize=8)

    fig2_path = os.path.join(report_dir, "p2_sweep_bar_chart.png")
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {fig2_path}")

    # ── commit to volume ──────────────────────────────────────────────────────
    volume.commit()

    # ── log figures to W&B ────────────────────────────────────────────────────
    with wandb.init(
        project=WANDB_PROJECT_SWEEP,
        entity=entity or None,
        job_type="figures",
        name="sweep_figures",
    ) as wrun:
        wrun.log({
            "sweep/loss_curves":   wandb.Image(fig1_path),
            "sweep/core_metric_bar": wandb.Image(fig2_path),
        })
    print("Figures logged to W&B ✓")


@app.local_entrypoint()
def main() -> None:
    """
    Full a2_mtp pipeline:
      1. d12 smoke test (validates MTP patches and code paths cheaply)
      2. Train baseline, MTP-2, MTP-4 in parallel (3 separate Modal containers)
      3. Eval + Report on all 3 checkpoints
    """
    w = 64
    print("\n" + "=" * w)
    print("a2_mtp: Meta-Style Multi-Token Prediction (Weight-Tied)")
    print(f"  depth={DEPTH}  seq=2048  steps={N_TOTAL_STEPS}")
    print(f"  chinchilla_tokens={CHINCHILLA_TOKENS / 1e9:.2f}B  batch={TOTAL_BATCH_SIZE}")
    print("=" * w + "\n")

    # Step 1: Smoke test (validates mtp_k, patching, eval end-to-end)
    print("[0/3] d12 smoke test (baseline + mtp2 + mtp4)...")
    quick_test_d12.remote()

    # Step 2: All 4 d16 training runs in parallel
    print("[1/3] Training baseline, MTP-2, MTP-4, MTP-2+YaRN in parallel...")
    baseline_h    = stage_train_baseline.spawn()
    mtp2_h        = stage_train_mtp2.spawn()
    mtp4_h        = stage_train_mtp4.spawn()
    mtp2_yarn_h   = stage_train_mtp2_yarn.spawn()
    baseline_h.get()
    mtp2_h.get()
    mtp4_h.get()
    mtp2_yarn_h.get()

    # Step 3: CORE eval + report
    print("[2/3] CORE eval + markdown report...")
    stage_eval_and_report.remote()

    print("\n" + "=" * w)
    print("a2_mtp pipeline complete!")
    print("=" * w)
