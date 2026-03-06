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
