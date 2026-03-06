"""
part2_arch: Architecture Ablations — SwiGLU and YaRN
=====================================================

Three training experiments, all at d16, seq=2048, full Chinchilla budget:
  1. Baseline — standard ReLU² MLP + standard RoPE
  2. SwiGLU   — SwiGLU gated MLP (parameter-equivalent), RoPE unchanged
  3. YaRN     — standard ReLU² MLP + YaRN NTK-by-Parts RoPE (scale=8)

All runs start from scratch, same hyperparameters, same token budget.
Only CORE benchmark evaluation is performed.

Usage
-----
Individual stages:
    modal run nanochat_modal.py::quick_test_d12
    modal run nanochat_modal.py::stage_train_baseline
    modal run nanochat_modal.py::stage_train_swiglu
    modal run nanochat_modal.py::stage_train_yarn
    modal run nanochat_modal.py::stage_eval_and_report

Full pipeline (smoke test → all 3 trains in parallel → eval):
    modal run nanochat_modal.py

Cost reference (8×H100 ~$31/hr)
---------------------------------
    d12 quick test    : ~15 min   (~$8)
    d16 baseline      : ~60 min   (~$31)
    d16 swiglu        : ~60 min   (~$31)  (parallel with baseline & yarn)
    d16 yarn          : ~60 min   (~$31)  (parallel with baseline & swiglu)
    eval (3 ckpts)    : ~30 min   (~$8, 4×H100)
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

# Chinchilla-optimal token budget for d16 (same as part2_mtp)
CHINCHILLA_TOKENS = 2_460_000_000
N_TOTAL_STEPS = CHINCHILLA_TOKENS // TOTAL_BATCH_SIZE  # ≈ 4693

# d12 quick-test step counts
N_D12_STEPS = 200

# Model tags (subdirectories under base_checkpoints/)
TAG_BASELINE = "a2arch/d16_baseline"
TAG_SWIGLU = "a2arch/d16_swiglu"
TAG_YARN = "a2arch/d16_yarn"

TAG_D12_BASELINE = "a2arch/d12_baseline"
TAG_D12_SWIGLU = "a2arch/d12_swiglu"
TAG_D12_YARN = "a2arch/d12_yarn"

WANDB_PROJECT = "part2_arch"
WANDB_RUN_BASELINE = "d16_baseline"
WANDB_RUN_SWIGLU = "d16_swiglu"
WANDB_RUN_YARN = "d16_yarn"

# YaRN scale factor for the yarn experiment
YARN_SCALE = 8.0

# Timeouts
TIMEOUT_TRAIN = 60 * 60 * 2  # 2 h per training run
TIMEOUT_EVAL = 60 * 60 * 2  # 2 h for eval on 3 checkpoints
TIMEOUT_QUICKTEST = 60 * 60 * 1  # 1 h

# Volume / cache paths
VOLUME_MOUNT = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"
BASE_DIR = "/data/.cache/nanochat"

# =============================================================================
# MODAL PRIMITIVES
# =============================================================================

app = modal.App("nanochat-part2-arch")
volume = Volume.from_name("nanochat-vol", create_if_missing=True)
secret = Secret.from_name("nanochat-secrets")

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
    # Apply patches: overwrite nanochat files with our arch-patched versions
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


def _python(
    module: str, args: list | None = None, *, cwd: str = "/root/nanochat"
) -> None:
    args = args or []
    _run(
        f"cd {cwd} && PYTHONPATH=/root/nanochat:$PYTHONPATH "
        f"uv run python -m {module} {' '.join(args)}"
    )


def _torchrun(module: str, args: list | None = None, *, nproc: int) -> None:
    args = args or []
    args_str = (" -- " + " ".join(args)) if args else ""
    cmd = (
        f"cd /root/nanochat && "
        f"PYTHONPATH=/root/nanochat:$PYTHONPATH "
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m {module}{args_str}"
    )
    _run(cmd)


def _setup_cache() -> None:
    os.makedirs(NANOCHAT_CACHE, exist_ok=True)
    if not os.path.lexists(BASE_DIR):
        os.makedirs("/data/.cache/", exist_ok=True)
        os.symlink(NANOCHAT_CACHE, BASE_DIR)
        print(f"Symlinked {BASE_DIR} -> {NANOCHAT_CACHE}")
    else:
        print(f"Cache symlink already exists: {BASE_DIR}")


def _find_last_step(model_tag: str) -> int:
    import glob

    ckpt_dir = os.path.join(BASE_DIR, "base_checkpoints", model_tag)
    files = glob.glob(os.path.join(ckpt_dir, "model_*.pt"))
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return max(int(os.path.basename(f).split("_")[1].split(".")[0]) for f in files)


def _collect_core_csv(tag: str) -> dict:
    """
    Read the CORE CSV written by scripts.base_eval and return
    { "core_metric": float | None, "tasks": { task_name: score, ... } }
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
    tag_swiglu: str,
    tag_yarn: str,
    depth: int,
    n_steps: int,
    chinchilla_tokens: int,
    total_batch_size: int,
    device_batch: int,
    n_gpus: int,
    title_suffix: str = "",
) -> str:
    """Build the part2_arch markdown report as a string."""
    lines = []

    def section(title: str) -> None:
        lines.append(f"\n## {title}\n")

    def table(headers: list, rows: list) -> None:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        lines.append("")

    n_embd = depth * 64
    n_head = n_embd // 128
    approx_params_m = round(12 * depth * n_embd**2 / 1e6)
    grad_accum = total_batch_size // (device_batch * 2048 * n_gpus)

    title_part = f" — {title_suffix}" if title_suffix else ""
    lines.append(f"# part2_arch: Architecture Ablations (SwiGLU & YaRN){title_part}\n")

    # ── 1. Architecture Variants ───────────────────────────────────────────────
    section("1. Architecture Variants")
    table(
        ["Parameter", "Baseline", "SwiGLU", "YaRN"],
        [
            ["depth (n_layer)", str(depth)] * 3 + [str(depth)],
            ["n_embd", str(n_embd)] * 3 + [str(n_embd)],
            ["approx params", f"~{approx_params_m}M"] * 3 + [f"~{approx_params_m}M"],
            ["activation", "ReLU²", "SwiGLU", "ReLU²"],
            ["positional enc.", "RoPE", "RoPE", "YaRN (scale=8)"],
            ["extra params", "—", "~0%", "—"],
        ],
    )

    # ── 2. Training Configuration ──────────────────────────────────────────────
    section("2. Training Configuration")
    table(
        ["Parameter", "Value"],
        [
            ["seq_len", "2048"],
            ["steps", str(n_steps)],
            ["tokens", f"{chinchilla_tokens / 1e9:.3f}B"],
            ["total_batch_size (tokens)", str(total_batch_size)],
            ["device_batch_size", str(device_batch)],
            ["grad_accum_steps", str(grad_accum)],
            ["GPUs", f"{n_gpus}×H100"],
            ["optimizer (matrix params)", "Muon"],
            ["optimizer (embeddings)", "AdamW"],
            ["WandB project", WANDB_PROJECT],
        ],
    )

    # ── 3. CORE Results ────────────────────────────────────────────────────────
    section("3. CORE Results")
    result_rows = []
    for label, tag in [
        ("Baseline", tag_baseline),
        ("SwiGLU", tag_swiglu),
        ("YaRN", tag_yarn),
    ]:
        r = results.get("training", {}).get(tag, {})
        result_rows.append(
            [
                label,
                f"{r['core_metric']:.4f}"
                if isinstance(r.get("core_metric"), float)
                else "N/A",
            ]
        )
    table(["Run", "CORE score ↑"], result_rows)

    # ── 4. CORE Per-Task Breakdown ─────────────────────────────────────────────
    section("4. CORE Per-Task Breakdown")
    core_tasks = results.get("core_tasks", {})
    all_tasks = sorted({t for tag_results in core_tasks.values() for t in tag_results})
    if all_tasks:
        core_headers = ["Task", "Baseline", "SwiGLU", "YaRN"]
        core_rows = []
        for task in all_tasks:
            row = [task]
            for tag in [tag_baseline, tag_swiglu, tag_yarn]:
                val = core_tasks.get(tag, {}).get(task, "N/A")
                row.append(f"{val:.4f}" if isinstance(val, float) else "N/A")
            core_rows.append(row)
        core_rows.append(
            ["**CORE aggregate**"]
            + [
                f"**{results['training'][tag]['core_metric']:.4f}**"
                if isinstance(
                    results.get("training", {}).get(tag, {}).get("core_metric"), float
                )
                else "N/A"
                for tag in [tag_baseline, tag_swiglu, tag_yarn]
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
    activation: str,
    rope_type: str,
    yarn_scale: float = YARN_SCALE,
    tag: str,
    wandb_run: str,
    n_steps: int,
    device_batch: int = DEVICE_BATCH,
    nproc: int = _N_TRAIN_GPUS,
    core_metric_every: int = 1000,
) -> None:
    """Shared training logic for all three architecture experiments."""
    _setup_cache()
    print(
        f"Training: depth={depth} seq=2048 activation={activation} "
        f"rope_type={rope_type} steps={n_steps} tag={tag}"
    )
    _torchrun(
        "scripts.base_train",
        [
            f"--depth={depth}",
            "--max-seq-len=2048",
            f"--model-tag={tag}",
            f"--activation={activation}",
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
    """Baseline: d16, seq=2048, ReLU², standard RoPE."""
    _train_run(
        depth=depth,
        activation="relu2",
        rope_type="rope",
        tag=TAG_BASELINE,
        wandb_run=WANDB_RUN_BASELINE,
        n_steps=n_steps,
    )


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_TRAIN,
)
def stage_train_swiglu(depth: int = DEPTH, n_steps: int = N_TOTAL_STEPS) -> None:
    """SwiGLU: d16, seq=2048, SwiGLU MLP (parameter-equivalent), standard RoPE."""
    _train_run(
        depth=depth,
        activation="swiglu",
        rope_type="rope",
        tag=TAG_SWIGLU,
        wandb_run=WANDB_RUN_SWIGLU,
        n_steps=n_steps,
    )


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_TRAIN,
)
def stage_train_yarn(depth: int = DEPTH, n_steps: int = N_TOTAL_STEPS) -> None:
    """YaRN: d16, seq=2048, ReLU², YaRN NTK-by-Parts RoPE (scale=8)."""
    _train_run(
        depth=depth,
        activation="relu2",
        rope_type="yarn",
        yarn_scale=YARN_SCALE,
        tag=TAG_YARN,
        wandb_run=WANDB_RUN_YARN,
        n_steps=n_steps,
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
    CORE evaluation on all 3 checkpoints → results JSON → markdown report.

      A. Download eval bundle (if not cached)
      B. CORE eval (scripts.base_eval --eval=core,bpb) on baseline, swiglu, yarn
      C. Merge CORE CSV results → JSON
      D. Build and write markdown report

    Writes to:
      nanochat_cache/a2arch_eval_results.json
      nanochat_cache/report/a2arch_report.md
    """
    volume.reload()
    _setup_cache()

    eval_bundle_dir = os.path.join(NANOCHAT_CACHE, "eval_bundle")
    if not os.path.isdir(eval_bundle_dir):
        print("Downloading eval bundle (~1GB)...")
        zip_path = "/tmp/eval_bundle.zip"
        _run(
            f"curl -L -o {zip_path} https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
        )
        _run(f"unzip -q {zip_path} -d {NANOCHAT_CACHE} && rm {zip_path}")
        volume.commit()

    tags = [TAG_BASELINE, TAG_SWIGLU, TAG_YARN]
    core_data: dict = {}

    # ── A. CORE eval on all three checkpoints ─────────────────────────────────
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
            "val_bpb": None,
        }
        if d.get("tasks"):
            results["core_tasks"][tag] = d["tasks"]

    results_path = os.path.join(NANOCHAT_CACHE, "a2arch_eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()

    # ── C. Build and write markdown report ────────────────────────────────────
    print(f"\n{'=' * 60}\nGenerating report\n{'=' * 60}")
    report_md = _build_report_markdown(
        results,
        tag_baseline=TAG_BASELINE,
        tag_swiglu=TAG_SWIGLU,
        tag_yarn=TAG_YARN,
        depth=DEPTH,
        n_steps=N_TOTAL_STEPS,
        chinchilla_tokens=CHINCHILLA_TOKENS,
        total_batch_size=TOTAL_BATCH_SIZE,
        device_batch=DEVICE_BATCH,
        n_gpus=_N_TRAIN_GPUS,
    )

    report_dir = os.path.join(NANOCHAT_CACHE, "report")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "a2arch_report.md")
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
    Smoke test for the full part2_arch pipeline at d12 scale:
      - Baseline: d12, seq=2048, relu2, rope, 200 steps
      - SwiGLU:   d12, seq=2048, swiglu, rope, 200 steps
      - YaRN:     d12, seq=2048, relu2, yarn (scale=8), 200 steps
      - Eval:     CORE on all 3 d12 checkpoints
    """
    _setup_cache()
    nproc = 4
    bs = 8
    total_bs = 65536
    n_steps = N_D12_STEPS

    common_args = [
        "--depth=12",
        "--max-seq-len=2048",
        f"--device-batch-size={bs}",
        f"--total-batch-size={total_bs}",
        f"--num-iterations={n_steps}",
        "--save-every=100",
        "--core-metric-every=999999",
        "--sample-every=-1",
        f"--wandb-project={WANDB_PROJECT}",
    ]

    # --- Baseline ---
    print("=== Quick test: Baseline (d12, relu2, rope, 200 steps) ===")
    _torchrun(
        "scripts.base_train",
        common_args
        + [
            f"--model-tag={TAG_D12_BASELINE}",
            "--activation=relu2",
            "--rope-type=rope",
            "--run=d12_baseline",
        ],
        nproc=nproc,
    )
    volume.commit()

    # --- SwiGLU ---
    print("=== Quick test: SwiGLU (d12, swiglu, rope, 200 steps) ===")
    _torchrun(
        "scripts.base_train",
        common_args
        + [
            f"--model-tag={TAG_D12_SWIGLU}",
            "--activation=swiglu",
            "--rope-type=rope",
            "--run=d12_swiglu",
        ],
        nproc=nproc,
    )
    volume.commit()

    # --- YaRN ---
    print("=== Quick test: YaRN (d12, relu2, yarn, 200 steps) ===")
    _torchrun(
        "scripts.base_train",
        common_args
        + [
            f"--model-tag={TAG_D12_YARN}",
            "--activation=relu2",
            "--rope-type=yarn",
            f"--yarn-scale={YARN_SCALE}",
            "--run=d12_yarn",
        ],
        nproc=nproc,
    )
    volume.commit()

    # --- Download eval bundle ---
    eval_bundle_dir = os.path.join(NANOCHAT_CACHE, "eval_bundle")
    if not os.path.isdir(eval_bundle_dir):
        print("Downloading eval bundle (~1GB)...")
        zip_path = "/tmp/eval_bundle.zip"
        _run(
            f"curl -L -o {zip_path} https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
        )
        _run(f"unzip -q {zip_path} -d {NANOCHAT_CACHE} && rm {zip_path}")
        volume.commit()

    # --- CORE eval on all 3 d12 checkpoints ---
    d12_tags = [TAG_D12_BASELINE, TAG_D12_SWIGLU, TAG_D12_YARN]
    core_data: dict = {}
    for tag in d12_tags:
        print(f"\n=== Quick test CORE eval: {tag} ===")
        _torchrun(
            "scripts.base_eval",
            [
                f"--model-tag={tag}",
                "--eval=core",
                "--max-per-task=50",
                "--device-batch-size=4",
            ],
            nproc=nproc,
        )
        core_data[tag] = _collect_core_csv(tag)

    # --- Build results and report ---
    results: dict = {"training": {}, "core_tasks": {}}
    for tag in d12_tags:
        d = core_data.get(tag, {})
        results["training"][tag] = {
            "core_metric": d.get("core_metric"),
            "val_bpb": None,
        }
        if d.get("tasks"):
            results["core_tasks"][tag] = d["tasks"]

    out_path = os.path.join(NANOCHAT_CACHE, "a2arch_d12_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    report_md = _build_report_markdown(
        results,
        tag_baseline=TAG_D12_BASELINE,
        tag_swiglu=TAG_D12_SWIGLU,
        tag_yarn=TAG_D12_YARN,
        depth=12,
        n_steps=n_steps,
        chinchilla_tokens=CHINCHILLA_TOKENS,
        total_batch_size=total_bs,
        device_batch=bs,
        n_gpus=nproc,
        title_suffix="d12 smoke test",
    )

    report_dir = os.path.join(NANOCHAT_CACHE, "report")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "a2arch_d12_report.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    volume.commit()

    print("\nQuick test passed! d12 pipeline end-to-end verified.")
    print(f"  Checkpoints : nanochat_cache/base_checkpoints/a2arch/  (3 dirs)")
    print(f"  Eval JSON   : {out_path}")
    print(f"  Report      : {report_path}")
    print("Ready to run the full d16 architecture experiments.")


# =============================================================================
# MAIN ENTRYPOINT — full part2_arch pipeline
# =============================================================================


@app.local_entrypoint()
def main() -> None:
    """
    Full part2_arch pipeline:
      1. d12 smoke test (validates patches and code paths cheaply)
      2. Train baseline, SwiGLU, YaRN in parallel (3 separate Modal containers)
      3. Eval + Report on all 3 checkpoints
    """
    w = 64
    print("\n" + "=" * w)
    print("part2_arch: Architecture Ablations (SwiGLU & YaRN)")
    print(f"  depth={DEPTH}  seq=2048  steps={N_TOTAL_STEPS}")
    print(
        f"  chinchilla_tokens={CHINCHILLA_TOKENS / 1e9:.2f}B  batch={TOTAL_BATCH_SIZE}"
    )
    print("=" * w + "\n")

    # Step 1: Smoke test
    print("[0/3] d12 smoke test (baseline + swiglu + yarn)...")
    quick_test_d12.remote()

    # Step 2: All 3 d16 training runs in parallel
    print("[1/3] Training baseline, SwiGLU, YaRN in parallel...")
    baseline_h = stage_train_baseline.spawn()
    swiglu_h = stage_train_swiglu.spawn()
    yarn_h = stage_train_yarn.spawn()
    baseline_h.get()
    swiglu_h.get()
    yarn_h.get()

    # Step 3: CORE eval + report
    print("[2/3] CORE eval + markdown report...")
    stage_eval_and_report.remote()

    print("\n" + "=" * w)
    print("part2_arch pipeline complete!")
    print("=" * w)
