"""
Part 3: Picochat Context Length Curriculum (d16)
=================================================

Three training experiments:
  1. Phase 1  — picochat (d16) at seq=512,  40% of Chinchilla token budget
  2. Phase 2  — continue from Phase 1 at seq=2048, remaining 60% of budget
  3. Baseline — picochat (d16) at seq=2048,  full Chinchilla budget from scratch

Phase 2 and Baseline run in parallel after Phase 1.
The d20 runs are preserved as Part 4 nanochat baselines.

All checkpoints land in nanochat_cache/base_checkpoints/part3/ on the volume.

Usage
-----
Sweep (justify curriculum choices):
    modal run nanochat_modal.py::stage_sweep_p3_s256 &
    modal run nanochat_modal.py::stage_sweep_p3_s512 &

Full pipeline (d12 smoke-test, then d16):
    modal run nanochat_modal.py

Individual stages:
    modal run nanochat_modal.py::quick_test_d12
    modal run nanochat_modal.py::stage_pretrain_phase1
    modal run nanochat_modal.py::stage_pretrain_phase2
    modal run nanochat_modal.py::stage_pretrain_baseline
    modal run nanochat_modal.py::stage_eval
    modal run nanochat_modal.py::stage_report

Cost reference (8×H100 ~$31/hr)
---------------------------------
    d12 quick test    : ~15 min   (~$8)
    sweep (2×H100:4)  : ~30 min   (~$8)   ]  run in parallel
    d16 Phase 1       : ~30 min   (~$16)
    d16 Phase 2       : ~45 min   (~$23)  ]  run in parallel
    d16 Baseline      : ~60 min   (~$31)  ]
    eval              : ~30 min   (~$8,  H100:4)
"""

import os
import json
import subprocess
import modal
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# CONFIGURATION
# =============================================================================

DEPTH = 16  # picochat = d16 (~234M scaling params)
GPU_TRAIN = "H100:8"
GPU_EVAL = "H100:4"

# Device batch sizes
DEVICE_BATCH_PHASE1 = 32  # seq=512
DEVICE_BATCH_PHASE2 = 32  # seq=2048, d16 comfortably fits 32/H100
DEVICE_BATCH_BASELINE = 32

TOTAL_BATCH_SIZE = 524288
CHINCHILLA_TOKENS = 2_460_000_000  # ≈ 10.5 × 234M scaling params
PHASE1_FRAC = 0.40  # 40% at seq=512  (chosen from sweep)
PHASE2_FRAC = 0.60  # 60% at seq=2048 (warm-started)

N_TOTAL_STEPS = CHINCHILLA_TOKENS // TOTAL_BATCH_SIZE  # ≈ 4693
N_PHASE1_STEPS = int(N_TOTAL_STEPS * PHASE1_FRAC)  # ≈ 1877
N_PHASE2_STEPS = N_TOTAL_STEPS - N_PHASE1_STEPS  # ≈ 2816
N_BASELINE_STEPS = N_TOTAL_STEPS  # full budget ≈ 4693

# d12 quick-test step counts (just enough to exercise all code paths)
N_D12_PHASE1_STEPS = 300
N_D12_PHASE2_STEPS = 300
N_D12_BASELINE_STEPS = 300

# Model tags (become subdirectories under base_checkpoints/)
TAG_PHASE1 = "part3/d16_ctx512"
TAG_PHASE2 = "part3/d16_ctx2048"
# Reuse the d16 baseline from Part 2 (same model, no need to re-train)
TAG_BASELINE = "a2mtp/d16_baseline"

TAG_D12_PHASE1 = "part3/d12_ctx512"
TAG_D12_PHASE2 = "part3/d12_ctx2048"
TAG_D12_BASELINE = "part3/d12_baseline"

WANDB_PROJECT = "nanochat-part3"
WANDB_RUN_PHASE1 = "p3_d16_phase1"
WANDB_RUN_PHASE2 = "p3_d16_phase2"
WANDB_RUN_BASELINE = "d16_baseline"  # Part 2 run name in part2_mtp project

# Timeouts (d16 is roughly half the cost of d20)
TIMEOUT_PHASE1 = 60 * 60 * 1  # 1 h
TIMEOUT_PHASE2 = 60 * 60 * 2  # 2 h
TIMEOUT_BASELINE = 60 * 60 * 2  # 2 h
TIMEOUT_EVAL = 60 * 60 * 2  # 2 h
TIMEOUT_QUICKTEST = 60 * 60 * 1  # 1 h

# =============================================================================
# HYPERPARAMETER SWEEP — 6 curriculum configs, H100:4, 300 steps per phase
# =============================================================================

# Design choices to sweep:
#   phase1_seq  : sequence length during Phase 1 (256 or 512)
#   phase1_frac : fraction of total budget spent in Phase 1 (0.2 / 0.4 / 0.6)
# Each combo runs 300 phase-1 steps then 300 phase-2 steps at seq=2048.
# Baseline (seq=2048 from scratch) is reused from Part 2 — not re-swept here.

SWEEP_P3_SEQS = [256, 512]
SWEEP_P3_FRACS = [0.2, 0.4, 0.6]
SWEEP_P3_STEPS = 300  # steps per phase within each sweep run
GPU_SWEEP_P3 = "H100:2"
DEVICE_BATCH_SWEEP_P3 = 16  # conservative; works for both seq=256 and seq=2048
TIMEOUT_SWEEP_P3 = 60 * 60 * 3  # 3 h for 3 combos × 2 phases sequentially
WANDB_PROJECT_SWEEP_P3 = "part3_sweep"

# Volume / cache paths (same volume as the parent nanochat_modal.py)
VOLUME_MOUNT = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"
BASE_DIR = "/data/.cache/nanochat"

# =============================================================================
# MODAL PRIMITIVES
# =============================================================================

app = modal.App("nanochat-part3")
volume = Volume.from_name("nanochat-vol", create_if_missing=True)
secret = Secret.from_name("nanochat-secrets")

image = (
    ModalImage.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git", "build-essential", "curl", "wget", "unzip")
    # nanochat repo lives at a3/nanochat/ — resolve absolute path from this file
    # so the path is correct regardless of where `modal run` is invoked from.
    .add_local_dir(
        local_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "nanochat"
        ),
        remote_path="/root/nanochat",
        copy=True,
    )
    # part3/ scripts (eval_longctx.py etc.) live alongside this file at a3/part3/
    .add_local_dir(
        local_path=os.path.dirname(os.path.abspath(__file__)),
        remote_path="/root/part3",
        copy=True,
    )
    # Apply Part 3 patches — overwrite the unmodified nanochat submodule scripts with
    # our patched versions (kept in part3/patches/) so the submodule stays clean.
    .add_local_file(
        local_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "patches", "base_train.py"
        ),
        remote_path="/root/nanochat/scripts/base_train.py",
        copy=True,
    )
    .add_local_file(
        local_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "patches", "base_eval.py"
        ),
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

# Lightweight CPU-only image for figure generation (no CUDA needed).
figures_image = ModalImage.debian_slim(python_version="3.11").pip_install(
    "wandb>=0.18", "matplotlib>=3.9", "numpy>=1.26"
)


def _run(cmd: str) -> None:
    """Shell out to bash, stream stdout/stderr, and raise on failure."""
    print(f"\n>>>  {cmd}\n")
    result = subprocess.run(["bash", "-c", cmd], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command exited with code {result.returncode}:\n  {cmd}")


def _python(
    module: str, args: list | None = None, *, cwd: str = "/root/nanochat"
) -> None:
    """Run `uv run python -m {module} [args]` — for non-distributed scripts."""
    args = args or []
    # PYTHONPATH ensures both nanochat (installed package) and /root (for part3) are importable.
    _run(
        f"cd {cwd} && PYTHONPATH=/root:/root/nanochat:$PYTHONPATH uv run python -m {module} {' '.join(args)}"
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
    print(cmd)
    _run(cmd)


def _setup_cache() -> None:
    """Create cache directories and symlink BASE_DIR → NANOCHAT_CACHE on the volume."""
    import os

    os.makedirs(NANOCHAT_CACHE, exist_ok=True)
    if not os.path.lexists(BASE_DIR):
        os.makedirs("/data/.cache/", exist_ok=True)
        os.symlink(NANOCHAT_CACHE, BASE_DIR)
        print(f"Symlinked {BASE_DIR} -> {NANOCHAT_CACHE}")
    else:
        print(f"Cache symlink already exists: {BASE_DIR}")


def _collect_core_csv(tag: str) -> dict:
    """
    After running scripts.base_eval for a given model tag, read the CSV it wrote and
    return a dict:
      {
        "val_bpb":    float | None,
        "core_metric": float | None,
        "tasks": { task_name: centered_score, ... }
      }
    base_eval writes to: {NANOCHAT_CACHE}/base_eval/{tag_slug}_{step:06d}.csv
    The step is discovered via _find_last_step.
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
            if name == "Task":  # header
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


def _merge_core_into_results(results: dict, tags: list, core_data: dict) -> None:
    """
    Merge the CORE CSV data (from _collect_core_csv) into the eval results dict
    in-place, under the keys used by _build_report_markdown:
      results["training"][tag]["core_metric"]
      results["core_tasks"][tag][task_name]
    """
    results.setdefault("training", {})
    results.setdefault("core_tasks", {})
    for tag in tags:
        d = core_data.get(tag, {})
        results["training"].setdefault(tag, {})
        if d.get("core_metric") is not None:
            results["training"][tag]["core_metric"] = d["core_metric"]
        if d.get("tasks"):
            results["core_tasks"][tag] = d["tasks"]


def _build_report_markdown(
    results: dict,
    *,
    tag_p1: str,
    tag_p2: str,
    tag_baseline: str,
    depth: int,
    n_p1_steps: int,
    n_p2_steps: int,
    n_baseline_steps: int,
    chinchilla_tokens: int,
    total_batch_size: int,
    device_batch_p1: int,
    device_batch_p2: int,
    device_batch_baseline: int,
    n_gpus: int,
    wandb_run_p1: str,
    wandb_run_p2: str,
    wandb_run_baseline: str,
    title_suffix: str = "",
) -> str:
    """
    Pure-Python helper that builds the Part 3 markdown report as a string.

    Outputs ONLY objective tables — no prose commentary or discussion.
    Called by both `stage_eval_and_report` (d20) and quick-test helpers (d12).
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

    # Derived model architecture values (nanochat defaults: aspect_ratio=64, head_dim=128)
    n_embd = depth * 64
    n_head = n_embd // 128
    n_kv_head = n_head  # GQA with kv_heads == q_heads (default)
    # Approx parameter count: embedding + 20 layers*(attn+mlp) + lm_head
    # rough formula used in nanochat: ~12 * n_layer * n_embd^2 / 1e6 M
    approx_params_m = round(12 * depth * n_embd**2 / 1e6)

    # Gradient accumulation steps per run
    def grad_accum(dev_bs: int, seq: int) -> int:
        return total_batch_size // (dev_bs * seq * n_gpus)

    title_part = f" — {title_suffix}" if title_suffix else ""
    lines.append(f"# Part 3: Picochat Context Length Curriculum{title_part}\n")

    # ── 1. Model Architecture (shared across all runs) ─────────────────────────
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
            ["position encoding", "RoPE (Rotary Position Embedding)"],
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
            ["warmdown_ratio", "0.5  (cosine)"],
            ["final_lr_frac", "0.0"],
        ],
    )

    # ── 2. Training Configuration (per-run) ───────────────────────────────────
    section("2. Training Configuration")
    table(
        ["Parameter", "Phase 1 (ckpt1)", "Phase 2 (ckpt2)", "Baseline (ckpt3)"],
        [
            ["seq_len", "512", "2048", "2048"],
            ["warm-start from", "—", "Phase 1 ckpt (weights only)", "—"],
            ["steps", str(n_p1_steps), str(n_p2_steps), str(n_baseline_steps)],
            [
                "tokens",
                f"{int(chinchilla_tokens * 0.4) / 1e9:.3f}B",
                f"{int(chinchilla_tokens * 0.6) / 1e9:.3f}B",
                f"{chinchilla_tokens / 1e9:.3f}B",
            ],
            [
                "total_batch_size (tokens)",
                str(total_batch_size),
                str(total_batch_size),
                str(total_batch_size),
            ],
            [
                "device_batch_size",
                str(device_batch_p1),
                str(device_batch_p2),
                str(device_batch_baseline),
            ],
            [
                "grad_accum_steps",
                str(grad_accum(device_batch_p1, 512)),
                str(grad_accum(device_batch_p2, 2048)),
                str(grad_accum(device_batch_baseline, 2048)),
            ],
            ["GPUs", f"{n_gpus}×H100", f"{n_gpus}×H100", f"{n_gpus}×H100"],
            ["WandB run", wandb_run_p1, wandb_run_p2, wandb_run_baseline],
        ],
    )

    # ── 3. Training Metrics ────────────────────────────────────────────────────
    section("3. Training Metrics")
    training_rows = []
    for label_run, tag in [
        ("Phase 1 (ckpt1)", tag_p1),
        ("Phase 2 (ckpt2)", tag_p2),
        ("Baseline (ckpt3)", tag_baseline),
    ]:
        r = results.get("training", {}).get(tag, {})
        training_rows.append(
            [
                label_run,
                f"{r['val_bpb']:.6f}" if isinstance(r.get("val_bpb"), float) else "N/A",
                f"{r['core_metric']:.4f}"
                if isinstance(r.get("core_metric"), float)
                else "N/A",
                f"{r['training_time_min']}"
                if r.get("training_time_min") is not None
                else "N/A",
            ]
        )
    table(["Run", "Final val BPB ↓", "CORE score ↑", "Train time (min)"], training_rows)

    # ── 4. Needle-in-Haystack Accuracy ────────────────────────────────────────
    section("4. Needle-in-Haystack Accuracy")
    distances = results.get("needle", {}).get("distances", [])
    nih_headers = [
        "Distance P (tokens)",
        "Phase 1 ctx=512",
        "Phase 2 ctx=2048",
        "Baseline ctx=2048",
    ]
    nih_rows = []
    for d in distances:
        row = [str(d)]
        for tag in [tag_p1, tag_p2, tag_baseline]:
            acc = results.get("needle", {}).get(tag, {}).get(str(d), "N/A")
            row.append(f"{acc:.4f}" if isinstance(acc, float) else "N/A")
        nih_rows.append(row)
    table(nih_headers, nih_rows)

    # ── 5. BPB by Context Position ────────────────────────────────────────────
    section("5. BPB by Context Position")
    seg_meta = [
        ("seg0", "0–511", "0"),
        ("seg1", "512–1023", "512"),
        ("seg2", "1024–1535", "512"),
        ("seg3", "1536–2047", "512"),
    ]
    bpb_headers = [
        "Segment",
        "Token range",
        "Prior ctx (ckpt1)",
        "Phase 1 BPB",
        "Phase 2 BPB",
        "Baseline BPB",
    ]
    bpb_rows = []
    for seg_key, tok_range, ctx1 in seg_meta:
        row = [seg_key, tok_range, ctx1]
        for tag in [tag_p1, tag_p2, tag_baseline]:
            val = results.get("bpb_by_position", {}).get(tag, {}).get(seg_key, "N/A")
            row.append(f"{val:.6f}" if isinstance(val, float) else "N/A")
        bpb_rows.append(row)
    table(bpb_headers, bpb_rows)

    # ── 6. CORE Per-Task Breakdown ────────────────────────────────────────────
    section("6. CORE Per-Task Breakdown")
    core_tasks = results.get("core_tasks", {})
    all_tasks = sorted({t for tag_results in core_tasks.values() for t in tag_results})
    if all_tasks:
        core_headers = ["Task", "Phase 1", "Phase 2", "Baseline"]
        core_rows = []
        for task in all_tasks:
            row = [task]
            for tag in [tag_p1, tag_p2, tag_baseline]:
                val = core_tasks.get(tag, {}).get(task, "N/A")
                row.append(f"{val:.4f}" if isinstance(val, float) else "N/A")
            core_rows.append(row)
        # Aggregate CORE row at the bottom
        core_rows.append(
            ["**CORE aggregate**"]
            + [
                f"**{results['training'][tag]['core_metric']:.4f}**"
                if isinstance(
                    results.get("training", {}).get(tag, {}).get("core_metric"), float
                )
                else "N/A"
                for tag in [tag_p1, tag_p2, tag_baseline]
            ]
        )
        table(core_headers, core_rows)
    else:
        lines.append(
            "_No CORE per-task data (CORE eval not run for this experiment)._\n"
        )

    return "\n".join(lines)


def _find_last_step(model_tag: str) -> int:
    """Return the highest checkpoint step saved under base_checkpoints/{model_tag}."""
    import glob

    ckpt_dir = os.path.join(BASE_DIR, "base_checkpoints", model_tag)
    files = glob.glob(os.path.join(ckpt_dir, "model_*.pt"))
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return max(int(os.path.basename(f).split("_")[1].split(".")[0]) for f in files)


_N_TRAIN_GPUS = int(GPU_TRAIN.split(":")[1]) if ":" in GPU_TRAIN else 1
_N_EVAL_GPUS = int(GPU_EVAL.split(":")[1]) if ":" in GPU_EVAL else 1
_N_SWEEP_P3_GPUS = int(GPU_SWEEP_P3.split(":")[1]) if ":" in GPU_SWEEP_P3 else 1


# =============================================================================
# STAGE: HYPERPARAMETER SWEEP — curriculum choices
# =============================================================================


def _run_sweep_combo_p3(phase1_seq: int, phase1_frac: float, depth: int) -> None:
    """
    Run one curriculum sweep combo:
      1. Train SWEEP_P3_STEPS at phase1_seq  (Phase 1 mini-run)
      2. Warm-start, train SWEEP_P3_STEPS at seq=2048 (Phase 2 mini-run)
    Both phases log to WandB project part3_sweep as separate runs.
    """
    n_steps = SWEEP_P3_STEPS
    bs = DEVICE_BATCH_SWEEP_P3
    nproc = _N_SWEEP_P3_GPUS
    frac_str = f"f{int(phase1_frac * 100):02d}"
    combo = f"s{phase1_seq}_{frac_str}"

    # ── Phase 1 mini-run ──────────────────────────────────────────────────────
    tag_p1 = f"part3/sweep/{combo}_p1"
    run_p1 = f"sweep_{combo}_phase1"
    print(
        f"\n{'=' * 64}\nSweep Phase 1: seq={phase1_seq}  frac={phase1_frac}  steps={n_steps}\n{'=' * 64}"
    )
    _torchrun(
        "scripts.base_train",
        [
            f"--depth={depth}",
            f"--max-seq-len={phase1_seq}",
            f"--model-tag={tag_p1}",
            f"--device-batch-size={bs}",
            f"--total-batch-size={TOTAL_BATCH_SIZE}",
            f"--num-iterations={n_steps}",
            "--save-every=9999",
            "--core-metric-every=9999",
            "--sample-every=-1",
            f"--wandb-project={WANDB_PROJECT_SWEEP_P3}",
            f"--run={run_p1}",
        ],
        nproc=nproc,
    )
    volume.commit()

    # ── Phase 2 mini-run (warm-start from Phase 1) ────────────────────────────
    p1_last_step = _find_last_step(tag_p1)
    p2_total_iters = p1_last_step + n_steps
    tag_p2 = f"part3/sweep/{combo}_p2"
    run_p2 = f"sweep_{combo}_phase2"
    print(
        f"\n{'=' * 64}\nSweep Phase 2: seq=2048  warm-start step={p1_last_step}\n{'=' * 64}"
    )
    _torchrun(
        "scripts.base_train",
        [
            f"--depth={depth}",
            "--max-seq-len=2048",
            f"--model-tag={tag_p2}",
            f"--resume-model-tag={tag_p1}",
            f"--resume-from-step={p1_last_step}",
            "--load-model-only",
            f"--device-batch-size={bs}",
            f"--total-batch-size={TOTAL_BATCH_SIZE}",
            f"--num-iterations={p2_total_iters}",
            "--save-every=9999",
            "--core-metric-every=9999",
            "--sample-every=-1",
            f"--wandb-project={WANDB_PROJECT_SWEEP_P3}",
            f"--run={run_p2}",
        ],
        nproc=nproc,
    )
    volume.commit()
    print(f"  Done combo: {combo}  (phase1 run={run_p1}, phase2 run={run_p2})")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_SWEEP_P3,
    timeout=TIMEOUT_SWEEP_P3,
)
def stage_sweep_p3_s256(depth: int = DEPTH) -> None:
    """
    Curriculum sweep — Phase 1 seq=256, three phase-fractions: 0.2 / 0.4 / 0.6.
    Runs 3 combos sequentially (each combo = 300-step phase1 + 300-step phase2).
    Run in parallel with stage_sweep_p3_s512.

    WandB project: part3_sweep
    Run names: sweep_s256_f{20,40,60}_phase{1,2}
    """
    _setup_cache()
    total = len(SWEEP_P3_FRACS)
    for i, frac in enumerate(SWEEP_P3_FRACS, 1):
        print(f"\n{'#' * 64}\n[{i}/{total}] seq=256  frac={frac}\n{'#' * 64}")
        _run_sweep_combo_p3(phase1_seq=256, phase1_frac=frac, depth=depth)
    print(
        f"\n{'=' * 64}\nseq=256 sweep done — {total} combos in '{WANDB_PROJECT_SWEEP_P3}'.\n{'=' * 64}"
    )


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_SWEEP_P3,
    timeout=TIMEOUT_SWEEP_P3,
)
def stage_sweep_p3_s512(depth: int = DEPTH) -> None:
    """
    Curriculum sweep — Phase 1 seq=512, three phase-fractions: 0.2 / 0.4 / 0.6.
    Runs 3 combos sequentially (each combo = 300-step phase1 + 300-step phase2).
    Run in parallel with stage_sweep_p3_s256.

    WandB project: part3_sweep
    Run names: sweep_s512_f{20,40,60}_phase{1,2}
    """
    _setup_cache()
    total = len(SWEEP_P3_FRACS)
    for i, frac in enumerate(SWEEP_P3_FRACS, 1):
        print(f"\n{'#' * 64}\n[{i}/{total}] seq=512  frac={frac}\n{'#' * 64}")
        _run_sweep_combo_p3(phase1_seq=512, phase1_frac=frac, depth=depth)
    print(
        f"\n{'=' * 64}\nseq=512 sweep done — {total} combos in '{WANDB_PROJECT_SWEEP_P3}'.\n{'=' * 64}"
    )


# =============================================================================
# STAGE: PHASE 1 — seq=512, 40% of Chinchilla budget
# =============================================================================


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_PHASE1,
)
def stage_pretrain_phase1(
    depth: int = DEPTH,
    n_steps: int = N_PHASE1_STEPS,
) -> None:
    """
    Phase 1: train picochat d16 at seq=512 for ~40% of the Chinchilla-optimal token budget.

    Justification for seq=512:
      1. Attention is O(n²): seq=512 is 16× cheaper per step than seq=2048, giving
         more gradient updates per dollar.
      2. RoPE (Su et al. 2022) generalises beyond training length. Chen et al. (2023,
         'Positional Interpolation') and LLaMA 2/3 confirm 4× extension is safe.
      3. Short-context curriculum (Xiong et al. 2023, 'Effective Long-Context Scaling'):
         local/syntactic patterns are learned faster at short context; then long-range
         dependencies are learned in phase 2.
    """
    _setup_cache()
    _python("nanochat.report", ["reset"])
    print(f"Phase 1: depth={depth} seq=512 steps={n_steps} tag={TAG_PHASE1}")
    _torchrun(
        "scripts.base_train",
        [
            f"--depth={depth}",
            "--max-seq-len=512",
            f"--model-tag={TAG_PHASE1}",
            f"--device-batch-size={DEVICE_BATCH_PHASE1}",
            f"--total-batch-size={TOTAL_BATCH_SIZE}",
            f"--num-iterations={n_steps}",
            "--save-every=500",
            f"--wandb-project={WANDB_PROJECT}",
            f"--run={WANDB_RUN_PHASE1}",
            "--core-metric-every=1000",  # CORE is slow, run less often
        ],
        nproc=_N_TRAIN_GPUS,
    )
    volume.commit()
    print(f"Phase 1 complete. Checkpoint: base_checkpoints/{TAG_PHASE1}/")


# =============================================================================
# STAGE: PHASE 2 — warm-start from Phase 1, extend to seq=2048
# =============================================================================


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_PHASE2,
)
def stage_pretrain_phase2(
    depth: int = DEPTH,
    n_steps: int = N_PHASE2_STEPS,
) -> None:
    """
    Phase 2: load Phase 1 weights, switch to seq=2048, train for remaining 60%.

    Uses --load-model-only so optimizer and dataloader state are NOT carried over
    from Phase 1. The optimizer restarts fresh (as if fine-tuning), which is
    correct because the gradient geometry changes when seq_len changes.

    IMPORTANT: --num-iterations must equal phase1_steps + phase2_steps (= N_TOTAL_STEPS)
    because base_train's step counter continues from the resume point. If you only
    pass phase2_steps and resume from phase1_steps, the loop exits immediately.
    """
    _setup_cache()
    # Dynamically find the last step saved by phase 1
    phase1_last_step = _find_last_step(TAG_PHASE1)
    # Total iterations = phase1 steps already done + phase2 steps to run
    total_iters = phase1_last_step + n_steps
    print(
        f"Phase 2: loading from {TAG_PHASE1} step={phase1_last_step}, saving to {TAG_PHASE2}"
    )
    print(f"         will train steps {phase1_last_step} → {total_iters}")
    _torchrun(
        "scripts.base_train",
        [
            f"--depth={depth}",
            "--max-seq-len=2048",
            f"--model-tag={TAG_PHASE2}",
            f"--resume-model-tag={TAG_PHASE1}",
            f"--resume-from-step={phase1_last_step}",
            "--load-model-only",
            f"--device-batch-size={DEVICE_BATCH_PHASE2}",
            f"--total-batch-size={TOTAL_BATCH_SIZE}",
            f"--num-iterations={total_iters}",
            "--save-every=500",
            f"--wandb-project={WANDB_PROJECT}",
            f"--run={WANDB_RUN_PHASE2}",
            "--core-metric-every=1000",
        ],
        nproc=_N_TRAIN_GPUS,
    )
    volume.commit()
    print(f"Phase 2 complete. Checkpoint: base_checkpoints/{TAG_PHASE2}/")


# =============================================================================
# STAGE: BASELINE — full budget at seq=2048 from scratch
# =============================================================================


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_BASELINE,
)
def stage_pretrain_baseline(
    depth: int = DEPTH,
    n_steps: int = N_BASELINE_STEPS,
) -> None:
    """
    Baseline: train d16 at seq=2048 for the full Chinchilla-optimal budget from scratch.

    This control run answers: 'is the 512→2048 curriculum better, worse, or the same
    as just training at 2048 the whole time?'  Runs in parallel with Phase 2.
    """
    _setup_cache()
    print(f"Baseline: depth={depth} seq=2048 steps={n_steps} tag={TAG_BASELINE}")
    _torchrun(
        "scripts.base_train",
        [
            f"--depth={depth}",
            "--max-seq-len=2048",
            f"--model-tag={TAG_BASELINE}",
            f"--device-batch-size={DEVICE_BATCH_BASELINE}",
            f"--total-batch-size={TOTAL_BATCH_SIZE}",
            f"--num-iterations={n_steps}",
            "--save-every=500",
            f"--wandb-project={WANDB_PROJECT}",
            f"--run={WANDB_RUN_BASELINE}",
            "--core-metric-every=1000",
        ],
        nproc=_N_TRAIN_GPUS,
    )
    volume.commit()
    print(f"Baseline complete. Checkpoint: base_checkpoints/{TAG_BASELINE}/")


# =============================================================================
# STAGE: EVAL + REPORT — CORE benchmark, custom evals, then full report
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
    Single stage that runs all evaluations then writes the markdown report.

      A. CORE (22 benchmarks via scripts.base_eval) on all 3 checkpoints
      B. Needle-in-Haystack + BPB-by-position (part3/eval_longctx.py)
      C. Build and write nanochat_cache/report/part3_report.md

    Reads checkpoints from:  base_checkpoints/part3/d20_{ctx512,ctx2048,baseline}
    Writes results to:       nanochat_cache/part3_eval_results.json
    Writes report to:        nanochat_cache/report/part3_report.md
    """
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

    tags = [TAG_PHASE1, TAG_PHASE2, TAG_BASELINE]
    core_data: dict = {}  # tag → {core_metric, val_bpb, tasks}

    # ── A. CORE eval on all three checkpoints ──────────────────────────────────
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
        # Collect immediately — before next eval can overwrite the default slug
        core_data[tag] = _collect_core_csv(tag)

    # ── B. Custom long-context eval ────────────────────────────────────────────
    results_path = os.path.join(NANOCHAT_CACHE, "part3_eval_results.json")
    print(
        f"\n{'=' * 60}\nCustom long-context eval (needle + BPB-by-position)\n{'=' * 60}"
    )
    _python(
        "part3.eval_longctx",
        [
            f"--tags={','.join(tags)}",
            f"--output={results_path}",
            "--n-samples=200",
        ],
    )
    volume.commit()

    # ── C. Build report ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}\nGenerating report\n{'=' * 60}")
    with open(results_path) as f:
        results = json.load(f)

    # Merge CORE results into the shared results dict
    _merge_core_into_results(results, tags, core_data)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()

    report_md = _build_report_markdown(
        results,
        tag_p1=TAG_PHASE1,
        tag_p2=TAG_PHASE2,
        tag_baseline=TAG_BASELINE,
        depth=DEPTH,
        n_p1_steps=N_PHASE1_STEPS,
        n_p2_steps=N_PHASE2_STEPS,
        n_baseline_steps=N_BASELINE_STEPS,
        chinchilla_tokens=CHINCHILLA_TOKENS,
        total_batch_size=TOTAL_BATCH_SIZE,
        device_batch_p1=DEVICE_BATCH_PHASE1,
        device_batch_p2=DEVICE_BATCH_PHASE2,
        device_batch_baseline=DEVICE_BATCH_BASELINE,
        n_gpus=_N_TRAIN_GPUS,
        wandb_run_p1=WANDB_RUN_PHASE1,
        wandb_run_p2=WANDB_RUN_PHASE2,
        wandb_run_baseline=WANDB_RUN_BASELINE,
    )

    report_dir = os.path.join(NANOCHAT_CACHE, "report")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "part3_report.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    volume.commit()

    print(f"\nEval JSON : {results_path}")
    print(f"Report    : {report_path}")
    print("\n" + "=" * 60)
    print("FULL REPORT:")
    print("=" * 60)
    print(report_md)


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_EVAL,
    timeout=TIMEOUT_EVAL,
)
def stage_needle_reeval() -> None:
    """
    Re-run the needle-in-haystack eval (likelihood-ranking variant) on all three
    d20 checkpoints, preserving the existing BPB results in the JSON.
    """
    volume.reload()
    _setup_cache()

    tags = [TAG_PHASE1, TAG_PHASE2, TAG_BASELINE]
    results_path = os.path.join(NANOCHAT_CACHE, "part3_eval_results.json")

    print(f"\n{'=' * 60}")
    print("Needle-in-haystack re-eval (likelihood ranking, 10-way)")
    print(f"{'=' * 60}")
    _python(
        "part3.eval_longctx",
        [
            f"--tags={','.join(tags)}",
            f"--output={results_path}",
            "--n-samples=200",
            "--skip-bpb",
        ],
    )
    volume.commit()
    print(f"\nResults written to: {results_path}")


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
    Smoke test for the full Part 3 pipeline at d12 scale:
      - Phase 1: d12, seq=512, 300 steps
      - Phase 2: d12, seq=2048, warm-started from phase 1, 300 steps
      - Baseline: d12, seq=2048, fresh, 300 steps
      - Custom eval: needle-in-haystack + BPB-by-position on all 3 d12 checkpoints
      - Report: exercise the report generation code

    This validates --resume-model-tag, --load-model-only, wandb-project, and
    eval_longctx.py end-to-end before spending money on d20.
    """
    _setup_cache()
    nproc = 4
    bs_small = 8  # small batch for quick test

    # --- Phase 1 (d12, seq=512) ---
    print("=== Quick test: Phase 1 (d12, seq=512, 300 steps) ===")
    _torchrun(
        "scripts.base_train",
        [
            "--depth=12",
            "--max-seq-len=512",
            f"--model-tag={TAG_D12_PHASE1}",
            f"--device-batch-size={bs_small}",
            "--total-batch-size=65536",  # small batch for quick test
            "--num-iterations=300",
            "--save-every=150",
            "--core-metric-every=999999",
            "--sample-every=-1",
            f"--wandb-project={WANDB_PROJECT}",
            "--run=d12_quicktest_phase1",
        ],
        nproc=nproc,
    )
    volume.commit()

    # --- Phase 2 (d12, resume at seq=2048) ---
    print("=== Quick test: Phase 2 (d12, seq=2048, 300 steps, warm-started) ===")
    p1_last_step = _find_last_step(TAG_D12_PHASE1)
    # num-iterations = phase1_steps_done + phase2_steps_to_run
    # (base_train's step counter continues from resume point)
    p2_total_iters = p1_last_step + N_D12_PHASE2_STEPS
    _torchrun(
        "scripts.base_train",
        [
            "--depth=12",
            "--max-seq-len=2048",
            f"--model-tag={TAG_D12_PHASE2}",
            f"--resume-model-tag={TAG_D12_PHASE1}",
            f"--resume-from-step={p1_last_step}",
            "--load-model-only",
            f"--device-batch-size={bs_small}",
            "--total-batch-size=65536",
            f"--num-iterations={p2_total_iters}",
            "--save-every=150",
            "--core-metric-every=999999",
            "--sample-every=-1",
            f"--wandb-project={WANDB_PROJECT}",
            "--run=d12_quicktest_phase2",
        ],
        nproc=nproc,
    )
    volume.commit()

    # --- Baseline (d12, seq=2048, fresh) ---
    print("=== Quick test: Baseline (d12, seq=2048, fresh, 300 steps) ===")
    _torchrun(
        "scripts.base_train",
        [
            "--depth=12",
            "--max-seq-len=2048",
            f"--model-tag={TAG_D12_BASELINE}",
            f"--device-batch-size={bs_small}",
            "--total-batch-size=65536",
            "--num-iterations=300",
            "--save-every=150",
            "--core-metric-every=999999",
            "--sample-every=-1",
            f"--wandb-project={WANDB_PROJECT}",
            "--run=d12_quicktest_baseline",
        ],
        nproc=nproc,
    )
    volume.commit()

    # --- Custom eval on d12 checkpoints ---
    print("=== Quick test: Custom long-context eval on d12 checkpoints ===")
    tags_d12 = f"{TAG_D12_PHASE1},{TAG_D12_PHASE2},{TAG_D12_BASELINE}"
    out_path = os.path.join(NANOCHAT_CACHE, "part3_d12_eval_results.json")
    _python(
        "part3.eval_longctx",
        [
            f"--tags={tags_d12}",
            f"--output={out_path}",
            "--n-samples=20",  # very few samples for quick test
        ],
        # cwd=/root/nanochat so uv picks up the project venv (torch etc.);
        # PYTHONPATH=/root (injected by _python) makes part3.eval_longctx importable.
    )
    volume.commit()

    # --- Generate d12 report (validates full report code path) ---
    print("=== Quick test: Generating d12 sanity report ===")
    with open(out_path) as _f:
        _d12_results = json.load(_f)

    # d12 quick-test uses fixed step counts; Chinchilla budget is tiny
    _d12_batch = 65536
    _d12_dev_bs = (
        8  # bs_small used in training (seq=512 and seq=2048 share same device_bs)
    )
    _d12_n_gpus = 4
    _d12_chinchilla = (N_D12_PHASE1_STEPS + N_D12_BASELINE_STEPS) * _d12_batch
    _d12_report_md = _build_report_markdown(
        _d12_results,
        tag_p1=TAG_D12_PHASE1,
        tag_p2=TAG_D12_PHASE2,
        tag_baseline=TAG_D12_BASELINE,
        depth=12,
        n_p1_steps=N_D12_PHASE1_STEPS,
        n_p2_steps=N_D12_PHASE2_STEPS,
        n_baseline_steps=N_D12_BASELINE_STEPS,
        chinchilla_tokens=_d12_chinchilla,
        total_batch_size=_d12_batch,
        device_batch_p1=_d12_dev_bs,
        device_batch_p2=_d12_dev_bs,
        device_batch_baseline=_d12_dev_bs,
        n_gpus=_d12_n_gpus,
        wandb_run_p1="d12_quicktest_phase1",
        wandb_run_p2="d12_quicktest_phase2",
        wandb_run_baseline="d12_quicktest_baseline",
        title_suffix="d12 Quick Test",
    )

    report_dir = os.path.join(NANOCHAT_CACHE, "report")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "part3_d12_quicktest_report.md")
    with open(report_path, "w") as _f:
        _f.write(_d12_report_md)
    volume.commit()
    print(f"Report written to: {report_path}")
    print(_d12_report_md[:2000])  # preview first 2000 chars in logs

    print("\nQuick test passed! d12 pipeline end-to-end verified.")
    print(f"  Checkpoints : nanochat_cache/base_checkpoints/part3/  (3 dirs)")
    print(f"  Eval JSON   : {out_path}")
    print(f"  Report      : {report_path}")
    print("Ready to run the full d20 experiments.")


# =============================================================================
# MAIN ENTRYPOINT — full Part 3 pipeline
# =============================================================================


@app.local_entrypoint()
def main() -> None:
    """
    Full Part 3 pipeline:
      1. d12 smoke test (validates all code paths cheaply)
      2. Phase 1: d16 at seq=512, 40% of Chinchilla budget
      3. Phase 2 + Baseline in parallel (separate Modal containers)
      4. Eval + Report: CORE, needle-in-haystack, BPB-by-position, then markdown
    """
    w = 64
    print("\n" + "=" * w)
    print("Part 3: Picochat Context Length Curriculum (d16)")
    print(
        f"  depth={DEPTH}  phase1_steps={N_PHASE1_STEPS}  phase2_steps={N_PHASE2_STEPS}"
    )
    print(f"  baseline_steps={N_BASELINE_STEPS}  total_batch={TOTAL_BATCH_SIZE}")
    print("=" * w + "\n")

    # Step 1: Smoke test (validates --load-model-only, --resume-model-tag, eval)
    print("[0/4] d12 smoke test...")
    quick_test_d12.remote()

    # Step 2: Phase 1 (must finish before Phase 2 can load its checkpoint)
    print("[1/4] Phase 1: d20, seq=512, 40% Chinchilla budget...")
    stage_pretrain_phase1.remote()

    # Step 3: Phase 2 + Baseline in parallel
    print("[2/4] Phase 2 + Baseline in parallel (separate Modal containers)...")
    phase2_handle = stage_pretrain_phase2.spawn()
    baseline_handle = stage_pretrain_baseline.spawn()
    phase2_handle.get()
    baseline_handle.get()

    # Step 4: Eval all 3 checkpoints then immediately write the report
    print("[3/4] Eval + Report (CORE + needle + BPB-by-position + markdown)...")
    stage_eval_and_report.remote()

    print("\n" + "=" * w)
    print("Part 3 (d16) complete!")
    print(f"  Report: nanochat_cache/report/part3_report.md (on nanochat-vol)")
    print(f"  WandB:  project '{WANDB_PROJECT}'")
    print("=" * w + "\n")


# =============================================================================
# D12 EVAL + REPORT ONLY — re-run on existing d12 checkpoints (no training)
# =============================================================================


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_EVAL,
    timeout=TIMEOUT_EVAL,
)
def quick_test_d12_eval_report() -> None:
    """
    Runs ONLY the eval + report steps of the d12 quick test.

    Assumes the three d12 checkpoints already exist on the volume:
      base_checkpoints/part3/d12_ctx512
      base_checkpoints/part3/d12_ctx2048
      base_checkpoints/part3/d12_baseline

    Produces:
      nanochat_cache/part3_d12_eval_results.json
      nanochat_cache/report/part3_d12_quicktest_report.md
    """
    _setup_cache()

    # --- Download eval bundle if needed ---
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

    tags_d12_list = [TAG_D12_PHASE1, TAG_D12_PHASE2, TAG_D12_BASELINE]
    core_data_d12: dict = {}

    # --- A. CORE eval on all three d12 checkpoints ---
    for tag in tags_d12_list:
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
        core_data_d12[tag] = _collect_core_csv(tag)

    # --- B. Needle-in-haystack + BPB-by-position ---
    print(f"\n{'=' * 60}\nd12 eval: needle-in-haystack + BPB-by-position\n{'=' * 60}")
    out_path = os.path.join(NANOCHAT_CACHE, "part3_d12_eval_results.json")
    _python(
        "part3.eval_longctx",
        [
            f"--tags={','.join(tags_d12_list)}",
            f"--output={out_path}",
            "--n-samples=50",
        ],
    )

    # Merge CORE results into JSON
    with open(out_path) as _f:
        _d12_results = json.load(_f)
    _merge_core_into_results(_d12_results, tags_d12_list, core_data_d12)
    with open(out_path, "w") as _f:
        json.dump(_d12_results, _f, indent=2)
    volume.commit()

    # --- Report ---
    print("=== d12 report ===")
    with open(out_path) as _f:
        _d12_results = json.load(_f)

    _d12_batch = 65536
    _d12_dev_bs = 8
    _d12_n_gpus = 4
    _d12_chinchilla = (N_D12_PHASE1_STEPS + N_D12_BASELINE_STEPS) * _d12_batch
    _d12_report_md = _build_report_markdown(
        _d12_results,
        tag_p1=TAG_D12_PHASE1,
        tag_p2=TAG_D12_PHASE2,
        tag_baseline=TAG_D12_BASELINE,
        depth=12,
        n_p1_steps=N_D12_PHASE1_STEPS,
        n_p2_steps=N_D12_PHASE2_STEPS,
        n_baseline_steps=N_D12_BASELINE_STEPS,
        chinchilla_tokens=_d12_chinchilla,
        total_batch_size=_d12_batch,
        device_batch_p1=_d12_dev_bs,
        device_batch_p2=_d12_dev_bs,
        device_batch_baseline=_d12_dev_bs,
        n_gpus=_d12_n_gpus,
        wandb_run_p1="d12_quicktest_phase1",
        wandb_run_p2="d12_quicktest_phase2",
        wandb_run_baseline="d12_quicktest_baseline",
        title_suffix="d12 Quick Test",
    )

    report_dir = os.path.join(NANOCHAT_CACHE, "report")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "part3_d12_quicktest_report.md")
    with open(report_path, "w") as _f:
        _f.write(_d12_report_md)
    volume.commit()

    print(f"\nEval JSON : {out_path}")
    print(f"Report    : {report_path}")
    print("\n" + "=" * 60)
    print("FULL REPORT:")
    print("=" * 60)
    print(_d12_report_md)


# =============================================================================
# STAGE: EVAL FIGURES  (CPU-only, no GPU)
# =============================================================================


@app.function(
    image=figures_image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    cpu=4,
    timeout=60 * 30,
)
def stage_make_eval_figures_p3() -> None:
    """CPU-only job: load part3_eval_results.json from the volume and fetch
    training-loss history from W&B to generate three publication figures:

    Figure 1 — p3_training_curves.png
        Phase 1 and Phase 2 loss/BPB curves on the same axes, with a vertical
        dashed line at the Phase 1 / Phase 2 boundary.
        Baseline shown as a separate dashed curve.

    Figure 2 — p3_bpb_by_position.png
        Grouped bar chart of BPB on each of four non-overlapping 512-token
        segments (seg0–seg3) of a 2048-token sequence.
        Three bar groups: Phase 1, Phase 2, Baseline.

    Figure 3 — p3_needle_accuracy.png
        Line plot of 10-way retrieval accuracy vs needle distance (tokens
        from end of context) for Phase 1, Phase 2, and Baseline.
        Random-chance baseline (10 %) shown as a dotted line.

    All PNGs are saved to nanochat_cache/report/ on the shared volume and
    logged as images to the nanochat-part3 W&B project.
    Requires stage_eval_and_report to have been run first.
    """
    import os
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import wandb

    report_dir = os.path.join(NANOCHAT_CACHE, "report")
    os.makedirs(report_dir, exist_ok=True)

    # ── load eval JSON ────────────────────────────────────────────────────────
    results_path = os.path.join(NANOCHAT_CACHE, "part3_eval_results.json")
    volume.reload()
    with open(results_path) as f:
        results = json.load(f)

    tags = [TAG_PHASE1, TAG_PHASE2, TAG_BASELINE]
    labels = {
        TAG_PHASE1: "Phase 1 (ctx=512)",
        TAG_PHASE2: "Phase 2 (ctx=2048, warm-start)",
        TAG_BASELINE: "Baseline (ctx=2048, full)",
    }
    colors = {
        TAG_PHASE1: "#4C72B0",  # blue
        TAG_PHASE2: "#DD8452",  # orange
        TAG_BASELINE: "#55A868",  # green
    }
    run_names = {
        TAG_PHASE1: WANDB_RUN_PHASE1,
        TAG_PHASE2: WANDB_RUN_PHASE2,
        TAG_BASELINE: WANDB_RUN_BASELINE,
    }
    # Baseline was trained in the Part 2 project, not nanochat-part3
    run_projects = {
        TAG_PHASE1: WANDB_PROJECT,
        TAG_PHASE2: WANDB_PROJECT,
        TAG_BASELINE: "part2_mtp",
    }

    # ── resolve W&B entity ────────────────────────────────────────────────────
    api = wandb.Api(timeout=120)
    entity = ""
    for getter in [
        lambda: api.viewer()["entity"],
        lambda: api.default_entity,
        lambda: os.environ.get("WANDB_ENTITY", ""),
    ]:
        try:
            entity = getter() or ""
            if entity:
                break
        except Exception:
            pass

    project_path = f"{entity}/{WANDB_PROJECT}" if entity else WANDB_PROJECT
    print(f"Fetching training curves from W&B: {project_path}")

    # ── helper: fetch run history ─────────────────────────────────────────────
    def fetch_run_history(run_name: str, project_override: str = None):
        """Return (steps, values) for the best available loss metric."""
        proj = project_override or WANDB_PROJECT
        cur_project_path = f"{entity}/{proj}" if entity else proj
        try:
            runs = api.runs(cur_project_path, filters={"config.run": run_name})
            if not runs:
                runs = api.runs(cur_project_path, filters={"display_name": run_name})
            if not runs:
                print(f"  WARNING: no W&B run found for name {run_name!r} in {cur_project_path!r}")
                return [], []
            run = runs[0]
            for key in ["val_bpb", "val/bpb", "train/loss", "loss", "train_loss"]:
                rows = list(run.scan_history(keys=["_step", key]))
                rows = [r for r in rows if r.get(key) is not None]
                if rows:
                    print(f"  {run_name}: {len(rows)} points for '{key}'")
                    return [r["_step"] for r in rows], [r[key] for r in rows]
            print(f"  WARNING: no loss metric found for {run_name!r}")
            return [], []
        except Exception as e:
            print(f"  WARNING: failed to fetch {run_name!r}: {e}")
            return [], []

    # ── Figure 1: Training curves ─────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 5), constrained_layout=True)

    for tag in tags:
        steps, vals = fetch_run_history(run_names[tag], project_override=run_projects[tag])
        if steps:
            ls = "--" if tag == TAG_BASELINE else "-"
            ax1.plot(
                steps,
                vals,
                label=labels[tag],
                color=colors[tag],
                linewidth=1.6,
                linestyle=ls,
                alpha=0.92,
            )

    ax1.axvline(
        x=N_PHASE1_STEPS,
        color="grey",
        linestyle=":",
        linewidth=1.4,
        label=f"Phase 1→2 boundary (step {N_PHASE1_STEPS})",
    )
    ax1.set_xlabel("Training Step", fontsize=11)
    ax1.set_ylabel("Loss / BPB ↓", fontsize=11)
    ax1.set_title(
        "Part 3 d16: Training Curves  (Phase 1, Phase 2, Baseline)", fontsize=12
    )
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(alpha=0.3)

    fig1_path = os.path.join(report_dir, "p3_training_curves.png")
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {fig1_path}")

    # ── Figure 2: BPB by context position ─────────────────────────────────────
    seg_keys = ["seg0", "seg1", "seg2", "seg3"]
    seg_labels = [
        "seg0\n(0–511)",
        "seg1\n(512–1023)",
        "seg2\n(1024–1535)",
        "seg3\n(1536–2047)",
    ]
    bpb_data = results.get("bpb_by_position", {})

    x = np.arange(len(seg_keys))
    n_mdl = len(tags)
    bar_w = 0.22
    offsets = np.linspace(-(n_mdl - 1) / 2 * bar_w, (n_mdl - 1) / 2 * bar_w, n_mdl)

    fig2, ax2 = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for i, tag in enumerate(tags):
        vals = [bpb_data.get(tag, {}).get(sk, float("nan")) for sk in seg_keys]
        ax2.bar(
            x + offsets[i],
            vals,
            width=bar_w,
            label=labels[tag],
            color=colors[tag],
            edgecolor="white",
            linewidth=0.5,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(seg_labels, fontsize=9)
    ax2.set_ylabel("BPB ↓", fontsize=11)
    ax2.set_title(
        "Part 3 d16: BPB by Context Position (512-token segments)", fontsize=12
    )
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig2_path = os.path.join(report_dir, "p3_bpb_by_position.png")
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {fig2_path}")

    # ── Figure 3: Needle-in-Haystack ──────────────────────────────────────────
    needle_data = results.get("needle", {})
    distances = needle_data.get("distances", [64, 256, 512, 768, 1024, 1536])

    fig3, ax3 = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax3.axhline(
        y=0.10,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="Random chance (10%)",
        zorder=2,
    )

    for tag in tags:
        accs = [needle_data.get(tag, {}).get(str(d), float("nan")) for d in distances]
        ax3.plot(
            distances,
            accs,
            marker="o",
            markersize=5,
            label=labels[tag],
            color=colors[tag],
            linewidth=1.8,
            zorder=3,
        )
        if tag == TAG_PHASE1:
            ax3.axvspan(
                512.5,
                max(distances) + 50,
                alpha=0.06,
                color=colors[tag],
                label="Beyond Phase 1 context (>512)",
                zorder=1,
            )

    ax3.set_xlabel("Needle Distance from End of Context (tokens)", fontsize=11)
    ax3.set_ylabel("10-way Retrieval Accuracy ↑", fontsize=11)
    ax3.set_title(
        "Part 3 d16: Needle-in-Haystack  (200 trials per distance)", fontsize=12
    )
    ax3.set_xticks(distances)
    ax3.set_ylim(0, None)
    ax3.legend(fontsize=9, loc="upper right")
    ax3.grid(alpha=0.3)

    fig3_path = os.path.join(report_dir, "p3_needle_accuracy.png")
    fig3.savefig(fig3_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {fig3_path}")

    # ── commit + log to W&B ───────────────────────────────────────────────────
    volume.commit()

    with wandb.init(
        project=WANDB_PROJECT,
        entity=entity or None,
        job_type="figures",
        name="p3_eval_figures",
    ) as wrun:
        wrun.log(
            {
                "eval/training_curves": wandb.Image(fig1_path),
                "eval/bpb_by_position": wandb.Image(fig2_path),
                "eval/needle_accuracy": wandb.Image(fig3_path),
            }
        )
    print("Eval figures logged to W&B ✓")


# =============================================================================
# STAGE: SWEEP FIGURES  (CPU-only, no GPU)
# =============================================================================


@app.function(
    image=figures_image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    cpu=4,
    timeout=60 * 30,
)
def stage_make_sweep_figures_p3() -> None:
    """CPU-only job: pull all Part 3 sweep runs from W&B and produce:

    Figure 1 — 2×2 panel training-loss curves.
        Rows = seq len (256, 512).  Columns = phase (Phase 1, Phase 2).
        3 lines per panel = phase1_frac (0.2, 0.4, 0.6).
        Same colour per frac across all panels.

    Figure 2 — grouped bar chart of final-step train loss.
        4 groups (s256_p1 | s256_p2 | s512_p1 | s512_p2), 3 bars each.
        Same colour per frac across groups.

    Both PNGs saved to the shared Volume (nanochat_cache/report/) and
    logged as images to the part3_sweep W&B project.
    """
    import re
    import os
    import wandb
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    report_dir = os.path.join(NANOCHAT_CACHE, "report")
    os.makedirs(report_dir, exist_ok=True)

    # ── fetch runs ────────────────────────────────────────────────────────────
    api = wandb.Api(timeout=120)
    entity = ""
    for getter in [
        lambda: api.viewer()["entity"],
        lambda: api.default_entity,
        lambda: os.environ.get("WANDB_ENTITY", ""),
    ]:
        try:
            entity = getter() or ""
            if entity:
                break
        except Exception:
            pass

    project_path = (
        f"{entity}/{WANDB_PROJECT_SWEEP_P3}" if entity else WANDB_PROJECT_SWEEP_P3
    )
    print(f"Fetching runs from: {project_path}  (entity={entity!r})")
    all_runs = api.runs(project_path)

    # ── layout constants ──────────────────────────────────────────────────────
    SEQS = [256, 512]
    FRACS = [0.2, 0.4, 0.6]
    PHASES = [1, 2]

    FRAC_LABELS = {f: f"frac={f}" for f in FRACS}
    # 3 colours — one per frac, consistent across every panel/group
    FRAC_COLORS = {f: c for f, c in zip(FRACS, plt.cm.tab10([0, 1, 2]))}

    # run name pattern: sweep_s{seq}_f{frac*100:02d}_phase{phase}
    # e.g. sweep_s256_f20_phase1
    _RE = re.compile(r"^sweep_s(\d+)_f(\d+)_phase(\d+)$")

    # histories[(seq, frac, phase)] = {"steps": [...], "loss": [...]}
    histories: dict = {}
    final_loss: dict = {}

    for run in all_runs:
        m = _RE.match(run.name)
        if not m:
            continue
        seq = int(m.group(1))
        frac = int(m.group(2)) / 100.0
        phase = int(m.group(3))
        if seq not in SEQS or frac not in FRACS or phase not in PHASES:
            continue

        try:
            rows = list(run.scan_history())
        except Exception as e:
            print(f"  WARNING — could not fetch {run.name}: {e}")
            continue

        # auto-detect loss key
        _LOSS_KEYS = ["train/loss", "loss", "train_loss"]
        loss_key: str | None = None
        for row in rows[:10]:
            for k in _LOSS_KEYS:
                if row.get(k) is not None:
                    loss_key = k
                    break
            if loss_key:
                break

        steps, losses = [], []
        for row in rows:
            if loss_key and row.get(loss_key) is not None:
                steps.append(row.get("_step", len(steps)))
                losses.append(float(row[loss_key]))

        if not losses:
            print(f"  WARNING — empty history for {run.name}, skipping")
            continue

        key = (seq, frac, phase)
        histories[key] = {"steps": steps, "loss": losses}
        final_loss[key] = losses[-1]
        print(f"  loaded {run.name}: {len(steps)} steps, final={losses[-1]:.4f}")

    print(f"Loaded {len(histories)} / 12 expected sweep runs")

    # ── Figure 1: 2×2 loss curves ─────────────────────────────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    fig1.suptitle(
        "Part 3 Curriculum Sweep — Training Loss Curves (d16, 300 steps per phase)",
        fontsize=13,
        y=1.02,
    )
    panel_titles = {
        (0, 0): "seq=256 — Phase 1",
        (0, 1): "seq=256 — Phase 2",
        (1, 0): "seq=512 — Phase 1",
        (1, 1): "seq=512 — Phase 2",
    }
    panel_keys = {
        (0, 0): (256, 1),
        (0, 1): (256, 2),
        (1, 0): (512, 1),
        (1, 1): (512, 2),
    }

    for (row_i, col_i), (seq, phase) in panel_keys.items():
        ax = axes[row_i][col_i]
        for frac in FRACS:
            key = (seq, frac, phase)
            if key in histories:
                d = histories[key]
                ax.plot(
                    d["steps"],
                    d["loss"],
                    color=FRAC_COLORS[frac],
                    label=FRAC_LABELS[frac],
                    linewidth=1.5,
                    alpha=0.85,
                )
        ax.set_title(panel_titles[(row_i, col_i)], fontsize=10, pad=4)
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("Train Loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    handles = [
        mpatches.Patch(color=FRAC_COLORS[f], label=FRAC_LABELS[f]) for f in FRACS
    ]
    fig1.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.06),
        fontsize=9,
        title="Phase-1 fraction of total budget  (colour consistent across all panels)",
        title_fontsize=8,
    )

    fig1_path = os.path.join(report_dir, "p3_sweep_loss_curves.png")
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {fig1_path}")

    # ── Figure 2: grouped bar chart — final loss ──────────────────────────────
    # 4 groups: s256_p1 | s256_p2 | s512_p1 | s512_p2
    # 3 bars per group (one per frac), same colours as Figure 1
    GROUPS = [(256, 1), (256, 2), (512, 1), (512, 2)]
    GROUP_LABELS = {
        (256, 1): "seq=256\nPhase 1",
        (256, 2): "seq=256\nPhase 2",
        (512, 1): "seq=512\nPhase 1",
        (512, 2): "seq=512\nPhase 2",
    }
    BAR_W = 0.22
    N_FRACS = len(FRACS)
    GROUP_GAP = 0.6

    fig2, ax2 = plt.subplots(figsize=(14, 6), constrained_layout=True)
    fig2.suptitle(
        "Part 3 Sweep — Final Training Loss at Step 300  (lower is better)",
        fontsize=13,
    )

    group_centers = []
    x = 0.0
    offsets = np.linspace(
        -(N_FRACS - 1) / 2 * BAR_W, (N_FRACS - 1) / 2 * BAR_W, N_FRACS
    )
    xtick_pos, xtick_lbl = [], []

    for grp in GROUPS:
        cx = x
        seq_g, phase_g = grp
        for i, frac in enumerate(FRACS):
            val = final_loss.get((seq_g, frac, phase_g))
            bx = cx + offsets[i]
            if val is not None:
                ax2.bar(
                    bx,
                    val,
                    width=BAR_W,
                    color=FRAC_COLORS[frac],
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                )
            else:
                ax2.bar(
                    bx,
                    0,
                    width=BAR_W,
                    color=FRAC_COLORS[frac],
                    alpha=0.15,
                    edgecolor="grey",
                    linewidth=0.4,
                    zorder=3,
                )
        xtick_pos.append(cx)
        xtick_lbl.append(GROUP_LABELS[grp])
        group_centers.append(cx)
        x += 1.0 + GROUP_GAP

    ax2.set_xticks(xtick_pos)
    ax2.set_xticklabels(xtick_lbl, fontsize=9)
    ax2.set_ylabel("Final Train Loss ↓", fontsize=10)
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    handles2 = [
        mpatches.Patch(color=FRAC_COLORS[f], label=FRAC_LABELS[f]) for f in FRACS
    ]
    ax2.legend(
        handles=handles2,
        fontsize=9,
        title="Phase-1 fraction",
        title_fontsize=8,
        loc="upper right",
    )

    fig2_path = os.path.join(report_dir, "p3_sweep_bar_chart.png")
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {fig2_path}")

    # ── commit + log to W&B ───────────────────────────────────────────────────
    volume.commit()

    with wandb.init(
        project=WANDB_PROJECT_SWEEP_P3,
        entity=entity or None,
        job_type="figures",
        name="sweep_figures_p3",
    ) as wrun:
        wrun.log(
            {
                "sweep/loss_curves": wandb.Image(fig1_path),
                "sweep/final_loss_bar": wandb.Image(fig2_path),
            }
        )
    print("Figures logged to W&B ✓")
