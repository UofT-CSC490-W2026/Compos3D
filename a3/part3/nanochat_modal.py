"""
Part 3: Picochat Context Length Curriculum
==========================================

Three training experiments:
  1. Phase 1  — picochat (d20) at seq=512,  40% of Chinchilla token budget
  2. Phase 2  — continue from Phase 1 at seq=2048, remaining 60% of budget
  3. Baseline — picochat (d20) at seq=2048,  full Chinchilla budget from scratch

Phase 2 and Baseline run in parallel after Phase 1.

All checkpoints land in nanochat_cache/base_checkpoints/part3/ on the volume.

Usage
-----
Full pipeline (d12 smoke-test, then d20):
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
    d20 Phase 1       : ~45 min   (~$23)
    d20 Phase 2       : ~60 min   (~$31) ]  run in parallel
    d20 Baseline      : ~90 min   (~$46) ]
    eval              : ~45 min   (~$12, 2×H100)
"""

import os
import json
import subprocess
import modal
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# CONFIGURATION
# =============================================================================

DEPTH = 20  # picochat = d20 (~560M params)
GPU_TRAIN = "H100:8"
GPU_EVAL = "H100:4"

# Device batch sizes
DEVICE_BATCH_PHASE1 = 32  # seq=512 use 4× shorter batch for faster training
DEVICE_BATCH_PHASE2 = 16  # seq=2048
DEVICE_BATCH_BASELINE = 16

# Fixed total batch size (tokens per optimizer step) for all three runs so
# gradient accumulation and LR scales are comparable.
TOTAL_BATCH_SIZE = 524288

# Chinchilla-optimal token budget for d20:
#   model_dim = depth * aspect_ratio = 20 * 64 = 1280
#   Each transformer layer (attention + MLP matrices):
#     attn: 4 * 1280^2 ≈ 6.55M  |  mlp: ~2 * 4 * 1280^2 ≈ 13.1M
#   20 layers ≈ 393M transformer_matrices + lm_head ≈ 65M → scaling_params ≈ 458M
#   target_param_data_ratio default = 10.5 → target_tokens ≈ 4.81B
#   At total_batch_size=524288 → total_steps ≈ 9174
# We use explicit --num-iterations for reproducibility across runs.
CHINCHILLA_TOKENS = 4_810_000_000  # ≈ 10.5 × 458M scaling params
PHASE1_FRAC = 0.40  # 40% at seq=512
PHASE2_FRAC = 0.60  # 60% at seq=2048 (warm-started)

N_TOTAL_STEPS = CHINCHILLA_TOKENS // TOTAL_BATCH_SIZE  # ≈ 9174
N_PHASE1_STEPS = int(N_TOTAL_STEPS * PHASE1_FRAC)  # ≈ 3670
N_PHASE2_STEPS = N_TOTAL_STEPS - N_PHASE1_STEPS  # ≈ 5504
N_BASELINE_STEPS = N_TOTAL_STEPS  # full budget

# d12 quick-test step counts (just enough to exercise all code paths)
N_D12_PHASE1_STEPS = 300
N_D12_PHASE2_STEPS = 300
N_D12_BASELINE_STEPS = 300

# Model tags (become subdirectories under base_checkpoints/)
TAG_PHASE1 = "part3/d20_ctx512"
TAG_PHASE2 = "part3/d20_ctx2048"
TAG_BASELINE = "part3/d20_baseline"

TAG_D12_PHASE1 = "part3/d12_ctx512"
TAG_D12_PHASE2 = "part3/d12_ctx2048"
TAG_D12_BASELINE = "part3/d12_baseline"

WANDB_PROJECT = "nanochat-part3"
WANDB_RUN_PHASE1 = "p3_phase1"
WANDB_RUN_PHASE2 = "p3_phase2"
WANDB_RUN_BASELINE = "p3_baseline"

# Timeouts
TIMEOUT_PHASE1 = 60 * 60 * 2  # 2 h
TIMEOUT_PHASE2 = 60 * 60 * 3  # 3 h
TIMEOUT_BASELINE = 60 * 60 * 3  # 3 h
TIMEOUT_EVAL = 60 * 60 * 3  # 3 h (3× CORE eval on d20 + longctx)
TIMEOUT_QUICKTEST = 60 * 60 * 1  # 1 h

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
    Phase 1: train picochat at seq=512 for ~40% of the Chinchilla-optimal token budget.

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
    Baseline: train d20 at seq=2048 for the full Chinchilla-optimal budget from scratch.

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
      2. Phase 1: d20 at seq=512, 40% of Chinchilla budget
      3. Phase 2 + Baseline in parallel (separate Modal containers)
      4. Eval + Report: CORE, needle-in-haystack, BPB-by-position, then markdown
    """
    w = 64
    print("\n" + "=" * w)
    print("Part 3: Picochat Context Length Curriculum")
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
    print("Part 3 complete!")
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
