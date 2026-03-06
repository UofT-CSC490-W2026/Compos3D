"""
Part 4: Training Your Final Nanochat  (d20 + MTP-2 + Context Curriculum)
=========================================================================

Experiments
-----------
  1. Scaling-law anchor runs — full Chinchilla budget at d8 and d12.
       d8 : ~42M scaling params  → ~440M tokens  → ~840 steps   (H100:2)
       d12: ~110M scaling params → ~1.16B tokens  → ~2204 steps  (H100:4)
     The d16 baseline result is *reused* from Part 2 (tag a2mtp/d16_baseline).

  2. Final nanochat d20 + MTP-2 + context curriculum.
       Phase 1: ctx=512,  40% budget (~3 669 steps)
       Phase 2: ctx=2048, 60% budget (~5 505 steps, warm-start from Phase 1)
     Tags: part4/d20_mtp2_ctx512, part4/d20_mtp2_ctx2048

References (no new training needed)
------------------------------------
  - d16 baseline            : Part 2  tag a2mtp/d16_baseline
  - d16 curriculum Phase 2  : Part 3  tag part3/d16_ctx2048  (trained + eval'd in Part 3)
  - d20 curriculum-only     : Part 3  tags part3/d20_ctx512, part3/d20_ctx2048
  - Hyperparameter sweep    : Part 2  project part2_sweep  (reused for Part 4)

Patches
-------
  Reuses a3/part2_mtp/patches/  (already supports --mtp-k and --rope-type).
  The nanochat submodule is never modified; patches overwrite files inside the
  container at image-build time.

WandB project: nanochat-part4

Cost reference (H100 pricing)
-------------------------------
  d8  scaling   (~840 steps, H100:2)  : ~$3
  d12 scaling   (~2204 steps, H100:4) : ~$12
  d20 P1        (~3669 steps, H100:8) : ~$24
  d20 P2        (~5505 steps, H100:8) : ~$36
  Eval  (3 ckpts, H100:4)             : ~$6
  Total                               : ~$81
  (d16 curriculum cost covered by Part 3)
"""

import json
import os
import subprocess

import modal
from modal import App
from modal import Image as ModalImage
from modal import Secret, Volume

# =============================================================================
# CONFIGURATION
# =============================================================================

# ── Shared ────────────────────────────────────────────────────────────────────
TOTAL_BATCH_SIZE = 524_288
TARGET_PARAM_DATA_RATIO = 10.5  # Chinchilla ratio used throughout nanochat
PHASE1_FRAC = 0.40  # 40 % of budget at short context
VOCAB_SIZE = 32_768

# MTP settings for the final nanochat
MTP_K = 2  # MTP-2
MTP_ROPE = "rope"  # standard RoPE (not YaRN — see Part 2 discussion)


def _sp(d: int) -> int:
    """Scaling parameters for depth d (using nanochat aspect ratio 64)."""
    n = d * 64
    return d * 12 * n * n + VOCAB_SIZE * n


def _ct(d: int) -> int:
    """Chinchilla-optimal token budget for depth d."""
    return int(TARGET_PARAM_DATA_RATIO * _sp(d))


def _ns(d: int, batch: int = TOTAL_BATCH_SIZE) -> int:
    """Chinchilla-optimal step count for depth d."""
    return _ct(d) // batch


# ── Scaling-law runs ──────────────────────────────────────────────────────────
DEPTH_D8 = 8
DEPTH_D12 = 12

CHINCHILLA_D8 = _ct(8)  # ≈  440 M tokens  (~42M scaling params)
CHINCHILLA_D12 = _ct(12)  # ≈ 1156 M tokens  (~110M scaling params)
N_STEPS_D8 = _ns(8)  # ≈  840 steps
N_STEPS_D12 = _ns(12)  # ≈ 2204 steps

# ── Nanochat d20 + MTP-2 + curriculum ─────────────────────────────────────────
DEPTH_NANO = 20
CHINCHILLA_D20 = 4_810_000_000  # 10.5 × ~458M  (same as original Part 3)
N_TOTAL_NANO = CHINCHILLA_D20 // TOTAL_BATCH_SIZE  # 9 174
N_PHASE1_NANO = int(N_TOTAL_NANO * PHASE1_FRAC)  # 3 669
N_PHASE2_NANO = N_TOTAL_NANO - N_PHASE1_NANO  # 5 505

# ── Checkpoint tags ───────────────────────────────────────────────────────────
TAG_D8 = "part4/d8_scaling"
TAG_D12 = "part4/d12_scaling"
TAG_D20_P1 = "part4/d20_mtp2_ctx512"
TAG_D20_P2 = "part4/d20_mtp2_ctx2048"

# Tags from earlier parts — no new training needed for these
TAG_D16_BASELINE = "a2mtp/d16_baseline"  # Part 2 — d16 no-curriculum baseline
TAG_D16_P2 = "part3/d16_ctx2048"  # Part 3 — d16 curriculum Phase 2 (plain, no MTP)
TAG_D20_CURRICULUM = (
    "part3/d20_ctx2048"  # old Part 3 d20 — d20 curriculum-only ablation
)

# ── WandB ─────────────────────────────────────────────────────────────────────
WANDB_PROJECT = "nanochat-part4"

# ── GPU / device batch ────────────────────────────────────────────────────────
GPU_SMALL = "H100:2"  # d8 (tiny model)
GPU_MED = "H100:2"  # d12
GPU_LARGE = "H100:8"  # d16, d20 main runs
GPU_EVAL = "H100:4"

DEVICE_BATCH_D8 = 64  # d8 is tiny — double the batch per GPU is fine
DEVICE_BATCH_D12 = 32
DEVICE_BATCH_P1 = 32  # ctx=512  (d20 Phase 1)
DEVICE_BATCH_P2 = 16  # ctx=2048 (d20 Phase 2) — halved vs Phase 1: MTP-2 retains
# k extra hidden-state slices, ~1.5× memory at seq=2048 on d20

_N_TRAIN_GPUS = 8
_N_EVAL_GPUS = 4

# ── Timeouts ──────────────────────────────────────────────────────────────────
TIMEOUT_D8 = 60 * 60 * 2  # 2 h  (840 steps, d8)
TIMEOUT_D12 = 60 * 60 * 8  # 8 h  (2204 steps, d12, H100:2)
TIMEOUT_P1_NANO = 60 * 60 * 3  # 3 h  (3669 steps, d20 Phase 1)
TIMEOUT_P2_NANO = 60 * 60 * 5  # 5 h  (5505 steps, d20 Phase 2)
TIMEOUT_EVAL = 60 * 60 * 3  # 3 h  (eval checkpoints)
TIMEOUT_SMOKE = 60 * 60 * 1  # 1 h

# ── Volume / cache ────────────────────────────────────────────────────────────
VOLUME_MOUNT = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"
BASE_DIR = "/data/.cache/nanochat"

# =============================================================================
# MODAL PRIMITIVES
# =============================================================================

app = modal.App("nanochat-part4")
volume = Volume.from_name("nanochat-vol", create_if_missing=True)
secret = Secret.from_name("nanochat-secrets")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_NANOCHAT_DIR = os.path.join(_THIS_DIR, "..", "nanochat")
# Reuse Part 2's patches — already support --mtp-k, --rope-type, --yarn-scale
_PATCHES_DIR = os.path.join(_THIS_DIR, "..", "part2_mtp", "patches")

image = (
    ModalImage.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git", "build-essential", "curl", "wget", "unzip")
    .add_local_dir(
        local_path=_NANOCHAT_DIR,
        remote_path="/root/nanochat",
        copy=True,
    )
    # Apply patches — overwrite nanochat internals so submodule stays clean
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
    """Shell out to bash, stream output, raise on failure."""
    print(f"\n>>>  {cmd}\n")
    result = subprocess.run(["bash", "-c", cmd], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command exited {result.returncode}:\n  {cmd}")


def _torchrun(module: str, args: list | None = None, *, nproc: int) -> None:
    """Distributed training via torchrun."""
    args = args or []
    args_str = (" -- " + " ".join(args)) if args else ""
    _run(
        f"cd /root/nanochat && "
        f"PYTHONPATH=/root/nanochat:$PYTHONPATH "
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m {module}{args_str}"
    )


def _setup_cache() -> None:
    """Create the cache directory and symlink to BASE_DIR."""
    os.makedirs(NANOCHAT_CACHE, exist_ok=True)
    if not os.path.exists(BASE_DIR):
        os.makedirs(os.path.dirname(BASE_DIR), exist_ok=True)
        os.symlink(NANOCHAT_CACHE, BASE_DIR)


def _find_last_step(model_tag: str) -> int:
    """Return the highest checkpoint step saved under base_checkpoints/<model_tag>/.

    Checkpoints are written as flat files model_{step:06d}.pt inside the tag directory.
    """
    import glob

    ckpt_dir = os.path.join(NANOCHAT_CACHE, "base_checkpoints", model_tag)
    volume.reload()
    files = glob.glob(os.path.join(ckpt_dir, "model_*.pt"))
    if not files:
        raise RuntimeError(f"No checkpoints found under {ckpt_dir}")
    return max(int(os.path.basename(f).split("_")[1].split(".")[0]) for f in files)


def _ensure_eval_bundle() -> None:
    """Download the eval bundle if not already cached on the volume."""
    eval_bundle_dir = os.path.join(NANOCHAT_CACHE, "eval_bundle")
    if not os.path.isdir(eval_bundle_dir):
        print("Downloading eval bundle...")
        zip_path = "/tmp/eval_bundle.zip"
        _run(
            f"curl -L -o {zip_path} "
            "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
        )
        _run(f"unzip -q {zip_path} -d {NANOCHAT_CACHE} && rm {zip_path}")
        volume.commit()


# =============================================================================
# SCALING-LAW RUNS — d8 and d12 at full Chinchilla budget
# =============================================================================
#
# These two short runs supply BPB and CORE anchor points for the scaling-law
# fit  log(BPB) = log(a) + b × log(N).  The d16 baseline from Part 2
# (tag a2mtp/d16_baseline) provides the third anchor.  The d20 prediction is
# made before running Part 4's main training and compared afterwards.


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_SMALL,
    timeout=TIMEOUT_D8,
)
def stage_scaling_d8() -> None:
    """
    d8 scaling-law anchor: full Chinchilla run (~840 steps, H100:2).

    Scaling params : ~42 M   (8 × 12 × 512² + 32768 × 512)
    Token budget   : ~440 M  (10.5 × 42M)
    Steps          : ~840    (440M / 524288)
    """
    _setup_cache()
    nproc = 2
    print(
        f"d8 scaling run — depth={DEPTH_D8}  steps={N_STEPS_D8}"
        f"  tokens={CHINCHILLA_D8 / 1e6:.0f}M"
    )
    _torchrun(
        "scripts.base_train",
        [
            f"--depth={DEPTH_D8}",
            "--max-seq-len=2048",
            f"--model-tag={TAG_D8}",
            f"--device-batch-size={DEVICE_BATCH_D8}",
            f"--total-batch-size={TOTAL_BATCH_SIZE}",
            f"--num-iterations={N_STEPS_D8}",
            "--save-every=500",
            "--core-metric-every=500",
            "--sample-every=-1",
            f"--wandb-project={WANDB_PROJECT}",
            "--run=p4_d8_scaling",
            "--mtp-k=0",
            "--rope-type=rope",
        ],
        nproc=nproc,
    )
    volume.commit()
    print(f"d8 scaling done.  Tag: {TAG_D8}")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_MED,
    timeout=TIMEOUT_D12,
)
def stage_scaling_d12() -> None:
    """
    d12 scaling-law anchor: full Chinchilla run (~2204 steps, H100:4).

    Scaling params : ~110 M   (12 × 12 × 768² + 32768 × 768)
    Token budget   : ~1.16 B  (10.5 × 110M)
    Steps          : ~2204    (1.16B / 524288)
    """
    _setup_cache()
    nproc = 2
    print(
        f"d12 scaling run — depth={DEPTH_D12}  steps={N_STEPS_D12}"
        f"  tokens={CHINCHILLA_D12 / 1e9:.3f}B"
    )
    _torchrun(
        "scripts.base_train",
        [
            f"--depth={DEPTH_D12}",
            "--max-seq-len=2048",
            f"--model-tag={TAG_D12}",
            f"--device-batch-size={DEVICE_BATCH_D12}",
            f"--total-batch-size={TOTAL_BATCH_SIZE}",
            f"--num-iterations={N_STEPS_D12}",
            "--save-every=500",
            "--core-metric-every=500",
            "--sample-every=-1",
            f"--wandb-project={WANDB_PROJECT}",
            "--run=p4_d12_scaling",
            "--mtp-k=0",
            "--rope-type=rope",
        ],
        nproc=nproc,
    )
    volume.commit()
    print(f"d12 scaling done.  Tag: {TAG_D12}")


# =============================================================================
# d20 NANOCHAT + MTP-2 + CONTEXT CURRICULUM  — FINAL MODEL
# =============================================================================
#
# Architecture choices:
#   MTP-2   — zero extra params, richer gradient signal per step.
#             Gloeckle et al. (2024) report growing gains with model size;
#             even if the d16 benefit is modest the d20 model should benefit more.
#   YaRN    — NOT included: benefit is mainly beyond the training context
#             length.  At our fixed 2048-token training context, NTK-by-Parts
#             scaling pushes most frequency dimensions back toward base RoPE.
#             Part 2 confirms the MTP-2+YaRN result barely beats MTP-2 alone.
#   Curriculum (ctx 512→2048) — consistent with Part 3 findings that the
#             warm-start does not hurt general CORE scores while preparing the
#             model for longer contexts at a 30% attention-FLOP discount.


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_LARGE,
    timeout=TIMEOUT_P1_NANO,
)
def stage_d20_mtp2_p1(
    depth: int = DEPTH_NANO,
    n_steps: int = N_PHASE1_NANO,
) -> None:
    """
    Nanochat d20 Phase 1: ctx=512, ~3669 steps, MTP-2.

    Scaling params : ~435–458 M
    Token budget   : ~4.81 B   (10.5 × 458M)
    Phase 1 tokens : ~1.92 B   (40 % of 4.81B)
    Phase 1 steps  : ~3669
    """
    _setup_cache()
    print(
        f"d20 MTP-2 Phase 1 — depth={depth}  seq=512  steps={n_steps}  tag={TAG_D20_P1}"
    )
    _torchrun(
        "scripts.base_train",
        [
            f"--depth={depth}",
            "--max-seq-len=512",
            f"--model-tag={TAG_D20_P1}",
            f"--device-batch-size={DEVICE_BATCH_P1}",
            f"--total-batch-size={TOTAL_BATCH_SIZE}",
            f"--num-iterations={n_steps}",
            "--save-every=9999",
            "--core-metric-every=9999",
            "--sample-every=-1",
            f"--wandb-project={WANDB_PROJECT}",
            "--run=p4_d20_mtp2_phase1",
            f"--mtp-k={MTP_K}",
            f"--rope-type={MTP_ROPE}",
        ],
        nproc=_N_TRAIN_GPUS,
    )
    volume.commit()
    print(f"d20 MTP-2 Phase 1 done.  Tag: {TAG_D20_P1}")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_LARGE,
    timeout=TIMEOUT_P2_NANO,
)
def stage_d20_mtp2_p2(
    depth: int = DEPTH_NANO,
    n_steps: int = N_PHASE2_NANO,
) -> None:
    """
    Nanochat d20 Phase 2: ctx=2048, ~5505 steps, MTP-2.
    THIS IS THE FINAL NANOCHAT MODEL.

    Warm-starts from Phase 1 checkpoint; optimizer and dataloader are reset so
    gradient curvature estimates specific to seq=512 do not contaminate Phase 2.

    Phase 2 tokens : ~2.89 B   (60 % of 4.81B)
    Phase 2 steps  : ~5505
    Total steps    : ~9174
    """
    _setup_cache()
    p1_step = _find_last_step(TAG_D20_P1)
    total_iters = p1_step + n_steps
    print(
        f"d20 MTP-2 Phase 2 — depth={depth}  seq=2048  "
        f"warm-start step={p1_step}  target={total_iters}  tag={TAG_D20_P2}"
    )
    _torchrun(
        "scripts.base_train",
        [
            f"--depth={depth}",
            "--max-seq-len=2048",
            f"--model-tag={TAG_D20_P2}",
            f"--resume-model-tag={TAG_D20_P1}",
            f"--resume-from-step={p1_step}",
            "--load-model-only",
            f"--device-batch-size={DEVICE_BATCH_P2}",
            f"--total-batch-size={TOTAL_BATCH_SIZE}",
            f"--num-iterations={total_iters}",
            "--save-every=500",
            "--core-metric-every=2000",
            "--sample-every=-1",
            f"--wandb-project={WANDB_PROJECT}",
            "--run=p4_d20_mtp2_phase2",
            f"--mtp-k={MTP_K}",
            f"--rope-type={MTP_ROPE}",
        ],
        nproc=_N_TRAIN_GPUS,
    )
    volume.commit()
    print(f"FINAL NANOCHAT done.  Tag: {TAG_D20_P2}  steps {p1_step}→{total_iters}")


# =============================================================================
# EVAL — CORE on all Part 4 checkpoints
# =============================================================================


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_EVAL,
    timeout=TIMEOUT_EVAL,
)
def stage_eval() -> None:
    """
    CORE evaluation on all new Part 4 checkpoints:
      - d8  scaling-law anchor          (part4/d8_scaling)
      - d12 scaling-law anchor          (part4/d12_scaling)
      - d20 + MTP-2 curriculum Phase 2  (part4/d20_mtp2_ctx2048) — the final nanochat

    The d16 curriculum Phase 2 (part3/d16_ctx2048) is already evaluated by
    Part 3's stage_eval_and_report; its CORE score is referenced directly in
    the Part 4 LaTeX report without re-running eval here.

    Results are written to nanochat_cache/part4_eval_results.json on the volume.
    """
    _setup_cache()
    _ensure_eval_bundle()

    eval_tags = [
        TAG_D8,
        TAG_D12,
        TAG_D20_P2,  # d20 + MTP-2 + curriculum  — the final nanochat
    ]

    results: dict[str, str | None] = {}
    for tag in eval_tags:
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
        # Locate the output CSV
        eval_dir = os.path.join(NANOCHAT_CACHE, "base_eval")
        slug = tag.replace("/", "-")
        csvs = sorted(f for f in os.listdir(eval_dir) if f.startswith(slug))
        results[tag] = csvs[-1] if csvs else None
        print(f"  → {tag}: {results[tag]}")

    out = os.path.join(NANOCHAT_CACHE, "part4_eval_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"\nAll Part 4 evals done.  Results: {out}")


# =============================================================================
# QUICK SMOKE TEST  — validates d8 train + MTP patch + warm-start in ~25 min
# =============================================================================


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu="H100:2",
    timeout=TIMEOUT_SMOKE,
)
def quick_test() -> None:
    """
    Smoke test: trains d8 for 100 steps (plain), then warm-starts a second
    50-step run with MTP-2 at seq=2048.  Validates:
      - patches applied correctly (--mtp-k flag works)
      - --load-model-only warm-start works at d8
      - WandB logging to nanochat-part4 project
    """
    _setup_cache()
    nproc = 2

    TAG_SMOKE_P1 = "part4/smoke_d8_ctx512"
    TAG_SMOKE_P2 = "part4/smoke_d8_ctx2048"

    print("=== Smoke test Phase 1: d8, seq=512, 100 steps ===")
    _torchrun(
        "scripts.base_train",
        [
            "--depth=8",
            "--max-seq-len=512",
            f"--model-tag={TAG_SMOKE_P1}",
            "--device-batch-size=16",
            "--total-batch-size=65536",
            "--num-iterations=100",
            "--save-every=50",
            "--core-metric-every=9999",
            "--sample-every=-1",
            f"--wandb-project={WANDB_PROJECT}",
            "--run=p4_smoke_p1",
            "--mtp-k=2",
            "--rope-type=rope",
        ],
        nproc=nproc,
    )
    volume.commit()

    p1_step = _find_last_step(TAG_SMOKE_P1)
    total_iters = p1_step + 50

    print("=== Smoke test Phase 2: d8, seq=2048, 50 steps, warm-start ===")
    _torchrun(
        "scripts.base_train",
        [
            "--depth=8",
            "--max-seq-len=2048",
            f"--model-tag={TAG_SMOKE_P2}",
            f"--resume-model-tag={TAG_SMOKE_P1}",
            f"--resume-from-step={p1_step}",
            "--load-model-only",
            "--device-batch-size=8",
            "--total-batch-size=65536",
            f"--num-iterations={total_iters}",
            "--save-every=9999",
            "--core-metric-every=9999",
            "--sample-every=-1",
            f"--wandb-project={WANDB_PROJECT}",
            "--run=p4_smoke_p2",
            "--mtp-k=2",
            "--rope-type=rope",
        ],
        nproc=nproc,
    )
    volume.commit()
    print("Smoke test passed ✓")


# =============================================================================
# EMERGENT ABILITIES — picochat (d16) vs nanochat (d20) text generation
# =============================================================================

_EMERGENT_PROMPTS = [
    "The water cycle is a natural process by which water evaporates from oceans,"
    " rises into the atmosphere, and",
    "The Pythagorean theorem states that in a right triangle, the square of the"
    " hypotenuse equals",
    "Democracy is a form of government in which political power is held by",
    "The main difference between a virus and a bacterium is that viruses",
    "To convert a temperature from Celsius to Fahrenheit, you multiply by 9/5 and",
    "Newton's first law of motion states that an object at rest will remain at rest"
    " unless",
    "Shakespeare wrote the tragedy Hamlet, in which Prince Hamlet seeks revenge"
    " against his uncle Claudius, who",
    "In computer science, an algorithm is a finite sequence of well-defined"
    " instructions that",
    "The mitochondria are often called the powerhouse of the cell because they",
    "The French Revolution began in 1789 when economic hardship and social"
    " inequality led",
    "Photosynthesis is the process by which plants use sunlight, water, and"
    " carbon dioxide to produce",
    "The circumference of a circle is calculated by multiplying pi by",
    "The immune system protects the body against disease by recognising"
    " and destroying",
    "A sonnet is a 14-line poem that typically follows a strict rhyme scheme."
    " Shakespeare's sonnets are famous for",
    "In economics, the law of supply and demand states that when the price of"
    " a good rises,",
]


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    secrets=[secret],
    gpu="H100:1",
    timeout=60 * 30,  # 30 min — single-GPU inference, two models
)
def stage_emergent_abilities(
    max_new_tokens: int = 80,
) -> None:
    """
    Load the d16 baseline (picochat) and d20 baseline (nanochat) models,
    run greedy generation on each prompt, and save results to the volume
    as JSON for later use in the LaTeX report.

    Runs inference inside the uv-managed venv (same pattern as training stages)
    to avoid 'No module named torch' errors when importing from system Python.
    """
    import json
    import textwrap

    volume.reload()
    _setup_cache()

    out_dir = os.path.join(NANOCHAT_CACHE, "report")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "emergent_abilities.json")

    # Serialise prompt list and model tags into the script so there are
    # no import/pickle complications across the subprocess boundary.
    prompts_repr = repr(_EMERGENT_PROMPTS)
    tag_d16 = TAG_D16_BASELINE
    tag_d20 = TAG_D20_CURRICULUM

    script = textwrap.dedent(f"""
        import glob, json, os, sys
        import torch
        from contextlib import nullcontext
        sys.path.insert(0, "/root/nanochat")
        os.environ["BASE_DIR"] = "{NANOCHAT_CACHE}"

        from nanochat.checkpoint_manager import load_model

        PROMPTS = {prompts_repr}
        TAGS = [("{tag_d16}", "picochat_d16"), ("{tag_d20}", "nanochat_d20")]
        MAX_TOK = {max_new_tokens}
        device = torch.device("cuda")
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

        def find_step(tag):
            ckpt_dir = os.path.join("{NANOCHAT_CACHE}", "base_checkpoints", tag)
            files = glob.glob(os.path.join(ckpt_dir, "model_*.pt"))
            if not files:
                raise RuntimeError(f"No checkpoints under {{ckpt_dir}}")
            return max(int(os.path.basename(f).split("_")[1].split(".")[0]) for f in files)

        results = []
        for tag, label in TAGS:
            step = find_step(tag)
            print(f"\\nLoading {{label}} ({{tag}}) @ step {{step}}", flush=True)
            model, tokenizer, _ = load_model("base", device=device, phase="eval",
                                             model_tag=tag, step=step)
            model.eval()
            for prompt in PROMPTS:
                ids = tokenizer.encode(prompt)
                generated = []
                with autocast_ctx:
                    for tok in model.generate(ids, max_tokens=MAX_TOK, temperature=0, seed=42):
                        generated.append(tok)
                cont = tokenizer.decode(generated)
                results.append(dict(model=label, prompt=prompt, continuation=cont,
                                    full=prompt + cont))
                print(f"  [{{label}}] {{prompt[:55]}}...\\n    → {{cont[:100]}}", flush=True)
            del model

        out = "{out_path}"
        with open(out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\\nSaved {{len(results)}} entries → {{out}}")
    """)

    script_path = "/tmp/run_emergent.py"
    with open(script_path, "w") as f:
        f.write(script)

    _run(
        f"cd /root/nanochat && "
        f"PYTHONPATH=/root/nanochat:$PYTHONPATH "
        f"uv run python {script_path}"
    )
    volume.commit()
    print("Done — results written to volume.")


# =============================================================================
# MAIN ENTRYPOINT — full Part 4 pipeline
# =============================================================================


@app.local_entrypoint()
def main() -> None:
    """
    Full Part 4 pipeline (run from a3/).

    NOTE: The d16 picochat plain curriculum is run via Part 3
    (part3/nanochat_modal.py::stage_pretrain_phase1 / stage_pretrain_phase2).
    Part 4 reads its checkpoint from tag 'part3/d16_ctx2048' directly.

      Phase A  — parallel:
        • stage_scaling_d8    (H100:2, ~2 h)
        • stage_scaling_d12   (H100:2, ~8 h)
        • stage_d20_mtp2_p1   (H100:8, ~3 h)

      Phase B  — sequential (after Phase A):
        • stage_d20_mtp2_p2   (H100:8, ~5 h,  warm-start from d20 P1)
          ← FINAL NANOCHAT

      Phase C:
        • stage_eval          (H100:4, ~3 h)
    """
    w = 64
    print("\n" + "=" * w)
    print("Part 4: Final Nanochat  (d20 + MTP-2 + Context Curriculum)")
    print(f"  d8  scaling  : {N_STEPS_D8} steps  ({CHINCHILLA_D8 / 1e6:.0f}M tokens)")
    print(f"  d12 scaling  : {N_STEPS_D12} steps  ({CHINCHILLA_D12 / 1e9:.3f}B tokens)")
    print(
        f"  d20 nanochat : P1={N_PHASE1_NANO} + P2={N_PHASE2_NANO} = {N_TOTAL_NANO} steps"
    )
    print("=" * w + "\n")

    # Phase A — scaling law anchors + d20 Phase 1 in parallel
    print("[A] Scaling d8, d12 + d20 MTP-2 Phase 1 in parallel...")
    h_d8 = stage_scaling_d8.spawn()
    h_d12 = stage_scaling_d12.spawn()
    h_d20p1 = stage_d20_mtp2_p1.spawn()
    h_d8.get()
    h_d12.get()
    h_d20p1.get()

    # Phase B — d20 Phase 2 (final nanochat, must follow d20 Phase 1)
    print("[B] d20 MTP-2 Phase 2 — final nanochat training...")
    stage_d20_mtp2_p2.remote()

    # Phase C — eval
    print("[C] CORE + BPB eval on all Part 4 checkpoints...")
    stage_eval.remote()

    print("\n" + "=" * w)
    print("Part 4 complete!")
    print(f"  Final nanochat checkpoint : {TAG_D20_P2}")
    print(f"  WandB project             : {WANDB_PROJECT}")
    print("=" * w + "\n")
