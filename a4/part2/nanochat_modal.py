"""
Part 2 (a4): SFT & Midtraining on d20 Baseline
================================================

Starting point: part3/d20_baseline — the plain d20 model trained at ctx=2048 from
scratch in Part 3, confirmed by the Part 4 LaTeX to be the best-performing d20 variant
on CORE (0.2460) ahead of the context-curriculum model (0.2358) and MTP-2+curriculum
(0.1760).  The LaTeX conclusion (part4.tex line 439) explicitly recommends this
checkpoint as the base for SFT.

Goal: evaluate how SFT and continued pretraining (midtraining) interact with a
strong pretrained base, and whether augmenting the SFT data mix with math-focused
datasets improves ChatCORE scores beyond the original nanochat data mixture.

Stages
------
  stage_setup          Download identity_conversations.jsonl to the volume   (CPU)
  stage_midtrain       Full-epoch midtraining from part3/d20_baseline (~650 steps) (H100:8)
  stage_sft_original   SFT on d20 baseline, original data mix                (H100:8)
  stage_sft_augmented  SFT on d20 baseline, augmented mix (+math datasets)   (H100:8)
  stage_eval           ChatCORE + BPB eval on all four output checkpoints    (H100:4)

Patches applied at image-build time (submodule stays clean)
-----------------------------------------------------------
  a3/part2_mtp/patches/gpt.py           — MTP-aware GPTConfig so the d20 checkpoint
                                           (which carries mtp_k=0 in its config dict)
                                           loads without KeyError.
  a3/part2_mtp/patches/base_eval.py     — unified CORE+BPB evaluation.
  a4/part2/patches/mid_train.py         — nanochat commit 348fbb3 mid_train.py
                                           (dedicated midtraining script, saves to
                                           mid_checkpoints/) + --wandb-project arg.
  a4/part2/patches/checkpoint_manager.py — restores "mid": "mid_checkpoints" entry
                                           that was removed after commit 348fbb3,
                                           allowing base_eval.py to load mid ckpts.
  a4/part2/patches/chat_sft.py          — adds --wandb-project, --base-model-tag,
                                           --use-augmented-data to the SFT script.
  a4/part2/tasks/metamath.py            — MetaMathQA Task subclass (395K rows).
  a4/part2/tasks/numina_math.py         — NuminaMathCoT Task subclass (100K rows).

WandB project: nanochat-a4-part2

Cost reference (H100 pricing ~$3.87/GPU/hr, node ~$31/hr for 8× H100)
-----------------------------------------------------------------------
  Midtraining  (1 epoch ~650 steps, H100:8) : ~$1
  SFT original (1 epoch,           H100:8) : ~$6
  SFT augmented(1 epoch,           H100:8) : ~$9  (+MetaMathQA 395K + NuminaMath 100K)
  Eval         (H100:4)                    : ~$3
  Total                                    : ~$19

Usage
-----
  # 1. One-time data setup (CPU, fast)
  modal run a4/part2/nanochat_modal.py::stage_setup

  # 2. Midtraining (can run in parallel with SFTs)
  modal run a4/part2/nanochat_modal.py::stage_midtrain

  # 3. Original SFT
  modal run a4/part2/nanochat_modal.py::stage_sft_original

  # 4. Augmented SFT
  modal run a4/part2/nanochat_modal.py::stage_sft_augmented

  # 5. Evaluation
  modal run a4/part2/nanochat_modal.py::stage_eval
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

# ── Base checkpoint ────────────────────────────────────────────────────────────
# part3/d20_baseline: plain d20 trained at ctx=2048 from scratch (no curriculum).
# CORE = 0.2460 — best among d20 variants per the Part 4 LaTeX evaluation table.
# (part3/d20_ctx2048 curriculum scored 0.2358; d20 MTP-2+curriculum scored 0.1760)
BASE_MODEL_TAG = "part3/d20_baseline"

# ── Output tags ───────────────────────────────────────────────────────────────
TAG_MIDTRAIN = "a4/d20_midtrain"
TAG_SFT_ORIG = "a4/d20_sft_orig"
TAG_SFT_AUG = "a4/d20_sft_aug"

# ── WandB ─────────────────────────────────────────────────────────────────────
WANDB_PROJECT = "nanochat-a4-part2"

# ── Training hyperparameters ──────────────────────────────────────────────────
TOTAL_BATCH_SIZE = 524_288
DEPTH = 20
DEVICE_BATCH_SFT = (
    16  # 16 sequences × 2048 tokens × 8 GPUs = 524288 tokens/step (2 grad-accum steps)
)
DEVICE_BATCH_EVAL = 16
MIDTRAIN_STEPS = (
    -1
)  # -1 = full epoch (~848K rows ≈ 650 steps); matches Karpathy speedrun exactly

# ── GPU / device counts ───────────────────────────────────────────────────────
GPU_TRAIN = "H100:8"
GPU_EVAL = "H100:4"
_N_TRAIN_GPUS = 8
_N_EVAL_GPUS = 4

# ── Timeouts ──────────────────────────────────────────────────────────────────
TIMEOUT_SETUP = 60 * 30  # 30 min (download only)
TIMEOUT_MIDTRAIN = 60 * 60 * 2  # 2 h  (full epoch ~650 steps on d20, H100:8)
TIMEOUT_SFT = 60 * 60 * 8  # 8 h  (full epoch SFT on ~1M+ rows, H100:8)
TIMEOUT_SFT_AUG = 60 * 60 * 10  # 10 h (augmented mix is larger)
TIMEOUT_EVAL = 60 * 60 * 4  # 4 h  (ChatCORE + BPB on three checkpoints)

# ── Volume / paths ────────────────────────────────────────────────────────────
VOLUME_MOUNT = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"
BASE_DIR = "/data/.cache/nanochat"  # symlinked to NANOCHAT_CACHE inside container

# Identity conversations URL (Karpathy's S3)
IDENTITY_JSONL_URL = (
    "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
)

# =============================================================================
# MODAL PRIMITIVES
# =============================================================================

app = modal.App("nanochat-a4-part2")
volume = Volume.from_name("nanochat-vol", create_if_missing=True)
secret = Secret.from_name("nanochat-secrets")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# a4 has its own nanochat submodule pinned to the commit the patches were written against
_NANOCHAT_DIR = os.path.join(_THIS_DIR, "..", "nanochat")
# Reuse MTP patches from a3/part2_mtp — gpt.py / base_eval.py
_MTP_PATCHES = os.path.join(_THIS_DIR, "..", "..", "a3", "part2_mtp", "patches")
# Our own patches for this part
_OWN_PATCHES = os.path.join(_THIS_DIR, "patches")
_OWN_TASKS = os.path.join(_THIS_DIR, "tasks")

image = (
    ModalImage.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git", "build-essential", "curl", "wget", "unzip")
    # Nanochat submodule (source of truth for all scripts / tasks / package)
    .add_local_dir(
        local_path=_NANOCHAT_DIR,
        remote_path="/root/nanochat",
        copy=True,
    )
    # ── MTP patches (keep submodule clean; overwrite at image-build time) ──────
    # gpt.py: adds mtp_k / rope_type to GPTConfig — required so the d20 checkpoint
    # (saved with mtp_k=0 in its config) deserialises without a KeyError.
    .add_local_file(
        local_path=os.path.join(_MTP_PATCHES, "gpt.py"),
        remote_path="/root/nanochat/nanochat/gpt.py",
        copy=True,
    )
    # base_eval.py: unified CORE + BPB eval with --eval= flag + --model-source flag
    # (local patch adds --model-source so mid_checkpoints/ can be evaluated)
    .add_local_file(
        local_path=os.path.join(_OWN_PATCHES, "base_eval.py"),
        remote_path="/root/nanochat/scripts/base_eval.py",
        copy=True,
    )
    # ── Part-2-specific patches ───────────────────────────────────────────────
    # mid_train.py: nanochat commit 348fbb3 dedicated midtraining script.
    # Saves to mid_checkpoints/<model-tag>/ (not base_checkpoints/).
    # Patch adds --wandb-project arg.
    .add_local_file(
        local_path=os.path.join(_OWN_PATCHES, "mid_train.py"),
        remote_path="/root/nanochat/scripts/mid_train.py",
        copy=True,
    )
    # checkpoint_manager.py: restores "mid": "mid_checkpoints" that was removed
    # from nanochat HEAD after the midtraining script was deleted.  Required so
    # base_eval.py can load checkpoints from mid_checkpoints/.
    .add_local_file(
        local_path=os.path.join(_OWN_PATCHES, "checkpoint_manager.py"),
        remote_path="/root/nanochat/nanochat/checkpoint_manager.py",
        copy=True,
    )
    # Patched chat_sft.py: --wandb-project, --base-model-tag, --use-augmented-data
    .add_local_file(
        local_path=os.path.join(_OWN_PATCHES, "chat_sft.py"),
        remote_path="/root/nanochat/scripts/chat_sft.py",
        copy=True,
    )
    # MetaMathQA Task subclass (395K augmented GSM8K+MATH rows)
    .add_local_file(
        local_path=os.path.join(_OWN_TASKS, "metamath.py"),
        remote_path="/root/nanochat/tasks/metamath.py",
        copy=True,
    )
    # NuminaMathCoT Task subclass (100K competition math CoT rows, sub-sampled)
    .add_local_file(
        local_path=os.path.join(_OWN_TASKS, "numina_math.py"),
        remote_path="/root/nanochat/tasks/numina_math.py",
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
    """Distributed launch via torchrun."""
    args = args or []
    args_str = (" -- " + " ".join(args)) if args else ""
    _run(
        f"cd /root/nanochat && "
        f"PYTHONPATH=/root/nanochat:$PYTHONPATH "
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m {module}{args_str}"
    )


def _setup_cache() -> None:
    """Create NANOCHAT_CACHE and symlink it to BASE_DIR."""
    os.makedirs(NANOCHAT_CACHE, exist_ok=True)
    if not os.path.exists(BASE_DIR):
        os.makedirs(os.path.dirname(BASE_DIR), exist_ok=True)
        os.symlink(NANOCHAT_CACHE, BASE_DIR)


def _find_last_step(model_tag: str) -> int:
    """Return the highest checkpoint step saved under base_checkpoints/<model_tag>/."""
    import glob

    ckpt_dir = os.path.join(NANOCHAT_CACHE, "base_checkpoints", model_tag)
    volume.reload()
    files = glob.glob(os.path.join(ckpt_dir, "model_*.pt"))
    if not files:
        raise RuntimeError(f"No checkpoints found under {ckpt_dir}")
    return max(int(os.path.basename(f).split("_")[1].split(".")[0]) for f in files)


def _find_last_sft_step(model_tag: str) -> int:
    """Return the highest checkpoint step under chatsft_checkpoints/<model_tag>/."""
    import glob

    ckpt_dir = os.path.join(NANOCHAT_CACHE, "chatsft_checkpoints", model_tag)
    volume.reload()
    files = glob.glob(os.path.join(ckpt_dir, "model_*.pt"))
    if not files:
        raise RuntimeError(f"No SFT checkpoints found under {ckpt_dir}")
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
# STAGE 1: SETUP  — download identity conversations to volume
# =============================================================================


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    timeout=TIMEOUT_SETUP,
)
def stage_setup() -> None:
    """
    Download identity_conversations.jsonl to the shared volume.

    This file teaches the model its name, creator, and basic self-awareness facts.
    It is a ~1000-row JSONL file served from Karpathy's S3 bucket and is included
    twice in the SFT data mixture (CustomJSON × 2) to give it sufficient weight.
    """
    _setup_cache()
    identity_dest = os.path.join(NANOCHAT_CACHE, "identity_conversations.jsonl")
    print(f"Downloading identity conversations → {identity_dest}")
    _run(f"curl -L -o {identity_dest} {IDENTITY_JSONL_URL}")
    volume.commit()
    print("Setup complete.")


# =============================================================================
# STAGE 2: MIDTRAINING — 300-step continued pretraining from d20 baseline
# =============================================================================


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_MIDTRAIN,
)
def stage_midtrain() -> None:
    """
    300-step continued pretraining (midtraining) from part3/d20_baseline.

    Methodology
    -----------
    Uses the dedicated mid_train.py script (nanochat commit 348fbb3), which runs
    one full epoch of the SFT data mixture (SmolTalk, MMLU, GSM8K, identity,
    SimpleSpelling, SpellingBee — ~848K rows) but with a fresh optimizer and no
    chat-format loss masking.  The script loads the base checkpoint from
    base_checkpoints/ and saves the result to mid_checkpoints/<TAG_MIDTRAIN>/.

    This differs from a plain pretraining resume: mid_train.py uses the chat/SFT
    data distribution rather than web-crawl text, acting as a "bridge" between
    the pretrained d20_baseline and the downstream SFT data.

    The resulting checkpoint is the input for SFT (both original and augmented),
    exactly as in the Karpathy speedrun where SFT always loads from mid_checkpoints/.

    The resulting checkpoint (a4/d20_midtrain) also provides a comparison point:
        d20_baseline (CORE 0.2460)  vs  d20_midtrain  (CORE measured in stage_eval)

    GPU     : H100:8
    Steps   : 1 full epoch (~848K rows ≈ 650 steps); no --num-iterations, matches speedrun
    Tag     : {TAG_MIDTRAIN}  → mid_checkpoints/{TAG_MIDTRAIN}/
    """
    _setup_cache()
    volume.reload()
    print(
        f"Midtraining: loading {BASE_MODEL_TAG} from base_checkpoints/, "
        f"running full epoch → mid_checkpoints/{TAG_MIDTRAIN}"
    )
    _torchrun(
        "scripts.mid_train",
        [
            f"--model-tag={BASE_MODEL_TAG}",  # load from base_checkpoints/part3/d20_baseline
            f"--save-tag={TAG_MIDTRAIN}",  # save to mid_checkpoints/a4/d20_midtrain
            f"--device-batch-size={DEVICE_BATCH_SFT}",
            f"--total-batch-size={TOTAL_BATCH_SIZE}",
            # no --num-iterations: runs a full epoch (~848K rows), matches Karpathy speedrun
            "--eval-every=50",
            f"--wandb-project={WANDB_PROJECT}",
            "--run=a4_d20_midtrain",
        ],
        nproc=_N_TRAIN_GPUS,
    )
    volume.commit()
    print(f"Midtraining done.  Tag: {TAG_MIDTRAIN}  → mid_checkpoints/{TAG_MIDTRAIN}/")


# =============================================================================
# STAGE 3: SFT ORIGINAL — original nanochat data mixture
# =============================================================================


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_SFT,
)
def stage_sft_original() -> None:
    """
    Supervised fine-tuning on part3/d20_baseline (CORE=0.2460, best d20 variant)
    with the original nanochat data mixture (identical to the nanochat speedrun
    SFT configuration).

    Data mixture
    ------------
      SmolTalk           (460K rows) — general multi-turn conversations
      CustomJSON × 2     (  1K rows) — identity / self-awareness conversations
      MMLU × 3           (300K rows) — multiple-choice knowledge questions
      GSM8K × 4          ( 32K rows) — math word problems (tool-use format)
      SimpleSpelling     (200K rows) — simple character-level spelling tasks
      SpellingBee        ( 80K rows) — harder spelling / letter-counting tasks
      Total              ~1.07M rows

    The checkpoint is saved to chatsft_checkpoints/a4/d20_sft_orig/ and evaluated
    every 200 steps via ChatCORE (logged to W&B project nanochat-a4-part2).

    GPU  : H100:8
    Tag  : {TAG_SFT_ORIG}
    """
    _setup_cache()
    volume.reload()
    print(f"SFT original — mid: {TAG_MIDTRAIN}  →  save: {TAG_SFT_ORIG}")
    _torchrun(
        "scripts.chat_sft",
        [
            f"--run=a4_d20_sft_orig",
            f"--wandb-project={WANDB_PROJECT}",
            f"--model-tag={TAG_SFT_ORIG}",
            f"--model-source=mid",  # load from mid_checkpoints/ (Karpathy default)
            f"--base-model-tag={TAG_MIDTRAIN}",  # load mid_checkpoints/a4/d20_midtrain
            f"--device-batch-size={DEVICE_BATCH_SFT}",
            f"--total-batch-size={TOTAL_BATCH_SIZE}",
            "--mmlu-epochs=3",
            "--gsm8k-epochs=4",
            "--chatcore-every=200",
            "--eval-every=200",
        ],
        nproc=_N_TRAIN_GPUS,
    )
    volume.commit()
    print(f"SFT original done.  Tag: {TAG_SFT_ORIG}")


# =============================================================================
# STAGE 4: SFT AUGMENTED — original + UltraChat200k
# =============================================================================


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_SFT_AUG,
)
def stage_sft_augmented() -> None:
    """
    Supervised fine-tuning on part3/d20_baseline (CORE=0.2460, best d20 variant)
    with the augmented data mixture: original nanochat data PLUS math datasets.

    Augmentation rationale
    ----------------------
    MetaMathQA (meta-math/MetaMathQA, 395K rows, Yu et al. 2023):
      Augments GSM8K + MATH training sets via 8 strategies — rephrasing, backwards
      reasoning (FOBAR), self-verification (SV), and answer augmentation (AnsAug).
      Fine-tuning on MetaMathQA improved GSM8K from 14.6% → 66.5% on LLaMA-2-7B
      and 40% → 77.7% on Mistral-7B.  The diverse reasoning perspectives train the
      model to approach problems from multiple angles beyond the base GSM8K × 4.

    NuminaMathCoT (AI-MO/NuminaMath-CoT, 100K rows sub-sampled, Apache 2.0):
      Competition math (AMC, AIME, Olympiad) with structured chain-of-thought.
      Powered the 2024 AI Mathematical Olympiad winning entry.  Teaches explicit
      step-by-step reasoning style.  We sub-sample to 100K to maintain balance —
      the hardest olympiad problems are beyond a sub-100M-param model, but the CoT
      formatting style transfers to all reasoning tasks.

    Data mixture
    ------------
      SmolTalk           (460K rows)
      CustomJSON × 2     (  1K rows)
      MMLU × 3           (300K rows)
      GSM8K × 4          ( 32K rows)
      SimpleSpelling     (200K rows)
      SpellingBee        ( 80K rows)
      MetaMathQA         (395K rows)  ← new
      NuminaMathCoT      (100K rows)  ← new
      Total              ~1.34M rows

    GPU  : H100:8
    Tag  : {TAG_SFT_AUG}
    """
    _setup_cache()
    volume.reload()
    print(f"SFT augmented — mid: {TAG_MIDTRAIN}  →  save: {TAG_SFT_AUG}")
    _torchrun(
        "scripts.chat_sft",
        [
            f"--run=a4_d20_sft_aug",
            f"--wandb-project={WANDB_PROJECT}",
            f"--model-tag={TAG_SFT_AUG}",
            f"--model-source=mid",  # load from mid_checkpoints/ (Karpathy default)
            f"--base-model-tag={TAG_MIDTRAIN}",  # load mid_checkpoints/a4/d20_midtrain
            f"--device-batch-size={DEVICE_BATCH_SFT}",
            f"--total-batch-size={TOTAL_BATCH_SIZE}",
            "--mmlu-epochs=3",
            "--gsm8k-epochs=4",
            "--chatcore-every=200",
            "--eval-every=200",
            "--use-augmented-data",
        ],
        nproc=_N_TRAIN_GPUS,
    )
    volume.commit()
    print(f"SFT augmented done.  Tag: {TAG_SFT_AUG}")


# =============================================================================
# STAGE 5: EVAL — ChatCORE + BPB on all output checkpoints
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
    Comprehensive evaluation of all Part 2 checkpoints.

    Evaluations performed
    ---------------------
      1. part3/d20_baseline (baseline)  — CORE + BPB via base_eval.py
      2. a4/d20_midtrain               — CORE + BPB via base_eval.py
      3. a4/d20_sft_orig  (SFT)        — ChatCORE (ARC, MMLU, GSM8K, HumanEval,
                                          SpellingBee) via chat_eval.py
      4. a4/d20_sft_aug   (SFT+ultra)  — same as above

    Results are saved as JSON to nanochat_cache/a4_part2_eval_results.json on the
    volume for use in the LaTeX report.

    GPU  : H100:4
    """
    _setup_cache()
    volume.reload()
    _ensure_eval_bundle()

    results: dict = {}
    out_path = os.path.join(NANOCHAT_CACHE, "a4_part2_eval_results.json")

    # ── 1: Base model CORE + BPB eval (base_checkpoints) ────────────────────
    print(f"\n{'=' * 60}\nBase eval (CORE+BPB): {BASE_MODEL_TAG}\n{'=' * 60}")
    _torchrun(
        "scripts.base_eval",
        [
            f"--model-tag={BASE_MODEL_TAG}",
            "--eval=core,bpb",
            "--max-per-task=500",
            f"--device-batch-size={DEVICE_BATCH_EVAL}",
        ],
        nproc=_N_EVAL_GPUS,
    )
    eval_dir = os.path.join(NANOCHAT_CACHE, "base_eval")
    slug = BASE_MODEL_TAG.replace("/", "-")
    if os.path.isdir(eval_dir):
        csvs = sorted(f for f in os.listdir(eval_dir) if f.startswith(slug))
        results[BASE_MODEL_TAG] = csvs[-1] if csvs else None
    print(f"  → {BASE_MODEL_TAG}: {results.get(BASE_MODEL_TAG)}")

    # ── 2: Midtrained model CORE + BPB eval (mid_checkpoints) ───────────────
    # checkpoint_manager.py is patched to support "mid" source → mid_checkpoints/
    print(f"\n{'=' * 60}\nMid eval (CORE+BPB): {TAG_MIDTRAIN}\n{'=' * 60}")
    _torchrun(
        "scripts.base_eval",
        [
            f"--model-tag={TAG_MIDTRAIN}",
            f"--model-source=mid",
            "--eval=core,bpb",
            "--max-per-task=500",
            f"--device-batch-size={DEVICE_BATCH_EVAL}",
        ],
        nproc=_N_EVAL_GPUS,
    )
    slug = TAG_MIDTRAIN.replace("/", "-")
    if os.path.isdir(eval_dir):
        csvs = sorted(f for f in os.listdir(eval_dir) if f.startswith(slug))
        results[TAG_MIDTRAIN] = csvs[-1] if csvs else None
    print(f"  → {TAG_MIDTRAIN}: {results.get(TAG_MIDTRAIN)}")

    # ── 3 & 4: SFT ChatCORE evals ────────────────────────────────────────────
    for sft_tag in [TAG_SFT_ORIG, TAG_SFT_AUG]:
        print(f"\n{'=' * 60}\nChatCORE eval (SFT): {sft_tag}\n{'=' * 60}")
        _torchrun(
            "scripts.chat_eval",
            [
                "-i",
                "sft",
                "-g",
                sft_tag,
            ],
            nproc=_N_EVAL_GPUS,
        )
        results[sft_tag] = "chat_eval complete"
        print(f"  → {sft_tag}: done")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"\nAll Part 2 (a4) evals done.  Results index: {out_path}")


# =============================================================================
# SMOKE TEST — validates the full pipeline in ~20 min before the big runs
# =============================================================================


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu="H100:2",
    timeout=60 * 60 * 1,  # 1 h — should finish in ~20 min
)
def quick_test() -> None:
    """
    End-to-end smoke test that validates every component before the real runs.

    What is tested
    --------------
    1. MTP-patched gpt.py loads part3/d20_baseline correctly (the checkpoint has no
       mtp_k in its config dict; the patched GPTConfig must supply the default of 0).
    2. Midtraining path: mid_train.py (commit 348fbb3) loads d20_baseline from
       base_checkpoints/, runs 20 steps, and saves to mid_checkpoints/.
    3. SFT path with --base-model-tag: patched chat_sft.py loads from the d20_baseline
       tag while saving to a distinct smoke tag.
    4. Augmented data path: --use-augmented-data imports and iterates MetaMathQA and
       NuminaMathCoT without error (confirms HuggingFace download + Task interface).

    All runs use trivially small batch sizes and step counts so the test finishes
    quickly.  On H100:2 this takes roughly 15–20 minutes total.

    GPU  : H100:2  (d20 with device_batch_size=4 fits comfortably)
    """
    _setup_cache()
    volume.reload()

    TAG_SMOKE_MIDTRAIN = "a4/smoke_midtrain"
    TAG_SMOKE_SFT_ORIG = "a4/smoke_sft_orig"
    TAG_SMOKE_SFT_AUG = "a4/smoke_sft_aug"
    SMOKE_BATCH = 4  # sequences per GPU
    SMOKE_TOTAL_BATCH = (
        65_536  # total tokens per step (4 × 2048 × 2 GPUs × 4 grad-accum)
    )
    SMOKE_NPROC = 2

    # ── Download identity file so CustomJSON in chat_sft doesn't error ─────────
    identity_dest = os.path.join(NANOCHAT_CACHE, "identity_conversations.jsonl")
    if not os.path.exists(identity_dest):
        print("Downloading identity_conversations.jsonl for smoke test...")
        _run(f"curl -L -o {identity_dest} {IDENTITY_JSONL_URL}")

    # ── 1. Midtraining smoke: load d20_baseline via mid_train.py → 20 steps ───
    print(
        f"\n=== Smoke 1/3: midtraining (20 steps via mid_train.py, model={BASE_MODEL_TAG}) ==="
    )
    _torchrun(
        "scripts.mid_train",
        [
            f"--model-tag={BASE_MODEL_TAG}",  # load from base_checkpoints/part3/d20_baseline
            f"--save-tag={TAG_SMOKE_MIDTRAIN}",  # save to mid_checkpoints/a4/smoke_midtrain
            f"--device-batch-size={SMOKE_BATCH}",
            f"--total-batch-size={SMOKE_TOTAL_BATCH}",
            "--num-iterations=20",
            "--eval-every=-1",
            f"--wandb-project={WANDB_PROJECT}",
            "--run=smoke_midtrain",
        ],
        nproc=SMOKE_NPROC,
    )
    volume.commit()
    print("Smoke 1/3 passed: mid_train.py → mid_checkpoints/ OK.")

    # ── 2. SFT original smoke: loads from smoke_midtrain (same 2-GPU run) ────
    # Optimizer state is compatible (both midtrain smoke and SFT smoke use 2 GPUs),
    # so --load-optimizer=1 (default) works correctly here.
    print(
        f"\n=== Smoke 2/3: SFT original (10 steps, --model-source=mid --base-model-tag={TAG_SMOKE_MIDTRAIN}) ==="
    )
    _torchrun(
        "scripts.chat_sft",
        [
            "--run=smoke_sft_orig",
            f"--wandb-project={WANDB_PROJECT}",
            f"--model-tag={TAG_SMOKE_SFT_ORIG}",
            f"--model-source=mid",
            f"--base-model-tag={TAG_SMOKE_MIDTRAIN}",
            f"--device-batch-size={SMOKE_BATCH}",
            f"--total-batch-size={SMOKE_TOTAL_BATCH}",
            "--num-iterations=10",
            "--chatcore-every=-1",
            "--eval-every=5",
        ],
        nproc=SMOKE_NPROC,
    )
    volume.commit()
    print("Smoke 2/3 passed: SFT original loads from mid_checkpoints/ OK.")

    # ── 3. SFT augmented smoke: same, with math datasets ─────────────────────
    print(f"\n=== Smoke 3/3: SFT augmented (10 steps, --use-augmented-data) ===")
    _torchrun(
        "scripts.chat_sft",
        [
            "--run=smoke_sft_aug",
            f"--wandb-project={WANDB_PROJECT}",
            f"--model-tag={TAG_SMOKE_SFT_AUG}",
            f"--model-source=mid",
            f"--base-model-tag={TAG_SMOKE_MIDTRAIN}",
            f"--device-batch-size={SMOKE_BATCH}",
            f"--total-batch-size={SMOKE_TOTAL_BATCH}",
            "--num-iterations=10",
            "--chatcore-every=-1",
            "--eval-every=5",
            "--use-augmented-data",
        ],
        nproc=SMOKE_NPROC,
    )
    volume.commit()
    print(
        "Smoke 3/3 passed: SFT augmented loads from mid_checkpoints/ + math datasets OK."
    )

    print("\n" + "=" * 60)
    print("All smoke tests passed!  Pipeline is ready for full runs.")
    print(f"  Base checkpoint        : {BASE_MODEL_TAG}")
    print(f"  MTP gpt.py patch       : OK (mtp_k defaulted to 0)")
    print(f"  mid_train.py           : OK (full epoch → mid_checkpoints/)")
    print(f"  SFT loads from mid     : OK (--model-source=mid)")
    print(f"  --use-augmented-data   : OK (MetaMathQA + NuminaMathCoT)")
    print("=" * 60)


# =============================================================================
# MAIN ENTRYPOINT — full sequential pipeline
# =============================================================================


@app.local_entrypoint()
def main() -> None:
    """
    Run the full a4/part2 pipeline in order.

    Stages 3 and 4 (SFT original + augmented) can safely run in parallel if
    sufficient GPU quota is available.  They are run sequentially here for
    simplicity and to avoid thundering-herd quota issues.
    """
    w = 64
    print("\n" + "=" * w)
    print("a4/part2: SFT & Midtraining on d20 baseline")
    print(f"  Base checkpoint : {BASE_MODEL_TAG}")
    print(f"  Midtrain steps  : {MIDTRAIN_STEPS}")
    print(f"  SFT original    → {TAG_SFT_ORIG}")
    print(f"  SFT augmented   → {TAG_SFT_AUG}")
    print(f"  WandB project   : {WANDB_PROJECT}")
    print("=" * w + "\n")

    print("[1] Setup — downloading identity conversations...")
    stage_setup.remote()

    print("[2] Midtraining...")
    stage_midtrain.remote()

    print("[3] SFT original...")
    stage_sft_original.remote()

    print("[4] SFT augmented...")
    stage_sft_augmented.remote()

    print("[5] Evaluation...")
    stage_eval.remote()

    print("\n" + "=" * w)
    print("a4/part2 complete!")
    print(f"  Midtrain ckpt  : {TAG_MIDTRAIN}")
    print(f"  SFT orig ckpt  : {TAG_SFT_ORIG}")
    print(f"  SFT aug  ckpt  : {TAG_SFT_AUG}")
    print(f"  WandB project  : {WANDB_PROJECT}")
    print("=" * w + "\n")
