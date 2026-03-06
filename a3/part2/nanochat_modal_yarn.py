"""
YaRN training wrapper for Part 2.

keep `nanochat/` untouched and overlay patched
files from `part2/patches/` into the Modal image at runtime.
"""

import os
import subprocess

import modal
from modal import Image as ModalImage, Secret, Volume


# Part 2 uses a smaller picochat than the default speedrun.
DEPTH = 16
NUM_SHARDS = 240
GPU_PRETRAIN = "H100:8"
GPU_FINETUNE = "H100:4"
DEVICE_BATCH_SIZE = 8
WANDB_RUN = "dummy"
VOLUME_MOUNT = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"
BASE_DIR = "/data/.cache/nanochat"
PRETRAIN_TIMEOUT_SEC = 60 * 60 * 6
FINETUNE_TIMEOUT_SEC = 60 * 60 * 2
DOWNLOAD_TIMEOUT_SEC = 60 * 90
_N_PRETRAIN_GPUS = int(GPU_PRETRAIN.split(":")[1]) if ":" in GPU_PRETRAIN else 1
_N_FINETUNE_GPUS = int(GPU_FINETUNE.split(":")[1]) if ":" in GPU_FINETUNE else 1
IDENTITY_JSONL_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"

YARN_MAX_SEQ_LEN = 8192
YARN_ORIGINAL_SEQUENCE_LEN = 2048
YARN_SCALE = YARN_MAX_SEQ_LEN / YARN_ORIGINAL_SEQUENCE_LEN
YARN_MODEL_TAG = f"d{DEPTH}-yarn"

PATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patches")

volume = Volume.from_name("nanochat-vol", create_if_missing=True)
secret = Secret.from_name("nanochat-secrets")

# YaRN patch: keep this file self-contained so Modal does not need to import the sibling wrapper.
image = (
    ModalImage.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git", "build-essential", "curl", "wget", "unzip")
    .add_local_dir(
        local_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nanochat"),
        remote_path="/root/nanochat",
        copy=True,
    )
    .add_local_file(
        local_path=os.path.join(PATCH_DIR, "gpt.py"),
        remote_path="/root/nanochat/nanochat/gpt.py",
        copy=True,
    )
    .add_local_file(
        local_path=os.path.join(PATCH_DIR, "checkpoint_manager.py"),
        remote_path="/root/nanochat/nanochat/checkpoint_manager.py",
        copy=True,
    )
    .add_local_file(
        local_path=os.path.join(PATCH_DIR, "base_train.py"),
        remote_path="/root/nanochat/scripts/base_train.py",
        copy=True,
    )
    .add_local_file(
        local_path=os.path.join(PATCH_DIR, "chat_sft.py"),
        remote_path="/root/nanochat/scripts/chat_sft.py",
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
    .env({
        "OMP_NUM_THREADS": "1",
        "NANOCHAT_BASE_DIR": BASE_DIR,
        "HF_HOME": "/data/.cache/huggingface",
    })
    .run_commands("ls /root/nanochat/.venv/bin/python || echo 'VENV NOT FOUND'")
    .run_commands("cd /root/nanochat && uv sync --extra gpu --no-install-project")
)

# YaRN patch: overlay copied files from part2/patches instead of editing nanochat directly.
app = modal.App("nanochat-speedrun-yarn")


def _run(cmd: str) -> None:
    print(f"\n>>>  {cmd}\n")
    result = subprocess.run(["bash", "-c", cmd], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command exited with code {result.returncode}:\n  {cmd}")


def _python(module: str, args: list | None = None, *, cwd: str = "/root/nanochat") -> None:
    args = args or []
    _run(f"cd {cwd} && uv run python -m {module} {' '.join(args)}")


def _torchrun(module: str, args: list | None = None, *, nproc: int) -> None:
    args = args or []
    args_str = (" -- " + " ".join(args)) if args else ""
    cmd = (
        f"cd /root/nanochat && "
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m {module}{args_str}"
    )
    print(cmd)
    _run(cmd)


def _setup_cache() -> None:
    os.makedirs(NANOCHAT_CACHE, exist_ok=True)
    if not os.path.lexists(BASE_DIR):
        os.makedirs("/data/.cache/", exist_ok=True)
        os.symlink(NANOCHAT_CACHE, BASE_DIR)
        print(f"Symlinked {BASE_DIR} -> {NANOCHAT_CACHE}")
    else:
        print(f"Cache symlink already exists: {BASE_DIR}")


def _curl(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"Already cached, skipping: {dest}")
        return
    _run(f"curl -L -o {dest} {url}")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    cpu=8,
    memory=16384,
    timeout=DOWNLOAD_TIMEOUT_SEC,
)
def stage_data_yarn(num_shards: int = NUM_SHARDS) -> None:
    # YaRN patch: rebind data download to this app so Modal can hydrate the function.
    _setup_cache()
    print(f"Downloading {num_shards} FineWeb-EDU shards...")
    _python("nanochat.dataset", [f"-n {num_shards}"])
    volume.commit()
    print(f"Done: {num_shards} shards downloaded.")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu="H100:1",
    timeout=60 * 30,
)
def stage_tokenizer_yarn() -> None:
    # YaRN patch: rebind tokenizer training to this app so the local entrypoint can invoke it.
    _setup_cache()

    tokenizer_path = os.path.join(NANOCHAT_CACHE, "tokenizer.model")
    if os.path.exists(tokenizer_path):
        print("Tokenizer already trained. Skipping tok_train.")
    else:
        print("Training tokenizer on 2B characters...")
        _python("scripts.tok_train", ["--max-chars=2000000000"])
        volume.commit()

    print("Evaluating tokenizer compression ratio...")
    _python("scripts.tok_eval")
    print("Tokenizer ready.")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_PRETRAIN,
    timeout=PRETRAIN_TIMEOUT_SEC,
)
def stage_pretrain_yarn(
    depth: int = DEPTH,
    device_batch_size: int = DEVICE_BATCH_SIZE,
    wandb_run: str = WANDB_RUN,
    max_seq_len: int = YARN_MAX_SEQ_LEN,
    original_seq_len: int = YARN_ORIGINAL_SEQUENCE_LEN,
) -> None:
    _setup_cache()

    print("Resetting training report...")
    _python("nanochat.report", ["reset"])

    yarn_scale = max_seq_len / original_seq_len
    model_tag = f"d{depth}-yarn"
    total_batch_size = device_batch_size * max_seq_len * _N_PRETRAIN_GPUS  # YaRN patch: keep total batch divisible by one full micro-batch.
    print(
        f"Starting YaRN pretraining: depth={depth}, "
        f"device_batch_size={device_batch_size}, nproc={_N_PRETRAIN_GPUS}, "
        f"total_batch_size={total_batch_size}, run={wandb_run}, max_seq_len={max_seq_len}, original_seq_len={original_seq_len}, "
        f"yarn_scale={yarn_scale:.2f}, model_tag={model_tag}"
    )

    _torchrun(
        "scripts.base_train",
        [
            f"--depth={depth}",
            f"--device-batch-size={device_batch_size}",
            f"--total-batch-size={total_batch_size}",
            f"--run={wandb_run}",
            f"--max-seq-len={max_seq_len}",
            "--positional-embedding=yarn",
            f"--yarn-original-sequence-len={original_seq_len}",
            f"--yarn-scale={yarn_scale}",
            f"--model-tag={model_tag}",
            "--save-every=1000",
        ],
        nproc=_N_PRETRAIN_GPUS,
    )

    volume.commit()
    print("YaRN pretraining complete.")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_FINETUNE,
    timeout=FINETUNE_TIMEOUT_SEC,
)
def stage_sft_yarn(wandb_run: str = WANDB_RUN, model_tag: str = YARN_MODEL_TAG) -> None:
    _setup_cache()

    identity_dest = os.path.join(NANOCHAT_CACHE, "identity_conversations.jsonl")
    print("Downloading identity conversations for SFT personality layer...")
    _curl(IDENTITY_JSONL_URL, identity_dest)

    print(f"Running SFT from base model tag {model_tag}...")
    _torchrun(
        "scripts.chat_sft",
        [
            f"--run={wandb_run}",
            f"--model-tag={model_tag}",
        ],
        nproc=_N_FINETUNE_GPUS,
    )

    print("Evaluating SFT checkpoint on task benchmarks...")
    _torchrun(
        "scripts.chat_eval",
        [
            "-i", "sft",
            "-g", model_tag,
        ],
        nproc=_N_FINETUNE_GPUS,
    )

    volume.commit()
    print("YaRN SFT complete.")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_FINETUNE,
    timeout=FINETUNE_TIMEOUT_SEC,
)
def stage_rl_yarn(wandb_run: str = WANDB_RUN, model_tag: str = YARN_MODEL_TAG) -> None:
    _setup_cache()

    print(f"Running RL from SFT model tag {model_tag}...")
    _torchrun(
        "scripts.chat_rl",
        [
            f"--run={wandb_run}",
            f"--model-tag={model_tag}",
        ],
        nproc=_N_FINETUNE_GPUS,
    )

    print("Evaluating RL checkpoint...")
    _torchrun(
        "scripts.chat_eval",
        [
            "-i", "rl",
            "-g", model_tag,
        ],
        nproc=_N_FINETUNE_GPUS,
    )

    volume.commit()
    print("YaRN RL complete.")


@app.local_entrypoint()
def main() -> None:
    width = 64
    print("\n" + "=" * width)
    print("nanochat Speedrun -- Modal Edition (YaRN)")
    print(
        f"  depth={DEPTH}  shards={NUM_SHARDS}  gpu={GPU_PRETRAIN}  "
        f"max_seq_len={YARN_MAX_SEQ_LEN}  yarn_scale={YARN_SCALE:.2f}"
    )
    print("=" * width + "\n")

    print("[0/3] Downloading FineWeb-EDU shards...")
    stage_data_yarn.remote(num_shards=NUM_SHARDS)

    print("[1/3] Training tokenizer...")
    stage_tokenizer_yarn.remote()

    print("[2/3] Pretraining YaRN base model...")
    stage_pretrain_yarn.remote(
        depth=DEPTH,
        device_batch_size=DEVICE_BATCH_SIZE,
        wandb_run=WANDB_RUN,
        max_seq_len=YARN_MAX_SEQ_LEN,
        original_seq_len=YARN_ORIGINAL_SEQUENCE_LEN,
    )

    print("\n" + "=" * width)
    print("YaRN pretraining complete.")
    print("  Optional SFT: modal run nanochat_modal_yarn.py::stage_sft_yarn")
    print("  Optional RL:  modal run nanochat_modal_yarn.py::stage_rl_yarn")
    print("=" * width + "\n")
