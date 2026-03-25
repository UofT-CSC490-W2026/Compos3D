import os
import subprocess

import modal
from modal import App
from modal import Image as ModalImage
from modal import Secret, Volume


SFT_MODEL_TAG = "a4/d20_sft_orig"


TAG_RL = "a4/d20_sft_orig"


WANDB_PROJECT = "nanochat-a4-part3"
WANDB_RUN = "a4_d20_rl"


GPU_TRAIN = "H100:8"
GPU_EVAL = "H100:8"
_N_TRAIN_GPUS = 8
_N_EVAL_GPUS = 8


TIMEOUT_RL = 60 * 60 * 3
TIMEOUT_EVAL = 60 * 60 * 2


VOLUME_MOUNT = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"
BASE_DIR = "/data/.cache/nanochat"


RESULTS_PATH = f"{NANOCHAT_CACHE}/rl_results.jsonl"


app = modal.App("nanochat-a4-part3")
volume = Volume.from_name("nanochat-vol", create_if_missing=True)
secret = Secret.from_name("nanochat-secrets")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_NANOCHAT_DIR = os.path.join(_THIS_DIR, "..", "nanochat")

_MTP_PATCHES = os.path.join(_THIS_DIR, "..", "..", "a3", "part2_mtp", "patches")

_PART2_PATCHES = os.path.join(_THIS_DIR, "..", "part2", "patches")

_OWN_PATCHES = os.path.join(_THIS_DIR, "patches")

image = (
    ModalImage.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git", "build-essential", "curl", "wget", "unzip")
    .add_local_dir(
        local_path=_NANOCHAT_DIR,
        remote_path="/root/nanochat",
        copy=True,
    )
    .add_local_file(
        local_path=os.path.join(_MTP_PATCHES, "gpt.py"),
        remote_path="/root/nanochat/nanochat/gpt.py",
        copy=True,
    )
    .add_local_file(
        local_path=os.path.join(_PART2_PATCHES, "checkpoint_manager.py"),
        remote_path="/root/nanochat/nanochat/checkpoint_manager.py",
        copy=True,
    )
    .add_local_file(
        local_path=os.path.join(_OWN_PATCHES, "chat_rl.py"),
        remote_path="/root/nanochat/scripts/chat_rl.py",
        copy=True,
    )
    .add_local_file(
        local_path=os.path.join(_OWN_PATCHES, "eval_gsm8k.py"),
        remote_path="/root/nanochat/scripts/eval_gsm8k.py",
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
    print(f"\n>>>  {cmd}\n")
    result = subprocess.run(["bash", "-c", cmd], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command exited {result.returncode}:\n  {cmd}")


def _torchrun(module: str, args: list | None = None, *, nproc: int) -> None:
    args = args or []
    args_str = (" -- " + " ".join(args)) if args else ""
    _run(
        f"cd /root/nanochat && "
        f"PYTHONPATH=/root/nanochat:$PYTHONPATH "
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m {module}{args_str}"
    )


def _setup_cache() -> None:
    os.makedirs(NANOCHAT_CACHE, exist_ok=True)
    if not os.path.exists(BASE_DIR):
        os.makedirs(os.path.dirname(BASE_DIR), exist_ok=True)
        os.symlink(NANOCHAT_CACHE, BASE_DIR)


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_RL,
)
def stage_rl() -> None:
    _setup_cache()
    volume.reload()
    print(
        f"RL training: loading {SFT_MODEL_TAG} from chatsft_checkpoints/, "
        f"running 1 epoch over GSM8K → chatrl_checkpoints/{TAG_RL}"
    )
    _torchrun(
        "scripts.chat_rl",
        [
            f"--run={WANDB_RUN}",
            f"--source=sft",
            f"--model-tag={SFT_MODEL_TAG}",
            f"--results-path={RESULTS_PATH}",
        ],
        nproc=_N_TRAIN_GPUS,
    )
    volume.commit()
    print(f"RL training done.  Tag: {TAG_RL}  → chatrl_checkpoints/{TAG_RL}/")
    print(f"Per-problem results: {RESULTS_PATH}")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_EVAL,
    timeout=TIMEOUT_EVAL,
)
def stage_eval() -> None:
    _setup_cache()
    volume.reload()
    print(f"Evaluating RL checkpoint: {TAG_RL}")
    _torchrun(
        "scripts.chat_eval",
        [
            "-i",
            "rl",
            "-a",
            "GSM8K",
            "-g",
            TAG_RL,
            "-x",
            "400",
        ],
        nproc=_N_EVAL_GPUS,
    )
    volume.commit()
    print(f"Eval done.  Check W&B project {WANDB_PROJECT} for final GSM8K score.")


RL_BEST_STEP = 60


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_EVAL,
    timeout=TIMEOUT_EVAL,
)
def stage_eval_sft() -> None:
    _setup_cache()
    volume.reload()
    print(f"Evaluating SFT checkpoint: {SFT_MODEL_TAG} (Karpathy protocol)")
    _torchrun(
        "scripts.eval_gsm8k",
        [
            f"--run=a4_d20_sft_karpathy_eval",
            f"--source=sft",
            f"--model-tag={SFT_MODEL_TAG}",
        ],
        nproc=_N_EVAL_GPUS,
    )
    volume.commit()
    print("SFT eval done. Check W&B project nanochat-a4-part3.")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_EVAL,
    timeout=TIMEOUT_EVAL,
)
def stage_eval_best_rl() -> None:
    _setup_cache()
    volume.reload()
    print(
        f"Evaluating best RL checkpoint: {TAG_RL} step {RL_BEST_STEP} (Karpathy protocol)"
    )
    _torchrun(
        "scripts.eval_gsm8k",
        [
            f"--run=a4_d20_rl_best_step{RL_BEST_STEP}",
            f"--source=rl",
            f"--model-tag={TAG_RL}",
            f"--model-step={RL_BEST_STEP}",
        ],
        nproc=_N_EVAL_GPUS,
    )
    volume.commit()
    print(f"Best RL eval done. Check W&B project nanochat-a4-part3.")


@app.local_entrypoint()
def main() -> None:
    w = 64
    print("\n" + "=" * w)
    print("a4/part3: Reinforcement Learning on GSM8K")
    print(f"  SFT checkpoint : {SFT_MODEL_TAG}")
    print(f"  RL output tag  : {TAG_RL}")
    print(f"  Results path   : {RESULTS_PATH}")
    print(f"  WandB project  : {WANDB_PROJECT}")
    print("=" * w + "\n")

    print("[1] RL training (~2 h, H100:8)...")
    stage_rl.remote()

    print("[2] GSM8K evaluation (H100:4)...")
    stage_eval.remote()

    print("\n" + "=" * w)
    print("a4/part3 complete!")
    print(f"  RL checkpoint  : chatrl_checkpoints/{TAG_RL}/")
    print(f"  EDA data       : {RESULTS_PATH}")
    print(f"  WandB project  : {WANDB_PROJECT}")
    print("=" * w + "\n")
