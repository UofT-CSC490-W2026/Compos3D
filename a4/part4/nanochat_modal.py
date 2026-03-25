"""
Part 4 (a4): Reinforcement Learning on GSM8K ablation study.
"""

from __future__ import annotations

import os
import subprocess
import shlex
from typing import Final

import modal
from modal import Image as ModalImage
from modal import Secret, Volume

# =============================================================================
# CONFIG
# =============================================================================

SFT_MODEL_TAG: Final[str] = "a4/d20_sft_orig"

TAG_RL_BASE_FORMAT: Final[str] = "a4/d20_part4_base_format_rl"
TAG_RL_BASE_ARITH: Final[str] = "a4/d20_part4_base_arithmetic_rl"
TAG_RL_ALL: Final[str] = "a4/d20_part4_all_rewards_rl"

WANDB_PROJECT: Final[str] = "nanochat-a4-part4"

GPU_TRAIN: Final[str] = "H100:8"
GPU_EVAL: Final[str] = "H100:4"
N_TRAIN_GPUS: Final[int] = 8
N_EVAL_GPUS: Final[int] = 4

TIMEOUT_RL: Final[int] = 60 * 60 * 4
TIMEOUT_EVAL: Final[int] = 60 * 60 * 2
TIMEOUT_ORCH: Final[int] = 60 * 60 * 24

VOLUME_MOUNT: Final[str] = "/vol"
NANOCHAT_CACHE: Final[str] = f"{VOLUME_MOUNT}/nanochat_cache"
BASE_DIR: Final[str] = "/data/.cache/nanochat"

RL_BEST_STEP: Final[int | None] = None

RUNS: Final[list[dict[str, object]]] = [
    {
        "key": "base_format",
        "run_name": "a4_d20_part4_base_format_rl",
        "tag_rl": TAG_RL_BASE_FORMAT,
        "eval_run_name": "a4_d20_part4_base_format_eval",
        "base_reward_weight": 0.5,
        "format_reward_weight": 0.5,
        "arithmetic_reward_weight": 0.0,
    },
    {
        "key": "base_arithmetic",
        "run_name": "a4_d20_part4_base_arithmetic_rl",
        "tag_rl": TAG_RL_BASE_ARITH,
        "eval_run_name": "a4_d20_part4_base_arithmetic_eval",
        "base_reward_weight": 0.5,
        "format_reward_weight": 0.0,
        "arithmetic_reward_weight": 0.5,
    },
    {
        "key": "all_rewards",
        "run_name": "a4_d20_part4_all_rewards_rl",
        "tag_rl": TAG_RL_ALL,
        "eval_run_name": "a4_d20_part4_all_rewards_eval",
        "base_reward_weight": 1 / 3,
        "format_reward_weight": 1 / 3,
        "arithmetic_reward_weight": 1 / 3,
    },
]

CFG_MAP: Final[dict[str, dict[str, object]]] = {
    str(cfg["key"]): cfg for cfg in RUNS
}

# =============================================================================
# MODAL SETUP
# =============================================================================

app = modal.App("nanochat-a4-part4")
volume = Volume.from_name("nanochat-vol", create_if_missing=True)
secret = Secret.from_name("nanochat-secrets")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_NANOCHAT_DIR = os.path.join(_THIS_DIR, "..", "nanochat")
_MTP_PATCHES = os.path.join(_THIS_DIR, "..", "..", "a3", "part2_mtp", "patches")
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
            "WANDB_PROJECT": WANDB_PROJECT,
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
    print(f"\n>>> {cmd}\n")
    result = subprocess.run(["bash", "-c", cmd], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command exited {result.returncode}:\n  {cmd}")


def _torchrun(script_path: str, args: list[str] | None = None, *, nproc: int) -> None:
    args = args or []
    quoted_args = " ".join(shlex.quote(a) for a in args)

    cmd = (
        "cd /root/nanochat && "
        "PYTHONPATH=/root/nanochat:$PYTHONPATH "
        f"uv run torchrun --standalone --nproc-per-node={nproc} "
        f"{shlex.quote(script_path)} -- {quoted_args}"
    )
    _run(cmd)



def _setup_cache() -> None:
    os.makedirs(NANOCHAT_CACHE, exist_ok=True)
    if not os.path.exists(BASE_DIR):
        os.makedirs(os.path.dirname(BASE_DIR), exist_ok=True)
        os.symlink(NANOCHAT_CACHE, BASE_DIR)



def _rl_results_path(tag_rl: str) -> str:
    return os.path.join(NANOCHAT_CACHE, f"{tag_rl.replace('/', '_')}_results.jsonl")



def _run_cfg(cfg: dict[str, object]) -> None:
    stage_rl.remote(
        run_name=str(cfg["run_name"]),
        tag_rl=str(cfg["tag_rl"]),
        base_reward_weight=float(cfg["base_reward_weight"]),
        format_reward_weight=float(cfg["format_reward_weight"]),
        arithmetic_reward_weight=float(cfg["arithmetic_reward_weight"]),
    ).get()

    stage_eval.remote(
        run_name=str(cfg["eval_run_name"]),
        tag_rl=str(cfg["tag_rl"]),
        model_step=RL_BEST_STEP,
    ).get()


# =============================================================================
# STAGES (ALL REMOTE)
# =============================================================================


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_RL,
)
def stage_rl(
    run_name: str,
    tag_rl: str,
    base_reward_weight: float = 1 / 3,
    format_reward_weight: float = 1 / 3,
    arithmetic_reward_weight: float = 1 / 3,
) -> None:
    _setup_cache()
    volume.reload()

    results_path = _rl_results_path(tag_rl)

    print(f"Starting RL run: {run_name}")
    print(f"  source SFT tag : {SFT_MODEL_TAG}")
    print(f"  output RL tag  : {tag_rl}")
    print(f"  results path   : {results_path}")
    print(
        "  reward weights : "
        f"base={base_reward_weight}, "
        f"format={format_reward_weight}, "
        f"arithmetic={arithmetic_reward_weight}"
    )

    _torchrun(
        "scripts/chat_rl.py",
        [
            f"--run={run_name}",
            "--source=sft",
            f"--model-tag={SFT_MODEL_TAG}",
            f"--results-path={results_path}",
            f"--base_reward_weight={base_reward_weight}",
            f"--format_reward_weight={format_reward_weight}",
            f"--arithmetic_reward_weight={arithmetic_reward_weight}",
        ],
        nproc=N_TRAIN_GPUS,
    )

    volume.commit()
    print(f"RL done: chatrl_checkpoints/{tag_rl}/")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_EVAL,
    timeout=TIMEOUT_EVAL,
)
def stage_eval(
    run_name: str,
    tag_rl: str,
    model_step: int | None = None,
) -> None:
    _setup_cache()
    volume.reload()

    print(f"Starting eval run: {run_name}")
    print(f"  RL tag     : {tag_rl}")
    print(f"  model step : {model_step}")

    args = [
        f"--run={run_name}",
        "--source=rl",
        f"--model-tag={tag_rl}",
    ]
    if model_step is not None:
        args.append(f"--model-step={model_step}")

    _torchrun(
        "scripts/eval_gsm8k.py",
        args,
        nproc=N_EVAL_GPUS,
    )

    volume.commit()
    print(f"Eval done: {tag_rl}")


# =============================================================================
# CLOUD ORCHESTRATION (REMOTE)
# =============================================================================


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    timeout=TIMEOUT_ORCH,
)
def run_ablation_remote(which: str) -> None:
    _setup_cache()
    volume.reload()

    if which not in CFG_MAP:
        raise ValueError(f"Unknown ablation '{which}'. Expected one of {list(CFG_MAP.keys())}")

    cfg = CFG_MAP[which]
    print(f"Running ablation remotely: {which}")
    print(f"  RL run name : {cfg['run_name']}")
    print(f"  RL tag      : {cfg['tag_rl']}")
    print(
        "  weights     : "
        f"base={cfg['base_reward_weight']}, "
        f"format={cfg['format_reward_weight']}, "
        f"arithmetic={cfg['arithmetic_reward_weight']}"
    )

    _run_cfg(cfg)

    volume.commit()
    print(f"Finished ablation: {which}")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    timeout=TIMEOUT_ORCH,
)
def run_all_remote() -> None:
    _setup_cache()
    volume.reload()

    print("Running all Part 4 ablations remotely.")
    for cfg in RUNS:
        print("-" * 72)
        print(f"Run name : {cfg['run_name']}")
        print(f"RL tag   : {cfg['tag_rl']}")
        print(
            "Weights  : "
            f"base={cfg['base_reward_weight']}, "
            f"format={cfg['format_reward_weight']}, "
            f"arithmetic={cfg['arithmetic_reward_weight']}"
        )
        _run_cfg(cfg)

    volume.commit()
    print("Finished all Part 4 ablations.")
