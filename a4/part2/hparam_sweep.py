import os
import subprocess

import modal
from modal import App, Image as ModalImage, Secret, Volume


BASE_MODEL_TAG = "a4/d20_midtrain"
WANDB_PROJECT = "nanochat-a4-part2"
SWEEP_STEPS = 200
TOTAL_BATCH_SIZE = 524_288
DEVICE_BATCH_SFT = 16
_N_GPUS = 8


LR_VALUES = [0.025, 0.05, 0.10]


MIX_CONFIGS = [
    {"name": "orig", "use_augmented": False, "numina": False},
    {"name": "plus_metamath", "use_augmented": True, "numina": False},
    {"name": "full_aug", "use_augmented": True, "numina": True},
]


TIMEOUT_SWEEP = 60 * 60 * 1

GPU_TRAIN = f"H100:{_N_GPUS}"


app = App("nanochat-a4-part2-sweep")
volume = Volume.from_name("nanochat-vol", create_if_missing=True)
secret = Secret.from_name("nanochat-secrets")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_NANOCHAT_DIR = os.path.join(_THIS_DIR, "..", "..", "a3", "nanochat")
_MTP_PATCHES = os.path.join(_THIS_DIR, "..", "..", "a3", "part2_mtp", "patches")
_OWN_PATCHES = os.path.join(_THIS_DIR, "patches")
_OWN_TASKS = os.path.join(_THIS_DIR, "tasks")

VOLUME_MOUNT = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"
BASE_DIR = "/data/.cache/nanochat"

image = (
    ModalImage.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git", "build-essential", "curl", "wget", "unzip")
    .add_local_dir(local_path=_NANOCHAT_DIR, remote_path="/root/nanochat", copy=True)
    .add_local_file(
        local_path=os.path.join(_MTP_PATCHES, "gpt.py"),
        remote_path="/root/nanochat/nanochat/gpt.py",
        copy=True,
    )
    .add_local_file(
        local_path=os.path.join(_OWN_PATCHES, "checkpoint_manager.py"),
        remote_path="/root/nanochat/nanochat/checkpoint_manager.py",
        copy=True,
    )
    .add_local_file(
        local_path=os.path.join(_OWN_PATCHES, "chat_sft.py"),
        remote_path="/root/nanochat/scripts/chat_sft.py",
        copy=True,
    )
    .add_local_file(
        local_path=os.path.join(_OWN_TASKS, "metamath.py"),
        remote_path="/root/nanochat/tasks/metamath.py",
        copy=True,
    )
    .add_local_file(
        local_path=os.path.join(_OWN_TASKS, "numina_math.py"),
        remote_path="/root/nanochat/tasks/numina_math.py",
        copy=True,
    )
    .workdir("/root/nanochat")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
    )
    .pip_install("uv")
    .env(
        {
            "OMP_NUM_THREADS": "1",
            "NANOCHAT_BASE_DIR": BASE_DIR,
            "HF_HOME": "/data/.cache/huggingface",
        }
    )
    .run_commands("cd /root/nanochat && uv sync --extra gpu --no-install-project")
)


def _run(cmd: str) -> None:
    print(f"\n>>>  {cmd}\n")
    result = subprocess.run(["bash", "-c", cmd], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command exited {result.returncode}:\n  {cmd}")


def _torchrun(module: str, args: list | None = None, *, nproc: int = _N_GPUS) -> None:
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
    gpu=GPU_TRAIN,
    volumes={VOLUME_MOUNT: volume},
    secrets=[secret],
    timeout=TIMEOUT_SWEEP,
)
def sweep_job(lr: float, mix_name: str, use_augmented: bool) -> dict:
    _setup_cache()

    run_name = f"sweep_lr{lr:.3f}_mix_{mix_name}"
    model_tag = f"a4/sweep/{run_name}"

    args = [
        f"--run={run_name}",
        f"--wandb-project={WANDB_PROJECT}",
        f"--model-tag={model_tag}",
        f"--model-source=mid",
        f"--base-model-tag={BASE_MODEL_TAG}",
        f"--load-optimizer=0",
        f"--num-iterations={SWEEP_STEPS}",
        f"--device-batch-size={DEVICE_BATCH_SFT}",
        f"--total-batch-size={TOTAL_BATCH_SIZE}",
        f"--matrix-lr={lr}",
        f"--mmlu-epochs=3",
        f"--gsm8k-epochs=4",
        f"--chatcore-every=-1",
        f"--eval-every={SWEEP_STEPS}",
    ]

    if use_augmented:
        args.append("--use-augmented-data")

    _torchrun("scripts.chat_sft", args)
    volume.commit()
    return {"run_name": run_name, "lr": lr, "mix": mix_name}


_ALL_CONFIGS = [
    {"lr": 0.025, "mix": MIX_CONFIGS[0]},
    {"lr": 0.050, "mix": MIX_CONFIGS[0]},
    {"lr": 0.100, "mix": MIX_CONFIGS[0]},
    {"lr": 0.025, "mix": MIX_CONFIGS[2]},
    {"lr": 0.050, "mix": MIX_CONFIGS[2]},
    {"lr": 0.100, "mix": MIX_CONFIGS[2]},
    {"lr": 0.050, "mix": MIX_CONFIGS[0]},
    {"lr": 0.050, "mix": MIX_CONFIGS[1]},
    {"lr": 0.050, "mix": MIX_CONFIGS[2]},
]


_SEEN: set = set()
_UNIQUE_CONFIGS = []
for _c in _ALL_CONFIGS:
    _name = f"sweep_lr{_c['lr']:.3f}_mix_{_c['mix']['name']}"
    if _name not in _SEEN:
        _SEEN.add(_name)
        _UNIQUE_CONFIGS.append(_c)


@app.local_entrypoint()
def run_sweep() -> None:
    print(f"Starting sequential sweep: {len(_UNIQUE_CONFIGS)} jobs on {GPU_TRAIN}")
    print(f"Monitor W&B at: https://wandb.ai/rishit_dagli/{WANDB_PROJECT}\n")

    for i, cfg in enumerate(_UNIQUE_CONFIGS, 1):
        lr = cfg["lr"]
        mix = cfg["mix"]
        run_name = f"sweep_lr{lr:.3f}_mix_{mix['name']}"
        print(f"[{i}/{len(_UNIQUE_CONFIGS)}] Running {run_name} ...")
        result = sweep_job.remote(
            lr=lr, mix_name=mix["name"], use_augmented=mix["use_augmented"]
        )
        print(f"  ✓  {result['run_name']} done\n")

    print("Sweep complete.")
    print(
        "Run  WANDB_API_KEY=<key> python a4/latex/gen_figures.py  to regenerate plots."
    )
