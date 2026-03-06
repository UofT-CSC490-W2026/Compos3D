## Setup 

> [!NOTE]
> Make sure to do a recursive clone of this repo to get the nanochat submodule.

Requires Python 3.10+ and `uv` package manager
```
requires-python = ">=3.10"
```

```
uv pip install modal    # install Modal
modal setup             # authenticate with Modal
```

pass API keys as a Modal secret
- W&B key:  https://wandb.ai/authorize
- HF token: https://huggingface.co/settings/tokens
- HF token is needed to download the FineWeb-EDU dataset
```
modal secret create nanochat-secrets \
           WANDB_API_KEY=your_wandb_key \
           HF_TOKEN=hf_your_huggingface_token
```

> [!NOTE]
> To avoid making changes to nanochat, we make all our changes to nanochat through the use of the `patches` directory which overwrites certain nanochat files during execution. This includes a patched `gpt.py` that adds `activation` and `rope_type` fields to `GPTConfig` and implements the SwiGLU MLP and YaRN NTK-by-Parts rotary embeddings.

## Run experiments

Run the following commands one after another from your environment from inside the `a3` directory.

1. Run the smoke test to validate the full pipeline end-to-end at d12 scale before spending money on d16.

```sh
modal run part2_arch/nanochat_modal.py::quick_test_d12 2>&1 | tee /tmp/a2arch_quicktest.log
```

2. Run all three d16 training experiments in parallel.

```sh
modal run part2_arch/nanochat_modal.py::stage_train_baseline 2>&1 | tee /tmp/a2arch_baseline.log &
modal run part2_arch/nanochat_modal.py::stage_train_swiglu 2>&1 | tee /tmp/a2arch_swiglu.log &
modal run part2_arch/nanochat_modal.py::stage_train_yarn 2>&1 | tee /tmp/a2arch_yarn.log &
wait
```

3. Compute metrics and generate a report.

```sh
modal run part2_arch/nanochat_modal.py::stage_eval_and_report 2>&1 | tee /tmp/a2arch_eval_report.log
```

## Credits

Our code is built on top of [nanochat](https://github.com/karpathy/nanochat/) as well as the [CSC490 tutorial](https://github.com/UofT-CSC490-W2026/022326-tutorial-nanochat/tree/main).
