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
> To avoid making changes to nanochat, we make all our changes to nanochat through the use of the `patches` directory which overwrites certain nanochat files during execution.

## Run experiments

Run the following commands one after another from your environment from inside the `a3` directory.

I. Part 2
1. Run the training on picochat depth=16 and using YaRN embedding:
```
modal run part2/nanochat_modal_yarn.py
```

II. Part 3
1. Run the training on 512 and 20248 context sizes.

```sh
modal run part3/nanochat_modal.py::stage_pretrain_phase1 2>&1 | tee /tmp/d20_phase1.log && \
modal run part3/nanochat_modal.py::stage_pretrain_phase2 2>&1 | tee /tmp/d20_phase2.log
```

2. Run the baseline model.

```sh
modal run part3/nanochat_modal.py::stage_pretrain_baseline 2>&1 | tee /tmp/d20_baseline.log
```

3. Compute metrics and generate a report.

```sh
modal run part3/nanochat_modal.py::stage_eval_and_report 2>&1 | tee /tmp/d20_eval_report.log
```

## Credits

Our code is built on top of [nanochat](https://github.com/karpathy/nanochat/) as well as the [CSC490 tutorial](https://github.com/UofT-CSC490-W2026/022326-tutorial-nanochat/tree/main).