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
> To avoid making changes to nanochat, we make all our changes to nanochat through the use of the `patches` directory which overwrites certain nanochat files during execution. This includes a patched `gpt.py` that adds the `mtp_k`, `rope_type`, and `yarn_scale` fields to `GPTConfig`, implements the weight-tied MTP loss in `forward()`, and adds the YaRN NTK-by-Parts rotary embedding variant.

## Run experiments

Run the following commands from inside the `a3` directory.

1. **Hyperparameter sweep** — 4 configs × 9 runs (3 LR × 3 batch-size) = 36 runs total,
   300 steps each on 2×H100. Run all four groups in parallel. Results log to the
   `part2_sweep` W&B project. For each config, pick the `(lr, batch)` with the lowest
   step-300 training loss. Run names: `sweep_<config>_lr<LR>_bs<batch>k`
   (e.g. `sweep_mtp2_lr0.02_bs524k`).

```sh
modal run part2_mtp/nanochat_modal.py::stage_sweep_baseline  2>&1 | tee /tmp/a2mtp_sweep_baseline.log &
modal run part2_mtp/nanochat_modal.py::stage_sweep_mtp2      2>&1 | tee /tmp/a2mtp_sweep_mtp2.log &
modal run part2_mtp/nanochat_modal.py::stage_sweep_mtp4      2>&1 | tee /tmp/a2mtp_sweep_mtp4.log &
modal run part2_mtp/nanochat_modal.py::stage_sweep_mtp2_yarn 2>&1 | tee /tmp/a2mtp_sweep_mtp2_yarn.log &
wait
```

2. **Smoke test** — validates the full pipeline end-to-end at d12 scale before spending money on d16.

```sh
modal run part2_mtp/nanochat_modal.py::quick_test_d12 2>&1 | tee /tmp/a2mtp_quicktest.log
```

3. **Main d16 training** — run all four ablations in parallel (each on 8×H100, ~60–70 min each).

```sh
modal run part2_mtp/nanochat_modal.py::stage_train_baseline 2>&1 | tee /tmp/a2mtp_baseline.log &
modal run part2_mtp/nanochat_modal.py::stage_train_mtp2 2>&1 | tee /tmp/a2mtp_mtp2.log &
modal run part2_mtp/nanochat_modal.py::stage_train_mtp4 2>&1 | tee /tmp/a2mtp_mtp4.log &
modal run part2_mtp/nanochat_modal.py::stage_train_mtp2_yarn 2>&1 | tee /tmp/a2mtp_mtp2_yarn.log &
wait
```

4. **Eval + report** — CORE benchmark on all four checkpoints, writes results JSON and markdown report.

```sh
modal run part2_mtp/nanochat_modal.py::stage_eval_and_report 2>&1 | tee /tmp/a2mtp_eval_report.log
```

## Credits

Our code is built on top of [nanochat](https://github.com/karpathy/nanochat/) as well as the [CSC490 tutorial](https://github.com/UofT-CSC490-W2026/022326-tutorial-nanochat/tree/main).
