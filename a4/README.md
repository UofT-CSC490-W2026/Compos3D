# a4

This directory contains all code and write-up for Part 2 of the assignment: SFT and midtraining on the d20 baseline.

## Setup

> [!NOTE]
> Uses the same nanochat submodule and Modal environment as `a3/`. Run all commands from `a3/`.

Requires Python 3.10+ and `uv`:

```
uv pip install modal
modal setup
```

Pass API keys as a Modal secret (same secret used in a3):

```sh
modal secret create nanochat-secrets \
           WANDB_API_KEY=your_wandb_key \
           HF_TOKEN=hf_your_huggingface_token
```

> [!NOTE]
> All nanochat modifications live in `a4/part2/patches/` and `a4/part2/tasks/` and are applied by
> overwriting files inside the Modal container at image-build time. The `a3/nanochat/` submodule
> is never modified.

---

## Part 2 — SFT & Midtraining on d20 Baseline

Base checkpoint: `part3/d20_baseline` (CORE = 0.2460, best d20 variant per Part 4 evaluation).
All commands from `a3/`.

### Smoke test
```sh
modal run a4/part2/nanochat_modal.py::quick_test 2>&1 | tee /tmp/a4p2_smoke.log
```

### Setup — download identity conversations
```sh
modal run a4/part2/nanochat_modal.py::stage_setup 2>&1 | tee /tmp/a4p2_setup.log
```

### Midtraining
One full epoch (~848K rows, ~650 steps) using the dedicated `mid_train.py` script (nanochat commit
`348fbb3`), matching Karpathy's speedrun exactly. Loads `part3/d20_baseline` from `base_checkpoints/`,
saves to `mid_checkpoints/a4/d20_midtrain/`. Both SFT runs then load from this midtrained checkpoint.
```sh
modal run a4/part2/nanochat_modal.py::stage_midtrain 2>&1 | tee /tmp/a4p2_midtrain.log
```

### SFT — original data mix
```sh
modal run a4/part2/nanochat_modal.py::stage_sft_original 2>&1 | tee /tmp/a4p2_sft_orig.log
```

### SFT — augmented data mix (+MetaMathQA + NuminaMathCoT)
Adds MetaMathQA (395K, Yu et al. 2023) and NuminaMathCoT (100K, AI-MO 2024) to the base
mixture. Both are math-reasoning datasets proven to substantially improve GSM8K scores.
```sh
modal run a4/part2/nanochat_modal.py::stage_sft_augmented 2>&1 | tee /tmp/a4p2_sft_aug.log
```

> [!NOTE]
> SFT original and SFT augmented are independent and can run in parallel if GPU quota allows.

### Evaluation
```sh
modal run a4/part2/nanochat_modal.py::stage_eval 2>&1 | tee /tmp/a4p2_eval.log
```

---

## Hyperparameter Sweep

3 × 3 grid (matrix LR × data mix, 200 steps each) to justify the SFT hyperparameters.
Runs all 9 jobs sequentially on a single 8×H100 node (~35–45 min total, ~$14).

```sh
modal run a4/part2/hparam_sweep.py::run_sweep 2>&1 | tee /tmp/a4p2_sweep.log
```

After the sweep finishes, regenerate the figures (replaces the placeholder sweep plots):

```sh
WANDB_API_KEY=<your_key> python a4/latex/gen_figures.py
```

---

## Credits

Built on top of [nanochat](https://github.com/karpathy/nanochat/) and the
[CSC490 tutorial](https://github.com/UofT-CSC490-W2026/022326-tutorial-nanochat/tree/main).
