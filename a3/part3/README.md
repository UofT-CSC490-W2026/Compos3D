## Overview

Part 3 trains a **d16 picochat** with a context-length curriculum: first at a short
sequence length (Phase 1), then extended to 2048 (Phase 2), and compares against a
d16 baseline trained at 2048 from scratch.

> [!NOTE]
> The d20 runs are also done through this code.
> (tags `part3/d20_ctx512`, `part3/d20_ctx2048`, `part3/d20_baseline`).
> Those results are reused in **Part 4** as the nanochat baseline
> All commands below operate exclusively on d16 checkpoints.

---

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

Pass API keys as a Modal secret
- W&B key:  https://wandb.ai/authorize
- HF token: https://huggingface.co/settings/tokens
- HF token is needed to download the FineWeb-EDU dataset
```
modal secret create nanochat-secrets \
           WANDB_API_KEY=your_wandb_key \
           HF_TOKEN=hf_your_huggingface_token
```

> [!NOTE]
> To avoid making changes to nanochat, all modifications are applied through the
> `patches/` directory, which overwrites specific nanochat files at container
> build time. The nanochat submodule itself stays clean.

---

## Run experiments

All commands are run from inside the `a3` directory.

I. Part 2
1. Run the training on picochat depth=16 and using YaRN embedding:
```
modal run part2/nanochat_modal_yarn.py
```

II. Part 3
1. Run the training on 512 and 20248 context sizes.

```sh
modal run part3/nanochat_modal.py::stage_sweep_p3_s256 2>&1 | tee /tmp/p3_sweep_s256.log &
modal run part3/nanochat_modal.py::stage_sweep_p3_s512 2>&1 | tee /tmp/p3_sweep_s512.log &
wait
```

W&B run names follow the pattern `sweep_s{seq}_f{frac}_phase{1,2}`
(e.g. `sweep_s512_f40_phase1`). Pick the `(seq, frac)` combo with the lowest
Phase 2 BPB at step 300 to justify the Phase 1 fraction used in the main runs.

Once the sweeps have finished, generate the figures (CPU-only job, no GPU needed):

```sh
modal run part3/nanochat_modal.py::stage_make_sweep_figures_p3 2>&1 | tee /tmp/p3_sweep_figures.log
```

This saves two PNGs to the volume under `nanochat_cache/report/` and logs them to the `part3_sweep` W&B project:
- `p3_sweep_loss_curves.png` — 2×2 panel loss curves (rows = seq len, columns = phase, 3 coloured lines per frac).
- `p3_sweep_bar_chart.png` — grouped bar chart of final-step loss across all 4 groups (s256_p1/p2, s512_p1/p2).

### 2. Smoke test

Validates the full two-phase curriculum pipeline end-to-end at d12 scale
before spending money on d16.

```sh
modal run part3/nanochat_modal.py::quick_test_d12 2>&1 | tee /tmp/p3_quicktest.log
```

3. Compute metrics and generate a report.

```sh
# Phase 1 first (sequential)
modal run part3/nanochat_modal.py::stage_pretrain_phase1 2>&1 | tee /tmp/p3_d16_phase1.log

# Then Phase 2 and Baseline in parallel
modal run part3/nanochat_modal.py::stage_pretrain_phase2    2>&1 | tee /tmp/p3_d16_phase2.log &
modal run part3/nanochat_modal.py::stage_pretrain_baseline  2>&1 | tee /tmp/p3_d16_baseline.log &
wait
```

### 4. Eval + report

Runs CORE benchmark + needle-in-haystack custom eval on all three d16
checkpoints, then writes a markdown report.

```sh
modal run part3/nanochat_modal.py::stage_eval_and_report 2>&1 | tee /tmp/p3_d16_eval.log
```

---

## Credits

Our code is built on top of [nanochat](https://github.com/karpathy/nanochat/) as well as the [CSC490 tutorial](https://github.com/UofT-CSC490-W2026/022326-tutorial-nanochat/tree/main).
