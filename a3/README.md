# a3

This directory contains all code and write-up for Parts 2, 3, and 4 of the assignment.

## Setup

> [!NOTE]
> Make sure to do a recursive clone of this repo to get the nanochat submodule.

Requires Python 3.10+ and `uv`:

```
uv pip install modal
modal setup
```

Pass API keys as a Modal secret:
- W&B key:  https://wandb.ai/authorize
- HF token: https://huggingface.co/settings/tokens

```sh
modal secret create nanochat-secrets \
           WANDB_API_KEY=your_wandb_key \
           HF_TOKEN=hf_your_huggingface_token
```

> [!NOTE]
> All nanochat modifications live in `part2_mtp/patches/` and are applied by overwriting files
> inside the Modal container at image-build time. The `nanochat/` submodule is never modified.

## Part 2 — Ablations on picochat (d16)

Five d16 models: Baseline, MTP-2, MTP-4, Yarn, MTP-2+YaRN. Run all commands from `a3/`.

### Hyperparameter sweep
4 configs × 9 runs (3 LR × 3 batch size) = 36 runs, 300 steps each on H100:2.
Results log to `part2_sweep` W&B project.

```sh
modal run part2_mtp/nanochat_modal.py::stage_sweep_baseline  2>&1 | tee /tmp/a2mtp_sweep_baseline.log &
modal run part2_mtp/nanochat_modal.py::stage_sweep_mtp2      2>&1 | tee /tmp/a2mtp_sweep_mtp2.log &
modal run part2_mtp/nanochat_modal.py::stage_sweep_mtp4      2>&1 | tee /tmp/a2mtp_sweep_mtp4.log &
modal run part2_mtp/nanochat_modal.py::stage_sweep_mtp2_yarn 2>&1 | tee /tmp/a2mtp_sweep_mtp2_yarn.log &
wait
```

### Smoke test
```sh
modal run part2_mtp/nanochat_modal.py::quick_test_d12 2>&1 | tee /tmp/a2mtp_quicktest.log
```

### Main d16 training

Training and evaluation for YaRN only model can be run separately with: 
```
modal run part2/nanochat_modal_yarn.py 2>&1 | tee /tmp/yarn_log
```
For all other models:

```sh
modal run part2_mtp/nanochat_modal.py::stage_train_baseline  2>&1 | tee /tmp/a2mtp_baseline.log &
modal run part2_mtp/nanochat_modal.py::stage_train_mtp2      2>&1 | tee /tmp/a2mtp_mtp2.log &
modal run part2_mtp/nanochat_modal.py::stage_train_mtp4      2>&1 | tee /tmp/a2mtp_mtp4.log &
modal run part2_mtp/nanochat_modal.py::stage_train_mtp2_yarn 2>&1 | tee /tmp/a2mtp_mtp2_yarn.log &
wait
```

### Eval + report
```sh
modal run part2_mtp/nanochat_modal.py::stage_eval_and_report 2>&1 | tee /tmp/a2mtp_eval_report.log
```

### Sweep figures
```sh
modal run part2_mtp/nanochat_modal.py::stage_make_sweep_figures 2>&1 | tee /tmp/a2mtp_sweep_figures.log
```

### Local eval figures
```sh
uv run python3 a3/make_p2_figures.py
```

---

## Part 3 — Context-length curriculum (d16)

Three d16 models: Phase 1 (ctx=512), Phase 2 (ctx=2048 warm-start), Baseline (ctx=2048 from scratch).
Run all commands from `a3/`.

> [!NOTE]
> The d20 runs (tags `part3/d20_ctx512`, `part3/d20_ctx2048`, `part3/d20_baseline`) are also
> triggered from this directory and are reused as Part 4 nanochat baselines.

### Hyperparameter sweep
6 curriculum configurations: 2 Phase 1 seq lengths × 3 budget fractions. Results log to `part3_sweep`.

```sh
modal run part3/nanochat_modal.py::stage_sweep_p3_s256 2>&1 | tee /tmp/p3_sweep_s256.log &
modal run part3/nanochat_modal.py::stage_sweep_p3_s512 2>&1 | tee /tmp/p3_sweep_s512.log &
wait
```

```sh
modal run part3/nanochat_modal.py::stage_make_sweep_figures_p3 2>&1 | tee /tmp/p3_sweep_figures.log
```

### Smoke test
```sh
modal run part3/nanochat_modal.py::quick_test_d12 2>&1 | tee /tmp/p3_quicktest.log
```

### Main d16 training
Phase 1 must finish before Phase 2. Phase 2 and Baseline can run in parallel.

```sh
modal run part3/nanochat_modal.py::stage_pretrain_phase1 2>&1 | tee /tmp/p3_d16_phase1.log

modal run part3/nanochat_modal.py::stage_pretrain_phase2   2>&1 | tee /tmp/p3_d16_phase2.log &
modal run part3/nanochat_modal.py::stage_pretrain_baseline 2>&1 | tee /tmp/p3_d16_baseline.log &
wait
```

### Eval + report
```sh
modal run part3/nanochat_modal.py::stage_eval_and_report 2>&1 | tee /tmp/p3_d16_eval.log
```

### Eval figures
```sh
modal run part3/nanochat_modal.py::stage_make_eval_figures_p3 2>&1 | tee /tmp/p3_eval_figures.log
```

---

## Part 4 — Final Nanochat (d20 + MTP-2 + curriculum)

| Tag | Model | Steps | GPU | Purpose |
|---|---|---|---|---|
| `part4/d8_scaling` | d8 baseline | ~840 | H100:2 | scaling-law anchor |
| `part4/d12_scaling` | d12 baseline | ~2204 | H100:4 | scaling-law anchor |
| `part4/d20_mtp2_ctx512` | d20 MTP-2 Phase 1 | ~3669 | H100:8 | final nanochat |
| `part4/d20_mtp2_ctx2048` | d20 MTP-2 Phase 2 | ~5505 | H100:8 | final nanochat |

References reused (no new training):

| Tag | Source |
|---|---|
| `a2mtp/d16_baseline` | Part 2 baseline |
| `part3/d16_ctx2048` | Part 3 d16 curriculum Phase 2 |
| `part3/d20_ctx2048` | Part 3 d20 curriculum-only |

All commands from `a3/`.

### Smoke test
```sh
modal run part4/nanochat_modal.py::quick_test 2>&1 | tee /tmp/p4_smoke.log
```

### Scaling-law anchor runs
```sh
modal run part4/nanochat_modal.py::stage_scaling_d8  2>&1 | tee /tmp/p4_d8.log &
modal run part4/nanochat_modal.py::stage_scaling_d12 2>&1 | tee /tmp/p4_d12.log &
wait
```

### Final nanochat d20 + MTP-2 + curriculum
```sh
modal run part4/nanochat_modal.py::stage_d20_mtp2_p1 2>&1 | tee /tmp/p4_d20p1.log

modal run part4/nanochat_modal.py::stage_d20_mtp2_p2 2>&1 | tee /tmp/p4_d20p2.log
```

### Evaluation
```sh
modal run part4/nanochat_modal.py::stage_eval 2>&1 | tee /tmp/p4_eval.log
```

### Scaling law figures
```sh
uv run python3 a3/make_p4_scaling_figures.py
uv run python3 a3/make_p4_results_figures.py
```

---

## Credits

Built on top of [nanochat](https://github.com/karpathy/nanochat/) and the
[CSC490 tutorial](https://github.com/UofT-CSC490-W2026/022326-tutorial-nanochat/tree/main).
