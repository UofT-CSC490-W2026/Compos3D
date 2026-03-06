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

Pass API keys as a Modal secret:
- W&B key:  https://wandb.ai/authorize
- HF token: https://huggingface.co/settings/tokens
- HF token is needed to download the FineWeb-EDU dataset

```sh
modal secret create nanochat-secrets \
           WANDB_API_KEY=your_wandb_key \
           HF_TOKEN=hf_your_huggingface_token
```

> [!NOTE]
> Part 4 reuses the patches from `a3/part2_mtp/patches/` — those files add
> `--mtp-k`, `--rope-type`, and `--yarn-scale` to `base_train.py` and implement
> weight-tied MTP + YaRN in `gpt.py`.  The nanochat submodule is never modified;
> patches overwrite files inside the container at image-build time only.

## Experiment overview

| Tag | Model | Variant | Steps | GPU | Purpose |
|---|---|---|---|---|---|
| `part4/d8_scaling` | d8 | baseline | ~840 | H100:2 | scaling-law anchor |
| `part4/d12_scaling` | d12 | baseline | ~2204 | H100:4 | scaling-law anchor |
| `part4/d20_mtp2_ctx512` | **d20** | **MTP-2** + curriculum Phase 1 | ~3669 | H100:8 | **final nanochat** |
| `part4/d20_mtp2_ctx2048` | **d20** | **MTP-2** + curriculum Phase 2 | ~5505 | H100:8 | **final nanochat** |

References reused from earlier parts (no new training):

| Tag | Source | Description |
|---|---|---|
| `a2mtp/d16_baseline` | Part 2 | d16 picochat baseline (no MTP, no curriculum) |
| `a2mtp/d16_mtp2` | Part 2 | d16 + MTP-2 (no curriculum) |
| `part3/d16_ctx2048` | **Part 3** | d16 + plain curriculum Phase 2 — picochat ablation (trained & eval'd in Part 3) |
| `part3/d20_ctx2048` | Part 3 (old d20) | d20 curriculum-only, no MTP |
| `part3/d20_baseline` | Part 3 (old d20) | d20 full-context baseline |

All training runs log to the **`nanochat-part4`** W&B project.

Scaling-law step counts use the formula:
```
scaling_params(d) = d × 12 × (64d)² + 32768 × (64d)
chinchilla_tokens = 10.5 × scaling_params
steps             = chinchilla_tokens // 524288
```

## Run experiments

Run the following commands from inside the `a3/` directory.

### 1. Smoke test

Validates patches, MTP, and the Phase-1→Phase-2 warm-start end-to-end at d8
scale in ~25 minutes before spending money on larger runs.

```sh
modal run part4/nanochat_modal.py::quick_test 2>&1 | tee /tmp/p4_smoke.log
```

### 2. Scaling-law anchor runs

Run d8 and d12 in parallel. Each trains to its full Chinchilla-optimal step
count.  Results log to W&B as `p4_d8_scaling` and `p4_d12_scaling`.

```sh
modal run part4/nanochat_modal.py::stage_scaling_d8  2>&1 | tee /tmp/p4_d8.log &
modal run part4/nanochat_modal.py::stage_scaling_d12 2>&1 | tee /tmp/p4_d12.log &
wait
```

> [!NOTE]
> The d16 picochat plain curriculum is run through **Part 3**, not Part 4.
> Run `part3/nanochat_modal.py::stage_pretrain_phase1` and
> `part3/nanochat_modal.py::stage_pretrain_phase2`, then
> `part3/nanochat_modal.py::stage_eval_and_report` and
> `part3/nanochat_modal.py::stage_make_eval_figures_p3`.
> Part 4 reads the resulting checkpoint from tag `part3/d16_ctx2048` directly.

### 3. Final nanochat d20 + MTP-2 + curriculum

Phase 2 must follow Phase 1. Phase 1 can run in parallel with d16 Phase 2.

```sh
# Phase 1 (ctx=512, ~3669 steps, H100:8, ~3 h)
modal run part4/nanochat_modal.py::stage_d20_mtp2_p1 2>&1 | tee /tmp/p4_d20p1.log

# Phase 2 (ctx=2048, ~5505 steps, H100:8, ~5 h) — run after Phase 1 completes
modal run part4/nanochat_modal.py::stage_d20_mtp2_p2 2>&1 | tee /tmp/p4_d20p2.log
```

### 5. Evaluation

CORE + BPB evaluation on the new Part 4 checkpoints (d8, d12, d20 Phase 2).
The d16 curriculum Phase 2 is evaluated by Part 3's `stage_eval_and_report`.
Requires steps 2–3 to be complete.

```sh
modal run part4/nanochat_modal.py::stage_eval 2>&1 | tee /tmp/p4_eval.log
```

### Full pipeline (automatic sequencing)

The `main` entrypoint runs all phases in the correct order, parallelising where
possible:

```sh
modal run part4/nanochat_modal.py 2>&1 | tee /tmp/p4_full.log
```

Execution order:
```
Phase A (parallel): stage_scaling_d8 + stage_scaling_d12 + stage_d20_mtp2_p1
Phase B (serial  ): stage_d20_mtp2_p2   ← final nanochat
Phase C (serial  ): stage_eval
```
Note: d16 curriculum is handled separately via Part 3.

## Recommended parallel run commands

For maximum parallelism without the `main` entrypoint:

```sh
# Terminal 1 — scaling laws + d20 Phase 1 in parallel
#   (run Part 3 d16 curriculum stages separately if not already done)
modal run part4/nanochat_modal.py::stage_scaling_d8  2>&1 | tee /tmp/p4_d8.log &
modal run part4/nanochat_modal.py::stage_scaling_d12 2>&1 | tee /tmp/p4_d12.log &
modal run part4/nanochat_modal.py::stage_d20_mtp2_p1 2>&1 | tee /tmp/p4_d20p1.log &
wait

# Terminal 2 — d20 Phase 2, then eval (after Terminal 1 done)
modal run part4/nanochat_modal.py::stage_d20_mtp2_p2 2>&1 | tee /tmp/p4_d20p2.log && \
modal run part4/nanochat_modal.py::stage_eval         2>&1 | tee /tmp/p4_eval.log
```

## Credits

Our code is built on top of [nanochat](https://github.com/karpathy/nanochat/)
as well as the [CSC490 tutorial](https://github.com/UofT-CSC490-W2026/022326-tutorial-nanochat/tree/main).
