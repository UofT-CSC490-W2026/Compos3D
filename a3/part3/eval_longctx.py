"""
Runs two evaluations on multiple model checkpoints:

  A. BPB by Context Position
     Take 2048-token validation sequences, split into 4 × 512-token segments.
     For each segment, compute BPB using only the prior tokens each model can access:
       seg0 (tokens   0– 511): 0 prior context for ALL models (same starting point)
       seg1 (tokens 512–1023): 0 prior for ckpt1,  512 for ckpt2/baseline
       seg2 (tokens 1024–1535): 0 prior for ckpt1, 1024 for ckpt2/baseline
       seg3 (tokens 1536–2047): 0 prior for ckpt1, 1536 for ckpt2/baseline
     A model with ctx=512 can only forward 512 tokens at once, so segments 1–3
     are evaluated as fresh 512-token windows (no cross-segment context).
     A model with ctx=2048 can see the full 2048-token sequence in one pass.

  B. Needle in Haystack (primary task)
     For each trial:
       1. Sample ~1800 tokens of real validation text as the "haystack".
       2. Encode a random 4-digit "secret code" DDDD.
       3. Inject the needle sentence "The secret code is DDDD." at token offset P
          from the END of the haystack (P ∈ {64, 256, 512, 768, 1024, 1536}).
       4. Append the suffix " The secret code is:" as the query.
       5. Measure whether the model's greedy prediction of the next tokens equals DDDD.
          Accuracy is per-digit (4 digits), to give a gradient even with partial retrieval.
       Distances P > ctx_len are "impossible" for that model (needle is outside its window)
       and expected to produce accuracy ≈ 0.25 (random chance on 0–9 digits).

Usage (run from the a3/ directory so that both nanochat and part3 are importable):
    cd /root/nanochat/..  # i.e. the a3/ directory
    python -m part3.eval_longctx \\
        --tags part3/d20_ctx512,part3/d20_ctx2048,part3/d20_baseline \\
        --output /vol/nanochat_cache/part3_eval_results.json \\
        --n-samples 200
"""

import os
import sys
import json
import math
import random
import string
import argparse
import logging
from typing import List, Dict

import torch
import torch.nn.functional as F

# The script must be run from the a3/ directory (or with nanochat on PYTHONPATH)
# so that `import nanochat` and `import part3` both work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nanochat.common import get_base_dir, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SEG_LEN = 512
N_SEGS = 4
FULL_LEN = SEG_LEN * N_SEGS


NEEDLE_DISTANCES = [64, 256, 512, 768, 1024, 1536]


HAYSTACK_TOKENS = 1800


def _load_nanochat_model(tag: str, device: torch.device):
    log.info(f"Loading model: {tag}")
    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=tag)
    ctx_len = meta["model_config"]["sequence_len"]
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        f"  Loaded tag='{tag}'  ctx={ctx_len}  params={n_params / 1e6:.1f}M  step={meta['step']}"
    )
    return model, tokenizer, ctx_len


@torch.no_grad()
def _compute_segment_bpb(
    model,
    token_bytes_tensor: torch.Tensor,
    x_seqs: torch.Tensor,
    y_seqs: torch.Tensor,
    ctx_len: int,
    seg_idx: int,
) -> float:
    device = x_seqs.device
    N = x_seqs.shape[0]
    seg_start = seg_idx * SEG_LEN
    seg_end = seg_start + SEG_LEN

    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)

    for i in range(N):
        x_seq = x_seqs[i]
        y_seq = y_seqs[i]

        if ctx_len >= FULL_LEN:
            inp = x_seq[:seg_end].unsqueeze(0)
            logits = model(inp)
            seg_logits = logits[0, seg_start:seg_end, :]
            seg_tgts = y_seq[seg_start:seg_end]
        else:
            inp = x_seq[seg_start:seg_end].unsqueeze(0)
            logits = model(inp)
            seg_logits = logits[0]
            seg_tgts = y_seq[seg_start:seg_end]

        losses = F.cross_entropy(
            seg_logits,
            seg_tgts,
            reduction="none",
        )

        tok_b = token_bytes_tensor[seg_tgts]
        valid = tok_b > 0
        total_nats += (losses * valid.float()).sum()
        total_bytes += tok_b.sum()

    nats_val = total_nats.item()
    bytes_val = total_bytes.item()
    if bytes_val == 0:
        return float("inf")
    return nats_val / (math.log(2) * bytes_val)


def eval_bpb_by_position(
    models_and_meta: List[dict],
    n_samples: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    log.info("=== BPB by Context Position ===")

    first = models_and_meta[0]
    tokenizer = first["tokenizer"]
    token_bytes_tensor = get_token_bytes(device=device)

    log.info(f"  Collecting {n_samples} validation sequences of length {FULL_LEN}...")
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, B=1, T=FULL_LEN, split="val", device=device
    )
    x_list, y_list = [], []
    for _ in range(n_samples):
        x, y = next(val_loader)

        x_list.append(x[0].clone())
        y_list.append(y[0].clone())
    x_seqs = torch.stack(x_list, dim=0)
    y_seqs = torch.stack(y_list, dim=0)

    results = {}
    for meta in models_and_meta:
        tag = meta["tag"]
        model = meta["model"]
        ctx_len = meta["ctx_len"]

        log.info(f"  Model: {tag} (ctx={ctx_len})")
        model.eval()
        seg_bpb = {}
        with (
            torch.no_grad(),
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            for seg_idx in range(N_SEGS):
                bpb = _compute_segment_bpb(
                    model, token_bytes_tensor, x_seqs, y_seqs, ctx_len, seg_idx
                )
                key = f"seg{seg_idx}"
                seg_bpb[key] = bpb
                log.info(
                    f"    {key} (tokens {seg_idx * SEG_LEN}–{(seg_idx + 1) * SEG_LEN - 1}): bpb={bpb:.6f}"
                )

        results[tag] = seg_bpb

    return results


def _make_needle_trial(
    tokenizer,
    val_sequences: list,
    code: str,
    distance: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    needle_text = f" The secret code is {code}."
    query_text = " The secret code is:"

    needle_ids = tokenizer.encode(needle_text)
    query_ids = tokenizer.encode(query_text)
    needle_len = len(needle_ids)
    query_len = len(query_ids)

    digit_ids = []
    for ch in code:
        ids = tokenizer.encode(f" {ch}")

        digit_ids.append(ids[-1] if ids else 0)

    tokens_before_needle = max(0, distance - needle_len)

    tokens_after_needle = HAYSTACK_TOKENS - tokens_before_needle

    haystack_token_pool = []
    for seq_tok in val_sequences:
        haystack_token_pool.extend(seq_tok)
        if len(haystack_token_pool) >= HAYSTACK_TOKENS + 100:
            break
    haystack_token_pool = haystack_token_pool[: HAYSTACK_TOKENS + 100]

    before_tokens = haystack_token_pool[:tokens_before_needle]
    after_tokens = haystack_token_pool[
        tokens_before_needle : tokens_before_needle + tokens_after_needle
    ]

    bos = tokenizer.get_bos_token_id()
    all_tokens = [bos] + before_tokens + needle_ids + after_tokens + query_ids

    input_tensor = torch.tensor([all_tokens], dtype=torch.long, device=device)
    return input_tensor, digit_ids


@torch.no_grad()
def _predict_digits(
    model,
    input_tensor: torch.Tensor,
    digit_ids: list[int],
    ctx_len: int,
) -> float:
    L = input_tensor.shape[1]
    if L > ctx_len:
        input_tensor = input_tensor[:, -ctx_len:]

    correct = 0
    for expected_id in digit_ids:
        logits = model(input_tensor)
        next_logits = logits[0, -1, :]
        predicted_id = int(next_logits.argmax().item())
        if predicted_id == expected_id:
            correct += 1

        next_tok = torch.tensor(
            [[expected_id]], dtype=torch.long, device=input_tensor.device
        )
        input_tensor = torch.cat([input_tensor, next_tok], dim=1)
        if input_tensor.shape[1] > ctx_len:
            input_tensor = input_tensor[:, -ctx_len:]

    return correct / len(digit_ids)


def eval_needle_in_haystack(
    models_and_meta: List[dict],
    n_samples: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    log.info("=== Needle in Haystack ===")

    first = models_and_meta[0]
    tokenizer = first["tokenizer"]

    n_pool = max(50, n_samples // 4)
    log.info(f"  Building haystack pool from {n_pool} validation sequences...")
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, B=1, T=FULL_LEN, split="val", device="cpu"
    )
    haystack_pool = []
    for _ in range(n_pool):
        x, _ = next(val_loader)
        haystack_pool.append(x[0].tolist())

    flat_pool = []
    for seq in haystack_pool:
        flat_pool.extend(seq)

    results = {meta["tag"]: {} for meta in models_and_meta}

    rng = random.Random(42)

    for distance in NEEDLE_DISTANCES:
        log.info(f"  Distance P={distance}...")

        trial_inputs = []
        trial_digit_ids = []
        for _ in range(n_samples):
            code = "".join(rng.choice(string.digits) for _ in range(4))

            offset = rng.randint(0, max(0, len(flat_pool) - HAYSTACK_TOKENS - 100))
            haystack_slice = [flat_pool[offset : offset + HAYSTACK_TOKENS]]
            inp, dids = _make_needle_trial(
                tokenizer, haystack_slice, code, distance, device
            )
            trial_inputs.append(inp)
            trial_digit_ids.append(dids)

        for meta in models_and_meta:
            tag = meta["tag"]
            model = meta["model"]
            ctx_len = meta["ctx_len"]

            model.eval()
            total_acc = 0.0
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                for inp, dids in zip(trial_inputs, trial_digit_ids):
                    inp = inp.to(device)
                    acc = _predict_digits(model, inp, dids, ctx_len)
                    total_acc += acc

            mean_acc = total_acc / n_samples
            results[tag][str(distance)] = mean_acc
            log.info(f"    {tag} (ctx={ctx_len}): acc={mean_acc:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Part 3 long-context evaluation: BPB-by-position + needle-in-haystack"
    )
    parser.add_argument(
        "--tags",
        type=str,
        required=True,
        help="Comma-separated list of model tags, e.g. 'part3/d20_ctx512,part3/d20_ctx2048'",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path for results",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of samples per distance/segment for each eval (default: 200)",
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="",
        help="cuda|cpu|mps (empty = autodetect)",
    )
    args = parser.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    log.info(f"Evaluating {len(tags)} models: {tags}")
    log.info(f"n_samples={args.n_samples}  output={args.output}")

    device_type_str = (
        autodetect_device_type() if args.device_type == "" else args.device_type
    )
    device = torch.device(device_type_str)

    models_and_meta = []
    for tag in tags:
        model, tokenizer, ctx_len = _load_nanochat_model(tag, device)
        models_and_meta.append(
            {
                "tag": tag,
                "model": model,
                "tokenizer": tokenizer,
                "ctx_len": ctx_len,
            }
        )

    bpb_results = eval_bpb_by_position(models_and_meta, args.n_samples, device)
    needle_results = eval_needle_in_haystack(models_and_meta, args.n_samples, device)

    output = {
        "bpb_by_position": bpb_results,
        "needle": {
            "distances": NEEDLE_DISTANCES,
            **needle_results,
        },
        "config": {
            "n_samples": args.n_samples,
            "needle_distances": NEEDLE_DISTANCES,
            "seg_len": SEG_LEN,
            "full_seq_len": FULL_LEN,
            "haystack_tokens": HAYSTACK_TOKENS,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Results written to: {args.output}")

    print("\n=== BPB by Context Position ===")
    seg_keys = [f"seg{i}" for i in range(N_SEGS)]
    header = f"{'Model':<35} " + "  ".join(f"{k:>10}" for k in seg_keys)
    print(header)
    print("-" * len(header))
    for tag in tags:
        row = f"{tag:<35} " + "  ".join(
            f"{bpb_results.get(tag, {}).get(k, float('nan')):>10.6f}" for k in seg_keys
        )
        print(row)

    print("\n=== Needle in Haystack Accuracy ===")
    dist_strs = [str(d) for d in NEEDLE_DISTANCES]
    header2 = f"{'Model':<35} " + "  ".join(f"P={d:>4}" for d in NEEDLE_DISTANCES)
    print(header2)
    print("-" * len(header2))
    for tag in tags:
        row2 = f"{tag:<35} " + "  ".join(
            f"{needle_results.get(tag, {}).get(ds, float('nan')):>7.3f}"
            for ds in dist_strs
        )
        print(row2)

    print(f"\nDone. Full results saved to: {args.output}")


if __name__ == "__main__":
    main()
