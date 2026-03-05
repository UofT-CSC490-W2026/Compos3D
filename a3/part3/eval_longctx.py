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

  B. Needle in Haystack (primary task) — likelihood-ranking variant
     For each trial:
       1. Sample ~1800 tokens of real validation text as the "haystack".
       2. Choose a random 4-digit "secret code" DDDD and N_DISTRACTORS other codes.
       3. Inject the needle sentence "The secret code is DDDD." at token offset P
          from the END of the haystack (P ∈ {64, 256, 512, 768, 1024, 1536}).
       4. Append the suffix " The secret code is:" as the query.
       5. Score each candidate code (correct + distractors) by summing log-probs of
          its 4 digit tokens using a single teacher-forced forward pass per candidate.
       6. A trial is correct if the true code scores higher than all N_DISTRACTORS
          distractors (10-way ranking; random-chance baseline = 10%).
     Distances P > ctx_len are "impossible" for that model (needle is outside its
     window) and expected to produce accuracy ≈ 10% (random chance).

Usage (run from the a3/ directory so that both nanochat and part3 are importable):
    cd /root/nanochat/..  # i.e. the a3/ directory
    python -m part3.eval_longctx \\
        --tags part3/d20_ctx512,part3/d20_ctx2048,part3/d20_baseline \\
        --output /vol/nanochat_cache/part3_eval_results.json \\
        --n-samples 200

    # needle-only re-run (skips BPB, merges into existing JSON):
    python -m part3.eval_longctx \\
        --tags part3/d20_ctx512,part3/d20_ctx2048,part3/d20_baseline \\
        --output /vol/nanochat_cache/part3_eval_results.json \\
        --n-samples 200 --skip-bpb
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

# Number of distractor codes per trial.  Total candidates = N_DISTRACTORS + 1,
# so random-chance accuracy = 1 / (N_DISTRACTORS + 1).
N_DISTRACTORS = 9  # 10-way ranking; random baseline = 10 %

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
    """Build the context tensor and correct digit IDs for one needle trial."""
    needle_text = f" The secret code is {code}."
    query_text = " The secret code is:"

    needle_ids = tokenizer.encode(needle_text)
    query_ids = tokenizer.encode(query_text)
    needle_len = len(needle_ids)

    digit_ids = _code_to_digit_ids(tokenizer, code)

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


def _code_to_digit_ids(tokenizer, code: str) -> list[int]:
    """Tokenize a code string into per-character token IDs (space-prefixed)."""
    ids = []
    for ch in code:
        toks = tokenizer.encode(f" {ch}")
        ids.append(toks[-1] if toks else 0)
    return ids


@torch.no_grad()
def _score_code_teacher_forced(
    model,
    context: torch.Tensor,
    digit_ids: list[int],
    ctx_len: int,
) -> float:
    """
    Score a candidate code via a single teacher-forced forward pass.

    Concatenates `context` with the first (n-1) digit tokens, runs one forward
    pass, then reads off the log-prob of each digit token at the corresponding
    output position.  This is equivalent to autoregressive scoring but requires
    only one model call instead of n_digits calls.

    Args:
        context:   [1, L] int64 tensor — the prompt up to (and including) the
                   query suffix " The secret code is:".
        digit_ids: list of n_digits token IDs for the candidate code.
        ctx_len:   maximum sequence length the model accepts.

    Returns:
        Sum of log-probs for the digit tokens (higher = more likely).
    """
    n = len(digit_ids)
    # Append all but the last digit to the context so the model predicts every
    # digit in one pass (classic teacher-forcing).
    prefix = torch.tensor([digit_ids[:-1]], dtype=torch.long, device=context.device)
    inp = torch.cat([context, prefix], dim=1)          # shape [1, L + n - 1]

    if inp.shape[1] > ctx_len:
        inp = inp[:, -ctx_len:]

    logits = model(inp)                                # [1, T, vocab]
    log_probs = F.log_softmax(logits[0], dim=-1)       # [T, vocab]

    # Positions [-n], [-n+1], ..., [-1] in the output predict digit_ids[0..n-1]
    total = 0.0
    for i, tok_id in enumerate(digit_ids):
        pos = -(n - i)          # e.g. for n=4: -4, -3, -2, -1
        total += log_probs[pos, tok_id].item()

    return total


@torch.no_grad()
def _rank_correct_code(
    model,
    input_tensor: torch.Tensor,
    correct_digit_ids: list[int],
    distractor_digit_ids_list: list[list[int]],
    ctx_len: int,
) -> float:
    """
    Return 1.0 if the correct code scores higher than every distractor, else 0.0.
    Uses teacher-forced scoring (one forward pass per candidate).
    """
    correct_score = _score_code_teacher_forced(
        model, input_tensor, correct_digit_ids, ctx_len
    )
    for dist_ids in distractor_digit_ids_list:
        if _score_code_teacher_forced(model, input_tensor, dist_ids, ctx_len) >= correct_score:
            return 0.0
    return 1.0


def eval_needle_in_haystack(
    models_and_meta: List[dict],
    n_samples: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Needle-in-haystack evaluation using likelihood ranking.

    For each trial, the correct 4-digit code is scored against N_DISTRACTORS
    random codes via teacher-forced log-prob.  A trial is marked correct (1.0)
    if the true code ranks first among all candidates.

    Random-chance baseline accuracy = 1 / (N_DISTRACTORS + 1) = {:.1%}.
    """.format(1 / (N_DISTRACTORS + 1))
    log.info("=== Needle in Haystack (likelihood ranking, %d-way) ===", N_DISTRACTORS + 1)

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

        # Pre-generate all trials (context tensors + correct and distractor digit IDs)
        trial_inputs = []
        trial_correct_ids = []
        trial_distractor_ids = []  # list of lists (N_DISTRACTORS per trial)

        for _ in range(n_samples):
            code = "".join(rng.choice(string.digits) for _ in range(4))

            # Generate N_DISTRACTORS unique codes different from the correct one
            distractors = []
            while len(distractors) < N_DISTRACTORS:
                d = "".join(rng.choice(string.digits) for _ in range(4))
                if d != code and d not in distractors:
                    distractors.append(d)

            offset = rng.randint(0, max(0, len(flat_pool) - HAYSTACK_TOKENS - 100))
            haystack_slice = [flat_pool[offset : offset + HAYSTACK_TOKENS]]
            inp, correct_dids = _make_needle_trial(
                tokenizer, haystack_slice, code, distance, device
            )
            distractor_dids_list = [
                _code_to_digit_ids(tokenizer, d) for d in distractors
            ]

            trial_inputs.append(inp)
            trial_correct_ids.append(correct_dids)
            trial_distractor_ids.append(distractor_dids_list)

        for meta in models_and_meta:
            tag = meta["tag"]
            model = meta["model"]
            ctx_len = meta["ctx_len"]

            model.eval()
            total_correct = 0.0
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                for inp, c_ids, d_ids_list in zip(
                    trial_inputs, trial_correct_ids, trial_distractor_ids
                ):
                    inp = inp.to(device)
                    total_correct += _rank_correct_code(
                        model, inp, c_ids, d_ids_list, ctx_len
                    )

            mean_acc = total_correct / n_samples
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
    parser.add_argument(
        "--skip-bpb",
        action="store_true",
        default=False,
        help="Skip BPB-by-position eval and only run needle-in-haystack. "
             "If --output already exists the BPB section is preserved.",
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

    # Load existing results if skipping BPB (to preserve previously computed values)
    existing = {}
    if args.skip_bpb and os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        log.info(f"Loaded existing results from {args.output} (BPB will be preserved)")

    if args.skip_bpb:
        bpb_results = existing.get("bpb_by_position", {})
    else:
        bpb_results = eval_bpb_by_position(models_and_meta, args.n_samples, device)

    needle_results = eval_needle_in_haystack(models_and_meta, args.n_samples, device)

    output = {
        "bpb_by_position": bpb_results,
        "needle": {
            "distances": NEEDLE_DISTANCES,
            "n_distractors": N_DISTRACTORS,
            **needle_results,
        },
        "config": {
            "n_samples": args.n_samples,
            "needle_distances": NEEDLE_DISTANCES,
            "n_distractors": N_DISTRACTORS,
            "needle_eval": "likelihood_ranking",
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
