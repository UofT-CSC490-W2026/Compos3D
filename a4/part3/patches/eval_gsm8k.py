import argparse
import torch
import torch.distributed as dist
from contextlib import nullcontext

from nanochat.common import (
    compute_init,
    compute_cleanup,
    print0,
    get_base_dir,
    DummyWandb,
    autodetect_device_type,
)
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

import wandb


parser = argparse.ArgumentParser(
    description="GSM8K pass@k evaluation (Karpathy protocol)"
)

parser.add_argument(
    "--run",
    type=str,
    default="dummy",
    help="wandb run name ('dummy' disables wandb logging)",
)

parser.add_argument(
    "--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)"
)
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")

parser.add_argument(
    "--source",
    type=str,
    default="sft",
    help="mid|sft|rl - which checkpoint directory to load from",
)
parser.add_argument(
    "--model-tag", type=str, default=None, help="model tag to load from"
)
parser.add_argument(
    "--model-step",
    type=int,
    default=None,
    help="specific checkpoint step to load (None = latest)",
)

parser.add_argument(
    "--device-batch-size",
    type=int,
    default=8,
    help="number of samples per problem (= pass@k k-value)",
)
parser.add_argument(
    "--eval-examples",
    type=int,
    default=400,
    help="number of GSM8K test problems to evaluate",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="sampling temperature (1.0 = Karpathy protocol)",
)
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling")
parser.add_argument(
    "--max-completion-tokens", type=int, default=256, help="max tokens per completion"
)
args = parser.parse_args()
user_config = vars(args).copy()


device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    if device_type == "cuda"
    else nullcontext()
)


use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = (
    DummyWandb()
    if use_dummy_wandb
    else wandb.init(
        project="nanochat-a4-part3",
        name=args.run,
        config=user_config,
    )
)


model, tokenizer, meta = load_model(
    args.source, device, phase="eval", model_tag=args.model_tag, step=args.model_step
)
engine = Engine(model, tokenizer)
model.eval()


val_task = GSM8K(subset="main", split="test")


@torch.no_grad()
def run_gsm8k_eval(
    task,
    tokenizer,
    engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50,
):
    max_examples = (
        min(max_examples, len(task)) if max_examples is not None else len(task)
    )
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        assert num_samples <= args.device_batch_size
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({"is_correct": is_correct})
        record = {"idx": idx, "outcomes": outcomes}
        yield record


print0(
    f"Evaluating {args.source} checkpoint (tag={args.model_tag}, step={args.model_step})"
)
print0(
    f"Protocol: {args.eval_examples} problems, {args.device_batch_size} samples each, temperature={args.temperature}"
)

passk = torch.zeros(args.device_batch_size, device=device)
with autocast_ctx:
    records_iter = run_gsm8k_eval(
        val_task,
        tokenizer,
        engine,
        num_samples=args.device_batch_size,
        max_examples=args.eval_examples,
        temperature=args.temperature,
        top_k=args.top_k,
        max_completion_tokens=args.max_completion_tokens,
    )
    records = list(records_iter)


for k in range(1, args.device_batch_size + 1):
    passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)

num_records = torch.tensor(len(records), dtype=torch.long, device=device)
if ddp:
    dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
    dist.all_reduce(passk, op=dist.ReduceOp.SUM)

passk = passk / num_records.item()

print_passk = [
    f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)
]
print0(f"Results: {', '.join(print_passk)}")
print0(f"Total problems evaluated: {num_records.item()}")

log_passk = {
    f"pass@{k}": passk[k - 1].item() for k in range(1, args.device_batch_size + 1)
}
wandb_run.log(log_passk)

if master_process and args.run != "dummy":
    print(f"Results logged to W&B project nanochat-a4-part3 run {args.run}")

compute_cleanup()
