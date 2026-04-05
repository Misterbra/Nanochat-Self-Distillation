"""
Simple Self-Distillation (SSD) - Data Synthesis Step.

Generates self-distillation training data by sampling solutions from the model
at elevated temperature with top-k/top-p truncation. The raw, unverified outputs
are saved as JSONL for subsequent SSD fine-tuning.

Reference: "Embarrassingly Simple Self-Distillation Improves Code Generation"
(Zhang et al., 2026, arXiv:2604.01193)

Usage:
    python -m scripts.chat_ssd_generate
    python -m scripts.chat_ssd_generate --source sft --temperature 1.5 --top-k 20 --top-p 0.8
"""

import argparse
import json
import os
import time
import torch

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="SSD data synthesis: sample solutions from the model")
# Model loading
parser.add_argument("--source", type=str, default="sft", help="Model source: base|sft|rl (default: sft)")
parser.add_argument("--model-tag", type=str, default=None, help="Model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="Model step to load from")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Sampling configuration (SSD hyperparameters)
parser.add_argument("--temperature", type=float, default=1.5, help="Training-time sampling temperature T_train (default: 1.5)")
parser.add_argument("--top-k", type=int, default=20, help="Top-k truncation during sampling (default: 20)")
parser.add_argument("--top-p", type=float, default=0.8, help="Top-p (nucleus) truncation during sampling (default: 0.8)")
parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max tokens to generate per solution (default: 1024)")
parser.add_argument("--num-samples", type=int, default=1, help="Number of samples per prompt (default: 1)")
# Data
parser.add_argument("--dataset", type=str, default="humaneval", help="Dataset to use for prompts: humaneval (default: humaneval)")
parser.add_argument("--output", type=str, default=None, help="Output JSONL file path (default: auto)")
args = parser.parse_args()

# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

# Load the model
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer)

# Output path
base_dir = get_base_dir()
if args.output:
    output_path = args.output
else:
    output_path = os.path.join(base_dir, "ssd_training_data.jsonl")

# Load the prompts
def load_prompts(dataset_name):
    """Load coding problem prompts for SSD data synthesis."""
    if dataset_name == "humaneval":
        from tasks.humaneval import HumanEval
        task = HumanEval()
        prompts = []
        for i in range(len(task)):
            conversation = task[i]
            # Use the problem prompt (function signature + docstring)
            user_content = conversation["messages"][0]["content"]
            prompts.append(user_content)
        return prompts
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

prompts = load_prompts(args.dataset)
num_prompts = len(prompts)
print0(f"Loaded {num_prompts} prompts from {args.dataset}")
print0(f"SSD sampling config: T_train={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
print0(f"Generating {args.num_samples} sample(s) per prompt, max {args.max_new_tokens} tokens each")

# Distribute prompts across ranks
rank_prompts = list(range(ddp_rank, num_prompts, ddp_world_size))
print0(f"This rank will process {len(rank_prompts)} prompts")

# Generate solutions
results = []
t0 = time.time()
for count, prompt_idx in enumerate(rank_prompts):
    prompt_text = prompts[prompt_idx]

    # Render the prompt as a conversation for the model
    conversation = {
        "messages": [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": "placeholder"}, # will be popped by render_for_completion
        ]
    }
    tokens = tokenizer.render_for_completion(conversation)

    # Generate samples with SSD decoding configuration
    for sample_idx in range(args.num_samples):
        seed = hash((prompt_idx, sample_idx)) & 0x7FFFFFFF
        generated_seqs, masks = engine.generate_batch(
            tokens,
            num_samples=1,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=seed,
        )

        # Decode the generated tokens (skip the prompt)
        prefix_length = len(tokens)
        generated_tokens = generated_seqs[0][prefix_length:]
        completion = tokenizer.decode(generated_tokens)

        # Minimal syntactic filtering: skip empty or single-line stubs
        stripped = completion.strip()
        if not stripped:
            continue
        if stripped.count('\n') == 0 and len(stripped) < 20:
            continue

        # Save as a conversation in CustomJSON format
        result_messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": completion},
        ]
        results.append(result_messages)

    # Progress logging
    if (count + 1) % 10 == 0 or count == len(rank_prompts) - 1:
        elapsed = time.time() - t0
        rate = (count + 1) / elapsed
        print(f"\rRank {ddp_rank} | {count + 1}/{len(rank_prompts)} prompts ({rate:.1f} prompts/sec) | {len(results)} samples collected", end='', flush=True)

print()  # newline after progress

# Gather results from all ranks (simple: each rank writes to its own temp file, rank 0 merges)
if ddp:
    import torch.distributed as dist
    # Write rank results to temp files
    temp_path = output_path + f".rank{ddp_rank}.tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        for messages in results:
            f.write(json.dumps(messages) + '\n')
    dist.barrier()

    # Rank 0 merges all temp files
    if master_process:
        with open(output_path, 'w', encoding='utf-8') as fout:
            for rank in range(ddp_world_size):
                rank_temp = output_path + f".rank{rank}.tmp"
                with open(rank_temp, 'r', encoding='utf-8') as fin:
                    fout.write(fin.read())
                os.remove(rank_temp)
        total_samples = sum(1 for _ in open(output_path))
        print0(f"Saved {total_samples} SSD training samples to {output_path}")
else:
    # Single process: write directly
    with open(output_path, 'w', encoding='utf-8') as f:
        for messages in results:
            f.write(json.dumps(messages) + '\n')
    print0(f"Saved {len(results)} SSD training samples to {output_path}")

elapsed_total = time.time() - t0
print0(f"Total time: {elapsed_total:.1f}s")

# Log to report
from nanochat.report import get_report
get_report().log(section="SSD Data Synthesis", data=[
    vars(args),
    {
        "num_prompts": num_prompts,
        "num_samples_generated": len(results),
        "total_time_seconds": elapsed_total,
    }
])

compute_cleanup()
