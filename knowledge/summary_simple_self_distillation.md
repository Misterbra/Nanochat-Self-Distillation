# Simple Self-Distillation (SSD) for Code Generation

**Paper:** "Embarrassingly Simple Self-Distillation Improves Code Generation"
**Authors:** Ruixiang Zhang, Richard He Bai, Huangjie Zheng, Navdeep Jaitly, Ronan Collobert, Yizhe Zhang (Apple)
**arXiv:** 2604.01193v1 (April 2026)
**Code:** https://github.com/apple/ml-ssd

---

## Core Idea

SSD is a three-step method that improves code generation using only the model's own unverified outputs:

1. **Sample** solutions from the frozen model at a non-unit temperature `T_train` with truncation (top-k/top-p)
2. **Fine-tune** on those raw samples using standard supervised cross-entropy loss (SFT)
3. **Decode** the fine-tuned model at evaluation with a separately tuned temperature `T_eval`

**No RL, no verifier, no teacher model, no execution environment, no correctness filtering.**

## Key Results

- Qwen3-30B-Instruct: 42.4% -> 55.3% pass@1 on LiveCodeBench v6 (+12.9pp, +30% relative)
- Gains concentrate on harder problems (+15.3pp on hard vs +6.5pp on easy)
- Generalizes across Qwen and Llama models at 4B, 8B, 30B scale
- Works for both instruct and thinking model variants
- pass@5 gains often exceed pass@1 gains (preserves diversity)

## Why It Works: The Precision-Exploration Conflict

Code has two types of token positions:
- **Locks**: positions where syntax/semantics demand one correct token (e.g., after `if n ==`). Need precision.
- **Forks**: positions where multiple valid continuations exist (e.g., choosing algorithm). Need exploration.

A single global decoding temperature can't satisfy both:
- Low T_eval: good at locks (suppresses distractors) but starves forks
- High T_eval: good at forks (enables diversity) but destabilizes locks

**SSD reshapes distributions context-dependently:**
- At locks: strips the distractor tail, concentrating mass on the dominant token
- At forks: preserves multiple viable continuations while removing useless tail

This is formalized as **support compression** (via truncation) and **within-support reshaping** (via temperature):

```
L(θ) = -log KeptMass_θ          [support compression]
     + (1-T) H_{1/T}(p_θ(·|S))  [within-support reshaping]
     + T · KL(q || p_{θ,T}(·|S)) [alignment to base model]
     + const
```

## Hyperparameters & Configuration

### Data Synthesis
- ~10K unique competitive programming problems (from rSTARcoder dataset, de-duplicated)
- **N=1**: a single sample per prompt already suffices
- Only minimal syntactic filtering (remove empty responses and single-line stubs)
- No correctness signal whatsoever
- Generation with vLLM, 128K max sequence length

### Training Configuration (from paper)
| Parameter | Instruct Models | Thinking Models |
|-----------|----------------|-----------------|
| Optimizer | AdamW with cosine decay | AdamW with cosine decay |
| Peak LR | 5 × 10⁻⁶ | 5 × 10⁻⁶ |
| Global batch size | 32 | 32 |
| Sequence length | 65,536 | 65,536 |
| Iterations | 2,500 | 300 |
| Warmup iterations | 250 | 50 |
| Hardware | 8×B200 GPUs | 8×B200 GPUs |

### Decoding Parameters (per model, from paper's Table 3)
| Model | T_train | top-k | top-p | T_eval | top-k_eval | top-p_eval |
|-------|---------|-------|-------|--------|------------|------------|
| Qwen3-30B-Instruct | 1.5 | 20 | 0.8 | 0.9 | 20 | 0.8 |
| Qwen3-4B-Instruct | 1.5 | 20 | 0.8 | 0.9 | 20 | 0.8 |
| Llama-3.1-8B-Instruct | 1.5 | 20 | 0.8 | 0.9 | 20 | 0.8 |
| Qwen3-30B-Thinking | 1.5 | 20 | 0.95 | 0.6 | 20 | 0.95 |
| Qwen3-4B-Thinking | 1.5 | 20 | 0.95 | 0.6 | 20 | 0.95 |

### Key Hyperparameter Insights
- **Effective temperature**: T_eff = T_train × T_eval governs performance (R²=0.75), peak near T_eff ≈ 1.2
- **Truncation raises the ceiling**: truncated runs consistently outperform no-truncation runs
- **Broad plateau**: performance is robust across a wide range of T_train × T_eval combinations
- Even "bad data" (T_train=2.0, gibberish outputs) still improves the model

## Connection to Nanochat

### What Nanochat Already Has
1. **SFT pipeline** (`scripts/chat_sft.py`): Standard supervised fine-tuning with BOS-aligned bestfit-pad packing, AdamW+Muon optimizer, LR scheduling. Currently trains on SmolTalk, MMLU, GSM8K, SpellingBee, and identity data.
2. **RL pipeline** (`scripts/chat_rl.py`): GRPO-style reinforcement learning on GSM8K.
3. **Engine** (`nanochat/engine.py`): Token generation with KV cache, tool use, top-k sampling, temperature control. Already supports `temperature` and `top_k` parameters.
4. **Evaluation** (`scripts/chat_eval.py`): Supports HumanEval (code), GSM8K (math), MMLU, ARC, SpellingBee.
5. **Task system** (`tasks/common.py`): Extensible Task base class with TaskMixture for combining data.
6. **CustomJSON task** (`tasks/customjson.py`): Can load arbitrary JSONL conversation data.

### How to Implement SSD in Nanochat

The SSD method maps naturally onto nanochat's existing infrastructure:

#### Step 1: Data Synthesis Script (`scripts/chat_ssd_generate.py`)
- Load the SFT model (or base model) using `load_model("sft", ...)`
- Create an `Engine` and use `engine.generate_batch()` with elevated temperature and top-k truncation
- For each coding problem prompt, sample N=1 solution
- Save as JSONL conversations in the CustomJSON format
- Apply minimal filtering (remove empty/single-line responses)
- Key params: `T_train=1.5`, `top_k=20`, `top_p=0.8` (adapt for nanochat's smaller scale)

#### Step 2: SSD Fine-tuning Script (`scripts/chat_ssd_train.py`)
- Very similar to existing `chat_sft.py`
- Load the base/SFT model
- Train on the self-generated data using CustomJSON task
- Key differences from standard SFT:
  - Lower learning rate (5 × 10⁻⁶ peak)
  - Fewer iterations (since dataset is smaller)
  - Train only on self-generated code data

#### Step 3: Evaluation with Tuned Decoding
- Use `chat_eval.py` with adjusted temperature/top-k for the SSD model
- `T_eval=0.9`, `top_k=20` for instruct models

### Adaptations for Nanochat's Scale

Nanochat is much smaller than the 4B-30B models in the paper. Key considerations:
1. **Temperature sensitivity**: Smaller models may need lower T_train (e.g., 1.0-1.2 instead of 1.5)
2. **Dataset size**: Even a few hundred coding problems may suffice given N=1
3. **Training iterations**: Scale down proportionally (maybe 100-500 iterations)
4. **Code generation quality**: The base model needs to generate somewhat coherent code for SSD to work, though even gibberish data showed improvements in the paper
5. **Evaluation**: HumanEval is already available as a task; could also add competitive programming prompts

### Data Sources for Coding Prompts
- HumanEval problems (164 problems, already in the repo)
- Could fetch competitive programming problems from datasets like APPS, CodeContests
- Even using the existing GSM8K/SpellingBee prompts might show improvements via the SSD mechanism

### What Makes This Exciting for Nanochat
1. **Zero external dependencies**: No teacher model, no verifier, no labeled solutions needed
2. **Complementary to RL**: SSD and GRPO target different mechanisms (distribution reshaping vs reward optimization)
3. **Simple implementation**: Reuses existing SFT pipeline almost entirely
4. **Could improve not just code but potentially other tasks** via the same mechanism
