# emo

Teaching Qwen3.5-4B to communicate more effectively in emoji. Inspired by [Pantheon](https://en.wikipedia.org/wiki/Pantheon_(TV_series)). Writeup coming soon.

![Pantheon low-bandwidth mode](emo.gif)

Also just wanted to experiment with RL + SFT on Tinker.  

## Approach

**SFT**

- Train on 25K generated (prompt, emoji response) pairs
- The model learns to map natural language inputs to 2-8 emoji
- Gets 100% format compliance but the emoji are generic

**RL**

- Starting from the SFT checkpoint, train with GRPO on a multi-turn retrieval game
- A frozen judge model (Qwen3-30B-A3B) tries to reconstruct the original message from the sender's emoji
- The sender is rewarded when the judge's reconstruction is semantically close to the original
- Penalties on format violations + turn usage + KL divergence (Tinker makes this easy)
- Ideally, this pushes the model towards more productive emoji usage

## Structure 

```
src/
├── data/
│   ├── prompts.py                # generate diverse NL prompts via Claude API
│   ├── generate_sft_dataset.py   # generate emoji responses via Claude batch API
│   ├── emoji_tokens.py           # build emoji token mask for logit constraining
│   ├── generate_rl_prompts.py    # curate target messages for RL
│   └── filter_rl_prompts.py      # filter prompts by emoji-communicability
├── rl/
│   ├── env.py                    # multi-turn retrieval game environment
│   ├── judge.py                  # frozen judge model client
│   ├── reward.py                 # similarity-based reward function
│   ├── emoji_completer.py        # emoji-constrained token completer
│   ├── train.py                  # GRPO training loop
│   ├── play_game.py              # interactive retrieval game
│   └── debug_rollout.py          # debug/inspect rollout episodes
├── train_sft.py                  # SFT training script
├── eval_sft.py                   # SFT evaluation (format compliance, emoji stats)
├── eval_retrieval.py             # retrieval game evaluation (base vs SFT vs RL)
└── chat.py                       # interactive chat with the emoji model
```

## Setup

Requires Anthropic API key and a [Tinker](https://tinker.thinkingmachines.ai) account.

```bash
uv sync
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=...   # for data generation
TINKER_API_KEY=...      # for training and inference
```

## Usage

### Generate data

Generate natural language prompts, then create emoji responses via Claude:

```bash
uv run python -m src.data.prompts
uv run python -m src.data.generate_sft_dataset
```

Build the emoji token mask (needed for constrained decoding):

```bash
uv run python -m src.data.emoji_tokens
```

### SFT training + evaluation

Train the SFT checkpoint on Tinker:
```bash
uv run python -m src.train_sft \
  --train-data data/sft_train.jsonl \
  --test-data data/sft_test.jsonl \
  --lora-rank 32 \
  --learning-rate 5e-4
```

Evaluate the SFT checkpoint:
```bash
uv run python -m src.eval_sft --checkpoint <tinker-state-name>
```

### RL training + evaluation

Generate RL prompts, then train with GRPO starting from the SFT checkpoint:

```bash
uv run python -m src.data.generate_rl_prompts
uv run python -m src.rl.train \
  --sft-checkpoint <tinker-state-name> \
  --batch-size 32 \
  --group-size 4 \
  --max-turns 5
```

Run some evaluations on the retrieval game:
```bash
uv run python -m src.eval_retrieval --checkpoint <tinker-state-name>
```

### For fun, chat with the model

```bash
uv run python -m src.chat --checkpoint <tinker-state-name>
```

## Models

- **Sender:** Qwen3.5-4B (LoRA fine-tuned on Tinker)
- **Judge:** Qwen3-30B-A3B-Instruct (frozen, used for RL reward signal)
- **Embeddings:** all-MiniLM-L6-v2 (for semantic similarity rewards)

