# emo

Teaching Qwen2.5-3B-Instruct to communicate in emoji by training it with GRPO on a multi-turn guessing game. Inspired by [Pantheon](https://en.wikipedia.org/wiki/Pantheon_(TV_series))'s "low-bandwidth mode."

![Pantheon low-bandwidth mode](emo.gif)

## Approach

A small language model is the sender. Its output space is constrained to emoji-only tokens via a logit mask. For each game it's given a target phrase and up to N turns to make a simulated guesser figure it out. Each turn it sees the guesser's previous attempts and tries again.

**Reward shape (per trajectory):**

```
trajectory_reward = final_sim + completion_bonus
                  - turn_cost × n_turns
                  - Σ rep_penalties
```

- `final_sim` — sentence-embedding similarity between the guesser's final guess and the target.
- `completion_bonus = +1.0` when `final_sim ≥ 0.65`.
- `turn_cost = 0.05` per turn — discourages dragging the game out.
- `rep_penalty = 1 − unique/total` per turn — discourages spamming a single emoji.

Training is GRPO on top of base `Qwen2.5-3B-Instruct` with a LoRA adapter — no SFT phase. The logit mask hard-constrains the output space to emoji at generation time, so SFT for format compliance is redundant; the base model already knows enough about emoji semantics from pretraining to give GRPO a usable starting point. The production checkpoint is `step_200` from run 4.

> **History note.** Earlier iterations of this project used [Tinker](https://tinker.thinkingmachines.ai) for both SFT and RL on `Qwen3.5-4B`. I hit enough friction — slow turnaround on multi-turn RL, limited control over the rollout loop, harder to debug reward shaping interactively — that I rewrote the training stack against Modal with a custom GRPO loop. The legacy Tinker code still lives under `src/rl/tinker/` for reference. Everything described above runs through `src/rl/custom/`.

## Stack

- **Sender:** `Qwen2.5-3B-Instruct` + LoRA (the model being trained)
- **Simulated guesser (training):** Claude Sonnet
- **Phrase generator (live games):** Claude Haiku
- **Embeddings:** `all-MiniLM-L6-v2` (for the similarity reward)
- **Training / serving:** Modal
- **Logging:** Weights & Biases
- **Web app:** Next.js (in `web/`)

## Repo layout

```
src/
├── data/
│   ├── prompts.py                # generate diverse NL prompts via Claude API
│   ├── generate_rl_prompts.py    # curate target phrases for RL (545 train + 50 held-out, 10 themes)
│   ├── filter_rl_prompts.py      # filter phrases by emoji-communicability
│   └── emoji_tokens.py           # build the emoji-safe token mask (1,258 of 151,643 tokens)
├── rl/
│   └── custom/
│       ├── env.py                # multi-turn guessing-game environment
│       ├── reward.py             # trajectory reward (final_sim + bonus - turn_cost - rep_penalty)
│       ├── generate.py           # rollout generation with the emoji logit mask
│       ├── train.py              # GRPO training loop
│       └── modal_train.py        # Modal entrypoint for training
├── viz/                          # W&B analysis, reward diagnostics, before/after plots
├── eval_retrieval.py             # held-out eval on the guessing game
└── chat.py                       # interactive chat with the emoji model

scripts/
├── generate_phrases.py           # generate phrase pools
├── eval_checkpoint.py            # evaluate a saved checkpoint
├── serve_model.py                # serve the production checkpoint via Modal
├── sweep_all.py                  # base vs intermediate vs current sweep
├── sweep_findings.md             # curated results for the writeup
└── ...

web/                              # Next.js game frontend + API routes
data/                             # generated phrase pools + emoji-test fixtures
```

## Setup

Requires an Anthropic API key and a Modal account.

```bash
uv sync
```

Create a `.env`:

```
ANTHROPIC_API_KEY=...
WANDB_API_KEY=...
```

Set up Modal:

```bash
modal setup
modal secret create anthropic-secret ANTHROPIC_API_KEY=...
modal secret create wandb-secret WANDB_API_KEY=...
```

## Usage

### Generate data

```bash
uv run python -m src.data.prompts
uv run python -m src.data.generate_rl_prompts
uv run python -m src.data.filter_rl_prompts
uv run python -m src.data.emoji_tokens   # build the emoji token mask
```

### Train

```bash
uv run python -m src.rl.custom.modal_train \
  --batch-size 32 \
  --group-size 4 \
  --max-turns 3 \
  --kl-coeff 0.05 \
  --lr 1e-5
```

Use `.spawn()` semantics — `modal_train.py` runs detached so the job survives closing your laptop.

### Evaluate

```bash
uv run python scripts/eval_checkpoint.py --checkpoint step_200
uv run python scripts/sweep_all.py    # base vs intermediate vs current
```

### Play (locally)

```bash
uv run python -m src.chat --checkpoint step_200
```

### Serve

```bash
uv run python scripts/serve_model.py
```

The web app (`web/`) hits the served endpoint and logs every completed game `(phrase, emoji_sequence, guesses, verdicts)` to a Modal Volume as JSONL for the next training pass.

## Numbers

- **Vocab:** 151,643 tokens. **Emoji-safe:** 1,258 (0.83%).
- **Phrase pool:** 545 train + 50 held-out across 10 themes.
- **Held-out completion rate** (`final_sim ≥ 0.65`) at `step_200`: **54%**. Base model: ~15–20%.
- **Avg held-out similarity** at `step_200`: **0.66**.

## Notes

- The live game uses Claude Sonnet to judge guesses and Claude Haiku to pick a phrase per game. The trained model only generates the emoji.
- The online training loop (retraining on real human gameplay) is not running yet. The deployed checkpoint has only ever been trained against the simulated guesser.