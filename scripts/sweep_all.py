"""Sweep the same phrase list across three checkpoints for side-by-side comparison.

Checkpoints:
  base      — Qwen2.5-3B-Instruct, no adapter
  phase2    — emoji-phase2-outputs / phase2-repetition-penalty (earlier GRPO run)
  current   — emo-checkpoints / step_200 (deployed model)

Usage:
    uv run modal run scripts/sweep_all.py
    uv run modal run scripts/sweep_all.py --turns 2
"""

import json
import modal

_MODEL_CACHE_DIR = "/model-cache"
_EMO_CKPT_DIR    = "/emo-checkpoints"
_PHASE2_DIR      = "/phase2-outputs"

_model_volume   = modal.Volume.from_name("emoji-model-weights")
_emo_volume     = modal.Volume.from_name("emo-checkpoints")
_phase2_volume  = modal.Volume.from_name("emoji-phase2-outputs")

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=5.0.0",
        "peft>=0.14.0",
    )
    .add_local_python_source("src")
    .add_local_file("data/emoji-test.txt", "/root/data/emoji-test.txt")
)

app = modal.App("emo-sweep-all")

PHRASES = [
    # situational / relatable
    "parallel parking fail",
    "printer jammed before deadline",
    "drunk texting",
    "sending wrong text to wrong person",
    "running into your ex",
    "amazon package stolen doorstep",
    "spilled red wine",
    "remote control batteries dead",
    # emotional arcs
    "breakup ice cream",
    "crying at a movie",
    "valentines day disappointment",
    "surprise birthday party",
    "forgetting an anniversary",
    # adventure / journey
    "honeymoon in paris",
    "lost luggage at airport",
    "flight delay at airport",
    "camping under the stars",
    # modern life
    "video conference on mute",
    "pulling an all nighter",
    "binge watching tv show",
    "skipping the gym",
    "ghost pepper challenge",
    "midlife crisis sports car",
    # harder / abstract
    "thanksgiving dinner arguments",
    "power outage dinner",
    "high school reunion cringe",
    "hospital visit",
]

CHECKPOINTS = [
    {
        "name": "base",
        "adapter_path": None,  # no adapter — base model only
    },
    {
        "name": "phase2",
        "adapter_path": f"{_PHASE2_DIR}/checkpoints/phase2-repetition-penalty",
    },
    {
        "name": "current",
        "adapter_path": f"{_EMO_CKPT_DIR}/step_200",
    },
]


@app.function(
    gpu="A10G",
    image=_image,
    volumes={
        _MODEL_CACHE_DIR: _model_volume,
        _EMO_CKPT_DIR:    _emo_volume,
        _PHASE2_DIR:      _phase2_volume,
    },
    timeout=600,
)
def run_checkpoint(checkpoint: dict, phrases: list[str], num_turns: int) -> list[dict]:
    """Load one checkpoint and generate emoji for all phrases."""
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
    from peft import PeftModel

    from src.rl.custom.generate import MODEL_NAME, build_emoji_mask, build_system_prompt, format_prompt

    os.environ["HF_HOME"] = _MODEL_CACHE_DIR

    class EmojiLogitsProcessor(LogitsProcessor):
        def __init__(self, emoji_mask):
            self.emoji_mask = emoji_mask

        def __call__(self, input_ids, scores):
            mask = self.emoji_mask.to(scores.device)
            vocab_size = scores.shape[-1]
            if mask.shape[0] < vocab_size:
                ext = torch.zeros(vocab_size - mask.shape[0], dtype=torch.bool, device=scores.device)
                mask = torch.cat([mask, ext])
            elif mask.shape[0] > vocab_size:
                mask = mask[:vocab_size]
            return scores.masked_fill(~mask, float("-inf"))

    name = checkpoint["name"]
    adapter_path = checkpoint["adapter_path"]

    print(f"[{name}] Loading base model {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    if adapter_path:
        print(f"[{name}] Attaching adapter from {adapter_path} ...")
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        print(f"[{name}] Running base model (no adapter)")
        model = base

    model.eval()
    emoji_mask = build_emoji_mask(tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits_processor = [EmojiLogitsProcessor(emoji_mask)]

    def generate_turn(phrase: str, history: list[dict]) -> str:
        sys_prompt = build_system_prompt(phrase)
        if not history:
            prompt = format_prompt(phrase, tokenizer, sys_prompt)
        else:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": phrase},
            ]
            for h in history:
                messages.append({"role": "assistant", "content": h["emoji"]})
                messages.append({
                    "role": "user",
                    "content": (
                        f'The player guessed: "{h["guess"]}". That\'s wrong. '
                        f'The correct phrase is still: "{phrase}". '
                        "Send more emoji to help them."
                    ),
                })
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=1.0,
                logits_processor=logits_processor,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = out[0, prompt_len:].tolist()
        if tokenizer.eos_token_id in new_ids:
            new_ids = new_ids[:new_ids.index(tokenizer.eos_token_id)]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    results = []
    for phrase in phrases:
        print(f"[{name}] {phrase}")
        history: list[dict] = []
        turns = []
        for _ in range(num_turns):
            emoji = generate_turn(phrase, history)
            turns.append(emoji)
            history.append({"emoji": emoji, "guess": "???"})
        results.append({"phrase": phrase, "turns": turns})

    return results


@app.local_entrypoint()
def main(turns: int = 3):
    # Run all three checkpoints in parallel
    futures = {
        ckpt["name"]: run_checkpoint.spawn(ckpt, PHRASES, turns)
        for ckpt in CHECKPOINTS
    }

    all_results = {name: f.get() for name, f in futures.items()}

    # Build comparison table
    col_order = ["base", "phase2", "current"]
    header = "| Phrase | Base | Phase 2 RL | Current RL |"
    sep    = "|--------|------|------------|------------|"
    rows = [header, sep]

    # Index results by phrase
    indexed = {
        name: {r["phrase"]: r["turns"] for r in results}
        for name, results in all_results.items()
    }

    for phrase in PHRASES:
        cells = []
        for col in col_order:
            turns_list = indexed.get(col, {}).get(phrase, [])
            cells.append(" → ".join(turns_list) if turns_list else "❌")
        rows.append(f"| {phrase} | {cells[0]} | {cells[1]} | {cells[2]} |")

    md = "\n".join(rows)
    print("\n" + md)

    out_path = "scripts/sweep_all_results.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md + "\n")

    jsonl_path = "scripts/sweep_all_results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for name, results in all_results.items():
            for r in results:
                f.write(json.dumps({"checkpoint": name, **r}, ensure_ascii=False) + "\n")

    print(f"\nWrote {out_path} and {jsonl_path}")
