"""Filter prompts by emoji-communicability using the SFT model + judge.

For each candidate message, run it through the full pipeline:
  SFT model encodes to emoji → judge decodes → compute similarity
If similarity > threshold, the message is emoji-communicable and kept.

Usage:
    uv run python -m src.data.filter_rl_prompts
    uv run python -m src.data.filter_rl_prompts --threshold 0.4 --input data/prompts.jsonl
"""

import argparse
import asyncio
import json
import logging

import tinker
from dotenv import load_dotenv
from tinker_cookbook import renderers, tokenizer_utils

from src.rl.emoji_completer import EmojiTokenCompleter
from src.rl.judge import JudgeClient
from src.rl.reward import RewardEmbedder

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SENDER_SYSTEM_PROMPT = (
    "You communicate exclusively using emoji. No text, numbers, or punctuation ever. "
    "Respond with 2-8 emoji that capture the core meaning of what you need to communicate."
)


async def test_one(
    target: str,
    policy: EmojiTokenCompleter,
    judge: JudgeClient,
    embedder: RewardEmbedder,
    renderer: renderers.Renderer,
    tokenizer,
) -> dict:
    """Run one round-trip: sender encodes, judge decodes, measure similarity."""
    messages = [
        {"role": "system", "content": SENDER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Communicate this message using only emoji: {target}"},
    ]
    prompt = renderer.build_generation_prompt(messages)
    stop = renderer.get_stop_sequences()

    result = await policy(prompt, stop)
    emoji = tokenizer.decode(result.tokens, skip_special_tokens=True).strip()

    guess = await judge.reconstruct_async([emoji])
    similarity = embedder.similarity(target, guess)

    return {
        "text": target,
        "emoji": emoji,
        "guess": guess,
        "similarity": similarity,
    }


async def main(args):
    service = tinker.ServiceClient()

    # Load SFT model
    tc = await service.create_lora_training_client_async(
        base_model="Qwen/Qwen3.5-4B", rank=32,
    )
    tc.load_state(args.checkpoint)
    sender_sc = tc.save_weights_and_get_sampling_client(name="filter-sender")
    tokenizer = tc.get_tokenizer()
    renderer = renderers.get_renderer("qwen3_disable_thinking", tokenizer)

    policy = EmojiTokenCompleter(
        sampling_client=sender_sc, max_tokens=20, temperature=0.7,
    )
    judge = JudgeClient.create(service, judge_model=args.judge_model)
    embedder = RewardEmbedder()

    # Load candidate prompts
    prompts = []
    with open(args.input) as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text") or row.get("prompt", "")
            text = text.strip()
            if 5 < len(text) < 80:  # pre-filter by length
                prompts.append(text)

    # Deduplicate
    prompts = list(set(prompts))
    logger.info("Loaded %d candidate prompts", len(prompts))

    # Test each prompt in batches
    kept = []
    dropped = []
    batch_size = 8  # concurrent requests

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        tasks = [
            test_one(t, policy, judge, embedder, renderer, tokenizer)
            for t in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                logger.warning("Failed: %s", r)
                continue
            if r["similarity"] >= args.threshold:
                kept.append(r)
            else:
                dropped.append(r)

        n_done = min(i + batch_size, len(prompts))
        if n_done % 50 < batch_size:
            logger.info(
                "Progress: %d/%d tested, %d kept (%.0f%%)",
                n_done, len(prompts), len(kept),
                100 * len(kept) / max(n_done, 1),
            )

    # Write filtered prompts
    with open(args.output, "w") as f:
        for r in kept:
            f.write(json.dumps({
                "text": r["text"],
                "difficulty": "easy",
                "filter_similarity": round(r["similarity"], 3),
            }) + "\n")

    logger.info("Done! Kept %d / %d (%.0f%%)", len(kept), len(prompts),
                100 * len(kept) / max(len(prompts), 1))
    logger.info("Output: %s", args.output)

    # Show samples
    kept_sorted = sorted(kept, key=lambda x: -x["similarity"])
    logger.info("\nTop 10 (easiest):")
    for r in kept_sorted[:10]:
        logger.info("  sim=%.3f  %s → %s → %s", r["similarity"], r["text"], r["emoji"], r["guess"])

    logger.info("\nBottom 10 (hardest kept):")
    for r in kept_sorted[-10:]:
        logger.info("  sim=%.3f  %s → %s → %s", r["similarity"], r["text"], r["emoji"], r["guess"])

    if dropped:
        dropped_sorted = sorted(dropped, key=lambda x: -x["similarity"])
        logger.info("\nTop 10 dropped (closest to threshold):")
        for r in dropped_sorted[:10]:
            logger.info("  sim=%.3f  %s → %s → %s", r["similarity"], r["text"], r["emoji"], r["guess"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/prompts.jsonl")
    parser.add_argument("--output", default="data/rl_prompts.jsonl")
    parser.add_argument("--checkpoint", default="final")
    parser.add_argument("--judge-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()
    asyncio.run(main(args))
