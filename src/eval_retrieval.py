"""Evaluate emoji models on the retrieval game.

Runs the full sender → judge → embedding pipeline on a held-out prompt set.
Supports three modes:
  - base:  prompted base model (no fine-tuning)
  - sft:   SFT checkpoint
  - rl:    SFT+RL checkpoint

Reports per-example and per-category metrics, plus aggregate comparison numbers.

Usage:
    uv run python -m src.eval_retrieval --mode base
    uv run python -m src.eval_retrieval --mode sft --checkpoint <tinker-state>
    uv run python -m src.eval_retrieval --mode rl --checkpoint <tinker-state>
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import regex
import tinker
from dotenv import load_dotenv
from tqdm import tqdm

from src.rl.judge import JudgeClient
from src.rl.reward import RewardEmbedder

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You communicate exclusively using emoji. No text, numbers, or punctuation ever. "
    "Use 2-8 emoji per response that capture the core meaning, emotion, and key concepts "
    "of the user's message."
)

# Categories for per-category analysis. Assign based on prompt characteristics.
CATEGORY_KEYWORDS = {
    "greeting": ["hello", "hi ", "hey ", "good morning", "good evening", "good night", "how are you", "how's your day"],
    "emotion": ["happy", "sad", "angry", "stressed", "nervous", "excited", "scared", "love", "hate", "afraid", "anxious", "proud", "lonely", "grateful"],
    "weather": ["weather", "rain", "snow", "sunny", "cloud", "storm", "wind", "temperature", "cold", "hot", "warm"],
    "food": ["food", "eat", "cook", "restaurant", "pizza", "hungry", "dinner", "lunch", "breakfast", "recipe", "meal"],
    "travel": ["travel", "flight", "airport", "hotel", "vacation", "trip", "moving", "city", "country"],
    "work": ["work", "job", "promoted", "office", "meeting", "boss", "career", "hired", "fired", "interview"],
    "animal": ["dog", "cat", "pet", "animal", "puppy", "kitten", "bird", "fish"],
    "nature": ["sunset", "ocean", "mountain", "forest", "river", "beach", "garden", "flower", "tree", "star"],
    "abstract": ["freedom", "meaning", "life", "death", "time", "truth", "justice", "love", "hope", "dream"],
    "question": ["what", "how", "why", "where", "when", "who", "can you", "tell me", "recommend"],
}


def categorize_prompt(text: str) -> str:
    """Assign a category based on keyword matching. Returns 'other' if no match."""
    text_lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return "other"


def validate_emoji_only(text: str) -> bool:
    """Check that text contains only emoji characters."""
    cleaned = (
        text.replace(" ", "")
        .replace("\u200d", "")
        .replace("\ufe0f", "")
        .replace("\ufe0e", "")
    )
    if not cleaned:
        return False
    for char in cleaned:
        if not regex.match(r"\p{Emoji}", char):
            if not regex.match(r"[\U0001F3FB-\U0001F3FF]", char):
                return False
    return True


def count_emoji(text: str) -> int:
    """Count the number of emoji in a string."""
    return len([c for c in text if regex.match(r"\p{Emoji}", c) and c not in "\ufe0f\ufe0e\u200d"])


def filter_to_emoji(tokens: list[int], tokenizer, emoji_token_ids: set[int]) -> str:
    """Filter token sequence to emoji-only tokens and decode, trimming broken Unicode."""
    filtered = [t for t in tokens if t in emoji_token_ids]
    if not filtered:
        return ""
    # Trim trailing tokens that form incomplete Unicode sequences
    while filtered:
        decoded = tokenizer.decode(filtered, skip_special_tokens=True)
        if "\ufffd" not in decoded:
            return decoded.strip()
        filtered.pop()
    return ""


def generate_response(
    sampling_client: tinker.SamplingClient,
    tokenizer,
    user_message: str,
    sampling_params: tinker.SamplingParams,
    emoji_token_ids: set[int] | None = None,
) -> str:
    """Generate a single emoji response from the sender model.

    If emoji_token_ids is provided, post-hoc filters output to emoji-only tokens.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    tokens = tokenizer.encode(text, add_special_tokens=False)
    model_input = tinker.ModelInput.from_ints(tokens)

    response = sampling_client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    ).result()

    raw_tokens = response.sequences[0].tokens

    if emoji_token_ids is not None:
        return filter_to_emoji(raw_tokens, tokenizer, emoji_token_ids)

    return tokenizer.decode(raw_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate emoji model on retrieval game")
    parser.add_argument("--mode", type=str, required=True, choices=["base", "sft", "rl"],
                        help="Evaluation mode: base (prompted), sft, or rl checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Tinker state path (required for sft/rl modes)")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--prompts", type=str, default="data/rl_prompts.jsonl",
                        help="Eval prompts (default: rl_prompts.jsonl)")
    parser.add_argument("--limit", type=int, default=200,
                        help="Number of prompts to evaluate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--emoji-filter", action="store_true",
                        help="Apply post-hoc emoji token filter (matches RL training setup)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: runs/<mode>)")
    args = parser.parse_args()

    if args.mode in ("sft", "rl") and not args.checkpoint:
        parser.error(f"--checkpoint required for mode={args.mode}")

    output_dir = Path(args.output_dir or f"runs/{args.mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Setup sender model ---
    service = tinker.ServiceClient()

    if args.mode == "base":
        logger.info("Loading base model (no fine-tuning): %s", args.base_model)
        sampling_client = service.create_sampling_client(base_model=args.base_model)
    else:
        logger.info("Loading checkpoint: %s", args.checkpoint)
        training_client = service.create_training_client_from_state(args.checkpoint)
        sampling_client = training_client.save_weights_and_get_sampling_client(
            name=f"eval-{args.mode}"
        )

    tokenizer = sampling_client.get_tokenizer()
    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # --- Load emoji filter if requested ---
    emoji_token_ids = None
    if args.emoji_filter:
        mask = np.load("data/emoji_mask.npy")
        emoji_token_ids = set(np.where(mask)[0].tolist())
        logger.info("Emoji filter enabled: %d allowed token IDs", len(emoji_token_ids))

    # --- Setup judge + embedder ---
    logger.info("Setting up judge model: %s", args.judge_model)
    judge = JudgeClient.create(service, judge_model=args.judge_model)
    embedder = RewardEmbedder()

    # --- Load prompts ---
    with open(args.prompts) as f:
        all_prompts = [json.loads(line) for line in f]

    # Sample evenly across difficulty levels
    prompts = all_prompts[: args.limit]
    logger.info("Evaluating %d prompts", len(prompts))

    # --- Run evaluation ---
    results = []
    start_time = time.time()

    for item in tqdm(prompts, desc=f"Eval ({args.mode})"):
        target = item["text"]
        difficulty = item.get("difficulty", "unknown")
        category = categorize_prompt(target)

        # Generate emoji response
        emoji_response = generate_response(
            sampling_client, tokenizer, target, sampling_params,
            emoji_token_ids=emoji_token_ids,
        )

        # Check format
        is_emoji_only = validate_emoji_only(emoji_response)
        n_emoji = count_emoji(emoji_response)

        # Judge reconstructs from emoji (single turn)
        judge_guess = judge.reconstruct(emoji_history=[emoji_response])

        # Compute similarity
        similarity = embedder.similarity(target, judge_guess)
        success = similarity >= 0.7

        result = {
            "target": target,
            "emoji": emoji_response,
            "judge_guess": judge_guess,
            "similarity": round(similarity, 4),
            "success": success,
            "emoji_only": is_emoji_only,
            "emoji_count": n_emoji,
            "category": category,
            "difficulty": difficulty,
        }
        results.append(result)

    elapsed = time.time() - start_time

    # --- Compute aggregates ---
    similarities = [r["similarity"] for r in results]
    successes = [r["success"] for r in results]
    emoji_only_count = sum(r["emoji_only"] for r in results)
    emoji_counts = [r["emoji_count"] for r in results]

    # Per-category breakdown
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    category_stats = {}
    for cat, cat_results in sorted(by_category.items()):
        cat_sims = [r["similarity"] for r in cat_results]
        cat_successes = [r["success"] for r in cat_results]
        category_stats[cat] = {
            "count": len(cat_results),
            "similarity_mean": round(float(np.mean(cat_sims)), 4),
            "similarity_std": round(float(np.std(cat_sims)), 4),
            "success_rate": round(float(np.mean(cat_successes)), 4),
        }

    # Per-difficulty breakdown
    by_difficulty = defaultdict(list)
    for r in results:
        by_difficulty[r["difficulty"]].append(r)

    difficulty_stats = {}
    for diff, diff_results in sorted(by_difficulty.items()):
        diff_sims = [r["similarity"] for r in diff_results]
        difficulty_stats[diff] = {
            "count": len(diff_results),
            "similarity_mean": round(float(np.mean(diff_sims)), 4),
            "success_rate": round(float(np.mean([r["success"] for r in diff_results])), 4),
        }

    summary = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "emoji_filter": args.emoji_filter,
        "n_examples": len(results),
        "time_s": round(elapsed, 1),
        "format": {
            "emoji_only_pct": round(emoji_only_count / len(results) * 100, 1),
            "avg_emoji_count": round(float(np.mean(emoji_counts)), 1),
        },
        "retrieval": {
            "similarity_mean": round(float(np.mean(similarities)), 4),
            "similarity_std": round(float(np.std(similarities)), 4),
            "similarity_median": round(float(np.median(similarities)), 4),
            "success_rate": round(float(np.mean(successes)) * 100, 1),
        },
        "by_category": category_stats,
        "by_difficulty": difficulty_stats,
    }

    # --- Print report ---
    print("\n" + "=" * 60)
    print(f"RETRIEVAL EVAL — {args.mode.upper()}")
    print("=" * 60)
    print(f"Examples:        {len(results)}")
    print(f"Emoji-only:      {emoji_only_count}/{len(results)} ({summary['format']['emoji_only_pct']}%)")
    print(f"Avg emoji/resp:  {summary['format']['avg_emoji_count']}")
    print(f"Similarity:      {summary['retrieval']['similarity_mean']:.3f} ± {summary['retrieval']['similarity_std']:.3f}")
    print(f"Success (≥0.7):  {summary['retrieval']['success_rate']:.1f}%")
    print(f"Time:            {elapsed:.0f}s")

    print("\n--- By Category ---")
    for cat, stats in sorted(category_stats.items(), key=lambda x: -x[1]["similarity_mean"]):
        print(f"  {cat:12s}  n={stats['count']:3d}  sim={stats['similarity_mean']:.3f}  success={stats['success_rate']:.0%}")

    print("\n--- By Difficulty ---")
    for diff, stats in sorted(difficulty_stats.items()):
        print(f"  {diff:8s}  n={stats['count']:3d}  sim={stats['similarity_mean']:.3f}  success={stats['success_rate']:.0%}")

    # Show best and worst examples
    sorted_by_sim = sorted(results, key=lambda r: r["similarity"], reverse=True)
    print("\n--- Top 5 (highest similarity) ---")
    for r in sorted_by_sim[:5]:
        print(f"  [{r['similarity']:.3f}] {r['target'][:60]}")
        print(f"          emoji: {r['emoji']}")
        print(f"          judge: {r['judge_guess']}")

    print("\n--- Bottom 5 (lowest similarity) ---")
    for r in sorted_by_sim[-5:]:
        print(f"  [{r['similarity']:.3f}] {r['target'][:60]}")
        print(f"          emoji: {r['emoji']}")
        print(f"          judge: {r['judge_guess']}")

    # --- Save ---
    results_path = output_dir / "retrieval_results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path = output_dir / "retrieval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Results saved to %s", results_path)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
