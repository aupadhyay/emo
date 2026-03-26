"""Evaluate the SFT-trained emoji model on Tinker.

Runs the full test set through the model and reports quantitative metrics:
- Emoji-only compliance rate (no text leaking)
- Emoji count distribution
- Response length stats

Also prints qualitative samples for eyeballing.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import regex
import tinker
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

SYSTEM_PROMPT = (
    "You communicate exclusively using emoji. No text, numbers, or punctuation ever. "
    "Use 2-8 emoji per response that capture the core meaning, emotion, and key concepts "
    "of the user's message."
)

SAMPLE_PROMPTS = [
    "I just got promoted at work!",
    "My dog passed away today",
    "What's the weather like outside?",
    "Can you recommend a good Italian restaurant?",
    "I'm moving to a new city and I'm nervous",
    "Happy birthday!",
    "The sunset over the ocean was beautiful",
    "I'm so hungry I could eat a horse",
    "What's the tallest mountain in the world?",
    "I think pizza is better than pasta",
    "Good morning! How's your day going?",
    "What does freedom mean to you?",
    "My flight got cancelled and I'm stuck at the airport overnight",
    "I just finished running a marathon",
    "Tell me a joke",
    "I'm stressed about my final exams",
]


def validate_emoji_only(text: str) -> bool:
    """Check that text contains only emoji characters (no text, numbers, punctuation)."""
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


def generate_response(
    sampling_client: tinker.SamplingClient,
    tokenizer,
    user_message: str,
    sampling_params: tinker.SamplingParams,
) -> str:
    """Generate a single response from the model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    tokens = tokenizer.encode(text, add_special_tokens=False)
    model_input = tinker.ModelInput.from_ints(tokens)

    response = sampling_client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    ).result()

    return tokenizer.decode(response.sequences[0].tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT emoji model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="tinker://51acb88b-35a0-5f3b-a102-a4b5d5643714:train:0/weights/final",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--test-data", type=str, default="data/sft_test.jsonl")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit test examples (default: all)")
    parser.add_argument("--output", type=str, default="runs/sft/eval_results.jsonl")
    args = parser.parse_args()

    service_client = tinker.ServiceClient()
    print(f"Loading checkpoint: {args.checkpoint}")

    training_client = service_client.create_training_client_from_state(args.checkpoint)
    sampling_client = training_client.save_weights_and_get_sampling_client(name="eval")
    tokenizer = sampling_client.get_tokenizer()

    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    results = []

    # --- Qualitative: fixed prompts ---
    print("\n=== Qualitative samples ===\n")
    for prompt in SAMPLE_PROMPTS:
        decoded = generate_response(sampling_client, tokenizer, prompt, sampling_params)
        print(f"  Q: {prompt}")
        print(f"  A: {decoded}")
        print()
        results.append({"prompt": prompt, "response": decoded, "source": "fixed"})

    # --- Quantitative: full test set ---
    print("\n=== Running full test set eval ===\n")

    with open(args.test_data) as f:
        test_lines = f.readlines()
    if args.test_limit:
        test_lines = test_lines[: args.test_limit]

    n_total = len(test_lines)
    n_emoji_only = 0
    n_empty = 0
    emoji_counts = []
    failures = []

    for line in tqdm(test_lines, desc="Evaluating"):
        row = json.loads(line)
        user_msg = row["messages"][1]["content"]
        expected = row["messages"][2]["content"]

        decoded = generate_response(sampling_client, tokenizer, user_msg, sampling_params)

        is_valid = validate_emoji_only(decoded)
        n_emoji = count_emoji(decoded)

        if not decoded:
            n_empty += 1
        elif is_valid:
            n_emoji_only += 1
        else:
            failures.append({"prompt": user_msg, "response": decoded, "expected": expected})

        emoji_counts.append(n_emoji)
        results.append({
            "prompt": user_msg,
            "response": decoded,
            "expected": expected,
            "emoji_only": is_valid,
            "emoji_count": n_emoji,
            "source": "test",
        })

    # --- Print report ---
    print("\n" + "=" * 50)
    print("EVAL RESULTS")
    print("=" * 50)
    print(f"Total test examples:    {n_total}")
    print(f"Emoji-only responses:   {n_emoji_only} ({n_emoji_only / n_total * 100:.1f}%)")
    print(f"Empty responses:        {n_empty} ({n_empty / n_total * 100:.1f}%)")
    print(f"Text leakage:           {len(failures)} ({len(failures) / n_total * 100:.1f}%)")
    print()

    # Emoji count distribution
    count_dist = Counter(emoji_counts)
    print("Emoji count distribution:")
    for k in sorted(count_dist.keys()):
        bar = "#" * min(count_dist[k] // max(1, n_total // 50), 50)
        print(f"  {k:2d} emoji: {count_dist[k]:5d} {bar}")
    print()

    avg_emoji = sum(emoji_counts) / len(emoji_counts) if emoji_counts else 0
    in_range = sum(1 for c in emoji_counts if 2 <= c <= 8)
    print(f"Avg emoji per response: {avg_emoji:.1f}")
    print(f"In target range (2-8):  {in_range} ({in_range / n_total * 100:.1f}%)")
    print()

    # Show some failures
    if failures:
        print(f"Sample failures (showing up to 10):")
        for f in failures[:10]:
            print(f"  Q: {f['prompt']}")
            print(f"  A: {f['response']}")
            print()

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save summary
    summary = {
        "total": n_total,
        "emoji_only": n_emoji_only,
        "emoji_only_pct": round(n_emoji_only / n_total * 100, 1),
        "empty": n_empty,
        "text_leakage": len(failures),
        "avg_emoji_count": round(avg_emoji, 1),
        "in_target_range_pct": round(in_range / n_total * 100, 1),
        "emoji_count_distribution": {str(k): v for k, v in sorted(count_dist.items())},
    }
    with open(output_path.parent / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {output_path}")
    print(f"Summary saved to {output_path.parent / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
