"""Generate SFT dataset: emoji-only responses for natural language prompts.

Uses the Anthropic Message Batches API for efficient bulk processing.
Workflow: submit → status → collect → filter → split
"""

import argparse
import json
import random
import time
from pathlib import Path

import anthropic
import regex
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are an emoji-only communicator. You must respond using ONLY emoji characters.

Rules:
- No text, letters, numbers, or punctuation whatsoever
- Use 2-8 emoji per response
- Capture the core meaning, emotion, and key concepts
- Prefer concrete, widely-understood emoji over obscure ones
- Use emoji sequences that a human could reasonably decode back to the original meaning
- For questions, respond with emoji that answer the question (not emoji that restate the question)

Examples:
User: "What's the weather like?" → 🌧️😔 (if rainy and sad)
User: "I just adopted a puppy!" → 🐶🎉❤️
User: "Can you recommend a good Italian restaurant?" → 🇮🇹🍝👨‍🍳👌
User: "I'm stressed about my final exams" → 😰📚📝💪"""

# Shorter system prompt for training data (what the model sees at inference)
TRAINING_SYSTEM_PROMPT = (
    "You communicate exclusively using emoji. No text, numbers, or punctuation ever. "
    "Use 2-8 emoji per response that capture the core meaning, emotion, and key concepts "
    "of the user's message."
)


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


def submit_batch(
    prompts_path: str = "data/prompts.jsonl",
    batch_id_path: str = "data/batch_id.txt",
    limit: int | None = None,
) -> None:
    """Submit all prompts as a Message Batch to the Anthropic API."""
    client = anthropic.Anthropic()

    prompts = []
    with open(prompts_path) as f:
        for line in f:
            prompts.append(json.loads(line)["text"])
    if limit:
        prompts = prompts[:limit]

    print(f"Preparing batch of {len(prompts)} requests...")

    requests = [
        {
            "custom_id": f"prompt-{i}",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 50,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
        }
        for i, prompt in enumerate(prompts)
    ]

    # Also save the prompt mapping for later collection
    mapping_path = Path(prompts_path).parent / "batch_prompt_mapping.jsonl"
    with open(mapping_path, "w") as f:
        for i, prompt in enumerate(prompts):
            f.write(json.dumps({"custom_id": f"prompt-{i}", "text": prompt}) + "\n")

    print(f"Submitting batch...")
    batch = client.messages.batches.create(requests=requests)

    Path(batch_id_path).parent.mkdir(parents=True, exist_ok=True)
    with open(batch_id_path, "w") as f:
        f.write(batch.id)

    print(f"Batch submitted!")
    print(f"  Batch ID: {batch.id}")
    print(f"  Status: {batch.processing_status}")
    print(f"  Saved batch ID to {batch_id_path}")
    print(f"\nRun 'python {__file__} status' to check progress")


def check_status(batch_id_path: str = "data/batch_id.txt", poll: bool = False) -> None:
    """Check the status of a submitted batch."""
    client = anthropic.Anthropic()

    with open(batch_id_path) as f:
        batch_id = f.read().strip()

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts

        print(f"Batch {batch_id}")
        print(f"  Status: {batch.processing_status}")
        print(f"  Processing: {counts.processing}")
        print(f"  Succeeded: {counts.succeeded}")
        print(f"  Errored: {counts.errored}")
        print(f"  Canceled: {counts.canceled}")
        print(f"  Expired: {counts.expired}")

        if batch.processing_status == "ended":
            print(f"\nBatch complete! Run 'python {__file__} collect' to retrieve results")
            break

        if not poll:
            break

        print(f"\n  Polling again in 30s...")
        time.sleep(30)


def collect_results(
    batch_id_path: str = "data/batch_id.txt",
    output_path: str = "data/sft_dataset.jsonl",
) -> None:
    """Collect results from a completed batch and write the SFT dataset."""
    client = anthropic.Anthropic()

    with open(batch_id_path) as f:
        batch_id = f.read().strip()

    # Load prompt mapping
    mapping_path = Path(batch_id_path).parent / "batch_prompt_mapping.jsonl"
    prompt_map = {}
    with open(mapping_path) as f:
        for line in f:
            row = json.loads(line)
            prompt_map[row["custom_id"]] = row["text"]

    print(f"Collecting results for batch {batch_id}...")

    succeeded = 0
    failed = 0
    invalid_emoji = 0
    results = []

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        user_message = prompt_map[custom_id]

        if result.result.type != "succeeded":
            failed += 1
            continue

        emoji_response = result.result.message.content[0].text.strip()

        if not validate_emoji_only(emoji_response):
            invalid_emoji += 1
            continue

        results.append({
            "messages": [
                {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": emoji_response},
            ]
        })
        succeeded += 1

    # Write output
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total = succeeded + failed + invalid_emoji
    print(f"\nResults: {total} total")
    print(f"  Succeeded (valid emoji): {succeeded}")
    print(f"  Invalid emoji (filtered): {invalid_emoji}")
    print(f"  API failures: {failed}")
    print(f"  Success rate: {succeeded/total*100:.1f}%")
    print(f"  Output: {output_path}")


def filter_dataset(input_path: str, output_path: str) -> None:
    """Filter out low-quality examples."""
    kept, dropped = 0, 0
    drop_reasons: dict[str, int] = {}

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            emoji_response = row["messages"][2]["content"]
            user_msg = row["messages"][1]["content"]

            if not validate_emoji_only(emoji_response):
                drop_reasons["invalid_emoji"] = drop_reasons.get("invalid_emoji", 0) + 1
                dropped += 1
                continue

            n_emoji = count_emoji(emoji_response)
            if n_emoji < 2 or n_emoji > 15:
                drop_reasons["emoji_count"] = drop_reasons.get("emoji_count", 0) + 1
                dropped += 1
                continue

            if len(user_msg) < 5 or len(user_msg) > 500:
                drop_reasons["msg_length"] = drop_reasons.get("msg_length", 0) + 1
                dropped += 1
                continue

            fout.write(line)
            kept += 1

    print(f"Kept: {kept}, Dropped: {dropped}")
    for reason, count in sorted(drop_reasons.items()):
        print(f"  {reason}: {count}")


def split_dataset(
    input_path: str,
    train_path: str,
    test_path: str,
    test_ratio: float = 0.1,
) -> None:
    """Split into train and test sets."""
    with open(input_path) as f:
        lines = f.readlines()
    random.shuffle(lines)

    split_idx = int(len(lines) * (1 - test_ratio))

    with open(train_path, "w") as f:
        f.writelines(lines[:split_idx])
    with open(test_path, "w") as f:
        f.writelines(lines[split_idx:])

    print(f"Train: {split_idx}, Test: {len(lines) - split_idx}")


def main():
    parser = argparse.ArgumentParser(description="Generate SFT emoji dataset")
    sub = parser.add_subparsers(dest="command", required=True)

    # Submit batch
    sm = sub.add_parser("submit", help="Submit prompts as a batch to Claude API")
    sm.add_argument("--input", type=str, default="data/prompts.jsonl")
    sm.add_argument("--limit", type=int, default=None)

    # Check status
    st = sub.add_parser("status", help="Check batch status")
    st.add_argument("--poll", action="store_true", help="Poll until complete")

    # Collect results
    co = sub.add_parser("collect", help="Collect batch results into dataset")
    co.add_argument("--output", type=str, default="data/sft_dataset.jsonl")

    # Filter
    filt = sub.add_parser("filter", help="Filter dataset")
    filt.add_argument("--input", type=str, default="data/sft_dataset.jsonl")
    filt.add_argument("--output", type=str, default="data/sft_dataset_filtered.jsonl")

    # Split
    sp = sub.add_parser("split", help="Train/test split")
    sp.add_argument("--input", type=str, default="data/sft_dataset_filtered.jsonl")
    sp.add_argument("--train", type=str, default="data/sft_train.jsonl")
    sp.add_argument("--test", type=str, default="data/sft_test.jsonl")
    sp.add_argument("--test-ratio", type=float, default=0.1)

    args = parser.parse_args()

    if args.command == "submit":
        submit_batch(prompts_path=args.input, limit=args.limit)
    elif args.command == "status":
        check_status(poll=args.poll)
    elif args.command == "collect":
        collect_results(output_path=args.output)
    elif args.command == "filter":
        filter_dataset(args.input, args.output)
    elif args.command == "split":
        split_dataset(args.input, args.train, args.test, args.test_ratio)


if __name__ == "__main__":
    main()
