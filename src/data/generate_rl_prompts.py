"""Curate target messages for the RL retrieval game.

Reuses the existing prompts.jsonl from the SFT data pipeline (which already has
30K diverse natural language messages). Filters and reformats for RL use.

We can also pull from HuggingFace datasets for additional coverage.

Usage:
    uv run python -m src.data.generate_rl_prompts
    uv run python -m src.data.generate_rl_prompts --source existing --target-count 8000
"""

import argparse
import json
import random


def build_from_existing(
    input_path: str = "data/prompts.jsonl",
    output_path: str = "data/rl_prompts.jsonl",
    target_count: int = 8000,
    seed: int = 42,
) -> None:
    """Build RL prompts from the existing SFT prompt pool.

    The SFT prompts.jsonl already contains 30K diverse natural language messages
    across 9 categories. We sample a subset and tag by difficulty.
    """
    prompts = []
    with open(input_path) as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text") or row.get("prompt", "")
            text = text.strip()
            # Filter: reasonable length, not too short
            if 10 < len(text) < 300:
                prompts.append(text)

    # Deduplicate
    prompts = list(set(prompts))
    random.seed(seed)
    random.shuffle(prompts)

    # Trim to target
    prompts = prompts[:target_count]

    with open(output_path, "w") as f:
        for p in prompts:
            difficulty = "easy" if len(p) < 50 else "medium" if len(p) < 150 else "hard"
            f.write(json.dumps({"text": p, "difficulty": difficulty}) + "\n")

    print(f"Wrote {len(prompts)} RL prompts to {output_path}")

    # Stats
    easy = sum(1 for p in prompts if len(p) < 50)
    medium = sum(1 for p in prompts if 50 <= len(p) < 150)
    hard = sum(1 for p in prompts if len(p) >= 150)
    print(f"  easy: {easy}, medium: {medium}, hard: {hard}")


def build_from_huggingface(
    output_path: str = "data/rl_prompts.jsonl",
    target_count: int = 8000,
    seed: int = 42,
) -> None:
    """Build RL prompts from HuggingFace instruction datasets."""
    from datasets import load_dataset

    prompts = []

    # Source 1: Alpaca instructions
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        for row in ds:
            instruction = row["instruction"].strip()
            inp = row.get("input", "").strip()
            msg = f"{instruction} {inp}".strip() if inp else instruction
            if 10 < len(msg) < 300:
                prompts.append(msg)
        print(f"  Loaded {len(prompts)} from Alpaca")
    except Exception as e:
        print(f"  Alpaca failed: {e}")

    # Source 2: Dolly
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        count = 0
        for row in ds:
            msg = row["instruction"].strip()
            if 10 < len(msg) < 300 and not any(c in msg for c in ["{", "}", "```", "def ", "import "]):
                prompts.append(msg)
                count += 1
        print(f"  Loaded {count} from Dolly")
    except Exception as e:
        print(f"  Dolly failed: {e}")

    # Deduplicate and shuffle
    prompts = list(set(prompts))
    random.seed(seed)
    random.shuffle(prompts)
    prompts = prompts[:target_count]

    with open(output_path, "w") as f:
        for p in prompts:
            difficulty = "easy" if len(p) < 50 else "medium" if len(p) < 150 else "hard"
            f.write(json.dumps({"text": p, "difficulty": difficulty}) + "\n")

    print(f"Wrote {len(prompts)} RL prompts to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate RL prompt dataset")
    parser.add_argument("--source", default="existing", choices=["existing", "huggingface"])
    parser.add_argument("--input-path", default="data/prompts.jsonl")
    parser.add_argument("--output-path", default="data/rl_prompts.jsonl")
    parser.add_argument("--target-count", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.source == "existing":
        build_from_existing(args.input_path, args.output_path, args.target_count, args.seed)
    else:
        build_from_huggingface(args.output_path, args.target_count, args.seed)


if __name__ == "__main__":
    main()
