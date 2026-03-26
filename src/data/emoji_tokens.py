"""Build a set of token IDs needed to encode any emoji in the Qwen3.5-4B tokenizer.

Uses the official Unicode emoji-test.txt as the source of all valid emoji sequences.
Tokenizes every sequence and collects the union of all resulting token IDs.

The resulting mask is used for logit masking during generation and RL training.
Only needs the tokenizer (no model weights).
"""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

EMOJI_TEST_URL = "https://unicode.org/Public/emoji/latest/emoji-test.txt"


def load_emoji_sequences(cache_path: str = "data/emoji-test.txt") -> list[str]:
    """Download and parse all emoji sequences from Unicode emoji-test.txt."""
    cache = Path(cache_path)
    if cache.exists():
        text = cache.read_text()
    else:
        print(f"Downloading {EMOJI_TEST_URL}...")
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(EMOJI_TEST_URL, cache)
        text = cache.read_text()

    sequences = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: "1F600 ; fully-qualified # 😀 E1.0 grinning face"
        codepoints_str, rest = line.split(";", 1)
        status = rest.strip().split()[0]
        # Include fully-qualified, minimally-qualified, unqualified, and component
        codepoints = codepoints_str.strip().split()
        emoji = "".join(chr(int(cp, 16)) for cp in codepoints)
        sequences.append(emoji)

    return sequences


def build_emoji_mask(
    model_name: str = "Qwen/Qwen3.5-4B",
    output_dir: str = "data",
) -> np.ndarray:
    """Tokenize every known emoji sequence, collect all token IDs produced."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size

    sequences = load_emoji_sequences(cache_path=f"{output_dir}/emoji-test.txt")
    print(f"Loaded {len(sequences)} emoji sequences from Unicode emoji-test.txt")

    emoji_token_ids = set()
    single_token_emoji = []

    for emoji_str in sequences:
        # Tokenize standalone
        token_ids = tokenizer.encode(emoji_str, add_special_tokens=False)
        emoji_token_ids.update(token_ids)
        if len(token_ids) == 1:
            single_token_emoji.append((emoji_str, token_ids[0]))
        # Tokenize with leading space (different token IDs due to BPE)
        space_ids = tokenizer.encode(" " + emoji_str, add_special_tokens=False)
        emoji_token_ids.update(space_ids)

    # Also add space token — emoji responses often have spaces between emoji
    space_ids = tokenizer.encode(" ", add_special_tokens=False)
    emoji_token_ids.update(space_ids)

    # Always allow EOS token for response termination
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        emoji_token_ids.add(eos_id)

    emoji_token_ids = sorted(emoji_token_ids)

    # Build boolean mask
    mask_size = max(vocab_size, max(emoji_token_ids) + 1)
    mask = np.zeros(mask_size, dtype=np.bool_)
    mask[emoji_token_ids] = True

    print(f"Found {len(emoji_token_ids)} unique token IDs needed for emoji encoding")
    print(f"  (out of {vocab_size} vocab size, mask size {mask_size})")
    print(f"  Single-token emoji: {len(single_token_emoji)}")
    if single_token_emoji[:10]:
        samples = " ".join(e for e, _ in single_token_emoji[:10])
        print(f"  Samples: {samples}")

    # Save outputs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "emoji_mask.npy", mask)

    with open(out / "emoji_tokens.json", "w") as f:
        json.dump(
            {
                "model": model_name,
                "vocab_size": vocab_size,
                "mask_size": mask_size,
                "emoji_token_count": len(emoji_token_ids),
                "single_token_emoji_count": len(single_token_emoji),
                "emoji_token_ids": emoji_token_ids,
                "single_token_emoji": {e: tid for e, tid in single_token_emoji},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved emoji_mask.npy and emoji_tokens.json to {output_dir}/")
    return mask


def main():
    parser = argparse.ArgumentParser(description="Build emoji token mask")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    build_emoji_mask(model_name=args.model, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
