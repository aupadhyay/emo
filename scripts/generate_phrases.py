"""Generate a diverse phrase dataset for emoji-communication training.

Calls Claude to produce ~500 training phrases + ~50 held-out phrases across
several thematic buckets, dedupes, and writes data/training_phrases.json.

Usage:
    uv run python scripts/generate_phrases.py
"""

import json
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

OUT_PATH = Path("data/training_phrases.json")
MODEL = "claude-sonnet-4-20250514"

THEMES = [
    ("daily life and routines",
     "morning routine, doing laundry, commuting, grocery shopping, cooking dinner"),
    ("emotions and inner states",
     "feeling overwhelmed, inner peace, jealousy, gratitude, homesickness"),
    ("relationships and social moments",
     "first date, breakup, reconnecting with old friends, awkward silence, family reunion"),
    ("work and career situations",
     "job interview, getting promoted, missed deadline, asking for a raise, quitting a job"),
    ("travel and adventure",
     "road trip, missed flight, getting lost in a new city, beach vacation, hiking a mountain"),
    ("nature, weather, and environment",
     "snowstorm, sunset over the ocean, autumn leaves, drought, thunderstorm at night"),
    ("idioms and figures of speech",
     "raining cats and dogs, spill the beans, break a leg, piece of cake, on cloud nine"),
    ("life events and milestones",
     "wedding day, graduation, retirement, having a baby, moving to a new city"),
    ("everyday objects and concepts",
     "favorite mug, broken phone, lost keys, library book, old photograph"),
    ("modern situations",
     "bad zoom call, dead phone battery, traffic jam, doom scrolling, online shopping spree"),
]

BATCH_PROMPT = """Generate 60 short English phrases suitable for an emoji-communication game.

Theme: {theme}
Examples in this theme: {examples}

Requirements:
- Each phrase 2-7 words, lowercase, no quotes or punctuation at the end
- Concrete and visualizable (someone should be able to describe it with emoji)
- Diverse — mix moods, tones, and specifics; don't repeat near-synonyms
- Avoid abstract concepts that have no visual hook
- One phrase per line, no numbering, no bullets, no extra commentary

Output exactly 60 phrases."""


def generate_batch(client: anthropic.Anthropic, theme: str, examples: str) -> list[str]:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": BATCH_PROMPT.format(theme=theme, examples=examples)}],
    )
    text = msg.content[0].text
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
        line = line.strip('"\'').rstrip(".!?,;:").lower()
        if 2 <= len(line.split()) <= 8 and len(line) <= 60:
            lines.append(line)
    return lines


def main():
    client = anthropic.Anthropic()
    seen: set[str] = set()
    phrases: list[str] = []

    for theme, examples in THEMES:
        print(f"Generating: {theme}...")
        batch = generate_batch(client, theme, examples)
        added = 0
        for p in batch:
            if p not in seen:
                seen.add(p)
                phrases.append(p)
                added += 1
        print(f"  +{added} new (total: {len(phrases)})")

    print(f"\nTotal unique phrases: {len(phrases)}")

    import random
    rng = random.Random(42)
    rng.shuffle(phrases)

    n_held = 50
    held_out = phrases[:n_held]
    training = phrases[n_held:]

    print(f"Training: {len(training)}  Held-out: {len(held_out)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"training": training, "held_out": held_out}, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
