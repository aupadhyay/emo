"""Generate simple, emoji-friendly messages for RL training.

These should be the kind of thing you'd text a friend — short, concrete,
everyday messages that can realistically be communicated in 2-8 emoji.

Usage:
    uv run python -m src.data.generate_rl_prompts_v2
"""

import asyncio
import json
import random

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

PROMPT = """\
Generate 50 short messages (under 40 characters each) that someone might text a friend. \
These should be simple, concrete, everyday messages that could be communicated using emoji.

Categories to cover:
- Greetings and goodbyes ("hey!", "see you tomorrow", "good night")
- Plans and logistics ("running late", "meet me at the park", "dinner at 7?")
- Emotions and reactions ("I'm so happy!", "that's hilarious", "I'm bored")
- Updates about your day ("just got home", "stuck in traffic", "eating lunch")
- Weather and environment ("it's freezing outside", "beautiful day today")
- Food and drink ("I'm starving", "let's get coffee", "pizza tonight?")
- Celebrations ("happy birthday!", "congrats!", "we won!")
- Complaints ("my phone died", "wifi is down", "I'm exhausted")
- Requests ("call me back", "send me the address", "pick up milk")
- Activities ("going for a run", "watching a movie", "at the gym")

Rules:
- Under 40 characters each
- Simple, concrete, everyday language
- No philosophical questions or abstract concepts
- Things a real person would actually text
- Vary the tone: happy, sad, angry, neutral, excited, tired, etc.

Output as a JSON array of strings, nothing else."""


async def generate_batch(client: AsyncAnthropic, batch_num: int) -> list[str]:
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=1.0,
        messages=[{"role": "user", "content": PROMPT}],
    )
    text = response.content[0].text.strip()
    # Parse JSON array
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


async def main():
    client = AsyncAnthropic()
    all_messages = []

    # Generate 20 batches of 50 = 1000 messages
    n_batches = 20
    print(f"Generating {n_batches} batches of 50 messages...")

    tasks = [generate_batch(client, i) for i in range(n_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  Batch {i} failed: {result}")
        else:
            all_messages.extend(result)
            print(f"  Batch {i}: {len(result)} messages")

    # Deduplicate and filter
    seen = set()
    filtered = []
    for msg in all_messages:
        msg = msg.strip()
        key = msg.lower()
        if key not in seen and 5 < len(msg) < 60:
            seen.add(key)
            filtered.append(msg)

    random.seed(42)
    random.shuffle(filtered)

    with open("data/rl_prompts.jsonl", "w") as f:
        for text in filtered:
            f.write(json.dumps({"text": text, "difficulty": "easy"}) + "\n")

    print(f"\nWrote {len(filtered)} messages to data/rl_prompts.jsonl")
    print("\nSamples:")
    for msg in filtered[:20]:
        print(f"  {msg}")


if __name__ == "__main__":
    asyncio.run(main())
