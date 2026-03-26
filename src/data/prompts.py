"""Generate diverse natural language prompts for SFT dataset creation.

Uses Claude API to synthetically generate prompts across categories,
then deduplicates and writes to JSONL.
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

CATEGORIES = [
    {
        "name": "factual_questions",
        "pct": 15,
        "description": "Factual questions about the world, science, history, geography, etc.",
        "examples": [
            "What's the tallest mountain in the world?",
            "How many planets are in the solar system?",
            "What year did World War II end?",
        ],
    },
    {
        "name": "emotional_statements",
        "pct": 15,
        "description": "Statements expressing emotions, feelings, personal experiences with emotional content.",
        "examples": [
            "I just got promoted and I'm so happy",
            "My dog passed away yesterday and I can't stop crying",
            "I'm so nervous about my job interview tomorrow",
        ],
    },
    {
        "name": "descriptions",
        "pct": 15,
        "description": "Descriptions of scenes, objects, places, people, weather, or situations.",
        "examples": [
            "A cat sleeping on a windowsill in the rain",
            "The sunset over the ocean with pink and orange clouds",
            "A busy farmer's market on a Saturday morning",
        ],
    },
    {
        "name": "requests_commands",
        "pct": 10,
        "description": "Requests, commands, instructions, or asking for help with something.",
        "examples": [
            "Remind me to call mom tomorrow",
            "Can you help me find a good Italian restaurant?",
            "Please send this to my boss",
        ],
    },
    {
        "name": "opinions_preferences",
        "pct": 10,
        "description": "Opinions, preferences, comparisons, hot takes, or personal taste.",
        "examples": [
            "I think pizza is better than pasta",
            "Summer is the best season and I'll die on that hill",
            "I prefer cats over dogs honestly",
        ],
    },
    {
        "name": "greetings_social",
        "pct": 10,
        "description": "Greetings, farewells, social pleasantries, small talk, congratulations.",
        "examples": [
            "Good morning! How's your day going?",
            "Happy birthday! Hope you have an amazing day",
            "See you later, have a great weekend!",
        ],
    },
    {
        "name": "abstract_concepts",
        "pct": 10,
        "description": "Abstract concepts, philosophical questions, hypotheticals, deep thoughts.",
        "examples": [
            "What does freedom mean to you?",
            "If you could have any superpower what would it be?",
            "What's the meaning of life?",
        ],
    },
    {
        "name": "complex_multipart",
        "pct": 10,
        "description": "Complex or multi-part messages combining multiple ideas, situations, or emotions.",
        "examples": [
            "I'm moving to a new city for work but I'm nervous about leaving my friends",
            "My flight got cancelled and I'm stuck at the airport overnight, can you help me find a hotel?",
            "I just finished a marathon in the rain and I'm exhausted but proud",
        ],
    },
    {
        "name": "edge_cases",
        "pct": 5,
        "description": "Very short messages (1-3 words), ambiguous statements, sarcasm, slang, or unusual inputs.",
        "examples": [
            "Hi",
            "Bruh",
            "Yeah sure whatever",
            "Oh great, another Monday",
            "It is what it is",
        ],
    },
]

BATCH_PROMPT = """\
Generate exactly {batch_size} diverse, natural user messages for the category: {category_name}.

Category description: {category_description}

Examples for reference (do NOT repeat these):
{examples}

Rules:
- Each message should be 1-2 sentences, under 200 characters
- Messages should sound like real humans talking to a chatbot
- Be diverse in topic, tone, and style within the category
- Include a mix of casual and slightly more formal messages
- No numbering, bullets, or formatting — just one message per line
- Do NOT include any labels, prefixes, or metadata
- Output ONLY the messages, one per line"""


async def generate_batch(
    client: anthropic.AsyncAnthropic,
    category: dict,
    batch_size: int = 50,
    max_retries: int = 5,
) -> list[str]:
    """Generate a batch of prompts for a single category."""
    examples_str = "\n".join(f"- {e}" for e in category["examples"])

    for attempt in range(max_retries + 1):
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": BATCH_PROMPT.format(
                            batch_size=batch_size,
                            category_name=category["name"].replace("_", " "),
                            category_description=category["description"],
                            examples=examples_str,
                        ),
                    }
                ],
            )
            break
        except anthropic.RateLimitError:
            wait = 15 * (attempt + 1)
            tqdm.write(f"  Rate limited, waiting {wait}s...")
            await asyncio.sleep(wait)
        except anthropic.APIError as e:
            tqdm.write(f"  API error: {e}")
            if attempt == max_retries:
                return []
            await asyncio.sleep(5)
    else:
        return []

    text = response.content[0].text.strip()
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    # Filter out lines that look like metadata/numbering
    prompts = []
    for line in lines:
        # Strip leading numbers like "1." or "1)"
        cleaned = line.lstrip("0123456789.-) ").strip()
        if cleaned and len(cleaned) <= 200:
            prompts.append(cleaned)
    return prompts


def deduplicate(prompts: list[str]) -> list[str]:
    """Deduplicate prompts by normalized lowercase text."""
    seen = set()
    unique = []
    for p in prompts:
        key = p.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


async def generate_all_prompts(
    total: int = 30_000,
    batch_size: int = 50,
    max_concurrent: int = 10,
    output_path: str = "data/prompts.jsonl",
) -> None:
    """Generate all prompts across categories."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)

    # Calculate how many prompts per category
    category_targets = []
    for cat in CATEGORIES:
        target = int(total * cat["pct"] / 100)
        # Number of batches needed (overshoot slightly for dedup losses)
        n_batches = (target * 12 // 10) // batch_size + 1
        category_targets.append((cat, target, n_batches))

    all_prompts: list[str] = []
    total_batches = sum(n for _, _, n in category_targets)

    async def run_batch(category: dict) -> list[str]:
        async with semaphore:
            return await generate_batch(client, category, batch_size)

    with tqdm(total=total_batches, desc="Generating prompts") as pbar:
        for cat, target, n_batches in category_targets:
            tasks = [run_batch(cat) for _ in range(n_batches)]
            cat_prompts = []
            for coro in asyncio.as_completed(tasks):
                result = await coro
                cat_prompts.extend(result)
                pbar.update(1)

            # Deduplicate within category and trim to target
            cat_prompts = deduplicate(cat_prompts)
            random.shuffle(cat_prompts)
            cat_prompts = cat_prompts[:target]
            all_prompts.extend(cat_prompts)
            tqdm.write(f"  {cat['name']}: {len(cat_prompts)} prompts (target: {target})")

    # Global dedup
    all_prompts = deduplicate(all_prompts)
    random.shuffle(all_prompts)

    # Write output
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for prompt in all_prompts:
            f.write(json.dumps({"text": prompt}) + "\n")

    print(f"\nGenerated {len(all_prompts)} unique prompts → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate prompt pool for SFT dataset")
    parser.add_argument("--count", type=int, default=30_000, help="Total prompts to generate")
    parser.add_argument("--batch-size", type=int, default=50, help="Prompts per API call")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent API calls")
    parser.add_argument("--output", type=str, default="data/prompts.jsonl", help="Output path")
    args = parser.parse_args()

    asyncio.run(
        generate_all_prompts(
            total=args.count,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
