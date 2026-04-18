"""Emoji frequency distribution — baseline before training.

Run: python -m src.viz.emoji_frequency
Requires: EmojiGenerator deployed to Modal, data/prompts.jsonl (or uses fallback)
Outputs: viz_outputs/emoji_frequency.png, viz_outputs/emoji_frequency.json,
         viz_outputs/emoji_frequency_summary.txt
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("module://mplcairo.macosx")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontEntry, fontManager
import modal

# Inject NotoColorEmoji directly, bypassing FreeType (which can't parse color fonts)
_noto_path = Path.home() / "Library/Fonts/NotoColorEmoji.ttf"
if _noto_path.exists():
    fontManager.ttflist.append(
        FontEntry(
            fname=str(_noto_path),
            name="Noto Color Emoji",
            style="normal",
            variant="normal",
            weight=400,
            stretch="normal",
            size="scalable",
        )
    )
    fontManager._findfont_cached.cache_clear()  # ty: ignore[unresolved-attribute]

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.rl.custom.generate import format_prompt, MODEL_NAME

OUT_DIR = Path("viz_outputs")
PROMPTS_PATH = Path("data/prompts.jsonl")
N_PROMPTS = 150

FALLBACK_PHRASES = [
    "pizza",
    "birthday party",
    "road trip",
    "I love coffee",
    "the sun is shining",
    "homework is hard",
    "my cat is sleeping",
    "I got a raise",
    "the ocean at sunset",
    "going hiking",
    "movie night",
    "I'm so tired",
    "new year new me",
    "winter is cold",
    "summer vacation",
    "feeling anxious",
    "great news",
    "traffic jam",
    "meditation",
    "cooking dinner",
    "friendship",
    "loneliness",
    "celebration",
    "rain outside",
    "mountain climbing",
    "a good book",
    "heartbreak",
    "first day of school",
    "retirement party",
    "getting married",
    "new baby",
    "graduation",
    "moving away",
    "running a marathon",
    "winning a game",
    "losing a game",
    "a lazy Sunday",
    "spring flowers",
    "autumn leaves",
    "a thunderstorm",
    "the stars at night",
    "fresh bread",
    "ice cream",
    "spicy food",
    "a long flight",
    "jet lag",
    "working from home",
    "the economy",
    "politics",
    "climate change",
    "technology",
    "music festival",
    "art museum",
    "science experiment",
    "learning to code",
    "yoga class",
    "gym workout",
    "swimming",
    "dancing",
    "singing",
    "painting",
    "gardening",
    "baking cake",
    "making coffee",
    "morning routine",
    "bedtime",
    "deadline stress",
    "project finished",
    "team meeting",
    "promotion",
    "job interview",
    "salary negotiation",
    "quitting a job",
    "starting a business",
    "travel adventure",
    "getting lost",
    "finding a shortcut",
    "missing a flight",
    "reuniting with family",
    "saying goodbye",
    "a surprise gift",
    "a broken promise",
    "forgiveness",
    "gratitude",
    "nostalgia",
    "hope",
    "fear",
    "excitement",
    "boredom",
    "curiosity",
    "inspiration",
    "confusion",
    "clarity",
    "peace",
    "chaos",
    "love",
    "jealousy",
    "pride",
    "shame",
    "courage",
    "kindness",
    "dog playing fetch",
    "cat knocking things over",
    "baby laughing",
    "grandparent",
    "the library",
    "the market",
    "the hospital",
    "the park",
    "the beach",
    "a thunderstorm at sea",
    "the city at night",
    "countryside morning",
    "northern lights",
    "a desert",
    "a jungle",
    "a snowy forest",
    "underwater",
    "a spaceship",
    "time travel",
    "robots",
    "alien contact",
    "the future",
    "the past",
    "a dream",
    "a nightmare",
    "meditation retreat",
    "disconnect",
    "adventure begins",
    "the journey home",
]


_MODIFIERS = {0x200D, 0xFE0F, *range(0x1F3FB, 0x1F3FF + 1)}


def extract_emoji(text: str) -> list[str]:
    result = []
    i = 0
    while i < len(text):
        cp = ord(text[i])
        if cp > 127 and cp not in _MODIFIERS:
            emoji = text[i]
            i += 1
            # Greedily consume modifiers and ZWJ-joined continuations
            while i < len(text):
                ncp = ord(text[i])
                if ncp in _MODIFIERS:
                    emoji += text[i]
                    i += 1
                elif emoji.endswith("\u200d") and ncp > 127:
                    emoji += text[i]
                    i += 1
                else:
                    break
            result.append(emoji)
        else:
            i += 1
    return result


def run():
    OUT_DIR.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    gen = modal.Cls.from_name("emoji-generator", "EmojiGenerator")()

    if PROMPTS_PATH.exists():
        import random

        phrases = []
        with open(PROMPTS_PATH) as f:
            for line in f:
                row = json.loads(line)
                phrases.append(row.get("text", row.get("prompt", "")))
        random.shuffle(phrases)
        phrases = [p for p in phrases if p][:N_PROMPTS]
        print(f"Loaded {len(phrases)} phrases from {PROMPTS_PATH}")
    else:
        phrases = FALLBACK_PHRASES[:N_PROMPTS]
        print(f"Using {len(phrases)} fallback phrases")

    prompts = [format_prompt(p, tokenizer) for p in phrases]

    all_completions = []
    batch_size = 50
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        print(
            f"Generating batch {i // batch_size + 1}/{-(-len(prompts) // batch_size)}..."
        )
        for completions in gen.generate.remote(
            batch, n=1, temperature=1.0, max_tokens=20
        ):
            all_completions.append(completions[0])

    counter: Counter = Counter()
    total_emoji = 0
    for text in all_completions:
        emojis = extract_emoji(text)
        counter.update(emojis)
        total_emoji += len(emojis)

    top30 = counter.most_common(30)
    all_counts = sorted(counter.values(), reverse=True)

    (OUT_DIR / "emoji_frequency.json").write_text(
        json.dumps(dict(counter.most_common()), indent=2, ensure_ascii=False)
    )

    top10 = counter.most_common(10)
    summary_lines = [
        f"Total generations: {len(all_completions)}",
        f"Total emoji generated: {total_emoji}",
        f"Unique emoji used: {len(counter)}",
        "",
        "Top 10 emoji:",
    ]
    for emoji, count in top10:
        summary_lines.append(f"  {emoji}  {count}  ({100 * count / total_emoji:.1f}%)")
    summary = "\n".join(summary_lines)
    (OUT_DIR / "emoji_frequency_summary.txt").write_text(summary)
    print(summary)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 18))

    emojis_top = [e for e, _ in top30]
    counts_top = [c for _, c in top30]
    ax1.barh(range(len(top30)), counts_top[::-1], color="#2196F3", height=0.8)
    ax1.set_yticks([])
    for i, emoji in enumerate(emojis_top[::-1]):
        display = emoji.split("\u200d")[0] if "\u200d" in emoji else emoji
        ax1.text(-0.02, i - 0.1, display, fontfamily="Noto Color Emoji", fontsize=22,
                 va="center_baseline", ha="right", transform=ax1.get_yaxis_transform())
    ax1.set_xlabel("Frequency", fontsize=13)
    ax1.set_title("Top 30 emoji (baseline, before training)", fontsize=14)
    ax1.grid(axis="x", alpha=0.3)

    ranks = list(range(1, len(all_counts) + 1))
    ax2.plot(ranks, all_counts, color="#FF5722")
    ax2.set_yscale("log")
    ax2.set_xlabel("Rank", fontsize=13)
    ax2.set_ylabel("Count (log scale)", fontsize=13)
    ax2.set_title("Emoji frequency distribution (Zipf-like tail)", fontsize=14)
    ax2.grid(alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.savefig(OUT_DIR / "emoji_frequency.png", dpi=150)
    plt.close()
    print(
        "Saved emoji_frequency.png, emoji_frequency.json, emoji_frequency_summary.txt"
    )


if __name__ == "__main__":
    run()
