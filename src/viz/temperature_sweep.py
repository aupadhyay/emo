"""Temperature sweep: diversity and length as a function of temperature.

Run: python -m src.viz.temperature_sweep
Requires: EmojiGenerator deployed to Modal
Outputs: viz_outputs/temperature_sweep.png, viz_outputs/temperature_sweep.json
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import modal
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.rl.custom.generate import format_prompt, MODEL_NAME

OUT_DIR = Path("viz_outputs")

PHRASES = [
    ("easy", "pizza"),
    ("medium", "road trip"),
    ("hard", "feeling nostalgic"),
]
TEMPERATURES = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5]
N_SAMPLES = 10


def count_emoji(text: str) -> int:
    return sum(1 for c in text if ord(c) > 127)


def run():
    OUT_DIR.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    gen = modal.Cls.from_name("emoji-generator", "EmojiGenerator")()

    all_results = {}
    unique_ratios = {label: [] for label, _ in PHRASES}
    avg_lengths = {label: [] for label, _ in PHRASES}

    for label, phrase in PHRASES:
        prompt = format_prompt(phrase, tokenizer)
        phrase_results = {}
        for temp in TEMPERATURES:
            print(f"Generating {N_SAMPLES} @ temp={temp} for '{phrase}'...")
            completions = gen.generate.remote(
                [prompt], n=N_SAMPLES, temperature=temp, max_tokens=20
            )[0]
            unique_ratios[label].append(len(set(completions)) / N_SAMPLES)
            avg_lengths[label].append(
                sum(count_emoji(c) for c in completions) / max(len(completions), 1)
            )
            phrase_results[str(temp)] = completions
        all_results[phrase] = phrase_results

    (OUT_DIR / "temperature_sweep.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#2196F3", "#FF9800", "#9C27B0"]
    for (label, _), color in zip(PHRASES, colors):
        ax1.plot(TEMPERATURES, unique_ratios[label], marker="o", label=label, color=color)
        ax2.plot(TEMPERATURES, avg_lengths[label], marker="s", linestyle="--",
                 label=label, color=color)

    ax1.set_title("Unique output ratio vs temperature")
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("Unique / Total")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_title("Average output length vs temperature")
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel("Avg emoji count")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "temperature_sweep.png", dpi=150)
    plt.close()
    print("Saved temperature_sweep.png and temperature_sweep.json")


if __name__ == "__main__":
    run()
