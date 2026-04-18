"""Before/after comparison: unconstrained vs emoji-constrained generation.

Run: python -m src.viz.before_after
Requires: EmojiGenerator deployed to Modal
Outputs: viz_outputs/before_after.md, viz_outputs/before_after.json
"""

import json
import sys
from pathlib import Path

import modal
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.rl.custom.generate import format_prompt, MODEL_NAME

OUT_DIR = Path("viz_outputs")

TEST_PHRASES = [
    "birthday party",
    "road trip",
    "feeling sad",
    "I just got promoted at work",
    "the economy is struggling",
]


def run():
    OUT_DIR.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    gen = modal.Cls.from_name("emoji-generator", "EmojiGenerator")()

    prompts = [format_prompt(p, tokenizer) for p in TEST_PHRASES]

    print("Generating unconstrained outputs...")
    unconstrained = gen.generate_unconstrained.remote(prompts, max_tokens=60)

    print("Generating constrained outputs (3 per phrase)...")
    constrained_batches = gen.generate.remote(prompts, n=3, temperature=1.0, max_tokens=20)

    results = []
    md_lines = []
    for phrase, unc, con_list in zip(TEST_PHRASES, unconstrained, constrained_batches):
        results.append({"phrase": phrase, "unconstrained": unc, "constrained": con_list})
        md_lines.append(f"### {phrase}")
        md_lines.append(f"**Unconstrained:** {unc}")
        md_lines.append("**Constrained:**")
        for i, c in enumerate(con_list, 1):
            md_lines.append(f"{i}. {c}")
        md_lines.append("")

    (OUT_DIR / "before_after.md").write_text("\n".join(md_lines))
    (OUT_DIR / "before_after.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    print("Saved before_after.md and before_after.json")
    print("\n".join(md_lines))


if __name__ == "__main__":
    run()
