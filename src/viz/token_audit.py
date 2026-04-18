"""Token audit: visualize the emoji mask constraint ratio.

Run: python -m src.viz.token_audit
Outputs: viz_outputs/token_audit.png, viz_outputs/token_audit.json, viz_outputs/multi_token_examples.md
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.rl.custom.generate import build_emoji_mask, _load_emoji_sequences, MODEL_NAME

OUT_DIR = Path("viz_outputs")

EXAMPLE_EMOJI = [
    "🚗", "🍕", "😀", "❤️", "🎉",
    "👨‍👩‍👧‍👦", "👩‍💻", "🏳️‍🌈", "👨‍🍳", "🧑‍🚀",
    "👋🏽", "👍🏿", "✊🏻", "🤝🏾", "🙌🏼",
]

MODIFIER_CODEPOINTS = set(range(0x1F3FB, 0x1F3FF + 1))
MODIFIER_CODEPOINTS.add(0x200D)
MODIFIER_CODEPOINTS.add(0xFE0F)


def run():
    OUT_DIR.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    mask = build_emoji_mask(tokenizer)

    vocab_size = tokenizer.vocab_size
    n_allowed = int(mask.sum().item())

    sequences = _load_emoji_sequences()
    single_token_count = sum(
        1 for seq in sequences
        if len(tokenizer.encode(seq, add_special_tokens=False)) == 1
    )
    multi_token_count = len(sequences) - single_token_count

    modifier_ids: set[int] = set()
    for cp in MODIFIER_CODEPOINTS:
        ch = chr(cp)
        modifier_ids.update(tokenizer.encode(ch, add_special_tokens=False))
        modifier_ids.update(tokenizer.encode(" " + ch, add_special_tokens=False))

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.barh(0, n_allowed, color="#4CAF50", label=f"Emoji tokens ({n_allowed:,})")
    ax.barh(0, vocab_size - n_allowed, left=n_allowed, color="#E0E0E0",
            label=f"Non-emoji ({vocab_size - n_allowed:,})")
    ax.set_xlim(0, vocab_size)
    ax.set_yticks([])
    ax.set_xlabel("Token ID count")
    ax.set_title(
        f"Emoji mask: {n_allowed:,} / {vocab_size:,} tokens allowed "
        f"({100 * n_allowed / vocab_size:.1f}%)"
    )
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "token_audit.png", dpi=150)
    plt.close()

    summary = {
        "model": MODEL_NAME,
        "vocab_size": vocab_size,
        "n_allowed": n_allowed,
        "pct_allowed": round(100 * n_allowed / vocab_size, 2),
        "single_token_emoji_sequences": single_token_count,
        "multi_token_emoji_sequences": multi_token_count,
        "modifier_token_ids_count": len(modifier_ids),
    }
    (OUT_DIR / "token_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    rows = ["| Emoji | Token IDs | Num Tokens |", "|-------|-----------|------------|"]
    for emoji in EXAMPLE_EMOJI:
        ids = tokenizer.encode(emoji, add_special_tokens=False)
        rows.append(f"| {emoji} | {ids} | {len(ids)} |")
    (OUT_DIR / "multi_token_examples.md").write_text("\n".join(rows))
    print("Saved token_audit.png, token_audit.json, multi_token_examples.md")


if __name__ == "__main__":
    run()
