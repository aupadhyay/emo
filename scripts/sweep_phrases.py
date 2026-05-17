"""Sweep a curated phrase list against the live /emoji endpoint.

Hits the Modal serve endpoint for each phrase up to MAX_TURNS times (simulating
multi-turn clue generation) and prints a markdown table of results.

Usage:
    uv run scripts/sweep_phrases.py
    uv run scripts/sweep_phrases.py --turns 3 --endpoint https://...
"""

import argparse
import json
import sys
import time
import urllib.request
from dataclasses import dataclass, field

ENDPOINT = "https://2c-nyc--emo-serve-web.modal.run"

# Phrases curated for interesting / visually anchored emoji output.
# Organised by category so the blog post can sample across them.
PHRASES = [
    # --- situational / relatable ---
    "parallel parking fail",
    "printer jammed before deadline",
    "drunk texting",
    "sending wrong text to wrong person",
    "running into your ex",
    "accidentally wearing pajama pants",
    "amazon package stolen doorstep",
    "spilled red wine",
    "bluetooth headphones not connecting",
    "remote control batteries dead",

    # --- emotional arcs ---
    "breakup ice cream",
    "crying at a movie",
    "valentines day disappointment",
    "surprise birthday party",
    "wedding day jitters",
    "forgetting an anniversary",

    # --- adventure / journey ---
    "honeymoon in paris",
    "lost luggage at airport",
    "flight delay at airport",
    "hitchhiking on empty highway",
    "road trip playlist",
    "camping under the stars",

    # --- modern life ---
    "video conference on mute",
    "pulling an all nighter",
    "binge watching tv show",
    "skipping the gym",
    "online dating profile",
    "ghost pepper challenge",
    "virtual reality headset gaming",
    "midlife crisis sports car",

    # --- harder / abstract ---
    "high school reunion cringe",
    "thanksgiving dinner arguments",
    "power outage dinner",
    "quarantine hotel room isolation",
    "retirement party",
    "hospital visit",
]


@dataclass
class Turn:
    emoji: str
    guess: str = ""  # empty — we're not running a guesser


@dataclass
class GameResult:
    phrase: str
    turns: list[Turn] = field(default_factory=list)
    error: str = ""


def post(url: str, payload: dict, timeout: int = 30) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def sweep(phrases: list[str], endpoint: str, num_turns: int) -> list[GameResult]:
    results = []
    for i, phrase in enumerate(phrases, 1):
        print(f"  [{i:2d}/{len(phrases)}] {phrase}", end=" ", flush=True)
        result = GameResult(phrase=phrase)
        history: list[dict] = []
        try:
            for _ in range(num_turns):
                resp = post(
                    f"{endpoint}/emoji",
                    {"phrase": phrase, "history": history},
                )
                emoji = resp["emoji"]
                result.turns.append(Turn(emoji=emoji))
                # Feed back a dummy "wrong" guess so the model produces a new clue.
                history.append({"emoji": emoji, "guess": "???"})
                time.sleep(0.1)  # be polite to the endpoint
            print("✓")
        except Exception as exc:
            result.error = str(exc)
            print(f"✗  {exc}")
        results.append(result)
    return results


def render_markdown(results: list[GameResult]) -> str:
    lines = [
        "| Phrase | Turn 1 | Turn 2 | Turn 3 |",
        "|--------|--------|--------|--------|",
    ]
    for r in results:
        if r.error:
            lines.append(f"| {r.phrase} | ❌ `{r.error[:40]}` | | |")
            continue
        turns = [t.emoji for t in r.turns]
        while len(turns) < 3:
            turns.append("")
        lines.append(f"| {r.phrase} | {turns[0]} | {turns[1]} | {turns[2]} |")
    return "\n".join(lines)


def render_jsonl(results: list[GameResult]) -> str:
    rows = []
    for r in results:
        rows.append(json.dumps({
            "phrase": r.phrase,
            "turns": [t.emoji for t in r.turns],
            "error": r.error,
        }, ensure_ascii=False))
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--turns", type=int, default=3, help="emoji turns per phrase")
    parser.add_argument("--out", choices=["markdown", "jsonl"], default="markdown")
    parser.add_argument(
        "--phrases",
        nargs="*",
        help="override phrase list (space-separated, quote multi-word phrases)",
    )
    args = parser.parse_args()

    phrases = args.phrases if args.phrases else PHRASES
    print(f"Sweeping {len(phrases)} phrases × {args.turns} turns against {args.endpoint}\n")

    results = sweep(phrases, args.endpoint, args.turns)

    ok = sum(1 for r in results if not r.error)
    print(f"\n{ok}/{len(results)} succeeded\n")

    output = render_markdown(results) if args.out == "markdown" else render_jsonl(results)
    print(output)

    # Also write to file
    suffix = "md" if args.out == "markdown" else "jsonl"
    out_path = f"scripts/sweep_results.{suffix}"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
