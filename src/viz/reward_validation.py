"""Phase 1 validation: reward function + simulated guesser diagnostics.

Steps:
  1. Similarity calibration
  2. Single-turn reward distribution (20 phrases, 8 rollouts each)
  3. Multi-turn episode validation (10 phrases, 4 rollouts each)
  4. Guesser response quality audit
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import modal
import numpy as np

from src.rl.custom.env import Episode, SimulatedGuesser, run_episode
from src.rl.custom.generate import DEFAULT_SYSTEM_PROMPT, MODEL_NAME, format_prompt
from src.rl.custom.reward import SimilarityScorer, compute_turn_rewards

logging.basicConfig(level=logging.WARNING)

PHRASES_EASY = ["pizza", "rainy day", "birthday party", "basketball", "cooking dinner"]
PHRASES_MEDIUM = ["road trip", "job interview", "moving to a new city", "watching a scary movie", "going to the gym"]
PHRASES_HARD = [
    "feeling nostalgic", "broken heart", "time is running out",
    "the economy is struggling", "making a difficult decision",
]
PHRASES_VERY_HARD = [
    "actions speak louder than words", "a fresh start after failure",
    "the calm before the storm", "learning from your mistakes", "finding balance in life",
]
ALL_PHRASES = PHRASES_EASY + PHRASES_MEDIUM + PHRASES_HARD + PHRASES_VERY_HARD

CALIBRATION_PAIRS = [
    ("road trip", "road trip", "exact match"),
    ("road trip", "car journey", "close synonym"),
    ("road trip", "driving", "partial"),
    ("road trip", "vacation", "related"),
    ("road trip", "pizza", "unrelated"),
    ("birthday party", "birthday celebration", "close synonym"),
    ("birthday party", "party", "partial"),
    ("birthday party", "cake", "tangential"),
    ("feeling nostalgic", "nostalgia", "close"),
    ("feeling nostalgic", "feeling sad", "adjacent emotion"),
    ("feeling nostalgic", "happy", "wrong"),
]


def step1_calibration(scorer: SimilarityScorer) -> float:
    print("\n" + "=" * 60)
    print("STEP 1: Similarity Calibration")
    print("=" * 60)

    rows = []
    for a, b, label in CALIBRATION_PAIRS:
        sim = scorer.score(a, b)
        rows.append((a, b, label, sim))
        print(f"  [{label:20s}] {a!r} vs {b!r}: {sim:.4f}")

    # Recommend threshold based on calibration
    exact_sims = [sim for _, _, label, sim in rows if label == "exact match"]
    close_sims = [sim for _, _, label, sim in rows if "close" in label or "synonym" in label]
    unrelated_sims = [sim for _, _, label, sim in rows if label in ("unrelated", "wrong")]

    print(f"\n  Exact match avg:    {np.mean(exact_sims):.4f}")
    print(f"  Close synonym avg:  {np.mean(close_sims):.4f}")
    print(f"  Unrelated avg:      {np.mean(unrelated_sims):.4f}")

    # Threshold = midpoint between close synonyms and unrelated
    threshold = (np.mean(close_sims) + np.mean(exact_sims)) / 2
    print(f"\n  Recommended exact_match_threshold: {threshold:.2f}")
    return threshold


def step2_single_turn(
    generator,
    guesser: SimulatedGuesser,
    scorer: SimilarityScorer,
    threshold: float,
) -> list[dict]:
    print("\n" + "=" * 60)
    print("STEP 2: Single-Turn Reward Distribution")
    print("=" * 60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    all_records = []
    all_sims = []
    turn1_correct = 0
    total = 0

    for phrase in ALL_PHRASES:
        prompts = [format_prompt(phrase, tokenizer, DEFAULT_SYSTEM_PROMPT)] * 8
        # Generate 8 rollouts at once
        batch_results = generator.generate.remote(prompts, n=1, temperature=1.0, max_tokens=20)
        emoji_outputs = [r[0] for r in batch_results]

        guess_inputs = [{"emoji": emoji, "previous_guesses": []} for emoji in emoji_outputs]
        guesses = guesser.guess_batch(guess_inputs)

        sims = [scorer.score(phrase, g) for g in guesses]
        correct = sum(1 for s in sims if s >= threshold)

        phrase_record = {
            "phrase": phrase,
            "emoji_outputs": emoji_outputs,
            "guesses": guesses,
            "similarities": sims,
            "mean_sim": float(np.mean(sims)),
            "std_sim": float(np.std(sims)),
            "turn1_correct": correct,
        }
        all_records.append(phrase_record)
        all_sims.extend(sims)
        turn1_correct += correct
        total += 8

        variance_flag = " [LOW VARIANCE]" if np.std(sims) < 0.1 else ""
        print(f"\n  {phrase!r}")
        print(f"    mean={np.mean(sims):.3f}  std={np.std(sims):.3f}  correct={correct}/8{variance_flag}")
        for emoji, guess, sim in zip(emoji_outputs, guesses, sims):
            marker = "✓" if sim >= threshold else " "
            print(f"    {marker} {emoji} -> {guess!r} ({sim:.3f})")

    accuracy = turn1_correct / total
    print(f"\n  Overall turn-1 accuracy: {accuracy:.1%} ({turn1_correct}/{total})")

    if accuracy > 0.50:
        print("  ⚠  Accuracy too high (>50%). Consider dumbing down the guesser prompt.")
    elif accuracy < 0.15:
        print("  ⚠  Accuracy too low (<15%). Check if phrases are too hard or emoji too bad.")

    # Histogram
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_sims, bins=20, range=(0, 1), edgecolor="black")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"Single-Turn Similarity Distribution (n={total})")
    hist_path = OUT_DIR / "single_turn_sim_histogram.png"
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Histogram saved: {hist_path}")

    raw_path = OUT_DIR / "single_turn_raw.json"
    raw_path.write_text(json.dumps(all_records, indent=2))
    print(f"  Raw data saved: {raw_path}")

    return all_records


def step3_multiturn(
    generator,
    guesser: SimulatedGuesser,
    scorer: SimilarityScorer,
    threshold: float,
) -> list[dict]:
    print("\n" + "=" * 60)
    print("STEP 3: Multi-Turn Episode Validation")
    print("=" * 60)

    # 10 phrases: 2 easy, 2 medium, 3 hard, 3 very hard
    phrases = (
        PHRASES_EASY[:2]
        + PHRASES_MEDIUM[:2]
        + PHRASES_HARD[:3]
        + PHRASES_VERY_HARD[:3]
    )

    all_episode_data = []

    for phrase in phrases:
        print(f"\n  Phrase: {phrase!r}")
        phrase_traj_rewards = []

        for rollout_idx in range(4):
            episode = run_episode(
                generator=generator,
                guesser=guesser,
                scorer=scorer,
                target_phrase=phrase,
                max_turns=5,
                exact_match_threshold=threshold,
            )

            guesses = [t.guess for t in episode.turns]
            result = compute_turn_rewards(
                target_phrase=phrase,
                guesses=guesses,
                scorer=scorer,
                exact_match_threshold=threshold,
            )

            traj_reward = result["trajectory_reward"]
            phrase_traj_rewards.append(traj_reward)

            # Print transcript
            print(f"\n    Rollout {rollout_idx + 1}:")
            for turn in episode.turns:
                marker = "✓" if turn.similarity >= threshold else " "
                print(f"      Turn {turn.turn_number}: {turn.emoji_output} -> {turn.guess!r} (sim: {turn.similarity:.2f}) {marker}")
            print(f"      Trajectory reward: {traj_reward:.4f}")
            print(f"      Deltas: {[f'{d:+.3f}' for d in result['deltas']]}")

            # Diagnostics: is model changing emoji on turn 2+?
            if len(episode.turns) > 1:
                t1 = episode.turns[0].emoji_output
                t2 = episode.turns[1].emoji_output
                if t1 == t2:
                    print("      ⚠  Turn 2 emoji identical to turn 1 (model ignoring feedback)")

            ep_data = {
                "phrase": phrase,
                "rollout": rollout_idx + 1,
                "turns": [
                    {
                        "turn_number": t.turn_number,
                        "emoji_output": t.emoji_output,
                        "guess": t.guess,
                        "similarity": t.similarity,
                    }
                    for t in episode.turns
                ],
                "completed": episode.completed,
                "completion_turn": episode.completion_turn,
                "similarities": result["similarities"],
                "deltas": result["deltas"],
                "turn_rewards": result["turn_rewards"],
                "trajectory_reward": traj_reward,
            }
            all_episode_data.append(ep_data)

        traj_std = np.std(phrase_traj_rewards)
        traj_mean = np.mean(phrase_traj_rewards)
        variance_flag = " [LOW VARIANCE]" if traj_std < 0.1 else ""
        print(f"\n    Trajectory reward: mean={traj_mean:.3f}  std={traj_std:.3f}{variance_flag}")

    transcripts_path = OUT_DIR / "multiturn_transcripts.json"
    transcripts_path.write_text(json.dumps(all_episode_data, indent=2))
    print(f"\n  Transcripts saved: {transcripts_path}")

    return all_episode_data


def step4_guesser_quality(
    guesser: SimulatedGuesser,
    scorer: SimilarityScorer,
    threshold: float,
) -> None:
    print("\n" + "=" * 60)
    print("STEP 4: Guesser Response Quality Audit")
    print("=" * 60)

    import logging as _logging

    # Temporarily capture WARNING logs to count cleaning events
    class _Counter(_logging.Handler):
        def __init__(self):
            super().__init__()
            self.count = 0
            self.samples = []

        def emit(self, record):
            self.count += 1
            if len(self.samples) < 5:
                self.samples.append(record.getMessage())

    counter = _Counter()
    env_logger = logging.getLogger("src.rl.custom.env")
    env_logger.addHandler(counter)
    env_logger.setLevel(logging.WARNING)

    # Run 50 guesses across varied emoji (we'll use single-turn)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Generate 50 emoji sequences for audit phrases
    audit_phrases = (ALL_PHRASES * 3)[:50]
    prompts = [format_prompt(p, tokenizer, DEFAULT_SYSTEM_PROMPT) for p in audit_phrases]

    gen = modal.Cls.from_name("emoji-generator", "EmojiGenerator")()
    batch = gen.generate.remote(prompts, n=1, temperature=1.0, max_tokens=20)
    emoji_list = [r[0] for r in batch]

    guess_inputs = [{"emoji": e, "previous_guesses": []} for e in emoji_list]
    guesses = guesser.guess_batch(guess_inputs)

    env_logger.removeHandler(counter)

    n_cleaned = counter.count
    n_empty = sum(1 for g in guesses if not g.strip())
    n_long = sum(1 for g in guesses if len(g.split()) > 10)

    print(f"  Total responses:      {len(guesses)}")
    print(f"  Needed cleaning:      {n_cleaned} ({100 * n_cleaned / len(guesses):.1f}%)")
    print(f"  Empty responses:      {n_empty}")
    print(f"  Suspiciously long:    {n_long}")

    if n_cleaned / len(guesses) >= 0.05:
        print("  ⚠  >5% responses needed cleaning — review guesser prompt")
    else:
        print("  ✓  Response quality looks good")

    if counter.samples:
        print("  Sample cleaned responses:")
        for s in counter.samples:
            print(f"    {s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stateless",
        action="store_true",
        help="Use stateless guesser (previous guesses as text list). Default is conversation mode.",
    )
    args = parser.parse_args()

    global OUT_DIR
    OUT_DIR = Path("viz_outputs/phase-1" if args.stateless else "viz_outputs/phase-1-conversation")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mode = "stateless" if args.stateless else "conversation"
    print(f"Phase 1 Validation: Reward Function + Simulated Guesser  [guesser mode: {mode}]")

    scorer = SimilarityScorer()
    guesser = SimulatedGuesser(
        model="claude-sonnet-4-20250514",
        difficulty="casual",
        conversation_mode=not args.stateless,
    )

    with modal.enable_output():
        generator = modal.Cls.from_name("emoji-generator", "EmojiGenerator")()

        threshold = step1_calibration(scorer)
        step2_single_turn(generator, guesser, scorer, threshold)
        step3_multiturn(generator, guesser, scorer, threshold)
        step4_guesser_quality(guesser, scorer, threshold)

    print("\n" + "=" * 60)
    print(f"Validation complete. Outputs in {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
