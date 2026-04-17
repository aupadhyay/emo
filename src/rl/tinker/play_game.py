"""Interactive retrieval game — watch or play the emoji communication game.

Usage:
    # Watch mode: sender and judge play, you observe
    uv run python -m src.rl.play_game
    uv run python -m src.rl.play_game --message "I just got promoted at work!"

    # Human mode: YOU are the judge, try to decode the emoji
    uv run python -m src.rl.play_game --human
"""

import argparse
import asyncio
import json
import random

import tinker
from dotenv import load_dotenv
from tinker_cookbook import renderers, tokenizer_utils

from src.rl.tinker.emoji_completer import EmojiTokenCompleter
from src.rl.tinker.judge import JudgeClient
from src.rl.tinker.reward import RetrievalGameReward

load_dotenv()

SENDER_SYSTEM_PROMPT = (
    "You communicate exclusively using emoji. No text, numbers, or punctuation ever. "
    "Respond with 2-8 emoji that capture the core meaning of what you need to communicate."
)


async def play_one(
    target: str,
    policy: EmojiTokenCompleter,
    judge: JudgeClient,
    reward_fn: RetrievalGameReward,
    renderer: renderers.Renderer,
    tokenizer,
    max_turns: int = 3,
):
    emoji_history = []
    judge_guesses = []
    similarities = []

    messages: list[renderers.Message] = [
        {"role": "system", "content": SENDER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Communicate this message using only emoji: {target}",
        },
    ]

    print(f"\n{'='*60}")
    print(f"TARGET: {target}")
    print(f"{'='*60}")

    for turn in range(1, max_turns + 1):
        prompt = renderer.build_generation_prompt(messages)
        stop = renderer.get_stop_sequences()

        result = await policy(prompt, stop)
        decoded = tokenizer.decode(result.tokens, skip_special_tokens=True).strip()
        emoji_history.append(decoded)

        guess = await judge.reconstruct_async(
            emoji_history, judge_guesses=judge_guesses
        )
        judge_guesses.append(guess)

        sim = reward_fn.embedder.similarity(target, guess)
        delta = sim - similarities[-1] if similarities else 0.0
        similarities.append(sim)

        status = "SUCCESS" if sim >= reward_fn.similarity_threshold else ""
        delta_str = f" ({delta:+.3f})" if turn > 1 else ""

        print(f"\n  Turn {turn}:")
        print(f"    Sender:  {decoded}")
        print(f"    Judge:   {guess}")
        print(f"    Sim:     {sim:.3f}{delta_str} {status}")

        if sim >= reward_fn.similarity_threshold:
            break

        messages.append({"role": "assistant", "content": decoded})
        messages.append(
            {
                "role": "user",
                "content": (
                    f'The receiver guessed: "{guess}"\n'
                    f'The original message was: "{target}"\n'
                    "Send emoji to correct what they got wrong."
                ),
            }
        )

    final = reward_fn.compute(
        target_message=target,
        judge_final_guess=judge_guesses[-1],
        num_turns=len(emoji_history),
        format_violation=False,
    )
    print(f"\n  Final reward: {final['reward']:.3f}")
    print(f"  Turns used:   {len(emoji_history)}/{max_turns}")


async def play_human(
    target: str,
    policy: EmojiTokenCompleter,
    reward_fn: RetrievalGameReward,
    renderer: renderers.Renderer,
    tokenizer,
    max_turns: int = 3,
):
    """Human plays as the judge — you try to guess the message from emoji."""
    emoji_history = []
    guesses = []
    similarities = []

    messages: list[renderers.Message] = [
        {"role": "system", "content": SENDER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Communicate this message using only emoji: {target}",
        },
    ]

    print(f"\n{'='*60}")
    print(f"  The sender has a secret message.")
    print(f"  Try to figure out what it is from the emoji!")
    print(f"{'='*60}")

    for turn in range(1, max_turns + 1):
        prompt = renderer.build_generation_prompt(messages)
        stop = renderer.get_stop_sequences()

        result = await policy(prompt, stop)
        decoded = tokenizer.decode(result.tokens, skip_special_tokens=True).strip()
        emoji_history.append(decoded)

        if turn == 1:
            print(f"\n  Emoji: {decoded}")
        else:
            print(f"\n  Clarification emoji: {decoded}")

        try:
            guess = input("  Your guess: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  The answer was: {target}")
            return

        if not guess:
            print(f"\n  The answer was: {target}")
            return

        guesses.append(guess)
        sim = reward_fn.embedder.similarity(target, guess)
        delta = sim - similarities[-1] if similarities else 0.0
        similarities.append(sim)

        if sim >= reward_fn.similarity_threshold:
            print(f"\n  ✓ CORRECT! (similarity: {sim:.3f})")
            print(f"  The message was: {target}")
            print(f"  Turns used: {turn}/{max_turns}")
            return

        delta_str = f" ({delta:+.3f})" if turn > 1 else ""
        print(f"  Not quite... (similarity: {sim:.3f}{delta_str})")

        if turn < max_turns:
            print(f"  The sender will send more emoji to help you.")
            messages.append({"role": "assistant", "content": decoded})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f'The receiver guessed: "{guess}"\n'
                        f'The original message was: "{target}"\n'
                        "Send emoji to correct what they got wrong."
                    ),
                }
            )

    print(f"\n  ✗ Out of turns!")
    print(f"  The message was: {target}")
    print(f"  Your best similarity: {max(similarities):.3f}")


async def main(args):
    service = tinker.ServiceClient()

    tc = await service.create_lora_training_client_async(
        base_model="Qwen/Qwen3.5-4B", rank=32
    )
    tc.load_state(args.checkpoint)
    sender_sc = tc.save_weights_and_get_sampling_client(name="play-sender")
    tokenizer = tc.get_tokenizer()
    renderer = renderers.get_renderer("qwen3_disable_thinking", tokenizer)

    policy = EmojiTokenCompleter(
        sampling_client=sender_sc,
        max_tokens=20,
        temperature=0.7,
    )
    reward_fn = RetrievalGameReward(similarity_threshold=args.similarity_threshold)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Similarity threshold: {args.similarity_threshold}")

    if args.human:
        # Load prompts for random selection
        prompts = []
        try:
            with open("data/rl_prompts.jsonl") as f:
                prompts = [json.loads(line)["text"] for line in f]
        except FileNotFoundError:
            prompts = [
                "I just got promoted at work!",
                "My flight got cancelled and I'm stuck at the airport",
                "The sunset over the ocean was beautiful",
                "Happy birthday! Hope you have a great day",
                "I need to go grocery shopping",
                "It's raining cats and dogs outside",
                "Can you recommend a good Italian restaurant?",
                "I'm so hungry I could eat a horse",
            ]

        while True:
            target = random.choice(prompts)
            await play_human(
                target, policy, reward_fn, renderer, tokenizer, args.max_turns
            )
            try:
                again = input("\nPlay again? (y/n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if again not in ("y", "yes", ""):
                break
    elif args.message:
        judge = JudgeClient.create(service, judge_model=args.judge_model)
        await play_one(
            args.message, policy, judge, reward_fn, renderer, tokenizer, args.max_turns
        )
    else:
        judge = JudgeClient.create(service, judge_model=args.judge_model)
        while True:
            try:
                msg = input("\nEnter a message (or 'q' to quit): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if msg.lower() in ("q", "quit", "exit"):
                break
            if msg:
                await play_one(
                    msg, policy, judge, reward_fn, renderer, tokenizer, args.max_turns
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="final")
    parser.add_argument("--judge-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--similarity-threshold", type=float, default=0.7)
    parser.add_argument("--message", type=str, default=None)
    parser.add_argument("--human", action="store_true", help="You play as the judge")
    args = parser.parse_args()
    asyncio.run(main(args))
