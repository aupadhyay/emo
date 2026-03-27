"""Debug script: run a few retrieval game episodes and inspect everything.

Usage:
    uv run python -m src.rl.debug_rollout --checkpoint final
    uv run python -m src.rl.debug_rollout --checkpoint rl-step-0010
"""

import argparse
import asyncio

import tinker
from dotenv import load_dotenv
from tinker_cookbook import renderers, tokenizer_utils
from src.rl.emoji_completer import EmojiTokenCompleter

from src.rl.judge import JudgeClient
from src.rl.reward import RetrievalGameReward

load_dotenv()

SENDER_SYSTEM_PROMPT = (
    "You communicate exclusively using emoji. No text, numbers, or punctuation ever. "
    "Respond with 2-8 emoji that capture the core meaning of what you need to communicate."
)

TEST_MESSAGES = [
    "I just got promoted at work!",
    "My flight got cancelled and I'm stuck at the airport",
    "Can you recommend a good Italian restaurant nearby?",
    "The sunset over the ocean was beautiful",
    "I'm so hungry I could eat a horse",
    "Happy birthday! Hope you have a great day",
    "It's raining cats and dogs outside",
    "I need to go grocery shopping",
]


async def run_episode(
    target: str,
    policy: EmojiTokenCompleter,
    judge: JudgeClient,
    reward_fn: RetrievalGameReward,
    renderer: renderers.Renderer,
    tokenizer,
    max_turns: int = 3,
):
    """Run one episode and return detailed trace."""
    emoji_history = []
    judge_guesses = []
    similarities = []

    # Turn 1: initial prompt
    messages = [
        {"role": "system", "content": SENDER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Communicate this message using only emoji: {target}"},
    ]

    for turn in range(1, max_turns + 1):
        prompt = renderer.build_generation_prompt(messages)
        stop = renderer.get_stop_sequences()

        # Sample from sender
        result = await policy(prompt, stop)
        raw_tokens = result.tokens
        decoded = tokenizer.decode(raw_tokens, skip_special_tokens=True).strip()
        emoji_history.append(decoded)

        # Judge reconstruction (pass previous guesses for multi-turn context)
        guess = await judge.reconstruct_async(emoji_history, judge_guesses=judge_guesses)
        judge_guesses.append(guess)

        # Similarity
        sim = reward_fn.embedder.similarity(target, guess)
        similarities.append(sim)

        # Check if done
        if sim >= reward_fn.similarity_threshold:
            break

        # Build next prompt with history
        messages.append({"role": "assistant", "content": decoded})
        messages.append({
            "role": "user",
            "content": (
                f'The receiver understood: "{guess}"\n'
                "Send more emoji to clarify or correct their understanding."
            ),
        })

    return {
        "target": target,
        "emoji_history": emoji_history,
        "judge_guesses": judge_guesses,
        "similarities": similarities,
        "raw_token_counts": [len(tokenizer.encode(e)) for e in emoji_history],
    }


async def main(args):
    service = tinker.ServiceClient()

    # Load sender
    tc = await service.create_lora_training_client_async(base_model="Qwen/Qwen3.5-4B", rank=32)
    tc.load_state(args.checkpoint)
    sender_sc = tc.save_weights_and_get_sampling_client(name="debug-sender")
    tokenizer = tc.get_tokenizer()
    renderer = renderers.get_renderer("qwen3_disable_thinking", tokenizer)

    policy = EmojiTokenCompleter(
        sampling_client=sender_sc,
        max_tokens=20,
        temperature=0.7,
    )

    # Load judge
    judge = JudgeClient.create(service, judge_model=args.judge_model)

    # Reward
    reward_fn = RetrievalGameReward(similarity_threshold=0.85)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Judge: {args.judge_model}")
    print(f"Max turns: {args.max_turns}")
    print("=" * 70)

    for target in TEST_MESSAGES:
        trace = await run_episode(
            target, policy, judge, reward_fn, renderer, tokenizer,
            max_turns=args.max_turns,
        )

        print(f"\nTARGET: {trace['target']}")
        for i, (emoji, guess, sim) in enumerate(
            zip(trace["emoji_history"], trace["judge_guesses"], trace["similarities"])
        ):
            status = "✓" if sim >= 0.85 else "✗"
            print(f"  Turn {i+1}: {emoji!r:40s} → Judge: {guess!r:50s} sim={sim:.3f} {status}")

        final_reward = reward_fn.compute(
            target_message=trace["target"],
            judge_final_guess=trace["judge_guesses"][-1],
            num_turns=len(trace["emoji_history"]),
            format_violation=False,
        )
        print(f"  Reward: {final_reward['reward']:.3f}")
        print("-" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="final", help="Tinker state name")
    parser.add_argument("--judge-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--max-turns", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main(args))
