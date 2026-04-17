"""Multi-turn retrieval game environment for Tinker RL training.

Follows the Env / EnvGroupBuilder / RLDataset interface from tinker_cookbook.rl.types.

The sender (trainable) sees a target message and communicates via emoji.
The judge (frozen) tries to reconstruct the original from the emoji.
"""

import json
import random
from collections.abc import Sequence

import regex
import tinker
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    StepResult,
    Trajectory,
)

from src.rl.tinker.judge import JudgeClient
from src.rl.tinker.reward import RetrievalGameReward

SENDER_SYSTEM_PROMPT = (
    "You communicate exclusively using emoji. No text, numbers, or punctuation ever. "
    "Respond with 2-8 emoji that capture the core meaning of what you need to communicate."
)


class RetrievalGameEnv(Env):
    """Single episode of the multi-turn emoji retrieval game.

    Flow:
      1. initial_observation: sender sees the target message
      2. step (turn 1): sender emits emoji → judge guesses → check similarity
      3. step (turn 2+): sender sees judge's guess → emits more emoji → judge re-guesses
      4. Terminates when similarity >= threshold or max_turns reached
    """

    def __init__(
        self,
        target_message: str,
        judge: JudgeClient,
        reward_fn: RetrievalGameReward,
        renderer: renderers.Renderer,
        tokenizer: tokenizer_utils.Tokenizer,
        max_turns: int = 5,
        max_tokens_per_turn: int = 20,
    ):
        self.target_message = target_message
        self.judge = judge
        self.reward_fn = reward_fn
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.max_tokens_per_turn = max_tokens_per_turn

        # State
        self.emoji_history: list[str] = []
        self.judge_guesses: list[str] = []
        self.similarities: list[float] = []
        self.current_turn = 0
        self.format_violation = False

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        messages: list[renderers.Message] = [
            {"role": "system", "content": SENDER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Communicate this message using only emoji: {self.target_message}",
            },
        ]
        prompt = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return prompt, stop

    async def step(
        self, action: Action, *, extra: ActionExtra | None = None
    ) -> StepResult:
        self.current_turn += 1

        # Decode sender's emoji output
        decoded = self.tokenizer.decode(action, skip_special_tokens=True)
        sender_emoji = (
            decoded if isinstance(decoded, str) else " ".join(decoded)
        ).strip()
        self.emoji_history.append(sender_emoji)

        # Check format compliance
        if not _is_emoji_only(sender_emoji):
            self.format_violation = True

        # Get judge's reconstruction (pass previous guesses for multi-turn context)
        judge_guess = await self.judge.reconstruct_async(
            self.emoji_history, judge_guesses=self.judge_guesses
        )
        self.judge_guesses.append(judge_guess)

        # Compute current similarity
        similarity = self.reward_fn.embedder.similarity(
            self.target_message, judge_guess
        )
        self.similarities.append(similarity)
        success = similarity >= self.reward_fn.similarity_threshold

        # Per-turn reward: reward improvement, penalize regression
        # This teaches the model that clarification turns should help, not hurt
        if len(self.similarities) > 1:
            sim_delta = similarity - self.similarities[-2]
            step_reward = sim_delta  # positive if improved, negative if worse
        else:
            step_reward = 0.0  # first turn gets no intermediate reward

        # Check termination
        done = success or self.current_turn >= self.max_turns

        if done:
            reward_info = self.reward_fn.compute(
                target_message=self.target_message,
                judge_final_guess=judge_guess,
                num_turns=self.current_turn,
                format_violation=self.format_violation,
            )
            # Terminal reward = final composite reward + last step delta
            next_obs, next_stop = await self._build_next_observation()
            return StepResult(
                reward=reward_info["reward"] + step_reward,
                episode_done=True,
                next_observation=next_obs,
                next_stop_condition=next_stop,
                metrics={
                    "similarity": similarity,
                    "success": float(success),
                    "num_turns": self.current_turn,
                    "format_violation": float(self.format_violation),
                },
            )
        else:
            # Continue: show sender the judge's guess so it can clarify
            next_obs, next_stop = await self._build_next_observation()
            return StepResult(
                reward=step_reward,
                episode_done=False,
                next_observation=next_obs,
                next_stop_condition=next_stop,
                metrics={
                    "turn": self.current_turn,
                    "similarity": similarity,
                    "sim_delta": sim_delta if len(self.similarities) > 1 else 0.0,
                },
            )

    async def _build_next_observation(self) -> tuple[Observation, StopCondition]:
        """Build the next prompt for the sender, including conversation history."""
        messages: list[renderers.Message] = [
            {"role": "system", "content": SENDER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Communicate this message using only emoji: {self.target_message}",
            },
        ]
        # Add conversation history
        for i in range(len(self.emoji_history)):
            messages.append({"role": "assistant", "content": self.emoji_history[i]})
            if i < len(self.judge_guesses):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f'The receiver guessed: "{self.judge_guesses[i]}"\n'
                            f'The original message was: "{self.target_message}"\n'
                            "Send emoji to correct what they got wrong."
                        ),
                    }
                )

        prompt = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return prompt, stop


class RetrievalGameGroupBuilder(EnvGroupBuilder):
    """Builds a group of identical environments (same target message) for GRPO.

    Each env in the group receives a different sampled rollout, enabling
    group-relative advantage computation.
    """

    def __init__(
        self,
        target_message: str,
        judge: JudgeClient,
        reward_fn: RetrievalGameReward,
        renderer: renderers.Renderer,
        tokenizer: tokenizer_utils.Tokenizer,
        group_size: int = 8,
        max_turns: int = 5,
        max_tokens_per_turn: int = 20,
    ):
        self.target_message = target_message
        self.judge = judge
        self.reward_fn = reward_fn
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.max_turns = max_turns
        self.max_tokens_per_turn = max_tokens_per_turn

    async def make_envs(self) -> Sequence[Env]:
        return [
            RetrievalGameEnv(
                target_message=self.target_message,
                judge=self.judge,
                reward_fn=self.reward_fn,
                renderer=self.renderer,
                tokenizer=self.tokenizer,
                max_turns=self.max_turns,
                max_tokens_per_turn=self.max_tokens_per_turn,
            )
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """No additional group-level reward — per-step rewards are sufficient."""
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ["retrieval_game"]


class RetrievalGameDataset(RLDataset):
    """Dataset of target messages for the retrieval game."""

    def __init__(
        self,
        prompts_path: str,
        judge: JudgeClient,
        reward_fn: RetrievalGameReward,
        renderer: renderers.Renderer,
        tokenizer: tokenizer_utils.Tokenizer,
        batch_size: int = 32,
        group_size: int = 8,
        max_turns: int = 5,
        max_tokens_per_turn: int = 20,
        shuffle: bool = True,
    ):
        with open(prompts_path) as f:
            self.prompts = [json.loads(line)["text"] for line in f]
        if shuffle:
            random.shuffle(self.prompts)

        self.judge = judge
        self.reward_fn = reward_fn
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.group_size = group_size
        self.max_turns = max_turns
        self.max_tokens_per_turn = max_tokens_per_turn

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = (index * self.batch_size) % len(self.prompts)
        batch_prompts = [
            self.prompts[(start + i) % len(self.prompts)]
            for i in range(self.batch_size)
        ]
        return [
            RetrievalGameGroupBuilder(
                target_message=p,
                judge=self.judge,
                reward_fn=self.reward_fn,
                renderer=self.renderer,
                tokenizer=self.tokenizer,
                group_size=self.group_size,
                max_turns=self.max_turns,
                max_tokens_per_turn=self.max_tokens_per_turn,
            )
            for p in batch_prompts
        ]

    def __len__(self) -> int:
        return len(self.prompts) // self.batch_size


def _is_emoji_only(text: str) -> bool:
    """Check if text contains only emoji characters (and whitespace/joiners)."""
    cleaned = (
        text.replace(" ", "")
        .replace("\u200d", "")
        .replace("\ufe0f", "")
        .replace("\ufe0e", "")
    )
    if not cleaned:
        return False
    for char in cleaned:
        if not regex.match(r"\p{Emoji}", char) and not regex.match(
            r"[\U0001F3FB-\U0001F3FF]", char
        ):
            return False
    return True
