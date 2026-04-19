"""Simulated guesser and multi-turn episode runner for the emoji communication game."""

import asyncio
import logging
import re
from dataclasses import dataclass, field

import anthropic

from src.rl.custom.reward import SimilarityScorer

logger = logging.getLogger(__name__)

_CASUAL_SYSTEM_PROMPT = """\
You are playing an emoji guessing game. Someone is trying to communicate a word or short phrase using only emoji. Look at the emoji and guess what they're trying to say.

Rules:
- Give your best guess as a short phrase (1-5 words)
- Respond with ONLY your guess, nothing else. No explanations, no "I think it's...", just the phrase.
- Guess like a normal person would — don't overthink it or try to find hidden meanings.
- If you've been told your previous guesses were wrong, try a different interpretation.\
"""

_NAIVE_SYSTEM_PROMPT = """\
You are playing an emoji guessing game. Someone is trying to communicate a word or short phrase using only emoji. Look at each emoji literally and guess what they're trying to say.

Rules:
- Give your best guess as a short phrase (1-5 words)
- Respond with ONLY your guess, nothing else. No explanations, just the phrase.
- Interpret each emoji as literally as possible — don't look for abstract meanings.
- If you've been told your previous guesses were wrong, try a different literal interpretation.\
"""

_DIFFICULTY_PROMPTS = {
    "casual": _CASUAL_SYSTEM_PROMPT,
    "naive": _NAIVE_SYSTEM_PROMPT,
}

_PREAMBLE_PATTERN = re.compile(
    r"^(i think (it'?s?|this is|this means)|it'?s?|this is|this means|my guess is|i'd guess|i would guess)\s*[:\-]?\s*",
    re.IGNORECASE,
)


def _clean_guess(raw: str) -> tuple[str, bool]:
    """Strip preamble/quotes from a guesser response. Returns (cleaned, was_cleaned)."""
    text = raw.strip().strip('"\'').strip()
    cleaned = _PREAMBLE_PATTERN.sub("", text).strip().strip('"\'').strip()
    if len(cleaned.split()) > 10:
        short = re.split(r"[,.]", cleaned)[0].strip()
        words = short.split()
        cleaned = " ".join(words[:5])
        return cleaned, True
    was_cleaned = cleaned != raw.strip()
    return cleaned, was_cleaned


def _build_guesser_messages(
    turn_history: list[tuple[str, str]],
    current_emoji: str,
) -> list[dict]:
    """Build a real multi-turn message thread for the guesser.

    turn_history: list of (emoji, guess) from previous turns.
    Returns an Anthropic messages list with the current emoji as the final user turn.
    """
    messages = []
    for emoji, guess in turn_history:
        messages.append({"role": "user", "content": emoji})
        messages.append({"role": "assistant", "content": guess})
        messages.append({"role": "user", "content": f'Wrong. Here are more emoji to help you guess:'})
    messages.append({"role": "user", "content": current_emoji})
    # Flatten: the "Wrong..." message and next emoji should be one user turn
    # Actually structure it properly: wrong + new emoji as a single user message
    if turn_history:
        messages = []
        for emoji, guess in turn_history:
            messages.append({"role": "user", "content": emoji})
            messages.append({"role": "assistant", "content": guess})
        messages.append({
            "role": "user",
            "content": f"Wrong. Here are more emoji to help you guess:\n{current_emoji}",
        })
    return messages


@dataclass
class Turn:
    turn_number: int
    emoji_output: str
    guess: str
    similarity: float


@dataclass
class Episode:
    target_phrase: str
    turns: list[Turn] = field(default_factory=list)
    completed: bool = False
    completion_turn: int | None = None


class SimulatedGuesser:
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        difficulty: str = "casual",
        conversation_mode: bool = True,
    ):
        """
        Args:
            conversation_mode: If True, maintain real multi-turn message history with
                the guesser instead of appending previous guesses as plain text.
                In conversation mode, guess() requires turn_history to be passed.
        """
        self.model = model
        self.system_prompt = _DIFFICULTY_PROMPTS[difficulty]
        self.conversation_mode = conversation_mode
        self._client = anthropic.Anthropic()
        self._async_client = anthropic.AsyncAnthropic()

    def _build_messages(
        self,
        emoji_sequence: str,
        previous_guesses: list[str] | None,
        turn_history: list[tuple[str, str]] | None,
    ) -> list[dict]:
        if self.conversation_mode and turn_history:
            return _build_guesser_messages(turn_history, emoji_sequence)
        # Stateless mode: append previous guesses as text
        user_content = emoji_sequence
        if previous_guesses:
            prior = ", ".join(f'"{g}"' for g in previous_guesses)
            user_content += f"\n\n(Previous wrong guesses: {prior})"
        return [{"role": "user", "content": user_content}]

    def guess(
        self,
        emoji_sequence: str,
        previous_guesses: list[str] | None = None,
        turn_history: list[tuple[str, str]] | None = None,
    ) -> str:
        messages = self._build_messages(emoji_sequence, previous_guesses, turn_history)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=64,
            system=self.system_prompt,
            messages=messages,
        )
        raw = response.content[0].text
        cleaned, was_cleaned = _clean_guess(raw)
        if was_cleaned:
            logger.warning("Guesser response needed cleaning: %r -> %r", raw, cleaned)
        return cleaned

    def guess_batch(self, episodes: list[dict]) -> list[str]:
        return asyncio.run(self._guess_batch_async(episodes))

    async def _guess_batch_async(self, episodes: list[dict]) -> list[str]:
        tasks = [self._guess_one_async(ep) for ep in episodes]
        return await asyncio.gather(*tasks)

    async def _guess_one_async(self, episode: dict) -> str:
        emoji = episode["emoji"]
        previous_guesses = episode.get("previous_guesses") or []
        turn_history = episode.get("turn_history") or []

        messages = self._build_messages(
            emoji,
            previous_guesses if previous_guesses else None,
            turn_history if turn_history else None,
        )
        response = await self._async_client.messages.create(
            model=self.model,
            max_tokens=64,
            system=self.system_prompt,
            messages=messages,
        )
        raw = response.content[0].text
        cleaned, was_cleaned = _clean_guess(raw)
        if was_cleaned:
            logger.warning("Guesser response needed cleaning: %r -> %r", raw, cleaned)
        return cleaned


def _build_multiturn_prompt(
    target_phrase: str,
    history: list[Turn],
    system_prompt: str,
    tokenizer,
) -> str:
    """Build a multi-turn chat prompt for the emoji model including conversation history."""
    messages = [{"role": "user", "content": f'Communicate this phrase using only emoji: "{target_phrase}"'}]
    for turn in history:
        messages.append({"role": "assistant", "content": turn.emoji_output})
        messages.append({
            "role": "user",
            "content": f'The player guessed: "{turn.guess}". That\'s wrong. Send more emoji to help them guess correctly.',
        })
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def run_episode(
    generator,
    guesser: SimulatedGuesser,
    scorer: SimilarityScorer,
    target_phrase: str,
    max_turns: int = 5,
    system_prompt: str = None,
    exact_match_threshold: float = 0.85,
) -> Episode:
    """Run a single multi-turn episode of the emoji communication game."""
    from src.rl.custom.generate import DEFAULT_SYSTEM_PROMPT, MODEL_NAME, format_prompt
    from transformers import AutoTokenizer

    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    episode = Episode(target_phrase=target_phrase)
    history: list[Turn] = []

    for turn_num in range(1, max_turns + 1):
        if turn_num == 1:
            prompt = format_prompt(target_phrase, tokenizer, system_prompt)
        else:
            prompt = _build_multiturn_prompt(target_phrase, history, system_prompt, tokenizer)

        results = generator.generate.remote([prompt], n=1, temperature=1.0, max_tokens=20)
        emoji_output = results[0][0]

        previous_guesses = [t.guess for t in history]
        turn_history = [(t.emoji_output, t.guess) for t in history]
        guess = guesser.guess(
            emoji_output,
            previous_guesses=previous_guesses if previous_guesses else None,
            turn_history=turn_history if turn_history else None,
        )
        similarity = scorer.score(target_phrase, guess)

        turn = Turn(
            turn_number=turn_num,
            emoji_output=emoji_output,
            guess=guess,
            similarity=similarity,
        )
        history.append(turn)
        episode.turns.append(turn)

        if similarity >= exact_match_threshold:
            episode.completed = True
            episode.completion_turn = turn_num
            break

    return episode


def run_episode_batch(
    generator,
    guesser: SimulatedGuesser,
    scorer: SimilarityScorer,
    target_phrases: list[str],
    n_per_phrase: int = 8,
    max_turns: int = 5,
) -> dict[str, list[Episode]]:
    """Run multiple episodes per phrase for GRPO groups."""
    results: dict[str, list[Episode]] = {phrase: [] for phrase in target_phrases}
    for phrase in target_phrases:
        for _ in range(n_per_phrase):
            episode = run_episode(
                generator=generator,
                guesser=guesser,
                scorer=scorer,
                target_phrase=phrase,
                max_turns=max_turns,
            )
            results[phrase].append(episode)
    return results
