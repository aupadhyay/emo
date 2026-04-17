"""Frozen judge model for the emoji retrieval game.

The judge sees emoji sequences and tries to reconstruct the original message.
Uses a separate Tinker SamplingClient — no weights are updated.
"""

import re

import tinker
from tinker_cookbook import renderers, tokenizer_utils

JUDGE_SYSTEM_PROMPT = """\
You are playing a communication game. Another AI was given an everyday English \
message (like something you'd text a friend) and encoded it using only emoji. \
You see the emoji and must guess the original message.

The original messages are things like: greetings, questions, complaints, \
requests, opinions, descriptions of events, or emotional statements.

Rules:
- Write ONE short sentence (under 15 words) guessing the original message.
- Think about what someone would actually say, not just what the emoji depict.
- Example: 🎂🎉🎈🎁 → "Happy birthday!"
- Example: ✈️❌😩🏨 → "My flight got cancelled and I'm stuck at a hotel"
- Do NOT repeat the emoji, do NOT add quotes or labels."""


def _clean_judge_output(text: str) -> str:
    """Strip special tokens, meta-commentary, and formatting artifacts from judge output."""
    # Remove special tokens that leak through
    text = re.sub(r"<\|[^|]+\|>", "", text)
    # Remove bold/markdown formatting
    text = re.sub(r"\*\*.*?\*\*:?\s*", "", text)
    # Remove common prefixes / meta-commentary the judge adds
    text = re.sub(
        r"^(The original message was:?\s*|"
        r"The sender (is |sent ).*?:\s*|"
        r"I think the original message was:?\s*|"
        r"My (best |updated )?guess:?\s*|"
        r"(Updated |New |Final )(guess|answer|chat|conversation):?\s*|"
        r"Assistant:\s*|"
        r"You're (playing|welcome).*?:\s*|"
        r"I need to (guess|analyze|reconsider).*?:\s*|"
        r"Your message is empty.*|"
        r"Please provide.*)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Remove surrounding quotes
    text = text.strip().strip('"').strip("'").strip()
    # If the response is multi-line, take only the first meaningful line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        text = lines[0]
    # If the output is mostly emoji (judge echoed input), return empty
    # so the caller can handle it
    non_emoji = re.sub(
        r"[\s\U0001F000-\U0001FFFF\u200d\ufe0f\u2600-\u27BF\u2300-\u23FF]+", "", text
    )
    if len(non_emoji) < 3 and len(text) > 0:
        return ""
    return text.strip()


class JudgeClient:
    """Frozen judge that reconstructs messages from emoji."""

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        judge_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    ):
        self.sampling_client = sampling_client
        self.tokenizer = tokenizer_utils.get_tokenizer(judge_model)
        self.renderer = renderers.get_renderer("qwen3_instruct", self.tokenizer)
        self.sampling_params = tinker.SamplingParams(
            max_tokens=40,
            temperature=0.3,
        )

    @classmethod
    def create(
        cls,
        service_client: tinker.ServiceClient,
        judge_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    ) -> "JudgeClient":
        sampling_client = service_client.create_sampling_client(base_model=judge_model)
        return cls(sampling_client=sampling_client, judge_model=judge_model)

    def _build_messages(
        self,
        emoji_history: list[str],
        judge_guesses: list[str] | None = None,
    ) -> list[renderers.Message]:
        """Build the multi-turn conversation for the judge.

        Includes the judge's own previous guesses as assistant turns so it can
        see what it said before and refine.
        """
        messages: list[renderers.Message] = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        ]
        if judge_guesses is None:
            judge_guesses = []

        for i, emoji_seq in enumerate(emoji_history):
            if i == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The sender sent this emoji message: {emoji_seq}\n\n"
                            "What do you think the original message was?"
                        ),
                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your guess was WRONG. The sender sent more emoji to help you: {emoji_seq}\n\n"
                            "Your previous guess was not close enough. Think about what else "
                            "these emoji could mean and make a DIFFERENT guess."
                        ),
                    }
                )

            # Add the judge's own previous guess as an assistant turn
            # (except for the last emoji, which is what we're about to generate for)
            if i < len(judge_guesses):
                messages.append(
                    {
                        "role": "assistant",
                        "content": judge_guesses[i],
                    }
                )

        return messages

    def _extract_response(self, tokens: list[int]) -> str:
        """Parse the judge's response tokens and clean up."""
        response_msg, success = self.renderer.parse_response(tokens)
        if success:
            raw = response_msg["content"]
        else:
            raw = self.tokenizer.decode(tokens, skip_special_tokens=True)
        cleaned = _clean_judge_output(str(raw))
        # If cleaning returned empty (judge echoed emoji or gave garbage),
        # return a placeholder that will score low in similarity
        if not cleaned:
            return "[unknown]"
        return cleaned

    def reconstruct(
        self,
        emoji_history: list[str],
        judge_guesses: list[str] | None = None,
    ) -> str:
        """Ask the judge to reconstruct the original message from emoji history.

        Args:
            emoji_history: list of emoji sequences, one per sender turn.
            judge_guesses: list of the judge's previous guesses (for multi-turn context).

        Returns:
            Judge's best guess as a string.
        """
        messages = self._build_messages(emoji_history, judge_guesses)
        prompt = self.renderer.build_generation_prompt(messages)

        output = self.sampling_client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=self.sampling_params,
        ).result()

        return self._extract_response(output.sequences[0].tokens)

    async def reconstruct_async(
        self,
        emoji_history: list[str],
        judge_guesses: list[str] | None = None,
    ) -> str:
        """Async version of reconstruct."""
        messages = self._build_messages(emoji_history, judge_guesses)
        prompt = self.renderer.build_generation_prompt(messages)

        output = await self.sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=self.sampling_params,
        )

        return self._extract_response(output.sequences[0].tokens)
