"""Token completer that constrains sampling to emoji-only tokens.

Since Tinker's SamplingParams doesn't support logit masking, we use a
filter-and-resample approach:
  1. Sample from the model normally
  2. Filter out non-emoji tokens from the output
  3. Validate that the decoded result is clean (no broken Unicode)

The emoji mask comes from data/emoji_mask.npy (built by src/data/emoji_tokens.py).
"""

import numpy as np
import tinker
from tinker_cookbook.completers import StopCondition, TokenCompleter, TokensWithLogprobs


class EmojiTokenCompleter(TokenCompleter):
    """Samples from the model, then filters to emoji-only tokens."""

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        emoji_mask_path: str = "data/emoji_mask.npy",
        max_tokens: int = 20,
        temperature: float = 0.8,
        max_retries: int = 3,
    ):
        self.sampling_client = sampling_client
        self.tokenizer = sampling_client.get_tokenizer()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

        # Load emoji mask and build allowed set
        mask = np.load(emoji_mask_path)
        self.emoji_token_ids: set[int] = set(np.where(mask)[0].tolist())

    def _filter_tokens(
        self, tokens: list[int], logprobs: list[float]
    ) -> tuple[list[int], list[float]]:
        """Keep only emoji tokens, then trim trailing incomplete byte sequences."""
        # Step 1: keep only tokens in the emoji mask
        filtered_tokens = []
        filtered_logprobs = []
        for tok, lp in zip(tokens, logprobs):
            if tok in self.emoji_token_ids:
                filtered_tokens.append(tok)
                filtered_logprobs.append(lp)

        if not filtered_tokens:
            return [], []

        # Step 2: trim trailing tokens that form incomplete Unicode sequences
        # Decode and check for replacement character (U+FFFD = broken bytes)
        while filtered_tokens:
            decoded = self.tokenizer.decode(filtered_tokens, skip_special_tokens=True)
            if "\ufffd" not in decoded:
                break
            # Remove last token and try again
            filtered_tokens.pop()
            filtered_logprobs.pop()

        return filtered_tokens, filtered_logprobs

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        """Sample and filter to emoji-only tokens."""
        for attempt in range(self.max_retries):
            temp = self.temperature + (attempt * 0.2)  # increase temp on retries

            result = await self.sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    stop=stop,
                    max_tokens=self.max_tokens,
                    temperature=min(temp, 1.5),
                ),
            )

            raw_tokens = result.sequences[0].tokens
            raw_logprobs = result.sequences[0].logprobs
            assert raw_logprobs is not None

            filtered_tokens, filtered_logprobs = self._filter_tokens(
                raw_tokens, raw_logprobs
            )

            if filtered_tokens:
                return TokensWithLogprobs(
                    tokens=filtered_tokens,
                    maybe_logprobs=filtered_logprobs,
                )

        # All retries exhausted — return whatever we got (even if empty)
        # The env will flag this as a format violation
        return TokensWithLogprobs(
            tokens=raw_tokens,
            maybe_logprobs=raw_logprobs,
        )
