"""Tests for src/rl/custom/generate.py — build_emoji_mask and format_prompt."""

import pytest
import torch
from transformers import AutoTokenizer

from src.rl.custom.generate import (
    MODEL_NAME,
    build_emoji_mask,
    format_prompt,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def mask(tokenizer):
    return build_emoji_mask(tokenizer)


# ---------------------------------------------------------------------------
# build_emoji_mask tests
# ---------------------------------------------------------------------------

def test_mask_shape(mask, tokenizer):
    """Mask is torch.bool and its size is >= vocab_size."""
    assert mask.dtype == torch.bool
    assert mask.shape[0] >= tokenizer.vocab_size


def test_mask_has_emoji(mask, tokenizer):
    """All token IDs produced by tokenizing 🍕 are True in the mask."""
    ids = tokenizer.encode("🍕", add_special_tokens=False)
    assert len(ids) > 0, "tokenizer produced no tokens for 🍕"
    for tid in ids:
        assert mask[tid], f"token {tid} (from 🍕) not allowed in mask"


def test_required_signal_emoji(mask, tokenizer):
    """Spot-check that key signal emoji are allowed."""
    spot_check = ["🟢", "🔴", "✅", "❌", "⬆️", "⬇️"]
    for emoji in spot_check:
        ids = tokenizer.encode(emoji, add_special_tokens=False)
        assert len(ids) > 0, f"tokenizer produced no tokens for {emoji}"
        for tid in ids:
            assert mask[tid], f"token {tid} (from {emoji}) not allowed in mask"


def test_eos_allowed(mask, tokenizer):
    """EOS token must be allowed so generation can terminate."""
    assert mask[tokenizer.eos_token_id], "EOS token not allowed in mask"


def test_ascii_blocked(mask, tokenizer):
    """Plain ASCII letters should NOT be in the mask."""
    for ch in "abcdefghijklmnopqrstuvwxyz":
        ids = tokenizer.encode(ch, add_special_tokens=False)
        if len(ids) == 1:
            assert not mask[ids[0]], f"ASCII letter '{ch}' (token {ids[0]}) should be blocked"


def test_reasonable_count(mask, tokenizer):
    """Allowed token count should be between 100 and 20% of vocab_size."""
    n_allowed = mask.sum().item()
    vocab_size = tokenizer.vocab_size
    assert n_allowed >= 100, f"Only {n_allowed} allowed tokens — suspiciously low"
    assert n_allowed <= 0.20 * vocab_size, (
        f"{n_allowed} allowed tokens exceeds 20% of vocab ({vocab_size})"
    )


# ---------------------------------------------------------------------------
# format_prompt tests
# ---------------------------------------------------------------------------

def test_format_prompt_contains_phrase(tokenizer):
    """The user phrase should appear verbatim in the formatted prompt."""
    phrase = "birthday party"
    result = format_prompt(phrase, tokenizer)
    assert phrase in result, f"'{phrase}' not found in formatted prompt"


def test_format_prompt_is_string(tokenizer):
    """format_prompt should return a non-empty string."""
    result = format_prompt("hello", tokenizer)
    assert isinstance(result, str)
    assert len(result) > 0


def test_format_prompt_custom_system(tokenizer):
    """A custom system_prompt should appear in the formatted output."""
    result = format_prompt("test phrase", tokenizer, system_prompt="Be concise.")
    assert "Be concise." in result


def test_format_prompt_has_generation_cue(tokenizer):
    """The formatted string should contain 'assistant' (chat template generation cue)."""
    result = format_prompt("something", tokenizer)
    assert "assistant" in result.lower(), "'assistant' not found in formatted prompt"
