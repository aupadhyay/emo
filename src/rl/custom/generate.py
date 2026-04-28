"""Constrained emoji generation with vLLM on Modal."""

import urllib.request
from pathlib import Path

import modal
import torch
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

DEFAULT_SYSTEM_PROMPT = (
    "You are an emoji communicator. Respond using only emoji to convey the meaning "
    "of the user's message. No text, numbers, or punctuation."
)

EMOJI_TEST_URL = "https://unicode.org/Public/emoji/latest/emoji-test.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_emoji_sequences(cache_path: str = "data/emoji-test.txt") -> list[str]:
    """Download and parse all emoji sequences from Unicode emoji-test.txt."""
    cache = Path(cache_path)
    if not cache.exists():
        print(f"Downloading {EMOJI_TEST_URL} ...")
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(EMOJI_TEST_URL, cache)

    text = cache.read_text(encoding="utf-8")
    sequences: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: "1F600 ; fully-qualified # 😀 E1.0 grinning face"
        codepoints_str, _ = line.split(";", 1)
        codepoints = codepoints_str.strip().split()
        emoji = "".join(chr(int(cp, 16)) for cp in codepoints)
        sequences.append(emoji)

    return sequences


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_ASCII_PUNCT = set('.,!?;:\'"()[]{}|\\#@$%^&*<>/`~+=-_')


def build_emoji_mask(tokenizer) -> torch.Tensor:
    """Return a boolean torch.Tensor mask of allowed token IDs for emoji-only generation.

    Downloads/caches Unicode emoji-test.txt, tokenizes every emoji sequence
    (standalone and with a leading space), adds EOS, then builds a boolean mask
    over the full vocabulary. Whitespace tokens and tokens that decode to ASCII
    alphanumeric/punctuation are explicitly excluded so the model cannot leak
    plain text even at high temperature.

    Args:
        tokenizer: A HuggingFace tokenizer (e.g. loaded via AutoTokenizer).

    Returns:
        torch.Tensor of dtype torch.bool with shape (mask_size,), where
        mask_size = max(vocab_size, max(emoji_token_ids) + 1).
    """
    vocab_size: int = tokenizer.vocab_size
    sequences = _load_emoji_sequences()

    emoji_token_ids: set[int] = set()

    # Tokenize every Unicode emoji sequence, standalone and with leading space.
    # The leading-space variant captures tokens that tokenizers encode differently
    # at word boundaries (common in BPE models).
    for emoji_str in sequences:
        for text in (emoji_str, " " + emoji_str):
            ids = tokenizer.encode(text, add_special_tokens=False)
            emoji_token_ids.update(ids)

    # Always allow EOS so generation can terminate
    if tokenizer.eos_token_id is not None:
        emoji_token_ids.add(tokenizer.eos_token_id)

    # Remove tokens that decode to whitespace or ASCII text/punctuation.
    # Without this filter, space tokens introduced by the " " + emoji_str
    # encodings leak through, and some byte-fallback tokens overlap with ASCII.
    eos = tokenizer.eos_token_id
    bad_ids: set[int] = set()
    for tid in emoji_token_ids:
        if tid == eos:
            continue
        decoded = tokenizer.decode([tid]).strip()
        if not decoded:  # pure whitespace token
            bad_ids.add(tid)
        elif any(ch.isascii() and (ch.isalnum() or ch in _ASCII_PUNCT) for ch in decoded):
            bad_ids.add(tid)

    if bad_ids:
        examples = [tokenizer.decode([t]) for t in list(bad_ids)[:10]]
        print(f"Removing {len(bad_ids)} non-emoji tokens from mask, e.g.: {examples}")
    emoji_token_ids -= bad_ids

    # Build the mask
    mask_size = max(vocab_size, max(emoji_token_ids) + 1)
    mask = torch.zeros(mask_size, dtype=torch.bool)
    for tid in emoji_token_ids:
        mask[tid] = True

    n_allowed = len(emoji_token_ids)
    pct = 100.0 * n_allowed / vocab_size
    print(f"Emoji mask: {n_allowed} allowed / {vocab_size} vocab ({pct:.1f}%)")

    return mask


def format_prompt(
    phrase: str,
    tokenizer,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Format a user phrase into a chat-template prompt ready for generation.

    Args:
        phrase: The user's input text to translate into emoji.
        tokenizer: A HuggingFace tokenizer with apply_chat_template support.
        system_prompt: System-role instruction; defaults to DEFAULT_SYSTEM_PROMPT.

    Returns:
        A formatted string (including special tokens / role markers) produced
        by the tokenizer's chat template with add_generation_prompt=True.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": phrase},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Modal app setup
# ---------------------------------------------------------------------------

_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "vllm>=0.6.0", "transformers>=4.45.0", "torch>=2.4.0"
)

_volume = modal.Volume.from_name("emoji-model-weights", create_if_missing=True)
_MODEL_CACHE_DIR = "/model-cache"

app = modal.App("emoji-generator")


@app.cls(
    gpu="A10G",
    image=_image,
    volumes={_MODEL_CACHE_DIR: _volume},
    timeout=600,
)
class EmojiGenerator:
    model_name: str = modal.parameter(default=MODEL_NAME)

    @modal.enter()
    def load(self):
        import os
        from vllm import LLM  # ty: ignore[unresolved-import]
        from transformers import AutoTokenizer

        os.environ["HF_HOME"] = _MODEL_CACHE_DIR
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=_MODEL_CACHE_DIR
        )
        self.llm = LLM(
            model=self.model_name,
            download_dir=_MODEL_CACHE_DIR,
            dtype="bfloat16",
        )
        self.emoji_mask = build_emoji_mask(self.tokenizer)
        # Pre-compute allowed token ID list for vLLM's allowed_token_ids param
        self.emoji_token_ids = self.emoji_mask.nonzero(as_tuple=False).squeeze(1).tolist()

    @modal.method()
    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        temperature: float = 1.0,
        max_tokens: int = 20,
    ) -> list[list[str]]:
        """Generate emoji-only outputs. Returns list[list[str]]: outer=per-prompt, inner=n completions."""
        from vllm import SamplingParams  # ty: ignore[unresolved-import]

        params = SamplingParams(
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            allowed_token_ids=self.emoji_token_ids,
        )

        outputs = self.llm.generate(prompts, params)

        results = []
        violations = 0
        for req_output in outputs:
            completions = []
            for completion in req_output.outputs:
                text = completion.text.strip()
                # Strip leading/trailing bare variation selectors (U+FE0F / U+FE0E).
                # The model sometimes generates them as the first token when a ZWJ
                # emoji sequence is split; they're meaningless without a base glyph.
                text = text.strip('\ufe0f\ufe0e')
                if any(
                    c.isascii() and (c.isalpha() or c.isdigit() or c in _ASCII_PUNCT)
                    for c in text
                ) or any(c.isspace() for c in text):
                    violations += 1
                completions.append(text)
            results.append(completions)

        if violations:
            print(
                f"WARNING: {violations} format violations in {sum(len(r) for r in results)} outputs"
            )
        return results

    @modal.method()
    def generate_unconstrained(
        self,
        prompts: list[str],
        max_tokens: int = 60,
    ) -> list[str]:
        """Unconstrained generation for before/after comparison."""
        from vllm import SamplingParams  # ty: ignore[unresolved-import]

        params = SamplingParams(temperature=0.7, max_tokens=max_tokens)
        outputs = self.llm.generate(prompts, params)
        return [out.outputs[0].text.strip() for out in outputs]


if __name__ == "__main__":
    _VALIDATION_PHRASES = [
        "pizza",
        "birthday party",
        "road trip",
        "feeling nostalgic",
        "the economy is struggling",
        "a dog playing in the park",
        "thunderstorm",
        "graduation day",
        "cooking dinner",
        "space exploration",
    ]
    _PUNCT = set('.,!?;:\'"()[]{}|\\#@$%^&*<>/`~+=-_')

    def _assert_clean(text: str, label: str) -> None:
        assert not any(ch.isascii() and ch.isalpha() for ch in text), \
            f"ASCII letter in [{label}]: {text!r}"
        assert not any(ch.isdigit() for ch in text), \
            f"Digit in [{label}]: {text!r}"
        assert not any(ch.isspace() for ch in text), \
            f"Whitespace in [{label}]: {text!r}"
        assert not text.startswith('\ufe0f'), \
            f"Leading VS16 in [{label}]: {text!r}"
        assert not any(ch in _PUNCT for ch in text), \
            f"Punctuation in [{label}]: {text!r}"

    from transformers import AutoTokenizer as _Tok

    _tok = _Tok.from_pretrained(MODEL_NAME)

    with modal.enable_output():
        gen = modal.Cls.from_name("emoji-generator", "EmojiGenerator")()
        prompts = [format_prompt(p, _tok) for p in _VALIDATION_PHRASES]

        for temp in [0.7, 1.0, 1.3, 1.5]:
            results = gen.generate.remote(prompts, n=10, temperature=temp, max_tokens=20)
            total = 0
            for phrase, completions in zip(_VALIDATION_PHRASES, results):
                for c in completions:
                    _assert_clean(c, f"{phrase} @ temp={temp}")
                    total += 1
            print(f"temp={temp}: all {total} outputs clean ✓")

        print("\nSample outputs (temp=1.0):")
        results = gen.generate.remote(prompts, n=4, temperature=1.0, max_tokens=20)
        for phrase, completions in zip(_VALIDATION_PHRASES, results):
            print(f"\n{phrase}:")
            for i, c in enumerate(completions, 1):
                print(f"  {i}. {c}")
