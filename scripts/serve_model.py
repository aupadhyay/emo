"""Serve the trained LoRA-adapted emoji generator over HTTP on Modal.

The web frontend posts the target phrase + the conversation so far; this
endpoint returns the next emoji turn from the policy. We mirror the exact
rollout path used during training (phrase-anchored system prompt, emoji-masked
greedy/sampled generation, same chat template) so the served behavior matches
what was learned.

Local sanity:
    curl -X POST $URL/emoji \
        -H 'Content-Type: application/json' \
        -d '{"phrase": "wild goose chase", "history": []}'

Deploy:
    uv run modal deploy scripts/serve_model.py
"""

import modal

_MODEL_CACHE_DIR = "/model-cache"
_CHECKPOINT_DIR = "/checkpoints"

# Hardcoded for the demo. The good "step_200" policy from run grpo_20260430_0306
# is at the legacy root path (run pre-dates the namespacing fix). Newer runs
# will live at "<run_name>/step_N".
DEFAULT_CHECKPOINT = "step_200"

# Same temperature + max tokens used at training-time rollout (train.py).
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 20

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=5.0.0",
        "peft>=0.14.0",
        "fastapi[standard]",
        "pydantic>=2",
    )
    .add_local_python_source("src")
    .add_local_file("data/emoji-test.txt", "/root/data/emoji-test.txt")
)

_model_volume = modal.Volume.from_name("emoji-model-weights", create_if_missing=True)
_checkpoint_volume = modal.Volume.from_name("emo-checkpoints", create_if_missing=True)

app = modal.App("emo-serve")


@app.cls(
    gpu="A10G",  # cheaper than A100; this is just inference
    image=_image,
    volumes={
        _MODEL_CACHE_DIR: _model_volume,
        _CHECKPOINT_DIR: _checkpoint_volume,
    },
    min_containers=0,  # scale to zero when idle
    scaledown_window=300,  # keep warm 5 min after last request
    timeout=300,
)
class EmojiModel:
    @modal.enter()
    def load(self):
        """One-shot model load on container start."""
        import os
        os.environ["HF_HOME"] = _MODEL_CACHE_DIR

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

        from src.rl.custom.generate import MODEL_NAME, build_emoji_mask

        # Inline copy of train.py's EmojiLogitsProcessor — defined here so we
        # don't import src.rl.custom.train (which pulls wandb + sentence-transformers).
        class EmojiLogitsProcessor(LogitsProcessor):
            def __init__(self, emoji_mask):
                self.emoji_mask = emoji_mask

            def __call__(self, input_ids, scores):
                mask = self.emoji_mask.to(scores.device)
                vocab_size = scores.shape[-1]
                if mask.shape[0] < vocab_size:
                    ext = torch.zeros(
                        vocab_size - mask.shape[0], dtype=torch.bool, device=scores.device
                    )
                    mask = torch.cat([mask, ext])
                elif mask.shape[0] > vocab_size:
                    mask = mask[:vocab_size]
                return scores.masked_fill(~mask, float("-inf"))

        self._EmojiLogitsProcessor = EmojiLogitsProcessor

        ckpt = f"{_CHECKPOINT_DIR}/{DEFAULT_CHECKPOINT}"
        print(f"Loading base {MODEL_NAME} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"Attaching LoRA adapter from {ckpt} ...")
        self.model = PeftModel.from_pretrained(base, ckpt)
        self.model.eval()
        self.emoji_mask = build_emoji_mask(self.tokenizer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Ready.")

    @modal.method()
    def generate(
        self,
        phrase: str,
        history: list[dict],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Generate the next emoji turn given the phrase and prior turns.

        history: list of {"emoji": str, "guess": str} — the conversation
        so far from the *policy's* point of view. Empty for turn 1.
        """
        import torch

        from src.rl.custom.generate import build_system_prompt, format_prompt

        sys_prompt = build_system_prompt(phrase)

        if not history:
            prompt = format_prompt(phrase, self.tokenizer, sys_prompt)
        else:
            # Mirror _build_multiturn_prompt_local from train.py without
            # importing train.py (which pulls heavy deps the serving image
            # doesn't need). Keep the message structure identical or the
            # served behavior will drift from training.
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": phrase},
            ]
            for h in history:
                messages.append({"role": "assistant", "content": h["emoji"]})
                messages.append({
                    "role": "user",
                    "content": (
                        f'The player guessed: "{h["guess"]}". That\'s wrong. '
                        f'The correct phrase is still: "{phrase}". '
                        "Send more emoji to help them."
                    ),
                })
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]
        logits_processor = [self._EmojiLogitsProcessor(self.emoji_mask)]

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                logits_processor=logits_processor,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_ids = out[0, prompt_len:].tolist()
        if self.tokenizer.eos_token_id in new_ids:
            new_ids = new_ids[: new_ids.index(self.tokenizer.eos_token_id)]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


_GAMES_LOG_PATH = f"{_CHECKPOINT_DIR}/games.jsonl"


@app.function(
    image=_image,
    timeout=120,
    volumes={_CHECKPOINT_DIR: _checkpoint_volume},
)
@modal.asgi_app()
def web():
    """FastAPI app exposed at https://<account>--emo-serve-web.modal.run/"""
    import json
    from datetime import datetime, timezone

    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    class HistoryTurn(BaseModel):
        emoji: str
        guess: str

    class EmojiRequest(BaseModel):
        phrase: str = Field(min_length=1, max_length=200)
        history: list[HistoryTurn] = Field(default_factory=list, max_length=10)
        temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
        max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=64)

    class EmojiResponse(BaseModel):
        emoji: str
        checkpoint: str

    class GameLogTurn(BaseModel):
        emoji: str
        guess: str | None = None
        verdict: str | None = None

    class GameLogRequest(BaseModel):
        phrase: str = Field(min_length=1, max_length=200)
        tier: str | None = None
        turns: list[GameLogTurn] = Field(min_length=1, max_length=10)
        outcome: str = Field(pattern="^(correct|partial|wrong)$")
        score: int | None = None

    api = FastAPI(title="emo emoji generator")

    @api.get("/")
    def root():
        return {"checkpoint": DEFAULT_CHECKPOINT, "ok": True}

    @api.post("/log")
    def log(req: GameLogRequest):
        """Append a finished-game record as one JSONL row on the volume.

        The log accumulates real human gameplay (phrase, model emoji,
        human guess, verdict) which is the gold signal for the next
        round of RL training.
        """
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "checkpoint": DEFAULT_CHECKPOINT,
            **req.model_dump(),
        }
        try:
            with open(_GAMES_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            _checkpoint_volume.commit()
            return {"ok": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"log write failed: {e!s}")

    @api.post("/emoji", response_model=EmojiResponse)
    def emoji(req: EmojiRequest):
        try:
            result = EmojiModel().generate.remote(
                phrase=req.phrase,
                history=[h.model_dump() for h in req.history],
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"generation failed: {e!s}")
        return EmojiResponse(emoji=result, checkpoint=DEFAULT_CHECKPOINT)

    return api


@app.local_entrypoint()
def smoke_test(phrase: str = "wild goose chase"):
    """Call .generate() directly for a quick sanity check without going through HTTP."""
    out = EmojiModel().generate.remote(phrase=phrase, history=[])
    print(f"phrase: {phrase}\nemoji:  {out}")
