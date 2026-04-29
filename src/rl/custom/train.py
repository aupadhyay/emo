"""GRPO training loop for the emoji communication game."""

import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import wandb

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

from src.rl.custom.env import Episode, SimulatedGuesser, Turn
from src.rl.custom.generate import (
    DEFAULT_SYSTEM_PROMPT,
    MODEL_NAME,
    build_emoji_mask,
    format_prompt,
)
from src.rl.custom.reward import (
    SimilarityScorer,
    compute_group_advantages,
    compute_repetition_penalty,
    compute_turn_rewards,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Emoji logits processor
# ---------------------------------------------------------------------------


class EmojiLogitsProcessor(LogitsProcessor):
    """Mask all non-emoji tokens to -inf during generation."""

    def __init__(self, emoji_mask: torch.Tensor):
        self.emoji_mask = emoji_mask

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
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


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def setup_models(
    model_name: str = MODEL_NAME,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_target_modules: list[str] | None = None,
) -> tuple[Any, Any, Any]:
    """Load policy model (LoRA-wrapped) and frozen reference model.

    Returns:
        (policy_model, ref_model, tokenizer)
    """
    tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
    assert tokenizer is not None

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Policy model: base + LoRA adapter
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.0,
        bias="none",
    )
    policy_model = get_peft_model(base, lora_cfg)
    policy_model.print_trainable_parameters()

    # Reference model: separate copy, fully frozen
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    return policy_model, ref_model, tokenizer


# ---------------------------------------------------------------------------
# Multi-turn sequence helpers
# ---------------------------------------------------------------------------


def _build_multiturn_prompt_local(
    phrase: str,
    history: list[Turn],
    system_prompt: str,
    tokenizer: Any,
) -> str:
    """Build chat prompt for turn > 1, consistent with format_prompt for turn 1.

    Uses just `phrase` as the initial user message (matching format_prompt) so
    that rollout context exactly matches the training sequence in build_multiturn_sequence.
    """
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": phrase},
    ]
    for turn in history:
        messages.append({"role": "assistant", "content": turn.emoji_output})
        messages.append(
            {
                "role": "user",
                "content": f'The player guessed: "{turn.guess}". That\'s wrong. Send more emoji to help them guess correctly.',
            }
        )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_multiturn_sequence(
    episode: Episode,
    tokenizer: Any,
    system_prompt: str,
) -> tuple[list[int], list[int]]:
    """Build full tokenized episode sequence and response mask for GRPO training.

    Constructs the full conversation for all turns in the episode and finds
    the token positions of each assistant turn by comparing tokenized prefixes.
    All assistant turns get gradient; user/system tokens are masked out.

    Returns (input_ids, response_mask) where response_mask[i]=1 for assistant tokens.
    """
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": episode.target_phrase},
    ]
    for i, turn in enumerate(episode.turns):
        messages.append({"role": "assistant", "content": turn.emoji_output})
        # Add feedback after all turns except the last
        if i < len(episode.turns) - 1:
            messages.append(
                {
                    "role": "user",
                    "content": f'The player guessed: "{turn.guess}". That\'s wrong. Send more emoji to help them guess correctly.',
                }
            )

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    response_mask = [0] * len(full_ids)

    # Find each assistant turn's token range by diffing prefix lengths.
    # prefix (with add_generation_prompt=True) ends at the opening of the assistant
    # turn header; end_text covers up through the closing <|im_end|> of that turn.
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix_text = tokenizer.apply_chat_template(
            messages[:i], tokenize=False, add_generation_prompt=True
        )
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        end_text = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=False, add_generation_prompt=False
        )
        end_ids = tokenizer.encode(end_text, add_special_tokens=False)
        for j in range(len(prefix_ids), min(len(end_ids), len(response_mask))):
            response_mask[j] = 1

    return full_ids, response_mask


# ---------------------------------------------------------------------------
# GRPO loss
# ---------------------------------------------------------------------------


def compute_grpo_loss(
    policy_model: Any,
    ref_model: Any,
    tokenizer: Any,
    full_ids: torch.Tensor,  # (B, L)
    response_mask: torch.Tensor,  # (B, L) float — 1 for assistant tokens
    attention_mask: torch.Tensor,  # (B, L) — 1 for non-padding
    advantages: torch.Tensor,  # (B,)
    kl_coeff: float = 0.05,
) -> dict[str, Any]:
    """Compute GRPO policy-gradient loss for one group of rollouts.

    Works for both single-turn and multi-turn episodes. response_mask marks ALL
    assistant tokens across all turns in the episode as trainable targets.

    Returns a dict with keys: loss, pg_loss, kl_loss, mean_kl, mean_logprob.
    """
    # Forward through policy (with gradient)
    logits_policy = policy_model(full_ids, attention_mask=attention_mask).logits

    # Forward through ref (no gradient — we only need log probs)
    with torch.no_grad():
        logits_ref = ref_model(full_ids, attention_mask=attention_mask).logits

    # Shift for next-token prediction: logits[t] predicts token[t+1]
    shift_logits_policy = logits_policy[:, :-1, :]  # (B, L-1, V)
    shift_logits_ref = logits_ref[:, :-1, :]
    shift_labels = full_ids[:, 1:]  # (B, L-1)
    shift_mask = response_mask[:, 1:]  # (B, L-1)

    log_probs_policy = F.log_softmax(shift_logits_policy, dim=-1)
    log_probs_ref = F.log_softmax(shift_logits_ref, dim=-1)

    # Per-token log probs for the actual tokens that were sampled
    token_lp_policy = log_probs_policy.gather(2, shift_labels.unsqueeze(-1)).squeeze(
        -1
    )  # (B, L-1)
    token_lp_ref = log_probs_ref.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Sum log probs over response tokens → one scalar per trajectory
    sum_lp_policy = (token_lp_policy * shift_mask).sum(dim=-1)  # (B,)

    # Policy gradient loss: -E[advantage * sum_log_prob]
    pg_loss = -(advantages * sum_lp_policy).mean()

    # KL(policy || ref) approximation: exp(r) - r - 1, where r = log(policy/ref)
    log_ratio = token_lp_policy - token_lp_ref
    kl_per_tok = torch.exp(log_ratio) - log_ratio - 1  # (B, L-1)
    kl_loss = (kl_per_tok * shift_mask).sum(dim=-1).mean()  # scalar

    n_resp_toks = shift_mask.sum().clamp(min=1)
    mean_kl = (kl_per_tok * shift_mask).sum() / n_resp_toks
    mean_logprob = sum_lp_policy.mean()

    return {
        "loss": pg_loss + kl_coeff * kl_loss,
        "pg_loss": pg_loss,
        "kl_loss": kl_loss,
        "mean_kl": mean_kl.item(),
        "mean_logprob": mean_logprob.item(),
    }


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def train_step(
    policy_model: Any,
    ref_model: Any,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    episodes: list[Episode],
    advantages: list[float],
    kl_coeff: float = 0.05,
    max_grad_norm: float = 1.0,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Execute one GRPO gradient step on a group of episodes.

    Supports single-turn and multi-turn: builds the full conversation sequence
    for each episode and masks ALL assistant turns as training targets.

    Returns dict of metrics: loss, pg_loss, kl_loss, mean_kl, grad_norm.
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_id = tokenizer.pad_token_id

    sequences = [
        build_multiturn_sequence(ep, tokenizer, system_prompt) for ep in episodes
    ]
    max_len = max(len(s[0]) for s in sequences)
    B = len(episodes)

    full_ids_tensor = torch.full((B, max_len), pad_id, dtype=torch.long)
    response_mask_tensor = torch.zeros(B, max_len, dtype=torch.float)
    attention_mask_tensor = torch.zeros(B, max_len, dtype=torch.long)

    for i, (input_ids, resp_mask) in enumerate(sequences):
        L = len(input_ids)
        full_ids_tensor[i, :L] = torch.tensor(input_ids, dtype=torch.long)
        response_mask_tensor[i, :L] = torch.tensor(resp_mask, dtype=torch.float)
        attention_mask_tensor[i, :L] = 1

    full_ids_tensor = full_ids_tensor.to(device)
    response_mask_tensor = response_mask_tensor.to(device)
    attention_mask_tensor = attention_mask_tensor.to(device)
    adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)

    policy_model.train()

    loss_dict = compute_grpo_loss(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        full_ids=full_ids_tensor,
        response_mask=response_mask_tensor,
        attention_mask=attention_mask_tensor,
        advantages=adv_tensor,
        kl_coeff=kl_coeff,
    )

    loss_dict["loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    return {
        "loss": loss_dict["loss"].item(),
        "pg_loss": loss_dict["pg_loss"].item(),
        "kl_loss": loss_dict["kl_loss"].item(),
        "mean_kl": loss_dict["mean_kl"],
        "grad_norm": (
            grad_norm.item()
            if isinstance(grad_norm, torch.Tensor)
            else float(grad_norm)
        ),
    }


# ---------------------------------------------------------------------------
# Rollout generation
# ---------------------------------------------------------------------------


def generate_rollouts(
    policy_model: Any,
    tokenizer: Any,
    emoji_mask: torch.Tensor,
    phrase: str,
    group_size: int = 8,
    temperature: float = 1.0,
    max_tokens: int = 20,
    system_prompt: str | None = None,
) -> list[str]:
    """Generate group_size emoji rollouts for a single-turn prompt.

    Uses batched generation for efficiency. For multi-turn episodes, use
    run_episode_hf instead.
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt = format_prompt(phrase, tokenizer, system_prompt)
    inputs = tokenizer(
        [prompt] * group_size,
        return_tensors="pt",
        padding=True,
    ).to(device)

    policy_model.eval()
    with torch.no_grad():
        output_ids = policy_model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
            logits_processor=[EmojiLogitsProcessor(emoji_mask)],
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    response_ids = output_ids[:, prompt_len:]

    results: list[str] = []
    for ids in response_ids:
        ids = ids.tolist()
        if tokenizer.eos_token_id in ids:
            ids = ids[: ids.index(tokenizer.eos_token_id)]
        text = tokenizer.decode(ids, skip_special_tokens=True).strip()
        results.append(text)

    return results


def _generate_single_hf(
    policy_model: Any,
    tokenizer: Any,
    emoji_mask: torch.Tensor,
    prompt_text: str,
    temperature: float = 1.0,
    max_tokens: int = 20,
) -> str:
    """Generate a single emoji response from the local policy model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    policy_model.eval()
    with torch.no_grad():
        output_ids = policy_model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
            logits_processor=[EmojiLogitsProcessor(emoji_mask)],
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    response_ids = output_ids[0, prompt_len:].tolist()
    if tokenizer.eos_token_id in response_ids:
        response_ids = response_ids[: response_ids.index(tokenizer.eos_token_id)]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def run_episode_hf(
    policy_model: Any,
    tokenizer: Any,
    emoji_mask: torch.Tensor,
    guesser: SimulatedGuesser,
    scorer: SimilarityScorer,
    phrase: str,
    max_turns: int = 5,
    temperature: float = 1.0,
    max_response_tokens: int = 20,
    system_prompt: str | None = None,
    exact_match_threshold: float = 0.65,
) -> Episode:
    """Run a single multi-turn episode using the local HF policy model.

    Uses _build_multiturn_prompt_local for turn > 1, which matches the format
    used in build_multiturn_sequence so rollout context == training context.
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    episode = Episode(target_phrase=phrase)
    history: list[Turn] = []

    for turn_num in range(1, max_turns + 1):
        if turn_num == 1:
            prompt_text = format_prompt(phrase, tokenizer, system_prompt)
        else:
            prompt_text = _build_multiturn_prompt_local(
                phrase, history, system_prompt, tokenizer
            )

        emoji_output = _generate_single_hf(
            policy_model,
            tokenizer,
            emoji_mask,
            prompt_text,
            temperature,
            max_response_tokens,
        )

        turn_history = [(t.emoji_output, t.guess) for t in history]
        guess = guesser.guess(
            emoji_output,
            previous_guesses=[t.guess for t in history] if history else None,
            turn_history=turn_history if turn_history else None,
        )
        sim = scorer.score(phrase, guess)

        turn = Turn(
            turn_number=turn_num, emoji_output=emoji_output, guess=guess, similarity=sim
        )
        history.append(turn)
        episode.turns.append(turn)

        if sim >= exact_match_threshold:
            episode.completed = True
            episode.completion_turn = turn_num
            break

    return episode


def _generate_batch_hf(
    policy_model: Any,
    tokenizer: Any,
    emoji_mask: torch.Tensor,
    prompts: list[str],
    temperature: float = 1.0,
    max_tokens: int = 20,
) -> list[str]:
    """Batch-generate emoji responses for multiple (different) prompts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    policy_model.eval()
    with torch.no_grad():
        output_ids = policy_model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
            logits_processor=[EmojiLogitsProcessor(emoji_mask)],
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    results: list[str] = []
    for ids in output_ids[:, prompt_len:]:
        ids = ids.tolist()
        if tokenizer.eos_token_id in ids:
            ids = ids[: ids.index(tokenizer.eos_token_id)]
        results.append(tokenizer.decode(ids, skip_special_tokens=True).strip())
    return results


def run_episode_group_hf(
    policy_model: Any,
    tokenizer: Any,
    emoji_mask: torch.Tensor,
    guesser: SimulatedGuesser,
    scorer: SimilarityScorer,
    phrase: str,
    group_size: int,
    max_turns: int = 5,
    temperature: float = 1.0,
    max_response_tokens: int = 20,
    system_prompt: str | None = None,
    exact_match_threshold: float = 0.65,
) -> list[Episode]:
    """Run group_size independent episodes with batched GPU + concurrent API per turn.

    Each turn: batch GPU inference across all active episodes, then fire all
    guesser requests concurrently via guess_batch. Replaces a sequential loop
    over run_episode_hf — same outputs, ~4-6x faster on API-bound workloads.
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    episodes = [Episode(target_phrase=phrase) for _ in range(group_size)]
    active = list(range(group_size))

    for turn_num in range(1, max_turns + 1):
        if not active:
            break

        # Batch GPU inference
        if turn_num == 1:
            emoji_outputs = generate_rollouts(
                policy_model, tokenizer, emoji_mask, phrase,
                len(active), temperature, max_response_tokens, system_prompt,
            )
        else:
            prompts = [
                _build_multiturn_prompt_local(
                    phrase, list(episodes[i].turns), system_prompt, tokenizer
                )
                for i in active
            ]
            emoji_outputs = _generate_batch_hf(
                policy_model, tokenizer, emoji_mask, prompts, temperature, max_response_tokens
            )

        # Concurrent API calls
        g_inputs = [
            {
                "emoji": emoji_outputs[idx],
                "previous_guesses": [t.guess for t in episodes[i].turns] or None,
                "turn_history": [(t.emoji_output, t.guess) for t in episodes[i].turns] or None,
            }
            for idx, i in enumerate(active)
        ]
        guesses = guesser.guess_batch(g_inputs)
        sims = scorer.score_batch([(phrase, g) for g in guesses])

        still_active = []
        for idx, i in enumerate(active):
            turn = Turn(
                turn_number=turn_num,
                emoji_output=emoji_outputs[idx],
                guess=guesses[idx],
                similarity=sims[idx],
            )
            episodes[i].turns.append(turn)
            if sims[idx] >= exact_match_threshold:
                episodes[i].completed = True
                episodes[i].completion_turn = turn_num
            else:
                still_active.append(i)
        active = still_active

    return episodes


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(
    model_name: str = MODEL_NAME,
    phrases: list[str] | None = None,
    n_steps: int = 50,
    group_size: int = 8,
    learning_rate: float = 1e-5,
    kl_coeff: float = 0.05,
    temperature: float = 1.0,
    max_turns: int = 1,
    lora_rank: int = 16,
    seed: int = 42,
    log_every: int = 5,
    eval_every: int = 10,
    save_dir: str = "checkpoints/",
    n_eval_episodes: int = 20,
    guesser_model: str = "claude-sonnet-4-20250514",
    eval_phrases: list[str] | None = None,
) -> dict[str, Any]:
    """Main GRPO training loop.

    Phase 2 (max_turns=1): single-turn, batch generation, fast.
    Phase 3 (max_turns>1): multi-turn episodes, phrase rotation, per-turn rewards.

    Returns history dict with training metrics and before/after episode data
    suitable for visualization.
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    if phrases is None:
        phrases = ["birthday party"]
    phrase_counts: dict[str, int] = {p: 0 for p in phrases}

    use_wandb = wandb is not None and bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        wandb.init(
            project="emo",
            name=f"grpo_{datetime.now().strftime('%Y%m%d_%H%M')}",
            config={
                "model_name": model_name,
                "n_steps": n_steps,
                "group_size": group_size,
                "learning_rate": learning_rate,
                "kl_coeff": kl_coeff,
                "temperature": temperature,
                "max_turns": max_turns,
                "lora_rank": lora_rank,
                "n_phrases": len(phrases),
                "phrases": phrases,
            },
        )

    logger.info("Loading models...")
    policy_model, ref_model, tokenizer = setup_models(
        model_name=model_name,
        lora_rank=lora_rank,
        lora_alpha=lora_rank * 2,
    )

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    emoji_mask = build_emoji_mask(tokenizer)
    # Multi-turn training uses conversation_mode=True so the guesser sees full history.
    guesser = SimulatedGuesser(
        model=guesser_model, difficulty="casual", conversation_mode=max_turns > 1
    )
    scorer = SimilarityScorer()

    history: dict[str, Any] = {
        "steps": [],
        "losses": [],
        "pg_losses": [],
        "kl_losses": [],
        "mean_rewards": [],
        "mean_kl": [],
        "grad_norms": [],
        "group_variances": [],
        "phrase_per_step": [],
        "eval_rewards": [],
        "eval_completion_rates": [],
        "eval_completion_turns": [],
        "sample_outputs": [],
        "before_samples": None,
        "after_samples": None,
        "before_episodes": [],
        "after_episodes": [],
        "phrase_counts": phrase_counts,
        "held_out_eval_rewards": [],
        "held_out_completion_rates": [],
    }

    # --- Capture baseline (before any training) ---
    phrase = phrases[0]
    logger.info("Generating baseline rollouts...")

    if max_turns == 1:
        before_emojis = generate_rollouts(
            policy_model, tokenizer, emoji_mask, phrase, group_size, temperature
        )
        before_inputs = [
            {"emoji": e, "previous_guesses": [], "turn_history": []}
            for e in before_emojis
        ]
        before_guesses = guesser.guess_batch(before_inputs)
        before_sims = scorer.score_batch([(phrase, g) for g in before_guesses])
        history["before_samples"] = {
            "emojis": before_emojis,
            "guesses": before_guesses,
            "sims": before_sims,
        }
        history["sample_outputs"].append((0, phrase, before_emojis))
        print(
            f"\n=== Baseline (step 0) — mean sim: {sum(before_sims)/len(before_sims):.4f} ==="
        )
        for e, g, s in zip(before_emojis, before_guesses, before_sims):
            print(f"  {e} → '{g}' ({s:.3f})")
    else:
        logger.info("Running multi-turn baseline episodes...")
        sampled_phrases = (
            phrases if len(phrases) <= n_eval_episodes
            else rng.sample(phrases, n_eval_episodes)
        )
        eps_per_phrase = max(1, n_eval_episodes // len(sampled_phrases))
        for p in sampled_phrases:
            eps = run_episode_group_hf(
                policy_model, tokenizer, emoji_mask, guesser, scorer,
                p, eps_per_phrase, max_turns, temperature,
            )
            history["before_episodes"].extend(eps)

        before_rewards = [
            compute_turn_rewards(
                ep.target_phrase,
                [t.guess for t in ep.turns],
                scorer,
                emoji_outputs=[t.emoji_output for t in ep.turns],
            )["trajectory_reward"]
            for ep in history["before_episodes"]
        ]
        avg_reward = sum(before_rewards) / len(before_rewards)
        completion_rate = sum(ep.completed for ep in history["before_episodes"]) / len(
            history["before_episodes"]
        )
        print(
            f"\n=== Baseline (before training) — mean reward: {avg_reward:.4f}  "
            f"completion: {completion_rate:.2f} ==="
        )

    policy_model.train()

    # Early stopping state
    _best_es_reward = float("-inf")
    _no_improve_count = 0
    _should_stop = False

    # --- Training loop ---
    for step in range(1, n_steps + 1):
        phrase = rng.choice(phrases)
        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        history["phrase_per_step"].append(phrase)

        if max_turns == 1:
            emoji_outputs = generate_rollouts(
                policy_model, tokenizer, emoji_mask, phrase, group_size, temperature
            )
            g_inputs = [
                {"emoji": e, "previous_guesses": [], "turn_history": []}
                for e in emoji_outputs
            ]
            guesses = guesser.guess_batch(g_inputs)
            sims = scorer.score_batch([(phrase, g) for g in guesses])
            rep_penalties = [compute_repetition_penalty(e) for e in emoji_outputs]
            trajectory_rewards = [s - p for s, p in zip(sims, rep_penalties)]
            episodes = [
                Episode(
                    target_phrase=phrase,
                    turns=[Turn(turn_number=1, emoji_output=e, guess=g, similarity=s)],
                )
                for e, g, s in zip(emoji_outputs, guesses, sims)
            ]
        else:
            episodes = run_episode_group_hf(
                policy_model, tokenizer, emoji_mask, guesser, scorer,
                phrase, group_size, max_turns, temperature,
            )

            trajectory_rewards = [
                compute_turn_rewards(
                    ep.target_phrase,
                    [t.guess for t in ep.turns],
                    scorer,
                    emoji_outputs=[t.emoji_output for t in ep.turns],
                )["trajectory_reward"]
                for ep in episodes
            ]

        advantages = compute_group_advantages(trajectory_rewards)

        metrics = train_step(
            policy_model=policy_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            episodes=episodes,
            advantages=advantages,
            kl_coeff=kl_coeff,
        )

        mean_reward = sum(trajectory_rewards) / len(trajectory_rewards)
        rewards_t = torch.tensor(trajectory_rewards, dtype=torch.float32)
        group_variance = rewards_t.std().item() if len(trajectory_rewards) > 1 else 0.0

        history["steps"].append(step)
        history["losses"].append(metrics["loss"])
        history["pg_losses"].append(metrics["pg_loss"])
        history["kl_losses"].append(metrics["kl_loss"])
        history["mean_rewards"].append(mean_reward)
        history["mean_kl"].append(metrics["mean_kl"])
        history["grad_norms"].append(metrics["grad_norm"])
        history["group_variances"].append(group_variance)

        if use_wandb:
            rewards_t = torch.tensor(trajectory_rewards, dtype=torch.float32)
            step_log: dict[str, Any] = {
                "train/loss": metrics["loss"],
                "train/pg_loss": metrics["pg_loss"],
                "train/kl_loss": metrics["kl_loss"],
                "train/mean_reward": mean_reward,
                "train/mean_kl": metrics["mean_kl"],
                "train/grad_norm": metrics["grad_norm"],
                "train/group_reward_std": (
                    rewards_t.std().item() if len(trajectory_rewards) > 1 else 0.0
                ),
                "train/group_reward_min": rewards_t.min().item(),
                "train/group_reward_max": rewards_t.max().item(),
                "train/phrase": phrase,
                "step": step,
            }
            if max_turns > 1:
                step_log["train/completed"] = sum(
                    ep.completed for ep in episodes
                ) / len(episodes)
                step_log["train/n_turns"] = sum(len(ep.turns) for ep in episodes) / len(
                    episodes
                )
            wandb.log(step_log)

        if step % log_every == 0:
            print(
                f"Step {step:3d}/{n_steps} | phrase={phrase!r:20s} | "
                f"loss={metrics['loss']:+.4f}  pg={metrics['pg_loss']:+.4f}  "
                f"kl={metrics['kl_loss']:.4f}  reward={mean_reward:.4f}  "
                f"grad={metrics['grad_norm']:.4f}"
            )

        if step % eval_every == 0:
            if max_turns == 1:
                eval_emojis = generate_rollouts(
                    policy_model, tokenizer, emoji_mask, phrase, group_size, temperature
                )
                eval_inputs = [
                    {"emoji": e, "previous_guesses": [], "turn_history": []}
                    for e in eval_emojis
                ]
                eval_guesses = guesser.guess_batch(eval_inputs)
                eval_sims = scorer.score_batch([(phrase, g) for g in eval_guesses])
                eval_reward = sum(eval_sims) / len(eval_sims)

                history["eval_rewards"].append((step, eval_reward))
                history["sample_outputs"].append((step, phrase, eval_emojis))

                print(f"\n=== Eval @ step {step} — mean sim: {eval_reward:.4f} ===")
                for e, g, s in zip(eval_emojis, eval_guesses, eval_sims):
                    print(f"  {e} → '{g}' ({s:.3f})")
                print()

                if use_wandb:
                    wandb.log({"eval/mean_reward": eval_reward, "step": step})
            else:
                eval_episodes = []
                sampled_eval_phrases = (
                    phrases if len(phrases) <= n_eval_episodes
                    else rng.sample(phrases, n_eval_episodes)
                )
                eval_eps_per_phrase = max(1, group_size // len(sampled_eval_phrases))
                for p in sampled_eval_phrases:
                    eval_episodes.extend(run_episode_group_hf(
                        policy_model, tokenizer, emoji_mask, guesser, scorer,
                        p, eval_eps_per_phrase, max_turns, temperature,
                    ))

                eval_rewards_list = [
                    compute_turn_rewards(
                        ep.target_phrase,
                        [t.guess for t in ep.turns],
                        scorer,
                    )["trajectory_reward"]
                    for ep in eval_episodes
                ]
                eval_reward = sum(eval_rewards_list) / len(eval_rewards_list)
                completion_rate = sum(ep.completed for ep in eval_episodes) / len(
                    eval_episodes
                )
                completed_eps = [ep for ep in eval_episodes if ep.completed]
                mean_completion_turn = (
                    sum(ep.completion_turn for ep in completed_eps) / len(completed_eps)
                    if completed_eps
                    else float(max_turns)
                )

                history["eval_rewards"].append((step, eval_reward))
                history["eval_completion_rates"].append((step, completion_rate))
                history["eval_completion_turns"].append((step, mean_completion_turn))

                print(
                    f"\n=== Eval @ step {step} — reward: {eval_reward:.4f}  "
                    f"completion: {completion_rate:.2f}  avg_turn: {mean_completion_turn:.2f} ==="
                )
                for ep in eval_episodes[:2]:
                    print(f"  Phrase: '{ep.target_phrase}'")
                    for t in ep.turns:
                        print(
                            f"    Turn {t.turn_number}: {t.emoji_output} → '{t.guess}' (sim: {t.similarity:.3f})"
                        )
                print()

                if use_wandb:
                    eval_log: dict[str, Any] = {
                        "eval/mean_reward": eval_reward,
                        "eval/completion_rate": completion_rate,
                        "eval/avg_completion_turn": mean_completion_turn,
                        "step": step,
                    }

                    # Per-phrase reward breakdown
                    phrase_rewards: dict[str, list[float]] = {}
                    for ep, r in zip(eval_episodes, eval_rewards_list):
                        phrase_rewards.setdefault(ep.target_phrase, []).append(r)
                    for p, rs in phrase_rewards.items():
                        eval_log[f"eval/phrase_reward/{p}"] = sum(rs) / len(rs)

                    # Per-turn similarity table
                    turn_sims: dict[int, list[float]] = {}
                    for ep in eval_episodes:
                        for t in ep.turns:
                            turn_sims.setdefault(t.turn_number, []).append(t.similarity)
                    sim_table = wandb.Table(
                        columns=["turn", "mean_sim", "std_sim", "n"],
                        data=[
                            [
                                turn,
                                sum(sims) / len(sims),
                                (
                                    sum((s - sum(sims) / len(sims)) ** 2 for s in sims)
                                    / len(sims)
                                )
                                ** 0.5,
                                len(sims),
                            ]
                            for turn, sims in sorted(turn_sims.items())
                        ],
                    )
                    eval_log["eval/similarity_over_turns"] = sim_table

                    # Sample transcripts (up to 5)
                    for i, ep in enumerate(eval_episodes[:5]):
                        lines = [f"Phrase: {ep.target_phrase}"]
                        for t in ep.turns:
                            lines.append(
                                f"  Turn {t.turn_number}: {t.emoji_output} → '{t.guess}' (sim: {t.similarity:.3f})"
                            )
                        lines.append(f"  Completed: {ep.completed}")
                        transcript = "\n".join(lines)
                        eval_log[f"eval/transcript_{i}"] = wandb.Html(
                            f"<pre>{transcript}</pre>"
                        )

                    wandb.log(eval_log)

            # Held-out eval (phrases never trained on)
            _held_out_reward: float | None = None
            if eval_phrases:
                policy_model.eval()
                held_out_episodes = []
                held_out_eps_per_phrase = max(1, group_size // len(eval_phrases))
                for p in eval_phrases:
                    held_out_episodes.extend(run_episode_group_hf(
                        policy_model, tokenizer, emoji_mask, guesser, scorer,
                        p, held_out_eps_per_phrase, max_turns, temperature,
                    ))

                held_out_rewards_list = [
                    compute_turn_rewards(
                        ep.target_phrase,
                        [t.guess for t in ep.turns],
                        scorer,
                    )["trajectory_reward"]
                    for ep in held_out_episodes
                ]
                _held_out_reward = sum(held_out_rewards_list) / len(held_out_rewards_list)
                held_out_completion = sum(ep.completed for ep in held_out_episodes) / len(
                    held_out_episodes
                )

                history["held_out_eval_rewards"].append((step, _held_out_reward))
                history["held_out_completion_rates"].append((step, held_out_completion))

                print(
                    f"  Held-out eval: reward={_held_out_reward:.4f}  "
                    f"completion={held_out_completion:.2f}"
                )

                if use_wandb:
                    wandb.log({
                        "eval/held_out_reward": _held_out_reward,
                        "eval/held_out_completion_rate": held_out_completion,
                        "step": step,
                    })

            # Mid-run checkpoint
            ckpt_path = Path(save_dir) / f"step_{step}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            policy_model.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))
            logger.info(f"Checkpoint saved to {ckpt_path}")
            print(f"  Checkpoint saved to {ckpt_path}")

            # Early stopping checks
            # Primary: held-out (or training) reward plateau
            _es_reward = (
                _held_out_reward
                if eval_phrases and _held_out_reward is not None
                else eval_reward
            )
            if _es_reward > _best_es_reward + 0.02:
                _best_es_reward = _es_reward
                _no_improve_count = 0
            else:
                _no_improve_count += 1

            if _no_improve_count >= 3:
                print(
                    f"\nEarly stopping: no improvement for {_no_improve_count} consecutive "
                    f"evals (best={_best_es_reward:.4f})"
                )
                _should_stop = True

            # Secondary: KL budget (use mean of last 5 steps)
            _recent_kl_window = history["mean_kl"][-5:]
            _recent_kl = sum(_recent_kl_window) / max(1, len(_recent_kl_window))
            if _recent_kl > 0.5:
                print(f"\nEarly stopping: KL divergence too high (recent_kl={_recent_kl:.4f})")
                _should_stop = True

            policy_model.train()

        if _should_stop:
            break

    # --- Capture after-training samples ---
    if max_turns == 1:
        phrase = phrases[0]
        after_emojis = generate_rollouts(
            policy_model, tokenizer, emoji_mask, phrase, group_size, temperature
        )
        after_inputs = [
            {"emoji": e, "previous_guesses": [], "turn_history": []}
            for e in after_emojis
        ]
        after_guesses = guesser.guess_batch(after_inputs)
        after_sims = scorer.score_batch([(phrase, g) for g in after_guesses])
        history["after_samples"] = {
            "emojis": after_emojis,
            "guesses": after_guesses,
            "sims": after_sims,
        }
    else:
        logger.info("Running post-training evaluation episodes...")
        # Run n_eval_episodes episodes per phrase (same count as baseline for fair comparison).
        # These serve as both the after-training comparison data (plots 3-4) and transcript
        # source (plots 5-6); n_eval_episodes=30 covers both the 20-episode comparison and
        # 30-transcript requirements from the Phase 3 spec.
        eps_per_phrase = max(1, n_eval_episodes // len(phrases))
        for p in phrases:
            history["after_episodes"].extend(run_episode_group_hf(
                policy_model, tokenizer, emoji_mask, guesser, scorer,
                p, eps_per_phrase, max_turns, temperature,
            ))

        after_rewards = [
            compute_turn_rewards(
                ep.target_phrase,
                [t.guess for t in ep.turns],
                scorer,
                emoji_outputs=[t.emoji_output for t in ep.turns],
            )["trajectory_reward"]
            for ep in history["after_episodes"]
        ]
        avg_reward = sum(after_rewards) / len(after_rewards)
        completion_rate = sum(ep.completed for ep in history["after_episodes"]) / len(
            history["after_episodes"]
        )
        print(
            f"\n=== Post-training — mean reward: {avg_reward:.4f}  "
            f"completion: {completion_rate:.2f} ==="
        )

    # --- Save LoRA checkpoint ---
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    logger.info(f"Saved LoRA checkpoint to {save_path}")
    print(f"\nCheckpoint saved to {save_path}")

    if use_wandb:
        artifact = wandb.Artifact(f"emo-lora-step{n_steps}", type="model")
        artifact.add_dir(str(save_path))
        wandb.log_artifact(artifact)
        wandb.finish()

    return history
