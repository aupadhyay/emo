"""GRPO training loop for the emoji communication game."""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

from src.rl.custom.env import Episode, SimulatedGuesser, Turn
from src.rl.custom.generate import (
    DEFAULT_SYSTEM_PROMPT,
    MODEL_NAME,
    build_emoji_mask,
    format_prompt,
)
from src.rl.custom.reward import SimilarityScorer, compute_group_advantages, compute_repetition_penalty

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
# GRPO loss
# ---------------------------------------------------------------------------


def compute_grpo_loss(
    policy_model: Any,
    ref_model: Any,
    tokenizer: Any,
    prompt_tokens: torch.Tensor,  # (B, prompt_len)
    response_tokens: torch.Tensor,  # (B, max_resp_len) — right-padded
    advantages: torch.Tensor,  # (B,)
    attention_mask: torch.Tensor,  # (B, prompt_len + max_resp_len)
    kl_coeff: float = 0.05,
) -> dict[str, Any]:
    """Compute GRPO policy-gradient loss for one group of rollouts.

    Returns a dict with keys: loss, pg_loss, kl_loss, mean_kl, mean_logprob.
    """
    prompt_len = prompt_tokens.shape[1]
    full_ids = torch.cat([prompt_tokens, response_tokens], dim=1)  # (B, L)

    # Response mask: 1 for actual response tokens, 0 for prompt and padding.
    response_mask = attention_mask.clone().float()
    response_mask[:, :prompt_len] = 0.0

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

    Returns dict of metrics: loss, pg_loss, kl_loss, mean_kl, grad_norm.
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_id = tokenizer.pad_token_id

    prompt_ids_list: list[list[int]] = []
    response_ids_list: list[list[int]] = []

    for ep in episodes:
        # Phase 2 is single-turn; prompt is the initial format_prompt output.
        prompt_text = format_prompt(ep.target_phrase, tokenizer, system_prompt)
        response_text = ep.turns[0].emoji_output

        p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        r_ids = tokenizer.encode(response_text, add_special_tokens=False)
        r_ids.append(tokenizer.eos_token_id)  # teach the model when to stop

        prompt_ids_list.append(p_ids)
        response_ids_list.append(r_ids)

    prompt_len = max(len(p) for p in prompt_ids_list)
    max_resp_len = max(len(r) for r in response_ids_list)
    B = len(episodes)

    prompt_tensor = torch.full((B, prompt_len), pad_id, dtype=torch.long)
    response_tensor = torch.full((B, max_resp_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, prompt_len + max_resp_len, dtype=torch.long)

    for i, (p_ids, r_ids) in enumerate(zip(prompt_ids_list, response_ids_list)):
        # Left-pad prompt so all prompts are right-aligned (safe for same-phrase batches)
        p_off = prompt_len - len(p_ids)
        prompt_tensor[i, p_off:] = torch.tensor(p_ids, dtype=torch.long)
        response_tensor[i, : len(r_ids)] = torch.tensor(r_ids, dtype=torch.long)
        attention_mask[i, p_off:prompt_len] = 1
        attention_mask[i, prompt_len : prompt_len + len(r_ids)] = 1

    prompt_tensor = prompt_tensor.to(device)
    response_tensor = response_tensor.to(device)
    attention_mask = attention_mask.to(device)
    adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)

    policy_model.train()

    loss_dict = compute_grpo_loss(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tensor,
        response_tokens=response_tensor,
        advantages=adv_tensor,
        attention_mask=attention_mask,
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
    """Generate group_size emoji rollouts from the current policy.

    Uses HuggingFace model.generate() with an EmojiLogitsProcessor so the
    in-memory LoRA weights are applied without vLLM.
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
) -> dict[str, Any]:
    """Main GRPO training loop.

    Phase 2 usage: one phrase, single-turn, 20-50 steps — prove the gradient works.

    Returns a history dict with training metrics and before/after sample data
    suitable for plotting in the sanity-check script.
    """
    torch.manual_seed(seed)
    if phrases is None:
        phrases = ["birthday party"]
    phrase = phrases[0]  # Phase 2: always the same phrase

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
    guesser = SimulatedGuesser(difficulty="casual", conversation_mode=False)
    scorer = SimilarityScorer()

    history: dict[str, Any] = {
        "steps": [],
        "losses": [],
        "pg_losses": [],
        "kl_losses": [],
        "mean_rewards": [],
        "mean_kl": [],
        "grad_norms": [],
        "eval_rewards": [],
        "sample_outputs": [],
        "before_samples": None,
        "after_samples": None,
    }

    # --- Capture baseline (before any training) ---
    logger.info(f"Generating baseline rollouts for '{phrase}'...")
    before_emojis = generate_rollouts(
        policy_model, tokenizer, emoji_mask, phrase, group_size, temperature
    )
    before_inputs = [
        {"emoji": e, "previous_guesses": [], "turn_history": []} for e in before_emojis
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

    policy_model.train()

    # --- Training loop ---
    for step in range(1, n_steps + 1):
        emoji_outputs = generate_rollouts(
            policy_model, tokenizer, emoji_mask, phrase, group_size, temperature
        )

        # Parallel guesser calls
        g_inputs = [
            {"emoji": e, "previous_guesses": [], "turn_history": []}
            for e in emoji_outputs
        ]
        guesses = guesser.guess_batch(g_inputs)
        sims = scorer.score_batch([(phrase, g) for g in guesses])
        rep_penalties = [compute_repetition_penalty(e) for e in emoji_outputs]
        trajectory_rewards = [s - p for s, p in zip(sims, rep_penalties)]
        advantages = compute_group_advantages(trajectory_rewards)

        episodes = [
            Episode(
                target_phrase=phrase,
                turns=[Turn(turn_number=1, emoji_output=e, guess=g, similarity=s)],
            )
            for e, g, s in zip(emoji_outputs, guesses, sims)
        ]

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
        history["steps"].append(step)
        history["losses"].append(metrics["loss"])
        history["pg_losses"].append(metrics["pg_loss"])
        history["kl_losses"].append(metrics["kl_loss"])
        history["mean_rewards"].append(mean_reward)
        history["mean_kl"].append(metrics["mean_kl"])
        history["grad_norms"].append(metrics["grad_norm"])

        if step % log_every == 0:
            print(
                f"Step {step:3d}/{n_steps} | "
                f"loss={metrics['loss']:+.4f}  pg={metrics['pg_loss']:+.4f}  "
                f"kl={metrics['kl_loss']:.4f}  reward={mean_reward:.4f}  "
                f"grad={metrics['grad_norm']:.4f}"
            )

        if step % eval_every == 0:
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

            policy_model.train()

    # --- Capture after-training samples ---
    after_emojis = generate_rollouts(
        policy_model, tokenizer, emoji_mask, phrase, group_size, temperature
    )
    after_inputs = [
        {"emoji": e, "previous_guesses": [], "turn_history": []} for e in after_emojis
    ]
    after_guesses = guesser.guess_batch(after_inputs)
    after_sims = scorer.score_batch([(phrase, g) for g in after_guesses])
    history["after_samples"] = {
        "emojis": after_emojis,
        "guesses": after_guesses,
        "sims": after_sims,
    }

    # --- Save LoRA checkpoint ---
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    logger.info(f"Saved LoRA checkpoint to {save_path}")
    print(f"\nCheckpoint saved to {save_path}")

    return history
