"""Reward computation for the multi-turn emoji communication game."""

import numpy as np
from sentence_transformers import SentenceTransformer

# Codepoints to ignore when counting base emoji (variation selectors, ZWJ, skin tones)
_SKIP_CODEPOINTS = frozenset((0xFE0F, 0xFE0E, 0x200D)) | frozenset(range(0x1F3FB, 0x1F3FF + 1))


class SimilarityScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def score(self, text_a: str, text_b: str) -> float:
        embeddings = self.model.encode([text_a, text_b], normalize_embeddings=True)
        return float(np.dot(embeddings[0], embeddings[1]))

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        texts = [t for pair in pairs for t in pair]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        results = []
        for i in range(len(pairs)):
            a, b = embeddings[2 * i], embeddings[2 * i + 1]
            results.append(float(np.dot(a, b)))
        return results


def compute_repetition_penalty(emoji_str: str, penalty_scale: float = 0.3) -> float:
    """Return a penalty in [0, penalty_scale] for repeated emoji in a sequence.

    Strips variation selectors, ZWJ, and skin-tone modifiers, then computes
    repeat_ratio = 1 - (unique_base_chars / total_base_chars).
    A fully-unique sequence scores 0; all-same scores penalty_scale.
    """
    if not emoji_str:
        return 0.0
    base_chars = [c for c in emoji_str if ord(c) not in _SKIP_CODEPOINTS]
    if len(base_chars) <= 1:
        return 0.0
    repeat_ratio = 1.0 - len(set(base_chars)) / len(base_chars)
    return penalty_scale * repeat_ratio


def compute_turn_rewards(
    target_phrase: str,
    guesses: list[str],
    scorer: SimilarityScorer,
    completion_bonus: float = 1.0,
    completion_decay: float = 0.2,
    exact_match_threshold: float = 0.85,
    repetition_penalty_scale: float = 0.3,
    emoji_outputs: list[str] | None = None,
) -> dict:
    """Compute per-turn rewards for a multi-turn episode.

    If emoji_outputs is provided, a repetition penalty is subtracted from each
    turn's reward proportional to how many emoji are repeated in that output.

    Returns dict with keys: similarities, deltas, completion_turn, turn_rewards,
    repetition_penalties, trajectory_reward.
    """
    pairs = [(target_phrase, guess) for guess in guesses]
    similarities = scorer.score_batch(pairs)

    deltas = []
    for i, sim in enumerate(similarities):
        deltas.append(sim if i == 0 else sim - similarities[i - 1])

    completion_turn = None
    for i, sim in enumerate(similarities):
        if sim >= exact_match_threshold:
            completion_turn = i + 1  # 1-indexed
            break

    rep_penalties = []
    for i in range(len(guesses)):
        emoji = (emoji_outputs[i] if emoji_outputs else None)
        penalty = compute_repetition_penalty(emoji, repetition_penalty_scale) if emoji else 0.0
        rep_penalties.append(penalty)

    turn_rewards = []
    for i, delta in enumerate(deltas):
        reward = delta - rep_penalties[i]
        if completion_turn is not None and (i + 1) == completion_turn:
            bonus = completion_bonus - i * completion_decay
            reward += max(0.0, bonus)
        turn_rewards.append(reward)

    return {
        "similarities": similarities,
        "deltas": deltas,
        "completion_turn": completion_turn,
        "turn_rewards": turn_rewards,
        "repetition_penalties": rep_penalties,
        "trajectory_reward": sum(turn_rewards),
    }


def compute_group_advantages(
    trajectory_rewards: list[float],
    normalize: bool = True,
) -> list[float]:
    """Compute GRPO advantages for a group of trajectories on the same phrase."""
    rewards = np.array(trajectory_rewards, dtype=float)
    advantages = rewards - rewards.mean()
    if normalize:
        std = rewards.std()
        advantages = advantages / (std + 1e-8)
    return advantages.tolist()
