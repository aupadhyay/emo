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
    exact_match_threshold: float = 0.65,
    repetition_penalty_scale: float = 0.3,
    turn_cost: float = 0.05,
    emoji_outputs: list[str] | None = None,
) -> dict:
    """Compute trajectory reward for a multi-turn episode.

        trajectory_reward = final_similarity
                            + completion_bonus_if_any
                            - turn_cost * n_turns_used
                            - sum(repetition_penalties)

    `final_similarity` is taken at the completion turn if the episode completed,
    otherwise at the last turn. Each turn carries a fixed turn_cost so taking
    more turns has a real cost (without this, the per-turn deltas telescope and
    the model has no incentive to be concise).

    Returns dict with keys: similarities, completion_turn, n_turns_used,
    repetition_penalties, completion_bonus, final_similarity, trajectory_reward.
    Also returns deltas and turn_rewards for diagnostic/viz consumers.
    """
    pairs = [(target_phrase, guess) for guess in guesses]
    similarities = scorer.score_batch(pairs)

    completion_turn = None
    for i, sim in enumerate(similarities):
        if sim >= exact_match_threshold:
            completion_turn = i + 1  # 1-indexed
            break

    n_turns = completion_turn if completion_turn is not None else len(guesses)
    final_similarity = similarities[n_turns - 1] if similarities else 0.0

    rep_penalties = []
    for i in range(n_turns):
        emoji = (emoji_outputs[i] if emoji_outputs else None)
        penalty = compute_repetition_penalty(emoji, repetition_penalty_scale) if emoji else 0.0
        rep_penalties.append(penalty)

    bonus = 0.0
    if completion_turn is not None:
        bonus = max(0.0, completion_bonus - (completion_turn - 1) * completion_decay)

    trajectory_reward = (
        final_similarity
        + bonus
        - turn_cost * n_turns
        - sum(rep_penalties)
    )

    # Diagnostic per-turn breakdown (kept for viz scripts; not used in training).
    deltas = [
        sim if i == 0 else sim - similarities[i - 1]
        for i, sim in enumerate(similarities)
    ]
    turn_rewards = [
        deltas[i] - (rep_penalties[i] if i < len(rep_penalties) else 0.0)
        for i in range(len(similarities))
    ]
    if completion_turn is not None and turn_rewards:
        turn_rewards[completion_turn - 1] += bonus

    return {
        "similarities": similarities,
        "completion_turn": completion_turn,
        "n_turns_used": n_turns,
        "final_similarity": final_similarity,
        "completion_bonus": bonus,
        "repetition_penalties": rep_penalties,
        "turn_cost_total": turn_cost * n_turns,
        "trajectory_reward": trajectory_reward,
        "deltas": deltas,
        "turn_rewards": turn_rewards,
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
