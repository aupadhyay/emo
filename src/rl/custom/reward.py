"""Reward computation for the multi-turn emoji communication game."""

import numpy as np
from sentence_transformers import SentenceTransformer


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


def compute_turn_rewards(
    target_phrase: str,
    guesses: list[str],
    scorer: SimilarityScorer,
    completion_bonus: float = 1.0,
    completion_decay: float = 0.2,
    exact_match_threshold: float = 0.85,
) -> dict:
    """Compute per-turn rewards for a multi-turn episode.

    Returns dict with keys: similarities, deltas, completion_turn, turn_rewards,
    trajectory_reward.
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

    turn_rewards = []
    for i, delta in enumerate(deltas):
        reward = delta
        if completion_turn is not None and (i + 1) == completion_turn:
            bonus = completion_bonus - i * completion_decay
            reward += max(0.0, bonus)
        turn_rewards.append(reward)

    return {
        "similarities": similarities,
        "deltas": deltas,
        "completion_turn": completion_turn,
        "turn_rewards": turn_rewards,
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
