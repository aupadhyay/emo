"""Reward computation for the emoji retrieval game.

Uses sentence embeddings to compute cosine similarity between the judge's
reconstruction and the original target message. Runs on CPU (client side).
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class RewardEmbedder:
    """Lightweight sentence embeddings for reward computation."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts. Returns float in [0, 1]."""
        embeddings = self.model.encode([text_a, text_b], normalize_embeddings=True)
        return float(np.dot(embeddings[0], embeddings[1]))


class RetrievalGameReward:
    """Compute reward for a completed retrieval game episode.

    reward = similarity + success_bonus (if success) - turn_penalty * num_turns
             - format_penalty (if non-emoji detected)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        success_bonus: float = 0.5,
        turn_penalty: float = 0.1,
        format_penalty: float = 0.5,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.embedder = RewardEmbedder(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.success_bonus = success_bonus
        self.turn_penalty = turn_penalty
        self.format_penalty = format_penalty

    def compute(
        self,
        target_message: str,
        judge_final_guess: str,
        num_turns: int,
        format_violation: bool,
    ) -> dict:
        similarity = self.embedder.similarity(target_message, judge_final_guess)
        success = similarity >= self.similarity_threshold

        reward = similarity
        if success:
            reward += self.success_bonus
        reward -= self.turn_penalty * num_turns
        if format_violation:
            reward -= self.format_penalty

        return {
            "similarity": similarity,
            "success": success,
            "num_turns": num_turns,
            "format_violation": format_violation,
            "reward": reward,
        }
