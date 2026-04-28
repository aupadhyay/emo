"""Phase 3 exit gate: multi-turn GRPO validation.

Trains with max_turns=5 on 3 phrases and produces 9 diagnostic plots that test
whether RL training taught the model turn-specific communication strategies.

Usage (remote, on A100):
    modal run src/viz/multiturn_grpo_validation.py

    # Re-plot from a saved history file (no re-training):
    modal run src/viz/multiturn_grpo_validation.py --history-json viz_outputs/phase3/history.json
"""

import json
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal image and volumes
# ---------------------------------------------------------------------------

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "peft>=0.12.0",
        "anthropic>=0.30.0",
        "sentence-transformers>=3.0.0",
        "numpy",
        "matplotlib",
    )
    .add_local_python_source("src")
)

_model_volume = modal.Volume.from_name("emoji-model-weights", create_if_missing=True)
_output_volume = modal.Volume.from_name("emoji-phase3-outputs", create_if_missing=True)
_MODEL_CACHE_DIR = "/model-cache"
_OUTPUT_DIR = "/outputs"

app = modal.App("grpo-multiturn-validation")


# ---------------------------------------------------------------------------
# Remote training function
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100",
    timeout=21600,  # 6 hours: 60 steps × 5 turns × 8 rollouts × API latency
    image=_image,
    secrets=[modal.Secret.from_name("anthropic-api-key")],
    volumes={
        _MODEL_CACHE_DIR: _model_volume,
        _OUTPUT_DIR: _output_volume,
    },
)
def run_phase3_training(
    run_name: str = "phase3-multiturn",
    n_steps: int = 60,
    group_size: int = 8,
    learning_rate: float = 1e-5,
    eval_every: int = 10,
    n_eval_episodes: int = 30,
    guesser_model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """Train multi-turn GRPO on 3 phrases and return serialized history."""
    import os

    os.environ["HF_HOME"] = _MODEL_CACHE_DIR
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = _MODEL_CACHE_DIR

    from src.rl.custom.train import train

    checkpoint_dir = f"{_OUTPUT_DIR}/checkpoints/{run_name}"
    history = train(
        phrases=["job interview", "road trip", "broken heart"],
        n_steps=n_steps,
        group_size=group_size,
        learning_rate=learning_rate,
        kl_coeff=0.05,
        temperature=1.0,
        max_turns=5,
        lora_rank=16,
        eval_every=eval_every,
        save_dir=checkpoint_dir,
        n_eval_episodes=n_eval_episodes,
        guesser_model=guesser_model,
    )

    serialized = _make_serializable(history)
    history_path = f"{_OUTPUT_DIR}/{run_name}-history.json"
    with open(history_path, "w") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)
    _output_volume.commit()

    return serialized


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _make_serializable(obj):
    """Recursively convert Episode/Turn dataclasses and tensors to plain dicts."""
    import dataclasses

    if obj is None:
        return None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _make_serializable(v) for k, v in dataclasses.asdict(obj).items()}
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, list):
        return [_make_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Per-turn metric helpers
# ---------------------------------------------------------------------------


def _per_turn_sims(episodes: list[dict], max_turns: int = 5) -> tuple[list, list, list]:
    """Mean, std, and n of similarity at each turn number."""
    by_turn: dict[int, list[float]] = {t: [] for t in range(1, max_turns + 1)}
    for ep in episodes:
        for turn in ep["turns"]:
            t = turn["turn_number"]
            if 1 <= t <= max_turns:
                by_turn[t].append(turn["similarity"])

    means, stds, ns = [], [], []
    for t in range(1, max_turns + 1):
        vals = by_turn[t]
        if vals:
            m = sum(vals) / len(vals)
            std = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
            means.append(m)
            stds.append(std)
            ns.append(len(vals))
        else:
            means.append(None)
            stds.append(None)
            ns.append(0)
    return means, stds, ns


def _per_turn_deltas(episodes: list[dict], max_turns: int = 5) -> tuple[list, list]:
    """Mean and std of similarity delta at each turn number."""
    by_turn: dict[int, list[float]] = {t: [] for t in range(1, max_turns + 1)}
    for ep in episodes:
        turns = ep["turns"]
        for i, turn in enumerate(turns):
            t = turn["turn_number"]
            if not (1 <= t <= max_turns):
                continue
            delta = turn["similarity"] if i == 0 else turn["similarity"] - turns[i - 1]["similarity"]
            by_turn[t].append(delta)

    means, stds = [], []
    for t in range(1, max_turns + 1):
        vals = by_turn[t]
        if vals:
            m = sum(vals) / len(vals)
            std = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
            means.append(m)
            stds.append(std)
        else:
            means.append(0.0)
            stds.append(0.0)
    return means, stds


def _ep_trajectory_reward(ep: dict) -> float:
    turns = ep["turns"]
    reward = 0.0
    for i, turn in enumerate(turns):
        reward += turn["similarity"] if i == 0 else turn["similarity"] - turns[i - 1]["similarity"]
    return reward


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_1_reward(history: dict, out_dir: Path) -> None:
    """Reward per step, color-coded by phrase."""
    import matplotlib.pyplot as plt
    import numpy as np

    steps = history["steps"]
    rewards = history["mean_rewards"]
    phrase_per_step = history.get("phrase_per_step", [""] * len(steps))

    phrase_colors = {
        "job interview": "steelblue",
        "road trip": "darkorange",
        "broken heart": "crimson",
    }
    all_phrases = sorted(set(phrase_per_step))

    fig, ax = plt.subplots(figsize=(10, 5))
    for phrase in all_phrases:
        color = phrase_colors.get(phrase, "gray")
        xs = [s for s, p in zip(steps, phrase_per_step) if p == phrase]
        ys = [r for r, p in zip(rewards, phrase_per_step) if p == phrase]
        ax.scatter(xs, ys, label=phrase, color=color, s=25, alpha=0.75, zorder=3)

    # Smoothed trend line across all phrases
    if len(rewards) >= 5:
        window = 5
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            steps[window - 1 :],
            smoothed,
            color="black",
            linewidth=1.5,
            linestyle="--",
            alpha=0.6,
            label=f"{window}-step avg",
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean trajectory reward")
    ax.set_title("Reward over training (color = phrase)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "1_reward.png", dpi=150)
    plt.close(fig)
    print("  Saved 1_reward.png")


def plot_2_kl(history: dict, out_dir: Path) -> None:
    """KL divergence over training steps."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["steps"], history["mean_kl"], marker="o", markersize=3, linewidth=1.5, color="darkorange")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean KL divergence")
    ax.set_title("KL divergence over training steps")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "2_kl.png", dpi=150)
    plt.close(fig)
    print("  Saved 2_kl.png")


def plot_3_similarity_over_turns(history: dict, out_dir: Path) -> None:
    """THE KEY CHART: mean similarity at each turn, before vs after."""
    import matplotlib.pyplot as plt

    before_eps = history.get("before_episodes", [])
    after_eps = history.get("after_episodes", [])
    if not before_eps or not after_eps:
        print("  Skipping 3_similarity_over_turns.png (missing episode data)")
        return

    max_turns = 5
    turns = list(range(1, max_turns + 1))

    fig, ax = plt.subplots(figsize=(8, 5))

    for eps, color, label in [
        (before_eps, "steelblue", "Before training"),
        (after_eps, "darkorange", "After training"),
    ]:
        means, stds, ns = _per_turn_sims(eps, max_turns)
        valid = [(t, m, s, n) for t, m, s, n in zip(turns, means, stds, ns) if m is not None]
        if not valid:
            continue
        ts, ms, ss, nns = zip(*valid)
        ax.plot(ts, ms, marker="o", linewidth=2, color=color, label=label)
        ax.fill_between(
            ts,
            [m - s for m, s in zip(ms, ss)],
            [m + s for m, s in zip(ms, ss)],
            color=color,
            alpha=0.15,
        )
        for t, m, n in zip(ts, ms, nns):
            ax.annotate(
                f"n={n}", (t, m), textcoords="offset points", xytext=(0, 9),
                fontsize=7, ha="center", color=color, alpha=0.8,
            )

    ax.set_xlabel("Turn number")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Similarity over turns: before vs after training\n(flat/upward trend after training = success)")
    ax.set_xticks(turns)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "3_similarity_over_turns.png", dpi=150)
    plt.close(fig)
    print("  Saved 3_similarity_over_turns.png")


def plot_4_delta_comparison(history: dict, out_dir: Path) -> None:
    """Per-turn similarity delta: before vs after, grouped bar chart."""
    import matplotlib.pyplot as plt
    import numpy as np

    before_eps = history.get("before_episodes", [])
    after_eps = history.get("after_episodes", [])
    if not before_eps or not after_eps:
        print("  Skipping 4_delta_comparison.png (missing episode data)")
        return

    max_turns = 5
    before_means, before_stds = _per_turn_deltas(before_eps, max_turns)
    after_means, after_stds = _per_turn_deltas(after_eps, max_turns)

    x = np.arange(max_turns)
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, before_means, width, yerr=before_stds, label="Before", color="steelblue", alpha=0.8, capsize=4)
    ax.bar(x + width / 2, after_means, width, yerr=after_stds, label="After", color="darkorange", alpha=0.8, capsize=4)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Turn number")
    ax.set_ylabel("Mean similarity delta")
    ax.set_title("Per-turn similarity delta: before vs after training\n(positive after-training delta on turns 2+ = success)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Turn {t}" for t in range(1, max_turns + 1)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "4_delta_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved 4_delta_comparison.png")


def plot_5_turn1_vs_turn2_emoji(history: dict, out_dir: Path) -> None:
    """Top-15 emoji for turn-1 vs turn-2+ from the trained model."""
    import matplotlib.pyplot as plt
    from collections import Counter

    _SKIP = frozenset((0xFE0F, 0xFE0E, 0x200D)) | frozenset(range(0x1F3FB, 0x1F3FF + 1))

    episodes = history.get("after_episodes", [])
    if not episodes:
        print("  Skipping 5_turn1_vs_turn2_emoji.png (no after_episodes)")
        return

    turn1_chars: list[str] = []
    turn2plus_chars: list[str] = []

    for ep in episodes:
        for turn in ep["turns"]:
            chars = [c for c in turn["emoji_output"] if ord(c) not in _SKIP]
            if turn["turn_number"] == 1:
                turn1_chars.extend(chars)
            else:
                turn2plus_chars.extend(chars)

    top1 = Counter(turn1_chars).most_common(15)
    top2 = Counter(turn2plus_chars).most_common(15)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))

    for ax, data, title, color in [
        (ax1, top1, "Turn 1 — opening clues", "steelblue"),
        (ax2, top2, "Turn 2+ — feedback responses", "darkorange"),
    ]:
        if data:
            emojis, counts = zip(*data)
            y_pos = range(len(emojis))
            ax.barh(list(y_pos), list(counts), color=color, alpha=0.8)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(list(emojis), fontsize=14)
            ax.invert_yaxis()
            ax.set_xlabel("Frequency")
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis="x")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)

    fig.suptitle("Emoji distribution: Turn 1 vs Turn 2+ (trained model)\nDifferent distributions = turn-specific strategy learned", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "5_turn1_vs_turn2_emoji.png", dpi=150)
    plt.close(fig)
    print("  Saved 5_turn1_vs_turn2_emoji.png")


def plot_6_transcripts(history: dict, out_dir: Path) -> None:
    """Save full transcripts and cherry-picked examples (best / worst / interesting)."""
    episodes = history.get("after_episodes", [])
    if not episodes:
        print("  Skipping 6_transcripts (no after_episodes)")
        return

    # Full transcripts
    full_path = out_dir / "6_transcripts.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)
    print(f"  Saved 6_transcripts.json ({len(episodes)} episodes)")

    # Rank by trajectory reward
    ranked = sorted(episodes, key=_ep_trajectory_reward)
    worst_3 = ranked[:3]
    best_3 = list(reversed(ranked[-3:]))

    def _is_interesting(ep: dict) -> bool:
        turns = ep["turns"]
        if len(turns) < 2:
            return False
        meta = set("✅❌🟢🔴⬆️⬇️👍👎➡️⬅️🔄🆗")
        for turn in turns[1:]:
            if any(c in meta for c in turn["emoji_output"]):
                return True
        t1_chars = set(turns[0]["emoji_output"])
        for turn in turns[1:]:
            t_chars = set(turn["emoji_output"])
            if t_chars and len(t_chars - t1_chars) / len(t_chars) > 0.5:
                return True
        return False

    interesting_3 = [ep for ep in episodes if _is_interesting(ep)][:3]

    def _format_ep(ep: dict) -> str:
        lines = [f'Phrase: "{ep["target_phrase"]}"']
        turns = ep["turns"]
        for i, turn in enumerate(turns):
            delta = turn["similarity"] if i == 0 else turn["similarity"] - turns[i - 1]["similarity"]
            mark = " ✓" if ep.get("completed") and ep.get("completion_turn") == turn["turn_number"] else ""
            lines.append(
                f'Turn {turn["turn_number"]}: {turn["emoji_output"]} → '
                f'Guess: "{turn["guess"]}" (sim: {turn["similarity"]:.2f}, Δ: {delta:+.2f}){mark}'
            )
        lines.append(f"Trajectory reward: {_ep_trajectory_reward(ep):.2f}")
        return "\n".join(lines)

    cherry_path = out_dir / "6_cherry_picked.md"
    with open(cherry_path, "w", encoding="utf-8") as f:
        for section, eps in [
            ("Best Episodes", best_3),
            ("Worst Episodes", worst_3),
            ("Most Interesting Episodes", interesting_3),
        ]:
            f.write(f"## {section}\n\n")
            for ep in eps:
                f.write(_format_ep(ep) + "\n\n")
            f.write("\n")
    print("  Saved 6_cherry_picked.md")


def plot_7_completion_rate(history: dict, out_dir: Path) -> None:
    """Completion rate at each eval checkpoint."""
    import matplotlib.pyplot as plt

    data = history.get("eval_completion_rates", [])
    if not data:
        print("  Skipping 7_completion_rate.png (no eval_completion_rates)")
        return

    steps = [x[0] for x in data]
    rates = [x[1] for x in data]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, rates, marker="o", linewidth=2, color="seagreen")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Completion rate")
    ax.set_title("Episode completion rate over training")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "7_completion_rate.png", dpi=150)
    plt.close(fig)
    print("  Saved 7_completion_rate.png")


def plot_8_avg_completion_turn(history: dict, out_dir: Path) -> None:
    """Mean turn at which completed episodes finish."""
    import matplotlib.pyplot as plt

    data = history.get("eval_completion_turns", [])
    if not data:
        print("  Skipping 8_avg_completion_turn.png (no eval_completion_turns)")
        return

    steps = [x[0] for x in data]
    turns = [x[1] for x in data]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, turns, marker="o", linewidth=2, color="mediumpurple")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean completion turn (completed eps only)")
    ax.set_title("Average completion turn over training\n(decreasing = model guides guesser faster)")
    ax.set_ylim(0.5, 5.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "8_avg_completion_turn.png", dpi=150)
    plt.close(fig)
    print("  Saved 8_avg_completion_turn.png")


def plot_9_group_variance(history: dict, out_dir: Path) -> None:
    """Reward variance within each group over training."""
    import matplotlib.pyplot as plt

    variances = history.get("group_variances", [])
    if not variances:
        print("  Skipping 9_group_variance.png (no group_variances)")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["steps"], variances, marker="o", markersize=3, linewidth=1.5, color="coral")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Reward std within group")
    ax.set_title("Reward group variance over training\n(collapsed = converged; sustained = still exploring)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "9_group_variance.png", dpi=150)
    plt.close(fig)
    print("  Saved 9_group_variance.png")


def plot_all(history: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating Phase 3 diagnostic plots → {out_dir}/")
    plot_1_reward(history, out_dir)
    plot_2_kl(history, out_dir)
    plot_3_similarity_over_turns(history, out_dir)
    plot_4_delta_comparison(history, out_dir)
    plot_5_turn1_vs_turn2_emoji(history, out_dir)
    plot_6_transcripts(history, out_dir)
    plot_7_completion_rate(history, out_dir)
    plot_8_avg_completion_turn(history, out_dir)
    plot_9_group_variance(history, out_dir)


def print_success_criteria(history: dict) -> None:
    """Print Phase 3 success criteria evaluation."""
    print("\n=== Phase 3 success criteria ===")

    before_eps = history.get("before_episodes", [])
    after_eps = history.get("after_episodes", [])

    # 1. Similarity-over-turns improvement
    if before_eps and after_eps:
        before_means, _, _ = _per_turn_sims(before_eps, 5)
        after_means, _, _ = _per_turn_sims(after_eps, 5)

        before_valid = [m for m in before_means if m is not None]
        after_valid = [m for m in after_means if m is not None]

        if len(before_valid) >= 2:
            before_slope = before_valid[-1] - before_valid[0]
            after_slope = after_valid[-1] - after_valid[0] if len(after_valid) >= 2 else 0
            improved = after_slope > before_slope
            print(f"  1. Sim-over-turns improved: before slope={before_slope:+.3f}  after slope={after_slope:+.3f}  {'✓' if improved else '✗'}")

    # 2. Turn 2+ deltas positive after training
    if after_eps:
        after_means, _ = _per_turn_deltas(after_eps, 5)
        turn2plus_positive = all(d > 0 for d in after_means[1:] if d is not None)
        avg_t2plus = sum(after_means[1:]) / max(1, len(after_means[1:]))
        print(f"  2. Turn 2+ deltas positive: avg={avg_t2plus:+.3f}  {'✓' if turn2plus_positive else '✗ (partial)'}")

    # 3. Turn-1 vs turn-2+ emoji differ (checked qualitatively via plot 5)
    print("  3. Turn 1 vs turn 2+ emoji distributions: see 5_turn1_vs_turn2_emoji.png")

    # 4. Completion rate increases
    cr_data = history.get("eval_completion_rates", [])
    if len(cr_data) >= 2:
        first_rate = cr_data[0][1]
        last_rate = cr_data[-1][1]
        print(f"  4. Completion rate: {first_rate:.2f} → {last_rate:.2f}  {'✓' if last_rate > first_rate else '✗'}")

    # 5. KL stability
    kls = history.get("mean_kl", [])
    if kls:
        max_kl = max(kls)
        print(f"  5. Max KL: {max_kl:.4f}  {'✓ stable' if max_kl < 2.0 else '✗ too high'}")

    # 6. Cherry-picked transcripts (see 6_cherry_picked.md)
    print("  6. Cherry-picked transcripts: see 6_cherry_picked.md")
    print()


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    run_name: str = "phase3-multiturn",
    n_steps: int = 60,
    group_size: int = 8,
    eval_every: int = 10,
    out_dir: str = "",
    history_json: str = "",
) -> None:
    """Run Phase 3 training remotely then plot locally.

    Pass --history-json to skip training and re-plot from an existing run.
    """
    out_path = Path(out_dir if out_dir else f"viz_outputs/{run_name}")

    if history_json:
        print(f"Loading history from {history_json}...")
        with open(history_json) as f:
            history = json.load(f)
    else:
        print(
            f"Starting Phase 3 training: run={run_name!r}  steps={n_steps}  "
            f"group_size={group_size}  eval_every={eval_every}"
        )
        history = run_phase3_training.remote(
            run_name=run_name,
            n_steps=n_steps,
            group_size=group_size,
            eval_every=eval_every,
        )

    # Save history locally for re-plotting
    out_path.mkdir(parents=True, exist_ok=True)
    history_out = out_path / "history.json"
    with open(history_out, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"History saved to {history_out}")

    plot_all(history, out_path)
    print_success_criteria(history)
