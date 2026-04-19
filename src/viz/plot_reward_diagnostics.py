"""Generate Phase 1 visualizations from validation JSON outputs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_per_phrase_variance():
    """Bar chart: mean similarity ± std per phrase, sorted by mean."""
    data = json.loads((OUT_DIR / "single_turn_raw.json").read_text())
    data.sort(key=lambda d: d["mean_sim"])

    phrases = [d["phrase"] for d in data]
    means = [d["mean_sim"] for d in data]
    stds = [d["std_sim"] for d in data]
    low_var = [d["std_sim"] < 0.1 for d in data]

    colors = ["#e74c3c" if lv else "#3498db" for lv in low_var]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(phrases, means, xerr=stds, color=colors, ecolor="black",
                   capsize=4, alpha=0.85)
    ax.set_xlabel("Cosine Similarity (mean ± std, 8 rollouts)")
    ax.set_title("Per-Phrase Similarity Distribution — Single Turn")
    ax.axvline(0.85, color="green", linestyle="--", linewidth=1.2, label="threshold (0.85)")

    red_patch = mpatches.Patch(color="#e74c3c", label="Low variance (std < 0.1)")
    blue_patch = mpatches.Patch(color="#3498db", label="Good variance")
    ax.legend(handles=[red_patch, blue_patch, ax.lines[0]], loc="lower right")
    ax.set_xlim(0, 1.15)

    fig.tight_layout()
    path = OUT_DIR / "per_phrase_variance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_trajectory_reward_distribution():
    """Histogram of trajectory rewards across all multi-turn episodes."""
    data = json.loads((OUT_DIR / "multiturn_transcripts.json").read_text())
    rewards = [ep["trajectory_reward"] for ep in data]
    completed = [ep["completed"] for ep in data]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist([r for r, c in zip(rewards, completed) if not c],
            bins=15, alpha=0.7, label="Not completed", color="#e74c3c")
    ax.hist([r for r, c in zip(rewards, completed) if c],
            bins=15, alpha=0.7, label="Completed", color="#2ecc71")
    ax.set_xlabel("Trajectory Reward")
    ax.set_ylabel("Count")
    ax.set_title("Trajectory Reward Distribution — Multi-Turn Episodes")
    ax.legend()
    fig.tight_layout()
    path = OUT_DIR / "trajectory_reward_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_similarity_over_turns():
    """Line plot: average similarity per turn number across all multi-turn episodes."""
    data = json.loads((OUT_DIR / "multiturn_transcripts.json").read_text())

    # Only episodes that went to turn 2+
    multi = [ep for ep in data if len(ep["turns"]) > 1]

    by_turn = {}
    for ep in multi:
        for t in ep["turns"]:
            by_turn.setdefault(t["turn_number"], []).append(t["similarity"])

    turns = sorted(by_turn.keys())
    means = [np.mean(by_turn[t]) for t in turns]
    stds = [np.std(by_turn[t]) for t in turns]
    counts = [len(by_turn[t]) for t in turns]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(turns, means, yerr=stds, marker="o", capsize=5,
                linewidth=2, color="#3498db", ecolor="gray")
    for t, m, n in zip(turns, means, counts):
        ax.annotate(f"n={n}", (t, m), textcoords="offset points",
                    xytext=(4, 6), fontsize=8, color="gray")
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Cosine Similarity (mean ± std)")
    ax.set_title("Similarity Over Turns — Multi-Turn Episodes (turn 2+ only)")
    ax.set_xticks(turns)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.85, color="green", linestyle="--", linewidth=1, label="threshold")
    ax.legend()
    fig.tight_layout()
    path = OUT_DIR / "similarity_over_turns.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_group_reward_variance():
    """Per-phrase trajectory reward spread across 4 rollouts (multi-turn)."""
    data = json.loads((OUT_DIR / "multiturn_transcripts.json").read_text())

    by_phrase = {}
    for ep in data:
        by_phrase.setdefault(ep["phrase"], []).append(ep["trajectory_reward"])

    phrases = list(by_phrase.keys())
    means = [np.mean(by_phrase[p]) for p in phrases]
    stds = [np.std(by_phrase[p]) for p in phrases]

    order = np.argsort(means)
    phrases = [phrases[i] for i in order]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]
    low_var = [s < 0.1 for s in stds]
    colors = ["#e74c3c" if lv else "#9b59b6" for lv in low_var]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(phrases, means, xerr=stds, color=colors, ecolor="black",
            capsize=4, alpha=0.85)
    ax.set_xlabel("Trajectory Reward (mean ± std, 4 rollouts)")
    ax.set_title("Per-Phrase Trajectory Reward Variance — Multi-Turn")
    red_patch = mpatches.Patch(color="#e74c3c", label="Low variance (std < 0.1)")
    purple_patch = mpatches.Patch(color="#9b59b6", label="Good variance")
    ax.legend(handles=[red_patch, purple_patch])
    fig.tight_layout()
    path = OUT_DIR / "group_reward_variance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_delta_heatmap():
    """Heatmap of per-turn similarity deltas across all multi-turn episodes."""
    data = json.loads((OUT_DIR / "multiturn_transcripts.json").read_text())
    multi = [ep for ep in data if len(ep["turns"]) > 1]

    max_turns = max(len(ep["turns"]) for ep in multi)
    labels = [f"{ep['phrase'][:20]} r{ep['rollout']}" for ep in multi]
    matrix = np.full((len(multi), max_turns), np.nan)

    for i, ep in enumerate(multi):
        for j, delta in enumerate(ep["deltas"]):
            matrix[i, j] = delta

    fig, ax = plt.subplots(figsize=(8, max(6, len(multi) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(max_turns))
    ax.set_xticklabels([f"Turn {i+1}" for i in range(max_turns)])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Per-Turn Similarity Deltas (green=improvement, red=worse)")
    plt.colorbar(im, ax=ax, label="Δ similarity")
    fig.tight_layout()
    path = OUT_DIR / "delta_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stateless",
        action="store_true",
        help="Read from phase-1-stateless output dir instead of phase-1.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output directory explicitly.",
    )
    args = parser.parse_args()

    if args.out_dir:
        OUT_DIR = args.out_dir
    elif args.stateless:
        OUT_DIR = Path("viz_outputs/phase-1")
    else:
        OUT_DIR = Path("viz_outputs/phase-1")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_per_phrase_variance()
    plot_trajectory_reward_distribution()
    plot_similarity_over_turns()
    plot_group_reward_variance()
    plot_delta_heatmap()
    print(f"\nAll Phase 1 plots saved to {OUT_DIR}/")
