"""Training analysis: pull metrics from W&B and plot training curves.

Usage:
    uv run python -m src.viz.wandb_run_analysis --run-id <run_name_or_id>
    uv run python -m src.viz.wandb_run_analysis --run-id <run_id> --entity myteam
    uv run python -m src.viz.wandb_run_analysis --run-id <run_id> --project emo
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import wandb

OUT_DIR = Path("viz_outputs/runs")


def fetch_run(entity: str | None, project: str, run_name: str):
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"display_name": run_name})
    runs = list(runs)
    if not runs:
        # fall back to treating run_name as run ID
        run = api.run(f"{path}/{run_name}")
    else:
        run = runs[0]
    print(f"Found run: {run.name} ({run.id}) — state: {run.state}")
    return run


def smooth(values: list[float], window: int = 10) -> list[float]:
    if len(values) < window:
        return values
    result = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        result.append(sum(values[lo:hi]) / (hi - lo))
    return result


def plot_training_curves(history: dict, out_dir: Path, run_id: str, run_name: str):
    steps = history["step"]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Training Curves — {run_name}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Mean reward (train)
    ax = fig.add_subplot(gs[0, 0])
    r = history.get("train/mean_reward", [])
    if r:
        ax.plot(steps[: len(r)], r, alpha=0.3, color="steelblue", linewidth=0.8)
        ax.plot(steps[: len(r)], smooth(r), color="steelblue", linewidth=2)
    ax.set_title("Train Reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Trajectory Reward")

    # 2. Held-out eval reward
    ax = fig.add_subplot(gs[0, 1])
    held_steps = history.get("eval/held_out_reward_steps", [])
    held_r = history.get("eval/held_out_reward", [])
    train_eval_steps = history.get("eval/mean_reward_steps", [])
    train_eval_r = history.get("eval/mean_reward", [])
    if held_r:
        ax.plot(held_steps, held_r, marker="o", color="tomato", label="held-out", linewidth=2)
    if train_eval_r:
        ax.plot(train_eval_steps, train_eval_r, marker="s", color="steelblue", label="train eval", linewidth=2, linestyle="--")
    ax.set_title("Eval Reward")
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)

    # 3. Completion rate
    ax = fig.add_subplot(gs[0, 2])
    comp_steps = history.get("eval/completion_rate_steps", [])
    comp_r = history.get("eval/completion_rate", [])
    held_comp_steps = history.get("eval/held_out_completion_rate_steps", [])
    held_comp_r = history.get("eval/held_out_completion_rate", [])
    if comp_r:
        ax.plot(comp_steps, comp_r, marker="s", color="steelblue", label="train eval", linewidth=2)
    if held_comp_r:
        ax.plot(held_comp_steps, held_comp_r, marker="o", color="tomato", label="held-out", linewidth=2)
    ax.set_title("Completion Rate")
    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction completed")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    # 4. KL divergence
    ax = fig.add_subplot(gs[1, 0])
    kl = history.get("train/mean_kl", [])
    if kl:
        ax.plot(steps[: len(kl)], kl, alpha=0.3, color="purple", linewidth=0.8)
        ax.plot(steps[: len(kl)], smooth(kl), color="purple", linewidth=2)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="KL budget")
    ax.set_title("KL Divergence (policy vs ref)")
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)

    # 5. Gradient norm
    ax = fig.add_subplot(gs[1, 1])
    gn = history.get("train/grad_norm", [])
    if gn:
        ax.plot(steps[: len(gn)], gn, alpha=0.3, color="darkorange", linewidth=0.8)
        ax.plot(steps[: len(gn)], smooth(gn), color="darkorange", linewidth=2)
    ax.set_title("Gradient Norm")
    ax.set_xlabel("Step")

    # 6. Group reward std (exploration signal)
    ax = fig.add_subplot(gs[1, 2])
    std = history.get("train/group_reward_std", [])
    if std:
        ax.plot(steps[: len(std)], std, alpha=0.3, color="teal", linewidth=0.8)
        ax.plot(steps[: len(std)], smooth(std), color="teal", linewidth=2)
    ax.set_title("Group Reward Std\n(exploration signal)")
    ax.set_xlabel("Step")

    # 7. Loss components
    ax = fig.add_subplot(gs[2, 0])
    pg = history.get("train/pg_loss", [])
    kl_loss = history.get("train/kl_loss", [])
    if pg:
        ax.plot(steps[: len(pg)], smooth(pg), label="pg_loss", color="steelblue", linewidth=2)
    if kl_loss:
        ax.plot(steps[: len(kl_loss)], smooth(kl_loss), label="kl_loss", color="purple", linewidth=2)
    ax.set_title("Loss Components (smoothed)")
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)

    # 8. Avg completion turn
    ax = fig.add_subplot(gs[2, 1])
    turn_steps = history.get("eval/avg_completion_turn_steps", [])
    turn_vals = history.get("eval/avg_completion_turn", [])
    if turn_vals:
        ax.plot(turn_steps, turn_vals, marker="o", color="green", linewidth=2)
    ax.set_title("Avg Completion Turn\n(lower = faster)")
    ax.set_xlabel("Step")

    # 9. Reward min/max band
    ax = fig.add_subplot(gs[2, 2])
    r_min = history.get("train/group_reward_min", [])
    r_max = history.get("train/group_reward_max", [])
    if r and r_min and r_max:
        n = min(len(steps), len(r), len(r_min), len(r_max))
        ax.fill_between(steps[:n], r_min[:n], r_max[:n], alpha=0.2, color="steelblue")
        ax.plot(steps[:n], r[:n], color="steelblue", linewidth=1.5, label="mean")
    ax.set_title("Group Reward Range")
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)

    out_path = out_dir / f"{run_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_summary(history: dict, run):
    steps = history.get("step", [])
    rewards = history.get("train/mean_reward", [])
    held = history.get("eval/held_out_reward", [])
    comp = history.get("eval/completion_rate", [])

    print("\n" + "=" * 50)
    print("RUN SUMMARY")
    print("=" * 50)
    print(f"  Run name:      {run.name}")
    print(f"  Run state:     {run.state}")
    print(f"  Steps logged:  {len(steps)}")
    if rewards:
        print(f"  Train reward:  start={rewards[0]:.4f}  end={rewards[-1]:.4f}  best={max(rewards):.4f}")
    if held:
        print(f"  Held-out rwd:  start={held[0]:.4f}  end={held[-1]:.4f}  best={max(held):.4f}")
    if comp:
        print(f"  Completion:    start={comp[0]:.2f}  end={comp[-1]:.2f}  best={max(comp):.2f}")
    print("=" * 50 + "\n")


def fetch_history(run) -> dict:
    """Pull all logged scalar metrics, keeping step alignment."""
    df = run.history(samples=10000, pandas=True)
    history: dict = {}

    scalar_keys = [
        "train/mean_reward", "train/pg_loss", "train/kl_loss", "train/mean_kl",
        "train/grad_norm", "train/group_reward_std", "train/group_reward_min",
        "train/group_reward_max",
    ]
    eval_keys = [
        "eval/mean_reward", "eval/completion_rate", "eval/avg_completion_turn",
        "eval/held_out_reward", "eval/held_out_completion_rate",
    ]

    if "_step" in df.columns:
        history["step"] = df["_step"].dropna().tolist()
    elif "step" in df.columns:
        history["step"] = df["step"].dropna().tolist()

    for key in scalar_keys:
        if key in df.columns:
            col = df[key].dropna()
            history[key] = col.tolist()

    for key in eval_keys:
        if key in df.columns:
            col = df[[key, "_step"]].dropna(subset=[key])
            history[key] = col[key].tolist()
            history[f"{key}_steps"] = col["_step"].tolist()

    return history


def main():
    parser = argparse.ArgumentParser(description="Analyze a W&B training run")
    parser.add_argument("--run-id", required=True, help="W&B run name or ID")
    parser.add_argument("--entity", default=None, help="W&B entity/username (default: uses logged-in user)")
    parser.add_argument("--project", default="emo", help="W&B project name (default: emo)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run = fetch_run(args.entity, args.project, args.run_id)
    history = fetch_history(run)

    print_summary(history, run)
    plot_training_curves(history, OUT_DIR, args.run_id, run.name)
    print(f"\nPlots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
