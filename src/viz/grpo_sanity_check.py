"""Phase 2 exit gate: run GRPO training on 'birthday party' and produce diagnostics.

Usage (remote, on A100):
    modal run src/viz/grpo_sanity_check.py

The Modal function trains for 30 steps, saves the LoRA checkpoint and history to
Modal volumes, and returns the full history dict. Local plotting runs after the
remote call completes.
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

_model_volume   = modal.Volume.from_name("emoji-model-weights",   create_if_missing=True)
_output_volume  = modal.Volume.from_name("emoji-phase2-outputs",  create_if_missing=True)
_MODEL_CACHE_DIR  = "/model-cache"
_OUTPUT_DIR       = "/outputs"

app = modal.App("grpo-sanity-check")


# ---------------------------------------------------------------------------
# Remote training function
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100",
    timeout=3600,
    image=_image,
    secrets=[modal.Secret.from_name("anthropic-api-key")],
    volumes={
        _MODEL_CACHE_DIR: _model_volume,
        _OUTPUT_DIR:      _output_volume,
    },
)
def run_phase2_training(
    phrase: str = "job interview",
    run_name: str = "phase2-repetition-penalty",
    n_steps: int = 30,
    group_size: int = 8,
    learning_rate: float = 1e-5,
    eval_every: int = 5,
) -> dict:
    """Train GRPO on a single phrase and return the history dict."""
    import os

    os.environ["HF_HOME"]                    = _MODEL_CACHE_DIR
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = _MODEL_CACHE_DIR

    from src.rl.custom.train import train

    checkpoint_dir = f"{_OUTPUT_DIR}/checkpoints/{run_name}"
    history = train(
        phrases=[phrase],
        n_steps=n_steps,
        group_size=group_size,
        learning_rate=learning_rate,
        kl_coeff=0.05,
        temperature=1.0,
        max_turns=1,
        lora_rank=16,
        eval_every=eval_every,
        save_dir=checkpoint_dir,
    )

    # Persist history to volume
    history_path = f"{_OUTPUT_DIR}/{run_name}-history.json"
    _serializable = _make_serializable(history)
    with open(history_path, "w") as f:
        json.dump(_serializable, f, indent=2)
    _output_volume.commit()

    return _serializable


def _make_serializable(history: dict) -> dict:
    """Convert tensors / non-JSON types to plain Python."""
    out = {}
    for k, v in history.items():
        if v is None:
            out[k] = None
        elif isinstance(v, list):
            out[k] = [_ser_item(x) for x in v]
        elif isinstance(v, dict):
            out[k] = {dk: _ser_item(dv) for dk, dv in v.items()}
        else:
            out[k] = _ser_item(v)
    return out


def _ser_item(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_ser_item(i) for i in x]
    return x


# ---------------------------------------------------------------------------
# Local plotting
# ---------------------------------------------------------------------------

def plot_diagnostics(history: dict, out_dir: Path) -> None:
    """Generate all 6 diagnostic plots from training history."""
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    steps = history["steps"]

    # 1. Reward over steps
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, history["mean_rewards"], marker="o", markersize=3, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean trajectory reward")
    ax.set_title("Reward over training steps")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "1_reward.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 1_reward.png")

    # 2. KL divergence over steps
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, history["mean_kl"], marker="o", markersize=3, linewidth=1.5, color="darkorange")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean KL divergence")
    ax.set_title("KL divergence over training steps")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "2_kl.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 2_kl.png")

    # 3. Loss components
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, history["pg_losses"],  label="pg_loss",  linewidth=1.5, color="crimson")
    ax.plot(steps, history["kl_losses"],  label="kl_loss",  linewidth=1.5, color="mediumpurple")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss components over training steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "3_loss_components.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 3_loss_components.png")

    # 4. Before / after emoji comparison (text → PNG)
    before = history.get("before_samples") or {}
    after  = history.get("after_samples")  or {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, samples, label in [(axes[0], before, "BEFORE (step 0)"), (axes[1], after, "AFTER (final step)")]:
        ax.axis("off")
        ax.set_title(label, fontsize=13, fontweight="bold")
        lines = []
        emojis  = samples.get("emojis",  [])
        guesses = samples.get("guesses", [])
        sims    = samples.get("sims",    [])
        for e, g, s in zip(emojis, guesses, sims):
            lines.append(f"{e}\n→ '{g}'\n({s:.3f})\n")
        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace")
    fig.tight_layout()
    fig.savefig(out_dir / "4_before_after.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 4_before_after.png")

    # 5. Similarity distribution shift (histogram)
    before_sims = before.get("sims", [])
    after_sims  = after.get("sims",  [])
    if before_sims and after_sims:
        fig, ax = plt.subplots(figsize=(8, 4))
        bins = np.linspace(0, 1, 15)
        ax.hist(before_sims, bins=bins, alpha=0.6, label="Before", color="steelblue")
        ax.hist(after_sims,  bins=bins, alpha=0.6, label="After",  color="coral")
        ax.axvline(sum(before_sims) / len(before_sims), color="steelblue", linestyle="--", linewidth=1.5)
        ax.axvline(sum(after_sims)  / len(after_sims),  color="coral",     linestyle="--", linewidth=1.5)
        ax.set_xlabel("Similarity score")
        ax.set_ylabel("Count")
        ax.set_title("Similarity distribution: before vs after")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "5_similarity_distribution.png", dpi=150)
        plt.close(fig)
        print(f"  Saved 5_similarity_distribution.png")

    # 6. Gradient norm over steps
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, history["grad_norms"], marker="o", markersize=3, linewidth=1.5, color="seagreen")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gradient norm (pre-clip)")
    ax.set_title("Gradient norm over training steps")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "6_grad_norm.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 6_grad_norm.png")


def print_before_after(history: dict) -> None:
    before = history.get("before_samples") or {}
    after  = history.get("after_samples")  or {}

    print("\n=== BEFORE (step 0) ===")
    for e, g, s in zip(before.get("emojis", []), before.get("guesses", []), before.get("sims", [])):
        print(f"  {e}  →  '{g}'  ({s:.3f})")

    print("\n=== AFTER (final step) ===")
    for e, g, s in zip(after.get("emojis", []), after.get("guesses", []), after.get("sims", [])):
        print(f"  {e}  →  '{g}'  ({s:.3f})")

    if before.get("sims") and after.get("sims"):
        b_mean = sum(before["sims"]) / len(before["sims"])
        a_mean = sum(after["sims"])  / len(after["sims"])
        print(f"\nMean similarity: {b_mean:.4f} → {a_mean:.4f}  (Δ = {a_mean - b_mean:+.4f})")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    phrase: str = "job interview",
    run_name: str = "phase2-repetition-penalty",
    n_steps: int = 30,
    group_size: int = 8,
    learning_rate: float = 1e-5,
    eval_every: int = 5,
    out_dir: str = "",
    history_json: str = "",
):
    """Run Phase 2 sanity check: train remotely, plot locally.

    --out-dir defaults to viz_outputs/<run-name>.
    --history-json skips training and re-plots from a saved history file.
    """
    out_path = Path(out_dir if out_dir else f"viz_outputs/{run_name}")

    if history_json:
        print(f"Loading history from {history_json}...")
        with open(history_json) as f:
            history = json.load(f)
    else:
        print(f"Starting training: phrase='{phrase}' run='{run_name}' steps={n_steps}")
        history = run_phase2_training.remote(
            phrase=phrase,
            run_name=run_name,
            n_steps=n_steps,
            group_size=group_size,
            learning_rate=learning_rate,
            eval_every=eval_every,
        )

    # Save history locally
    out_path.mkdir(parents=True, exist_ok=True)
    history_out = out_path / "history.json"
    with open(history_out, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved to {history_out}")

    # Print before/after comparison
    print_before_after(history)

    # Generate plots
    print(f"\nGenerating plots → {out_path}/")
    plot_diagnostics(history, out_path)

    # Print success criteria summary
    rewards = history.get("mean_rewards", [])
    kls     = history.get("mean_kl", [])
    if rewards:
        print(f"\n=== Success criteria ===")
        print(f"  Reward trend:  {rewards[0]:.4f} → {rewards[-1]:.4f}  {'✓ increasing' if rewards[-1] > rewards[0] else '✗ not increasing'}")
    if kls:
        max_kl = max(kls)
        print(f"  Max KL:        {max_kl:.4f}  {'✓ < 1.0' if max_kl < 1.0 else '✗ too high'}")
    before_sims = (history.get("before_samples") or {}).get("sims") or []
    after_sims  = (history.get("after_samples")  or {}).get("sims") or []
    if before_sims and after_sims:
        b_mean = sum(before_sims) / len(before_sims)
        a_mean = sum(after_sims)  / len(after_sims)
        print(f"  Sim shift:     {b_mean:.4f} → {a_mean:.4f}  {'✓ improved' if a_mean > b_mean else '✗ not improved'}")
    print()
