"""Evaluate a saved LoRA checkpoint on the held-out phrase set.

Loads a checkpoint from the Modal `emo-checkpoints` volume, attaches the LoRA
adapter to the base model, and runs multi-turn held-out eval. Prints summary
metrics + transcripts.

Checkpoint layout on the volume (after the run-name namespacing fix):
    /checkpoints/<run_name>/step_<N>/
    /checkpoints/<run_name>/final/

Pre-fix legacy checkpoints sit at the root: /checkpoints/step_<N>/

Usage:
    # List what's on the volume
    uv run modal run scripts/eval_checkpoint.py::list_ckpts

    # New-style (after namespacing fix): pass run name + step
    uv run modal run scripts/eval_checkpoint.py --run-name grpo_20260430_0306 --step 150

    # Legacy / no run-name: pass --step alone (resolves to /checkpoints/step_N)
    uv run modal run scripts/eval_checkpoint.py --step 150

    # Or pin the path explicitly
    uv run modal run scripts/eval_checkpoint.py --checkpoint-path /checkpoints/grpo_.../step_150
"""

import json

import modal

_MODEL_CACHE_DIR = "/model-cache"
_CHECKPOINT_DIR = "/checkpoints"

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=5.0.0",
        "peft>=0.14.0",
        "sentence-transformers>=3.0.0",
        "anthropic>=0.40.0",
        "wandb",
        "python-dotenv",
        "regex",
        "numpy",
        "tqdm",
    )
    .add_local_python_source("src")
    .add_local_file("data/training_phrases.json", "/root/data/training_phrases.json")
    .add_local_file("data/emoji-test.txt", "/root/data/emoji-test.txt")
)

_model_volume = modal.Volume.from_name("emoji-model-weights", create_if_missing=True)
_checkpoint_volume = modal.Volume.from_name("emo-checkpoints", create_if_missing=True)

app = modal.App("emo-eval-checkpoint")


@app.function(
    gpu="A100",
    image=_image,
    volumes={
        _MODEL_CACHE_DIR: _model_volume,
        _CHECKPOINT_DIR: _checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("anthropic-api-key")],
    timeout=3600,
)
def eval_checkpoint(
    checkpoint_path: str,
    eps_per_phrase: int = 1,
    max_turns: int = 5,
    temperature: float = 1.0,
    guesser_model: str = "claude-sonnet-4-20250514",
) -> dict:
    import os
    from pathlib import Path

    os.environ["HF_HOME"] = _MODEL_CACHE_DIR

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.rl.custom.env import SimulatedGuesser
    from src.rl.custom.generate import MODEL_NAME, build_emoji_mask
    from src.rl.custom.reward import SimilarityScorer, compute_turn_rewards
    from src.rl.custom.train import run_episode_group_hf

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        available = sorted(p.name for p in Path(_CHECKPOINT_DIR).iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Available under {_CHECKPOINT_DIR}: {available}"
        )

    print(f"Loading base model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Attaching LoRA from {ckpt}...")
    model = PeftModel.from_pretrained(base, str(ckpt))
    model.eval()

    emoji_mask = build_emoji_mask(tokenizer)
    guesser = SimulatedGuesser(model=guesser_model, difficulty="casual", conversation_mode=max_turns > 1)
    scorer = SimilarityScorer()

    with open("/root/data/training_phrases.json") as f:
        phrase_data = json.load(f)
    held_out = phrase_data["held_out"]
    print(f"Evaluating on {len(held_out)} held-out phrases × {eps_per_phrase} eps each...")

    all_episodes = []
    for p in held_out:
        eps = run_episode_group_hf(
            model, tokenizer, emoji_mask, guesser, scorer,
            p, eps_per_phrase, max_turns, temperature,
        )
        all_episodes.extend(eps)

    rewards = []
    for ep in all_episodes:
        r = compute_turn_rewards(
            ep.target_phrase,
            [t.guess for t in ep.turns],
            scorer,
            emoji_outputs=[t.emoji_output for t in ep.turns],
        )
        rewards.append(r["trajectory_reward"])

    completion_rate = sum(ep.completed for ep in all_episodes) / len(all_episodes)
    completed_eps = [ep for ep in all_episodes if ep.completed]
    mean_completion_turn = (
        sum(ep.completion_turn for ep in completed_eps) / len(completed_eps)
        if completed_eps else float(max_turns)
    )
    mean_reward = sum(rewards) / len(rewards)

    print("\n" + "=" * 60)
    print(f"HELD-OUT EVAL — {ckpt.name}")
    print("=" * 60)
    print(f"  Phrases:           {len(held_out)}")
    print(f"  Episodes:          {len(all_episodes)}")
    print(f"  Mean reward:       {mean_reward:.4f}")
    print(f"  Completion rate:   {completion_rate:.2%}")
    print(f"  Avg completion:    {mean_completion_turn:.2f} turns (when completed)")
    print("=" * 60)

    print("\n=== TRANSCRIPTS ===")
    for ep in all_episodes:
        marker = "✓" if ep.completed else "✗"
        print(f"\n{marker} {ep.target_phrase}")
        for t in ep.turns:
            print(f"    Turn {t.turn_number}: {t.emoji_output} → '{t.guess}' (sim: {t.similarity:.3f})")

    return {
        "checkpoint": str(ckpt),
        "mean_reward": mean_reward,
        "completion_rate": completion_rate,
        "mean_completion_turn": mean_completion_turn,
        "n_phrases": len(held_out),
        "n_episodes": len(all_episodes),
        "episodes": [
            {
                "phrase": ep.target_phrase,
                "completed": ep.completed,
                "completion_turn": ep.completion_turn,
                "turns": [
                    {
                        "turn": t.turn_number,
                        "emoji": t.emoji_output,
                        "guess": t.guess,
                        "sim": t.similarity,
                    }
                    for t in ep.turns
                ],
                "trajectory_reward": rewards[i],
            }
            for i, ep in enumerate(all_episodes)
        ],
    }


@app.function(
    image=_image,
    volumes={_CHECKPOINT_DIR: _checkpoint_volume},
    timeout=120,
)
def list_checkpoints() -> list[str]:
    """Return relative paths of every checkpoint dir, depth 1 or 2."""
    from pathlib import Path
    root = Path(_CHECKPOINT_DIR)
    out: list[str] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        # Depth-1 dir is either a legacy step_N or a run-name directory.
        children = [c for c in entry.iterdir() if c.is_dir()]
        if children:
            for c in sorted(children):
                out.append(f"{entry.name}/{c.name}")
        else:
            out.append(entry.name)
    return out


@app.local_entrypoint()
def list_ckpts():
    """Print available checkpoints on the volume (recursive, depth 2)."""
    ckpts = list_checkpoints.remote()
    print(f"Found {len(ckpts)} checkpoints on emo-checkpoints volume:")
    for c in ckpts:
        print(f"  {c}")


@app.local_entrypoint()
def main(
    step: int = 0,
    run_name: str = "",
    checkpoint_path: str = "",
    eps_per_phrase: int = 1,
    max_turns: int = 5,
    temperature: float = 1.0,
):
    """Evaluate a LoRA checkpoint on held-out phrases.

    Pass --checkpoint-path for an explicit path, OR --step N (with optional --run-name)
    to resolve the volume path:
      - --step N --run-name <name>   ->  /checkpoints/<name>/step_N
      - --step N                     ->  /checkpoints/step_N (legacy layout)

    Writes a JSON report to eval_outputs/<slug>_<timestamp>.json with full
    transcripts so you don't have to scrape stdout.
    """
    import json
    from datetime import datetime
    from pathlib import Path

    if not checkpoint_path:
        if not step:
            raise SystemExit("Pass --step N (optionally with --run-name) or --checkpoint-path /path")
        if run_name:
            checkpoint_path = f"{_CHECKPOINT_DIR}/{run_name}/step_{step}"
        else:
            checkpoint_path = f"{_CHECKPOINT_DIR}/step_{step}"

    result = eval_checkpoint.remote(
        checkpoint_path=checkpoint_path,
        eps_per_phrase=eps_per_phrase,
        max_turns=max_turns,
        temperature=temperature,
    )

    out_dir = Path("eval_outputs")
    out_dir.mkdir(exist_ok=True)
    slug = checkpoint_path.replace("/", "_").strip("_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{slug}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\nSummary: reward={result['mean_reward']:.4f}  "
          f"completion={result['completion_rate']:.2%}  "
          f"avg_turn={result['mean_completion_turn']:.2f}")
    print(f"Report: {out_path}")
