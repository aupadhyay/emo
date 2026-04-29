"""Modal A100 runner for GRPO training."""

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

app = modal.App("emo-trainer")


@app.function(
    gpu="A100",
    image=_image,
    volumes={
        _MODEL_CACHE_DIR: _model_volume,
        _CHECKPOINT_DIR: _checkpoint_volume,
    },
    secrets=[
        modal.Secret.from_name("anthropic-api-key"),
        modal.Secret.from_name("wandb-secret"),
    ],
    timeout=36000,  # 10 hours
)
def run_training(
    n_steps: int = 500,
    group_size: int = 8,
    max_turns: int = 5,
    learning_rate: float = 1e-5,
    kl_coeff: float = 0.05,
    temperature: float = 1.0,
    lora_rank: int = 16,
    seed: int = 42,
    eval_every: int = 50,
    log_every: int = 10,
    n_eval_episodes: int = 30,
) -> dict:
    import os

    os.environ["HF_HOME"] = _MODEL_CACHE_DIR

    from src.rl.custom.train import train

    with open("/root/data/training_phrases.json") as f:
        phrase_data = json.load(f)

    history = train(
        phrases=phrase_data["training"],
        eval_phrases=phrase_data["held_out"],
        n_steps=n_steps,
        group_size=group_size,
        max_turns=max_turns,
        learning_rate=learning_rate,
        kl_coeff=kl_coeff,
        temperature=temperature,
        lora_rank=lora_rank,
        seed=seed,
        eval_every=eval_every,
        log_every=log_every,
        n_eval_episodes=n_eval_episodes,
        save_dir=_CHECKPOINT_DIR,
    )

    _checkpoint_volume.commit()
    return {
        "status": "complete",
        "steps_completed": history["steps"][-1] if history["steps"] else 0,
        "final_reward": history["mean_rewards"][-1] if history["mean_rewards"] else 0.0,
        "held_out_rewards": history.get("held_out_eval_rewards", []),
    }


@app.local_entrypoint()
def main(
    n_steps: int = 500,
    group_size: int = 8,
    max_turns: int = 5,
    kl_coeff: float = 0.05,
    learning_rate: float = 1e-5,
):
    """Launch GRPO training on Modal.

    Usage:
        modal run src/rl/custom/modal_train.py                        # blocking
        modal run --detach src/rl/custom/modal_train.py               # detached
        modal run --detach src/rl/custom/modal_train.py --kl-coeff 0.15 --learning-rate 5e-6
    """
    result = run_training.remote(
        n_steps=n_steps,
        group_size=group_size,
        max_turns=max_turns,
        kl_coeff=kl_coeff,
        learning_rate=learning_rate,
    )
    print(f"Training complete: {result}")
