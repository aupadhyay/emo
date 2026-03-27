"""GRPO training loop for the emoji retrieval game on Tinker.

Starts from the SFT checkpoint and trains the sender via PPO loss
with group-relative advantages computed across rollouts.

Usage:
    uv run python -m src.rl.train --sft-checkpoint <tinker-state-name>
    uv run python -m src.rl.train --resume <tinker-state-name>
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import numpy as np
import tinker
import torch
from dotenv import load_dotenv
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.metrics import incorporate_kl_penalty
from tinker_cookbook.rl.rollouts import do_group_rollout

from src.rl.emoji_completer import EmojiTokenCompleter
from src.rl.env import RetrievalGameDataset
from src.rl.judge import JudgeClient
from src.rl.reward import RetrievalGameReward

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SENDER_MODEL = "Qwen/Qwen3.5-4B"


async def train(args: argparse.Namespace) -> None:
    # ── Setup ──
    service = tinker.ServiceClient()

    # Training client (sender model with LoRA)
    if args.resume:
        training_client = service.create_training_client_from_state_with_optimizer(args.resume)
        logger.info("Resumed from %s", args.resume)
    elif args.sft_checkpoint:
        training_client = await service.create_lora_training_client_async(
            base_model=args.sender_model,
            rank=args.lora_rank,
        )
        training_client.load_state(args.sft_checkpoint)
        logger.info("Loaded SFT checkpoint: %s", args.sft_checkpoint)
    else:
        raise ValueError("Must provide --sft-checkpoint or --resume")

    tokenizer = training_client.get_tokenizer()
    renderer = renderers.get_renderer("qwen3_disable_thinking", tokenizer)

    # Sampling client for the sender (rollouts)
    sender_sampling_client = training_client.save_weights_and_get_sampling_client(
        name="rl-sender-init"
    )

    # Frozen reference sampling client for KL penalty (stays at SFT weights)
    ref_sampling_client: tinker.SamplingClient | None = None
    if args.kl_coef > 0:
        ref_sampling_client = training_client.save_weights_and_get_sampling_client(
            name="rl-ref-sft"
        )
        logger.info("KL penalty enabled: coef=%.4f", args.kl_coef)

    # Judge model (frozen, separate sampling client)
    judge = JudgeClient.create(service, judge_model=args.judge_model)

    # Reward function
    reward_fn = RetrievalGameReward(
        similarity_threshold=args.similarity_threshold,
        success_bonus=0.5,
        turn_penalty=0.1,
        format_penalty=0.5,
    )

    # Dataset
    dataset = RetrievalGameDataset(
        prompts_path=args.prompts_path,
        judge=judge,
        reward_fn=reward_fn,
        renderer=renderer,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_turns=args.max_turns,
    )

    # Token completer for rollouts (emoji-constrained)
    policy = EmojiTokenCompleter(
        sampling_client=sender_sampling_client,
        max_tokens=20,
        temperature=args.temperature,
    )

    # Logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    metrics_file = open(log_dir / "metrics.jsonl", "a")

    # ── Training loop ──
    for iteration in range(args.num_iterations):
        iter_start = time.time()
        logger.info("=" * 60)
        logger.info("Iteration %d/%d", iteration + 1, args.num_iterations)

        # Get batch of env group builders
        env_group_builders = dataset.get_batch(iteration)

        # ── Rollout phase ──
        # Run all group rollouts (each builder = one target message × group_size rollouts)
        trajectory_groups = []
        n_failed = 0
        for group_idx, builder in enumerate(env_group_builders):
            try:
                traj_group = await do_group_rollout(
                    env_group_builder=builder,
                    policy=policy,
                )
                trajectory_groups.append(traj_group)
            except Exception as e:
                n_failed += 1
                logger.warning("  Group %d failed: %s", group_idx, e)
                continue

            if group_idx % 8 == 0 and group_idx > 0:
                rewards = traj_group.get_total_rewards()
                logger.info(
                    "  Group %d/%d: mean_reward=%.3f",
                    group_idx, len(env_group_builders), np.mean(rewards),
                )

        if n_failed:
            logger.warning("  %d/%d groups failed this iteration", n_failed, len(env_group_builders))

        # ── Compute advantages ──
        advantages = compute_advantages(trajectory_groups)

        # ── Assemble training data ──
        training_data, metadata = assemble_training_data(trajectory_groups, advantages)

        if not training_data:
            logger.warning("No training data produced this iteration, skipping update")
            continue

        # ── KL penalty (before stripping mask) ──
        kl_metrics = {}
        if ref_sampling_client is not None and args.kl_coef > 0:
            try:
                kl_metrics = await incorporate_kl_penalty(
                    data_D=training_data,
                    base_sampling_client=ref_sampling_client,
                    kl_penalty_coef=args.kl_coef,
                    kl_discount_factor=0.0,
                )
                logger.info("  KL penalty: kl_policy_base=%.4f", kl_metrics.get("kl_policy_base", 0))
            except Exception as e:
                logger.warning("  KL penalty failed: %s", e)

        # Fix datum format: Tinker PPO only accepts target_tokens + logprobs + advantages.
        # The cookbook produces a "mask" field that Tinker's server can't serialize.
        # Fold mask into advantages (zero-masked positions already have 0 advantage).
        PPO_FIELDS = {"target_tokens", "logprobs", "advantages"}
        fixed_data = []
        for datum in training_data:
            old = datum.loss_fn_inputs
            mask_t = old["mask"].to_torch() if "mask" in old else None
            fixed_inputs: dict[str, tinker.TensorData] = {}
            for k in PPO_FIELDS:
                if k not in old:
                    continue
                t = old[k].to_torch()
                if k == "target_tokens":
                    fixed_inputs[k] = tinker.TensorData.from_torch(t.to(torch.int64))
                elif k == "advantages" and mask_t is not None:
                    # Apply mask: zero out advantages for observation tokens
                    fixed_inputs[k] = tinker.TensorData.from_torch(
                        (t * mask_t).to(torch.float32)
                    )
                else:
                    fixed_inputs[k] = tinker.TensorData.from_torch(t.to(torch.float32))
            fixed_data.append(tinker.Datum(
                model_input=datum.model_input,
                loss_fn_inputs=fixed_inputs,
            ))
        training_data = fixed_data

        logger.info("  Training on %d datums", len(training_data))

        # ── Training phase ──
        # Forward-backward with PPO loss
        fwd_bwd_future = training_client.forward_backward(
            training_data,
            loss_fn="ppo",
        )
        optim_future = training_client.optim_step(
            tinker.AdamParams(
                learning_rate=args.learning_rate,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
            )
        )

        fwd_bwd_result = fwd_bwd_future.result()
        optim_result = optim_future.result()

        # Sync sampler weights with updated training weights
        sender_sampling_client = training_client.save_weights_and_get_sampling_client(
            name=f"rl-sender-iter-{iteration + 1}"
        )
        policy = EmojiTokenCompleter(
            sampling_client=sender_sampling_client,
            max_tokens=20,
            temperature=args.temperature,
        )

        # ── Metrics ──
        all_rewards = []
        all_similarities = []
        all_successes = []
        all_turns = []
        for tg in trajectory_groups:
            total_rewards = tg.get_total_rewards()
            all_rewards.extend(total_rewards)
            for traj, metrics_g in zip(tg.trajectories_G, tg.metrics_G):
                # Collect metrics from the final transition
                final_metrics = traj.transitions[-1].metrics if traj.transitions else {}
                all_similarities.append(final_metrics.get("similarity", 0.0))
                all_successes.append(final_metrics.get("success", 0.0))
                all_turns.append(final_metrics.get("num_turns", len(traj.transitions)))

        iter_elapsed = time.time() - iter_start
        metrics = {
            "iteration": iteration + 1,
            "reward/mean": float(np.mean(all_rewards)),
            "reward/std": float(np.std(all_rewards)),
            "reward/min": float(np.min(all_rewards)),
            "reward/max": float(np.max(all_rewards)),
            "similarity/mean": float(np.mean(all_similarities)) if all_similarities else 0.0,
            "success_rate": float(np.mean(all_successes)) if all_successes else 0.0,
            "turns/mean": float(np.mean(all_turns)) if all_turns else 0.0,
            "n_training_datums": len(training_data),
            **{f"kl/{k}": v for k, v in kl_metrics.items()},
            "time_s": round(iter_elapsed, 1),
        }

        logger.info(
            "  reward=%.3f | similarity=%.3f | success=%.1f%% | turns=%.1f | %.1fs",
            metrics["reward/mean"],
            metrics["similarity/mean"],
            metrics["success_rate"] * 100,
            metrics["turns/mean"],
            iter_elapsed,
        )

        metrics_file.write(json.dumps(metrics) + "\n")
        metrics_file.flush()

        # ── Checkpointing ──
        if (iteration + 1) % args.save_every == 0:
            ckpt_name = f"rl-step-{iteration + 1:04d}"
            save_result = training_client.save_state(ckpt_name, ttl_seconds=604800).result()
            logger.info("  Checkpoint saved: %s", save_result)

    # Final save
    final_result = training_client.save_state("rl-final", ttl_seconds=None).result()
    logger.info("Training complete. Final checkpoint: %s", final_result)
    metrics_file.close()


def main():
    parser = argparse.ArgumentParser(description="emoji-maxxing RL training (GRPO)")
    parser.add_argument("--sft-checkpoint", type=str, help="Tinker state name for SFT checkpoint")
    parser.add_argument("--resume", type=str, help="Resume from RL checkpoint")
    parser.add_argument("--sender-model", default=SENDER_MODEL)
    parser.add_argument("--judge-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--prompts-path", default="data/rl_prompts.jsonl")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--kl-coef", type=float, default=0.01, help="KL penalty coefficient (0 to disable)")
    parser.add_argument("--num-iterations", type=int, default=75)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--log-dir", default="runs/rl")
    args = parser.parse_args()
    asyncio.run(train(args))


if __name__ == "__main__":
    main()
